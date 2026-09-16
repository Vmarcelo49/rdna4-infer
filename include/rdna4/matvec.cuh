// M2 step 3 — fused matvec (quantized weights x q8_1 activations) on gfx1201.
//
// Structure mirrors llama.cpp's mmvq (ggml-cuda/mmvq.cu, MIT) for the
// ncols_dst=1 / no-fusion case, one warp per output row:
//
//   blocks_per_iter = vdr * warp_size / qi
//   kqs             = vdr * (tid % (qi/vdr))      // iqs slot inside the block
//   slot            = tid / (qi/vdr)              // which block within the iter
//   kby             = kb * (qk / QK8_1)           // aligned activation block
//
// with per-type (qk, qi, vdr) from ggml-common.h / vecdotq.cuh. Differences
// from llama.cpp: no DMMV-style prefetch, no fusion, no multi-column batching,
// no tuned nwarps table (one warp per row, grid-strided over rows).
#pragma once
#include <hip/hip_runtime.h>

#include "rdna4/tuning.h"
#include "rdna4/quants.h"
#include "rdna4/fp16.h"
#include "rdna4/vecdotq.cuh"

namespace rdna4 {

// ---------------------------------------------------------------------------
// Activation quantization: one warp turns 32 floats into one block_q8_1,
// exactly like ggml's quantize_row_q8_1_ref:
//   d = amax/127, qs[i] = round(x[i]/d), s = d * sum(qs)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void quantize_q8_1_block(const float *x, block_q8_1 *y) {
  const int lane = threadIdx.x & 31;
  const float xi = x[lane];
  float amax = fabsf(xi);
  // Only the max needs a reduction here; the sum of the QUANTIZED values is
  // reduced further down, because that is what ggml's reference stores.
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  const int q = (int)roundf(xi * id);  // ggml's ref uses roundf (half away from zero)
  y->qs[lane] = (int8_t)q;

  int qsum = q;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor(qsum, off);
  if (lane == 0) {
    // ds = {d, s} with s = d * sum(qs), reproducing ggml's
    // quantize_row_q8_1_ref byte-for-byte (verified by the activation
    // cross-check in check-matvec-gpu). NOTE: llama.cpp's CUDA mmvq instead
    // stores make_half2(d, sum_of_raw_inputs) - a different value that only
    // vec_dot_iq1_s_q8_1 reads, and only ~1% off on that one type. Matching the
    // reference keeps the whole block_q8_1 bit-exact and the activation side
    // independently verifiable, which is worth more than matching mmvq.
    y->ds = (uint32_t)float_to_fp16(d) | ((uint32_t)float_to_fp16(d * (float)qsum) << 16);
  }
}

__global__ void quantize_q8_1_kernel(const float *__restrict__ x, block_q8_1 *__restrict__ y,
                                     int64_t nblocks) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= nblocks) return;
  quantize_q8_1_block(x + (int64_t)warp * QK8_1, y + warp);
}

// Quantize N activation rows in one launch: one warp per (row, 32-element block),
// the same `quantize_q8_1_block` the per-row kernel uses, so the blocks are
// bit-identical to N separate launches.
__global__ void quantize_q8_1_batch_kernel(const float *__restrict__ x,
                                           block_q8_1 *__restrict__ y,
                                           std::int64_t nblocks_per_row, std::int64_t nrows,
                                           std::int64_t row_stride) {
  const std::int64_t warp = ((std::int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= nrows * nblocks_per_row) return;
  const std::int64_t row = warp / nblocks_per_row;
  const std::int64_t blk = warp % nblocks_per_row;
  quantize_q8_1_block(x + row * row_stride + blk * QK8_1, y + warp);
}

inline bool quantize_q8_1_batch_launch(const float *d_x, block_q8_1 *d_y,
                                       std::int64_t nblocks_per_row, std::int64_t nrows,
                                       std::int64_t row_stride, hipStream_t stream = nullptr) {
  const std::int64_t warps = nrows * nblocks_per_row;
  if (warps <= 0) return false;
  const int threads = 128;  // 4 warps per CTA
  const unsigned grid = (unsigned)((warps + 3) / 4);
  quantize_q8_1_batch_kernel<<<grid, threads, 0, stream>>>(d_x, d_y, nblocks_per_row, nrows,
                                                           row_stride);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Per-type traits: block size, quants-per-int, values-per-dot, byte size,
// and the vec_dot entry point.
// ---------------------------------------------------------------------------
// Nada a hoistar (tipos sem dequantizacao cara por token no lote).
struct NoPrep {};

#define RD_MATVEC_TRAITS(Name, Vd, QK, QI, VDR, BlockT)                                 \
  struct Name {                                                                         \
    static constexpr int qk = QK, qi = QI, vdr = VDR;                                   \
    using block_t = BlockT;                                                             \
    static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,   \
                                                const int &kbx, const int &iqs) {       \
      return Vd(vbq, a, kbx, iqs);                                                      \
    }                                                                                   \
    /* dot com a LUT vinda de um ponteiro (LDS). Aqui a tabela nao existe: a LUT     */ \
    /* global e ignorada, o que mantem UMA assinatura para o kernel todo.            */ \
    static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a, \
                                                    const int &kbx, const int &iqs,     \
                                                    const void * /*lut*/) {             \
      return Vd(vbq, a, kbx, iqs);                                                      \
    }                                                                                   \
    /* Caminho em lote: por omissao nao ha' nada a hoistar, entao o par            */ \
    /* prep/dot_prep reproduz EXATAMENTE a chamada de hoje (bit-exato).            */ \
    using prep_t = NoPrep;                                                              \
    static __device__ __forceinline__ const void *lut_source() { return nullptr; }        \
    static __device__ __forceinline__ prep_t prep(const void * /*vbq*/, const int &/*kbx*/, \
                                                  const int &/*iqs*/, const void */*lut*/) { \
      return prep_t{};                                                                  \
    }                                                                                   \
    static __device__ __forceinline__ float dot_prep(const prep_t &, const void *vbq,   \
                                                     const block_q8_1 *a, const int &kbx, \
                                                     const int &iqs, const void */*lut*/) { \
      return Vd(vbq, a, kbx, iqs);                                                      \
    }                                                                                   \
  }

RD_MATVEC_TRAITS(TQ8_0, vec_dot_q8_0_q8_1, 32, QI8_0, VDR_Q8_0_Q8_1_MMVQ, block_q8_0);
RD_MATVEC_TRAITS(TQ2K, vec_dot_q2_K_q8_1, 256, QI2_K, VDR_Q2_K_Q8_1_MMVQ, block_q2_K);
RD_MATVEC_TRAITS(TQ3K, vec_dot_q3_K_q8_1, 256, QI3_K, VDR_Q3_K_Q8_1_MMVQ, block_q3_K);
RD_MATVEC_TRAITS(TQ4K, vec_dot_q4_K_q8_1, 256, QI4_K, VDR_Q4_K_Q8_1_MMVQ, block_q4_K);
RD_MATVEC_TRAITS(TQ5K, vec_dot_q5_K_q8_1, 256, QI5_K, VDR_Q5_K_Q8_1_MMVQ, block_q5_K);
RD_MATVEC_TRAITS(TQ6K, vec_dot_q6_K_q8_1, 256, QI6_K, VDR_Q6_K_Q8_1_MMVQ, block_q6_K);
RD_MATVEC_TRAITS(TIQ2XXS, vec_dot_iq2_xxs_q8_1, 256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs);
RD_MATVEC_TRAITS(TIQ2XS, vec_dot_iq2_xs_q8_1, 256, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, block_iq2_xs);
RD_MATVEC_TRAITS(TIQ2S, vec_dot_iq2_s_q8_1, 256, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, block_iq2_s);
RD_MATVEC_TRAITS(TIQ3XXS, vec_dot_iq3_xxs_q8_1, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS(TIQ3S, vec_dot_iq3_s_q8_1, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ1S, vec_dot_iq1_s_q8_1, 256, QI1_S, VDR_IQ1_S_Q8_1_MMVQ, block_iq1_s);
RD_MATVEC_TRAITS(TIQ4NL, vec_dot_iq4_nl_q8_1, 32, QI4_NL, VDR_IQ4_NL_Q8_1_MMVQ, block_iq4_nl);
RD_MATVEC_TRAITS(TIQ4XS, vec_dot_iq4_xs_q8_1, 256, QI4_XS, VDR_IQ4_XS_Q8_1_MMVQ, block_iq4_xs);

// iq3_s A/B variants (bench only; see vecdotq.cuh)
RD_MATVEC_TRAITS(TIQ3S_LIN, vec_dot_iq3_s_q8_1_lin, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ3S_PERM, vec_dot_iq3_s_q8_1_perm, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ3S_XORADD, vec_dot_iq3_s_q8_1_xoradd, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ2XXS_PERM, vec_dot_iq2_xxs_q8_1_perm, 256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs);
RD_MATVEC_TRAITS(TIQ2XXS_PERM2, vec_dot_iq2_xxs_q8_1_perm2, 256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs);
RD_MATVEC_TRAITS(TIQ2XS_PERM2, vec_dot_iq2_xs_q8_1_perm2, 256, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, block_iq2_xs);
RD_MATVEC_TRAITS(TIQ2S_PERM2, vec_dot_iq2_s_q8_1_perm2, 256, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, block_iq2_s);
// Shipping traits for the sign-using IQ types: the perm+lin bodies below were
// measured 1.26-2.02x faster than the vendored form and are bit-identical
// (checked by check-matvec-gpu --bench-ab and the per-type CPU-oracle test).
RD_MATVEC_TRAITS(TIQ2XXS_S, vec_dot_iq2_xxs_q8_1_perm2, 256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs);
RD_MATVEC_TRAITS(TIQ2XS_S, vec_dot_iq2_xs_q8_1_perm2, 256, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, block_iq2_xs);
RD_MATVEC_TRAITS(TIQ3XXS_S, vec_dot_iq3_xxs_q8_1_perm2, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS(TIQ2S_S, vec_dot_iq2_s_q8_1_perm2, 256, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, block_iq2_s);
RD_MATVEC_TRAITS(TIQ3S_S, vec_dot_iq3_s_q8_1_perm, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ2XS_PERM, vec_dot_iq2_xs_q8_1_perm, 256, QI2_XS, VDR_IQ2_XS_Q8_1_MMVQ, block_iq2_xs);
RD_MATVEC_TRAITS(TIQ2S_PERM, vec_dot_iq2_s_q8_1_perm, 256, QI2_S, VDR_IQ2_S_Q8_1_MMVQ, block_iq2_s);
RD_MATVEC_TRAITS(TIQ3XXS_PERM, vec_dot_iq3_xxs_q8_1_perm, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS(TIQ3XXS_PERM2, vec_dot_iq3_xxs_q8_1_perm2, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS(TIQ3XXS_NOSIGN, vec_dot_iq3_xxs_q8_1_diag_nosign, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS(TIQ3S_NOSIGN, vec_dot_iq3_s_q8_1_diag_nosign, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);
RD_MATVEC_TRAITS(TIQ3S_NOLOOKUP, vec_dot_iq3_s_q8_1_diag_nolookup, 256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s);

#undef RD_MATVEC_TRAITS

// ---------------------------------------------------------------------------
// LUT em LDS (docs/journal-kernels.md; docs/vulkan-vs-hip.md §1.3/§4.2: o
// backend Vulkan poe a iq3s_grid em `shared` e cobra os 2 KB no orcamento de
// LDS do GEMM). Aqui a MESMA tabela e copiada para a LDS uma vez por CTA e o
// `vec_dot` le de la em vez de fazer 8 `global_load_b32` por chamada.
//
// Bit-exato por construcao: a LDS recebe exatamente os mesmos bytes da tabela
// global (copiada com `LUT::src_words()`), e o corpo do `vec_dot` nao muda --
// so o endereco de onde o valor vem. Gate: `check-matvec-gpu --check-lds`
// (memcmp do resultado tipo a tipo contra o caminho de producao).
//
// `words` = palavras de 32 bits a copiar (0 = tipo sem LUT => a LDS nao existe
// e o kernel e' identico ao de producao).
struct NoLut {
  static constexpr int words = 0;
  static __device__ __forceinline__ const uint32_t *src_words() { return nullptr; }
};

#define RD_LUT(Name, Words, Table, ElemT)                                              \
  struct Name {                                                                        \
    static constexpr int words = Words;                                                \
    static __device__ __forceinline__ const uint32_t *src_words() {                    \
      return (const uint32_t *)Table;                                                  \
    }                                                                                  \
    using elem_t = ElemT;                                                              \
  }

RD_LUT(LutIq3S,   512,  iq3s_grid,    uint32_t);   //  2 KB
RD_LUT(LutIq3XXS, 256,  iq3xxs_grid,  uint32_t);   //  1 KB
RD_LUT(LutIq2XXS, 512,  iq2xxs_grid,  uint64_t);   //  2 KB
RD_LUT(LutIq2XS,  1024, iq2xs_grid,   uint64_t);   //  4 KB
RD_LUT(LutIq2S,   2048, iq2s_grid,    uint64_t);   //  8 KB
#undef RD_LUT

// Traits que leem a LUT de um ponteiro (a LDS) -- mesmo corpo de vec_dot.
#define RD_MATVEC_TRAITS_LDS(Name, Vd, VdLut, QK, QI, VDR, BlockT)                      \
  struct Name {                                                                         \
    static constexpr int qk = QK, qi = QI, vdr = VDR;                                   \
    using block_t = BlockT;                                                             \
    static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,   \
                                                const int &kbx, const int &iqs) {       \
      return Vd(vbq, a, kbx, iqs);                                                      \
    }                                                                                   \
    static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a, \
                                                    const int &kbx, const int &iqs,     \
                                                    const void *lut) {                  \
      return VdLut(vbq, a, kbx, iqs, lut);                                              \
    }                                                                                   \
  }

RD_MATVEC_TRAITS_LDS(TIQ3S_LDS,   vec_dot_iq3_s_q8_1_perm,      vec_dot_iq3_s_q8_1_perm_lut,      256, QI3_S,   VDR_IQ3_S_Q8_1_MMVQ,   block_iq3_s);
RD_MATVEC_TRAITS_LDS(TIQ3XXS_LDS, vec_dot_iq3_xxs_q8_1_perm2,   vec_dot_iq3_xxs_q8_1_perm2_lut,   256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ, block_iq3_xxs);
RD_MATVEC_TRAITS_LDS(TIQ2XXS_LDS, vec_dot_iq2_xxs_q8_1_perm2,   vec_dot_iq2_xxs_q8_1_perm2_lut,   256, QI2_XXS, VDR_IQ2_XXS_Q8_1_MMVQ, block_iq2_xxs);
RD_MATVEC_TRAITS_LDS(TIQ2XS_LDS,  vec_dot_iq2_xs_q8_1_perm2,    vec_dot_iq2_xs_q8_1_perm2_lut,    256, QI2_XS,  VDR_IQ2_XS_Q8_1_MMVQ,  block_iq2_xs);
RD_MATVEC_TRAITS_LDS(TIQ2S_LDS,   vec_dot_iq2_s_q8_1_perm2,     vec_dot_iq2_s_q8_1_perm2_lut,     256, QI2_S,   VDR_IQ2_S_Q8_1_MMVQ,   block_iq2_s);
#undef RD_MATVEC_TRAITS_LDS

// Traits do caminho em LOTE com a dequantizacao hoistada (ver o bloco "prep" em
// vecdotq.cuh): o gather da LUT + a mascara de sinal + o V_PERM saem do laco por
// token. Mesma aritmetica, mesma ordem -> bit-exato (check-matmul-gpu /
// check-batch-gpu).
#define RD_MATVEC_TRAITS_PREP(Name, Vd, VdLut, PrepT, PrepFn, DotPrepFn, QK, QI, VDR, BlockT, \
                              LutSrc)                                                   \
  struct Name {                                                                         \
    static constexpr int qk = QK, qi = QI, vdr = VDR;                                   \
    using block_t = BlockT;                                                             \
    using prep_t = PrepT;                                                               \
    static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,   \
                                                const int &kbx, const int &iqs) {       \
      return Vd(vbq, a, kbx, iqs);                                                      \
    }                                                                                   \
    static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a, \
                                                    const int &kbx, const int &iqs,     \
                                                    const void *lut) {                  \
      return VdLut(vbq, a, kbx, iqs, lut);                                              \
    }                                                                                   \
    static __device__ __forceinline__ const void *lut_source() { return (const void *)LutSrc; } \
    static __device__ __forceinline__ prep_t prep(const void *vbq, const int &kbx,      \
                                                  const int &iqs, const void *lut) {    \
      return PrepFn(vbq, kbx, iqs, lut);                                                \
    }                                                                                   \
    static __device__ __forceinline__ float dot_prep(const prep_t &w, const void *vbq,  \
                                                     const block_q8_1 *a, const int &kbx, \
                                                     const int &iqs, const void *lut) { \
      return DotPrepFn(w, vbq, a, kbx, iqs, lut);                                       \
    }                                                                                   \
  }

RD_MATVEC_TRAITS_PREP(TIQ3S_PREP, vec_dot_iq3_s_q8_1_perm, vec_dot_iq3_s_q8_1_perm_lut,
                      iq3s_prep_t, vec_prep_iq3_s_q8_1_perm, vec_dot_prep_iq3_s_q8_1_perm,
                      256, QI3_S, VDR_IQ3_S_Q8_1_MMVQ, block_iq3_s, iq3s_grid);
RD_MATVEC_TRAITS_PREP(TIQ3XXS_PREP, vec_dot_iq3_xxs_q8_1_perm2, vec_dot_iq3_xxs_q8_1_perm2_lut,
                      iq3xxs_prep_t, vec_prep_iq3_xxs_q8_1_perm2,
                      vec_dot_prep_iq3_xxs_q8_1_perm2, 256, QI3_XXS, VDR_IQ3_XXS_Q8_1_MMVQ,
                      block_iq3_xxs, iq3xxs_grid);
#undef RD_MATVEC_TRAITS_PREP

// O caminho em lote le a LUT global (no lote o gather ja' e' amortizado em N
// tokens, entao a LDS nao e' o que decide ali).
struct LutGlobal {
  static constexpr int words = 0;
  static __device__ __forceinline__ const uint32_t *src_words() { return nullptr; }
};

// Tipos que TEM kernel com LUT em LDS (os que carregam 60 % dos bytes do matvec).
inline bool matvec_has_lut_lds(int dt) {
  return dt == 7 || dt == 8 || dt == 9 || dt == 12 || dt == 13;
}

// L2 prefetch (from llama.cpp mmvq.cu, MIT).
//
// WARNING (measured, docs/rocha-estudo): `__builtin_prefetch` compiles to
// NOTHING on gfx1201 — the kernel it is called from contains no prefetch
// instruction at all (verified by dumping the ISA with --save-temps). So the
// PF=true variants below are only meaningful with the arch's own intrinsic,
// `__builtin_amdgcn_s_prefetch_data`. That one requires a WAVE-UNIFORM address
// (the compiler emits v_readfirstlane + s_prefetch_data), which the matvec
// happens to have: every lane of a warp walks the same weight row at the same
// block index. M2's "prefetch is neutral/harmful" was therefore a measurement of
// a no-op, and it is re-tested here with an instruction that actually executes.
static __device__ __forceinline__ void rdna4_prefetch_l2(const void *p) {
#if defined(__HIP_DEVICE_COMPILE__)
  __builtin_amdgcn_s_prefetch_data(p, 64);
#else
  (void)p;
#endif
}

// Generalized kernel: ROWS rows per CTA, WPR warps cooperating per row
// (NWARPS = ROWS*WPR). Covers both shapes:
//   (ROWS=4, WPR=1) -> the 4-warps-4-rows layout (best aggregate here)
//   (ROWS=1, WPR=8) -> llama.cpp's mmvq layout for ncols_dst=1
// Per-row-group indexing follows llama.cpp: kqs = vdr*(tg % (qi/vdr)),
// slot = tg / (qi/vdr), blocks_per_iter = vdr*(WPR*32)/qi, where tg is the
// thread index inside its row group.
// MINB > 0 emits __launch_bounds__(threads, MINB), which caps the register
// budget so more warps fit per CU. The lookup-heavy IQ vec_dots otherwise use
// 80-120 registers (iq4_xs: 119 -> only ~8 warps/CU) and become latency bound:
// read-only walks of the same blocks run at 1700 GB/s while the matvec gets 196.
//
// UNROLL > 1 processes UNROLL blocks per loop iteration into the SAME
// accumulator, in the SAME order the ILP=1 loop walks them: the two bodies are
// independent (different weight blocks, different activation blocks) so their
// loads and dp4a chains overlap, but the floating-point adds happen in exactly
// the sequence they would without the unroll. That makes it a bit-exact way to
// buy memory-level parallelism -- the thing PLAN.md M2 identified as the actual
// limiter ("the per-thread loop has little ILP (5-20 iterations, dependency
// sum += dot(...)), so memory latency is not hidden"). ILP does NOT have this
// property: it changes the summation order, so it stays a measured per-type
// constant.
// LUT = NoLut (producao ate hoje) ou uma das LutIq* acima: nesse caso a tabela do
// tipo e' copiada para a LDS uma vez por CTA e o vec_dot le de la. Tudo o mais
// (ROWS/WPR/ILP/UNROLL, ordem das somas) e' identico -- e' o unico jeito de
// atribuir uma diferenca medida a fonte da LUT e nao a outro knob.
template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0, int UNROLL = 1,
          class LUT = NoLut>
__global__ void
#if defined(__HIP_DEVICE_COMPILE__)
__launch_bounds__(ROWS * WPR * 32, MINB > 0 ? MINB : 1)
#endif
matvec_kernel_gen(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                  float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row) {
  static_assert(ILP >= 1, "ILP must be >= 1");
  static_assert(UNROLL >= 1, "UNROLL must be >= 1");
  static_assert(ILP == 1 || UNROLL == 1, "ILP and UNROLL are mutually exclusive");
  constexpr int PF_DIST = 2;  // prefetch distance, in loop iterations
  constexpr int vdr = T::vdr;
  constexpr int qi = T::qi;
  constexpr int qk = T::qk;
  constexpr int slots_per_block = qi / vdr;
  constexpr int blocks_per_iter = vdr * (WPR * 32) / qi;

  const int tid = threadIdx.x;          // 0 .. ROWS*WPR*32-1
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = warp / WPR;     // which of the CTA's rows
  const int w_in_group = warp % WPR;    // this warp's index inside the row group
  const int tg = w_in_group * 32 + lane;  // thread index inside the row group

  const int row_raw = (int)(blockIdx.x * ROWS + row_group);
  const bool active = row_raw < nrows;
  const int row = active ? row_raw : 0;  // inactive groups compute a dummy row

  const char *rowp =
      (const char *)vx + (int64_t)row * blocks_per_row * (int64_t)sizeof(typename T::block_t);
  const int kqs = vdr * (tg % slots_per_block);
  const int slot = tg / slots_per_block;

  // LUT em LDS: copia cooperativa no inicio do CTA (nenhum numero muda -- a LDS
  // recebe os mesmos bytes da tabela global) e uma barreira. Nao ha retorno
  // divergente acima deste ponto, entao a barreira e' segura.
  __shared__ uint32_t s_lut[LUT::words > 0 ? LUT::words : 1];
  if (LUT::words > 0) {
    const uint32_t *src = LUT::src_words();
    for (int i = tid; i < LUT::words; i += ROWS * WPR * 32) s_lut[i] = src[i];
    __syncthreads();
  }

  // ILP independent accumulators: the naive loop is one dependent chain of
  // float adds, which leaves memory latency exposed.
  float acc[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) acc[i] = 0.0f;

  int64_t kb = slot;
  if (UNROLL == 1) {
    for (; kb + (ILP - 1) * blocks_per_iter < blocks_per_row; kb += ILP * blocks_per_iter) {
      if (PF) {
        const int64_t kp = kb + PF_DIST * blocks_per_iter;
        if (kp < blocks_per_row) {
          rdna4_prefetch_l2((const char *)rowp + kp * (int64_t)sizeof(typename T::block_t));
        }
      }
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        acc[u] += T::dot_lut((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs,
                             (const void *)s_lut);
      }
    }
  } else {
    // Single accumulator, UNROLL blocks per iteration, walked in the SAME order
    // the ILP=1 loop walks them: the arithmetic of every output element is
    // unchanged (same ops, same order), only the memory-level parallelism grows.
    for (; kb + (UNROLL - 1) * blocks_per_iter < blocks_per_row; kb += UNROLL * blocks_per_iter) {
      if (PF) {
        const int64_t kp = kb + PF_DIST * blocks_per_iter;
        if (kp < blocks_per_row) {
          rdna4_prefetch_l2((const char *)rowp + kp * (int64_t)sizeof(typename T::block_t));
        }
      }
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        acc[0] += T::dot_lut((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs,
                             (const void *)s_lut);
      }
    }
  }
  for (; kb < blocks_per_row; kb += blocks_per_iter) {
    acc[0] += T::dot_lut((const void *)rowp, vy + kb * (qk / QK8_1), (const int)kb, kqs,
                         (const void *)s_lut);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ILP; ++i) sum += acc[i];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);

  if (WPR == 1) {
    if (active && lane == 0) dst[row] = sum;
    return;
  }
  __shared__ float part[ROWS][WPR];
  if (lane == 0) part[row_group][w_in_group] = sum;
  __syncthreads();
  if (active && w_in_group == 0 && lane == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < WPR; ++i) t += part[row_group][i];
    dst[row] = t;
  }
}


// ---------------------------------------------------------------------------
// Batched matvec (PLAN.md M6): N activation vectors share ONE pass over the
// weights, so the weight traffic per token drops by N. This is what makes prefill
// fast without touching the decode path.
//
// Bit-exactness by construction: for a given token the k-walk, the ILP slots, the
// per-slot sum and both reduction stages are byte-for-byte the same operations in
// the same order as matvec_kernel_gen, so each output element equals what N
// separate GEMV calls produce (checked by tests/check_matmul_gpu.hip). That is
// what lets a validated graph switch to it without invalidating the M2/M3 gates.
//
//   vy         : N activation rows of q8_1 blocks, row stride `act_stride` blocks
//   dst        : N rows of `nrows` floats, row stride `nrows` (token-major)
// ---------------------------------------------------------------------------
template <class T, int ROWS, int WPR, int ILP = 1, int N = 1, int UNROLL = 1>
__global__ void
#if defined(__HIP_DEVICE_COMPILE__)
__launch_bounds__(ROWS * WPR * 32, 1)
#endif
matvec_kernel_batch(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                    float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row,
                    int64_t act_stride) {
  static_assert(ILP >= 1, "ILP must be >= 1");
  // UNROLL e' o MESMO knob de MLP bit-exato do matvec_kernel_gen: processa UNROLL
  // blocos por iteracao no MESMO acumulador e na MESMA ordem do caminho por token
  // (as somas de um elemento de saida acontecem exatamente na sequencia em que
  // aconteceriam sem o unroll; so' ha mais cargas em voo). O caminho por token ja
  // o usa onde kMtIlp == 1 (kMtUnroll=2 para iq3_s, iq3_xxs, iq2_s, iq4_nl -- 59 %
  // dos bytes deste modelo) e o lote nao o tinha: era a diferenca estrutural entre
  // os dois caminhos, e e' o que a medicao de N=1..16 (pass ~ 13,9 + 6,04N) manda
  // atacar, porque o custo por token NAO e' trafego (o controle de reuso da
  // ativacao mostra 6 % de ganho) e sim latencia/issue.
  static_assert(UNROLL >= 1, "UNROLL must be >= 1");
  static_assert(ILP == 1 || UNROLL == 1, "ILP and UNROLL are mutually exclusive");
  constexpr int vdr = T::vdr;
  constexpr int qi = T::qi;
  constexpr int qk = T::qk;
  constexpr int slots_per_block = qi / vdr;
  constexpr int blocks_per_iter = vdr * (WPR * 32) / qi;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = warp / WPR;
  const int w_in_group = warp % WPR;
  const int tg = w_in_group * 32 + lane;

  const int row_raw = (int)(blockIdx.x * ROWS + row_group);
  const bool active = row_raw < nrows;
  const int row = active ? row_raw : 0;

  const char *rowp = (const char *)vx +
                     (int64_t)row * blocks_per_row * (int64_t)sizeof(typename T::block_t);
  const int kqs = vdr * (tg % slots_per_block);
  const int slot = tg / slots_per_block;

  float acc[N][ILP];
#pragma unroll
  for (int n = 0; n < N; ++n) {
#pragma unroll
    for (int i = 0; i < ILP; ++i) acc[n][i] = 0.0f;
  }

  // AMPLIFICACAO EM LOTE: a dequantizacao do peso (gather da LUT, mascara de sinal,
  // V_PERM) e' feita UMA vez por bloco com `T::prep` e o resultado em registrador e'
  // consumido pelas N linhas de ativacao com `T::dot_prep` -- so' o dp4a e' por
  // token. Para os tipos sem prep (`NoPrep`), `prep`/`dot_prep` chamam exatamente o
  // `dot` de antes, entao a sequencia de operacoes e' IDENTICA e o resultado segue
  // bit-exato (gate: check-matmul-gpu e check-batch-gpu, os dois memcmp/rel-L2 0).
  int64_t kb = slot;
  if (UNROLL == 1) {
    for (; kb + (ILP - 1) * blocks_per_iter < blocks_per_row; kb += ILP * blocks_per_iter) {
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        const block_q8_1 *abase = vy + k * (qk / QK8_1);
        const typename T::prep_t pre =
            T::prep((const void *)rowp, (const int)k, kqs, (const void *)T::lut_source());
#pragma unroll
        for (int n = 0; n < N; ++n) {
          acc[n][u] += T::dot_prep(pre, (const void *)rowp, abase + (int64_t)n * act_stride,
                                   (const int)k, kqs, (const void *)T::lut_source());
        }
      }
    }
  } else {
    for (; kb + (UNROLL - 1) * blocks_per_iter < blocks_per_row; kb += UNROLL * blocks_per_iter) {
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        const block_q8_1 *abase = vy + k * (qk / QK8_1);
        const typename T::prep_t pre =
            T::prep((const void *)rowp, (const int)k, kqs, (const void *)T::lut_source());
#pragma unroll
        for (int n = 0; n < N; ++n) {
          acc[n][0] += T::dot_prep(pre, (const void *)rowp, abase + (int64_t)n * act_stride,
                                   (const int)k, kqs, (const void *)T::lut_source());
        }
      }
    }
  }
  for (; kb < blocks_per_row; kb += blocks_per_iter) {
    const block_q8_1 *abase = vy + kb * (qk / QK8_1);
    const typename T::prep_t pre =
        T::prep((const void *)rowp, (const int)kb, kqs, (const void *)T::lut_source());
#pragma unroll
    for (int n = 0; n < N; ++n) {
      acc[n][0] += T::dot_prep(pre, (const void *)rowp, abase + (int64_t)n * act_stride,
                               (const int)kb, kqs, (const void *)T::lut_source());
    }
  }

  float sum[N];
#pragma unroll
  for (int n = 0; n < N; ++n) {
    sum[n] = 0.0f;
#pragma unroll
    for (int i = 0; i < ILP; ++i) sum[n] += acc[n][i];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) sum[n] += __shfl_xor(sum[n], off);
  }

  if (WPR == 1) {
    if (active && lane == 0) {
#pragma unroll
      for (int n = 0; n < N; ++n) dst[(int64_t)n * nrows + row] = sum[n];
    }
    return;
  }
  __shared__ float part[ROWS][WPR][N];
  if (lane == 0) {
#pragma unroll
    for (int n = 0; n < N; ++n) part[row_group][w_in_group][n] = sum[n];
  }
  __syncthreads();
  if (active && w_in_group == 0 && lane == 0) {
#pragma unroll
    for (int n = 0; n < N; ++n) {
      float t = 0.0f;
#pragma unroll
      for (int i = 0; i < WPR; ++i) t += part[row_group][i][n];
      dst[(int64_t)n * nrows + row] = t;
    }
  }
}

// ---------------------------------------------------------------------------
// Host launch helpers
// ---------------------------------------------------------------------------
struct MatvecShape {
  int qk;         // elements per weight block
  int block_bytes;
};

// dtype ordinals follow rdna4::DType (include/rdna4/dtype.h).
inline bool matvec_shape(int dt, MatvecShape &out) {
  switch (dt) {
    case 1:  out = {32, (int)sizeof(block_q8_0)};   return true;   // Q8_0
    case 2:  out = {256, (int)sizeof(block_q2_K)};  return true;
    case 3:  out = {256, (int)sizeof(block_q3_K)};  return true;
    case 4:  out = {256, (int)sizeof(block_q4_K)};  return true;
    case 5:  out = {256, (int)sizeof(block_q5_K)};  return true;
    case 6:  out = {256, (int)sizeof(block_q6_K)};  return true;
    case 7:  out = {256, (int)sizeof(block_iq2_xxs)}; return true;
    case 8:  out = {256, (int)sizeof(block_iq2_xs)};  return true;
    case 9:  out = {256, (int)sizeof(block_iq3_xxs)}; return true;
    case 10: out = {256, (int)sizeof(block_iq1_s)};   return true;
    case 11: out = {32,  (int)sizeof(block_iq4_nl)};  return true;
    case 12: out = {256, (int)sizeof(block_iq3_s)};   return true;
    case 13: out = {256, (int)sizeof(block_iq2_s)};   return true;
    case 14: out = {256, (int)sizeof(block_iq4_xs)};  return true;
    default: return false;  // F32 and anything outside the M1 union
  }
}

// Matvec launch configuration: rows per CTA and warps cooperating per row.
//
// Chosen by measurement on gfx1201 (RX 9070 XT): `check-matvec-gpu --bench`
// sweeps 8 shapes x 14 types on the largest tensor of each type, 3 runs
// averaged (full table in PLAN.md "Passo 5").
//
// NOTE on benchmarking this GPU: it drops to a deep DPM state between kernels
// (SCLK observed at 9-16 MHz), so a short warmup measures the clock ramp, not
// the kernel. The bench warms up until ~300 ms of GPU time has elapsed and
// then times 50 iterations; without that, every number here is ~1.5x too low.
struct MatvecConfig {
  int rows;  // rows per CTA
  int wpr;   // warps cooperating per row
};


template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0, int UNROLL = 1,
          class LUT = NoLut>
inline bool launch_gen(const void *d_weights, const block_q8_1 *d_act, float *d_out, int64_t nrows,
                       int64_t bpr, hipStream_t stream) {
  const int grid = (int)((nrows + ROWS - 1) / ROWS);
  const int threads = ROWS * WPR * 32;
  matvec_kernel_gen<T, ROWS, WPR, ILP, PF, MINB, UNROLL, LUT>
      <<<grid, threads, 0, stream>>>(d_weights, d_act, d_out, nrows, bpr);
  return hipGetLastError() == hipSuccess;
}


// The per-type numbers live in exactly one place: include/rdna4/tuning.h
// (namespace rdna4::tuned), which check-tuning compares against
// tests/golden/ml_tuning.txt. These tables only give them type-level names so the
// dispatch below can use them as template arguments.
template <int Dt> struct MtShape;             // { rows, wpr } per dtype ordinal
template <> struct MtShape<1>  { static constexpr int rows = tuned::kMtRows[0], wpr = tuned::kMtWpr[0]; };   // q8_0
template <> struct MtShape<2>  { static constexpr int rows = tuned::kMtRows[1], wpr = tuned::kMtWpr[1]; };   // q2_K
template <> struct MtShape<3>  { static constexpr int rows = tuned::kMtRows[2], wpr = tuned::kMtWpr[2]; };   // q3_K
template <> struct MtShape<4>  { static constexpr int rows = tuned::kMtRows[3], wpr = tuned::kMtWpr[3]; };   // q4_K
template <> struct MtShape<5>  { static constexpr int rows = tuned::kMtRows[4], wpr = tuned::kMtWpr[4]; };   // q5_K
template <> struct MtShape<6>  { static constexpr int rows = tuned::kMtRows[5], wpr = tuned::kMtWpr[5]; };   // q6_K
template <> struct MtShape<7>  { static constexpr int rows = tuned::kMtRows[6], wpr = tuned::kMtWpr[6]; };   // iq2_xxs
template <> struct MtShape<8>  { static constexpr int rows = tuned::kMtRows[7], wpr = tuned::kMtWpr[7]; };   // iq2_xs
template <> struct MtShape<9>  { static constexpr int rows = tuned::kMtRows[8], wpr = tuned::kMtWpr[8]; };   // iq3_xxs
template <> struct MtShape<10> { static constexpr int rows = tuned::kMtRows[9], wpr = tuned::kMtWpr[9]; };   // iq1_s
template <> struct MtShape<11> { static constexpr int rows = tuned::kMtRows[10], wpr = tuned::kMtWpr[10]; }; // iq4_nl
template <> struct MtShape<12> { static constexpr int rows = tuned::kMtRows[11], wpr = tuned::kMtWpr[11]; }; // iq3_s
template <> struct MtShape<13> { static constexpr int rows = tuned::kMtRows[12], wpr = tuned::kMtWpr[12]; }; // iq2_s
template <> struct MtShape<14> { static constexpr int rows = tuned::kMtRows[13], wpr = tuned::kMtWpr[13]; }; // iq4_xs

// ILP (independent accumulators) per type, measured with --bench-tune.
// Helps the latency-bound types a lot (q4_K 610->739 GB/s), does nothing or
// slightly regresses the ALU-bound IQ types, so those stay at 1.
template <int Dt> struct MtIlp { static constexpr int value = 1; };
template <> struct MtIlp<1>  { static constexpr int value = tuned::kMtIlp[0]; };   // q8_0
template <> struct MtIlp<2>  { static constexpr int value = tuned::kMtIlp[1]; };   // q2_K
template <> struct MtIlp<3>  { static constexpr int value = tuned::kMtIlp[2]; };   // q3_K
template <> struct MtIlp<4>  { static constexpr int value = tuned::kMtIlp[3]; };   // q4_K
template <> struct MtIlp<5>  { static constexpr int value = tuned::kMtIlp[4]; };   // q5_K
template <> struct MtIlp<6>  { static constexpr int value = tuned::kMtIlp[5]; };   // q6_K
template <> struct MtIlp<7>  { static constexpr int value = tuned::kMtIlp[6]; };   // iq2_xxs
template <> struct MtIlp<8>  { static constexpr int value = tuned::kMtIlp[7]; };   // iq2_xs
template <> struct MtIlp<9>  { static constexpr int value = tuned::kMtIlp[8]; };   // iq3_xxs
template <> struct MtIlp<10> { static constexpr int value = tuned::kMtIlp[9]; };   // iq1_s
template <> struct MtIlp<11> { static constexpr int value = tuned::kMtIlp[10]; };  // iq4_nl
template <> struct MtIlp<12> { static constexpr int value = tuned::kMtIlp[11]; };  // iq3_s
template <> struct MtIlp<13> { static constexpr int value = tuned::kMtIlp[12]; };  // iq2_s
template <> struct MtIlp<14> { static constexpr int value = tuned::kMtIlp[13]; };  // iq4_xs

// UNROLL (blocks per iteration into the SAME accumulator) per type: the
// bit-exact MLP lever (same ops, same order, more loads in flight). Only used
// where ILP is 1 -- the kernel static_asserts that the two are exclusive, and
// that is why iq3_s/iq3_xxs/iq2_s/iq4_nl bought their +3..8% by moving from
// ILP to UNROLL.
template <int Dt> struct MtUnroll { static constexpr int value = 1; };
template <> struct MtUnroll<1>  { static constexpr int value = tuned::kMtUnroll[0]; };   // q8_0
template <> struct MtUnroll<2>  { static constexpr int value = tuned::kMtUnroll[1]; };   // q2_K
template <> struct MtUnroll<3>  { static constexpr int value = tuned::kMtUnroll[2]; };   // q3_K
template <> struct MtUnroll<4>  { static constexpr int value = tuned::kMtUnroll[3]; };   // q4_K
template <> struct MtUnroll<5>  { static constexpr int value = tuned::kMtUnroll[4]; };   // q5_K
template <> struct MtUnroll<6>  { static constexpr int value = tuned::kMtUnroll[5]; };   // q6_K
template <> struct MtUnroll<7>  { static constexpr int value = tuned::kMtUnroll[6]; };   // iq2_xxs
template <> struct MtUnroll<8>  { static constexpr int value = tuned::kMtUnroll[7]; };   // iq2_xs
template <> struct MtUnroll<9>  { static constexpr int value = tuned::kMtUnroll[8]; };   // iq3_xxs
template <> struct MtUnroll<10> { static constexpr int value = tuned::kMtUnroll[9]; };   // iq1_s
template <> struct MtUnroll<11> { static constexpr int value = tuned::kMtUnroll[10]; };  // iq4_nl
template <> struct MtUnroll<12> { static constexpr int value = tuned::kMtUnroll[11]; };  // iq3_s
template <> struct MtUnroll<13> { static constexpr int value = tuned::kMtUnroll[12]; };  // iq2_s
template <> struct MtUnroll<14> { static constexpr int value = tuned::kMtUnroll[13]; };  // iq4_xs

// MINB (blocos minimos por multiprocessador no __launch_bounds__) por tipo: o
// cap de registradores bit-exato (so muda ocupacao/spills, nunca a aritmetica).
// 0 = comportamento atual (sem cap). Valores >0 precisam de sonda de spill
// limpa (localSizeBytes==0) + gate do oraculo -- ver tuning.h (kMtMinb).
template <int Dt> struct MtMinb { static constexpr int value = 0; };
template <> struct MtMinb<1>  { static constexpr int value = tuned::kMtMinb[0]; };   // q8_0
template <> struct MtMinb<2>  { static constexpr int value = tuned::kMtMinb[1]; };   // q2_K
template <> struct MtMinb<3>  { static constexpr int value = tuned::kMtMinb[2]; };   // q3_K
template <> struct MtMinb<4>  { static constexpr int value = tuned::kMtMinb[3]; };   // q4_K
template <> struct MtMinb<5>  { static constexpr int value = tuned::kMtMinb[4]; };   // q5_K
template <> struct MtMinb<6>  { static constexpr int value = tuned::kMtMinb[5]; };   // q6_K
template <> struct MtMinb<7>  { static constexpr int value = tuned::kMtMinb[6]; };   // iq2_xxs
template <> struct MtMinb<8>  { static constexpr int value = tuned::kMtMinb[7]; };   // iq2_xs
template <> struct MtMinb<9>  { static constexpr int value = tuned::kMtMinb[8]; };   // iq3_xxs
template <> struct MtMinb<10> { static constexpr int value = tuned::kMtMinb[9]; };   // iq1_s
template <> struct MtMinb<11> { static constexpr int value = tuned::kMtMinb[10]; };  // iq4_nl
template <> struct MtMinb<12> { static constexpr int value = tuned::kMtMinb[11]; };  // iq3_s
template <> struct MtMinb<13> { static constexpr int value = tuned::kMtMinb[12]; };  // iq2_s
template <> struct MtMinb<14> { static constexpr int value = tuned::kMtMinb[13]; };  // iq4_xs

inline MatvecConfig matvec_default_config(int dt) {
  // Derived from the compile-time tables so the reported/shipping shape and the
  // instantiated kernel can never disagree (review finding L7).
  switch (dt) {
    case 1:  return {MtShape<1>::rows,  MtShape<1>::wpr};
    case 2:  return {MtShape<2>::rows,  MtShape<2>::wpr};
    case 3:  return {MtShape<3>::rows,  MtShape<3>::wpr};
    case 4:  return {MtShape<4>::rows,  MtShape<4>::wpr};
    case 5:  return {MtShape<5>::rows,  MtShape<5>::wpr};
    case 6:  return {MtShape<6>::rows,  MtShape<6>::wpr};
    case 7:  return {MtShape<7>::rows,  MtShape<7>::wpr};
    case 8:  return {MtShape<8>::rows,  MtShape<8>::wpr};
    case 9:  return {MtShape<9>::rows,  MtShape<9>::wpr};
    case 10: return {MtShape<10>::rows, MtShape<10>::wpr};
    case 11: return {MtShape<11>::rows, MtShape<11>::wpr};
    case 12: return {MtShape<12>::rows, MtShape<12>::wpr};
    case 13: return {MtShape<13>::rows, MtShape<13>::wpr};
    case 14: return {MtShape<14>::rows, MtShape<14>::wpr};
    default: return {4, 1};
  }
}

// UNROLL the shipping path uses for this type (see MtUnroll / tuning.h).
inline int matvec_default_unroll(int dt) {
  switch (dt) {
    case 1:  return MtUnroll<1>::value;   case 2:  return MtUnroll<2>::value;
    case 3:  return MtUnroll<3>::value;   case 4:  return MtUnroll<4>::value;
    case 5:  return MtUnroll<5>::value;   case 6:  return MtUnroll<6>::value;
    case 7:  return MtUnroll<7>::value;   case 8:  return MtUnroll<8>::value;
    case 9:  return MtUnroll<9>::value;   case 10: return MtUnroll<10>::value;
    case 11: return MtUnroll<11>::value;  case 12: return MtUnroll<12>::value;
    case 13: return MtUnroll<13>::value;  case 14: return MtUnroll<14>::value;
    default: return 1;
  }
}

inline int matvec_default_ilp(int dt) {
  switch (dt) {
    case 1:  return MtIlp<1>::value;   case 2:  return MtIlp<2>::value;
    case 3:  return MtIlp<3>::value;   case 4:  return MtIlp<4>::value;
    case 5:  return MtIlp<5>::value;   case 6:  return MtIlp<6>::value;
    case 7:  return MtIlp<7>::value;   case 8:  return MtIlp<8>::value;
    case 9:  return MtIlp<9>::value;   case 10: return MtIlp<10>::value;
    case 11: return MtIlp<11>::value;  case 12: return MtIlp<12>::value;
    case 13: return MtIlp<13>::value;  case 14: return MtIlp<14>::value;
    default: return 1;
  }
}

// Shipping path: shape and ILP are compile-time per type (both measured), so
// this instantiates exactly one kernel per type.
inline bool matvec_launch_lut_lds(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                  int64_t nrows, int64_t ncols, hipStream_t stream);
inline bool matvec_lut_lds_ok(int dt, int64_t nrows, int64_t ncols);

inline bool matvec_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                          int64_t nrows, int64_t ncols, hipStream_t stream) {
  // LUT em LDS quando ela e' pequena em relacao aos pesos lidos por CTA: medido
  // 1,070x em iq3_s e 1,182x em iq3_xxs, bit-exato (check-matvec-gpu --check-lds
  // compara com memcmp). iq2_*/iq2_s ficam de fora pela razao -- ver
  // matvec_lut_lds_ok. Nada mais muda: mesma forma, mesma aritmetica.
  if (matvec_lut_lds_ok(dt, nrows, ncols)) return matvec_launch_lut_lds(dt, d_w, d_a, d_o, nrows, ncols, stream);
#define RD_SHIP(Traits, Dt, QK)                                                            \
  case Dt:                                                                                 \
    return launch_gen<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, MtIlp<Dt>::value, false, \
                      MtMinb<Dt>::value, MtUnroll<Dt>::value>(d_w, d_a, d_o, nrows, ncols / QK, stream);
  switch (dt) {
    RD_SHIP(TQ8_0, 1, 32)
    RD_SHIP(TQ2K, 2, 256)
    RD_SHIP(TQ3K, 3, 256)
    RD_SHIP(TQ4K, 4, 256)
    RD_SHIP(TQ5K, 5, 256)
    RD_SHIP(TQ6K, 6, 256)
    RD_SHIP(TIQ2XXS_S, 7, 256)
    RD_SHIP(TIQ2XS_S, 8, 256)
    RD_SHIP(TIQ3XXS_S, 9, 256)
    RD_SHIP(TIQ1S, 10, 256)
    RD_SHIP(TIQ4NL, 11, 32)
    RD_SHIP(TIQ3S_S, 12, 256)
    RD_SHIP(TIQ2S_S, 13, 256)
    RD_SHIP(TIQ4XS, 14, 256)
    default:
      return false;  // no kernel: caller must fail loudly (SPEC 1.3)
  }
#undef RD_SHIP
}

// ---------------------------------------------------------------------------
// Caminho com a LUT do tipo em LDS. Mesma forma (ROWS/WPR/ILP/UNROLL) do caminho
// de producao: a unica diferenca e' de onde vem a tabela, o que e' o que torna a
// medicao atribuivel (e o resultado bit-exato -- ver `check-matvec-gpu
// --check-lds`, que compara os dois caminhos com memcmp).
//
// Tipos cobertos: os IQ que usam LUT e carregam os bytes de verdade
// (iq3_s, iq3_xxs, iq2_s, iq2_xs, iq2_xxs = 60 % do trafego do matvec). Os
// outros caem no caminho de producao, entao chamar isto para qualquer dt e'
// seguro.
// ---------------------------------------------------------------------------
// Bytes da LUT de cada tipo IQ (0 = sem kernel de LDS). Ver `LutIq*` acima.
inline int matvec_lut_bytes(int dt) {
  switch (dt) {
    case 7:  return 256 * 8;    // iq2_xxs
    case 8:  return 512 * 8;    // iq2_xs
    case 9:  return 256 * 4;    // iq3_xxs
    case 12: return 512 * 4;    // iq3_s
    case 13: return 1024 * 8;   // iq2_s
    default: return 0;
  }
}

// A LDS so' compensa quando a tabela e' pequena em relacao aos pesos que CADA
// CTA le: a copia e' por CTA (ROWS*bpr*block_bytes de peso). Medido no
// inventario real (docs/journal-kernels.md §2, piso de ruido 0,999x):
//   iq3_s    2 KB de LUT vs 17,6 KB de peso por CTA -> 1,070x   (GANHO)
//   iq3_xxs  1 KB        vs  3,9 KB                 -> 1,182x   (GANHO)
//   iq2_xxs  2 KB        vs  1,3 KB                 -> 0,914x   (perde)
//   iq2_xs   4 KB        vs  1,5 KB                 -> 0,680x   (perde)
//   iq2_s    8 KB        vs  3,3 KB                 -> 0,601x   (perde)
// O corte em 2x separa exatamente os dois grupos (razoes 3,9x/8,8x contra
// 0,37x/0,41x/0,66x) e e' conservador: ele so' impede um caso que ja' foi medido
// como perda.
inline bool matvec_lut_lds_ok(int dt, int64_t nrows, int64_t ncols) {
  const int lut_bytes = matvec_lut_bytes(dt);
  if (lut_bytes == 0 || nrows <= 0) return false;
  MatvecShape shape{};
  if (!matvec_shape(dt, shape) || shape.qk <= 0) return false;
  const int64_t bpr = ncols / shape.qk;
  const int64_t rows = matvec_default_config(dt).rows;
  const int64_t weights_per_cta = rows * bpr * shape.block_bytes;
  return weights_per_cta >= 2 * (int64_t)lut_bytes;
}

inline bool matvec_launch_lut_lds(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                  int64_t nrows, int64_t ncols, hipStream_t stream) {
#define RD_LDS(TraitsLds, Dt, QK, LutT)                                                        \
  case Dt: {                                                                                    \
    constexpr int R = MtShape<Dt>::rows, W = MtShape<Dt>::wpr, I = MtIlp<Dt>::value,            \
                  U = MtUnroll<Dt>::value;                                                      \
    return launch_gen<TraitsLds, R, W, I, false, MtMinb<Dt>::value, U, LutT>(                   \
        d_w, d_a, d_o, nrows, ncols / QK, stream);                                               \
  }
  switch (dt) {
    RD_LDS(TIQ2XXS_LDS, 7, 256, LutIq2XXS)
    RD_LDS(TIQ2XS_LDS, 8, 256, LutIq2XS)
    RD_LDS(TIQ3XXS_LDS, 9, 256, LutIq3XXS)
    RD_LDS(TIQ3S_LDS, 12, 256, LutIq3S)
    RD_LDS(TIQ2S_LDS, 13, 256, LutIq2S)
    default:
      return matvec_launch(dt, d_w, d_a, d_o, nrows, ncols, stream);
  }
#undef RD_LDS
}

// Mesmo caminho, forcando UNROLL (blocos por iteracao no MESMO acumulador; a
// alavanca bit-exata de paralelismo de memoria). Existe para medir se o unroll
// que a tabela rejeitou com a LUT GLOBAL passa a compensar com a LUT na LDS --
// o orcamento de registradores e de espera de load muda quando o gather sai do
// caminho global (ver docs/journal-kernels.md §1.3).
inline bool matvec_launch_lut_lds_unroll(int dt, const void *d_w, const block_q8_1 *d_a,
                                         float *d_o, int64_t nrows, int64_t ncols,
                                         hipStream_t stream, int unroll) {
#define RD_LDSU(TraitsLds, Dt, QK, LutT)                                                       \
  case Dt: {                                                                                   \
    constexpr int R = MtShape<Dt>::rows, W = MtShape<Dt>::wpr;                                  \
    switch (unroll) {                                                                          \
      case 1: return launch_gen<TraitsLds, R, W, 1, false, 0, 1, LutT>(d_w, d_a, d_o, nrows,    \
                                                                       ncols / QK, stream);    \
      case 2: return launch_gen<TraitsLds, R, W, 1, false, 0, 2, LutT>(d_w, d_a, d_o, nrows,    \
                                                                       ncols / QK, stream);    \
      case 4: return launch_gen<TraitsLds, R, W, 1, false, 0, 4, LutT>(d_w, d_a, d_o, nrows,    \
                                                                       ncols / QK, stream);    \
      default: return false;                                                                   \
    }                                                                                          \
  }
  switch (dt) {
    RD_LDSU(TIQ2XXS_LDS, 7, 256, LutIq2XXS)
    RD_LDSU(TIQ2XS_LDS, 8, 256, LutIq2XS)
    RD_LDSU(TIQ3XXS_LDS, 9, 256, LutIq3XXS)
    RD_LDSU(TIQ3S_LDS, 12, 256, LutIq3S)
    RD_LDSU(TIQ2S_LDS, 13, 256, LutIq2S)
    default:
      // Tipo sem kernel de LDS: medir o caminho de producao (senao o bench
      // cronometraria um lancamento que nao acontece e reportaria "12x").
      return matvec_launch(dt, d_w, d_a, d_o, nrows, ncols, stream);
  }
#undef RD_LDSU
}



// Batched shipping path. N is a compile-time constant (2/4/8/16) and the shape
// per type is the *same* one the GEMV path uses, which is what makes the results
// bit-identical; n_tokens is capped at the largest instantiation and the caller
// loops over sub-batches. The batched kernel has no UNROLL knob (its `N` already
// provides the memory-level parallelism), and since UNROLL does not change the
// summation order the batched result stays bit-identical to the GEMV path for
// every type that ships with unroll>1 -- tests/check_batch_gpu.hip asserts it.
// 64 = teto das INSTANCIACOES (inclui N=32/64 bench-only do D1); contrato de
// producao = chunk_ok (graph.cuh:210): GEMV-batch so' n<=16 (H0).
inline int matvec_batch_cap() { return 64; }

template <int N>
inline bool matvec_launch_batch_n(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                  int64_t nrows, int64_t ncols, int64_t act_stride,
                                  hipStream_t stream) {
#define RD_BATCH(Traits, Dt, QK)                                                              \
  case Dt: {                                                                                  \
    const int grid = (int)((nrows + MtShape<Dt>::rows - 1) / MtShape<Dt>::rows);              \
    const int threads = MtShape<Dt>::rows * MtShape<Dt>::wpr * 32;                            \
    /* kMtUnroll NAO entra aqui: medido e neutro no lote (-1 % a +5 %, dentro do piso  */    \
    /* de ruido; unr2 e' 5 % PIOR em N=16). O knob fica no template + no lancador      */   \
    /* bench-only abaixo como registro da hipotese refutada -- ver docs/journal-lote.md. */   \
    constexpr int kIlp = MtIlp<Dt>::value;                                                    \
    constexpr int kUnr = 1;                                                                   \
    matvec_kernel_batch<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, kIlp, N, kUnr>           \
        <<<grid, threads, 0, stream>>>(d_w, d_a, d_o, nrows, ncols / QK, act_stride);         \
    return hipGetLastError() == hipSuccess;                                                   \
  }
  switch (dt) {
    RD_BATCH(TQ8_0, 1, 32)
    RD_BATCH(TQ2K, 2, 256)
    RD_BATCH(TQ3K, 3, 256)
    RD_BATCH(TQ4K, 4, 256)
    RD_BATCH(TQ5K, 5, 256)
    RD_BATCH(TQ6K, 6, 256)
    RD_BATCH(TIQ2XXS_S, 7, 256)
    RD_BATCH(TIQ2XS_S, 8, 256)
    RD_BATCH(TIQ3XXS_S, 9, 256)  // PREP: ver comentario abaixo
    RD_BATCH(TIQ1S, 10, 256)
    RD_BATCH(TIQ4NL, 11, 32)
    RD_BATCH(TIQ3S_S, 12, 256)  // PREP: ver comentario abaixo
    RD_BATCH(TIQ2S_S, 13, 256)
    RD_BATCH(TIQ4XS, 14, 256)
    default:
      return false;  // no kernel: caller must fail loudly (SPEC 1.3)
  }
#undef RD_BATCH
}

// ---------------------------------------------------------------------------
// Bench-only: o UNROLL do lote, forcado. Existe para MEDIR o knob de MLP
// bit-exato no caminho em lote nos tipos em que kMtIlp == 1 (os 59 % de bytes
// deste modelo: iq3_s, iq3_xxs, iq2_s). Instanciado so' para esses tipos e para
// N em {8,16} -- o lote de prefill -- para manter o numero de instanciacoes
// limitado. Nao e' caminho de producao: quem embarca e' matvec_launch_batch,
// que ja' usa kMtUnroll onde kMtIlp == 1.
// ---------------------------------------------------------------------------
template <int N, int UNROLL>
inline bool matvec_launch_batch_unroll_n(int dt, const void *d_w, const block_q8_1 *d_a,
                                         float *d_o, int64_t nrows, int64_t ncols,
                                         int64_t act_stride, hipStream_t stream) {
#define RD_BU(Traits, Dt, QK)                                                                 \
  case Dt: {                                                                                  \
    const int grid = (int)((nrows + MtShape<Dt>::rows - 1) / MtShape<Dt>::rows);              \
    const int threads = MtShape<Dt>::rows * MtShape<Dt>::wpr * 32;                            \
    matvec_kernel_batch<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, 1, N, UNROLL>            \
        <<<grid, threads, 0, stream>>>(d_w, d_a, d_o, nrows, ncols / QK, act_stride);         \
    return hipGetLastError() == hipSuccess;                                                   \
  }
  switch (dt) {
    RD_BU(TIQ3XXS_S, 9, 256)
    RD_BU(TIQ3S_S, 12, 256)
    RD_BU(TIQ2S_S, 13, 256)
    default: return false;
  }
#undef RD_BU
}

template <int N>
inline bool matvec_launch_batch_unroll_n2(int dt, const void *d_w, const block_q8_1 *d_a,
                                          float *d_o, int64_t nrows, int64_t ncols,
                                          int64_t act_stride, int unroll, hipStream_t stream) {
  switch (unroll) {
    case 1: return matvec_launch_batch_unroll_n<N, 1>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 2: return matvec_launch_batch_unroll_n<N, 2>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 4: return matvec_launch_batch_unroll_n<N, 4>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    default: return false;
  }
}

inline bool matvec_launch_batch_unroll(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                      int64_t nrows, int64_t ncols, int64_t act_stride,
                                      int n_tokens, int unroll, hipStream_t stream) {
  switch (n_tokens) {
    case 8: return matvec_launch_batch_unroll_n2<8>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, unroll, stream);
    case 16: return matvec_launch_batch_unroll_n2<16>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, unroll, stream);
    default: return false;
  }
}

// ---------------------------------------------------------------------------
// NAO LIGADO -- e a leitura de ISA mostra que NAO DEVE ser ligado: o compilador
// JA FAZ esse hoist. Medido com `--save-temps` no kernel em lote com N=16
// (docs/journal-kernels.md §10): por bloco, `v_perm_b32` aparece 8x (nao 8x16) e
// as cargas da LUT 8x (nao 8x16), enquanto `v_dot4_i32_iu8` e as cargas de
// ativacao escalam exatamente 16x. Ou seja: o lado do peso (gather da LUT,
// mascara de sinal, V_PERM) ja esta fora do laco por token, e o teto desta
// mudanca e' ~0 -- nao os ~10 % da contagem de instrucoes do §8.
//
// O par `T::prep`/`T::dot_prep` fica no arquivo como registro do beco sem saida
// (e como ponto de partida se um dia o `T::dot` deixar de ser CSE-ado pelo
// compilador), com o fallback `NoPrep` identico ao codigo anterior nos outros 12
// tipos. O que limita o lote e' o que NAO pode ser amortizado (dp4a + cargas de
// ativacao por token) e a latencia dessas cargas: IPC implicito ~0,15 do pico.
// ---------------------------------------------------------------------------

// `n_tokens` must be <= matvec_batch_cap(); d_a holds `n_tokens` activation rows
// of `act_stride` q8_1 blocks each (act_stride >= ncols / 32).
inline bool matvec_launch_batch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                int64_t nrows, int64_t ncols, int64_t act_stride, int n_tokens,
                                hipStream_t stream) {
  switch (n_tokens) {
    case 2:  return matvec_launch_batch_n<2>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 3:  return matvec_launch_batch_n<3>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 4:  return matvec_launch_batch_n<4>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 8:  return matvec_launch_batch_n<8>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 16: return matvec_launch_batch_n<16>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    // N=32/64 existem para MEDIR o degrau D1 do plano (chunk maior com o kernel de hoje).
    // A previsao do modelo de custo (13,9 ms + 6,04 ms/token) e' de apenas +10 % em N=64 --
    // previsao falsificavel, e e' isto que a mede. Custo: acc[N] floats = N registradores.
    case 32: return matvec_launch_batch_n<32>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    case 64: return matvec_launch_batch_n<64>(dt, d_w, d_a, d_o, nrows, ncols, act_stride, stream);
    default: return false;  // N is a compile-time instantiation, not a runtime knob
  }
}

// Occupancy diagnostic for the BATCHED kernel: register/local/shared usage of
// one (type, N) instantiation. Same purpose as matvec_kernel_attrs above; it is
// what turns "o lote e' lento" into "o lote gasta N registradores por token e
// transborda para memoria local", which is the first thing to rule out
// (docs/journal-noite.md, matvec em lote).
template <int N>
inline bool matvec_batch_kernel_attrs(int dt, hipFuncAttributes &attr) {
#define RD_BATCH_ATTR(Traits, Dt)                                                              \
  case Dt: {                                                                                   \
    constexpr int kIlp = MtIlp<Dt>::value;                                                     \
    constexpr int kUnr = kIlp == 1 ? MtUnroll<Dt>::value : 1;                                  \
    auto *fn = &matvec_kernel_batch<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, kIlp, N,      \
                                    kUnr>;                                                     \
    return hipFuncGetAttributes(&attr, (const void *)fn) == hipSuccess;                         \
  }
  switch (dt) {
    RD_BATCH_ATTR(TQ8_0, 1) RD_BATCH_ATTR(TQ2K, 2) RD_BATCH_ATTR(TQ3K, 3) RD_BATCH_ATTR(TQ4K, 4)
    RD_BATCH_ATTR(TQ5K, 5) RD_BATCH_ATTR(TQ6K, 6) RD_BATCH_ATTR(TIQ2XXS_S, 7)
    RD_BATCH_ATTR(TIQ2XS_S, 8) RD_BATCH_ATTR(TIQ3XXS_S, 9) RD_BATCH_ATTR(TIQ1S, 10)
    RD_BATCH_ATTR(TIQ4NL, 11) RD_BATCH_ATTR(TIQ3S_S, 12) RD_BATCH_ATTR(TIQ2S_S, 13)
    RD_BATCH_ATTR(TIQ4XS, 14)
    default: return false;
  }
#undef RD_BATCH_ATTR
}

// ---------------------------------------------------------------------------
// Tuned dispatch: the shipping shape per type is a compile-time constant (it is
// measured, see matvec_default_config) so that the ILP/prefetch knobs can be
// selected without a runtime shape switch, keeping instantiations bounded.
// ---------------------------------------------------------------------------
template <class T, int ROWS, int WPR>
inline bool launch_shape_tuned(const void *d_w, const block_q8_1 *d_a, float *d_o, int64_t nrows,
                               int64_t bpr, hipStream_t stream, int ilp, bool pf) {
  if (ilp == 1 && !pf) return launch_gen<T, ROWS, WPR, 1, false>(d_w, d_a, d_o, nrows, bpr, stream);
  if (ilp == 2 && !pf) return launch_gen<T, ROWS, WPR, 2, false>(d_w, d_a, d_o, nrows, bpr, stream);
  if (ilp == 4 && !pf) return launch_gen<T, ROWS, WPR, 4, false>(d_w, d_a, d_o, nrows, bpr, stream);
  if (ilp == 1 &&  pf) return launch_gen<T, ROWS, WPR, 1, true >(d_w, d_a, d_o, nrows, bpr, stream);
  if (ilp == 2 &&  pf) return launch_gen<T, ROWS, WPR, 2, true >(d_w, d_a, d_o, nrows, bpr, stream);
  if (ilp == 4 &&  pf) return launch_gen<T, ROWS, WPR, 4, true >(d_w, d_a, d_o, nrows, bpr, stream);
  return false;
}

// Sweeps the ILP/prefetch knobs on each type's shipping shape (bench use).
inline bool matvec_launch_tuned(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                int64_t nrows, int64_t ncols, hipStream_t stream, int ilp, bool pf) {
#define RD_TUNED(Traits, Dt, QK)                                                          \
  case Dt:                                                                                \
    return launch_shape_tuned<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr>(               \
        d_w, d_a, d_o, nrows, ncols / QK, stream, ilp, pf);
  switch (dt) {
    RD_TUNED(TQ8_0, 1, 32)
    RD_TUNED(TQ2K, 2, 256)
    RD_TUNED(TQ3K, 3, 256)
    RD_TUNED(TQ4K, 4, 256)
    RD_TUNED(TQ5K, 5, 256)
    RD_TUNED(TQ6K, 6, 256)
    RD_TUNED(TIQ2XXS, 7, 256)
    RD_TUNED(TIQ2XS, 8, 256)
    RD_TUNED(TIQ3XXS, 9, 256)
    RD_TUNED(TIQ1S, 10, 256)
    RD_TUNED(TIQ4NL, 11, 32)
    RD_TUNED(TIQ3S, 12, 256)
    RD_TUNED(TIQ2S, 13, 256)
    RD_TUNED(TIQ4XS, 14, 256)
    default:
      return false;  // no kernel: caller must fail loudly (SPEC 1.3)
  }
#undef RD_TUNED
}


// ---------------------------------------------------------------------------
// Diagnostic: walks the exact same blocks as the matvec (same row/block/thread
// mapping) but only reads the weights, so the measured bandwidth is the ceiling
// of this access pattern. Used to tell "memory pattern" from "ALU" limits
// (bench: check-matvec-gpu --bench-read).
// ---------------------------------------------------------------------------
template <class T, int ROWS, int WPR>
__global__ void read_only_kernel(const void *__restrict__ vx, float *__restrict__ dst,
                                 int64_t nrows, int64_t blocks_per_row) {
  constexpr int qi = T::qi;
  constexpr int vdr = T::vdr;
  constexpr int slots_per_block = qi / vdr;
  constexpr int blocks_per_iter = vdr * (WPR * 32) / qi;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = warp / WPR;
  const int w_in_group = warp % WPR;
  const int tg = w_in_group * 32 + lane;

  const int row_raw = (int)(blockIdx.x * ROWS + row_group);
  const bool active = row_raw < nrows;
  const int row = active ? row_raw : 0;
  const char *rowp =
      (const char *)vx + (int64_t)row * blocks_per_row * (int64_t)sizeof(typename T::block_t);
  const int slot = tg / slots_per_block;
  const int lane_in_block = tg % slots_per_block;
  constexpr int block_u32 = (int)(sizeof(typename T::block_t) / 4);

  // Read EVERY byte of every visited block: the warp's slots_per_block threads
  // walk the block's uint32s cooperatively. (Reading only one uint32 per thread
  // touches ~20% of a large block while still being credited with the whole
  // tensor size, which overstates the ceiling by up to 5x.)
  uint32_t acc = 0;
  for (int64_t kb = slot; kb < blocks_per_row; kb += blocks_per_iter) {
    const uint32_t *blk = (const uint32_t *)(rowp + kb * (int64_t)sizeof(typename T::block_t));
    for (int u = lane_in_block; u < block_u32; u += slots_per_block) {
      acc += blk[u];
    }
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor(acc, off);
  if (active && lane == 0 && w_in_group == 0) dst[row] = (float)acc;
}

template <class T, int ROWS, int WPR>
inline bool launch_read_only(const void *d_w, float *d_o, int64_t nrows, int64_t bpr,
                             hipStream_t stream) {
  const int grid = (int)((nrows + ROWS - 1) / ROWS);
  read_only_kernel<T, ROWS, WPR><<<grid, ROWS * WPR * 32, 0, stream>>>(d_w, d_o, nrows, bpr);
  return hipGetLastError() == hipSuccess;
}

inline bool matvec_launch_read_only(int dt, const void *d_w, float *d_o, int64_t nrows,
                                    int64_t ncols, hipStream_t stream) {
#define RD_RO(Traits, Dt, QK)                                                              \
  case Dt:                                                                                 \
    return launch_read_only<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr>(                   \
        d_w, d_o, nrows, ncols / QK, stream);
  switch (dt) {
    RD_RO(TQ8_0, 1, 32)
    RD_RO(TQ2K, 2, 256)
    RD_RO(TQ3K, 3, 256)
    RD_RO(TQ4K, 4, 256)
    RD_RO(TQ5K, 5, 256)
    RD_RO(TQ6K, 6, 256)
    RD_RO(TIQ2XXS, 7, 256)
    RD_RO(TIQ2XS, 8, 256)
    RD_RO(TIQ3XXS, 9, 256)
    RD_RO(TIQ1S, 10, 256)
    RD_RO(TIQ4NL, 11, 32)
    RD_RO(TIQ3S, 12, 256)
    RD_RO(TIQ2S, 13, 256)
    RD_RO(TIQ4XS, 14, 256)
    default: return false;
  }
#undef RD_RO
}


// Occupancy diagnostic: register/shared usage of each type's shipping kernel.
inline bool matvec_kernel_attrs(int dt, hipFuncAttributes &attr) {
#define RD_ATTR(Traits, Dt)                                                              \
  case Dt: {                                                                             \
    auto *fn = &matvec_kernel_gen<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr,           \
                                  MtIlp<Dt>::value, false>;                               \
    return hipFuncGetAttributes(&attr, (const void *)fn) == hipSuccess;                   \
  }
  switch (dt) {
    RD_ATTR(TQ8_0, 1) RD_ATTR(TQ2K, 2) RD_ATTR(TQ3K, 3) RD_ATTR(TQ4K, 4) RD_ATTR(TQ5K, 5)
    RD_ATTR(TQ6K, 6) RD_ATTR(TIQ2XXS, 7) RD_ATTR(TIQ2XS, 8) RD_ATTR(TIQ3XXS, 9)
    RD_ATTR(TIQ1S, 10) RD_ATTR(TIQ4NL, 11) RD_ATTR(TIQ3S, 12) RD_ATTR(TIQ2S, 13)
    RD_ATTR(TIQ4XS, 14)
    default: return false;
  }
#undef RD_ATTR
}


// ---------------------------------------------------------------------------
// ROWS is a FREE knob: it changes only which CTA computes which output row, so
// the arithmetic of every output element (the thread -> (slot, kqs) mapping, the
// k-walk, the ILP slots and both reduction stages) is bit-identical for any
// ROWS at a fixed (WPR, ILP). That is what makes the adaptive selection below
// safe to ship: it is validated as bit-exact by tests/check_matmul_gpu.hip
// (batch == N GEMV) and by the oracle gate, not merely inside a tolerance.
//
// WPR and ILP are NOT free: both change the summation order of an output
// element, so they stay compile-time per type (measured once, PLAN.md M2/M6).
//
// ROWS is clamped to the largest value that keeps ROWS*WPR*32 <= 1024 (the
// hardware workgroup limit): with WPR=8 (q3_K) only ROWS=1..4 is legal, so asking
// for 8 must not produce a silent launch failure (hipGetLastError reports it, but
// a caller looping over types would have to special-case the table).
inline bool matvec_launch_rows(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                               int64_t nrows, int64_t ncols, hipStream_t stream, int rows) {
  const int max_rows = 1024 / (matvec_default_config(dt).wpr * 32);
  if (rows > max_rows) rows = max_rows;
  if (rows < 1) return false;
#define RD_ROWS(Traits, Dt, QK)                                                              \
  case Dt:                                                                                   \
    switch (rows) {                                                                          \
      case 1: return launch_gen<Traits, 1, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(       \
                  d_w, d_a, d_o, nrows, ncols / QK, stream);                                 \
      case 2: return launch_gen<Traits, 2, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(       \
                  d_w, d_a, d_o, nrows, ncols / QK, stream);                                 \
      case 4: return launch_gen<Traits, 4, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(       \
                  d_w, d_a, d_o, nrows, ncols / QK, stream);                                 \
      case 8: return launch_gen<Traits, 8, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(       \
                  d_w, d_a, d_o, nrows, ncols / QK, stream);                                 \
      case 16: return launch_gen<Traits, 16, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(     \
                  d_w, d_a, d_o, nrows, ncols / QK, stream);                                 \
      default: return false;                                                                 \
    }
  switch (dt) {
    RD_ROWS(TQ8_0, 1, 32)   RD_ROWS(TQ2K, 2, 256)    RD_ROWS(TQ3K, 3, 256)   RD_ROWS(TQ4K, 4, 256)
    RD_ROWS(TQ5K, 5, 256)   RD_ROWS(TQ6K, 6, 256)    RD_ROWS(TIQ2XXS_S, 7, 256)
    RD_ROWS(TIQ2XS_S, 8, 256) RD_ROWS(TIQ3XXS_S, 9, 256) RD_ROWS(TIQ1S, 10, 256)
    RD_ROWS(TIQ4NL, 11, 32) RD_ROWS(TIQ3S_S, 12, 256) RD_ROWS(TIQ2S_S, 13, 256)
    RD_ROWS(TIQ4XS, 14, 256)
    default: return false;
  }
#undef RD_ROWS
}

// Manual single-accumulator unroll (bit-exact MLP) and the real L2 prefetch.
// Both are bench-only candidates: they are instantiated here so a measurement can
// compare them against the shipping kernel with the same shapes.
inline bool matvec_launch_unroll(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                 int64_t nrows, int64_t ncols, hipStream_t stream, int unroll,
                                 bool pf) {
#define RD_UNROLL(Traits, Dt, QK)                                                          \
  case Dt: {                                                                               \
    constexpr int R = MtShape<Dt>::rows, W = MtShape<Dt>::wpr;                             \
    if (unroll == 2 && !pf)                                                                \
      return launch_gen<Traits, R, W, 1, false, 0, 2>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
    if (unroll == 4 && !pf)                                                                \
      return launch_gen<Traits, R, W, 1, false, 0, 4>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
    if (unroll == 2 && pf)                                                                 \
      return launch_gen<Traits, R, W, 1, true, 0, 2>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
    if (unroll == 4 && pf)                                                                 \
      return launch_gen<Traits, R, W, 1, true, 0, 4>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
    if (unroll == 1 && pf)                                                                 \
      return launch_gen<Traits, R, W, MtIlp<Dt>::value, true>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
    return false;                                                                          \
  }
  switch (dt) {
    RD_UNROLL(TQ8_0, 1, 32)   RD_UNROLL(TQ2K, 2, 256)    RD_UNROLL(TQ3K, 3, 256)
    RD_UNROLL(TQ4K, 4, 256)   RD_UNROLL(TQ5K, 5, 256)    RD_UNROLL(TQ6K, 6, 256)
    RD_UNROLL(TIQ2XXS_S, 7, 256) RD_UNROLL(TIQ2XS_S, 8, 256) RD_UNROLL(TIQ3XXS_S, 9, 256)
    RD_UNROLL(TIQ1S, 10, 256) RD_UNROLL(TIQ4NL, 11, 32)  RD_UNROLL(TIQ3S_S, 12, 256)
    RD_UNROLL(TIQ2S_S, 13, 256) RD_UNROLL(TIQ4XS, 14, 256)
    default: return false;
  }
#undef RD_UNROLL
}

// Rows per CTA the shipping path would use for this type if nothing else is
// known (the historical, measured-per-type value).
inline int matvec_default_rows(int dt) { return matvec_default_config(dt).rows; }

// Sweeps __launch_bounds__ min-blocks (register budget) on each type's
// shipping shape; used by the bench to see if capping registers buys occupancy.
inline bool matvec_launch_minb(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                               int64_t nrows, int64_t ncols, hipStream_t stream, int minb) {
#define RD_MB(Traits, Dt, QK)                                                              \
  case Dt: {                                                                               \
    constexpr int R = MtShape<Dt>::rows, W = MtShape<Dt>::wpr, I = MtIlp<Dt>::value;       \
    switch (minb) {                                                                        \
      case 0: return launch_gen<Traits, R, W, I, false, 0>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
      case 2: return launch_gen<Traits, R, W, I, false, 2>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
      case 3: return launch_gen<Traits, R, W, I, false, 3>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
      case 4: return launch_gen<Traits, R, W, I, false, 4>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
      case 6: return launch_gen<Traits, R, W, I, false, 6>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
      default: return false;                                                               \
    }                                                                                      \
  }
  switch (dt) {
    RD_MB(TQ8_0, 1, 32)   RD_MB(TQ2K, 2, 256)    RD_MB(TQ3K, 3, 256)   RD_MB(TQ4K, 4, 256)
    RD_MB(TQ5K, 5, 256)   RD_MB(TQ6K, 6, 256)    RD_MB(TIQ2XXS, 7, 256) RD_MB(TIQ2XS, 8, 256)
    RD_MB(TIQ3XXS, 9, 256) RD_MB(TIQ1S, 10, 256) RD_MB(TIQ4NL, 11, 32) RD_MB(TIQ3S, 12, 256)
    RD_MB(TIQ2S, 13, 256) RD_MB(TIQ4XS, 14, 256)
    default: return false;
  }
#undef RD_MB
}


// A/B variants for iq3_s only (bench use; see vecdotq.cuh for what each means).
//   variant 0 = shipping, 1 = diagnostic no-sign, 2 = diagnostic no-lookup,
//   3 = correct dp4a-linearity candidate.
inline bool matvec_launch_variant(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                  int64_t nrows, int64_t ncols, hipStream_t stream, int variant) {
  // variant 0 = shipping kernel; 1 = perm+lin candidate; for iq3_s only:
  // 2 = DIAG-no-sign, 3 = lin-only, 4 = DIAG-no-lookup.
  switch (dt) {
    case 7: {  // iq2_xxs
      constexpr int R = MtShape<7>::rows, W = MtShape<7>::wpr, I = MtIlp<7>::value, QK = 256;
      return variant == 1 ? launch_gen<TIQ2XXS_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream)
                          : launch_gen<TIQ2XXS, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
    }
    case 8: {  // iq2_xs
      constexpr int R = MtShape<8>::rows, W = MtShape<8>::wpr, I = MtIlp<8>::value, QK = 256;
      return variant == 1 ? launch_gen<TIQ2XS_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream)
                          : launch_gen<TIQ2XS, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
    }
    case 9: {  // iq3_xxs: 0 = vendored, 1 = perm2 (shipping), 2 = perm (intermediate), 3 = DIAG
      constexpr int R = MtShape<9>::rows, W = MtShape<9>::wpr, I = MtIlp<9>::value, QK = 256;
      switch (variant) {
        case 0: return launch_gen<TIQ3XXS, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        case 1: return launch_gen<TIQ3XXS_PERM2, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        case 2: return launch_gen<TIQ3XXS_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        default: return launch_gen<TIQ3XXS_NOSIGN, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
      }
    }
    case 12: {  // iq3_s
      constexpr int R = MtShape<12>::rows, W = MtShape<12>::wpr, I = MtIlp<12>::value, QK = 256;
      switch (variant) {
        case 0: return launch_gen<TIQ3S_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);      // shipped
        case 1: return launch_gen<TIQ3S_NOSIGN, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);    // DIAG
        case 2: return launch_gen<TIQ3S, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);           // vendored
        case 3: return launch_gen<TIQ3S_LIN, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);       // DIAG/ctrl
        case 4: return launch_gen<TIQ3S_NOLOOKUP, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);  // DIAG
        default: return launch_gen<TIQ3S_XORADD, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);   // cand
      }
    }
    case 13: {  // iq2_s
      constexpr int R = MtShape<13>::rows, W = MtShape<13>::wpr, I = MtIlp<13>::value, QK = 256;
      return variant == 1 ? launch_gen<TIQ2S_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream)
                          : launch_gen<TIQ2S, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
    }
    default:
      return false;  // no variants for this type
  }
}

}  // namespace rdna4
