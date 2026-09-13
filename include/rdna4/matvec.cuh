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

// ---------------------------------------------------------------------------
// Per-type traits: block size, quants-per-int, values-per-dot, byte size,
// and the vec_dot entry point.
// ---------------------------------------------------------------------------
#define RD_MATVEC_TRAITS(Name, Vd, QK, QI, VDR, BlockT)                                 \
  struct Name {                                                                         \
    static constexpr int qk = QK, qi = QI, vdr = VDR;                                   \
    using block_t = BlockT;                                                             \
    static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,   \
                                                const int &kbx, const int &iqs) {       \
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

// L2 prefetch (from llama.cpp mmvq.cu, MIT). Only used where measured to help.
static __device__ __forceinline__ void rdna4_prefetch_l2(const void *p) {
  // __builtin_prefetch instead of llama.cpp's inline asm: the asm form uses the
  // "l" (64-bit register) constraint, which the host pass of amdclang++ rejects.
  __builtin_prefetch(p, 0, 3);
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
template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0>
__global__ void
#if defined(__HIP_DEVICE_COMPILE__)
__launch_bounds__(ROWS * WPR * 32, MINB > 0 ? MINB : 1)
#endif
matvec_kernel_gen(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                  float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row) {
  static_assert(ILP >= 1, "ILP must be >= 1");
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

  // ILP independent accumulators: the naive loop is one dependent chain of
  // float adds, which leaves memory latency exposed.
  float acc[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) acc[i] = 0.0f;

  int64_t kb = slot;
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
      acc[u] += T::dot((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs);
    }
  }
  for (; kb < blocks_per_row; kb += blocks_per_iter) {
    acc[0] += T::dot((const void *)rowp, vy + kb * (qk / QK8_1), (const int)kb, kqs);
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

inline MatvecConfig matvec_default_config(int dt) {
  switch (dt) {
    case 1:  return {2, 1};  // q8_0     671 GB/s
    case 2:  return {2, 1};  // q2_K     337
    case 3:  return {1, 8};  // q3_K     321  (very long rows: token_embd)
    case 4:  return {8, 1};  // q4_K     694
    case 5:  return {4, 1};  // q5_K     610
    case 6:  return {1, 4};  // q6_K     484
    case 7:  return {4, 1};  // iq2_xxs  227
    case 8:  return {2, 2};  // iq2_xs   248
    case 9:  return {2, 1};  // iq3_xxs  336
    case 10: return {1, 4};  // iq1_s    433
    case 11: return {8, 1};  // iq4_nl   179
    case 12: return {8, 1};  // iq3_s    374
    case 13: return {2, 1};  // iq2_s    289
    case 14: return {8, 1};  // iq4_xs   241
    default: return {4, 1};
  }
}

template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0>
inline bool launch_gen(const void *d_weights, const block_q8_1 *d_act, float *d_out, int64_t nrows,
                       int64_t bpr, hipStream_t stream) {
  const int grid = (int)((nrows + ROWS - 1) / ROWS);
  const int threads = ROWS * WPR * 32;
  matvec_kernel_gen<T, ROWS, WPR, ILP, PF, MINB>
      <<<grid, threads, 0, stream>>>(d_weights, d_act, d_out, nrows, bpr);
  return hipGetLastError() == hipSuccess;
}

// Supported (ROWS, WPR) shapes; the bench sweeps exactly these.
#define RD_CFG_CASES(Traits, QK)                                                                \
  if (cfg.rows == 4 && cfg.wpr == 1) return launch_gen<Traits, 4, 1>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 2 && cfg.wpr == 1) return launch_gen<Traits, 2, 1>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 1 && cfg.wpr == 1) return launch_gen<Traits, 1, 1>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 8 && cfg.wpr == 1) return launch_gen<Traits, 8, 1>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 1 && cfg.wpr == 2) return launch_gen<Traits, 1, 2>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 1 && cfg.wpr == 4) return launch_gen<Traits, 1, 4>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 1 && cfg.wpr == 8) return launch_gen<Traits, 1, 8>(d_w, d_a, d_o, nrows, ncols / QK, stream); \
  if (cfg.rows == 2 && cfg.wpr == 2) return launch_gen<Traits, 2, 2>(d_w, d_a, d_o, nrows, ncols / QK, stream);

// Dispatches on type and shape. Returns false for a type with no kernel: the
// caller must fail loudly, never fall back to CPU (SPEC 1.3).
inline bool matvec_launch_cfg(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                              int64_t nrows, int64_t ncols, hipStream_t stream, MatvecConfig cfg) {
  switch (dt) {
    case 1:  RD_CFG_CASES(TQ8_0, 32)  break;
    case 2:  RD_CFG_CASES(TQ2K, 256)  break;
    case 3:  RD_CFG_CASES(TQ3K, 256)  break;
    case 4:  RD_CFG_CASES(TQ4K, 256)  break;
    case 5:  RD_CFG_CASES(TQ5K, 256)  break;
    case 6:  RD_CFG_CASES(TQ6K, 256)  break;
    case 7:  RD_CFG_CASES(TIQ2XXS, 256) break;
    case 8:  RD_CFG_CASES(TIQ2XS, 256) break;
    case 9:  RD_CFG_CASES(TIQ3XXS, 256) break;
    case 10: RD_CFG_CASES(TIQ1S, 256) break;
    case 11: RD_CFG_CASES(TIQ4NL, 32) break;
    case 12: RD_CFG_CASES(TIQ3S, 256) break;
    case 13: RD_CFG_CASES(TIQ2S, 256) break;
    case 14: RD_CFG_CASES(TIQ4XS, 256) break;
    default: return false;
  }
  return false;  // unsupported (rows, wpr) shape
}

#undef RD_CFG_CASES

template <int Dt> struct MtShape;             // { rows, wpr } per dtype ordinal
template <> struct MtShape<1>  { static constexpr int rows = 2, wpr = 1; };  // q8_0
template <> struct MtShape<2>  { static constexpr int rows = 2, wpr = 1; };  // q2_K
template <> struct MtShape<3>  { static constexpr int rows = 1, wpr = 8; };  // q3_K
template <> struct MtShape<4>  { static constexpr int rows = 8, wpr = 1; };  // q4_K
template <> struct MtShape<5>  { static constexpr int rows = 4, wpr = 1; };  // q5_K
template <> struct MtShape<6>  { static constexpr int rows = 1, wpr = 4; };  // q6_K
template <> struct MtShape<7>  { static constexpr int rows = 4, wpr = 1; };  // iq2_xxs
template <> struct MtShape<8>  { static constexpr int rows = 2, wpr = 2; };  // iq2_xs
template <> struct MtShape<9>  { static constexpr int rows = 2, wpr = 1; };  // iq3_xxs
template <> struct MtShape<10> { static constexpr int rows = 1, wpr = 4; };  // iq1_s
template <> struct MtShape<11> { static constexpr int rows = 8, wpr = 1; };  // iq4_nl
template <> struct MtShape<12> { static constexpr int rows = 8, wpr = 1; };  // iq3_s
template <> struct MtShape<13> { static constexpr int rows = 2, wpr = 1; };  // iq2_s
template <> struct MtShape<14> { static constexpr int rows = 8, wpr = 1; };  // iq4_xs

// ILP (independent accumulators) per type, measured with --bench-tune.
// Helps the latency-bound types a lot (q4_K 610->739 GB/s), does nothing or
// slightly regresses the ALU-bound IQ types, so those stay at 1.
template <int Dt> struct MtIlp { static constexpr int value = 1; };
template <> struct MtIlp<1>  { static constexpr int value = 4; };  // q8_0
template <> struct MtIlp<2>  { static constexpr int value = 2; };  // q2_K
template <> struct MtIlp<3>  { static constexpr int value = 2; };  // q3_K
template <> struct MtIlp<4>  { static constexpr int value = 2; };  // q4_K
template <> struct MtIlp<5>  { static constexpr int value = 2; };  // q5_K
template <> struct MtIlp<6>  { static constexpr int value = 2; };  // q6_K
template <> struct MtIlp<7>  { static constexpr int value = 2; };  // iq2_xxs
template <> struct MtIlp<8>  { static constexpr int value = 4; };  // iq2_xs
template <> struct MtIlp<9>  { static constexpr int value = 1; };  // iq3_xxs
template <> struct MtIlp<10> { static constexpr int value = 1; };  // iq1_s
template <> struct MtIlp<11> { static constexpr int value = 1; };  // iq4_nl
template <> struct MtIlp<12> { static constexpr int value = 1; };  // iq3_s
template <> struct MtIlp<13> { static constexpr int value = 1; };  // iq2_s
template <> struct MtIlp<14> { static constexpr int value = 2; };  // iq4_xs

// Shipping path: shape and ILP are compile-time per type (both measured), so
// this instantiates exactly one kernel per type.
inline bool matvec_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                          int64_t nrows, int64_t ncols, hipStream_t stream) {
#define RD_SHIP(Traits, Dt, QK)                                                            \
  case Dt:                                                                                 \
    return launch_gen<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, MtIlp<Dt>::value, false>(\
        d_w, d_a, d_o, nrows, ncols / QK, stream);
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
        case 0: return launch_gen<TIQ3S, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        case 1: return launch_gen<TIQ3S_PERM, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        case 2: return launch_gen<TIQ3S_NOSIGN, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        case 3: return launch_gen<TIQ3S_LIN, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
        default: return launch_gen<TIQ3S_NOLOOKUP, R, W, I, false>(d_w, d_a, d_o, nrows, ncols / QK, stream);
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
