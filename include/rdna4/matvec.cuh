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
template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0, int UNROLL = 1>
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
        acc[u] += T::dot((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs);
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
        acc[0] += T::dot((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs);
      }
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
template <class T, int ROWS, int WPR, int ILP = 1, int N = 1>
__global__ void
#if defined(__HIP_DEVICE_COMPILE__)
__launch_bounds__(ROWS * WPR * 32, 1)
#endif
matvec_kernel_batch(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                    float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row,
                    int64_t act_stride) {
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

  int64_t kb = slot;
  for (; kb + (ILP - 1) * blocks_per_iter < blocks_per_row; kb += ILP * blocks_per_iter) {
#pragma unroll
    for (int u = 0; u < ILP; ++u) {
      const int64_t k = kb + u * blocks_per_iter;
      const block_q8_1 *abase = vy + k * (qk / QK8_1);
#pragma unroll
      for (int n = 0; n < N; ++n) {
        acc[n][u] +=
            T::dot((const void *)rowp, abase + (int64_t)n * act_stride, (const int)k, kqs);
      }
    }
  }
  for (; kb < blocks_per_row; kb += blocks_per_iter) {
    const block_q8_1 *abase = vy + kb * (qk / QK8_1);
#pragma unroll
    for (int n = 0; n < N; ++n) {
      acc[n][0] += T::dot((const void *)rowp, abase + (int64_t)n * act_stride, (const int)kb, kqs);
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


template <class T, int ROWS, int WPR, int ILP = 1, bool PF = false, int MINB = 0, int UNROLL = 1>
inline bool launch_gen(const void *d_weights, const block_q8_1 *d_act, float *d_out, int64_t nrows,
                       int64_t bpr, hipStream_t stream) {
  const int grid = (int)((nrows + ROWS - 1) / ROWS);
  const int threads = ROWS * WPR * 32;
  matvec_kernel_gen<T, ROWS, WPR, ILP, PF, MINB, UNROLL>
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
inline bool matvec_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                          int64_t nrows, int64_t ncols, hipStream_t stream) {
#define RD_SHIP(Traits, Dt, QK)                                                            \
  case Dt:                                                                                 \
    return launch_gen<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, MtIlp<Dt>::value, false, \
                      0, MtUnroll<Dt>::value>(d_w, d_a, d_o, nrows, ncols / QK, stream);
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



// Batched shipping path. N is a compile-time constant (2/4/8/16) and the shape
// per type is the *same* one the GEMV path uses, which is what makes the results
// bit-identical; n_tokens is capped at the largest instantiation and the caller
// loops over sub-batches. The batched kernel has no UNROLL knob (its `N` already
// provides the memory-level parallelism), and since UNROLL does not change the
// summation order the batched result stays bit-identical to the GEMV path for
// every type that ships with unroll>1 -- tests/check_batch_gpu.hip asserts it.
inline int matvec_batch_cap() { return 16; }

template <int N>
inline bool matvec_launch_batch_n(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                                  int64_t nrows, int64_t ncols, int64_t act_stride,
                                  hipStream_t stream) {
#define RD_BATCH(Traits, Dt, QK)                                                              \
  case Dt: {                                                                                  \
    const int grid = (int)((nrows + MtShape<Dt>::rows - 1) / MtShape<Dt>::rows);              \
    const int threads = MtShape<Dt>::rows * MtShape<Dt>::wpr * 32;                            \
    matvec_kernel_batch<Traits, MtShape<Dt>::rows, MtShape<Dt>::wpr, MtIlp<Dt>::value, N>     \
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
    RD_BATCH(TIQ3XXS_S, 9, 256)
    RD_BATCH(TIQ1S, 10, 256)
    RD_BATCH(TIQ4NL, 11, 32)
    RD_BATCH(TIQ3S_S, 12, 256)
    RD_BATCH(TIQ2S_S, 13, 256)
    RD_BATCH(TIQ4XS, 14, 256)
    default:
      return false;  // no kernel: caller must fail loudly (SPEC 1.3)
  }
#undef RD_BATCH
}

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
    default: return false;  // N is a compile-time instantiation, not a runtime knob
  }
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
