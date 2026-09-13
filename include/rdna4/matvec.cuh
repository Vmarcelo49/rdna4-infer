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
  float sum = xi;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor(amax, off));
    sum += __shfl_xor(sum, off);
  }
  const float d = amax / 127.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  const int q = (int)roundf(xi * id);  // ggml uses roundf (half away from zero)
  y->qs[lane] = (int8_t)q;
  if (lane == 0) {
    const uint16_t dh = float_to_fp16(d);
    // s = d * sum(qs): recompute the integer sum on the whole warp below
    y->ds = (uint32_t)dh;
  }
  // warp-wide integer sum of the quantized values
  int qsum = q;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor(qsum, off);
  if (lane == 0) {
    const uint16_t dh = (uint16_t)(y->ds & 0xFFFFu);
    const uint16_t sh = float_to_fp16(fp16_to_float(dh) * (float)qsum);
    y->ds = (uint32_t)dh | ((uint32_t)sh << 16);
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

#undef RD_MATVEC_TRAITS

// Generalized kernel: ROWS rows per CTA, WPR warps cooperating per row
// (NWARPS = ROWS*WPR). Covers both shapes:
//   (ROWS=4, WPR=1) -> the 4-warps-4-rows layout (best aggregate here)
//   (ROWS=1, WPR=8) -> llama.cpp's mmvq layout for ncols_dst=1
// Per-row-group indexing follows llama.cpp: kqs = vdr*(tg % (qi/vdr)),
// slot = tg / (qi/vdr), blocks_per_iter = vdr*(WPR*32)/qi, where tg is the
// thread index inside its row group.
template <class T, int ROWS, int WPR>
__global__ void matvec_kernel_gen(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                                  float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row) {
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

  float sum = 0.0f;
  for (int64_t kb = slot; kb < blocks_per_row; kb += blocks_per_iter) {
    const int64_t kby = kb * (qk / QK8_1);
    sum += T::dot((const void *)rowp, vy + kby, (const int)kb, kqs);
  }
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
// Measured on gfx1201 (RX 9070 XT) with `check-matvec-gpu --bench` on the
// largest tensor of each type, averaged over 3 runs (values in GB/s in
// PLAN.md "Passo 5"). Row shape matters as much as type: tensors with few
// blocks per row (q8_0/iq4_nl) and very long rows (token_embd 248320 cols)
// prefer different shapes, so this keys on the type of the tensor being run.
struct MatvecConfig {
  int rows;  // rows per CTA
  int wpr;   // warps cooperating per row
};

inline MatvecConfig matvec_default_config(int dt) {
  switch (dt) {
    case 1:  return {4, 1};  // q8_0     (399; 1x1=407 within noise, better occupancy)
    case 2:  return {2, 2};  // q2_K     (190)
    case 3:  return {1, 4};  // q3_K     (187, long rows)
    case 4:  return {2, 2};  // q4_K     (408)
    case 5:  return {1, 2};  // q5_K     (493, long rows)
    case 6:  return {1, 1};  // q6_K     (312)
    case 7:  return {1, 1};  // iq2_xxs  (128)
    case 8:  return {1, 1};  // iq2_xs   (146)
    case 9:  return {8, 1};  // iq3_xxs  (193)
    case 10: return {2, 2};  // iq1_s    (249)
    case 11: return {8, 1};  // iq4_nl   (99)
    case 12: return {1, 1};  // iq3_s    (215)
    case 13: return {8, 1};  // iq2_s    (166)
    case 14: return {8, 1};  // iq4_xs   (124)
    default: return {4, 1};
  }
}

template <class T, int ROWS, int WPR>
inline bool launch_gen(const void *d_weights, const block_q8_1 *d_act, float *d_out, int64_t nrows,
                       int64_t bpr, hipStream_t stream) {
  const int grid = (int)((nrows + ROWS - 1) / ROWS);
  const int threads = ROWS * WPR * 32;
  matvec_kernel_gen<T, ROWS, WPR><<<grid, threads, 0, stream>>>(d_weights, d_act, d_out, nrows, bpr);
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

inline bool matvec_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                          int64_t nrows, int64_t ncols, hipStream_t stream) {
  return matvec_launch_cfg(dt, d_w, d_a, d_o, nrows, ncols, stream, matvec_default_config(dt));
}

}  // namespace rdna4
