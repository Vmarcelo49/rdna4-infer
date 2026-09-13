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

// ---------------------------------------------------------------------------
// matvec: one warp per row; each thread walks its own (block, iqs) slots.
// ---------------------------------------------------------------------------
template <class T>
__global__ void matvec_kernel(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                              float *__restrict__ dst, int64_t nrows, int64_t blocks_per_row) {
  constexpr int vdr = T::vdr;
  constexpr int qi = T::qi;
  constexpr int qk = T::qk;
  constexpr int slots_per_block = qi / vdr;
  constexpr int blocks_per_iter = vdr * 32 / qi;

  const int tid = threadIdx.x & 31;
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (warp >= nrows) return;

  const char *row = (const char *)vx + (int64_t)warp * blocks_per_row * (int64_t)sizeof(typename T::block_t);
  const int kqs = vdr * (tid % slots_per_block);
  const int slot = tid / slots_per_block;

  float sum = 0.0f;
  for (int64_t kb = slot; kb < blocks_per_row; kb += blocks_per_iter) {
    const int64_t kby = kb * (qk / QK8_1);
    sum += T::dot((const void *)row, vy + kby, (const int)kb, kqs);
  }
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);
  if (tid == 0) dst[warp] = sum;
}

// ---------------------------------------------------------------------------
// Host launch helpers
// ---------------------------------------------------------------------------
struct MatvecShape {
  int qk;         // elements per weight block
  int block_bytes;
};

// Returns false for a type with no matvec kernel (never a silent CPU fallback).
bool matvec_shape(int dtype_ordinal, MatvecShape &out);

bool matvec_launch(int dtype_ordinal, const void *d_weights, const block_q8_1 *d_act,
                   float *d_out, int64_t nrows, int64_t ncols, hipStream_t stream);


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

#define RD_LAUNCH(Traits, Dt, QK)                                                          \
  case Dt: {                                                                               \
    const int warps_per_block = 4;                                                         \
    const int threads = 32 * warps_per_block;                                              \
    const int blocks = (int)((nrows + warps_per_block - 1) / warps_per_block);             \
    matvec_kernel<Traits><<<blocks, threads, 0, stream>>>(d_weights, d_act, d_out, nrows,  \
                                                          ncols / QK);                     \
    return hipGetLastError() == hipSuccess;                                                \
  }

inline bool matvec_launch(int dt, const void *d_weights, const block_q8_1 *d_act, float *d_out,
                          int64_t nrows, int64_t ncols, hipStream_t stream) {
  switch (dt) {
    RD_LAUNCH(TQ8_0, 1, 32)
    RD_LAUNCH(TQ2K, 2, 256)
    RD_LAUNCH(TQ3K, 3, 256)
    RD_LAUNCH(TQ4K, 4, 256)
    RD_LAUNCH(TQ5K, 5, 256)
    RD_LAUNCH(TQ6K, 6, 256)
    RD_LAUNCH(TIQ2XXS, 7, 256)
    RD_LAUNCH(TIQ2XS, 8, 256)
    RD_LAUNCH(TIQ3XXS, 9, 256)
    RD_LAUNCH(TIQ1S, 10, 256)
    RD_LAUNCH(TIQ4NL, 11, 32)
    RD_LAUNCH(TIQ3S, 12, 256)
    RD_LAUNCH(TIQ2S, 13, 256)
    RD_LAUNCH(TIQ4XS, 14, 256)
    default:
      return false;  // no kernel for this type: caller must fail loudly
  }
}

#undef RD_LAUNCH

}  // namespace rdna4
