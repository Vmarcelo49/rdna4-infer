// M4 — dequantize one contiguous span of quantized rows into f32 on device.
//
// Factorised out of tests/check_dequant_gpu.hip (where the per-type kernels and
// the 14-type dispatch were validated bit-exact against llama.cpp's
// dequantize_row_* for every type in the UD union) so the engine can reuse the
// *same* code for the token embedding instead of duplicating it: the CLI
// dequantizes the embedding row on the GPU, with no host-side round trip.
#pragma once

#include "rdna4/dequant.cuh"
#include "rdna4/dtype.h"

namespace rdna4 {

template <class Fn>
__global__ void dequant_kernel_256(const void *vx, float *yy, std::int64_t nblocks) {
  for (std::int64_t i = blockIdx.x; i < nblocks; i += gridDim.x) {
    Fn::apply(vx, i, yy + i * 256, threadIdx.x);
  }
}

__global__ void dequant_kernel_q8_0(const void *vx, float *yy, std::int64_t nblocks) {
  for (std::int64_t i = blockIdx.x; i < nblocks; i += gridDim.x) {
    float2 v;
    rdna4::dequantize_q8_0(vx, i, 2 * (int)threadIdx.x, v);
    yy[i * 32 + 2 * threadIdx.x + 0] = v.x;
    yy[i * 32 + 2 * threadIdx.x + 1] = v.y;
  }
}

#define RD_FN(Name, FnName)                                                                     \
  struct Name {                                                                                 \
    static __device__ __forceinline__ void apply(const void *vx, std::int64_t i, float *y,      \
                                                 int tid) {                                     \
      rdna4::FnName(vx, i, y, tid);                                                             \
    }                                                                                           \
  }

RD_FN(FnQ2K, dequantize_q2_K);
RD_FN(FnQ3K, dequantize_q3_K);
RD_FN(FnQ4K, dequantize_q4_K);
RD_FN(FnQ5K, dequantize_q5_K);
RD_FN(FnQ6K, dequantize_q6_K);
RD_FN(FnIQ2XXS, dequantize_iq2_xxs);
RD_FN(FnIQ2XS, dequantize_iq2_xs);
RD_FN(FnIQ3XXS, dequantize_iq3_xxs);
RD_FN(FnIQ1S, dequantize_iq1_s);
RD_FN(FnIQ4NL, dequantize_iq4_nl);
RD_FN(FnIQ3S, dequantize_iq3_s);
RD_FN(FnIQ2S, dequantize_iq2_s);
RD_FN(FnIQ4XS, dequantize_iq4_xs);
#undef RD_FN

// Dequantizes `nelem` elements (nelem % block_elems == 0) starting at d_src.
inline bool dequant_row_launch(DType dt, const void *d_src, float *d_dst, std::int64_t nelem,
                               hipStream_t stream = nullptr) {
  if (dt == DType::F32) {
    return hipMemcpyAsync(d_dst, d_src, (std::size_t)nelem * sizeof(float),
                          hipMemcpyDeviceToDevice, stream) == hipSuccess;
  }
  const std::int64_t be = (std::int64_t)dtype_block_elems(dt);
  if (be == 0 || nelem % be != 0) return false;
  const std::int64_t nblocks = nelem / be;

  if (dt == DType::Q8_0) {
    // 16 threads x 2 elements (the float2 form llama.cpp's get_rows uses for
    // q8_0); a 32-thread launch would index past the 32-element block
    dequant_kernel_q8_0<<<(unsigned)nblocks, 16, 0, stream>>>(d_src, d_dst, nblocks);
    return hipGetLastError() == hipSuccess;
  }
  if (dt == DType::IQ4_NL) {
    dequant_kernel_256<FnIQ4NL><<<(unsigned)nblocks, 32, 0, stream>>>(d_src, d_dst, nblocks);
    return hipGetLastError() == hipSuccess;
  }
  const unsigned grid = (unsigned)nblocks;
  switch (dt) {
    case DType::Q2_K: dequant_kernel_256<FnQ2K><<<grid, 64, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::Q3_K: dequant_kernel_256<FnQ3K><<<grid, 64, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::Q4_K: dequant_kernel_256<FnQ4K><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::Q5_K: dequant_kernel_256<FnQ5K><<<grid, 64, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::Q6_K: dequant_kernel_256<FnQ6K><<<grid, 64, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ2_XXS: dequant_kernel_256<FnIQ2XXS><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ2_XS: dequant_kernel_256<FnIQ2XS><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ3_XXS: dequant_kernel_256<FnIQ3XXS><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ1_S: dequant_kernel_256<FnIQ1S><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ3_S: dequant_kernel_256<FnIQ3S><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ2_S: dequant_kernel_256<FnIQ2S><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    case DType::IQ4_XS: dequant_kernel_256<FnIQ4XS><<<grid, 32, 0, stream>>>(d_src, d_dst, nblocks); break;
    default: return false;
  }
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
