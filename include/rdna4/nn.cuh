#pragma once
// M3 — neural-network primitives (device kernels + host launchers).
//
// Semantics copied from the llama.cpp reference graph (src/models/qwen35.cpp and
// the ggml CPU ops it lowers to), so that a layer built from these kernels can be
// compared against llama-eval-callback dumps (see tests/check_nn_gpu.hip).
#include <hip/hip_runtime.h>

#include <cstdint>

#include "rdna4/fp16.h"

namespace rdna4 {

// ---------------------------------------------------------------------------
// RMSNorm: y = x / sqrt(mean(x^2) + eps), one row per block.
// ggml computes scale = 1/sqrt(sum(x^2)/n + eps) (ggml-cpu/ops.cpp
// ggml_compute_forward_rms_norm_f32); the weight multiply is a separate node in
// the reference graph, provided here by `w` (pass nullptr to skip).
// ---------------------------------------------------------------------------
constexpr int kRmsNormThreads = 256;

__global__ void rms_norm_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                float *__restrict__ y, std::int64_t ncols, float eps) {
  __shared__ float partial[kRmsNormThreads];
  const std::int64_t row = blockIdx.x;
  const float *xr = x + row * ncols;
  float *yr = y + row * ncols;

  float acc = 0.0f;
  for (std::int64_t i = threadIdx.x; i < ncols; i += kRmsNormThreads) {
    const float v = xr[i];
    acc = fmaf(v, v, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kRmsNormThreads / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)ncols + eps);
  for (std::int64_t i = threadIdx.x; i < ncols; i += kRmsNormThreads) {
    yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
  }
}

// ---------------------------------------------------------------------------
// L2 norm (gated delta net q/k): y = x / sqrt(sum(x^2) + eps).
// Reference: build_gdn_l2_norm() = scale(rms_norm(x, eps/n), 1/sqrt(n))
// (src/models/models.h:14), which is algebraically x / sqrt(sum(x^2) + eps).
// ---------------------------------------------------------------------------
__global__ void l2_norm_kernel(const float *__restrict__ x, float *__restrict__ y,
                               std::int64_t ncols, float eps) {
  __shared__ float partial[kRmsNormThreads];
  const std::int64_t row = blockIdx.x;
  const float *xr = x + row * ncols;
  float *yr = y + row * ncols;

  float acc = 0.0f;
  for (std::int64_t i = threadIdx.x; i < ncols; i += kRmsNormThreads) {
    const float v = xr[i];
    acc = fmaf(v, v, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kRmsNormThreads / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] + eps);
  for (std::int64_t i = threadIdx.x; i < ncols; i += kRmsNormThreads) yr[i] = xr[i] * scale;
}

// ---------------------------------------------------------------------------
// Elementwise ops (all ggml ops used by the reference graph).
//   silu(x) = x * sigmoid(x)              (ggml_silu)
//   softplus(x) = log(1 + exp(x))         (ggml_compute_softplus_f32, same clamp)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) {
  // ggml_compute_softplus_f32 (ggml-impl.h): (x > 20) ? x : log(1 + exp(x)).
  // log1pf(expf(x)) is mathematically the same and differs by <= 1e-8, but the
  // reference formula is what we compare against, so use it verbatim.
  return (x > 20.0f) ? x : logf(1.0f + expf(x));
}

enum class UnOp { Silu, Sigmoid, Softplus };

__global__ void unary_kernel(const float *__restrict__ x, float *__restrict__ y, std::int64_t n,
                             UnOp op) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  y[i] = op == UnOp::Silu ? silu_f(v) : (op == UnOp::Sigmoid ? sigmoid_f(v) : softplus_f(v));
}

__global__ void mul_kernel(const float *__restrict__ a, const float *__restrict__ b,
                           float *__restrict__ y, std::int64_t n) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i];
}

__global__ void add_kernel(const float *__restrict__ a, const float *__restrict__ b,
                           float *__restrict__ y, std::int64_t n) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i];
}

__global__ void scale_kernel(const float *__restrict__ x, float *__restrict__ y, std::int64_t n,
                             float s) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * s;
}

// ---------------------------------------------------------------------------
// Broadcast elementwise, for the BATCHED prefill path (feat/noite-prefill).
// y[i] = a[i] (+|*) b[i % nb]: the per-token path applies a length-`nb` array
// (ssm_dt, ssm_a) to one token at a time, so folding the token index into the
// thread index reproduces each token's arithmetic exactly -- same operand values,
// same single rounding, one launch for the whole batch instead of one per token.
__global__ void add_bcast_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                 float *__restrict__ y, std::int64_t n, int nb) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] + b[i % nb];
}

__global__ void mul_bcast_kernel(const float *__restrict__ a, const float *__restrict__ b,
                                 float *__restrict__ y, std::int64_t n, int nb) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a[i] * b[i % nb];
}

// ---------------------------------------------------------------------------
// Host launchers. All return false on a failed launch (never a silent fallback).
// ---------------------------------------------------------------------------
inline bool rms_norm_launch(const float *d_x, const float *d_w, float *d_y, std::int64_t nrows,
                            std::int64_t ncols, float eps, hipStream_t stream = nullptr) {
  rms_norm_kernel<<<(unsigned)nrows, kRmsNormThreads, 0, stream>>>(d_x, d_w, d_y, ncols, eps);
  return hipGetLastError() == hipSuccess;
}

inline bool l2_norm_launch(const float *d_x, float *d_y, std::int64_t nrows, std::int64_t ncols,
                           float eps, hipStream_t stream = nullptr) {
  l2_norm_kernel<<<(unsigned)nrows, kRmsNormThreads, 0, stream>>>(d_x, d_y, ncols, eps);
  return hipGetLastError() == hipSuccess;
}

inline bool unary_launch(const float *d_x, float *d_y, std::int64_t n, UnOp op,
                         hipStream_t stream = nullptr) {
  const int threads = 256;
  unary_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_x, d_y, n, op);
  return hipGetLastError() == hipSuccess;
}

inline bool mul_launch(const float *d_a, const float *d_b, float *d_y, std::int64_t n,
                       hipStream_t stream = nullptr) {
  const int threads = 256;
  mul_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_a, d_b, d_y, n);
  return hipGetLastError() == hipSuccess;
}

inline bool add_launch(const float *d_a, const float *d_b, float *d_y, std::int64_t n,
                       hipStream_t stream = nullptr) {
  const int threads = 256;
  add_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_a, d_b, d_y, n);
  return hipGetLastError() == hipSuccess;
}

inline bool scale_launch(const float *d_x, float *d_y, std::int64_t n, float s,
                         hipStream_t stream = nullptr) {
  const int threads = 256;
  scale_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_x, d_y, n, s);
  return hipGetLastError() == hipSuccess;
}

// nb = length of the broadcast operand (elements per token in the batch).
inline bool add_bcast_launch(const float *d_a, const float *d_b, float *d_y, std::int64_t n, int nb,
                             hipStream_t stream = nullptr) {
  const int threads = 256;
  add_bcast_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_a, d_b, d_y,
                                                                                    n, nb);
  return hipGetLastError() == hipSuccess;
}

inline bool mul_bcast_launch(const float *d_a, const float *d_b, float *d_y, std::int64_t n, int nb,
                             hipStream_t stream = nullptr) {
  const int threads = 256;
  mul_bcast_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_a, d_b, d_y,
                                                                                    n, nb);
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
