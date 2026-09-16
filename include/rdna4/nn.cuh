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

  // A cadeia de fma tem a MESMA ordem de antes (i = tid, tid+256, tid+512, ...),
  // mas com 4 cargas independentes em voo por iteracao: o kernel e' limitado por
  // LATENCIA de load (docs/journal-kernels.md §4: 1 CTA de 256 threads para 5120
  // elementos, 20 cargas por thread numa cadeia de fma). Issa nao muda nenhum
  // valor -- mesmos fma, mesma ordem, mesma arvore de reducao -- entao e'
  // bit-exato e nao precisa de gate numerico, so' dos gates normais.
  float acc = 0.0f;
  std::int64_t i = threadIdx.x;
  for (; i + 3 * kRmsNormThreads < ncols; i += 4 * kRmsNormThreads) {
    const float v0 = xr[i];
    const float v1 = xr[i + kRmsNormThreads];
    const float v2 = xr[i + 2 * kRmsNormThreads];
    const float v3 = xr[i + 3 * kRmsNormThreads];
    acc = fmaf(v0, v0, acc);
    acc = fmaf(v1, v1, acc);
    acc = fmaf(v2, v2, acc);
    acc = fmaf(v3, v3, acc);
  }
  for (; i < ncols; i += kRmsNormThreads) {
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

  // Mesmo tratamento do rms_norm acima: 4 cargas em voo, mesma ordem de fma.
  float acc = 0.0f;
  std::int64_t i = threadIdx.x;
  for (; i + 3 * kRmsNormThreads < ncols; i += 4 * kRmsNormThreads) {
    const float v0 = xr[i];
    const float v1 = xr[i + kRmsNormThreads];
    const float v2 = xr[i + 2 * kRmsNormThreads];
    const float v3 = xr[i + 3 * kRmsNormThreads];
    acc = fmaf(v0, v0, acc);
    acc = fmaf(v1, v1, acc);
    acc = fmaf(v2, v2, acc);
    acc = fmaf(v3, v3, acc);
  }
  for (; i < ncols; i += kRmsNormThreads) {
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
// ---------------------------------------------------------------------------
// Argmax on the device (greedy decoding).
//
// The greedy path used to copy the whole vocab (248 320 floats = 993 KB) back to
// the host just to take a maximum on the CPU: 0.2-0.4 ms per token of pure
// round-trip, and one more host sync in the middle of the decode loop.
//
// The rule is the one the host sampler uses for temp <= 0: **the first maximum in
// id order**. That is what makes this kernel's result identical to the host
// argmax and not merely equivalent to it: ties keep the lower id, both inside a
// thread's stride and when combining partial results (strict `>` comparisons,
// lower index wins).
// ---------------------------------------------------------------------------
constexpr int kArgmaxThreads = 256;
constexpr int kArgmaxBlocks = 64;

__global__ void argmax_kernel(const float *__restrict__ x, std::int64_t n, float *__restrict__ best,
                              int *__restrict__ best_i) {
  __shared__ float sval[kArgmaxThreads];
  __shared__ int sidx[kArgmaxThreads];
  const std::int64_t stride = (std::int64_t)gridDim.x * blockDim.x;
  float v = -INFINITY;
  std::int64_t vi = -1;
  // The host scan (Sampler::filter, temp <= 0) starts with index 0 as the
  // incumbent and only replaces it on a STRICTLY greater value -- so with
  // logits[0] = NaN it answers 0, while a plain max-of-finite would answer the
  // argmax of the finite entries (review finding R4). Seeding thread 0 of block 0
  // with index 0 reproduces the host rule exactly: nothing is "greater" than NaN,
  // and ties keep the lower index.
  if (blockIdx.x == 0 && threadIdx.x == 0 && n > 0) {
    v = x[0];
    vi = 0;
  }
  for (std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float xi = x[i];
    // strict > : equal values keep the lower id (and the first one seen, which is
    // the lower id because the stride walks upwards)
    if (xi > v || (vi < 0 && xi == xi)) {
      v = xi;
      vi = i;
    }
  }
  const int t = threadIdx.x;
  sval[t] = v;
  sidx[t] = (int)vi;
  __syncthreads();
  for (int off = kArgmaxThreads / 2; off > 0; off >>= 1) {
    if (t < off) {
      const float ov = sval[t + off];
      const int oi = sidx[t + off];
      if (ov > sval[t] || (sidx[t] < 0 && oi >= 0)) {
        sval[t] = ov;
        sidx[t] = oi;
      }
    }
    __syncthreads();
  }
  if (t == 0 && sidx[0] >= 0) {
    // One candidate per CTA: append it (the caller reduces the 64 partials).
    best[blockIdx.x] = sval[0];
    best_i[blockIdx.x] = sidx[0];
  }
}

// 64 partials -> the final index, on the host (4 bytes copied instead of 993 KB).
inline bool argmax_launch(const float *d_x, std::int64_t n, float *d_partial_val,
                          int *d_partial_idx, hipStream_t stream = nullptr) {
  argmax_kernel<<<kArgmaxBlocks, kArgmaxThreads, 0, stream>>>(d_x, n, d_partial_val,
                                                              d_partial_idx);
  return hipGetLastError() == hipSuccess;
}

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

// ---------------------------------------------------------------------------
// Fused unary+mul pairs for the DECODE path (each erases a launch + barrier).
//
// Bit-exactness contract: the intermediate (silu/sigmoid output) lives in a
// register and is written back to EXACTLY the buffer the unfused sequence
// wrote, so downstream dumps (gate_sigmoid, ...) and any later reader see the
// same bytes. Same operand values, same op order, same single rounding per op
// => the fused launch stores exactly what the two launches stored.
// ---------------------------------------------------------------------------

// y[i] = silu(a[i]) * b[i], with the silu written back to a_io. Serves the FFN
// (a_io == y == d_ffn_a_, b == d_ffn_b_) and the GDN tail
// (a_io == d_z_, b == y == v_c): every output index depends only on its own
// inputs, so all aliasing combinations are safe.
__global__ void silu_gate_kernel(float *__restrict__ a_io, const float *__restrict__ b,
                                 float *__restrict__ y, std::int64_t n) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float s = silu_f(a_io[i]);
  a_io[i] = s;
  y[i] = s * b[i];
}

inline bool silu_gate_launch(float *d_a_io, const float *d_b, float *d_y, std::int64_t n,
                             hipStream_t stream = nullptr) {
  const int threads = 256;
  silu_gate_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_a_io, d_b,
                                                                                    d_y, n);
  return hipGetLastError() == hipSuccess;
}

// y[i] *= sigmoid(g[i]), with the sigmoid written back to g_io. Serves the
// attention gate (g_io == d_attngate_, y_io == d_attnout_): d_attngate_ still
// holds the sigmoid output, so the gate_sigmoid dump reads the same values.
__global__ void sigmoid_gate_kernel(float *__restrict__ g_io, float *__restrict__ y_io,
                                    std::int64_t n) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float s = sigmoid_f(g_io[i]);
  g_io[i] = s;
  y_io[i] = y_io[i] * s;
}

inline bool sigmoid_gate_launch(float *d_g_io, float *d_y_io, std::int64_t n,
                                hipStream_t stream = nullptr) {
  const int threads = 256;
  sigmoid_gate_kernel<<<(unsigned)((n + threads - 1) / threads), threads, 0, stream>>>(d_g_io,
                                                                                       d_y_io, n);
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
