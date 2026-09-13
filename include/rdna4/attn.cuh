#pragma once
// M3 — full-attention branch primitives for qwen35 (full-attention layers).
//
// Layouts and semantics follow llama.cpp's qwen35 graph (src/models/qwen35.cpp,
// build_layer_attn) and its ggml lowering:
//   attn_q output per head is [ q (head_dim) | gate (head_dim) ], stride 2*head_dim
//     Qcur = view at offset 0, gate = view at offset head_dim
//   Q/K are RMS-normalized per head (attn_q_norm / attn_k_norm, size head_dim)
//   RoPE: GGML_ROPE_TYPE_IMROPE, which ggml lowerers through
//     rotate_pairs(n_dims, n_dims/2, ...) — SPLIT-HALF pairs (i with i+n_rot/2)
//     over n_rot dims of each head (rope.dimension_count)
//   attention scale = 1/sqrt(head_dim); the attention output is multiplied by
//     sigmoid(gate) before attn_output
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "rdna4/kv.h"

namespace rdna4 {

// ---------------------------------------------------------------------------
// RoPE on split-half pairs of the first n_rot dims of each head:
//   angle(pair p, position t) = pos[t] * freq_base^(-2p/n_rot)
//   (x0,x1) -> (x0*cos - x1*sin, x0*sin + x1*cos)
// x is [n_tokens, n_heads, head_dim].
// ---------------------------------------------------------------------------
__global__ void rope_kernel(float *__restrict__ x, int n_tokens, int n_heads, int head_dim,
                            int n_rot, float freq_base, const int *__restrict__ pos) {
  const int pairs = n_rot / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_tokens * n_heads * pairs) return;
  const int p = idx % pairs;
  const int h = (idx / pairs) % n_heads;
  const int t = idx / (pairs * n_heads);

  const float theta = (float)pos[t] * powf(freq_base, -2.0f * (float)p / (float)n_rot);
  const float c = cosf(theta);
  const float s = sinf(theta);

  // SPLIT-HALF pairing, not adjacent pairs: qwen35 is LLAMA_ROPE_TYPE_IMROPE and
  // ggml routes NEOX/MROPE/IMROPE through rotate_pairs(n_dims, n_dims/2, ...)
  // (ggml-cpu/ops.cpp) — element i rotates with element i + n_rot/2 while the
  // cache index still steps by 2 (cache[i0] belongs to pair i0/2, so the angle
  // of pair p is pos * freq_base^(-2p/n_rot), unchanged).
  float *base = x + ((std::int64_t)(t * n_heads + h)) * head_dim;
  const int jc = p + n_rot / 2;
  const float x0 = base[p];
  const float x1 = base[jc];
  base[p] = x0 * c - x1 * s;
  base[jc] = x0 * s + x1 * c;
}

inline bool rope_launch(float *d_x, int n_tokens, int n_heads, int head_dim, int n_rot,
                        float freq_base, const int *d_pos, hipStream_t stream = nullptr) {
  const int total = n_tokens * n_heads * (n_rot / 2);
  const int threads = 128;
  rope_kernel<<<(total + threads - 1) / threads, threads, 0, stream>>>(
      d_x, n_tokens, n_heads, head_dim, n_rot, freq_base, d_pos);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Causal attention for ONE query token against a KV cache.
//   q, out: [n_head, head_dim] of token t (single token, not the cache)
//   k, v  : cache rows [max_ctx, n_head_kv]; row j = token j, stored in the
//           type given by the KT/VT template parameters (kv.h)
// The cache must already hold token t at row t; keys 0..t are attended.
//
// Flash-attention style: one block per head, `WPB` warps splitting the keys,
// ONLINE softmax with a running max/sum — so there is no per-key shared array
// and the kernel works at any context length (the naive "score per key in
// shared memory" version needs O(t) shared memory and a block-wide reduction
// per key, which caps t at a few thousand and is very slow).
// GQA: query head h reads kv head h / (n_head/n_head_kv) (contiguous grouping).
//
// Per warp: lane owns head_dim/32 dims of every row; the score is a warp
// shuffle reduction, the accumulation is local, and the WPB slices are merged
// through a small shared buffer at the end.
// ---------------------------------------------------------------------------
constexpr int kAttnWarpsPerBlock = 8;  // key slices per block
constexpr int kAttnMaxDimsPerLane = 16; // head_dim/32 <= 16 (head_dim <= 512)

template <KvType KT, KvType VT>
__global__ void attn_kernel(const float *__restrict__ q, const void *__restrict__ k,
                            const void *__restrict__ v, float *__restrict__ out, int t,
                            int n_head, int n_head_kv, int head_dim, float dscale) {
  extern __shared__ float smem[];

  const int h = blockIdx.x;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kvh = h / (n_head / n_head_kv);
  const int dpw = head_dim / 32;  // dims per lane
  // this warp's slice: [running max, running sum, accumulator[head_dim]]
  float *pm = smem + (std::int64_t)w * (2 + head_dim);

  const std::uint64_t krow = kv_row_bytes(KT, head_dim);
  const std::uint64_t vrow = kv_row_bytes(VT, head_dim);
  const char *kbase = (const char *)k + ((std::int64_t)kvh * head_dim) * 0;  // set per key below
  (void)kbase;

  float qv[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) qv[i] = q[h * head_dim + lane * dpw + i];

  float m = -INFINITY;
  float l = 0.0f;
  float acc[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) acc[i] = 0.0f;

  for (int j = w; j <= t; j += kAttnWarpsPerBlock) {
    const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krow;
    float kk[kAttnMaxDimsPerLane];
    if (dpw == 8) {
      // the common case: one vectorized, coalesced load per lane per row
      kv_load8<KT>(kr, lane, kk);
    } else {
      for (int i = 0; i < dpw; ++i) kk[i] = kv_load<KT>(kr, lane * dpw + i);
    }
    float partial = 0.0f;
    for (int i = 0; i < dpw; ++i) partial = fmaf(qv[i], kk[i], partial);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffffull, partial, off);
    const float score = partial * dscale;

    // online softmax
    if (score > m) {
      const float corr = (m == -INFINITY) ? 0.0f : expf(m - score);
      l *= corr;
      for (int i = 0; i < dpw; ++i) acc[i] *= corr;
      m = score;
    }
    const float p = (m == -INFINITY) ? 0.0f : expf(score - m);
    l += p;
    const char *vr = (const char *)v + ((std::int64_t)j * n_head_kv + kvh) * vrow;
    float vv[kAttnMaxDimsPerLane];
    if (dpw == 8) {
      kv_load8<VT>(vr, lane, vv);
    } else {
      for (int i = 0; i < dpw; ++i) vv[i] = kv_load<VT>(vr, lane * dpw + i);
    }
    for (int i = 0; i < dpw; ++i) acc[i] = fmaf(p, vv[i], acc[i]);
  }

  // Merge the WPB key slices: every lane publishes its own dims, so the whole
  // head_dim-wide accumulator of each slice is available after the barrier.
  if (lane == 0) {
    pm[0] = m;
    pm[1] = l;
  }
  for (int i = 0; i < dpw; ++i) pm[2 + lane * dpw + i] = acc[i];
  __syncthreads();

  float mm = -INFINITY;
  for (int s = 0; s < kAttnWarpsPerBlock; ++s) mm = fmaxf(mm, smem[(std::int64_t)s * (2 + head_dim)]);
  float ll = 0.0f;
  for (int s = 0; s < kAttnWarpsPerBlock; ++s) {
    const float *ps = smem + (std::int64_t)s * (2 + head_dim);
    ll += ps[1] * expf(ps[0] - mm);
  }
  const float inv = (ll > 0.0f) ? 1.0f / ll : 0.0f;
  for (int i = 0; i < dpw; ++i) {
    float a = 0.0f;
    for (int s = 0; s < kAttnWarpsPerBlock; ++s) {
      const float *ps = smem + (std::int64_t)s * (2 + head_dim);
      a += ps[2 + lane * dpw + i] * expf(ps[0] - mm);
    }
    out[h * head_dim + lane * dpw + i] = a * inv;
  }
}

template <KvType KT, KvType VT>
inline bool attn_launch_typed(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                              int t, int n_head, int n_head_kv, int head_dim, float scale,
                              hipStream_t stream) {
  if (head_dim % 32 != 0 || head_dim / 32 > kAttnMaxDimsPerLane) return false;
  const int threads = kAttnWarpsPerBlock * 32;
  const std::size_t smem =
      (std::size_t)kAttnWarpsPerBlock * (2 + (std::size_t)head_dim) * sizeof(float);
  attn_kernel<KT, VT><<<n_head, threads, smem, stream>>>(d_q, d_k, d_v, d_out, t, n_head,
                                                         n_head_kv, head_dim, scale);
  return hipGetLastError() == hipSuccess;
}

inline bool attn_launch(const float *d_q, const void *d_k, const void *d_v, float *d_out, int t,
                        int n_head, int n_head_kv, int head_dim, float scale, KvType kt,
                        KvType vt, hipStream_t stream = nullptr) {
#define RD_ATTN_CASE(K, V)                                                                        \
  if (kt == KvType::K && vt == KvType::V)                                                         \
  return attn_launch_typed<KvType::K, KvType::V>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv,      \
                                                 head_dim, scale, stream)
  RD_ATTN_CASE(F32, F32);
  RD_ATTN_CASE(F32, F16);
  RD_ATTN_CASE(F32, Q8_0);
  RD_ATTN_CASE(F32, Q4_0);
  RD_ATTN_CASE(F16, F32);
  RD_ATTN_CASE(F16, F16);
  RD_ATTN_CASE(F16, Q8_0);
  RD_ATTN_CASE(F16, Q4_0);
  RD_ATTN_CASE(Q8_0, F32);
  RD_ATTN_CASE(Q8_0, F16);
  RD_ATTN_CASE(Q8_0, Q8_0);
  RD_ATTN_CASE(Q8_0, Q4_0);
  RD_ATTN_CASE(Q4_0, F32);
  RD_ATTN_CASE(Q4_0, F16);
  RD_ATTN_CASE(Q4_0, Q8_0);
  RD_ATTN_CASE(Q4_0, Q4_0);
#undef RD_ATTN_CASE
  return false;
}

}  // namespace rdna4
