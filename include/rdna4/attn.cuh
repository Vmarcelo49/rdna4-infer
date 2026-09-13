#pragma once
// M3 — full-attention branch primitives for qwen35 (full-attention layers).
//
// Layouts and semantics follow llama.cpp's qwen35 graph (src/models/qwen35.cpp,
// build_layer_attn) and its ggml lowering:
//   attn_q output per head is [ q (head_dim) | gate (head_dim) ], stride 2*head_dim
//     Qcur = view at offset 0, gate = view at offset head_dim
//   Q/K are RMS-normalized per head (attn_q_norm / attn_k_norm, size head_dim)
//   RoPE: GGML_ROPE_TYPE_IMROPE, which for text (t=h=w positions) is plain
//     adjacent-pair RoPE over n_rot dims of each head (rope.dimension_count)
//   attention scale = 1/sqrt(head_dim); the attention output is multiplied by
//     sigmoid(gate) before attn_output
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

namespace rdna4 {

// ---------------------------------------------------------------------------
// RoPE on adjacent pairs of the first n_rot dims of each head:
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
//   k, v  : cache layout [max_ctx, n_head_kv, head_dim] (row j = token j)
// The cache must already hold token t at row t; keys 0..t are attended.
// Token t attends to cache rows 0..t (the cache holds every token processed so
// far, so the causal bound is simply j <= t).
// One block per head, head_dim threads; GQA: head h reads kv head
// h / (n_head/n_head_kv). Shared memory: head_dim (reduction) + (t+2) scores.
// Correctness-first: the scores are materialised, so softmax is a single pass.
// ---------------------------------------------------------------------------
__global__ void attn_kernel(const float *__restrict__ q, const float *__restrict__ k,
                            const float *__restrict__ v, float *__restrict__ out, int t,
                            int n_head, int n_head_kv, int head_dim, float scale) {
  extern __shared__ float smem[];
  float *red = smem;            // head_dim
  float *sc = smem + head_dim;  // t + 2

  const int h = blockIdx.x;
  const int d = threadIdx.x;
  // GQA grouping is CONTIGUOUS: the kv heads are tiled per group (HF's
  // repeat_kv, and what llama.cpp does for this model), so query head h reads
  // kv head h / (n_head/n_head_kv). Verified against the per-token oracle at
  // layer 3: contiguous gives attn_pregate-3 sum rel 1.1e-04, while grouping
  // by h % n_head_kv gives 3.2e-02 and a wrong argmax.
  const int kvh = h / (n_head / n_head_kv);

  // q and out hold the CURRENT token only: [n_head, head_dim]. The cache holds
  // the whole history: k/v are [t+1, n_head_kv, head_dim] rows 0..t.
  const float qv = q[h * head_dim + d];

  for (int j = 0; j <= t; ++j) {
    red[d] = qv * k[((std::int64_t)j * n_head_kv + kvh) * head_dim + d];
    __syncthreads();
    for (int s = head_dim / 2; s > 0; s >>= 1) {
      if (d < s) red[d] += red[d + s];
      __syncthreads();
    }
    if (d == 0) sc[j] = red[0] * scale;
    __syncthreads();  // sc[j] is written before red is reused
  }

  if (d == 0) {
    float m = -INFINITY;
    for (int j = 0; j <= t; ++j) m = fmaxf(m, sc[j]);
    float sum = 0.0f;
    for (int j = 0; j <= t; ++j) {
      sc[j] = expf(sc[j] - m);
      sum += sc[j];
    }
    sc[t + 1] = sum;
  }
  __syncthreads();
  const float inv = 1.0f / sc[t + 1];

  float acc = 0.0f;
  for (int j = 0; j <= t; ++j) {
    acc += (sc[j] * inv) * v[((std::int64_t)j * n_head_kv + kvh) * head_dim + d];
  }
  out[h * head_dim + d] = acc;
}

inline bool attn_launch(const float *d_q, const float *d_k, const float *d_v, float *d_out, int t,
                        int n_head, int n_head_kv, int head_dim, float scale,
                        hipStream_t stream = nullptr) {
  const std::size_t smem = (std::size_t)(head_dim + t + 2) * sizeof(float);
  attn_kernel<<<n_head, head_dim, smem, stream>>>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv,
                                                 head_dim, scale);
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
