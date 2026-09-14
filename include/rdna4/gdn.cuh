#pragma once
// M3 — gated delta net (GDN / linear attention) kernels for qwen35.
//
// Semantics transcribed from llama.cpp's reference:
//   src/models/qwen35.cpp  build_layer_attn_linear (L335-468)
//   src/models/delta-net-base.cpp build_conv_state / build_recurrent_attn
//   ggml/src/ggml-cpu/ops.cpp  ggml_compute_forward_gated_delta_net_one_chunk
// In particular the state is stored TRANSPOSED: M[j*S + i] = S[i][j], so row j of
// M is column j of S, which makes every step of the recurrence local to one
// thread (thread j owns row j) and needs no synchronisation inside a head.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "rdna4/nn.cuh"  // silu_f

namespace rdna4 {

// ---------------------------------------------------------------------------
// Causal depthwise conv1d with a rolling state, then SiLU (ggml_ssm_conv +
// ggml_silu). conv_w is [K, channels]; state holds the previous K-1 inputs.
// One thread per channel.
// ---------------------------------------------------------------------------
__global__ void conv1d_state_kernel(const float *__restrict__ qkv, const float *__restrict__ w,
                                    float *__restrict__ out, float *__restrict__ state, int channels,
                                    int K) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  // conv_w is [K, channels] in ggml order (ne0 = K is the fastest dim), so the
  // tap k of channel c lives at c*K + k — NOT k*channels + c.
  float acc = 0.0f;
  for (int k = 0; k < K - 1; ++k) acc += w[c * K + k] * state[k * channels + c];
  acc += w[c * K + (K - 1)] * qkv[c];
  out[c] = silu_f(acc);
  for (int k = 0; k < K - 2; ++k) state[k * channels + c] = state[(k + 1) * channels + c];
  state[(K - 2) * channels + c] = qkv[c];
}

inline bool conv1d_state_launch(const float *d_qkv, const float *d_w, float *d_out, float *d_state,
                                int channels, int K, hipStream_t stream = nullptr) {
  const int threads = 256;
  conv1d_state_kernel<<<(channels + threads - 1) / threads, threads, 0, stream>>>(
      d_qkv, d_w, d_out, d_state, channels, K);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Gated delta rule for one token, one block per value head, one thread per row j
// of the transposed state. q/k heads are the GQA-repeated ones (head h reads
// k head h % n_k_heads).
//   S *= exp(gate[h])                      (per value head; gate is scalar per head)
//   delta[j] = (v[j] - dot(M[j], k)) * beta[h]
//   M[j] += k * delta[j]
//   out[j] = dot(M[j], q) / sqrt(S)
// ---------------------------------------------------------------------------
__global__ void delta_rule_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                  const float *__restrict__ v, const float *__restrict__ gate,
                                  const float *__restrict__ beta, float *__restrict__ state,
                                  float *__restrict__ out, [[maybe_unused]] int n_v_heads,
                                  int n_k_heads, int S) {
  const int h = blockIdx.x;      // value head
  const int j = threadIdx.x;     // row index (value dim)
  if (j >= S) return;
  const int kh = h % n_k_heads;  // repeated key head

  const float *kd = k + (std::int64_t)kh * S;
  const float *qd = q + (std::int64_t)kh * S;
  float *row = state + ((std::int64_t)h * S + j) * S;

  const float g = expf(gate[h]);
  const float b = beta[h];

  float sum = 0.0f;
  for (int i = 0; i < S; ++i) {
    row[i] *= g;
    sum = fmaf(row[i], kd[i], sum);
  }
  const float delta = (v[(std::int64_t)h * S + j] - sum) * b;

  float acc = 0.0f;
  for (int i = 0; i < S; ++i) {
    row[i] = fmaf(kd[i], delta, row[i]);
    acc = fmaf(row[i], qd[i], acc);
  }
  out[(std::int64_t)h * S + j] = acc * rsqrtf((float)S);
}

inline bool delta_rule_launch(const float *d_q, const float *d_k, const float *d_v,
                              const float *d_gate, const float *d_beta, float *d_state,
                              float *d_out, int n_v_heads, int n_k_heads, int S,
                              hipStream_t stream = nullptr) {
  delta_rule_kernel<<<n_v_heads, ((S + 31) / 32) * 32, 0, stream>>>(
      d_q, d_k, d_v, d_gate, d_beta, d_state, d_out, n_v_heads, n_k_heads, S);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// De-interleave the attn_q projection output into contiguous Q and gate buffers.
// Per head the layout is [ q (head_dim) | gate (head_dim) ] (build_layer_attn).
// ---------------------------------------------------------------------------
__global__ void deinterleave_q_gate_kernel(const float *__restrict__ yq, float *__restrict__ q,
                                           float *__restrict__ gate, int n_head, int head_dim) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n_head * head_dim) return;
  const int d = idx % head_dim;
  const int h = idx / head_dim;
  q[idx] = yq[(std::int64_t)h * 2 * head_dim + d];
  gate[idx] = yq[(std::int64_t)h * 2 * head_dim + head_dim + d];
}

inline bool deinterleave_q_gate_launch(const float *d_yq, float *d_q, float *d_gate, int n_head,
                                       int head_dim, hipStream_t stream = nullptr) {
  const int n = n_head * head_dim;
  const int threads = 256;
  deinterleave_q_gate_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(d_yq, d_q, d_gate,
                                                                                 n_head, head_dim);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// BATCHED PREFILL (feat/noite-prefill): the same three kernels with the N tokens
// of one forward_batch chunk folded into a single launch each.
//
// The invariant is the one the whole batched path is built on (graph.cuh:153-163,
// tests/check_batch_gpu.hip): every token keeps the arithmetic it has in the
// per-token path, so the result is BIT-IDENTICAL and the gate stays meaningful.
// That is why the token loop lives INSIDE each kernel and walks the tokens in
// order: the state is sequential, so a token's inputs must be produced by the
// previous token before it is consumed, and the operation order inside a token is
// copied verbatim from the per-token kernels above.
//
// What it buys (measured, docs/journal-prefill.md): the per-token path launched
// these kernels once per token and re-read the whole recurrent state from
// L2/DRAM every time (measured 152 GB/s of an available 633 GB/s, 69.6 us per
// delta_rule launch against a 20 us memory floor); with the whole chunk in one
// launch the state of the layer stays resident across the N tokens.
// ---------------------------------------------------------------------------

// conv1d_state_kernel, but walking the chunk's tokens in order. One thread per
// channel; the channel's K-1 taps live in `state` and are shifted exactly as
// before after every token. `out` is SPLIT (q and k into qk_out at c, v into
// v_out) so that the following l2_norm / rms_norm / delta_rule see contiguous
// rows -- the values written are the same ones the per-token kernel writes.
__global__ void conv1d_state_batch_kernel(const float *__restrict__ qkv,
                                          const float *__restrict__ w,
                                          float *__restrict__ qk_out, float *__restrict__ v_out,
                                          float *__restrict__ state, int channels, int K,
                                          int key_dim, int d_inner, int n) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  const bool is_v = c >= 2 * key_dim;
  const int out_col = is_v ? c - 2 * key_dim : c;
  float *dst = is_v ? v_out + out_col : qk_out + out_col;
  const int out_stride = is_v ? d_inner : 2 * key_dim;
  for (int t = 0; t < n; ++t) {
    const float x = qkv[(std::int64_t)t * channels + c];
    float acc = 0.0f;
    for (int k = 0; k < K - 1; ++k) acc += w[c * K + k] * state[k * channels + c];
    acc += w[c * K + (K - 1)] * x;
    dst[(std::int64_t)t * out_stride] = silu_f(acc);
    for (int k = 0; k < K - 2; ++k) state[k * channels + c] = state[(k + 1) * channels + c];
    state[(K - 2) * channels + c] = x;
  }
}

inline bool conv1d_state_batch_launch(const float *d_qkv, const float *d_w, float *d_qk_out,
                                      float *d_v_out, float *d_state, int channels, int K,
                                      int key_dim, int d_inner, int n,
                                      hipStream_t stream = nullptr) {
  const int threads = 256;
  conv1d_state_batch_kernel<<<(channels + threads - 1) / threads, threads, 0, stream>>>(
      d_qkv, d_w, d_qk_out, d_v_out, d_state, channels, K, key_dim, d_inner, n);
  return hipGetLastError() == hipSuccess;
}

// delta_rule_kernel with the chunk's tokens walked in order inside one CTA per
// value head: the state row of thread j stays hot across the N tokens instead of
// being re-fetched by a new launch per token. `ROWS` is the number of j rows one
// CTA owns (a SHAPE choice: rows are independent, so splitting them across CTAs
// changes no arithmetic); grid.y indexes the row blocks.
//
// Per token, per row, the body is the per-token kernel's body word for word:
//   S *= exp(gate); delta = (v - <M,k>) * beta; M += k*delta; out = <M,q>/sqrt(S)
template <int ROWS>
__global__ void delta_rule_batch_rows_kernel(const float *__restrict__ qk,
                                             const float *__restrict__ v,
                                             const float *__restrict__ gate,
                                             const float *__restrict__ beta,
                                             float *__restrict__ state, float *__restrict__ out,
                                             int n_k_heads, int S, int n_vh, int key_dim,
                                             int d_inner, int n) {
  const int h = blockIdx.x;                 // value head
  const int j = blockIdx.y * ROWS + threadIdx.x;  // row index (value dim)
  if (j >= S) return;
  const int kh = h % n_k_heads;             // repeated key head
  const float *kb = qk + key_dim + (std::int64_t)kh * S;
  const float *qb = qk + (std::int64_t)kh * S;
  float *row = state + ((std::int64_t)h * S + j) * S;
  for (int t = 0; t < n; ++t) {
    const float *kd = kb + (std::int64_t)t * 2 * key_dim;
    const float *qd = qb + (std::int64_t)t * 2 * key_dim;
    const float g = expf(gate[(std::int64_t)t * n_vh + h]);
    const float b = beta[(std::int64_t)t * n_vh + h];
    float sum = 0.0f;
    for (int i = 0; i < S; ++i) {
      row[i] *= g;
      sum = fmaf(row[i], kd[i], sum);
    }
    const float delta = (v[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] - sum) * b;
    float acc = 0.0f;
    for (int i = 0; i < S; ++i) {
      row[i] = fmaf(kd[i], delta, row[i]);
      acc = fmaf(row[i], qd[i], acc);
    }
    out[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] = acc * rsqrtf((float)S);
  }
}

// rows per CTA: S when S <= 128 (one CTA per head, the per-token geometry), else
// split. 32 rows = 16 KiB of state per CTA, which is what keeps the row working
// set inside L1/L2 for the whole chunk.
inline bool delta_rule_batch_launch(const float *d_qk, const float *d_v, const float *d_gate,
                                    const float *d_beta, float *d_state, float *d_out,
                                    int n_v_heads, int n_k_heads, int S, int n_vh, int key_dim,
                                    int d_inner, int n, hipStream_t stream = nullptr) {
  constexpr int kRows = 32;
  const unsigned ny = (unsigned)((S + kRows - 1) / kRows);
  delta_rule_batch_rows_kernel<kRows><<<dim3((unsigned)n_v_heads, ny), kRows, 0, stream>>>(
      d_qk, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S, n_vh, key_dim, d_inner, n);
  return hipGetLastError() == hipSuccess;
}

// deinterleave_q_gate_kernel over the whole chunk: q/gate are [n][n_head*head_dim]
// and the projection output is [n][n_head*2*head_dim].
__global__ void deinterleave_q_gate_batch_kernel(const float *__restrict__ yq,
                                                 float *__restrict__ q,
                                                 float *__restrict__ gate, int n_head, int head_dim,
                                                 int n) {
  const std::int64_t total = (std::int64_t)n * n_head * head_dim;
  for (std::int64_t idx = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
       idx += (std::int64_t)gridDim.x * blockDim.x) {
    const int d = (int)(idx % head_dim);
    const int h = (int)((idx / head_dim) % n_head);
    const std::int64_t t = idx / ((std::int64_t)n_head * head_dim);
    q[idx] = yq[(t * n_head + h) * 2 * head_dim + d];
    gate[idx] = yq[(t * n_head + h) * 2 * head_dim + head_dim + d];
  }
}

inline bool deinterleave_q_gate_batch_launch(const float *d_yq, float *d_q, float *d_gate,
                                             int n_head, int head_dim, int n,
                                             hipStream_t stream = nullptr) {
  const std::int64_t total = (std::int64_t)n * n_head * head_dim;
  const int threads = 256;
  const unsigned blocks = (unsigned)((total + threads - 1) / threads);
  deinterleave_q_gate_batch_kernel<<<blocks, threads, 0, stream>>>(d_yq, d_q, d_gate, n_head,
                                                                  head_dim, n);
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
