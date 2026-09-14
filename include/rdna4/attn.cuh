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
#include "rdna4/tuning.h"

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
// key slices per block (shipped value). O valor e o de include/rdna4/tuning.h
// (tabela unica, medida nesta placa): nao ha uma segunda copia do numero aqui.
constexpr int kAttnWarpsPerBlock = tuned::kAttnWarpsPerBlock;
constexpr int kAttnMaxDimsPerLane = 16; // head_dim/32 <= 16 (head_dim <= 512)

// WPB is a TEMPLATE knob (default = the shipped constant) so the bench can A/B
// it without a second copy of the kernel. Changing it changes the number of
// partial slices merged at the end of the CTA, i.e. the summation order across
// key slices -- so the shipped default is kept wherever a bit-exact path is
// expected, and any other value is validated as a numeric-equivalence change
// (scripts/check_attn_split.sh, check-kvctx-gpu).
template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
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
  if (mm == -INFINITY) {  // no keys at all
    for (int i = 0; i < dpw; ++i) out[h * head_dim + lane * dpw + i] = 0.0f;
    return;
  }
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

template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
inline bool attn_launch_typed(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                              int t, int n_head, int n_head_kv, int head_dim, float scale,
                              hipStream_t stream) {
  if (head_dim % 32 != 0 || head_dim / 32 > kAttnMaxDimsPerLane) return false;
  const int threads = WPB * 32;
  const std::size_t smem = (std::size_t)WPB * (2 + (std::size_t)head_dim) * sizeof(float);
  attn_kernel<KT, VT, WPB><<<n_head, threads, smem, stream>>>(d_q, d_k, d_v, d_out, t, n_head,
                                                              n_head_kv, head_dim, scale);
  return hipGetLastError() == hipSuccess;
}

// WPB dispatch for the single-CTA (unsplit) kernel (bench use). Same
// numeric-equivalence caveat as the split one: the number of partial slices
// merged per CTA changes the summation order across key slices.
template <KvType KT, KvType VT>
inline bool attn_launch_wpb_typed(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                                  int t, int n_head, int n_head_kv, int head_dim, float scale,
                                  int wpb, hipStream_t stream) {
  switch (wpb) {
    case 8:
      return attn_launch_typed<KT, VT, 8>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv, head_dim,
                                          scale, stream);
    case 16:
      return attn_launch_typed<KT, VT, 16>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv, head_dim,
                                           scale, stream);
    case 32:
      return attn_launch_typed<KT, VT, 32>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv, head_dim,
                                           scale, stream);
    default: return false;
  }
}

inline bool attn_launch_wpb(const float *d_q, const void *d_k, const void *d_v, float *d_out, int t,
                            int n_head, int n_head_kv, int head_dim, float scale, KvType kt,
                            KvType vt, int wpb, hipStream_t stream = nullptr) {
#define RD_ATTN_U_CASE(K, V)                                                                      \
  if (kt == KvType::K && vt == KvType::V)                                                         \
  return attn_launch_wpb_typed<KvType::K, KvType::V>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv,  \
                                                     head_dim, scale, wpb, stream)
  RD_ATTN_U_CASE(F32, F32);
  RD_ATTN_U_CASE(F16, F16);
  RD_ATTN_U_CASE(Q8_0, Q8_0);
  RD_ATTN_U_CASE(Q4_0, Q4_0);
  RD_ATTN_U_CASE(F16, Q8_0);
  RD_ATTN_U_CASE(F16, Q4_0);
  RD_ATTN_U_CASE(Q8_0, F16);
  RD_ATTN_U_CASE(Q4_0, F16);
  RD_ATTN_U_CASE(F32, F16);
  RD_ATTN_U_CASE(F16, F32);
  RD_ATTN_U_CASE(F32, Q8_0);
  RD_ATTN_U_CASE(F32, Q4_0);
  RD_ATTN_U_CASE(Q8_0, F32);
  RD_ATTN_U_CASE(Q4_0, F32);
  RD_ATTN_U_CASE(Q8_0, Q4_0);
  RD_ATTN_U_CASE(Q4_0, Q8_0);
  // The new formats, diagonal only: this entry point is the WPB A/B knob used by
  // tests/bench_attn_gpu.hip, which benches K and V at the same type. The shipped
  // path (attn_launch below) carries the full 6x6 product.
  RD_ATTN_U_CASE(Q5_0, Q5_0);
  RD_ATTN_U_CASE(Q4_1, Q4_1);
#undef RD_ATTN_U_CASE
  return false;
}

inline bool attn_launch(const float *d_q, const void *d_k, const void *d_v, float *d_out, int t,
                        int n_head, int n_head_kv, int head_dim, float scale, KvType kt,
                        KvType vt, hipStream_t stream = nullptr) {
  // The full 6x6 product: every (K, V) pair the CLI can name is instantiated, so
  // `--cache-type-k q5_0 --cache-type-v q4_1` cannot silently fall off the end of
  // this dispatch and return false (the previous list was 4x4 minus f32/f32
  // mirrors, i.e. a hand-maintained subset).
#define RD_ATTN_CASE(K, V)                                                                        \
  if (kt == KvType::K && vt == KvType::V)                                                         \
  return attn_launch_typed<KvType::K, KvType::V>(d_q, d_k, d_v, d_out, t, n_head, n_head_kv,      \
                                                 head_dim, scale, stream)
  RD_ATTN_CASE(F32, F32);
  RD_ATTN_CASE(F32, F16);
  RD_ATTN_CASE(F32, Q8_0);
  RD_ATTN_CASE(F32, Q4_0);
  RD_ATTN_CASE(F32, Q5_0);
  RD_ATTN_CASE(F32, Q4_1);
  RD_ATTN_CASE(F16, F32);
  RD_ATTN_CASE(F16, F16);
  RD_ATTN_CASE(F16, Q8_0);
  RD_ATTN_CASE(F16, Q4_0);
  RD_ATTN_CASE(F16, Q5_0);
  RD_ATTN_CASE(F16, Q4_1);
  RD_ATTN_CASE(Q8_0, F32);
  RD_ATTN_CASE(Q8_0, F16);
  RD_ATTN_CASE(Q8_0, Q8_0);
  RD_ATTN_CASE(Q8_0, Q4_0);
  RD_ATTN_CASE(Q8_0, Q5_0);
  RD_ATTN_CASE(Q8_0, Q4_1);
  RD_ATTN_CASE(Q4_0, F32);
  RD_ATTN_CASE(Q4_0, F16);
  RD_ATTN_CASE(Q4_0, Q8_0);
  RD_ATTN_CASE(Q4_0, Q4_0);
  RD_ATTN_CASE(Q4_0, Q5_0);
  RD_ATTN_CASE(Q4_0, Q4_1);
  RD_ATTN_CASE(Q5_0, F32);
  RD_ATTN_CASE(Q5_0, F16);
  RD_ATTN_CASE(Q5_0, Q8_0);
  RD_ATTN_CASE(Q5_0, Q4_0);
  RD_ATTN_CASE(Q5_0, Q5_0);
  RD_ATTN_CASE(Q5_0, Q4_1);
  RD_ATTN_CASE(Q4_1, F32);
  RD_ATTN_CASE(Q4_1, F16);
  RD_ATTN_CASE(Q4_1, Q8_0);
  RD_ATTN_CASE(Q4_1, Q4_0);
  RD_ATTN_CASE(Q4_1, Q5_0);
  RD_ATTN_CASE(Q4_1, Q4_1);
#undef RD_ATTN_CASE
  return false;
}


// ---------------------------------------------------------------------------
// BATCHED PREFILL (feat/noite-prefill): the unsplit kernel above, with all the
// query tokens of one forward_batch chunk in blockIdx.y.
//
// The body is attn_kernel's, element for element -- same warp-sliced key walk
// (j = w; j <= t; j += WPB), same online softmax, same in-CTA merge across WPB
// slices, same expf/1/l rounding -- with only the query/output addressing moved
// from "the single token" to "token qt of the batch". Each (qt, h) CTA therefore
// computes exactly what the per-token launch computed for that token, which is
// what keeps tests/check_batch_gpu.hip BIT-EXACT.
//
// Precondition (checked by the caller): every query token in the batch uses the
// UNSPLIT path (keys < kAttnSplitMin). A token whose key range would be split
// across CTAs sums its slices in a different order (attn.cuh:289), so for those
// chunks the per-token loop is kept and this kernel is not used.
// ---------------------------------------------------------------------------
template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
__global__ void attn_batch_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                  const void *__restrict__ v, float *__restrict__ out,
                                  const int *__restrict__ pos, int n_head, int n_head_kv,
                                  int head_dim, float dscale) {
  extern __shared__ float smem[];

  const int h = blockIdx.x;
  const int qt = blockIdx.y;
  const int t = pos[qt];  // this query token's position == the last key it attends
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kvh = h / (n_head / n_head_kv);
  const int dpw = head_dim / 32;  // dims per lane
  float *pm = smem + (std::int64_t)w * (2 + head_dim);

  const std::uint64_t krow = kv_row_bytes(KT, head_dim);
  const std::uint64_t vrow = kv_row_bytes(VT, head_dim);
  const float *qp = q + ((std::int64_t)qt * n_head + h) * head_dim;
  float *op = out + ((std::int64_t)qt * n_head + h) * head_dim;

  float qv[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) qv[i] = qp[lane * dpw + i];

  float m = -INFINITY;
  float l = 0.0f;
  float acc[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) acc[i] = 0.0f;

  for (int j = w; j <= t; j += kAttnWarpsPerBlock) {
    const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krow;
    float kk[kAttnMaxDimsPerLane];
    if (dpw == 8) {
      kv_load8<KT>(kr, lane, kk);
    } else {
      for (int i = 0; i < dpw; ++i) kk[i] = kv_load<KT>(kr, lane * dpw + i);
    }
    float partial = 0.0f;
    for (int i = 0; i < dpw; ++i) partial = fmaf(qv[i], kk[i], partial);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffffull, partial, off);
    const float score = partial * dscale;

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

  if (lane == 0) {
    pm[0] = m;
    pm[1] = l;
  }
  for (int i = 0; i < dpw; ++i) pm[2 + lane * dpw + i] = acc[i];
  __syncthreads();

  float mm = -INFINITY;
  for (int s = 0; s < kAttnWarpsPerBlock; ++s) mm = fmaxf(mm, smem[(std::int64_t)s * (2 + head_dim)]);
  if (mm == -INFINITY) {  // no keys at all
    for (int i = 0; i < dpw; ++i) op[lane * dpw + i] = 0.0f;
    return;
  }
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
    op[lane * dpw + i] = a * inv;
  }
}

template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
inline bool attn_batch_launch_typed(const float *d_q, const void *d_k, const void *d_v,
                                    float *d_out, const int *d_pos, int n_tok, int n_head,
                                    int n_head_kv, int head_dim, float scale, hipStream_t stream) {
  if (head_dim % 32 != 0 || head_dim / 32 > kAttnMaxDimsPerLane) return false;
  const int threads = WPB * 32;
  const std::size_t smem = (std::size_t)WPB * (2 + (std::size_t)head_dim) * sizeof(float);
  dim3 grid((unsigned)n_head, (unsigned)n_tok);
  attn_batch_kernel<KT, VT, WPB><<<grid, threads, smem, stream>>>(d_q, d_k, d_v, d_out, d_pos,
                                                                 n_head, n_head_kv, head_dim,
                                                                 scale);
  return hipGetLastError() == hipSuccess;
}

inline bool attn_batch_launch(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                              const int *d_pos, int n_tok, int n_head, int n_head_kv, int head_dim,
                              float scale, KvType kt, KvType vt, hipStream_t stream = nullptr) {
#define RD_ATTN_B_CASE(K, V)                                                                      \
  if (kt == KvType::K && vt == KvType::V)                                                          \
  return attn_batch_launch_typed<KvType::K, KvType::V>(d_q, d_k, d_v, d_out, d_pos, n_tok, n_head, \
                                                       n_head_kv, head_dim, scale, stream)
  RD_ATTN_B_CASE(F32, F32);
  RD_ATTN_B_CASE(F32, F16);
  RD_ATTN_B_CASE(F32, Q8_0);
  RD_ATTN_B_CASE(F32, Q4_0);
  RD_ATTN_B_CASE(F16, F32);
  RD_ATTN_B_CASE(F16, F16);
  RD_ATTN_B_CASE(F16, Q8_0);
  RD_ATTN_B_CASE(F16, Q4_0);
  RD_ATTN_B_CASE(Q8_0, F32);
  RD_ATTN_B_CASE(Q8_0, F16);
  RD_ATTN_B_CASE(Q8_0, Q8_0);
  RD_ATTN_B_CASE(Q8_0, Q4_0);
  RD_ATTN_B_CASE(Q4_0, F32);
  RD_ATTN_B_CASE(Q4_0, F16);
  RD_ATTN_B_CASE(Q4_0, Q8_0);
  RD_ATTN_B_CASE(Q4_0, Q4_0);
  // Q5_0/Q4_1 (frente KV): the full product, for the same reason as attn_launch
  // above -- this dispatcher was added by the prefill front while the KV front was
  // adding the two formats, and the merge left it as a 4x4 minus-mirrors list. The
  // caller in graph.cuh treats `false` as a hard error ("batch attn (batched)
  // launch failed"), so a missing pair here does NOT degrade: it aborts the batched
  // prefill of any run with --cache-type-k q5_0. Found by re-reading the merge.
  RD_ATTN_B_CASE(F32, Q5_0);
  RD_ATTN_B_CASE(F32, Q4_1);
  RD_ATTN_B_CASE(F16, Q5_0);
  RD_ATTN_B_CASE(F16, Q4_1);
  RD_ATTN_B_CASE(Q8_0, Q5_0);
  RD_ATTN_B_CASE(Q8_0, Q4_1);
  RD_ATTN_B_CASE(Q4_0, Q5_0);
  RD_ATTN_B_CASE(Q4_0, Q4_1);
  RD_ATTN_B_CASE(Q5_0, F32);
  RD_ATTN_B_CASE(Q5_0, F16);
  RD_ATTN_B_CASE(Q5_0, Q8_0);
  RD_ATTN_B_CASE(Q5_0, Q4_0);
  RD_ATTN_B_CASE(Q5_0, Q5_0);
  RD_ATTN_B_CASE(Q5_0, Q4_1);
  RD_ATTN_B_CASE(Q4_1, F32);
  RD_ATTN_B_CASE(Q4_1, F16);
  RD_ATTN_B_CASE(Q4_1, Q8_0);
  RD_ATTN_B_CASE(Q4_1, Q4_0);
  RD_ATTN_B_CASE(Q4_1, Q5_0);
  RD_ATTN_B_CASE(Q4_1, Q4_1);
#undef RD_ATTN_B_CASE
  return false;
}

// ---------------------------------------------------------------------------
// Split-KV attention (M7 step 2): the same flash-style kernel with the key range
// ALSO split across CTAs.
//
// Why: the single-CTA version is memory-latency-bound, not bandwidth-bound --
// measured 69 GB/s of ~600 GB/s at 64K with 24 CTAs (one per query head) on
// 64 CUs (docs/medicoes-m7.md). More in-flight requests need more CTAs, and the
// key range is the only dimension with room (the head dimension is already
// spread over the lanes).
//
// Layout: CTA (h, s) walks keys j = w + 32*s, ... step 32*S (so consecutive CTAs
// read adjacent 32-key blocks: coalesced, and neighbouring in L2), writing one
// partial (m, l, acc[head_dim]) per split. attn_merge_kernel then combines the S
// partials with the same online-softmax arithmetic the in-kernel merge uses.
//
// The result is NOT bit-identical to the single-CTA path (the summation order
// across keys changes), so callers must accept ~1e-7 relative; the gate is
// tests/bench_attn_gpu.hip's comparison against the unsplit kernel.
// ---------------------------------------------------------------------------
template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
__global__ void attn_split_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                  const void *__restrict__ v, float *__restrict__ partial,
                                  int t, int n_head, int n_head_kv, int head_dim, float dscale,
                                  int n_splits) {
  extern __shared__ float smem[];
  const int h = blockIdx.x;
  const int s = blockIdx.y;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int kvh = h / (n_head / n_head_kv);
  const int dpw = head_dim / 32;
  float *pm = smem + (std::int64_t)w * (2 + head_dim);

  const std::uint64_t krow = kv_row_bytes(KT, head_dim);
  const std::uint64_t vrow = kv_row_bytes(VT, head_dim);

  float qv[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) qv[i] = q[h * head_dim + lane * dpw + i];

  float m = -INFINITY;
  float l = 0.0f;
  float acc[kAttnMaxDimsPerLane];
  for (int i = 0; i < dpw; ++i) acc[i] = 0.0f;

  const int step = WPB * n_splits;
  for (int j = w + WPB * s; j <= t; j += step) {
    const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krow;
    float kk[kAttnMaxDimsPerLane];
    if (dpw == 8) {
      kv_load8<KT>(kr, lane, kk);
    } else {
      for (int i = 0; i < dpw; ++i) kk[i] = kv_load<KT>(kr, lane * dpw + i);
    }
    float partial_dot = 0.0f;
    for (int i = 0; i < dpw; ++i) partial_dot = fmaf(qv[i], kk[i], partial_dot);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      partial_dot += __shfl_xor_sync(0xffffffffull, partial_dot, off);
    const float score = partial_dot * dscale;
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

  // merge this CTA's WPB warps, then publish one partial for the CTA
  if (lane == 0) {
    pm[0] = m;
    pm[1] = l;
  }
  for (int i = 0; i < dpw; ++i) pm[2 + lane * dpw + i] = acc[i];
  __syncthreads();

  // Reduce this CTA's WPB warp slices into one partial, spread over the block:
  // the weights exp(m_i - mm) are computed once (one per warp), then every dim is
  // a 32-term dot product with them. (A single-thread version costs 32*head_dim
  // expf calls in one lane and becomes the kernel's serial tail.)
  __shared__ float wts[WPB];
  __shared__ float cmax;
  __syncthreads();
  if (threadIdx.x == 0) {
    float mm = -INFINITY;
    for (int i = 0; i < WPB; ++i)
      mm = fmaxf(mm, smem[(std::int64_t)i * (2 + head_dim)]);
    cmax = mm;
  }
  __syncthreads();
  if (threadIdx.x < WPB) {
    // An EMPTY split (fewer keys than CTAs, e.g. two splits at position 0) leaves
    // every warp slice at m = -INFINITY; expf(-inf - -inf) is NaN and poisons the
    // whole attention output (found by forcing more splits than keys in the M7
    // gate). An empty split contributes nothing, so its weights are 0.
    wts[threadIdx.x] = (cmax == -INFINITY)
                           ? 0.0f
                           : expf(smem[(std::int64_t)threadIdx.x * (2 + head_dim)] - cmax);
  }
  __syncthreads();
  float *out = partial + ((std::int64_t)h * n_splits + s) * (2 + head_dim);
  if (threadIdx.x < 32) {
    float lsum = 0.0f;
    if (threadIdx.x < WPB)
      lsum = wts[threadIdx.x] * smem[(std::int64_t)threadIdx.x * (2 + head_dim) + 1];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffull, lsum, off);
    if (threadIdx.x == 0) {
      out[0] = cmax;
      out[1] = lsum;
    }
  }
  // Strided over the whole block, not `threadIdx.x < head_dim`: with head_dim
  // > WPB*32 the old form wrote only the first WPB*32 dims of the partial and the
  // merge then read uninitialized VRAM (review finding M2). For head_dim == 256
  // and WPB == 8 (the shipped configuration) the loop runs exactly once per
  // thread, so this is the same work in the same order. It also removes the
  // unsigned/signed comparison that produced 98 -Wsign-compare warnings, which is
  // what kept -Wall -Wextra off the engine targets (finding B1).
  for (int d = (int)threadIdx.x; d < head_dim; d += (int)blockDim.x) {
    float a = 0.0f;
    for (int i = 0; i < WPB; ++i)
      a += smem[(std::int64_t)i * (2 + head_dim) + 2 + d] * wts[i];
    out[2 + d] = a;
  }
}

// Combine the n_splits partials of each query head (one warp per head).
__global__ void attn_merge_kernel(const float *__restrict__ partial, float *__restrict__ out,
                                  [[maybe_unused]] int n_head, int head_dim, int n_splits) {
  const int h = blockIdx.x;
  const int lane = threadIdx.x & 31;
  const int dpw = head_dim / 32;
  const float *base = partial + (std::int64_t)h * n_splits * (2 + head_dim);

  float mm = -INFINITY;
  for (int s = 0; s < n_splits; ++s) mm = fmaxf(mm, base[(std::int64_t)s * (2 + head_dim)]);
  if (mm == -INFINITY) {  // every split empty: no keys at all
    for (int i = 0; i < dpw; ++i) out[h * head_dim + lane * dpw + i] = 0.0f;
    return;
  }
  float ll = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    const float *ps = base + (std::int64_t)s * (2 + head_dim);
    ll += ps[1] * expf(ps[0] - mm);
  }
  const float inv = (ll > 0.0f) ? 1.0f / ll : 0.0f;
  for (int i = 0; i < dpw; ++i) {
    float a = 0.0f;
    for (int s = 0; s < n_splits; ++s) {
      const float *ps = base + (std::int64_t)s * (2 + head_dim);
      a += ps[2 + lane * dpw + i] * expf(ps[0] - mm);
    }
    out[h * head_dim + lane * dpw + i] = a * inv;
  }
}

template <KvType KT, KvType VT, int WPB = kAttnWarpsPerBlock>
inline bool attn_launch_split_typed(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                                    float *d_partial, int t, int n_head, int n_head_kv,
                                    int head_dim, float scale, int n_splits,
                                    hipStream_t stream) {
  if (head_dim % 32 != 0 || head_dim / 32 > kAttnMaxDimsPerLane) return false;
  if (n_splits < 1) return false;
  const int threads = WPB * 32;
  const std::size_t smem = (std::size_t)WPB * (2 + (std::size_t)head_dim) * sizeof(float);
  dim3 grid((unsigned)n_head, (unsigned)n_splits);
  attn_split_kernel<KT, VT, WPB><<<grid, threads, smem, stream>>>(d_q, d_k, d_v, d_partial, t,
                                                                 n_head, n_head_kv, head_dim, scale,
                                                                 n_splits);
  if (hipGetLastError() != hipSuccess) return false;
  attn_merge_kernel<<<n_head, 32, 0, stream>>>(d_partial, d_out, n_head, head_dim, n_splits);
  return hipGetLastError() == hipSuccess;
}

// WPB is a template knob (bench use); the shipping path fixes it at
// kAttnWarpsPerBlock. Only 8/16/32 are instantiated (32 = the 1024-thread cap).
template <KvType KT, KvType VT>
inline bool attn_launch_split_wpb(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                                  float *d_partial, int t, int n_head, int n_head_kv, int head_dim,
                                  float scale, int n_splits, int wpb, hipStream_t stream) {
  switch (wpb) {
    case 8:
      return attn_launch_split_typed<KT, VT, 8>(d_q, d_k, d_v, d_out, d_partial, t, n_head,
                                                n_head_kv, head_dim, scale, n_splits, stream);
    case 16:
      return attn_launch_split_typed<KT, VT, 16>(d_q, d_k, d_v, d_out, d_partial, t, n_head,
                                                 n_head_kv, head_dim, scale, n_splits, stream);
    case 32:
      return attn_launch_split_typed<KT, VT, 32>(d_q, d_k, d_v, d_out, d_partial, t, n_head,
                                                 n_head_kv, head_dim, scale, n_splits, stream);
    default: return false;
  }
}

inline bool attn_launch_split_wpb(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                                  float *d_partial, int t, int n_head, int n_head_kv, int head_dim,
                                  float scale, KvType kt, KvType vt, int n_splits, int wpb,
                                  hipStream_t stream = nullptr) {
#define RD_ATTN_WPB_CASE(K, V)                                                                    \
  if (kt == KvType::K && vt == KvType::V)                                                         \
  return attn_launch_split_wpb<KvType::K, KvType::V>(d_q, d_k, d_v, d_out, d_partial, t, n_head,  \
                                                     n_head_kv, head_dim, scale, n_splits, wpb,   \
                                                     stream)
  RD_ATTN_WPB_CASE(F32, F32);
  RD_ATTN_WPB_CASE(F16, F16);
  RD_ATTN_WPB_CASE(F16, Q8_0);
  RD_ATTN_WPB_CASE(F16, Q4_0);
  RD_ATTN_WPB_CASE(Q8_0, Q8_0);
  RD_ATTN_WPB_CASE(Q4_0, Q4_0);
  RD_ATTN_WPB_CASE(F32, F16);
  RD_ATTN_WPB_CASE(F32, Q8_0);
  RD_ATTN_WPB_CASE(F32, Q4_0);
  RD_ATTN_WPB_CASE(Q8_0, F16);
  RD_ATTN_WPB_CASE(Q8_0, Q4_0);
  RD_ATTN_WPB_CASE(Q4_0, F16);
  RD_ATTN_WPB_CASE(Q4_0, Q8_0);
  RD_ATTN_WPB_CASE(Q8_0, F32);
  RD_ATTN_WPB_CASE(Q4_0, F32);
  RD_ATTN_WPB_CASE(F16, F32);
  // same reason as attn_launch_wpb above: bench-only WPB knob, diagonal entries.
  RD_ATTN_WPB_CASE(Q5_0, Q5_0);
  RD_ATTN_WPB_CASE(Q4_1, Q4_1);
#undef RD_ATTN_WPB_CASE
  return false;
}

// Byte size of the partial buffer for `n_splits` (caller allocates once).
inline std::size_t attn_partial_bytes(int n_head, int head_dim, int n_splits) {
  return (std::size_t)n_head * (std::size_t)n_splits * (2 + (std::size_t)head_dim) * sizeof(float);
}

// ---------------------------------------------------------------------------
// Warps per CTA for the SPLIT kernel, by the number of key splits.
//
// Measured on gfx1201 (tests/bench_attn_gpu.hip, f16 KV, head_dim 256, 24 heads /
// 4 kv heads, one layer, one query token):
//
//   splits  WPB=8     WPB=16    WPB=32      grid CTAs
//   2       0.1777    0.1051    0.1057      48
//   4       0.1014    0.0719    0.0884      96
//   8       0.0744    0.0775    0.0969     192
//   16      0.0819    0.0844    0.1111     384
//
// The CTA count fixes the parallelism the GPU is willing to take from this
// kernel; what is left is how the KEY RANGE of each (head, split) pair is spread
// over its warps. At few splits per head, 8 warps leave each warp with a long
// serial key walk (latency exposed, measured 3.3x slower than the best config at
// 4096 keys), so more warps win; once there are 8+ splits per head the key walk
// is already short and the extra LDS traffic of a wider CTA (the merge reads
// WPB*(2+head_dim) floats per CTA through shared memory) starts to dominate.
//
// This is a SHAPE decision (like a block-size policy), not a change to the split
// count: the number of partials and the summation order across splits are
// untouched, and the short-context path (keys < kAttnSplitMin -> the unsplit
// kernel) never enters here, so the oracle/golden gates keep the arithmetic they
// were recorded with. The numeric effect of a WPB change is the same class as the
// M7 split change (rel-L2 ~2.5e-7 at the kernel level) and is gated by
// scripts/check_attn_split.sh + check-kvctx-gpu.
// The threshold is 4, not 8: at 8 splits per head the two independent kernel
// runs disagree (WPB=16 measured 1.10x better in one and 1.003x in the other)
// while at 2-4 splits every run shows 1.4-1.7x, and the end-to-end measurement
// at 16K keys (7 splits) showed -1.3% with the wider CTA. Above 4 splits the
// shipping 8 warps are therefore kept, which also means every context from 16K
// up is byte-for-byte the code path it was before this change.
//
// REVISADO (task 6, docs/autotuning-gfx1201.md): a varredura de 2026-09-13 com
// A/B intercalado mostrou que o topo da faixa e diferente do meio. Com KV f16
// (4K-131K) 8 warps continuam ganhando em 5..15 splits, mas a partir de 16
// splits -- o que so acontece em contexto longo, e sobretudo com KV q4_0, cuja
// linha e 3,5x menor -- a CTA larga volta a ganhar (131K q4_0: 16 splits x
// 16 warps = 1,050x em 12/15 rodadas; 24 splits x 8 warps = 0,954x a 64K f16).
// Por isso a regra tem DUAS pontas; o trecho 5..15 splits, que e o que 4K/16K
// usam, nao mudou.
constexpr int kAttnSplitWpbLimit = tuned::kAttnSplitWpbLimit;
// A partir de quantos splits a CTA volta a ser larga (ver tuning.h e
// docs/autotuning-gfx1201.md §Atencao: 131K com KV q4_0, 16 splits x 16 warps
// mediu 1,050x contra 8 warps, 12/15 rodadas de A/B intercalado).
constexpr int kAttnSplitWpbWide = tuned::kAttnSplitWpbWide;

inline int attn_split_wpb(int n_splits) {
  // Regra de duas pontas, medida (nao e monotonica): ver o comentario acima de
  // kAttnSplitWpbLimit/kAttnSplitWpbWide e a tabela em docs/autotuning-gfx1201.md.
  if (n_splits <= kAttnSplitWpbLimit) return 16;
  if (n_splits >= kAttnSplitWpbWide) return 16;
  return kAttnWarpsPerBlock;
}

inline bool attn_launch_split(const float *d_q, const void *d_k, const void *d_v, float *d_out,
                              float *d_partial, int t, int n_head, int n_head_kv, int head_dim,
                              float scale, KvType kt, KvType vt, int n_splits,
                              hipStream_t stream = nullptr) {
  // Shape dispatch: the WPB chosen above, through the templated launcher.
#define RD_ATTN_SPLIT_WPB(K, V)                                                                   \
  if (kt == KvType::K && vt == KvType::V) {                                                        \
    if (attn_split_wpb(n_splits) == 16)                                                            \
      return attn_launch_split_typed<KvType::K, KvType::V, 16>(d_q, d_k, d_v, d_out, d_partial, t,  \
                                                               n_head, n_head_kv, head_dim, scale,  \
                                                               n_splits, stream);                   \
    return attn_launch_split_typed<KvType::K, KvType::V, 8>(d_q, d_k, d_v, d_out, d_partial, t,     \
                                                            n_head, n_head_kv, head_dim, scale,     \
                                                            n_splits, stream);                      \
  }
  RD_ATTN_SPLIT_WPB(F32, F32);
  RD_ATTN_SPLIT_WPB(F32, F16);
  RD_ATTN_SPLIT_WPB(F32, Q8_0);
  RD_ATTN_SPLIT_WPB(F32, Q4_0);
  RD_ATTN_SPLIT_WPB(F32, Q5_0);
  RD_ATTN_SPLIT_WPB(F32, Q4_1);
  RD_ATTN_SPLIT_WPB(F16, F32);
  RD_ATTN_SPLIT_WPB(F16, F16);
  RD_ATTN_SPLIT_WPB(F16, Q8_0);
  RD_ATTN_SPLIT_WPB(F16, Q4_0);
  RD_ATTN_SPLIT_WPB(F16, Q5_0);
  RD_ATTN_SPLIT_WPB(F16, Q4_1);
  RD_ATTN_SPLIT_WPB(Q8_0, F32);
  RD_ATTN_SPLIT_WPB(Q8_0, F16);
  RD_ATTN_SPLIT_WPB(Q8_0, Q8_0);
  RD_ATTN_SPLIT_WPB(Q8_0, Q4_0);
  RD_ATTN_SPLIT_WPB(Q8_0, Q5_0);
  RD_ATTN_SPLIT_WPB(Q8_0, Q4_1);
  RD_ATTN_SPLIT_WPB(Q4_0, F32);
  RD_ATTN_SPLIT_WPB(Q4_0, F16);
  RD_ATTN_SPLIT_WPB(Q4_0, Q8_0);
  RD_ATTN_SPLIT_WPB(Q4_0, Q4_0);
  RD_ATTN_SPLIT_WPB(Q4_0, Q5_0);
  RD_ATTN_SPLIT_WPB(Q4_0, Q4_1);
  RD_ATTN_SPLIT_WPB(Q5_0, F32);
  RD_ATTN_SPLIT_WPB(Q5_0, F16);
  RD_ATTN_SPLIT_WPB(Q5_0, Q8_0);
  RD_ATTN_SPLIT_WPB(Q5_0, Q4_0);
  RD_ATTN_SPLIT_WPB(Q5_0, Q5_0);
  RD_ATTN_SPLIT_WPB(Q5_0, Q4_1);
  RD_ATTN_SPLIT_WPB(Q4_1, F32);
  RD_ATTN_SPLIT_WPB(Q4_1, F16);
  RD_ATTN_SPLIT_WPB(Q4_1, Q8_0);
  RD_ATTN_SPLIT_WPB(Q4_1, Q4_0);
  RD_ATTN_SPLIT_WPB(Q4_1, Q5_0);
  RD_ATTN_SPLIT_WPB(Q4_1, Q4_1);
#undef RD_ATTN_SPLIT_WPB
  return false;
}

}  // namespace rdna4
