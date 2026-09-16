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

// ---------------------------------------------------------------------------
// Variante com RPT linhas por thread (frente kernels de decode, noite
// 2026-09-14; diario docs/journal-kernels.md §3).
//
// O kernel de cima tem 48 CTAs de 128 threads numa placa de 64 CU (9 % de
// ocupacao) e cada thread faz DUAS cadeias seriais de 128 fma com ~600 ciclos
// de latencia de memoria por load. Medido: 69,6 us por lançamento isolado
// contra um piso de memoria de ~20 us (12,6 MB / 633 GB/s).
//
// Com T threads por CTA e RPT linhas por thread (T*RPT == S), cada thread
// carrega RPT cadeias INDEPENDENTES: a ordem de soma de cada linha e'
// exatamente a mesma (`i` crescente, dois passes, mesmos `fma`), o que torna a
// mudanca bit-exata -- nao ha' reordenamento nenhum, so' mais trabalho em voo.
// VEC usa cargas float4 na linha (mesmos valores, mesma ordem de fma).
//
// O caso de producao e' T=128/RPT=1 (o kernel de cima); os outros ficam
// instanciados para o bench e para a medicao que escolheu a configuracao.
// ---------------------------------------------------------------------------
template <int T, int RPT, bool VEC>
__global__ void delta_rule_kernel_ilp(const float *__restrict__ q, const float *__restrict__ k,
                                      const float *__restrict__ v, const float *__restrict__ gate,
                                      const float *__restrict__ beta, float *__restrict__ state,
                                      float *__restrict__ out, int n_k_heads, int S) {
  const int h = blockIdx.x;   // value head
  const int t = threadIdx.x;  // 0 .. T-1
  const int kh = h % n_k_heads;

  const float *kd = k + (std::int64_t)kh * S;
  const float *qd = q + (std::int64_t)kh * S;

  const float g = expf(gate[h]);
  const float b = beta[h];

  float *rowp[RPT];
  float sum[RPT], delta[RPT], acc[RPT];
#pragma unroll
  for (int r = 0; r < RPT; ++r) {
    const int j = t + r * T;  // as linhas de um thread ficam separadas por T
    rowp[r] = state + ((std::int64_t)h * S + j) * S;
    sum[r] = 0.0f;
    acc[r] = 0.0f;
  }

  if (VEC) {
    const float4 *kd4 = (const float4 *)kd;
#pragma unroll 4
    for (int i4 = 0; i4 < S / 4; ++i4) {
      const float4 kk = kd4[i4];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        float4 *p = (float4 *)rowp[r];
        float4 x = p[i4];
        x.x *= g; x.y *= g; x.z *= g; x.w *= g;
        p[i4] = x;
        sum[r] = fmaf(x.x, kk.x, sum[r]);
        sum[r] = fmaf(x.y, kk.y, sum[r]);
        sum[r] = fmaf(x.z, kk.z, sum[r]);
        sum[r] = fmaf(x.w, kk.w, sum[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < RPT; ++r) delta[r] = (v[(std::int64_t)h * S + (t + r * T)] - sum[r]) * b;
    const float4 *qd4 = (const float4 *)qd;
#pragma unroll 4
    for (int i4 = 0; i4 < S / 4; ++i4) {
      const float4 kk = kd4[i4];
      const float4 qq = qd4[i4];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        float4 *p = (float4 *)rowp[r];
        float4 x = p[i4];
        x.x = fmaf(kk.x, delta[r], x.x);
        x.y = fmaf(kk.y, delta[r], x.y);
        x.z = fmaf(kk.z, delta[r], x.z);
        x.w = fmaf(kk.w, delta[r], x.w);
        p[i4] = x;
        acc[r] = fmaf(x.x, qq.x, acc[r]);
        acc[r] = fmaf(x.y, qq.y, acc[r]);
        acc[r] = fmaf(x.z, qq.z, acc[r]);
        acc[r] = fmaf(x.w, qq.w, acc[r]);
      }
    }
  } else {
#pragma unroll 4
    for (int i = 0; i < S; ++i) {
      const float kk = kd[i];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        const float x = rowp[r][i] * g;
        rowp[r][i] = x;
        sum[r] = fmaf(x, kk, sum[r]);
      }
    }
#pragma unroll
    for (int r = 0; r < RPT; ++r) delta[r] = (v[(std::int64_t)h * S + (t + r * T)] - sum[r]) * b;
#pragma unroll 4
    for (int i = 0; i < S; ++i) {
      const float kk = kd[i];
      const float qq = qd[i];
#pragma unroll
      for (int r = 0; r < RPT; ++r) {
        const float x = fmaf(kk, delta[r], rowp[r][i]);
        rowp[r][i] = x;
        acc[r] = fmaf(x, qq, acc[r]);
      }
    }
  }

  const float rs = rsqrtf((float)S);
#pragma unroll
  for (int r = 0; r < RPT; ++r) out[(std::int64_t)h * S + (t + r * T)] = acc[r] * rs;
}

// Caminho de producao.
//
// MEDIDO (docs/journal-kernels.md §3, bench-delta-gpu, cadeia de 100 lancamentos,
// min de 12 rodadas x 3 passadas rotacionadas, piso de ruido do harness 1,0008x):
//
//   ship   T128 RPT1 escalar   71,70 us / camada   (175,7 GB/s de estado)
//   T128   RPT1 float4          8,42 us / camada  = 8,515x   (1496 GB/s)
//   T64    RPT2 float4         21,06 us
//   T32    RPT4 float4         36,21 us
//   piso do padrao (le+escreve, sem cadeia) 5,72 us
//
// O que muda e' SO' a largura da carga da linha do estado: escalar (128 cargas
// de 4 B por passe, cada lane num setor de 32 B proprio) -> float4 (32 cargas de
// 16 B). As operacoes, a ordem e os acumuladores sao os mesmos, e o bench
// confere estado e saida com memcmp (bit-identico). Manter RPT=1/T=128 e' o que
// MEDE melhor: com RPT>1 ha' menos warps no ar (192 -> 96 -> 48) e o float4 ja'
// da' o paralelismo de memoria que faltava.
//
// S != 128 (outro modelo) cai no kernel historico, sem mudanca de semantica.
inline bool delta_rule_launch(const float *d_q, const float *d_k, const float *d_v,
                              const float *d_gate, const float *d_beta, float *d_state,
                              float *d_out, int n_v_heads, int n_k_heads, int S,
                              hipStream_t stream = nullptr) {
  if (S == 128 && n_v_heads > 0) {
    delta_rule_kernel_ilp<128, 1, true><<<n_v_heads, 128, 0, stream>>>(
        d_q, d_k, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S);
    return hipGetLastError() == hipSuccess;
  }
  delta_rule_kernel<<<n_v_heads, ((S + 31) / 32) * 32, 0, stream>>>(
      d_q, d_k, d_v, d_gate, d_beta, d_state, d_out, n_v_heads, n_k_heads, S);
  return hipGetLastError() == hipSuccess;
}

// O kernel historico fica instanciado e acessivel para o bench (e para quem
// quiser re-medir a comparacao que escolheu o float4).
inline bool delta_rule_launch_scalar(const float *d_q, const float *d_k, const float *d_v,
                                     const float *d_gate, const float *d_beta, float *d_state,
                                     float *d_out, int n_v_heads, int n_k_heads, int S,
                                     hipStream_t stream = nullptr) {
  delta_rule_kernel<<<n_v_heads, ((S + 31) / 32) * 32, 0, stream>>>(
      d_q, d_k, d_v, d_gate, d_beta, d_state, d_out, n_v_heads, n_k_heads, S);
  return hipGetLastError() == hipSuccess;
}

// Mesmo lancamento, forcando uma configuracao de (T, RPT, VEC) -- usado pelo
// bench e para medir antes de trocar o caminho de producao.
inline bool delta_rule_launch_ilp(int t, int rpt, bool vec, const float *d_q, const float *d_k,
                                  const float *d_v, const float *d_gate, const float *d_beta,
                                  float *d_state, float *d_out, int n_v_heads, int n_k_heads, int S,
                                  hipStream_t stream = nullptr) {
  if (S != 128) return false;  // as instanciacoes medidas sao para S=128
#define RD_DELTA(T_, RPT_, VEC_)                                                          \
  if (t == T_ && rpt == RPT_ && vec == VEC_) {                                            \
    delta_rule_kernel_ilp<T_, RPT_, VEC_><<<n_v_heads, T_, 0, stream>>>(                   \
        d_q, d_k, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S);                      \
    return hipGetLastError() == hipSuccess;                                                \
  }
  RD_DELTA(128, 1, true)
  RD_DELTA(64, 2, true)
  RD_DELTA(32, 4, true)
  RD_DELTA(64, 2, false)
  RD_DELTA(32, 4, false)
#undef RD_DELTA
  return false;
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
//
// VEC=true walks the row with float4 loads/stores: thread j owns row j, so a
// warp's 32 lanes touch addresses 512 B apart and each 4-byte access pulls a
// whole 32 B sector (8x amplification, and the row is walked twice per token).
// Four elements per access cuts that to 2x and cuts the instruction count of the
// serial i-loop by 4 without touching the arithmetic: the four FMAs of one vector
// are still issued in increasing i order, so the accumulation rounds exactly as
// the scalar loop did.
template <int ROWS, bool VEC>
__global__ void delta_rule_batch_rows_kernel(const float *__restrict__ qk,
                                             const float *__restrict__ v,
                                             const float *__restrict__ gate,
                                             const float *__restrict__ beta,
                                             float *__restrict__ state, float *__restrict__ out,
                                             int n_k_heads, int S, int n_vh, int key_dim,
                                             int d_inner, int n) {
  const int h = blockIdx.x;                       // value head
  const int j = blockIdx.y * ROWS + threadIdx.x;  // row index (value dim)
  if (j >= S) return;
  const int kh = h % n_k_heads;  // repeated key head
  const float *kb = qk + key_dim + (std::int64_t)kh * S;
  const float *qb = qk + (std::int64_t)kh * S;
  float *row = state + ((std::int64_t)h * S + j) * S;
  const int i4n = VEC ? (S & ~3) : 0;

  for (int t = 0; t < n; ++t) {
    const float *kd = kb + (std::int64_t)t * 2 * key_dim;
    const float *qd = qb + (std::int64_t)t * 2 * key_dim;
    const float g = expf(gate[(std::int64_t)t * n_vh + h]);
    const float b = beta[(std::int64_t)t * n_vh + h];
    float sum = 0.0f;
    for (int i = 0; i < i4n; i += 4) {
      float4 r = *reinterpret_cast<const float4 *>(row + i);
      const float4 kk = *reinterpret_cast<const float4 *>(kd + i);
      r.x *= g;
      r.y *= g;
      r.z *= g;
      r.w *= g;
      *reinterpret_cast<float4 *>(row + i) = r;
      sum = fmaf(r.x, kk.x, sum);
      sum = fmaf(r.y, kk.y, sum);
      sum = fmaf(r.z, kk.z, sum);
      sum = fmaf(r.w, kk.w, sum);
    }
    for (int i = i4n; i < S; ++i) {
      row[i] *= g;
      sum = fmaf(row[i], kd[i], sum);
    }
    const float delta = (v[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] - sum) * b;
    float acc = 0.0f;
    for (int i = 0; i < i4n; i += 4) {
      const float4 kk = *reinterpret_cast<const float4 *>(kd + i);
      const float4 qq = *reinterpret_cast<const float4 *>(qd + i);
      float4 r = *reinterpret_cast<const float4 *>(row + i);
      r.x = fmaf(kk.x, delta, r.x);
      r.y = fmaf(kk.y, delta, r.y);
      r.z = fmaf(kk.z, delta, r.z);
      r.w = fmaf(kk.w, delta, r.w);
      *reinterpret_cast<float4 *>(row + i) = r;
      acc = fmaf(r.x, qq.x, acc);
      acc = fmaf(r.y, qq.y, acc);
      acc = fmaf(r.z, qq.z, acc);
      acc = fmaf(r.w, qq.w, acc);
    }
    for (int i = i4n; i < S; ++i) {
      row[i] = fmaf(kd[i], delta, row[i]);
      acc = fmaf(row[i], qd[i], acc);
    }
    out[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] = acc * rsqrtf((float)S);
  }
}

// rows per CTA: 32 rows = 16 KiB of state per CTA, which is what keeps the row
// working set inside L1/L2 for the whole chunk.
inline bool delta_rule_batch_launch(const float *d_qk, const float *d_v, const float *d_gate,
                                    const float *d_beta, float *d_state, float *d_out,
                                    int n_v_heads, int n_k_heads, int S, int n_vh, int key_dim,
                                    int d_inner, int n, hipStream_t stream = nullptr) {
  constexpr int kRows = 32;
  const unsigned ny = (unsigned)((S + kRows - 1) / kRows);
  const dim3 grid((unsigned)n_v_heads, ny);
  if ((S & 3) == 0) {
    delta_rule_batch_rows_kernel<kRows, true><<<grid, kRows, 0, stream>>>(
        d_qk, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S, n_vh, key_dim, d_inner, n);
  } else {
    delta_rule_batch_rows_kernel<kRows, false><<<grid, kRows, 0, stream>>>(
        d_qk, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S, n_vh, key_dim, d_inner, n);
  }
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Blocked multi-token delta_rule for PREFILL (frente gdn-resident): the same
// scan as delta_rule_batch_rows_kernel above -- grid (n_v_heads, S/ROWS), the
// chunk's n tokens walked IN ORDER inside one launch -- but the CTA's state
// tile (ROWS x S floats) is staged once into LDS and kept resident across the
// n tokens, instead of round-tripping through DRAM twice per token.
//
// Bit-exactness (why the gate stays meaningful): thread j still owns row j
// for the whole launch, and per (token, row) the loop nest is the scalar path
// of the kernel above line for line -- scale by expf(gate), dot with k
// ascending-i with a single accumulator, delta = (v - sum) * beta, rank-1
// update, dot with q ascending-i, times rsqrt(S). Only the TIER holding the
// row changes (LDS instead of global); loads and stores do not round, and no
// two threads ever touch the same element, so the single barrier below cannot
// change any value either. check-batch-gpu (n <= 16 BIT-EXACT) proves it on
// every run.
//
// Layout note: the tile is TRANSPOSED in LDS (element i of thread tx at
// tile[i*ROWS + tx]). Row-major (tile[tx*S + i]) would land all 32 threads of
// the wave on the same LDS bank (S = 128 dwords is 0 mod 32 banks) -- a
// 32-way conflict on every access; transposed, simultaneous threads hit bank
// tx, conflict-free by construction. LDS traffic is scalar on purpose (those
// loads stay hidden under the serial fma chains); only the global tile move
// is float4. The static tile is sized for S <= 128 (production S == 128);
// anything else falls back to delta_rule_batch_launch in the launcher below.
//
// What it buys: per token the old kernel moved the whole head state twice
// through global memory (2 passes x read+write = 4 x 64 KiB per head); this
// one moves it exactly twice per CHUNK (tile in on entry, tile out on exit).
// k/q/v/gate/beta/out still stream per token, but those are ~1% of the old
// traffic (3 x S + 2 floats read per thread vs 4 x S floats of state per pass
// moved twice).
// ---------------------------------------------------------------------------
template <int ROWS, bool VEC>
__global__ void delta_rule_batch_resident_kernel(const float *__restrict__ qk,
                                                 const float *__restrict__ v,
                                                 const float *__restrict__ gate,
                                                 const float *__restrict__ beta,
                                                 float *__restrict__ state, float *__restrict__ out,
                                                 int n_k_heads, int S, int n_vh, int key_dim,
                                                 int d_inner, int n) {
  const int h = blockIdx.x;              // value head
  const int tx = threadIdx.x;
  const int j = blockIdx.y * ROWS + tx;  // row index (value dim)
  const bool active = j < S;
  const int kh = h % n_k_heads;  // repeated key head

  // Transposed tile: TROW(i) is element i of this thread's row.
  __shared__ __align__(16) float tile[128 * ROWS];
#define RD_GDN_TROW(i) (tile[(i)*ROWS + tx])

  // Stage the tile once. Thread j owns row j and no row is shared, so the one
  // barrier below is documentation-grade (one per launch, not per token).
  if (active) {
    const float *srow = state + ((std::int64_t)h * S + j) * S;
    if (VEC) {
      const float4 *sp = (const float4 *)srow;
      for (int i4 = 0; i4 < S / 4; ++i4) {
        const float4 g = sp[i4];
        RD_GDN_TROW(4 * i4 + 0) = g.x;
        RD_GDN_TROW(4 * i4 + 1) = g.y;
        RD_GDN_TROW(4 * i4 + 2) = g.z;
        RD_GDN_TROW(4 * i4 + 3) = g.w;
      }
      for (int i = (S & ~3); i < S; ++i) RD_GDN_TROW(i) = srow[i];
    } else {
      for (int i = 0; i < S; ++i) RD_GDN_TROW(i) = srow[i];
    }
  }
  __syncthreads();

  if (active) {
    const float *kb = qk + key_dim + (std::int64_t)kh * S;
    const float *qb = qk + (std::int64_t)kh * S;

    for (int t = 0; t < n; ++t) {
      const float *kd = kb + (std::int64_t)t * 2 * key_dim;
      const float *qd = qb + (std::int64_t)t * 2 * key_dim;
      const float g = expf(gate[(std::int64_t)t * n_vh + h]);
      const float b = beta[(std::int64_t)t * n_vh + h];
      float sum = 0.0f;
      for (int i = 0; i < S; ++i) {
        const float x = RD_GDN_TROW(i) * g;
        RD_GDN_TROW(i) = x;
        sum = fmaf(x, kd[i], sum);
      }
      const float delta = (v[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] - sum) * b;
      float acc = 0.0f;
      for (int i = 0; i < S; ++i) {
        const float x = fmaf(kd[i], delta, RD_GDN_TROW(i));
        RD_GDN_TROW(i) = x;
        acc = fmaf(x, qd[i], acc);
      }
      out[(std::int64_t)t * d_inner + (std::int64_t)h * S + j] = acc * rsqrtf((float)S);
    }

    // Write the tile back once: the recurrent state the next chunk (or the MTP
    // snapshot) reads. Same layout, same dtype, same buffer as before.
    float *srow = state + ((std::int64_t)h * S + j) * S;
    if (VEC) {
      float4 *dp = (float4 *)srow;
      for (int i4 = 0; i4 < S / 4; ++i4) {
        const float4 g = make_float4(RD_GDN_TROW(4 * i4 + 0), RD_GDN_TROW(4 * i4 + 1),
                                     RD_GDN_TROW(4 * i4 + 2), RD_GDN_TROW(4 * i4 + 3));
        dp[i4] = g;
      }
      for (int i = (S & ~3); i < S; ++i) srow[i] = RD_GDN_TROW(i);
    } else {
      for (int i = 0; i < S; ++i) srow[i] = RD_GDN_TROW(i);
    }
  }
#undef RD_GDN_TROW
}

// rows per CTA: same 32 as delta_rule_batch_launch (same grid, so the only
// variable the measurement moves is the storage tier of the tile; 16 was
// measured identical, 314.6 vs 315.1 tok/s -- the scan sits on its serial
// floor, not on waves).
inline bool delta_rule_batch_resident_launch(const float *d_qk, const float *d_v, const float *d_gate,
                                             const float *d_beta, float *d_state, float *d_out,
                                             int n_v_heads, int n_k_heads, int S, int n_vh, int key_dim,
                                             int d_inner, int n, hipStream_t stream = nullptr) {
  constexpr int kRows = 32;
  if (S <= 0 || S > 128 || n_v_heads <= 0) {
    return delta_rule_batch_launch(d_qk, d_v, d_gate, d_beta, d_state, d_out, n_v_heads, n_k_heads,
                                   S, n_vh, key_dim, d_inner, n, stream);
  }
  const unsigned ny = (unsigned)((S + kRows - 1) / kRows);
  const dim3 grid((unsigned)n_v_heads, ny);
  if ((S & 3) == 0) {
    delta_rule_batch_resident_kernel<kRows, true><<<grid, kRows, 0, stream>>>(
        d_qk, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S, n_vh, key_dim, d_inner, n);
  } else {
    delta_rule_batch_resident_kernel<kRows, false><<<grid, kRows, 0, stream>>>(
        d_qk, d_v, d_gate, d_beta, d_state, d_out, n_k_heads, S, n_vh, key_dim, d_inner, n);
  }
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

// ---------------------------------------------------------------------------
// Fused GDN scalars for the DECODE path: sigmoid(beta) + (alpha + ssm_dt) +
// softplus + gate*ssm_a in ONE launch (4 launches -> 1, erases 3 + barriers).
//
// Bit-exactness: the same four ops in the same order with the same single
// rounding each -- intermediates live in registers. Buffer states are kept
// dump-compatible on purpose: d_beta_ still ends up holding sigmoid(beta) and
// d_gate_ softplus(a)*ssm_a, so the beta_sigmoid/gate node dumps read the same
// values. The one exception is d_alpha_: it receives the softplus value
// instead of alpha+ssm_dt, which is what lets the a_softplus dump keep its
// source (its emit is re-pointed at d_alpha_). d_alpha_ is dead after this
// point on the decode path -- the delta rule reads only gate/beta and the next
// token re-projects alpha fresh -- so no later reader can observe it.
// ---------------------------------------------------------------------------
__global__ void gdn_scalars_kernel(float *__restrict__ beta_io, float *__restrict__ alpha_sp,
                                   float *__restrict__ gate_out, const float *__restrict__ dt,
                                   const float *__restrict__ A, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float b = sigmoid_f(beta_io[i]);
  beta_io[i] = b;
  const float s = softplus_f(alpha_sp[i] + dt[i]);
  alpha_sp[i] = s;
  gate_out[i] = s * A[i];
}

inline bool gdn_scalars_launch(float *d_beta_io, float *d_alpha_sp, float *d_gate_out,
                               const float *d_dt, const float *d_A, int n,
                               hipStream_t stream = nullptr) {
  const int threads = 256;
  gdn_scalars_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
      d_beta_io, d_alpha_sp, d_gate_out, d_dt, d_A, n);
  return hipGetLastError() == hipSuccess;
}

}  // namespace rdna4
