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

}  // namespace rdna4
