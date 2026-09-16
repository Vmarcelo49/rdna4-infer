// Bench isolado da `delta_rule` do GDN (frente kernels de decode, noite 2026-09-14).
//
// Por que existe: `docs/medicoes-banda-e-gargalos.md` §4.2 mede a `delta_rule` em
// 69,6 us por lançamento (formas reais: nvh 48, nkh 16, S 128) contra um piso de
// memória de ~20 us, com 48 CTAs de 128 threads numa placa de 64 CU (9 % de
// ocupacao) e 4 passadas sobre 3,15 MB de estado. O `bench-phases-gpu --gdn` mede
// o kernel QUE EXISTE; este mede VARIANTES do mesmo kernel, na mesma cadeia de
// 100 lançamentos, com A/B intercalado e piso de ruído do proprio harness.
//
// Protocolo (docs/noite-regras.md §3): aquecimento até >=300 ms de GPU ocupada,
// cadeia de CH lançamentos por par de eventos, `--rounds` rodadas, MINIMO por
// variante (a placa cai de relogio entre kernels e o minimo e' a medida menos
// contaminada), e a variante 0 medida DUAS vezes como piso de ruido do harness
// (`ship` vs `ship2`, mesma config, celulas independentes).
//
// Bit-exatidao: toda variante tem que produzir o MESMO estado e a MESMA saida,
// bit a bit, que a que embarca. As variantes aqui só mexem em (a) quantas linhas
// cada thread processa (cadeias independentes: nao muda a ordem de soma de
// nenhuma linha), (b) cargas vetorizadas float4 (mesmos valores, mesma ordem de
// fma) e (c) a ordem de emissao das instrucoes (o compilador ja faz isso). O
// teste compara com memcmp nos floats.
//
// usage: bench-delta-gpu [--rounds N] [--chain N] [--no-noise]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <functional>
#include <vector>

#include "rdna4/gdn.cuh"

namespace {

constexpr int NVH = 48;   // value heads (time_step_rank)
constexpr int NKH = 16;   // key heads (group_count)
constexpr int S = 128;    // state dimension
constexpr int CH = 100;   // launches per event pair

__global__ void fill_kernel(float *__restrict__ x, long long n, unsigned seed) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned s = seed * 1664525u + 1013904223u + (unsigned)i * 2654435761u;
  s ^= s >> 13;
  s *= 1274126177u;
  x[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
}

double now_ms() {
  using namespace std::chrono;
  return (double)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// Piso de memoria do MESMO padrao de acesso: le a linha inteira (float4) e
// escreve de volta, sem cadeia serial (4 acumuladores independentes). Nao e'
// uma variante correta, e' so o teto do padrao: 3,15 MB lidos + 3,15 MB
// escritos + kd/qd por cabeca.
__global__ void pattern_floor_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                     float *__restrict__ state, float *__restrict__ out) {
  const int h = blockIdx.x;
  const int t = threadIdx.x;
  float *rowp = state + ((long long)h * S + t) * S;
  const float4 *kd4 = (const float4 *)(k + (long long)(h % NKH) * S);
  const float4 *qd4 = (const float4 *)(q + (long long)(h % NKH) * S);
  float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  float4 *p = (float4 *)rowp;
#pragma unroll 4
  for (int i4 = 0; i4 < S / 4; ++i4) {
    float4 x = p[i4];
    const float4 kk = kd4[i4];
    const float4 qq = qd4[i4];
    x.x += kk.x + qq.x; x.y += kk.y + qq.y; x.z += kk.z + qq.z; x.w += kk.w + qq.w;
    p[i4] = x;
    a0 += x.x; a1 += x.y; a2 += x.z; a3 += x.w;
  }
  out[(long long)h * S + t] = a0 + a1 + a2 + a3;
}

struct Variant {
  const char *name;
  std::function<void()> launch;
  int T;
};

// ---------------------------------------------------------------------------
// GDN chunked-port prototype (docs/estudo-ninfer-gdn.md §2, autorizado em
// docs/plano-ninfer-pendente.md §2). CPU-only, synthetic data, single head:
// the within-chunk parallel form in fp32 AND fp64 vs the production GPU path.
// Conventions under test (each a documented trap, estudo §3.5): inclusive
// cumsum, include-diagonal output mask, row-wise beta. `--chunked` runs it.
// ---------------------------------------------------------------------------
namespace gdn_chunked {

constexpr int CS = 128;  // state dim (production S)
constexpr int CB = 16;   // chunk size (proposal: B=16, estudo §6.2)
constexpr int CT = 32;   // tokens (2 chunks)

static unsigned long long rng_s = 0x12345678u;
static double randu() {
  rng_s = rng_s * 6364136223846793005ull + 1442695040888963407ull;
  return (double)(rng_s >> 11) * (1.0 / 9007199254740992.0);
}

template <typename F>
static F xexp(F x) {
  return (sizeof(F) > 4) ? (F)std::exp((double)x) : (F)expf((float)x);
}

// Sequential reference, transcribed from delta_rule_kernel (gdn.cuh:72-87):
// per-token expf gate, ascending-i single-accumulator fma dots, post-update out.
template <typename F>
static void sequential(const std::vector<F> &q, const std::vector<F> &k, const std::vector<F> &v,
                       const std::vector<F> &gate, const std::vector<F> &beta, std::vector<F> &st,
                       std::vector<F> &out, int T) {
  const F rs = (F)(1.0 / std::sqrt((double)CS));
  for (int t = 0; t < T; ++t) {
    const F g = xexp(gate[t]), b = beta[t];
    const F *kd = &k[(std::size_t)t * CS], *qd = &q[(std::size_t)t * CS];
    for (int j = 0; j < CS; ++j) {
      F *row = &st[(std::size_t)j * CS];
      F sum = (F)0;
      for (int i = 0; i < CS; ++i) {
        const F x = row[i] * g;
        row[i] = x;
        sum = std::fma(x, kd[i], sum);
      }
      const F delta = (v[(std::size_t)t * CS + j] - sum) * b;
      F acc = (F)0;
      for (int i = 0; i < CS; ++i) {
        const F x = std::fma(kd[i], delta, row[i]);
        row[i] = x;
        acc = std::fma(x, qd[i], acc);
      }
      out[(std::size_t)t * CS + j] = acc * rs;
    }
  }
}

struct Traps {
  bool exclusive_cumsum = false;  // wrong: 0.275 (estudo §3.5)
  bool strict_diag = false;       // wrong: 0.824
  bool col_beta = false;          // wrong: 1.1e-2..2.6e-2
};

// Within-chunk parallel form, transcribed from estudo §2.3-2.5. H is kept in
// S[i][j] layout (i=key, j=value), i.e. H[i*CS+j] == st[j*CS+i] of sequential.
template <typename F>
static void chunked(const std::vector<F> &q, const std::vector<F> &k, const std::vector<F> &v,
                    const std::vector<F> &gate, const std::vector<F> &beta, std::vector<F> &st,
                    std::vector<F> &out, int T, Traps tr = Traps()) {
  const F rs = (F)(1.0 / std::sqrt((double)CS));
  std::vector<F> G(CB), A(CB * CB), Tm(CB * CB), U(CB * CS), W(CB * CS), vn(CB * CS),
      Ao(CB * CB), Hin(CS * CS);
  for (int t0 = 0; t0 < T; t0 += CB) {
    Hin = st;  // output term reads H at block ENTRY (output.cuh:233-234)
    // G: INCLUSIVE local cumsum (prepare_wy_wu.cuh:417-443)
    for (int r = 0; r < CB; ++r) {
      const F g = gate[t0 + r];
      if (tr.exclusive_cumsum)
        G[r] = (r ? G[r - 1] + gate[t0 + r - 1] : (F)0);  // WRONG variant: 0.275
      else
        G[r] = (r ? G[r - 1] : (F)0) + g;
    }
    // A = -diag(beta_row) * strict-tril(KK^T o D); select-mask, never *mask
    // (upper triangle of exp(Gr-Gc) overflows: inf*0=NaN, estudo §2.3a)
    for (int r = 0; r < CB; ++r) {
      for (int c = 0; c < CB; ++c) {
        F e = (F)0;
        if (c < r) {
          F dot = (F)0;
          const F *kr = &k[(std::size_t)(t0 + r) * CS], *kc = &k[(std::size_t)(t0 + c) * CS];
          for (int i = 0; i < CS; ++i) dot = std::fma(kr[i], kc[i], dot);
          e = dot * xexp(G[r] - G[c]);
        }
        const F b = tr.col_beta ? beta[t0 + c] : beta[t0 + r];  // ROW is correct
        A[r * CB + c] = -b * e;
      }
    }
    // Tm = (I-A)^{-1}, unit-diagonal forward substitution per column
    for (int r = 0; r < CB; ++r)
      for (int c = 0; c < CB; ++c) Tm[r * CB + c] = (r == c) ? (F)1 : (F)0;
    for (int c = 0; c < CB; ++c) {  // solve (I-A) x = e_c
      for (int r = 0; r < CB; ++r) {
        F acc = (r == c) ? (F)1 : (F)0;
        for (int s = 0; s < r; ++s) acc += A[r * CB + s] * Tm[s * CB + c];
        Tm[r * CB + c] = acc;  // (I-A) has unit diagonal: no division
      }
    }
    // U = Tm (beta V); W = Tm (beta exp(G) K)
    for (int r = 0; r < CB; ++r)
      for (int i = 0; i < CS; ++i) {
        F su = (F)0, sw = (F)0;
        for (int c = 0; c < CB; ++c) {
          const F t = Tm[r * CB + c] * beta[t0 + c];
          su += t * v[(std::size_t)(t0 + c) * CS + i];
          sw += t * (xexp(G[c]) * k[(std::size_t)(t0 + c) * CS + i]);
        }
        U[(std::size_t)r * CS + i] = su;
        W[(std::size_t)r * CS + i] = sw;
      }
    // vnew = U - W H_in; H_out = e^{G[B-1]} H_in + K^T vd (state_passing.cuh)
    const F glast = xexp(G[CB - 1]);
    for (int r = 0; r < CB; ++r) {
      for (int j = 0; j < CS; ++j) {
        F s = (F)0;
        for (int i = 0; i < CS; ++i) s += W[(std::size_t)r * CS + i] * Hin[(std::size_t)j * CS + i];
        // note: st[j*CS+i] == H[i][j]
        vn[(std::size_t)r * CS + j] = U[(std::size_t)r * CS + j] - s;  // vnew, unscaled
      }
    }
    for (int i = 0; i < CS; ++i)
      for (int j = 0; j < CS; ++j) {
        F s = (F)0;
        for (int r = 0; r < CB; ++r)
          s += k[(std::size_t)(t0 + r) * CS + i] *
               (vn[(std::size_t)r * CS + j] * xexp(G[CB - 1] - G[r]));  // vd
        st[(std::size_t)j * CS + i] = Hin[(std::size_t)j * CS + i] * glast + s;
      }
    // out (output.cuh:10-11): scale*(e^{G_r} q H_in^T + A vnew), diagonal INCLUDED
    for (int r = 0; r < CB; ++r) {
      const F gr = xexp(G[r]);
      const F *qr = &q[(std::size_t)(t0 + r) * CS];
      for (int s = 0; s < CB; ++s) {
        F d = (F)0;
        if (tr.strict_diag ? (s < r) : (s <= r)) {
          const F *ks = &k[(std::size_t)(t0 + s) * CS];
          for (int i = 0; i < CS; ++i) d = std::fma(qr[i], ks[i], d);
          d *= xexp(G[r] - G[s]);
        }
        Ao[r * CB + s] = d;
      }
      for (int j = 0; j < CS; ++j) {
        F qh = (F)0;
        for (int i = 0; i < CS; ++i) qh = std::fma(qr[i], Hin[(std::size_t)j * CS + i], qh);
        F intra = (F)0;
        for (int s = 0; s < CB; ++s) intra += Ao[r * CB + s] * vn[(std::size_t)s * CS + j];
        out[(std::size_t)(t0 + r) * CS + j] = (gr * qh + intra) * rs;
      }
    }
  }
}

template <typename F>
static double rel_l2_vec(const std::vector<F> &a, const std::vector<F> &b) {
  double num = 0, den = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)a[i] * (double)a[i];
  }
  return std::sqrt(num / den);
}

// `--chunked`: exactness of the within-chunk parallel form on synthetic data
// + per-token cost of the production prefill (batch) path. Returns 0.
constexpr int PB = 16;    // chunk size of the GPU prototype below
constexpr int PROWS = 32;  // state rows per CTA of the GPU prototype below
__global__ void gdn_chunked_proto_kernel(const float *qk, const float *v, const float *gate,
                                         const float *beta, float *state, float *out,
                                         int n_k_heads, int n_vh, int key_dim, int d_inner, int n);
static int run_chunked_check() {
  constexpr int T = CT;
  rng_s = 0xC0FFEEu;
  std::vector<double> qd(T * CS), kd(T * CS), vd(T * CS), gd(T), bd(T), st0(CS * CS);
  for (int t = 0; t < T; ++t) {
    double nq = 0, nk = 0;
    for (int i = 0; i < CS; ++i) {
      qd[(std::size_t)t * CS + i] = randu() - 0.5;
      kd[(std::size_t)t * CS + i] = randu() - 0.5;
      nq += qd[(std::size_t)t * CS + i] * qd[(std::size_t)t * CS + i];
      nk += kd[(std::size_t)t * CS + i] * kd[(std::size_t)t * CS + i];
      vd[(std::size_t)t * CS + i] = (randu() - 0.5) * 2.0;
    }
    nq = std::sqrt(nq);
    nk = std::sqrt(nk);  // L2-normalized q/k, as l2_norm delivers
    for (int i = 0; i < CS; ++i) {
      qd[(std::size_t)t * CS + i] /= nq;
      kd[(std::size_t)t * CS + i] /= nk;
    }
    gd[t] = -0.5 * randu() - 0.01;                        // log-gate <= 0
    bd[t] = 1.0 / (1.0 + std::exp(-(randu() - 0.5) * 4));  // beta in (0,1)
  }
  for (auto &x : st0) x = randu() - 0.5;  // non-zero initial state

  auto as_float = [](const std::vector<double> &d) {
    std::vector<float> f(d.size());
    for (std::size_t i = 0; i < d.size(); ++i) f[i] = (float)d[i];
    return f;
  };
  const std::vector<float> qf = as_float(qd), kf = as_float(kd), vf = as_float(vd),
                           gf = as_float(gd), bf = as_float(bd), sf0 = as_float(st0);

  // CPU references
  std::vector<double> st_seq = st0, out_seq(T * CS);
  sequential(qd, kd, vd, gd, bd, st_seq, out_seq, T);
  std::vector<double> st_ch = st0, out_ch(T * CS);
  chunked(qd, kd, vd, gd, bd, st_ch, out_ch, T);
  std::vector<float> st_seq32 = sf0, out_seq32(T * CS);
  sequential(qf, kf, vf, gf, bf, st_seq32, out_seq32, T);
  std::vector<float> st_ch32 = sf0, out_ch32(T * CS);
  chunked(qf, kf, vf, gf, bf, st_ch32, out_ch32, T);

  std::printf("chunked prototype (1 head, S=128, B=16, T=32, synthetic, nonzero state):\n");
  std::printf("  fp64 chunked vs fp64 sequential : out %.2e  state %.2e  (spec: 1e-15 exact)\n",
              rel_l2_vec(out_ch, out_seq), rel_l2_vec(st_ch, st_seq));
  std::printf("  fp32 chunked vs fp32 sequential : out %.2e  state %.2e  (spec: 5e-7..1e-6)\n",
              rel_l2_vec(out_ch32, out_seq32), rel_l2_vec(st_ch32, st_seq32));
  // the three convention traps, each must blow up (estudo §3.5)
  for (int tr = 0; tr < 3; ++tr) {
    Traps tp;
    if (tr == 0) tp.exclusive_cumsum = true;
    if (tr == 1) tp.strict_diag = true;
    if (tr == 2) tp.col_beta = true;
    std::vector<double> stt = st0, outt(T * CS);
    chunked(qd, kd, vd, gd, bd, stt, outt, T, tp);
    std::printf("  trap %-16s vs sequential  : out %.2e  (must be O(1e-2..1))\n",
                tr == 0 ? "excl-cumsum" : tr == 1 ? "strict-diag" : "col-beta",
                rel_l2_vec(outt, out_seq));
  }

  // Production GPU path (prefill): batch-resident, nvh=1 shape, same inputs
  float *dqk = nullptr, *dvv = nullptr, *dgg = nullptr, *dbb = nullptr, *dst = nullptr,
        *dou = nullptr;
  const std::size_t qkb = (std::size_t)T * 2 * CS * 4, vb = (std::size_t)T * CS * 4,
                    gb = (std::size_t)T * 4, sb = (std::size_t)CS * CS * 4;
  if (hipMalloc(&dqk, qkb) != hipSuccess || hipMalloc(&dvv, vb) != hipSuccess ||
      hipMalloc(&dgg, gb) != hipSuccess || hipMalloc(&dbb, gb) != hipSuccess ||
      hipMalloc(&dst, sb) != hipSuccess || hipMalloc(&dou, vb) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  std::vector<float> qk(T * 2 * CS);
  for (int t = 0; t < T; ++t)
    for (int i = 0; i < CS; ++i) {
      qk[(std::size_t)t * 2 * CS + i] = qf[(std::size_t)t * CS + i];
      qk[(std::size_t)t * 2 * CS + CS + i] = kf[(std::size_t)t * CS + i];
    }
  hipMemcpy(dqk, qk.data(), qkb, hipMemcpyHostToDevice);
  hipMemcpy(dvv, vf.data(), vb, hipMemcpyHostToDevice);
  hipMemcpy(dgg, gf.data(), gb, hipMemcpyHostToDevice);
  hipMemcpy(dbb, bf.data(), gb, hipMemcpyHostToDevice);
  hipMemcpy(dst, sf0.data(), sb, hipMemcpyHostToDevice);
  (void)rdna4::delta_rule_batch_resident_launch(dqk, dvv, dgg, dbb, dst, dou, 1, 1, CS, 1, CS,
                                                CS, T);
  hipDeviceSynchronize();
  std::vector<float> gpu_out(T * CS), gpu_st(CS * CS);
  hipMemcpy(gpu_out.data(), dou, vb, hipMemcpyDeviceToHost);
  hipMemcpy(gpu_st.data(), dst, sb, hipMemcpyDeviceToHost);
  std::printf("  fp32 chunked-CPU vs PRODUCTION-GPU (batch-resident): out %.2e  state %.2e\n",
              rel_l2_vec(out_ch32, gpu_out), rel_l2_vec(st_ch32, gpu_st));
  std::printf("  fp32 sequential-CPU vs PRODUCTION-GPU             : out %.2e  state %.2e\n",
              rel_l2_vec(out_seq32, gpu_out), rel_l2_vec(st_seq32, gpu_st));

  // GPU prototype on the same inputs: correctness vs CPU-chunked + production
  hipMemcpy(dst, sf0.data(), sb, hipMemcpyHostToDevice);
  {
    const dim3 grid(1, CS / PROWS);
    gdn_chunked_proto_kernel<<<grid, PROWS>>>(dqk, dvv, dgg, dbb, dst, dou, 1, 1, CS, CS, T);
  }
  hipDeviceSynchronize();
  if (hipGetLastError() != hipSuccess) {
    std::printf("  GPU prototype LAUNCH FAILED: %s\n", hipGetErrorString(hipGetLastError()));
  } else {
    hipMemcpy(gpu_out.data(), dou, vb, hipMemcpyDeviceToHost);
    hipMemcpy(gpu_st.data(), dst, sb, hipMemcpyDeviceToHost);
    // refetch production numbers for the proto-vs-production comparison
    hipMemcpy(dst, sf0.data(), sb, hipMemcpyHostToDevice);
    (void)rdna4::delta_rule_batch_resident_launch(dqk, dvv, dgg, dbb, dst, dou, 1, 1, CS, 1, CS,
                                                  CS, T);
    hipDeviceSynchronize();
    std::vector<float> prod_out(T * CS);
    hipMemcpy(prod_out.data(), dou, vb, hipMemcpyDeviceToHost);
    std::printf("  GPU-proto vs fp32 chunked-CPU : out %.2e  state %.2e\n",
                rel_l2_vec(out_ch32, gpu_out), rel_l2_vec(st_ch32, gpu_st));
    std::printf("  GPU-proto vs PRODUCTION-GPU   : out %.2e  (gate: <=1e-5)\n",
                rel_l2_vec(prod_out, gpu_out));
  }
  for (auto *p : {dqk, dvv, dgg, dbb, dst, dou}) hipFree(p);

  // Production-shape timing: 48 heads, n=16 (prefill chunk), chain of 100
  {
    constexpr int NVH = 48, NKH = 16, N = 16, KEYD = NKH * CS, DIN = NVH * CS;
    float *qk2 = nullptr, *vv2 = nullptr, *gg2 = nullptr, *bb2 = nullptr, *st2 = nullptr,
          *ou2 = nullptr;
    (void)hipMalloc(&qk2, (std::size_t)N * 2 * KEYD * 4);
    (void)hipMalloc(&vv2, (std::size_t)N * DIN * 4);
    (void)hipMalloc(&gg2, (std::size_t)N * NVH * 4);
    (void)hipMalloc(&bb2, (std::size_t)N * NVH * 4);
    (void)hipMalloc(&st2, (std::size_t)NVH * CS * CS * 4);
    (void)hipMalloc(&ou2, (std::size_t)N * DIN * 4);
    if (!qk2 || !vv2 || !gg2 || !bb2 || !st2 || !ou2) {
      std::fprintf(stderr, "hipMalloc failed (timing)\n");
      return 1;
    }
    hipEvent_t e0, e1;
    hipEventCreate(&e0);
    hipEventCreate(&e1);
    auto time_launch = [&](const std::function<void()> &fn) {
      const double w0 = now_ms();
      while (now_ms() - w0 < 320.0) {
        for (int i = 0; i < 100; ++i) fn();
        hipDeviceSynchronize();
      }
      double best = 1e30;
      for (int r = 0; r < 12; ++r) {
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < 100; ++i) fn();
        hipEventRecord(e1, nullptr);
        hipDeviceSynchronize();
        float ms = 0;
        hipEventElapsedTime(&ms, e0, e1);
        best = std::min(best, (double)ms * 1000.0 / 100.0);
      }
      return best;
    };
    for (int i = 0; i < 200; ++i) {
      (void)rdna4::delta_rule_batch_resident_launch(qk2, vv2, gg2, bb2, st2, ou2, NVH, NKH, CS,
                                                    NVH, KEYD, DIN, N);
      (void)rdna4::delta_rule_batch_launch(qk2, vv2, gg2, bb2, st2, ou2, NVH, NKH, CS, NVH,
                                           KEYD, DIN, N);
    }
    hipDeviceSynchronize();
    const double res_us = time_launch([&] {
      (void)rdna4::delta_rule_batch_resident_launch(qk2, vv2, gg2, bb2, st2, ou2, NVH, NKH,
                                                    CS, NVH, KEYD, DIN, N);
    });
    const double old_us = time_launch([&] {
      (void)rdna4::delta_rule_batch_launch(qk2, vv2, gg2, bb2, st2, ou2, NVH, NKH, CS, NVH,
                                           KEYD, DIN, N);
    });
    // GPU chunked prototype, same shape: grid (48, 4), 32 threads/CTA
    for (int i = 0; i < 200; ++i) {
      const dim3 grid((unsigned)NVH, (unsigned)(CS / PROWS));
      gdn_chunked_proto_kernel<<<grid, PROWS>>>(qk2, vv2, gg2, bb2, st2, ou2, NKH, NVH, KEYD,
                                                DIN, N);
    }
    hipDeviceSynchronize();
    const double proto_us = time_launch([&] {
      const dim3 grid((unsigned)NVH, (unsigned)(CS / PROWS));
      gdn_chunked_proto_kernel<<<grid, PROWS>>>(qk2, vv2, gg2, bb2, st2, ou2, NKH, NVH, KEYD,
                                                DIN, N);
    });
    // one launch covers 48 heads x 16 tokens: per-token = /16, x48 layers
    std::printf("prefill batch (48 heads, n=16, chain-of-100, best of 12):\n");
    std::printf("  resident %.2f us/launch = %.3f ms/token(48 layers); non-resident %.2f us\n",
                res_us, res_us / 16.0 * 48.0 / 1000.0, old_us);
    std::printf("  chunked-GPU-proto %.2f us/launch = %.3f ms/token (%.3fx vs resident)\n", proto_us,
                proto_us / 16.0 * 48.0 / 1000.0, res_us / proto_us);
    std::printf("  target per pendente §2: gdn_delta 0.625 -> <=0.30 ms/token\n");
    for (auto *p : {qk2, vv2, gg2, bb2, st2, ou2}) hipFree(p);
    hipEventDestroy(e0);
    hipEventDestroy(e1);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// GPU prototype of the within-chunk parallel form (estudo §6.4, S1+S2 fused).
// Grid (nvh, S/32); CTA owns 32 state rows, staged once in LDS and kept
// resident across the chunk's blocks. B=16. T must be a multiple of B.
// Prototype scope: correctness-checked vs the CPU chunked form above and
// timed vs delta_rule_batch_resident_launch. S==128 only.
// (forward-declared above run_chunked_check; defined here).
// ---------------------------------------------------------------------------
__global__ void gdn_chunked_proto_kernel(const float *__restrict__ qk, const float *__restrict__ v,
                                         const float *__restrict__ gate,
                                         const float *__restrict__ beta, float *__restrict__ state,
                                         float *__restrict__ out, int n_k_heads, int n_vh,
                                         int key_dim, int d_inner, int n) {
  const int h = blockIdx.x;
  const int tx = threadIdx.x;  // 0..31
  const int j0 = blockIdx.y * PROWS;
  const int kh = h % n_k_heads;
  const float rs = rsqrtf(128.0f);
  __shared__ float H[32 * 128];    // Htile[j][i], resident across blocks
  __shared__ float Kb[16 * 128];   // K block rows
  __shared__ float Vb[16 * 128];   // V block rows
  __shared__ float Wm[16 * 128], Um[16 * 128];
  __shared__ float Tm[16 * 16], Gv[16];

  // stage the 32 state rows once (M[j][i] layout, row j owned logic per tx row)
  {
    const float4 *sp = (const float4 *)(state + ((std::int64_t)h * 128 + j0) * 128);
    // 32 rows x 32 vec4 = 1024 vec4 / 32 threads = 32 each
    for (int u = tx; u < 32 * 32; u += 32) {
      const int r = u / 32, c4 = u % 32;
      const float4 g = sp[r * 32 + c4];
      H[(r * 128) + c4 * 4 + 0] = g.x;
      H[(r * 128) + c4 * 4 + 1] = g.y;
      H[(r * 128) + c4 * 4 + 2] = g.z;
      H[(r * 128) + c4 * 4 + 3] = g.w;
    }
  }
  __syncthreads();

  const float *kb = qk + key_dim + (std::int64_t)kh * 128;
  const float *qb = qk + (std::int64_t)kh * 128;
  const int nblk = n / PB;
  for (int b = 0; b < nblk; ++b) {
    const int t0 = b * PB;
    // K/V block rows into LDS
    for (int u = tx; u < PB * 32; u += 32) {
      const int r = u / 32, c4 = u % 32;
      const float4 kk = ((const float4 *)(kb + (std::int64_t)(t0 + r) * 2 * key_dim))[c4];
      const float4 vv =
          ((const float4 *)(v + (std::size_t)(t0 + r) * d_inner + (std::size_t)h * 128))[c4];
      Kb[r * 128 + c4 * 4 + 0] = kk.x;
      Kb[r * 128 + c4 * 4 + 1] = kk.y;
      Kb[r * 128 + c4 * 4 + 2] = kk.z;
      Kb[r * 128 + c4 * 4 + 3] = kk.w;
      Vb[r * 128 + c4 * 4 + 0] = vv.x;
      Vb[r * 128 + c4 * 4 + 1] = vv.y;
      Vb[r * 128 + c4 * 4 + 2] = vv.z;
      Vb[r * 128 + c4 * 4 + 3] = vv.w;
    }
    if (tx < PB) {  // inclusive cumsum + beta/gate fetch (one thread per row)
      float acc = 0;
      for (int r = 0; r <= tx; ++r) acc += gate[(std::size_t)(t0 + r) * n_vh + h];
      Gv[tx] = acc;
    }
    __syncthreads();
    // A = -beta_row * strict-tril(KK^T o D); Tm = (I-A)^{-1} col by col
    for (int u = tx; u < PB * PB; u += 32) {
      const int r = u / PB, c = u % PB;
      float e = 0;
      if (c < r) {
        float dot = 0;
        for (int i = 0; i < 128; ++i) dot = fmaf(Kb[r * 128 + i], Kb[c * 128 + i], dot);
        e = dot * expf(Gv[r] - Gv[c]);
      }
      Tm[r * PB + c] = -beta[(std::size_t)(t0 + r) * n_vh + h] * e;  // A in place
    }
    __syncthreads();
    if (tx == 0) {
      // forward substitution: Tm <- (I-A)^{-1}, one column at a time
      float A[16 * 16], X[16 * 16];
      for (int i = 0; i < 16 * 16; ++i) {
        A[i] = Tm[i];
        X[i] = (i / 16 == i % 16) ? 1.0f : 0.0f;
      }
      for (int c = 0; c < 16; ++c)
        for (int r = 0; r < 16; ++r) {
          float acc = X[r * 16 + c];
          for (int s = 0; s < r; ++s) acc += A[r * 16 + s] * X[s * 16 + c];
          X[r * 16 + c] = acc;
        }
      for (int i = 0; i < 16 * 16; ++i) Tm[i] = X[i];
    }
    __syncthreads();
    // U = Tm(beta V); W = Tm(beta e^G K)
    for (int u = tx; u < PB * 128; u += 32) {
      const int r = u / 128, i = u % 128;
      float su = 0, sw = 0;
      for (int c = 0; c < PB; ++c) {
        const float t = Tm[r * PB + c] * beta[(std::size_t)(t0 + c) * n_vh + h];
        su += t * Vb[c * 128 + i];
        sw += t * (expf(Gv[c]) * Kb[c * 128 + i]);
      }
      Um[r * 128 + i] = su;
      Wm[r * 128 + i] = sw;
    }
    __syncthreads();
    // per owned state row j: vnew (16), output (uses H_in: computed BEFORE update)
    for (int jr = tx; jr < PROWS; jr += 32) {
      const int j = j0 + jr;
      float vnew[16];
      for (int r = 0; r < PB; ++r) {
        float s = 0;
        for (int i = 0; i < 128; ++i) s += Wm[r * 128 + i] * H[jr * 128 + i];
        vnew[r] = Um[r * 128 + j] - s;
      }
      // output rows for the 16 tokens of this block
      for (int r = 0; r < PB; ++r) {
        const float *qr = qb + (std::size_t)(t0 + r) * 2 * key_dim;
        float qh = 0;
        for (int i = 0; i < 128; ++i) qh = fmaf(qr[i], H[jr * 128 + i], qh);
        float intra = 0;
        for (int s = 0; s <= r; ++s) {
          const float *ks = Kb + s * 128;
          float d = 0;
          for (int i = 0; i < 128; ++i) d = fmaf(qr[i], ks[i], d);
          intra += d * expf(Gv[r] - Gv[s]) * vnew[s];
        }
        out[(std::size_t)(t0 + r) * d_inner + (std::size_t)h * 128 + j] =
            (expf(Gv[r]) * qh + intra) * rs;
      }
      // state update with vd = vnew * e^{G[B-1]-G[r]}
      const float gl = expf(Gv[PB - 1]);
      for (int i = 0; i < 128; ++i) {
        float s = 0;
        for (int r = 0; r < PB; ++r) s += Kb[r * 128 + i] * (vnew[r] * expf(Gv[PB - 1] - Gv[r]));
        H[jr * 128 + i] = H[jr * 128 + i] * gl + s;
      }
    }
    __syncthreads();
  }
  // write back once
  {
    float4 *dp = (float4 *)(state + ((std::int64_t)h * 128 + j0) * 128);
    for (int u = tx; u < 32 * 32; u += 32) {
      const int r = u / 32, c4 = u % 32;
      dp[r * 32 + c4] = make_float4(H[r * 128 + c4 * 4], H[r * 128 + c4 * 4 + 1],
                                    H[r * 128 + c4 * 4 + 2], H[r * 128 + c4 * 4 + 3]);
    }
  }
}

}  // namespace gdn_chunked

// ---------------------------------------------------------------------------
// H2 prototypes (plano-hipfire-gfx1201 §4 H2, steps 2-3). PROTOTYPE-ONLY: all
// kernels below live in this bench file; include/ and src/ are untouched.
//
// H2.2 occupancy: the shipping decode kernel is T128/RPT1/float4, grid = 48
//   CTAs x 128 threads (one CTA per value head, one thread per state row).
//   48 CTAs on a 64-CU part = 16 CUs idle. The variants below keep the EXACT
//   per-row summation order of the ship kernel (ascending-i single-accumulator
//   fma chains, two passes) so the gate stays bit-exact via memcmp:
//     rows16/32/64 : split each head's 128 rows over 8/4/2 CTAs (384/192/96
//                    CTAs total). Same per-thread work as ship, only the CTA
//                    grouping changes -> identical bytes by construction.
//     4trow        : 4 threads cooperate per state row (32 rows x 4 lanes per
//                    CTA of 128, grid 48x4 = 192 CTAs). Lanes stage their row
//                    quarters (float4 loads) into a TRANSPOSED LDS mailbox
//                    (mbox[i*32+rrow], conflict-free: simultaneous owner
//                    threads hit banks rrow, distinct); lane 0 of each group
//                    then issues the two fma passes in the SAME ascending-i
//                    order as ship. Loads/stores/syncs do not round, and no
//                    two threads share an element, so the result is bit-equal.
//   Killing bar (triage): delta is 0.856 ms of 26.4 ms per token; >=1% E2E
//   needs >=0.264 ms off delta => >=1.45x on us/launch. The pattern floor
//   (5.72 us, no chains) caps the headroom at ~1.47x over ship (8.42 us).
//
// H2.3 fusion sketch: conv1d(4 taps, 10240 chan) + l2_norm Q + l2_norm K are
//   3 launches/layer/token. The fused sketch does all three in ONE launch:
//   32 row-CTAs (256 threads, one Q/K row each, conv body + l2 tree copied
//   LINE FOR LINE incl. the 256-thread tree with its zero upper half) + 24
//   V-passthrough CTAs. Bit-exact by construction (fmaf(v,v,0)==v*v, x+0==x
//   for the +0 tree entries, lossless float store/load); proven by memcmp.
//   Prize: 2 launches x 48 layers = 96 launches/token ≈ 0.2-0.3 ms.
// ---------------------------------------------------------------------------
namespace h2 {

constexpr int HNVH = 48;
constexpr int HNKH = 16;
constexpr int HS = 128;

// --- H2.2a: rows-split float4 kernel (identical per-row body to ship) ------
template <int ROWS>
__global__ void delta_rows_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                  const float *__restrict__ v, const float *__restrict__ gate,
                                  const float *__restrict__ beta, float *__restrict__ state,
                                  float *__restrict__ out, int n_k_heads, int S) {
  const int h = blockIdx.x;                       // value head
  const int j = blockIdx.y * ROWS + threadIdx.x;  // row index (value dim)
  if (j >= S) return;
  const int kh = h % n_k_heads;
  const float *kd = k + (std::int64_t)kh * S;
  const float *qd = q + (std::int64_t)kh * S;
  float *row = state + ((std::int64_t)h * S + j) * S;
  const float g = expf(gate[h]);
  const float b = beta[h];
  const float4 *kd4 = (const float4 *)kd;
  float4 *p = (float4 *)row;
  float sum = 0.0f;
#pragma unroll 4
  for (int i4 = 0; i4 < S / 4; ++i4) {
    float4 x = p[i4];
    x.x *= g;
    x.y *= g;
    x.z *= g;
    x.w *= g;
    p[i4] = x;
    const float4 kk = kd4[i4];
    sum = fmaf(x.x, kk.x, sum);
    sum = fmaf(x.y, kk.y, sum);
    sum = fmaf(x.z, kk.z, sum);
    sum = fmaf(x.w, kk.w, sum);
  }
  const float delta = (v[(std::int64_t)h * S + j] - sum) * b;
  const float4 *qd4 = (const float4 *)qd;
  float acc = 0.0f;
#pragma unroll 4
  for (int i4 = 0; i4 < S / 4; ++i4) {
    const float4 kk = kd4[i4];
    const float4 qq = qd4[i4];
    float4 x = p[i4];
    x.x = fmaf(kk.x, delta, x.x);
    x.y = fmaf(kk.y, delta, x.y);
    x.z = fmaf(kk.z, delta, x.z);
    x.w = fmaf(kk.w, delta, x.w);
    p[i4] = x;
    acc = fmaf(x.x, qq.x, acc);
    acc = fmaf(x.y, qq.y, acc);
    acc = fmaf(x.z, qq.z, acc);
    acc = fmaf(x.w, qq.w, acc);
  }
  out[(std::int64_t)h * S + j] = acc * rsqrtf((float)S);
}

// --- H2.2b: 4-threads/row cooperative, owner-accumulates (bit-exact) -------
__global__ void delta_4trow_kernel(const float *__restrict__ q, const float *__restrict__ k,
                                   const float *__restrict__ v, const float *__restrict__ gate,
                                   const float *__restrict__ beta, float *__restrict__ state,
                                   float *__restrict__ out, int n_k_heads, int S) {
  constexpr int ROWS = 32;
  const int h = blockIdx.x;                 // value head
  const int rrow = threadIdx.x >> 2;        // 0..31 row-in-CTA
  const int lane = threadIdx.x & 3;         // 0..3 quarter-lane
  const int j = blockIdx.y * ROWS + rrow;   // row index (value dim)
  const int kh = h % n_k_heads;
  __shared__ float mbox[ROWS * 128];        // transposed: TROW(i) = mbox[i*32+rrow]
#define H2_TROW(i) (mbox[(i)*ROWS + rrow])
  const float *srow = state + ((std::int64_t)h * S + j) * S;
  // Stage: each lane float4-loads its quarter (8 vec4 = 32 floats).
  {
    const float4 *sp = (const float4 *)srow;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      const float4 gv = sp[lane * 8 + u];
      H2_TROW(lane * 32 + 4 * u + 0) = gv.x;
      H2_TROW(lane * 32 + 4 * u + 1) = gv.y;
      H2_TROW(lane * 32 + 4 * u + 2) = gv.z;
      H2_TROW(lane * 32 + 4 * u + 3) = gv.w;
    }
  }
  __syncthreads();
  if (lane == 0) {
    const float *kd = k + (std::int64_t)kh * S;
    const float *qd = q + (std::int64_t)kh * S;
    const float g = expf(gate[h]);
    const float b = beta[h];
    float sum = 0.0f;
    for (int i = 0; i < S; ++i) {
      const float x = H2_TROW(i) * g;  // same ascending-i order as ship
      H2_TROW(i) = x;
      sum = fmaf(x, kd[i], sum);
    }
    const float delta = (v[(std::int64_t)h * S + j] - sum) * b;
    float acc = 0.0f;
    for (int i = 0; i < S; ++i) {
      const float x = fmaf(kd[i], delta, H2_TROW(i));
      H2_TROW(i) = x;
      acc = fmaf(x, qd[i], acc);
    }
    out[(std::int64_t)h * S + j] = acc * rsqrtf((float)S);
    float4 *dp = (float4 *)(state + ((std::int64_t)h * S + j) * S);
#pragma unroll
    for (int i4 = 0; i4 < S / 4; ++i4)
      dp[i4] = make_float4(H2_TROW(4 * i4 + 0), H2_TROW(4 * i4 + 1), H2_TROW(4 * i4 + 2),
                           H2_TROW(4 * i4 + 3));
  }
#undef H2_TROW
}

// --- H2.3: fused conv(4tap)+SiLU+l2norm-Q/K, one launch --------------------
// Production shapes: chan=10240, K=4, key_dim=2048, d_inner=6144, nkh=16,
// S=128, eps=1e-6. Grid: 32 row-CTAs (one Q/K row each) + 24 V CTAs.
constexpr int FCHAN = 10240, FK = 4, FKEYD = 2048, FDIN = 6144, FNKH = 16, FS = 128;
constexpr float FEPS = 1e-6f;

__global__ void conv_silu_l2_fused_kernel(const float *__restrict__ qkv, const float *__restrict__ w,
                                          float *__restrict__ conv, float *__restrict__ cstate,
                                          int channels, int K) {
  constexpr int NROW = 2 * FNKH;  // 32 Q/K row CTAs
  if (blockIdx.x < NROW) {
    // One Q/K row: conv body verbatim, then the l2_norm_kernel body verbatim
    // (256 threads, ncols=128: threads >=128 contribute +0.0f, same tree).
    const int r = blockIdx.x;
    const bool is_q = r < FNKH;
    const int head = is_q ? r : r - FNKH;
    const int base = (is_q ? 0 : FKEYD) + head * FS;
    const int t = threadIdx.x;
    float val = 0.0f;
    if (t < FS) {
      const int c = base + t;
      float acc = 0.0f;
      for (int k = 0; k < K - 1; ++k) acc += w[c * K + k] * cstate[k * channels + c];
      acc += w[c * K + (K - 1)] * qkv[c];
      val = rdna4::silu_f(acc);
      for (int k = 0; k < K - 2; ++k) cstate[k * channels + c] = cstate[(k + 1) * channels + c];
      cstate[(K - 2) * channels + c] = qkv[c];
    }
    __shared__ float partial[256];
    float a = 0.0f;
    for (long long i = t; i < FS; i += 256) {
      const float vv = val;  // exactly one iter for t<128 (== xr[i] load, lossless)
      a = fmaf(vv, vv, a);
    }
    partial[t] = a;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
      if (t < stride) partial[t] += partial[t + stride];
      __syncthreads();
    }
    const float scale = rsqrtf(partial[0] + FEPS);
    if (t < FS) conv[base + t] = val * scale;
  } else {
    // V passthrough: conv+SiLU verbatim, no norm.
    const int t = threadIdx.x;
    const long long vc = (long long)(blockIdx.x - NROW) * 256 + t;
    if (vc < FDIN) {
      const int c = 2 * FKEYD + (int)vc;
      float acc = 0.0f;
      for (int k = 0; k < K - 1; ++k) acc += w[c * K + k] * cstate[k * channels + c];
      acc += w[c * K + (K - 1)] * qkv[c];
      conv[c] = rdna4::silu_f(acc);
      for (int k = 0; k < K - 2; ++k) cstate[k * channels + c] = cstate[(k + 1) * channels + c];
      cstate[(K - 2) * channels + c] = qkv[c];
    }
  }
}

struct Timed {
  double best = 1e30, mean = 0;
};

static int run_h2() {
  int CUs = 64;
  hipDeviceGetAttribute(&CUs, hipDeviceAttributeMultiprocessorCount, 0);
  std::printf("H2 prototypes (gfx1201, %d CUs). Bar: >=1.45x on delta us/launch\n", CUs);
  std::printf("(delta 0.856ms of 26.4ms; >=1%% E2E needs >=0.264ms off delta).\n");

  // ================= H2.2 delta occupancy =================
  const int dim = HNVH * HS;
  const std::size_t fb = (std::size_t)dim * sizeof(float);
  const std::size_t sb = (std::size_t)HNVH * HS * HS * sizeof(float);
  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr, *dst = nullptr,
        *dou = nullptr, *dref_s = nullptr, *dref_o = nullptr;
  bool ok = hipMalloc(&dq, fb) == hipSuccess && hipMalloc(&dk, fb) == hipSuccess &&
            hipMalloc(&dv, fb) == hipSuccess && hipMalloc(&dg, fb) == hipSuccess &&
            hipMalloc(&db, fb) == hipSuccess && hipMalloc(&dst, sb) == hipSuccess &&
            hipMalloc(&dou, fb) == hipSuccess && hipMalloc(&dref_s, sb) == hipSuccess &&
            hipMalloc(&dref_o, fb) == hipSuccess;
  if (!ok) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  {
    const unsigned bl = 256;
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dq, dim, 11u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dk, dim, 12u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dv, dim, 13u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dg, dim, 14u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(db, dim, 15u);
    fill_kernel<<<(unsigned)((sb / 4 + bl - 1) / bl), bl>>>((float *)dst, (long long)(sb / 4), 7u);
    hipDeviceSynchronize();
    hipMemcpy(dref_s, dst, sb, hipMemcpyDeviceToDevice);
  }
  auto ship = [&] {
    (void)rdna4::delta_rule_launch(dq, dk, dv, dg, db, dst, dou, HNVH, HNKH, HS);
  };
  auto rows16 = [&] { delta_rows_kernel<16><<<dim3((unsigned)HNVH, 8), 16>>>(dq, dk, dv, dg, db, dst, dou, HNKH, HS); };
  auto rows32 = [&] { delta_rows_kernel<32><<<dim3((unsigned)HNVH, 4), 32>>>(dq, dk, dv, dg, db, dst, dou, HNKH, HS); };
  auto rows64 = [&] { delta_rows_kernel<64><<<dim3((unsigned)HNVH, 2), 64>>>(dq, dk, dv, dg, db, dst, dou, HNKH, HS); };
  auto trow4 = [&] {
    delta_4trow_kernel<<<dim3((unsigned)HNVH, 4), 128>>>(dq, dk, dv, dg, db, dst, dou, HNKH, HS);
  };
  struct V {
    const char *name;
    std::function<void()> fn;
    int ctas, blk;
  };
  std::vector<V> vs = {{"ship T128/RPT1/float4", ship, 48, 128}, {"rows16 (384 CTA)", rows16, 384, 16},
                       {"rows32 (192 CTA)", rows32, 192, 32},    {"rows64 (96 CTA)", rows64, 96, 64},
                       {"4trow coop (192 CTA)", trow4, 192, 128}};

  // 1. bit-exactness gate (memcmp state+out vs ship)
  std::vector<float> ref_o(dim), got_o(dim), ref_s(sb / 4), got_s(sb / 4);
  hipMemcpy(dst, dref_s, sb, hipMemcpyDeviceToDevice);
  ship();
  hipDeviceSynchronize();
  if (hipGetLastError() != hipSuccess) {
    std::printf("ship launch failed: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  hipMemcpy(ref_o.data(), dou, fb, hipMemcpyDeviceToHost);
  hipMemcpy(ref_s.data(), dst, sb, hipMemcpyDeviceToHost);
  std::printf("\n[H2.2] bit-exatidao vs ship (memcmp %zu estado + %d saida):\n", ref_s.size(), dim);
  int bad = 0;
  for (std::size_t i = 1; i < vs.size(); ++i) {
    hipMemcpy(dst, dref_s, sb, hipMemcpyDeviceToDevice);
    vs[i].fn();
    hipDeviceSynchronize();
    if (hipGetLastError() != hipSuccess) {
      std::printf("  %-22s LAUNCH FAILED: %s\n", vs[i].name, hipGetErrorString(hipGetLastError()));
      ++bad;
      continue;
    }
    hipMemcpy(got_o.data(), dou, fb, hipMemcpyDeviceToHost);
    hipMemcpy(got_s.data(), dst, sb, hipMemcpyDeviceToHost);
    const bool so = std::memcmp(got_o.data(), ref_o.data(), fb) == 0;
    const bool ss = std::memcmp(got_s.data(), ref_s.data(), sb) == 0;
    std::printf("  %-22s saida %s  estado %s\n", vs[i].name, so ? "IDENTICA" : "DIFERE",
                ss ? "IDENTICO" : "DIFERE");
    if (!so || !ss) ++bad;
  }

  // 2. occupancy (runtime API) per variant kernel
  std::printf("\n[H2.2] ocupacao (hipOccupancyMaxActiveBlocksPerMultiprocessor):\n");
  {
    int n = 0;
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, rdna4::delta_rule_kernel_ilp<128, 1, true>, 128, 0);
    std::printf("  ship T128/RPT1/float4 : %d blks/CU x 48 CTA = %d waves/CU-equiv\n", n, n * 48);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, delta_rows_kernel<16>, 16, 0);
    std::printf("  rows16                : %d blks/CU (384 CTA)\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, delta_rows_kernel<32>, 32, 0);
    std::printf("  rows32                : %d blks/CU (192 CTA)\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, delta_rows_kernel<64>, 64, 0);
    std::printf("  rows64                : %d blks/CU (96 CTA)\n", n);
    hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, delta_4trow_kernel, 128, 0);
    std::printf("  4trow coop            : %d blks/CU (192 CTA, 16KB LDS)\n", n);
  }

  // 3. interleaved A/B timing: best+mean, chain-of-100, warmup>=300ms
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  constexpr int CHN = 100, ROUNDS = 12;
  auto measure = [&](const std::function<void()> &fn) {
    const double w0 = now_ms();
    while (now_ms() - w0 < 320.0) {
      for (int i = 0; i < CHN; ++i) fn();
      hipDeviceSynchronize();
    }
    double best = 1e30, sum = 0;
    for (int r = 0; r < ROUNDS; ++r) {
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < CHN; ++i) fn();
      hipEventRecord(e1, nullptr);
      hipDeviceSynchronize();
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      const double us = (double)ms * 1000.0 / CHN;
      best = std::min(best, us);
      sum += us;
    }
    return Timed{best, sum / ROUNDS};
  };
  for (auto &v : vs)
    for (int i = 0; i < 200; ++i) v.fn();
  hipDeviceSynchronize();
  std::vector<Timed> tm(vs.size());
  for (int r = 0; r < 3; ++r)
    for (std::size_t k = 0; k < vs.size(); ++k) {
      const std::size_t idx = (k + r) % vs.size();
      const Timed t = measure(vs[idx].fn);
      if (t.best < tm[idx].best) tm[idx].best = t.best;
      tm[idx].mean = tm[idx].mean == 0 ? t.mean : (tm[idx].mean + t.mean) / 2;
    }
  std::printf("\n[H2.2] us/launch (chain %d, %d rounds x 3 passadas rotacionadas, best+mean):\n",
              CHN, ROUNDS);
  for (std::size_t i = 0; i < vs.size(); ++i)
    std::printf("  %-22s best %7.2f us  mean %7.2f us  %6.3fx  (x48 = %.3f ms/token)\n", vs[i].name,
                tm[i].best, tm[i].mean, tm[0].best / tm[i].best, tm[i].best * 48.0 / 1000.0);
  std::printf("  piso 1%% E2E = 0.264ms => precisa <= %.2f us/launch (%.2fx)\n",
              tm[0].best - 264.0 / 48.0 * 1000.0, tm[0].best / (tm[0].best - 264.0 / 48.0 * 1000.0));

  // ================= H2.3 conv+SiLU+norm fusion =================
  std::printf("\n[H2.3] conv+SiLU+l2norm-Q/K fusion sketch (chan=%d K=%d, 3 launches -> 1):\n",
              FCHAN, FK);
  float *fq = nullptr, *fw = nullptr, *fc = nullptr, *fcs = nullptr, *frs = nullptr, *frc = nullptr;
  const std::size_t qb = (std::size_t)FCHAN * 4, wb = (std::size_t)FCHAN * FK * 4,
                    csb = (std::size_t)(FK - 1) * FCHAN * 4;
  ok = hipMalloc(&fq, qb) == hipSuccess && hipMalloc(&fw, wb) == hipSuccess &&
       hipMalloc(&fc, qb) == hipSuccess && hipMalloc(&fcs, csb) == hipSuccess &&
       hipMalloc(&frs, csb) == hipSuccess && hipMalloc(&frc, qb) == hipSuccess;
  if (!ok) {
    std::fprintf(stderr, "hipMalloc failed (fusion)\n");
    return 1;
  }
  {
    const unsigned bl = 256;
    fill_kernel<<<(unsigned)((qb / 4 + bl - 1) / bl), bl>>>(fq, FCHAN, 21u);
    fill_kernel<<<(unsigned)((wb / 4 + bl - 1) / bl), bl>>>(fw, (long long)(wb / 4), 22u);
    fill_kernel<<<(unsigned)((csb / 4 + bl - 1) / bl), bl>>>((float *)fcs, (long long)(csb / 4),
                                                            23u);
    hipDeviceSynchronize();
    hipMemcpy(frs, fcs, csb, hipMemcpyDeviceToDevice);
  }
  auto unfused = [&] {
    (void)rdna4::conv1d_state_launch(fq, fw, fc, fcs, FCHAN, FK);
    (void)rdna4::l2_norm_launch(fc, fc, FNKH, FS, FEPS);
    (void)rdna4::l2_norm_launch(fc + FKEYD, fc + FKEYD, FNKH, FS, FEPS);
  };
  auto fused = [&] {
    conv_silu_l2_fused_kernel<<<2 * FNKH + (FDIN + 255) / 256, 256>>>(fq, fw, fc, fcs, FCHAN, FK);
  };
  hipMemcpy(fcs, frs, csb, hipMemcpyDeviceToDevice);
  unfused();
  hipDeviceSynchronize();
  std::vector<float> fref(qb / 4), fgot(qb / 4), frefs(csb / 4), fgots(csb / 4);
  hipMemcpy(fref.data(), fc, qb, hipMemcpyDeviceToHost);
  hipMemcpy(frefs.data(), fcs, csb, hipMemcpyDeviceToHost);
  hipMemcpy(fcs, frs, csb, hipMemcpyDeviceToDevice);
  fused();
  hipDeviceSynchronize();
  if (hipGetLastError() != hipSuccess) {
    std::printf("  fused LAUNCH FAILED: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  hipMemcpy(fgot.data(), fc, qb, hipMemcpyDeviceToHost);
  hipMemcpy(fgots.data(), fcs, csb, hipMemcpyDeviceToHost);
  const bool fo = std::memcmp(fgot.data(), fref.data(), qb) == 0;
  const bool fs = std::memcmp(fgots.data(), frefs.data(), csb) == 0;
  std::printf("  bit-identidade vs unfused: conv-out %s  conv-state %s\n", fo ? "IDENTICO" : "DIFERE",
              fs ? "IDENTICO" : "DIFERE");
  for (int i = 0; i < 200; ++i) {
    unfused();
    fused();
  }
  hipDeviceSynchronize();
  // interleave A/B at the CHAIN level for launch-amortized honesty
  Timed tu{1e30, 0}, tf{1e30, 0};
  for (int r = 0; r < 3; ++r) {
    Timed a = measure([&] {
      for (int i = 0; i < CHN; ++i) unfused();
    });
    Timed b = measure([&] {
      for (int i = 0; i < CHN; ++i) fused();
    });
    tu.best = std::min(tu.best, a.best);
    tf.best = std::min(tf.best, b.best);
    tu.mean = tu.mean == 0 ? a.mean : (tu.mean + a.mean) / 2;
    tf.mean = tf.mean == 0 ? b.mean : (tf.mean + b.mean) / 2;
  }
  // each chain iter ran CHN trios; per-trio = /CHN already in measure? No:
  // measure() divides by CHN (launches). Here one fn() = CHN trios, so the
  // reported "us/launch" is us per CHN-trios; divide once more by CHN.
  std::printf("  unfused-trio best %.2f us  mean %.2f us | fused best %.2f us  mean %.2f us\n",
              tu.best / CHN, tu.mean / CHN, tf.best / CHN, tf.mean / CHN);
  std::printf("  diff/layer-trio best %+.2f us (x48 = %+.3f ms/token) | speedup %.3fx\n",
              tu.best / CHN - tf.best / CHN, (tu.best / CHN - tf.best / CHN) * 48.0 / 1000.0,
              tu.best / tf.best);
  std::printf("  launches/layer/token: 3 -> 1 (saves 96 launches/token)\n");

  for (auto *p : {dq, dk, dv, dg, db, dst, dou, dref_s, dref_o, fq, fw, fc, fcs, frs, frc})
    if (p) hipFree(p);
  hipEventDestroy(e0);
  hipEventDestroy(e1);
  return bad ? 1 : 0;
}

}  // namespace h2

}  // namespace

int main(int argc, char **argv) {
  int rounds = 20;
  int chain = CH;
  bool noise = true;
  bool chunked = false;
  bool h2flag = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-noise")) noise = false;
    else if (!std::strcmp(argv[i], "--chunked")) chunked = true;
    else if (!std::strcmp(argv[i], "--h2")) h2flag = true;
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 1; }
  }
  if (chunked) return gdn_chunked::run_chunked_check();
  if (h2flag) return h2::run_h2();

  const int dim = NVH * S;
  const std::size_t fb = (std::size_t)dim * sizeof(float);
  const std::size_t sb = (std::size_t)NVH * S * S * sizeof(float);
  float *dq = nullptr, *dk = nullptr, *dv = nullptr, *dg = nullptr, *db = nullptr, *dstate = nullptr,
        *dout = nullptr, *dref_state = nullptr, *dref_out = nullptr;
  bool ok = hipMalloc(&dq, fb) == hipSuccess && hipMalloc(&dk, fb) == hipSuccess &&
            hipMalloc(&dv, fb) == hipSuccess && hipMalloc(&dg, fb) == hipSuccess &&
            hipMalloc(&db, fb) == hipSuccess && hipMalloc(&dstate, sb) == hipSuccess &&
            hipMalloc(&dout, fb) == hipSuccess && hipMalloc(&dref_state, sb) == hipSuccess &&
            hipMalloc(&dref_out, fb) == hipSuccess;
  if (!ok) { std::fprintf(stderr, "hipMalloc failed\n"); return 1; }
  {
    const unsigned bl = 256;
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dq, dim, 11u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dk, dim, 12u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dv, dim, 13u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(dg, dim, 14u);
    fill_kernel<<<(unsigned)((fb / 4 + bl - 1) / bl), bl>>>(db, dim, 15u);
    // gamas de gate realistas (o expf nao muda de preco, mas estado!=0 importa)
    fill_kernel<<<(unsigned)((sb / 4 + bl - 1) / bl), bl>>>((float *)dstate, (long long)(sb / 4), 7u);
    hipDeviceSynchronize();
    // copia de referencia do estado inicial
    hipMemcpy(dref_state, dstate, sb, hipMemcpyDeviceToDevice);
  }

  // "ship" = o caminho que EMBARCA hoje (gdn.cuh); "scalar" = o kernel historico,
  // que fica no arquivo para esta comparacao poder ser re-medida.
  auto shipping = [&] {
    (void)rdna4::delta_rule_launch(dq, dk, dv, dg, db, dstate, dout, NVH, NKH, S);
  };
  auto scalar = [&] {
    (void)rdna4::delta_rule_launch_scalar(dq, dk, dv, dg, db, dstate, dout, NVH, NKH, S);
  };

  // Variantes: todas chamam o caminho que pode embarcar (gdn.cuh), nao uma copia
  // local do kernel -- o que for medido aqui e' o que pode entrar no motor.
  auto ilp = [&](int t, int rpt, bool vec) {
    return [&, t, rpt, vec] {
      (void)rdna4::delta_rule_launch_ilp(t, rpt, vec, dq, dk, dv, dg, db, dstate, dout, NVH, NKH, S);
    };
  };
  std::vector<Variant> variants;
  variants.push_back({"ship  (o que embarca)", shipping, 128});
  variants.push_back({"historico T128 escalar", scalar, 128});
  if (noise) variants.push_back({"ship2 (piso de ruido)", shipping, 128});
  variants.push_back({"T128 RPT1 float4", ilp(128, 1, true), 128});
  variants.push_back({"T64  RPT2 float4", ilp(64, 2, true), 64});
  variants.push_back({"T32  RPT4 float4", ilp(32, 4, true), 32});
  variants.push_back({"T64  RPT2 escalar", ilp(64, 2, false), 64});
  variants.push_back({"T32  RPT4 escalar", ilp(32, 4, false), 32});
  variants.push_back({"piso do padrao (le+escreve)", [&] {
    pattern_floor_kernel<<<NVH, 128>>>(dq, dk, dstate, dout);
  }, 128});

  // -----------------------------------------------------------------------
  // 1. Bit-exatidao: cada variante tem que reproduzir estado e saida do ship.
  // -----------------------------------------------------------------------
  std::vector<float> ref_out(dim), got_out(dim);
  std::vector<float> ref_st(sb / 4), got_st(sb / 4);
  hipMemcpy(dstate, dref_state, sb, hipMemcpyDeviceToDevice);
  shipping();
  hipDeviceSynchronize();
  hipMemcpy(ref_out.data(), dout, fb, hipMemcpyDeviceToHost);
  hipMemcpy(ref_st.data(), dstate, sb, hipMemcpyDeviceToHost);

  int bad = 0;
  std::printf("bit-exatidao (vs ship, memcmp de %zu floats de estado + %d de saida):\n", ref_st.size(), dim);
  for (std::size_t v = 1; v < variants.size(); ++v) {
    if (std::strstr(variants[v].name, "piso do padrao")) continue;
    hipMemcpy(dstate, dref_state, sb, hipMemcpyDeviceToDevice);
    variants[v].launch();
    hipDeviceSynchronize();
    hipMemcpy(got_out.data(), dout, fb, hipMemcpyDeviceToHost);
    hipMemcpy(got_st.data(), dstate, sb, hipMemcpyDeviceToHost);
    const bool same_o = std::memcmp(got_out.data(), ref_out.data(), fb) == 0;
    const bool same_s = std::memcmp(got_st.data(), ref_st.data(), sb) == 0;
    std::printf("  %-28s saida %s  estado %s\n", variants[v].name, same_o ? "IDENTICA" : "DIFERE",
                same_s ? "IDENTICO" : "DIFERE");
    if (!same_o || !same_s) ++bad;
  }
  if (bad) std::printf("  ATENCAO: %d variante(s) nao batem bit a bit — medir mesmo assim, mas a\n"
                       "  variante so entra no motor com gate numerico (check_regression/compare_ppl).\n", bad);

  // -----------------------------------------------------------------------
  // 2. Tempo: cadeia de `chain` lançamentos, min de `rounds` rodadas.
  // -----------------------------------------------------------------------
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto measure = [&](const std::function<void()> &fn) {
    const double w0 = now_ms();
    while (now_ms() - w0 < 320.0) {
      for (int i = 0; i < chain; ++i) fn();
      hipDeviceSynchronize();
    }
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < chain; ++i) fn();
      hipEventRecord(e1, nullptr);
      hipDeviceSynchronize();
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      best = std::min(best, (double)ms * 1000.0 / chain);
    }
    return best;
  };
  // O aquecimento do PRIMEIRO medido nao pode ser o unico: aquece tudo uma vez.
  for (auto &v : variants) { for (int i = 0; i < 200; ++i) v.launch(); }
  hipDeviceSynchronize();

  std::vector<double> us(variants.size(), 0.0);
  for (int r = 0; r < 3; ++r) {  // 3 passadas com ordem rotacionada
    for (std::size_t k = 0; k < variants.size(); ++k) {
      const std::size_t idx = (k + r) % variants.size();
      const double t = measure(variants[idx].launch);
      if (us[idx] == 0.0 || t < us[idx]) us[idx] = t;
    }
  }

  std::printf("\nus/lançamento (cadeia de %d, min de %d rodadas x 3 passadas rotacionadas):\n", chain,
              rounds);
  const double ship_us = us[0];
  for (std::size_t v = 0; v < variants.size(); ++v) {
    std::printf("  %-28s %8.2f us   %.3fx   (%.1f GB/s de estado)\n", variants[v].name, us[v],
                ship_us / us[v], 12.6e6 / (us[v] * 1e-6) / 1e9);
  }
  std::printf("\n  por camada recorrente; x48 camadas = %.3f ms/token (ship) / %.3f ms/token (melhor)\n",
              ship_us * 48.0 / 1000.0,
              *std::min_element(us.begin() + 1, us.end()) * 48.0 / 1000.0);
  if (noise) {
    std::size_t i2 = 0;
    for (std::size_t v = 0; v < variants.size(); ++v)
      if (std::strstr(variants[v].name, "ship2")) i2 = v;
    std::printf("  piso de ruido do harness: ship %.2f us vs ship2 %.2f us = %.4fx\n", us[0],
                us[i2], us[0] / us[i2]);
  }
  return 0;
}
