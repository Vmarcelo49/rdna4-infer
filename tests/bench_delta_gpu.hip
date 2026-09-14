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

}  // namespace

int main(int argc, char **argv) {
  int rounds = 20;
  int chain = CH;
  bool noise = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-noise")) noise = false;
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 1; }
  }

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
    std::printf("  piso de ruido do harness: ship %.2f us vs ship2 %.2f us = %.4fx\n", us[0], us[1],
                us[0] / us[1]);
  }
  return 0;
}
