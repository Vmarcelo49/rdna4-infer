// M2 step 3 — GPU fused matvec vs CPU reference, per quantization type.
//
// GPU:  y = W_quant * quantize_q8_1(x)        (include/rdna4/matvec.cuh)
// CPU:  y = dequant_row_<type>(W_quant) . dequant_q8_1(x_q8_1)
//       (llama.cpp's dequantize_row_* + the fp16 d*qs[i] rule for the
//        activation block, so BOTH sides consume identical bytes and the
//        only difference is float accumulation order / FMA.)
//
// usage: check-matvec-gpu <file.gguf> [nrows]
#include "rdna4/matvec.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/loader.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);
extern "C" void rdna4_cpu_quantize_q8_1(const float *x, void *vy, std::int64_t nelem);

namespace {

using rdna4::DType;

// dtype ordinal -> short type name (as used in the tables/PLAN.md)
inline const char *tc_name_of(int dt) {
  switch (dt) {
    case 1: return "q8_0";   case 2: return "q2_k";    case 3: return "q3_k";
    case 4: return "q4_k";   case 5: return "q5_k";    case 6: return "q6_k";
    case 7: return "iq2_xxs";case 8: return "iq2_xs";  case 9: return "iq3_xxs";
    case 10: return "iq1_s"; case 11: return "iq4_nl"; case 12: return "iq3_s";
    case 13: return "iq2_s"; case 14: return "iq4_xs"; default: return "?";
  }
}

struct TypeCase {
  DType dt;
  const char *name;
  const char *tensor;   // preferred tensor name (falls back to first usable of type)
  double tol_rel_l2;    // documented tolerance (see below)
};

// Tolerances: the GPU path uses llama.cpp's integer-optimized vec_dot, which
// scales sub-block partial sums in INT arithmetic, e.g.
//   iq2_xs/iq2_s: sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0+sumi1)/2)/4
//   iq3_xxs:      sumi = (ls*sumi + sumi/2)/2
//   iq2_xxs:      sumi = sumi*ls/8
//   iq1_s:        uses the fp16 `s` field (d*sum(qs)) for its delta term
// That truncation is NOT present in the exact-f32 reference (dequantize_row_*
// followed by an f32 dot), so those types differ by ~1e-5..1e-4 relative
// (growing with the number of sub-blocks). Types using exact float math match
// the reference at fp32 noise level (~1e-7). Tolerances below carry >=5x
// margin over the values observed on both UD files.
constexpr double kExactTol = 1e-6;
constexpr double kIntTruncTol = 1e-3;

const TypeCase kCases[] = {
    {DType::Q8_0, "q8_0", nullptr, kExactTol},
    {DType::Q2_K, "q2_k", nullptr, kExactTol},
    {DType::Q3_K, "q3_k", nullptr, kExactTol},
    {DType::Q4_K, "q4_k", nullptr, kExactTol},
    {DType::Q5_K, "q5_k", nullptr, kExactTol},
    {DType::Q6_K, "q6_k", nullptr, kExactTol},
    {DType::IQ2_XXS, "iq2_xxs", nullptr, kIntTruncTol},
    {DType::IQ2_XS, "iq2_xs", nullptr, kIntTruncTol},
    {DType::IQ3_XXS, "iq3_xxs", nullptr, kIntTruncTol},
    {DType::IQ1_S, "iq1_s", nullptr, kIntTruncTol},
    {DType::IQ4_NL, "iq4_nl", nullptr, kExactTol},
    {DType::IQ3_S, "iq3_s", nullptr, kExactTol},
    {DType::IQ2_S, "iq2_s", nullptr, kIntTruncTol},
    {DType::IQ4_XS, "iq4_xs", nullptr, kExactTol},
};

// Deterministic pseudo-random activation vector.
void fill_x(std::vector<float> &x, int64_t n, unsigned seed) {
  x.resize(n);
  unsigned s = seed;
  for (int64_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 3.0f;
  }
}

struct Stats {
  double rel_l2, max_abs;
  long long nelem;
};

Stats compare(const std::vector<float> &a, const std::vector<float> &b) {
  Stats s{0, 0, (long long)a.size()};
  double num = 0, da = 0, db = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    da += (double)a[i] * a[i];
    db += (double)b[i] * b[i];
    s.max_abs = std::fmax(s.max_abs, std::fabs(d));
  }
  s.rel_l2 = std::sqrt(num / (da + db));
  return s;
}


// ---------------------------------------------------------------------------
// `--check-lds`: bit-exatidao do caminho com a LUT em LDS contra o caminho de
// producao, tipo a tipo, com o tensor REAL do inventario.
//
// Por que este gate existe separado: a LUT em LDS nao muda nenhuma operacao (a
// LDS recebe os mesmos bytes da tabela global), entao o resultado tem que ser
// BIT-IDENTICO -- nao "dentro da tolerancia do oraculo de CPU", que e' frouxa
// para os tipos IQ (a referencia exata em f32 difere do vec_dot inteiro por
// 1e-5..1e-4 de proposito). Comparar com memcmp e' o unico jeito de provar
// isso; uma tolerancia aceitaria uma tabela errada.
//
// Anti-teste-vazio: exige pelo menos um tipo com kernel de LDS testado; se
// nenhum tipo do modelo tiver tensor usavel, ele FALHA em vez de passar vazio.
// ---------------------------------------------------------------------------
int run_check_lds(rdna4::GgufLoader &ld) {
  int tested = 0, failures = 0;
  std::printf("check-lds: LUT em LDS vs LUT global, bit-exatidao por tipo\n");
  for (int dt = 1; dt <= 14; ++dt) {
    if (!rdna4::matvec_has_lut_lds(dt)) continue;
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape(dt, shape)) continue;
    std::size_t idx = std::size_t(-1);
    std::uint64_t best = 0;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if ((int)ld.dtype(i) != dt) continue;
      const auto &d = ld.meta().tensors[i].dims;
      if (d.size() != 2 || d[1] % shape.qk != 0 || d[0] <= 0) continue;
      if (ld.tensor_bytes(i) > best) { best = ld.tensor_bytes(i); idx = i; }
    }
    if (idx == std::size_t(-1)) {
      std::printf("  %-9s sem tensor usavel no modelo\n", tc_name_of(dt));
      continue;
    }
    const auto &info = ld.meta().tensors[idx];
    // convencao do loader/Graph::proj: dims[0] = ncols (dim de reducao),
    // dims[1] = nrows. Trocar isso aqui mediria uma forma que nao existe.
    const int64_t ncols = info.dims[0];
    const int64_t nrows = std::min<int64_t>(info.dims[1], 256);  // 256 linhas bastam
    std::vector<std::uint8_t> w;
    std::string err;
    const std::uint64_t row_bytes = (std::uint64_t)(ncols / shape.qk) * shape.block_bytes;
    if (!ld.load_tensor_range(idx, 0, row_bytes * (std::uint64_t)nrows, w, err)) {
      std::printf("  %-9s carga falhou: %s\n", tc_name_of(dt), err.c_str());
      ++failures;
      continue;
    }
    std::vector<float> x;
    fill_x(x, ncols, 777u + (unsigned)dt);
    const int64_t nq8 = ncols / QK8_1;
    void *d_w = nullptr;
    rdna4::block_q8_1 *d_a = nullptr;
    float *d_x = nullptr;
    float *d_y0 = nullptr, *d_y1 = nullptr;
    if (hipMalloc(&d_w, w.size()) != hipSuccess ||
        hipMalloc(&d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1)) != hipSuccess ||
        hipMalloc(&d_x, (std::size_t)ncols * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_y0, (std::size_t)nrows * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_y1, (std::size_t)nrows * sizeof(float)) != hipSuccess) {
      std::printf("  %-9s hipMalloc falhou\n", tc_name_of(dt));
      ++failures;
      continue;
    }
    hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((nq8 + 3) / 4), 128>>>(d_x, d_a, nq8);
    const bool ok0 = rdna4::matvec_launch(dt, d_w, d_a, d_y0, nrows, ncols, nullptr);
    const bool ok1 = rdna4::matvec_launch_lut_lds(dt, d_w, d_a, d_y1, nrows, ncols, nullptr);
    hipDeviceSynchronize();
    if (!ok0 || !ok1) {
      std::printf("  %-9s lancamento falhou (global %d lds %d)\n", tc_name_of(dt), (int)ok0, (int)ok1);
      ++failures;
    } else {
      std::vector<float> y0(nrows), y1(nrows);
      hipMemcpy(y0.data(), d_y0, (std::size_t)nrows * sizeof(float), hipMemcpyDeviceToHost);
      hipMemcpy(y1.data(), d_y1, (std::size_t)nrows * sizeof(float), hipMemcpyDeviceToHost);
      const bool same = std::memcmp(y0.data(), y1.data(), (std::size_t)nrows * sizeof(float)) == 0;
      double maxd = 0;
      for (int64_t i = 0; i < nrows; ++i) maxd = std::fmax(maxd, (double)std::fabs(y0[i] - y1[i]));
      std::printf("  %-9s %s  %lld x %lld  %s  max|d| %.3e  (regra de producao usa LDS: %s)\n",
                  tc_name_of(dt), info.name.c_str(), (long long)nrows, (long long)ncols,
                  same ? "BIT-IDENTICO" : "DIFERE", maxd,
                  rdna4::matvec_lut_lds_ok(dt, nrows, ncols) ? "sim" : "nao");
      if (!same) ++failures;
      ++tested;
    }
    hipFree(d_w); hipFree(d_a); hipFree(d_x); hipFree(d_y0); hipFree(d_y1);
  }
  if (tested == 0) {
    std::printf("check-lds: FAIL: nenhum tipo testado (teste vazio)\n");
    return 1;
  }
  if (failures) {
    std::printf("check-lds: FAIL: %d falha(s) em %d tipo(s)\n", failures, tested);
    return 1;
  }
  std::printf("check-lds: OK (%d tipos bit-identicos entre LUT global e LDS)\n", tested);
  return 0;
}

// Bench mode: for each type, matvec the LARGEST usable tensor over ALL its
// rows, timed with hipEvents. Decode is memory-bandwidth bound, so the
// reported GB/s is the number that matters (weights are re-read per token).
int run_bench(rdna4::GgufLoader &ld, int force_ilp = 0, int force_pf = 0) {
  // force_ilp == 0 -> each type's shipping (measured) ILP.
  auto launch = [&](int dt, const void *dw, const rdna4::block_q8_1 *da, float *dy,
                    int64_t nrows, int64_t ncols) {
    if (force_ilp > 0) {
      return rdna4::matvec_launch_tuned(dt, dw, da, dy, nrows, ncols, nullptr, force_ilp,
                                        force_pf != 0);
    }
    return rdna4::matvec_launch(dt, dw, da, dy, nrows, ncols, nullptr);
  };
  double total_bytes = 0, total_sec = 0;
  static double bw_by_type[16] = {0};
  for (int i = 0; i < 16; ++i) bw_by_type[i] = 0;
  std::printf("%-9s %-24s %10s %10s %8s %10s\n", "type", "tensor", "rows", "cols", "GB/s", "ms");
  for (const TypeCase &tc : kCases) {
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape((int)tc.dt, shape)) continue;

    std::size_t best = std::size_t(-1);
    std::uint64_t best_bytes = 0;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if (ld.dtype(i) != tc.dt) continue;
      const auto &d = ld.meta().tensors[i].dims;
      if (d.size() != 2 || d[1] % shape.qk != 0 || d[0] <= 0) continue;
      if (ld.tensor_bytes(i) > best_bytes) { best_bytes = ld.tensor_bytes(i); best = i; }
    }
    if (best == std::size_t(-1)) {
      std::printf("%-9s (absent)\n", tc.name);
      continue;
    }
    const auto &info = ld.meta().tensors[best];
    const int64_t ncols = info.dims[1], nrows = info.dims[0];

    std::vector<std::uint8_t> w;
    std::string err;
    if (!ld.load_tensor(best, w, err)) { std::fprintf(stderr, "%s: %s\n", tc.name, err.c_str()); continue; }

    std::vector<float> x;
    fill_x(x, ncols, 999u);
    void *d_w = nullptr; rdna4::block_q8_1 *d_a = nullptr; float *d_y = nullptr; float *d_x = nullptr;
    hipMalloc(&d_w, w.size());
    hipMalloc(&d_a, (std::size_t)(ncols / QK8_1) * sizeof(rdna4::block_q8_1));
    hipMalloc(&d_y, (std::size_t)nrows * sizeof(float));
    hipMalloc(&d_x, (std::size_t)ncols * sizeof(float));
    hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((ncols / QK8_1 + 3) / 4), 128>>>(d_x, d_a, ncols / QK8_1);

    const int iters = 50;
    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    // Clock-ramp aware: the GPU drops to a low DPM state (SCLK down to ~10 MHz)
    // between kernels, so a short warmup measures the ramp, not the kernel.
    // Warm up until ~300 ms of GPU time has elapsed, then time enough
    // iterations to be stable.
    auto time_cfg = [&](rdna4::MatvecConfig c) {
      (void)c;
      float warm = 0;
      while (warm < 300.0f) {
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < 20; ++i) {
          launch((int)tc.dt, d_w, d_a, d_y, nrows, ncols);
        }
        hipEventRecord(e1, nullptr);
        hipEventSynchronize(e1);
        float ms = 0;
        hipEventElapsedTime(&ms, e0, e1);
        warm += ms;
      }
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < iters; ++i) {
        launch((int)tc.dt, d_w, d_a, d_y, nrows, ncols);
      }
      hipEventRecord(e1, nullptr);
      hipEventSynchronize(e1);
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      return (double)ms / iters;
    };

    // Times exactly what ships (per-type shape + ILP), or a forced global
    // (ILP, PF) when the caller asked for one.
    const double ms = time_cfg(rdna4::matvec_default_config((int)tc.dt));
    hipEventDestroy(e0); hipEventDestroy(e1);
    const rdna4::MatvecConfig def = rdna4::matvec_default_config((int)tc.dt);
    const double def_gbs = (double)w.size() / (ms * 1e-3) / 1e9;
    bw_by_type[(int)tc.dt] = def_gbs;   // decode estimate uses the SHIPPED config
    total_bytes += (double)w.size();
    total_sec += (double)w.size() / (def_gbs * 1e9);
    std::printf("%-9s %-22s %6lld x %6lld  %2dx%d  %6.1f GB/s  (%.3f ms%s%s)\n", tc.name,
                info.name.c_str(), (long long)nrows, (long long)ncols, def.rows, def.wpr, def_gbs,
                ms, force_ilp > 0 ? ", forced ILP=" : "", force_ilp > 0 ? std::to_string(force_ilp).c_str() : "");
    hipFree(d_w); hipFree(d_a); hipFree(d_y); hipFree(d_x);
  }
  std::printf("\nbench total: %.2f GB of weights, %.2f ms -> %.1f GB/s aggregate\n",
              total_bytes / 1e9, total_sec * 1e3, total_bytes / total_sec / 1e9);

  // Decode estimate: one token reads every quantized weight once, so
  //   t_decode ~= sum over tensors of (bytes / measured bandwidth[type]).
  // Attention/GDN/norm/KV work is NOT included -> this is an upper bound.
  double est_sec = 0;
  double model_bytes = 0;
  int skipped = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    const auto dt = ld.dtype(i);
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape((int)dt, shape)) { skipped++; continue; }  // F32 etc.
    const double bw = bw_by_type[(int)dt];
    if (bw <= 0) { skipped++; continue; }
    const double bytes = (double)ld.tensor_bytes(i);
    model_bytes += bytes;
    est_sec += bytes / (bw * 1e9);
  }
  std::printf("decode estimate (matvec only): %.2f GB of weights in %.2f ms -> %.1f tok/s\n",
              model_bytes / 1e9, est_sec * 1e3, 1.0 / est_sec);

  // Per-type breakdown of the estimated decode time: shows which types to
  // optimise (the estimate is bytes / measured bandwidth for the largest
  // tensor of that type).
  {
    struct Row { const char *name; double bytes; double sec; double bw; };
    std::vector<Row> rows;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      const auto dt = ld.dtype(i);
      rdna4::MatvecShape sh{};
      if (!rdna4::matvec_shape((int)dt, sh)) continue;
      if (bw_by_type[(int)dt] <= 0) continue;
      bool found = false;
      for (auto &r : rows) {
        if (r.name == std::string(tc_name_of((int)dt))) { r.bytes += (double)ld.tensor_bytes(i); found = true; break; }
      }
      if (!found) {
        rows.push_back({tc_name_of((int)dt), (double)ld.tensor_bytes(i), 0, bw_by_type[(int)dt]});
      }
    }
    for (auto &r : rows) r.sec = r.bytes / (r.bw * 1e9);
    std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.sec > b.sec; });
    std::printf("  where the time goes:\n");
    for (const auto &r : rows) {
      std::printf("    %-9s %6.2f GB  %8.2f ms  %6.1f GB/s  %5.1f%% of decode\n", r.name,
                  r.bytes / 1e9, r.sec * 1e3, r.bw, 100.0 * r.sec / est_sec);
    }
  }
  std::printf("  (F32 / unsupported tensors skipped: %d; baseline Vulkan/RADV decode = 37-38 tok/s)\n",
              skipped);
  return 0;
}


// Tuning sweep: ILP/prefetch knobs on each type's shipping shape.
int run_bench_tune(rdna4::GgufLoader &ld) {
  std::printf("%-9s %-22s %6s   %s\n", "type", "tensor", "shape", "GB/s per (ILP,PF)");
  for (const TypeCase &tc : kCases) {
    std::size_t best = std::size_t(-1);
    std::uint64_t best_bytes = 0;
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape((int)tc.dt, shape)) continue;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if (ld.dtype(i) != tc.dt) continue;
      const auto &d = ld.meta().tensors[i].dims;
      if (d.size() != 2 || d[1] % shape.qk != 0 || d[0] <= 0) continue;
      if (ld.tensor_bytes(i) > best_bytes) { best_bytes = ld.tensor_bytes(i); best = i; }
    }
    if (best == std::size_t(-1)) { std::printf("%-9s (absent)\n", tc.name); continue; }
    const auto &info = ld.meta().tensors[best];
    const int64_t ncols = info.dims[1], nrows = info.dims[0];
    std::vector<std::uint8_t> w;
    std::string err;
    if (!ld.load_tensor(best, w, err)) continue;
    std::vector<float> x;
    fill_x(x, ncols, 4242u);
    void *d_w = nullptr; rdna4::block_q8_1 *d_a = nullptr; float *d_y = nullptr; float *d_x = nullptr;
    hipMalloc(&d_w, w.size());
    hipMalloc(&d_a, (std::size_t)(ncols / QK8_1) * sizeof(rdna4::block_q8_1));
    hipMalloc(&d_y, (std::size_t)nrows * sizeof(float));
    hipMalloc(&d_x, (std::size_t)ncols * sizeof(float));
    hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((ncols / QK8_1 + 3) / 4), 128>>>(d_x, d_a, ncols / QK8_1);

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    char line[256] = {0};
    double best_gbs = 0;
    int best_ilp = 0; bool best_pf = false;
    for (int ilp : {1, 2, 4}) {
      for (bool pf : {false, true}) {
        float warm = 0;
        while (warm < 200.0f) {  // clock ramp (see PLAN.md): short warmups lie
          hipEventRecord(e0, nullptr);
          for (int i = 0; i < 10; ++i)
            rdna4::matvec_launch_tuned((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr, ilp, pf);
          hipEventRecord(e1, nullptr);
          hipEventSynchronize(e1);
          float ms = 0; hipEventElapsedTime(&ms, e0, e1); warm += ms;
        }
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < 30; ++i)
          rdna4::matvec_launch_tuned((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr, ilp, pf);
        hipEventRecord(e1, nullptr);
        hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); ms /= 30;
        const double gbs = (double)w.size() / (ms * 1e-3) / 1e9;
        if (gbs > best_gbs) { best_gbs = gbs; best_ilp = ilp; best_pf = pf; }
        char tmp[32];
        std::snprintf(tmp, sizeof(tmp), "%d%s:%.0f ", ilp, pf ? "p" : "-", gbs);
        std::strncat(line, tmp, sizeof(line) - std::strlen(line) - 1);
      }
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    const rdna4::MatvecConfig sc = rdna4::matvec_default_config((int)tc.dt);
    std::printf("%-9s %-22s %2dx%-2d   %s | best ILP=%d PF=%d\n", tc.name, info.name.c_str(),
                sc.rows, sc.wpr, line, best_ilp, best_pf ? 1 : 0);
    hipFree(d_w); hipFree(d_a); hipFree(d_y); hipFree(d_x);
  }
  return 0;
}


// Diagnostic mode: matvec vs a weights-only walk with the same access pattern.
int run_bench_read(rdna4::GgufLoader &ld) {
  std::printf("%-9s %-22s %10s %10s %10s %8s\n", "type", "tensor", "read GB/s", "matvec", "ratio", "shape");
  double tr = 0, tm = 0, tb = 0;
  for (const TypeCase &tc : kCases) {
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape((int)tc.dt, shape)) continue;
    std::size_t best = std::size_t(-1);
    std::uint64_t best_bytes = 0;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if (ld.dtype(i) != tc.dt) continue;
      const auto &d = ld.meta().tensors[i].dims;
      if (d.size() != 2 || d[1] % shape.qk != 0 || d[0] <= 0) continue;
      if (ld.tensor_bytes(i) > best_bytes) { best_bytes = ld.tensor_bytes(i); best = i; }
    }
    if (best == std::size_t(-1)) continue;
    const auto &info = ld.meta().tensors[best];
    const int64_t ncols = info.dims[1], nrows = info.dims[0];
    std::vector<std::uint8_t> w;
    std::string err;
    if (!ld.load_tensor(best, w, err)) continue;
    std::vector<float> x;
    fill_x(x, ncols, 777u);
    void *d_w = nullptr; rdna4::block_q8_1 *d_a = nullptr; float *d_y = nullptr; float *d_x = nullptr;
    hipMalloc(&d_w, w.size());
    hipMalloc(&d_a, (std::size_t)(ncols / QK8_1) * sizeof(rdna4::block_q8_1));
    hipMalloc(&d_y, (std::size_t)nrows * sizeof(float));
    hipMalloc(&d_x, (std::size_t)ncols * sizeof(float));
    hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
    hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((ncols / QK8_1 + 3) / 4), 128>>>(d_x, d_a, ncols / QK8_1);

    hipEvent_t e0, e1;
    hipEventCreate(&e0); hipEventCreate(&e1);
    auto timeit = [&](bool ro) {
      float warm = 0;
      while (warm < 250.0f) {
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < 10; ++i) {
          if (ro) rdna4::matvec_launch_read_only((int)tc.dt, d_w, d_y, nrows, ncols, nullptr);
          else rdna4::matvec_launch((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr);
        }
        hipEventRecord(e1, nullptr); hipEventSynchronize(e1);
        float ms = 0; hipEventElapsedTime(&ms, e0, e1); warm += ms;
      }
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < 30; ++i) {
        if (ro) rdna4::matvec_launch_read_only((int)tc.dt, d_w, d_y, nrows, ncols, nullptr);
        else rdna4::matvec_launch((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr);
      }
      hipEventRecord(e1, nullptr); hipEventSynchronize(e1);
      float ms = 0; hipEventElapsedTime(&ms, e0, e1);
      return (double)ms / 30.0;
    };
    const double ms_ro = timeit(true);
    const double ms_mv = timeit(false);
    // __launch_bounds__ sweep: does capping registers (more warps/CU) help?
    char mb_line[128] = {0};
    for (int mb : {0, 2, 3, 4, 6}) {
      float warm = 0;
      while (warm < 150.0f) {
        hipEventRecord(e0, nullptr);
        for (int i = 0; i < 10; ++i)
          rdna4::matvec_launch_minb((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr, mb);
        hipEventRecord(e1, nullptr); hipEventSynchronize(e1);
        float w = 0; hipEventElapsedTime(&w, e0, e1); warm += w;
      }
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < 30; ++i)
        rdna4::matvec_launch_minb((int)tc.dt, d_w, d_a, d_y, nrows, ncols, nullptr, mb);
      hipEventRecord(e1, nullptr); hipEventSynchronize(e1);
      float m2 = 0; hipEventElapsedTime(&m2, e0, e1); m2 /= 30;
      char tmp[24];
      std::snprintf(tmp, sizeof(tmp), "%d:%.0f ", mb, (double)w.size() / (m2 * 1e-3) / 1e9);
      std::strncat(mb_line, tmp, sizeof(mb_line) - std::strlen(mb_line) - 1);
    }
    hipEventDestroy(e0); hipEventDestroy(e1);
    const double gro = (double)w.size() / (ms_ro * 1e-3) / 1e9;
    const double gmv = (double)w.size() / (ms_mv * 1e-3) / 1e9;
    const rdna4::MatvecConfig sc = rdna4::matvec_default_config((int)tc.dt);
    hipFuncAttributes at{};
    int regs = 0, maxthr = 0;
    if (rdna4::matvec_kernel_attrs((int)tc.dt, at)) { regs = at.numRegs; maxthr = at.maxThreadsPerBlock; }
    const int threads = sc.rows * sc.wpr * 32;
    const int warps_per_cu = regs > 0 ? (regs <= 32 ? 64 : (regs <= 64 ? 32 : 16)) : 0;
    std::printf("%-9s %-22s %10.1f %10.1f %9.2fx %5dx%d  regs=%3d  minb[%s]\n",
                tc.name, info.name.c_str(), gro, gmv, gro / gmv, sc.rows, sc.wpr, regs, mb_line);
    tr += (double)w.size() / (gro * 1e9);
    tm += (double)w.size() / (gmv * 1e9);
    tb += (double)w.size();
    hipFree(d_w); hipFree(d_a); hipFree(d_y); hipFree(d_x);
  }
  std::printf("\nweights only: %.1f GB/s   matvec: %.1f GB/s   (read-only ceiling vs shipped kernel)\n",
              tb / tr / 1e9, tb / tm / 1e9);
  return 0;
}


// ---------------------------------------------------------------------------
// A/B mode: interleaved paired timing of iq3_s kernel variants, in ONE process,
// so that clock drift (this GPU varies ~20% run-to-run) cancels out.
// Variant 0 = shipping, 1 = diag no-sign, 2 = diag no-lookup, 3 = lin candidate.
// Diagnostic variants are wrong on purpose; only their TIMING is meaningful, and
// candidates (3) are compared numerically against variant 0.
// ---------------------------------------------------------------------------
int run_bench_ab(rdna4::GgufLoader &ld, int rounds, int dt) {
  // largest iq3_s tensor
  rdna4::MatvecShape shape{};
  if (!rdna4::matvec_shape(dt, shape)) return 1;
  std::size_t best = std::size_t(-1);
  std::uint64_t best_bytes = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    if ((int)ld.dtype(i) != dt) continue;
    const auto &d = ld.meta().tensors[i].dims;
    if (d.size() != 2 || d[1] % shape.qk != 0 || d[0] <= 0) continue;
    if (ld.tensor_bytes(i) > best_bytes) { best_bytes = ld.tensor_bytes(i); best = i; }
  }
  if (best == std::size_t(-1)) { std::printf("no usable iq3_s tensor\n"); return 1; }
  const auto &info = ld.meta().tensors[best];
  const int64_t ncols = info.dims[1], nrows = info.dims[0];
  std::printf("A/B on %s %s  %lld x %lld  (%.1f MB)\n", tc_name_of(dt), info.name.c_str(), (long long)nrows,
              (long long)ncols, ld.tensor_bytes(best) / 1e6);

  std::vector<std::uint8_t> w;
  std::string err;
  if (!ld.load_tensor(best, w, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
  std::vector<float> x;
  fill_x(x, ncols, 31337u);
  void *d_w = nullptr; rdna4::block_q8_1 *d_a = nullptr; float *d_y = nullptr; float *d_x = nullptr;
  hipMalloc(&d_w, w.size());
  hipMalloc(&d_a, (std::size_t)(ncols / QK8_1) * sizeof(rdna4::block_q8_1));
  hipMalloc(&d_y, (std::size_t)nrows * sizeof(float));
  hipMalloc(&d_x, (std::size_t)ncols * sizeof(float));
  hipMemcpy(d_w, w.data(), w.size(), hipMemcpyHostToDevice);
  hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
  rdna4::quantize_q8_1_kernel<<<(unsigned)((ncols / QK8_1 + 3) / 4), 128>>>(d_x, d_a, ncols / QK8_1);

  const bool diag = (dt == 12);                 // iq3_s has 6 variants
  const bool xx = (dt == 9);                    // iq3_xxs has 4
  const int kVariants = diag ? 6 : (xx ? 4 : 2);
  const char *names[6] = {"shipping", "DIAG-no-sign", "perm+lin (cand)", "lin (candidate)",
                          "DIAG-no-lookup", "xoradd (cand)"};
  if (xx) {
    names[0] = "vendored"; names[1] = "perm2 (shipping)"; names[2] = "perm (mid)";
    names[3] = "DIAG-no-sign";
  }
  std::vector<std::vector<double>> times(kVariants);
  std::vector<std::vector<float>> out(kVariants);

  hipEvent_t e0, e1;
  hipEventCreate(&e0); hipEventCreate(&e1);
  auto av_of = [&](int v) { return (diag || xx) ? v : (v == 0 ? 0 : 1); };  // variant index == dispatch index
  auto measure = [&](int v) {
    const int av = av_of(v);
    hipEventRecord(e0, nullptr);
    for (int i = 0; i < 25; ++i) {
      if (!rdna4::matvec_launch_variant(dt, d_w, d_a, d_y, nrows, ncols, nullptr, av)) {
        std::fprintf(stderr, "variant launch failed: dt=%d variant=%d\n", dt, av);
        std::exit(3);
      }
    }
    hipEventRecord(e1, nullptr);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    return (double)ms / 25.0;
  };

  // clock ramp
  float warm = 0;
  while (warm < 300.0f) {
    hipEventRecord(e0, nullptr);
    for (int i = 0; i < 10; ++i) rdna4::matvec_launch_variant(dt, d_w, d_a, d_y, nrows, ncols, nullptr, 0);
    hipEventRecord(e1, nullptr); hipEventSynchronize(e1);
    float ms = 0; hipEventElapsedTime(&ms, e0, e1); warm += ms;
  }

  // capture each variant's output once (for the numeric check)
  for (int v = 0; v < kVariants; ++v) {
    const int avo = (diag || xx) ? v : (v == 0 ? 0 : 1);
    rdna4::matvec_launch_variant(dt, d_w, d_a, d_y, nrows, ncols, nullptr, avo);
    out[v].resize(nrows);
    hipMemcpy(out[v].data(), d_y, (std::size_t)nrows * sizeof(float), hipMemcpyDeviceToHost);
  }

  for (int r = 0; r < rounds; ++r) {
    // rotate the starting variant so no variant is always measured first
    for (int k = 0; k < kVariants; ++k) {
      const int v = (k + r) % kVariants;
      times[v].push_back(measure(v));
    }
  }
  hipEventDestroy(e0); hipEventDestroy(e1);

  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double gb = (double)w.size() / 1e9;
  const double base = median(times[0]);
  std::printf("\n%-16s %10s %10s %8s   %s\n", "variant", "median ms", "GB/s", "vs ship", "numeric vs shipping");
  for (int v = 0; v < kVariants; ++v) {
    const double m = median(times[v]);
    const Stats st = compare(out[0], out[v]);
    std::printf("%-16s %10.3f %10.1f %7.2fx   max|d|=%.3e rel-L2=%.3e %s\n", names[v], m,
                gb / (m * 1e-3), base / m, st.max_abs, st.rel_l2,
                v == 0 ? "" : (v == 1 ? (st.rel_l2 < 1e-5 ? "(matches)" : "(MISMATCH!)")
                                    : (st.rel_l2 < 1e-5 ? "(matches)" : "(expected to differ)")));
  }
  hipFree(d_w); hipFree(d_a); hipFree(d_y); hipFree(d_x);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-matvec-gpu <file.gguf> [nrows]\n");
    return 2;
  }
  const bool bench = (argc >= 3 && std::strcmp(argv[2], "--bench") == 0);
  const int nrows_req = (!bench && argc >= 3) ? std::atoi(argv[2]) : 8;

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open %s: %s\n", argv[1], err.c_str());
    return 1;
  }

  if (argc >= 3 && std::strcmp(argv[2], "--check-lds") == 0) {
    return run_check_lds(ld);
  }
  if (argc >= 3 && std::strcmp(argv[2], "--bench-ab") == 0) {
    return run_bench_ab(ld, argc >= 4 ? std::atoi(argv[3]) : 9, argc >= 5 ? std::atoi(argv[4]) : 12);
  }
  if (argc >= 3 && std::strcmp(argv[2], "--bench-read") == 0) {
    return run_bench_read(ld);
  }
  if (argc >= 3 && std::strcmp(argv[2], "--bench-tune") == 0) {
    return run_bench_tune(ld);
  }
  if (bench) {
    const int f_ilp = argc >= 4 ? std::atoi(argv[3]) : 0;
    const int f_pf = argc >= 5 ? std::atoi(argv[4]) : 0;
    return run_bench(ld, f_ilp, f_pf);
  }

  int failures = 0;
  int tested = 0;

  for (const TypeCase &tc : kCases) {
    rdna4::MatvecShape shape{};
    if (!rdna4::matvec_shape((int)tc.dt, shape)) {
      // SPEC 1.3: a type with no kernel must fail loudly, never fall back to CPU.
      std::printf("%-9s no matvec kernel (explicit failure, no CPU fallback)\n", tc.name);
      ++failures;
      continue;
    }

    // A usable tensor is 2-D with a full number of blocks per row.
    auto usable = [&](std::size_t i) {
      if (ld.dtype(i) != tc.dt) return false;
      const auto &d = ld.meta().tensors[i].dims;
      return d.size() == 2 && d[1] >= shape.qk && d[1] % shape.qk == 0 && d[0] > 0;
    };

    std::size_t idx = std::size_t(-1);
    if (tc.tensor) {
      const auto *t = ld.find(tc.tensor);
      if (t) {
        const std::size_t ci = (std::size_t)(t - ld.meta().tensors.data());
        if (usable(ci)) idx = ci;
      }
    }
    if (idx == std::size_t(-1)) {
      for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
        if (usable(i)) {
          idx = i;
          break;
        }
      }
    }
    if (idx == std::size_t(-1)) {
      std::printf("%-9s (absent, or no 2-D tensor with cols %% %d == 0)\n", tc.name, shape.qk);
      continue;
    }

    const auto &info = ld.meta().tensors[idx];
    const int64_t ncols = info.dims[1];
    const int64_t nrows = info.dims[0];
    const int64_t rows = std::min<int64_t>(nrows, nrows_req);

    // weights: raw quantized bytes for the first `rows` rows
    const std::uint64_t row_bytes = (std::uint64_t)(ncols / shape.qk) * shape.block_bytes;
    std::vector<std::uint8_t> wbytes;
    if (!ld.load_tensor_range(idx, 0, row_bytes * (std::uint64_t)rows, wbytes, err)) {
      std::fprintf(stderr, "%-9s load failed: %s\n", tc.name, err.c_str());
      ++failures;
      continue;
    }

    std::vector<float> x;
    fill_x(x, ncols, 12345u + (unsigned)tc.dt);

    // ---- GPU: quantize activations, then matvec -----------------------------
    const int64_t nq8 = ncols / QK8_1;
    void *d_w = nullptr;
    rdna4::block_q8_1 *d_a = nullptr;
    float *d_y = nullptr;
    float *d_x = nullptr;
    auto fail_alloc = [&](const char *what) {
      std::fprintf(stderr, "%-9s hipMalloc(%s) failed: %s\n", tc.name, what,
                   hipGetErrorString(hipGetLastError()));
      if (d_w) hipFree(d_w);
      if (d_a) hipFree(d_a);
      if (d_y) hipFree(d_y);
      if (d_x) hipFree(d_x);
      return true;
    };
    if (hipMalloc(&d_w, wbytes.size()) != hipSuccess) { if (fail_alloc("weights")) { ++failures; continue; } }
    if (hipMalloc(&d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1)) != hipSuccess) { if (fail_alloc("act")) { ++failures; continue; } }
    if (hipMalloc(&d_y, (std::size_t)rows * sizeof(float)) != hipSuccess) { if (fail_alloc("out")) { ++failures; continue; } }
    hipMemcpy(d_w, wbytes.data(), wbytes.size(), hipMemcpyHostToDevice);

    if (hipMalloc(&d_x, (std::size_t)ncols * sizeof(float)) != hipSuccess) {
      if (fail_alloc("x")) { ++failures; continue; }
    }
    hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((nq8 + 3) / 4), 128>>>(d_x, d_a, nq8);
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      std::fprintf(stderr, "%-9s quantize launch failed: %s\n", tc.name, hipGetErrorString(e));
      ++failures;
      continue;
    }

    if (!rdna4::matvec_launch((int)tc.dt, d_w, d_a, d_y, rows, ncols, nullptr)) {
      std::fprintf(stderr, "%-9s matvec launch failed\n", tc.name);
      ++failures;
      continue;
    }
    std::vector<float> y_gpu(rows);
    hipMemcpy(y_gpu.data(), d_y, (std::size_t)rows * sizeof(float), hipMemcpyDeviceToHost);

    // pull the quantized activations back so the CPU side uses identical data
    std::vector<rdna4::block_q8_1> a_host(nq8);
    hipMemcpy(a_host.data(), d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1), hipMemcpyDeviceToHost);

    // Activation cross-check (independent of the matvec): compare against
    // llama.cpp's quantize_row_q8_1_ref. d and qs must match bit-exactly; `s`
    // follows mmvq's convention (fp16 of the raw float sum) instead of ggml's
    // fp16(d*sum(qs)), so it is checked within fp16 precision and reported.
    {
      std::vector<rdna4::block_q8_1> a_ref(nq8);
      rdna4_cpu_quantize_q8_1(x.data(), a_ref.data(), ncols);
      long long d_bad = 0, qs_bad = 0, s_bad = 0;
      for (std::int64_t b = 0; b < nq8; ++b) {
        if ((a_host[b].ds & 0xFFFFu) != (a_ref[b].ds & 0xFFFFu)) ++d_bad;
        if (std::memcmp(a_host[b].qs, a_ref[b].qs, QK8_1) != 0) ++qs_bad;
        if ((a_host[b].ds >> 16) != (a_ref[b].ds >> 16)) ++s_bad;
      }
      const bool act_ok = (d_bad == 0 && qs_bad == 0 && s_bad == 0);
      std::printf("          activation q8_1 vs quantize_row_q8_1_ref: d=%lld qs=%lld s=%lld mismatches"
                  " (bit-exact)  %s\n",
                  d_bad, qs_bad, s_bad, act_ok ? "OK" : "MISMATCH");
      failures += act_ok ? 0 : 1;
      if (!act_ok) continue;
    }

    // ---- CPU reference ------------------------------------------------------
    std::vector<float> xq(ncols);
    for (int64_t b = 0; b < nq8; ++b) {
      const float d = rdna4::fp16_to_float((uint16_t)(a_host[b].ds & 0xFFFFu));
      for (int i = 0; i < QK8_1; ++i) xq[b * QK8_1 + i] = d * (float)a_host[b].qs[i];
    }
    std::vector<float> wf((std::size_t)rows * ncols);
    for (int64_t r = 0; r < rows; ++r) {
      if (!rdna4_cpu_dequant_row((int)tc.dt, wbytes.data() + (std::size_t)r * row_bytes,
                                 wf.data() + (std::size_t)r * ncols, ncols)) {
        std::fprintf(stderr, "%-9s no CPU oracle\n", tc.name);
        ++failures;
        break;
      }
    }
    std::vector<float> y_cpu(rows, 0.0f);
    for (int64_t r = 0; r < rows; ++r) {
      double acc = 0;
      for (int64_t c = 0; c < ncols; ++c) acc += (double)wf[(std::size_t)r * ncols + c] * xq[c];
      y_cpu[r] = (float)acc;
    }

    const Stats s = compare(y_cpu, y_gpu);
    // The kernel accumulates in f32 (per-thread partials + butterfly) while the
    // reference is f64, so the accumulation-order difference grows with the
    // number of terms. Scale the gate with the row length instead of using a
    // fixed bound that a 248320-column row can approach on its own (this test
    // compares 1-4 rows; a permutation of the thread->(block,slot) assignment is
    // mathematically invisible here - only coverage bugs move the number).
    const double tol = tc.tol_rel_l2 * std::max(1.0, std::sqrt((double)ncols / 256.0));
    const bool ok = s.rel_l2 < tol;
    std::printf("%-9s %-24s rows=%3lld cols=%7lld  max|d|=%.3e  rel-L2=%.3e (tol %.1e)  %s\n",
                tc.name, info.name.c_str(), (long long)rows, (long long)ncols, s.max_abs, s.rel_l2,
                tol, ok ? "OK" : "MISMATCH");
    failures += ok ? 0 : 1;
    ++tested;

    hipFree(d_w);
    hipFree(d_a);
    hipFree(d_y);
    hipFree(d_x);
  }

  std::printf("check-matvec-gpu: %s (%d types tested)\n", failures ? "FAILED" : "OK", tested);
  return failures ? 1 : 0;
}
