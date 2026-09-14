// Measurement tool (worktree rdna4-wt-medicoes-gpu) — how lossy is each
// quantization format, measured on the REAL tensors of a real GGUF.
//
//   check-quant-precision-gpu <file.gguf> [--max-elems N] [--rows N] [--seed N] [--time]
//   check-quant-precision-gpu --cross <A.gguf> <B.gguf> [--max-elems N]
//
// Why not "perplexity per format": we have no F16/fp32 model of this network
// (only IQ3_S and IQ4_XS), so the error of a format is measured where it is
// actually injected — in the matvec — against a reference that does NOT depend
// on the format's own kernel: a double-precision dot product over the
// *dequantized* weights (rdna4_cpu_dequant_row, validated bit-exact against
// llama.cpp by tests/check_dequant_gpu.hip).
//
// Three numbers per type, which together are the whole error budget of a matvec:
//
//   total      y_engine  vs  W_f32 . x_f32      (what the token actually gets:
//                                                activation q8_1 + kernel rounding)
//   act        W_f32 . x_q8_1  vs  W_f32 . x_f32 (the activation half: q8_1)
//   kernel     y_engine  vs  W_f32 . x_q8_1      (the kernel half: the
//                                                integer/fp16 arithmetic of vec_dot)
//
// The weights' own quantization error is NOT in these numbers (both sides see
// the same dequantized weights) — that is what --cross measures instead, by
// comparing the same tensor dequantized from two different formats (an upper
// bound on their distance from the common original).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/fp16.h"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"
#include "rdna4/quants.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);
extern "C" void rdna4_cpu_quantize_q8_1(const float *x, void *vy, std::int64_t nelem);

namespace {

using rdna4::DType;

double now_ms() {
  return (double)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

const char *tc_name_of(int dt) {
  switch (dt) {
    case 1: return "q8_0";    case 2: return "q2_k";   case 3: return "q3_k";
    case 4: return "q4_k";    case 5: return "q5_k";   case 6: return "q6_k";
    case 7: return "iq2_xxs"; case 8: return "iq2_xs"; case 9: return "iq3_xxs";
    case 10: return "iq1_s";  case 11: return "iq4_nl"; case 12: return "iq3_s";
    case 13: return "iq2_s";  case 14: return "iq4_xs"; default: return "?";
  }
}

// Gaussian-ish activation with unit RMS. Real activations reaching a matvec are
// post-RMS-norm, i.e. ~N(0,1) per row, which is what the q8_1 quantizer sees in
// the engine. (tests/check_matvec_gpu.hip uses uniform +-1.5; the q8_1 error is
// scale-invariant, so both give the same relative numbers.)
void fill_gauss(std::vector<float> &x, std::int64_t n, unsigned seed) {
  x.resize((std::size_t)n);
  unsigned s = seed * 2654435761u + 1u;
  auto next_u = [&]() {
    s = s * 1664525u + 1013904223u;
    return (float)(s >> 8) / (float)(1u << 24);
  };
  for (std::int64_t i = 0; i < n; ++i) {
    const float u1 = std::fmax(next_u(), 1e-7f), u2 = next_u();
    x[(std::size_t)i] = std::sqrt(-2.0f * std::log(u1)) * std::cos(6.2831853f * u2);
  }
}

struct Stats {
  double rel_l2 = 0, max_abs = 0, rms_a = 0, rms_b = 0;
};

Stats compare(const std::vector<float> &a, const std::vector<float> &b) {
  Stats s;
  double num = 0, da = 0, db = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    da += (double)a[i] * (double)a[i];
    db += (double)b[i] * (double)b[i];
    s.max_abs = std::fmax(s.max_abs, std::fabs(d));
  }
  s.rel_l2 = std::sqrt(num / (da + db));
  s.rms_a = std::sqrt(da / (double)a.size());
  s.rms_b = std::sqrt(db / (double)b.size());
  return s;
}

struct TypeRow {
  std::string type, tensor;
  std::int64_t rows = 0, cols = 0;
  double bpw = 0;
  Stats total, act, kernel, xq;
  double gbs = 0;   // cache-resident throughput of this one tensor (see the note)
  double ms = 0;
  bool ok = false;
};

// The type inventory of a file: every dtype that appears, with the largest
// usable 2-D tensor of that type under the element cap.
std::vector<TypeRow> measure_all(rdna4::GgufLoader &ld, std::int64_t max_elems, std::int64_t max_rows,
                                 bool do_time) {
  std::vector<TypeRow> out;
  std::vector<int> seen;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    const int dt = (int)ld.dtype(i);
    if (dt == 0) continue;  // F32/other non-quant types are not formats of interest
    if (std::find(seen.begin(), seen.end(), dt) != seen.end()) continue;
    seen.push_back(dt);
  }
  std::sort(seen.begin(), seen.end());
  for (int dt : seen) {
    TypeRow row;
    row.type = tc_name_of(dt);
    // largest tensor of this type with <= max_elems elements (best statistics
    // for the same runtime); fall back to the largest available, then to any.
    std::size_t pick = std::size_t(-1);
    std::int64_t best_elems = -1;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if ((int)ld.dtype(i) != dt) continue;
      const auto &t = ld.meta().tensors[i];
      if (t.dims.size() != 2) continue;
      const std::int64_t cols = t.dims[0], rows = t.dims[1];
      if (cols % QK8_1 != 0 || rows <= 0) continue;
      const std::int64_t elems = cols * rows;
      if (elems > max_elems) continue;
      if (elems > best_elems) {
        best_elems = elems;
        pick = i;
      }
    }
    if (pick == std::size_t(-1)) {
      std::printf("%-9s (nenhum tensor 2-D com cols %% %d == 0 abaixo do limite)\n", row.type.c_str(),
                  QK8_1);
      out.push_back(row);
      continue;
    }
    const auto &info = ld.meta().tensors[pick];
    const std::int64_t ncols = info.dims[0];
    const std::int64_t nrows = std::min<std::int64_t>(info.dims[1], max_rows);
    row.tensor = info.name;
    row.rows = nrows;
    row.cols = ncols;
    const std::uint64_t tbytes = ld.tensor_bytes(pick);
    row.bpw = 8.0 * (double)tbytes / (double)(info.dims[0] * info.dims[1]);
    const std::uint64_t row_bytes = tbytes / (std::uint64_t)info.dims[1];
    std::string err;
    std::vector<std::uint8_t> wbytes;
    if (!ld.load_tensor_range(pick, 0, row_bytes * (std::uint64_t)nrows, wbytes, err)) {
      std::printf("%-9s load: %s\n", row.type.c_str(), err.c_str());
      out.push_back(row);
      continue;
    }

    std::vector<float> x;
    fill_gauss(x, ncols, 777u + (unsigned)dt);

    // GPU: the engine's own path (q8_1 activation -> matvec_launch).
    const std::int64_t nq8 = ncols / QK8_1;
    void *d_w = nullptr, *d_a = nullptr, *d_y = nullptr, *d_x = nullptr;
    if (hipMalloc(&d_w, wbytes.size()) != hipSuccess ||
        hipMalloc(&d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1)) != hipSuccess ||
        hipMalloc(&d_y, (std::size_t)nrows * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_x, (std::size_t)ncols * sizeof(float)) != hipSuccess) {
      std::printf("%-9s hipMalloc failed\n", row.type.c_str());
      if (d_w) (void)hipFree(d_w);
      if (d_a) (void)hipFree(d_a);
      if (d_y) (void)hipFree(d_y);
      if (d_x) (void)hipFree(d_x);
      out.push_back(row);
      continue;
    }
    (void)hipMemcpy(d_w, wbytes.data(), wbytes.size(), hipMemcpyHostToDevice);
    (void)hipMemcpy(d_x, x.data(), (std::size_t)ncols * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((nq8 + 3) / 4), 128>>>(
        (const float *)d_x, (rdna4::block_q8_1 *)d_a, nq8);
    if (!rdna4::matvec_launch(dt, d_w, (const rdna4::block_q8_1 *)d_a, (float *)d_y, nrows, ncols,
                              nullptr)) {
      std::printf("%-9s matvec_launch failed\n", row.type.c_str());
      (void)hipFree(d_w);
      (void)hipFree(d_a);
      (void)hipFree(d_y);
      (void)hipFree(d_x);
      out.push_back(row);
      continue;
    }
    std::vector<rdna4::block_q8_1> a_host((std::size_t)nq8);
    std::vector<float> y_gpu((std::size_t)nrows);
    (void)hipMemcpy(a_host.data(), d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1),
              hipMemcpyDeviceToHost);
    (void)hipMemcpy(y_gpu.data(), d_y, (std::size_t)nrows * sizeof(float), hipMemcpyDeviceToHost);

    // Cache-resident throughput of this tensor: max over rounds of one single
    // launch, with the DPM warm-up. It is NOT DRAM bandwidth (a single tensor
    // fits in the Infinity Cache) — see docs/medicoes-banda-e-gargalos.md for
    // the DRAM numbers; kept here only to relate bytes to time for the same
    // tensor whose precision is being reported.
    if (do_time) {
      for (int i = 0; i < 200; ++i) {
        (void)rdna4::matvec_launch(dt, d_w, (const rdna4::block_q8_1 *)d_a, (float *)d_y, nrows,
                                   ncols, nullptr);
      }
      (void)hipDeviceSynchronize();
      double best = 1e30;
      for (int r = 0; r < 5; ++r) {
        const double t0 = now_ms();
        for (int i = 0; i < 10; ++i) {
          (void)rdna4::matvec_launch(dt, d_w, (const rdna4::block_q8_1 *)d_a, (float *)d_y, nrows,
                                     ncols, nullptr);
        }
        (void)hipDeviceSynchronize();
        best = std::min(best, (now_ms() - t0) / 10.0);
      }
      row.ms = best;
      row.gbs = (double)wbytes.size() / (best / 1000.0) / 1e9;
    }

    // Activation dequantized the way the kernel consumes it (fp16 d * int8 qs).
    std::vector<float> xq((std::size_t)ncols);
    for (std::int64_t b = 0; b < nq8; ++b) {
      const float dd = rdna4::fp16_to_float((std::uint16_t)(a_host[(std::size_t)b].ds & 0xFFFFu));
      for (int i = 0; i < QK8_1; ++i) {
        xq[(std::size_t)(b * QK8_1 + i)] = dd * (float)a_host[(std::size_t)b].qs[i];
      }
    }
    row.xq = compare(xq, x);

    // Reference: double dot over the dequantized weights, two right-hand sides
    // (exact activation, and the q8_1 activation).
    std::vector<float> wrow((std::size_t)ncols);
    std::vector<float> y_ref((std::size_t)nrows), y_actq((std::size_t)nrows);
    bool cpu_ok = true;
    for (std::int64_t r = 0; r < nrows && cpu_ok; ++r) {
      if (!rdna4_cpu_dequant_row(dt, wbytes.data() + (std::size_t)r * row_bytes, wrow.data(), ncols)) {
        std::printf("%-9s sem oraculo CPU para este tipo\n", row.type.c_str());
        cpu_ok = false;
        break;
      }
      double acc = 0, accq = 0;
      for (std::int64_t c = 0; c < ncols; ++c) {
        const double w = (double)wrow[(std::size_t)c];
        acc += w * (double)x[(std::size_t)c];
        accq += w * (double)xq[(std::size_t)c];
      }
      y_ref[(std::size_t)r] = (float)acc;
      y_actq[(std::size_t)r] = (float)accq;
    }
    (void)hipFree(d_w);
    (void)hipFree(d_a);
    (void)hipFree(d_y);
    (void)hipFree(d_x);
    if (!cpu_ok) {
      out.push_back(row);
      continue;
    }
    row.total = compare(y_gpu, y_ref);
    row.act = compare(y_actq, y_ref);
    row.kernel = compare(y_gpu, y_actq);
    row.ok = true;
    out.push_back(row);
  }
  return out;
}

// Same tensor dequantized from two files: the distance between two FORMATS,
// which is the only measurement of the weight-side error available without an
// fp32 original.
int cross(const char *path_a, const char *path_b, std::int64_t max_elems) {
  rdna4::GgufLoader a, b;
  std::string err;
  if (!a.open(path_a, err)) { std::fprintf(stderr, "open A: %s\n", err.c_str()); return 1; }
  if (!b.open(path_b, err)) { std::fprintf(stderr, "open B: %s\n", err.c_str()); return 1; }
  std::printf("A: %s\nB: %s\n", path_a, path_b);
  std::printf("%-34s %-9s %-9s %8s %10s %10s\n", "tensor", "tipo A", "tipo B", "bpw A", "rel-L2", "max|d|");
  double worst = 0, agg_num = 0, agg_den = 0;
  int n = 0;
  for (std::size_t i = 0; i < a.n_tensors(); ++i) {
    const auto &ta = a.meta().tensors[i];
    const auto *tb = b.find(ta.name.c_str());
    if (!tb) continue;
    if (ta.dims.size() != 2 || tb->dims.size() != 2) continue;
    if (ta.dims != tb->dims) continue;
    const std::int64_t cols = ta.dims[0], rows = ta.dims[1];
    if (cols * rows > max_elems) continue;
    const int dta = (int)a.dtype(i);
    const std::size_t ib = (std::size_t)(tb - b.meta().tensors.data());
    const int dtb = (int)b.dtype(ib);
    if (dta == 0 || dtb == 0) continue;
    std::vector<std::uint8_t> wa, wb;
    if (!a.load_tensor(i, wa, err) || !b.load_tensor(ib, wb, err)) continue;
    const int64_t rca = (int64_t)(wa.size() / (std::size_t)rows);
    const int64_t rcb = (int64_t)(wb.size() / (std::size_t)rows);
    std::vector<float> ra((std::size_t)cols), rb((std::size_t)cols);
    double num = 0, den = 0, mx = 0;
    for (std::int64_t r = 0; r < rows; ++r) {
      if (!rdna4_cpu_dequant_row(dta, wa.data() + (std::size_t)r * rca, ra.data(), cols) ||
          !rdna4_cpu_dequant_row(dtb, wb.data() + (std::size_t)r * rcb, rb.data(), cols)) {
        num = den = 0;
        break;
      }
      for (std::int64_t c = 0; c < cols; ++c) {
        const double d = (double)ra[(std::size_t)c] - (double)rb[(std::size_t)c];
        num += d * d;
        den += (double)ra[(std::size_t)c] * ra[(std::size_t)c] + (double)rb[(std::size_t)c] * rb[(std::size_t)c];
        mx = std::fmax(mx, std::fabs(d));
      }
    }
    if (!(num > 0)) continue;
    const double rel = std::sqrt(num / den);
    worst = std::fmax(worst, rel);
    agg_num += num;
    agg_den += den;
    ++n;
    std::printf("%-34s %-9s %-9s %8.3f %10.3e %10.3e\n", ta.name.c_str(), tc_name_of(dta),
                tc_name_of(dtb), 8.0 * (double)wa.size() / (double)(cols * rows), rel, mx);
  }
  std::printf("\n%d tensores comparados; agregado rel-L2 %.3e, pior %.3e\n", n,
              agg_den > 0 ? std::sqrt(agg_num / agg_den) : 0.0, worst);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  std::int64_t max_elems = 512 * 1024 * 1024, max_rows = 2048;
  bool do_time = false;
  std::vector<const char *> files;
  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    if (!std::strcmp(a, "--cross")) {
      if (i + 2 >= argc) { std::fprintf(stderr, "--cross precisa de dois arquivos\n"); return 2; }
      return cross(argv[i + 1], argv[i + 2], max_elems);
    } else if (!std::strcmp(a, "--max-elems")) {
      if (i + 1 < argc) max_elems = std::atoll(argv[++i]);
    } else if (!std::strcmp(a, "--rows")) {
      if (i + 1 < argc) max_rows = std::atoll(argv[++i]);
    } else if (!std::strcmp(a, "--time")) {
      do_time = true;
    } else if (a[0] == '-') {
      std::fprintf(stderr, "argumento desconhecido %s\n", a);
      return 2;
    } else {
      files.push_back(a);
    }
  }
  if (files.empty()) {
    std::fprintf(stderr,
                 "usage: check-quant-precision-gpu <file.gguf> [--max-elems N] [--rows N] [--time]\n"
                 "       check-quant-precision-gpu --cross <A.gguf> <B.gguf>\n");
    return 2;
  }
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(files[0], err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  std::printf("== precisao por formato (vs matvec fp64 sobre os pesos desquantizados)\n");
  std::printf("modelo: %s\n", files[0]);
  std::printf("ativacao: gaussiana, RMS 1 (= saida de um RMS-norm), semente 777+dt\n");
  std::printf("limite: <= %lld elementos e <= %lld linhas por tipo\n", (long long)max_elems,
              (long long)max_rows);
  const std::vector<TypeRow> rows = measure_all(ld, max_elems, max_rows, do_time);
  std::printf("\n%-9s %-30s %6s %8s %11s %11s %11s %11s %9s %10s\n", "tipo", "tensor", "linhas",
              "bpw", "total relL2", "total max", "ativ relL2", "kernel relL2", "x_q8 relL2",
              "RMS(y)");
  for (const TypeRow &r : rows) {
    if (!r.ok) {
      std::printf("%-9s %-30s %6s %8s %11s %11s %11s %11s %9s %10s\n", r.type.c_str(),
                  r.tensor.empty() ? "-" : r.tensor.c_str(), "-", "-", "-", "-", "-", "-", "-",
                  "-");
      continue;
    }
    std::printf("%-9s %-30s %6lld %8.3f %11.3e %11.3e %11.3e %11.3e %9.3e %10.3f\n",
                r.type.c_str(), r.tensor.c_str(), (long long)r.rows, r.bpw, r.total.rel_l2,
                r.total.max_abs, r.act.rel_l2, r.kernel.rel_l2, r.xq.rel_l2, r.total.rms_a);
  }
  std::printf("\n%-9s %8s %10s %14s %12s\n", "tipo", "linhas", "ms/matvec", "GB/s (cache)", "M elems");
  for (const TypeRow &r : rows) {
    if (!r.ok || r.ms <= 0) continue;
    std::printf("%-9s %8lld %10.4f %14.1f %12.2f\n", r.type.c_str(), (long long)r.rows, r.ms, r.gbs,
                (double)(r.rows * r.cols) / 1e6);
  }
  std::printf("\n(total relL2 = erro que chega ao token; ativ relL2 = so a quantizacao q8_1 da\n"
              " ativacao; kernel relL2 = so a aritmetica inteira/fp16 do vec_dot, com a mesma\n"
              " ativacao; x_q8 relL2 = erro do vetor de ativacao antes de qualquer peso)\n");
  std::printf("GB/s (cache) = 1 tensor em loop fechado: mede Infinity Cache, NAO DRAM.\n");
  return 0;
}
