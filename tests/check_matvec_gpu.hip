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
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/loader.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);

namespace {

using rdna4::DType;

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

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-matvec-gpu <file.gguf> [nrows]\n");
    return 2;
  }
  const int nrows_req = argc >= 3 ? std::atoi(argv[2]) : 8;

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open %s: %s\n", argv[1], err.c_str());
    return 1;
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
    hipMalloc(&d_w, wbytes.size());
    hipMalloc(&d_a, (std::size_t)nq8 * sizeof(rdna4::block_q8_1));
    hipMalloc(&d_y, (std::size_t)rows * sizeof(float));
    hipMemcpy(d_w, wbytes.data(), wbytes.size(), hipMemcpyHostToDevice);

    float *d_x = nullptr;
    hipMalloc(&d_x, (std::size_t)ncols * sizeof(float));
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
    const bool ok = s.rel_l2 < tc.tol_rel_l2;
    std::printf("%-9s %-24s rows=%3lld cols=%7lld  max|d|=%.3e  rel-L2=%.3e (tol %.0e)  %s\n",
                tc.name, info.name.c_str(), (long long)rows, (long long)ncols, s.max_abs, s.rel_l2,
                tc.tol_rel_l2, ok ? "OK" : "MISMATCH");
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
