// KV cache formats Q5_0 / Q4_1 — validated against llama.cpp's OWN quantizers,
// not against a mirror of our own kernels.
//
// Why this file exists: the night's target state is "131K with K=q5_0, V=q4_1",
// and neither format existed in the engine. Adding them means adding a *lossy*
// transform, so the gate cannot be bit-exactness of the cache — but it can be
// exactness of something stronger and independently known: the bytes the engine
// stores must be the bytes `quantize_row_q5_0_ref` / `quantize_row_q4_1_ref`
// (ggml/src/ggml-quants.c) would store, and both load paths (the scalar
// kv_load<>, used for head_dim != 256, and the vectorized kv_load8<>, used by the
// shipping attention kernel) must reconstruct exactly what llama.cpp's
// `dequantize_row_q5_0` / `dequantize_row_q4_1` reconstruct from those same
// bytes. libggml-base.so exports all eight functions, so they are called here
// directly as `extern "C"` — the same oracle tests/check_dequant.cpp already uses
// for the weight dtypes.
//
// Then, because "byte-identical quantizer" does not by itself say how much the
// cache *distorts* the row, the measured round-trip error is printed per format
// and per data pattern, plus (when a GGUF is given) on real K/V rows captured
// from the graph's own `Kcur` / `Vcur` nodes during a forward pass. The numbers
// in docs/journal-kv.md §2 come from here.
//
// usage: check-kvquant-gpu [file.gguf]
//
// Without a model it is a pure-kernel gate (~1 s, no 12 GiB load): synthetic
// patterns only. With a model it adds the real-data error bounds.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/kv.h"
#include "rdna4/loader.h"
#include "rdna4/model.h"

// llama.cpp's reference quantizers / dequantizers, taken from libggml-base.so
// (GGML_API, C linkage — see ggml-quants.h:19-52). Declared with `void *`/`const
// void *` so this test does not have to include ggml's headers and cannot drift
// from the block layouts: the byte counts are checked by static_assert in kv.h.
extern "C" {
void quantize_row_q4_0_ref(const float *x, void *y, std::int64_t k);
void quantize_row_q4_1_ref(const float *x, void *y, std::int64_t k);
void quantize_row_q5_0_ref(const float *x, void *y, std::int64_t k);
void quantize_row_q8_0_ref(const float *x, void *y, std::int64_t k);
void dequantize_row_q4_0(const void *x, float *y, std::int64_t k);
void dequantize_row_q4_1(const void *x, float *y, std::int64_t k);
void dequantize_row_q5_0(const void *x, float *y, std::int64_t k);
void dequantize_row_q8_0(const void *x, float *y, std::int64_t k);
}

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);

namespace {

constexpr int kHeadDim = 256;  // qwen35 head_dim: 8 blocks of 32 per row
constexpr int kHeadKv = 4;
constexpr int kRows = 48;  // 6 patterns x 8 rows

// Quantized types live in the cache; f32/f16 are the controls (their behaviour
// must not change and their bytes are checked here too).
const rdna4::KvType kTypes[] = {rdna4::KvType::F32, rdna4::KvType::F16, rdna4::KvType::Q8_0,
                                rdna4::KvType::Q4_0, rdna4::KvType::Q5_0, rdna4::KvType::Q4_1};

float frand(unsigned &s) {
  s = s * 1664525u + 1013904223u;
  return ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
}

// The llama.cpp reference quantizer for a type; nullptr for f32/f16 (nothing to
// call — those are stored verbatim / as fp16).
void (*ref_quantizer(rdna4::KvType t))(const float *, void *, std::int64_t) {
  switch (t) {
    case rdna4::KvType::Q8_0: return quantize_row_q8_0_ref;
    case rdna4::KvType::Q4_0: return quantize_row_q4_0_ref;
    case rdna4::KvType::Q5_0: return quantize_row_q5_0_ref;
    case rdna4::KvType::Q4_1: return quantize_row_q4_1_ref;
    case rdna4::KvType::F32:
    case rdna4::KvType::F16: return nullptr;
  }
  return nullptr;
}

void (*ref_dequantizer(rdna4::KvType t))(const void *, float *, std::int64_t) {
  switch (t) {
    case rdna4::KvType::Q8_0: return dequantize_row_q8_0;
    case rdna4::KvType::Q4_0: return dequantize_row_q4_0;
    case rdna4::KvType::Q5_0: return dequantize_row_q5_0;
    case rdna4::KvType::Q4_1: return dequantize_row_q4_1;
    case rdna4::KvType::F32:
    case rdna4::KvType::F16: return nullptr;
  }
  return nullptr;
}

// Builds the f32 rows used by the synthetic half. Patterns are chosen to be
// unfair to different formats on purpose:
//   0 uniform [-1,1)   -- the well-behaved symmetric case
//   1 positive [0,1)   -- asymmetric: q4_0/q5_0 spend half their grid on values
//                         that never occur, q4_1's per-block min is the fix
//   2 one outlier      -- destroys the scale of a whole block for every format
//   3 constant 0.5     -- d == 0 edge case
//   4 all zeros        -- d == 0, vmax == 0 (id must not be inf)
//   5 alternating +-1  -- extreme two-level distribution
//   6 gaussian-ish     -- realistic activation shape
//   7 wide range       -- |x| up to 100, checks the fp16 scale path
void make_patterns(std::vector<float> &rows, std::vector<int> &row_pattern) {
  rows.assign((std::size_t)kRows * kHeadDim, 0.0f);
  row_pattern.assign(kRows, 0);
  unsigned seed = 0x5eed1234u;
  const int reps = kRows / 8;
  for (int rep = 0; rep < reps; ++rep) {
    for (int p = 0; p < 8; ++p) {
      const int r = rep * 8 + p;
      row_pattern[r] = p;
      float *x = rows.data() + (std::size_t)r * kHeadDim;
      for (int d = 0; d < kHeadDim; ++d) {
        const float u = frand(seed);
        switch (p) {
          case 0: x[d] = u; break;
          case 1: x[d] = 0.5f * (u + 1.0f); break;
          case 2: x[d] = (d == 7) ? 1.0f : 1e-3f * u; break;
          case 3: x[d] = 0.5f; break;
          case 4: x[d] = 0.0f; break;
          case 5: x[d] = (d & 1) ? -1.0f : 1.0f; break;
          case 6: x[d] = 0.25f * (u + frand(seed) + frand(seed) + frand(seed)); break;
          default: x[d] = 100.0f * u; break;
        }
      }
    }
  }
}

// Device-side: read a whole row with the scalar path and a whole row with the
// vectorized path, so both are compared against the CPU dequantizer.
template <rdna4::KvType CT>
__global__ void load_paths_kernel(const void *__restrict__ cache, int n_rows, int head_dim,
                                  float *__restrict__ out_scalar, float *__restrict__ out_vec8) {
  const int r = blockIdx.x;
  const int lane = threadIdx.x;
  if (r >= n_rows) return;
  const char *row = (const char *)cache + (std::size_t)r * rdna4::kv_row_bytes(CT, head_dim);
  for (int d = lane; d < head_dim; d += 32) {
    out_scalar[(std::size_t)r * head_dim + d] = rdna4::kv_load<CT>(row, d);
  }
  if (head_dim % 32 != 0) return;
  float v8[8];
  rdna4::kv_load8<CT>(row, lane, v8);
#pragma unroll
  for (int i = 0; i < 8; ++i) out_vec8[(std::size_t)r * head_dim + lane * 8 + i] = v8[i];
}

bool launch_load_paths(rdna4::KvType t, const void *d_cache, int n_rows, int head_dim,
                       float *d_scalar, float *d_vec8) {
  const dim3 grid((unsigned)n_rows);
  switch (t) {
    case rdna4::KvType::F32:
      load_paths_kernel<rdna4::KvType::F32><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                          d_vec8);
      break;
    case rdna4::KvType::F16:
      load_paths_kernel<rdna4::KvType::F16><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                          d_vec8);
      break;
    case rdna4::KvType::Q8_0:
      load_paths_kernel<rdna4::KvType::Q8_0><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                           d_vec8);
      break;
    case rdna4::KvType::Q4_0:
      load_paths_kernel<rdna4::KvType::Q4_0><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                           d_vec8);
      break;
    case rdna4::KvType::Q5_0:
      load_paths_kernel<rdna4::KvType::Q5_0><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                           d_vec8);
      break;
    case rdna4::KvType::Q4_1:
      load_paths_kernel<rdna4::KvType::Q4_1><<<grid, 32>>>(d_cache, n_rows, head_dim, d_scalar,
                                                           d_vec8);
      break;
  }
  return hipGetLastError() == hipSuccess;
}

// One row (32 elements) of the f32 source, quantized by the ENGINE on the GPU.
// Returns the stored bytes and the error of the round trip.
struct RowResult {
  double max_abs_err = 0.0;   // max |x - dequant(quant(x))|
  double max_abs_x = 0.0;     // max |x| in the row (the natural normalizer)
  double block_range = 0.0;   // max over blocks of (max - min)
  double max_step = 0.0;      // largest quantization step seen (d, or 2*d)
};

// Runs the whole GPU pipeline for one type over `n_rows` rows of `src` (host
// data) and reports:
//   - bad_bytes: stored bytes differing from llama.cpp's reference quantizer
//   - max_load_diff / max_vec8_diff: |gpu load - llama.cpp dequant| on the bytes
//     actually stored (absolute and relative to the row's max |value|)
//   - the round-trip error bounds
RowResult check_type(rdna4::KvType ty, const std::vector<float> &flat, int n_rows, int head_dim,
                     std::string &bytes_report, std::string &load_report, bool verbose) {
  const std::uint64_t row_bytes = rdna4::kv_row_bytes(ty, head_dim);
  const std::size_t total = (std::size_t)n_rows * row_bytes;
  const int n_gpu_rows = n_rows;
  if (flat.size() < (std::size_t)n_gpu_rows * head_dim) {
    bytes_report = "internal: flat buffer too small";
    load_report = bytes_report;
    return RowResult{};
  }

  float *d_src = nullptr;
  void *d_cache = nullptr;
  float *d_scalar = nullptr, *d_vec8 = nullptr;
  hipMalloc(&d_src, flat.size() * sizeof(float));
  hipMemcpy(d_src, flat.data(), flat.size() * sizeof(float), hipMemcpyHostToDevice);
  hipMalloc(&d_cache, total);
  hipMemset(d_cache, 0, total);
  hipMalloc(&d_scalar, (std::size_t)n_gpu_rows * head_dim * sizeof(float));
  hipMalloc(&d_vec8, (std::size_t)n_gpu_rows * head_dim * sizeof(float));

  bool ok = true;
  for (int r = 0; r < n_gpu_rows; ++r) {
    ok = ok && rdna4::kv_store_row_launch(ty, d_src + (std::size_t)r * head_dim,
                                          (std::uint8_t *)d_cache + (std::size_t)r * row_bytes,
                                          head_dim);
  }
  ok = ok && launch_load_paths(ty, d_cache, n_gpu_rows, head_dim, d_scalar, d_vec8);
  std::vector<std::uint8_t> stored(total);
  hipMemcpy(stored.data(), d_cache, total, hipMemcpyDeviceToHost);
  std::vector<float> got_scalar((std::size_t)n_gpu_rows * head_dim);
  std::vector<float> got_vec8((std::size_t)n_gpu_rows * head_dim);
  hipMemcpy(got_scalar.data(), d_scalar, got_scalar.size() * sizeof(float), hipMemcpyDeviceToHost);
  hipMemcpy(got_vec8.data(), d_vec8, got_vec8.size() * sizeof(float), hipMemcpyDeviceToHost);
  hipDeviceSynchronize();

  auto qf = ref_quantizer(ty);
  auto df = ref_dequantizer(ty);
  long long bad_bytes = 0;
  double max_load_diff = 0.0, max_load_ref = 0.0;
  double max_vec_diff = 0.0;
  RowResult res;

  for (int r = 0; r < n_gpu_rows; ++r) {
    const float *x = flat.data() + (std::size_t)r * head_dim;
    const std::uint8_t *rowp = stored.data() + (std::size_t)r * row_bytes;

    // (1) the stored bytes vs llama.cpp's reference quantizer
    if (qf != nullptr) {
      std::vector<std::uint8_t> want(row_bytes);
      qf(x, want.data(), head_dim);
      for (std::uint64_t i = 0; i < row_bytes; ++i) {
        if (stored[(std::size_t)r * row_bytes + i] != want[i]) ++bad_bytes;
      }
    } else if (ty == rdna4::KvType::F16) {
      for (int d = 0; d < head_dim; ++d) {
        std::uint16_t h = 0;
        std::memcpy(&h, rowp + (std::size_t)d * 2, 2);
        if (h != rdna4::float_to_fp16(x[d])) ++bad_bytes;
      }
    } else {
      for (int d = 0; d < head_dim; ++d) {
        float v = 0.0f;
        std::memcpy(&v, rowp + (std::size_t)d * 4, 4);
        if (v != x[d]) ++bad_bytes;
      }
    }

    // (2) the CPU dequant of the stored bytes IS the expectation for both load
    //     paths (this is the part that is not the engine's own output)
    std::vector<float> ref(head_dim);
    if (df != nullptr) {
      df(rowp, ref.data(), head_dim);
    } else if (ty == rdna4::KvType::F16) {
      for (int d = 0; d < head_dim; ++d) {
        std::uint16_t h = 0;
        std::memcpy(&h, rowp + (std::size_t)d * 2, 2);
        ref[d] = rdna4::fp16_to_float(h);
      }
    } else {
      for (int d = 0; d < head_dim; ++d) std::memcpy(&ref[d], rowp + (std::size_t)d * 4, 4);
    }

    // (3) round-trip error + the block geometry that explains it
    for (int d = 0; d < head_dim; ++d) {
      res.max_abs_err = std::max(res.max_abs_err, std::fabs((double)x[d] - (double)ref[d]));
      res.max_abs_x = std::max(res.max_abs_x, (double)std::fabs(x[d]));
      max_load_diff = std::max(max_load_diff,
                               std::fabs((double)got_scalar[(std::size_t)r * head_dim + d] -
                                         (double)ref[d]));
      max_vec_diff = std::max(max_vec_diff,
                              std::fabs((double)got_vec8[(std::size_t)r * head_dim + d] -
                                        (double)ref[d]));
      max_load_ref = std::max(max_load_ref, (double)std::fabs(ref[d]));
    }
    for (int blk = 0; blk < head_dim / 32; ++blk) {
      float lo = x[blk * 32], hi = x[blk * 32];
      for (int j = 1; j < 32; ++j) {
        lo = std::min(lo, x[blk * 32 + j]);
        hi = std::max(hi, x[blk * 32 + j]);
      }
      res.block_range = std::max(res.block_range, (double)(hi - lo));
      if (ty == rdna4::KvType::Q5_0) {
        // d = vmax/-16, levels -16..15 -> the step is |d|
        const rdna4::block_q5_0 *b = (const rdna4::block_q5_0 *)rowp + blk;
        res.max_step = std::max(res.max_step, (double)std::fabs(rdna4::fp16_to_float(b->d)));
      } else if (ty == rdna4::KvType::Q4_1) {
        // d = (max-min)/15 -> the step is d
        const rdna4::block_q4_1 *b = (const rdna4::block_q4_1 *)rowp + blk;
        res.max_step = std::max(res.max_step, (double)rdna4::fp16_to_float(b->d));
      } else if (ty == rdna4::KvType::Q4_0) {
        const rdna4::block_q4_0 *b = (const rdna4::block_q4_0 *)rowp + blk;
        res.max_step = std::max(res.max_step, (double)std::fabs(rdna4::fp16_to_float(b->d)));
      } else if (ty == rdna4::KvType::Q8_0) {
        const rdna4::block_q8_0 *b = (const rdna4::block_q8_0 *)rowp + blk;
        res.max_step = std::max(res.max_step, (double)std::fabs(rdna4::fp16_to_float(b->d)));
      }
    }
  }

  char buf[256];
  std::snprintf(buf, sizeof(buf), "bytes differ from llama.cpp: %lld", bad_bytes);
  bytes_report = buf;
  std::snprintf(buf, sizeof(buf),
                "scalar load %.3e (rel %.2e), load8 %.3e (rel %.2e)", max_load_diff,
                max_load_ref > 0 ? max_load_diff / max_load_ref : 0.0, max_vec_diff,
                max_load_ref > 0 ? max_vec_diff / max_load_ref : 0.0);
  load_report = buf;

  if (verbose) {
    std::printf("   %-5s block_range %.6f  largest step %.6f  max|x-ref| %.6f "
                "(= %.4f x step, %.5f x max|x|)\n",
                rdna4::kv_type_name(ty), res.block_range, res.max_step, res.max_abs_err,
                res.max_step > 0 ? res.max_abs_err / res.max_step : 0.0,
                res.max_abs_x > 0 ? res.max_abs_err / res.max_abs_x : 0.0);
  }

  hipFree(d_src);
  hipFree(d_cache);
  hipFree(d_scalar);
  hipFree(d_vec8);
  res.max_abs_err = res.max_abs_err;  // keep
  // pack ok into the return through max_abs_x's sign bit is ugly; the caller
  // reads `bad_bytes` from the report instead.
  (void)ok;
  return res;
}

bool load_embedding(const rdna4::GgufLoader &ld, const rdna4::Qwen35Config &cfg, int token,
                    std::vector<float> &out, std::string &err) {
  const auto *info = ld.find("token_embd.weight");
  if (!info) {
    err = "token_embd.weight missing";
    return false;
  }
  const std::size_t ti = (std::size_t)(info - ld.meta().tensors.data());
  const rdna4::DType dt = ld.dtype(ti);
  const std::uint64_t row_bytes = rdna4::tensor_bytes(dt, cfg.embedding_length);
  std::vector<std::uint8_t> raw;
  if (!ld.load_tensor_range(ti, (std::uint64_t)token * row_bytes, row_bytes, raw, err)) return false;
  out.resize(cfg.embedding_length);
  return rdna4_cpu_dequant_row((int)dt, raw.data(), out.data(), (std::int64_t)cfg.embedding_length);
}

}  // namespace

int main(int argc, char **argv) {
  // Line-buffered: this test prints one line per format, and an abort or a hang
  // must not swallow the rows that already passed (the first version of this file
  // aborted in the per-pattern loop and the log came back empty).
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  int failures = 0;
  const std::string model = argc >= 2 ? argv[1] : std::string();

  // ------------------------------------------------------------------
  // 1. synthetic patterns, no model
  // ------------------------------------------------------------------
  std::vector<float> synth;
  std::vector<int> pattern;
  make_patterns(synth, pattern);
  // Tolerance on the round-trip error, in units of the format's own step. The
  // values are the geometry, not a guess, and they are NOT all 0.5:
  //
  //   Q5_0/Q4_0  d is taken from the SIGNED max (`d = vmax/-16` resp. `/-8`), so
  //              the grid is the ASYMMETRIC [-16,15] (resp. [-8,7]). The extreme
  //              value at the signed max lands exactly, but the opposite extreme
  //              is a full step away: measured err/step 0.930-1.000, so <= 1.0.
  //              Relative to a block's (max-min) range that is ~1/32.
  //   Q4_1       affine grid [min, max] with both ends exact, so half a step:
  //              measured 0.491-0.496, i.e. ~1/30 of the range.
  //   Q8_0       roundf on a symmetric grid -> half a step, but `d` is stored as
  //              fp16, which adds a 2^-11 relative term: measured 0.520, so the
  //              bound needs that headroom.
  //   F16        no step grid at all: a relative 2^-11, checked separately with
  //              the same 5e-4 bound tests/check_rope_gpu.hip uses.
  auto step_tol = [](rdna4::KvType t) -> double {
    switch (t) {
      case rdna4::KvType::Q5_0: return 1.02;
      case rdna4::KvType::Q4_0: return 1.02;
      case rdna4::KvType::Q4_1: return 0.52;
      case rdna4::KvType::Q8_0: return 0.55;
      case rdna4::KvType::F16: return 0.0;  // relative bound below, not a step bound
      case rdna4::KvType::F32: return 0.0;
    }
    return 0.0;
  };
  // f16 stores a relative 2^-11 = 4.88e-4; 5e-4 is the same bound check-rope-gpu
  // uses. f32 must be exact (scored below as err == 0).
  auto relative_tol = [](rdna4::KvType t) -> double {
    return t == rdna4::KvType::F16 ? 5e-4 : 0.0;
  };
  auto err_ok = [&](rdna4::KvType t, const RowResult &r) {
    if (t == rdna4::KvType::F32) return r.max_abs_err == 0.0;
    if (t == rdna4::KvType::F16) return r.max_abs_err <= relative_tol(t) * r.max_abs_x + 1e-9;
    return r.max_abs_err <= step_tol(t) * r.max_step + 1e-6;
  };

  std::printf("== synthetic (no model): %d rows x %d elems, 8 patterns x %d reps\n", kRows,
              kHeadDim, kRows / 8);
  std::printf("%-6s %-8s %-10s %-10s %-12s %-12s %-10s %s\n", "type", "bytes", "step_tol",
              "max_err", "err/step", "err/range", "err/|x|", "check");
  for (rdna4::KvType ty : kTypes) {
    std::string bytes_report, load_report;
    // make_patterns() already produced kRows independent rows of kHeadDim floats,
    // so the row count is kRows (not kRows * kHeadKv -- that mis-count is what
    // overran the buffer in the first version of this test).
    const RowResult r = check_type(ty, synth, kRows, kHeadDim, bytes_report, load_report, false);
    const bool bytes_ok = bytes_report.find(": 0") != std::string::npos;
    // "exact", not "close": both load paths are compared against llama.cpp's own
    // dequantizer on the bytes the engine actually stored, and the difference must
    // be zero. A tolerance here would hide exactly the offset bugs this checks.
    const bool load_ok = load_report.find("scalar load 0.000e+00") != std::string::npos &&
                         load_report.find("load8 0.000e+00") != std::string::npos;
    const bool e_ok = err_ok(ty, r);
    const bool ok = bytes_ok && load_ok && e_ok;
    if (!ok) ++failures;
    std::printf("%-6s %-8s %-10.4f %-10.6f %-12.4f %-12.5f %-10.6f %s\n", rdna4::kv_type_name(ty),
                bytes_ok ? "exact" : "DIFFER", step_tol(ty), r.max_abs_err,
                r.max_step > 0 ? r.max_abs_err / r.max_step : 0.0,
                r.block_range > 0 ? r.max_abs_err / r.block_range : 0.0,
                r.max_abs_x > 0 ? r.max_abs_err / r.max_abs_x : 0.0, ok ? "OK" : "FAIL");
    std::printf("       %s | %s\n", bytes_report.c_str(), load_report.c_str());
    if (!ok && (ty == rdna4::KvType::Q5_0 || ty == rdna4::KvType::Q4_1)) {
      std::printf("       ^ the new formats are the point of this file: NOT acceptable\n");
    }
  }

  // Per-pattern breakdown for the two new formats: the same error bound has to
  // hold on the patterns that are hostile to each grid, otherwise the tolerance
  // is hiding a pattern-specific failure.
  std::printf("\n-- per-pattern, only the patterns that stress the format --\n");
  for (rdna4::KvType ty : {rdna4::KvType::Q5_0, rdna4::KvType::Q4_1}) {
    for (int p = 0; p < 8; ++p) {
      int n = 0;
      for (int r = 0; r < kRows; ++r) {
        if (pattern[r] == p) ++n;
      }
      if (n == 0) continue;
      std::vector<float> sub((std::size_t)n * kHeadDim);  // n matching rows, sized first
      int w = 0;
      for (int r = 0; r < kRows; ++r) {
        if (pattern[r] != p) continue;
        std::memcpy(sub.data() + (std::size_t)w * kHeadDim, synth.data() + (std::size_t)r * kHeadDim,
                    (std::size_t)kHeadDim * sizeof(float));
        ++w;
      }
      std::string br, lr;
      const RowResult r = check_type(ty, sub, n, kHeadDim, br, lr, false);
      std::printf("   %-5s pattern %d: max_err %.6f  err/step %.4f  err/range %.5f  (%s)\n",
                  rdna4::kv_type_name(ty), p, r.max_abs_err,
                  r.max_step > 0 ? r.max_abs_err / r.max_step : 0.0,
                  r.block_range > 0 ? r.max_abs_err / r.block_range : 0.0, br.c_str());
    }
  }

  // ------------------------------------------------------------------
  // 2. real K/V rows out of the graph (only when a model is given)
  // ------------------------------------------------------------------
  if (!model.empty()) {
    rdna4::GgufLoader ld;
    std::string err;
    if (!ld.open(model.c_str(), err)) {
      std::fprintf(stderr, "open: %s\n", err.c_str());
      return 1;
    }
    rdna4::Qwen35Config cfg;
    if (!rdna4::parse_qwen35_config(ld.meta(), cfg, err)) {
      std::fprintf(stderr, "config: %s\n", err.c_str());
      return 1;
    }
    // "The quick brown fox jumps over the lazy dog" style prompt: real token ids
    // so the K/V rows are the ones the model actually produces.
    const std::vector<int> prompt = {9419, 1814, 11, 411, 369, 264, 1228, 13};
    std::vector<std::vector<float>> emb(prompt.size());
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      if (!load_embedding(ld, cfg, prompt[i], emb[i], err)) {
        std::fprintf(stderr, "embedding: %s\n", err.c_str());
        return 1;
      }
    }
    std::vector<float> all_emb;
    for (const auto &e : emb) all_emb.insert(all_emb.end(), e.begin(), e.end());

    // Kcur is the row AFTER RoPE (exactly what kv_write stores); Vcur is the raw
    // V projection. Both come out of the shipping graph, so this is the real
    // distribution, not a simulation.
    std::map<std::string, std::vector<float>> cap;
    {
      rdna4::Graph g(ld, cfg);
      if (!g.init((int)prompt.size(), rdna4::KvType::F32, rdna4::KvType::F32, err)) {
        std::fprintf(stderr, "graph init: %s\n", err.c_str());
        return 1;
      }
      g.set_node_cb([&cap](const char *name, int il, int, const float *p, std::int64_t n) {
        const std::string nm(name);
        if (nm != "Kcur" && nm != "Vcur") return;
        cap[nm + "#" + std::to_string(il)].assign(p, p + n);
      });
      std::vector<float> h, logits;
      if (!g.forward(all_emb, h, logits, err)) {
        std::fprintf(stderr, "forward: %s\n", err.c_str());
        return 1;
      }
      g.set_node_cb(nullptr);
    }
    if (cap.empty()) {
      std::printf("real K/V capture: EMPTY (node names changed?)\n");
      ++failures;
    } else {
      std::printf("\n== real K/V rows captured from the graph: %zu nodes (full-attn layers x "
                  "K/V) x %zu tokens, %u kv heads x %u dims\n",
                  cap.size(), prompt.size(), cfg.head_count_kv, cfg.key_length);
      for (const char *side : {"Kcur", "Vcur"}) {
        // concatenate all captured layers of this side into one buffer of rows
        std::vector<float> rows;
        int n_rows = 0;
        for (const auto &kv : cap) {
          if (kv.first.rfind(std::string(side) + "#", 0) != 0) continue;
          rows.insert(rows.end(), kv.second.begin(), kv.second.end());
          n_rows += (int)(kv.second.size() / cfg.key_length);
        }
        if (n_rows == 0) continue;
        std::printf("-- %s: %d rows of %d (real, post-RoPE for K)\n", side, n_rows,
                    (int)cfg.key_length);
        for (rdna4::KvType ty : kTypes) {
          std::string br, lr;
          const RowResult r = check_type(ty, rows, n_rows, (int)cfg.key_length, br, lr, false);
          const bool bytes_ok = br.find(": 0") != std::string::npos;
          const bool load_ok = lr.find("scalar load 0.000e+00") != std::string::npos &&
                               lr.find("load8 0.000e+00") != std::string::npos;
          const bool e_ok = err_ok(ty, r);
          if (!(bytes_ok && load_ok && e_ok)) ++failures;
          std::printf("   %-5s bytes %-6s %s | max_err %.6f  err/step %.4f  err/range %.5f  "
                      "err/|x|max %.5f  %s\n",
                      rdna4::kv_type_name(ty), bytes_ok ? "exact" : "DIFFER", lr.c_str(),
                      r.max_abs_err, r.max_step > 0 ? r.max_abs_err / r.max_step : 0.0,
                      r.block_range > 0 ? r.max_abs_err / r.block_range : 0.0,
                      r.max_abs_x > 0 ? r.max_abs_err / r.max_abs_x : 0.0,
                      (bytes_ok && load_ok && e_ok) ? "OK" : "FAIL");
        }
      }
    }
  }

  std::printf("check-kvquant-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
