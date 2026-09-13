// M6 step 1 — does weight sharing across tokens actually scale?
//
// Batched prefill only pays off if N tokens can share one pass over the weights
// (per-token weight traffic drops by N). Before restructuring the graph for it,
// this measures the curve directly on real tensors: for each dominant quant type,
// N = 1,2,4,8,16 activation vectors, comparing
//   - N separate GEMV calls (the shipping decode path), and
//   - one batched call (matvec_kernel_batch),
// both bit-exactly and in time (ms per token, weight GB/s).
//
// usage: check-matmul-gpu <file.gguf> [n_tokens...]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"

namespace {

std::string human_bytes(std::uint64_t b) {
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%.1f MB", (double)b / 1e6);
  return buf;
}

// Fills the activation matrix with a deterministic pattern (values of the same
// magnitude as real post-norm activations).
void fill_act(std::vector<float> &a, int n_tokens, int ncols, unsigned seed) {
  a.resize((std::size_t)n_tokens * (std::size_t)ncols);
  for (std::size_t i = 0; i < a.size(); ++i) {
    unsigned h = (unsigned)(i * 2654435761u) ^ (seed * 2246822519u);
    h ^= h >> 13;
    h *= 1274126177u;
    a[i] = ((float)(h >> 8) / 8388608.0f) * 2.0f - 1.0f;
  }
}

struct Timing {
  double ms;
  double gbs;   // weight bytes / time
  double tps;   // tokens per second through this tensor
};

// Warm-up measured in *GPU time*, not iterations: this part drops to a deep DPM
// state when idle (SCLK 9-16 MHz), and the first type measured in a process
// otherwise reports a clock-ramp number instead of a kernel number (M2 "Passo 5";
// re-confirmed in M6 when the very first case came out 15x too slow).
Timing timeit(const std::function<void()> &fn, int reps) {
  // Sync inside the warm-up loop: without it the command queue absorbs thousands
  // of iterations and the wall clock measures launch throughput instead of GPU
  // time. That mistake queued ~20 minutes of GPU work in the first M6 run.
  const auto w0 = std::chrono::steady_clock::now();
  do {
    fn();
    hipDeviceSynchronize();
  } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() <
           300.0);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) fn();
  hipDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  Timing t{};
  t.ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
  return t;
}

struct Case {
  rdna4::DType dt;
  std::size_t tensor_index;
  const char *name;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-matmul-gpu <file.gguf> [n_tokens...]\n");
    return 2;
  }
  std::vector<int> ns;
  for (int i = 2; i < argc; ++i) ns.push_back(std::atoi(argv[i]));
  if (ns.empty()) ns = {1, 2, 4, 8, 16};

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(argv[1], err)) {
    std::fprintf(stderr, "gguf load failed: %s\n", err.c_str());
    return 1;
  }
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    std::fprintf(stderr, "no HIP device\n");
    return 1;
  }
  std::printf("device: %s  VRAM=%.2f GiB\n", prop.name, prop.totalGlobalMem / 1073741824.0);
  std::printf("N is the number of tokens sharing one weight pass; "
              "\"tok/s\" is the prefill rate this tensor alone would allow\n\n");

  // The types that dominate the decode cost (PLAN.md M2 "Passo 5").
  using rdna4::DType;
  const DType kTypes[] = {DType::IQ3_S,   DType::IQ4_XS, DType::IQ3_XXS, DType::Q4_K,
                          DType::Q5_K,    DType::Q3_K,   DType::Q6_K,    DType::IQ2_S,
                          DType::IQ2_XS,  DType::IQ2_XXS, DType::IQ1_S,  DType::Q2_K,
                          DType::Q8_0,    DType::IQ4_NL};
  int failures = 0;
  for (const DType want : kTypes) {
    // ALL tensors of that type with the same shape, so the timed loop streams more
    // than the L2 (64 MB on this part): timing one small tensor would let it live
    // in L2 across iterations and hide the DRAM traffic batching is meant to save.
    std::vector<std::size_t> idxs;
    std::int64_t want_cols = -1, want_rows = -1;
    for (std::size_t i = 0; i < loader.meta().tensors.size(); ++i) {
      if (loader.dtype(i) != want) continue;
      const auto &t = loader.meta().tensors[i];
      if (t.dims.size() != 2) continue;
      if (want_cols < 0) {
        want_cols = t.dims[0];
        want_rows = t.dims[1];
      }
      if (t.dims[0] == want_cols && t.dims[1] == want_rows) idxs.push_back(i);
      if (idxs.size() >= 16) break;
    }
    if (idxs.empty()) {
      std::printf("%-9s (absent from this file)\n", rdna4::dtype_name(want));
      continue;
    }
    const auto &info = loader.meta().tensors[idxs[0]];
    const int nc = (int)info.dims[0];
    const int nr = (int)info.dims[1];
    std::vector<void *> ws;
    std::uint64_t bytes_each = 0;
    for (std::size_t idx : idxs) {
      std::vector<std::uint8_t> bytes;
      if (!loader.load_tensor(idx, bytes, err)) {
        std::fprintf(stderr, "%-9s load failed: %s\n", rdna4::dtype_name(want), err.c_str());
        return 1;
      }
      void *d = nullptr;
      if (hipMalloc(&d, bytes.size()) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc failed\n");
        return 1;
      }
      hipMemcpy(d, bytes.data(), bytes.size(), hipMemcpyHostToDevice);
      ws.push_back(d);
      bytes_each = bytes.size();
    }
    const int ncols = nc, nrows = nr;
    const std::uint64_t wbytes = bytes_each * ws.size();
    std::printf("  (%zu tensors of this shape, %.0f MB total, exceeds L2)\n", ws.size(),
                (double)wbytes / 1e6);

    const int act_blocks = ncols / 32;  // q8_1 blocks per activation row
    const int max_n = ns.back();
    std::vector<float> act;
    fill_act(act, max_n, ncols, 7u);
    float *d_act_f = nullptr;
    rdna4::block_q8_1 *d_act_q = nullptr;
    float *d_ref = nullptr, *d_bat = nullptr;
    hipMalloc(&d_act_f, (std::size_t)max_n * ncols * sizeof(float));
    hipMalloc(&d_act_q, (std::size_t)max_n * act_blocks * sizeof(rdna4::block_q8_1));
    hipMalloc(&d_ref, (std::size_t)max_n * nrows * sizeof(float));
    hipMalloc(&d_bat, (std::size_t)max_n * nrows * sizeof(float));
    hipMemcpy(d_act_f, act.data(), act.size() * sizeof(float), hipMemcpyHostToDevice);
    // one quantize_q8_1 launch per row, exactly like proj() does per token
    for (int n = 0; n < max_n; ++n) {
      rdna4::quantize_q8_1_kernel<<<(act_blocks + 3) / 4, 128>>>(
          d_act_f + (std::size_t)n * ncols, d_act_q + (std::size_t)n * act_blocks, act_blocks);
    }
    if (hipGetLastError() != hipSuccess) {
      std::fprintf(stderr, "quantize_q8_1 launch failed\n");
      return 1;
    }

    std::printf("%-9s %-28s [%d x %d] %s\n", rdna4::dtype_name(want), info.name.c_str(), ncols,
                nrows, human_bytes(wbytes).c_str());
    for (int n : ns) {
      if (n > rdna4::matvec_batch_cap()) continue;
      // ---- correctness: batched vs N separate GEMV calls -------------------
      for (void *d_w : ws) {
        for (int t = 0; t < n; ++t) {
          if (!rdna4::matvec_launch((int)want, d_w, d_act_q + (std::size_t)t * act_blocks,
                                    d_ref + (std::size_t)t * nrows, nrows, ncols, nullptr)) {
            std::fprintf(stderr, "  matvec_launch failed\n");
            return 1;
          }
        }
      }
      const void *d_w = ws[0];  // the comparison below uses the last written tensor
      hipDeviceSynchronize();
      bool launched = true;
      if (n == 1) {
        hipMemcpy(d_bat, d_ref, (std::size_t)nrows * sizeof(float), hipMemcpyDeviceToDevice);
      } else {
        launched = rdna4::matvec_launch_batch((int)want, ws.back(), d_act_q, d_bat, nrows, ncols,
                                              act_blocks, n, nullptr);
      }
      if (!launched) {
        std::printf("  N=%-2d batched launch unsupported\n", n);
        continue;
      }
      hipDeviceSynchronize();
      std::vector<float> ref((std::size_t)n * nrows), bat((std::size_t)n * nrows);
      hipMemcpy(ref.data(), d_ref, ref.size() * sizeof(float), hipMemcpyDeviceToHost);
      hipMemcpy(bat.data(), d_bat, bat.size() * sizeof(float), hipMemcpyDeviceToHost);
      std::size_t exact = 0;
      double maxd = 0.0;
      for (std::size_t i = 0; i < ref.size(); ++i) {
        if (ref[i] == bat[i]) ++exact;
        maxd = std::max(maxd, (double)std::fabs((double)ref[i] - (double)bat[i]));
      }
      const bool bitexact = exact == ref.size();

      // ---- timing ---------------------------------------------------------
      const int reps = 20;
      const Timing tg = timeit(
          [&] {
            for (void *d_w : ws) {
              for (int t = 0; t < n; ++t) {
                rdna4::matvec_launch((int)want, d_w, d_act_q + (std::size_t)t * act_blocks,
                                     d_ref + (std::size_t)t * nrows, nrows, ncols, nullptr);
              }
            }
          },
          reps);
      const Timing tb = timeit(
          [&] {
            for (void *d_w : ws) {
              (void)rdna4::matvec_launch_batch((int)want, d_w, d_act_q, d_bat, nrows, ncols,
                                               act_blocks, n, nullptr);
            }
          },
          reps);
      const double wgib = (double)wbytes / 1073741824.0;
      const double gemv_tps = (double)n / (tg.ms / 1000.0);
      const double bat_tps = (double)n / (tb.ms / 1000.0);
      std::printf("  N=%-2d gemv %7.3f ms (%6.1f tok/s, %6.1f GB/s) | batch %7.3f ms "
                  "(%6.1f tok/s, %6.1f GB/s) | %s speedup %.2fx\n",
                  n, tg.ms, gemv_tps, wgib * n / (tg.ms / 1000.0), tb.ms, bat_tps,
                  wgib / (tb.ms / 1000.0), bitexact ? "BIT-EXACT" : "MISMATCH", bat_tps / gemv_tps);
      if (!bitexact) {
        std::fprintf(stderr, "  FAIL: N=%d batched output differs from N GEMV calls "
                             "(exact %zu/%zu, max|d|=%.3e)\n",
                     n, exact, ref.size(), maxd);
        ++failures;
      }
      if (n > 1 && bat_tps < gemv_tps) {
        std::printf("      NOTE: batching is SLOWER than per-token here\n");
      }
      if (n == 1) {
        // warm the GPU so the first measured N is not a cold-clock artifact
        for (int i = 0; i < 200; ++i) {
          (void)rdna4::matvec_launch_batch((int)want, ws.back(), d_act_q, d_bat, nrows, ncols,
                                           act_blocks, 4, nullptr);
        }
        hipDeviceSynchronize();
      }
    }
    for (void *d : ws) hipFree(d);
    hipFree(d_act_f);
    hipFree(d_act_q);
    hipFree(d_ref);
    hipFree(d_bat);
    std::printf("\n");
  }
  if (failures == 0) {
    std::printf("check-matmul-gpu: OK (bit-exact in every configuration tested)\n");
    return 0;
  }
  std::printf("check-matmul-gpu: FAILED (%d mismatch(es))\n", failures);
  return 1;
}
