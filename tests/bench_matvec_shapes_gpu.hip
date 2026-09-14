// M8 (ROCM study) — shape-aware matvec bench: what the shipping GEMV really
// costs per tensor, and whether ROWS per CTA / the register budget are free money.
//
// Why this tool exists (the three methodology gaps it closes):
//
//  1. `check-matvec-gpu --bench` measures ONE tensor per type — the largest of
//     that type — and applies the resulting GB/s to every tensor of the type.
//     Two problems with that: (a) the largest tensor of a type is often NOT
//     read per token (`token_embd.weight` is 556 MB of q3_K and decode reads a
//     single 2.2 KB row of it, the q6_K block of the MTP layer is never
//     executed), so a type's "share of the model" is not its share of the
//     per-token traffic; (b) it measures with dims[0]/dims[1] SWAPPED relative
//     to how `Graph::proj` calls the kernel, so the measured shape is not the
//     shape that ships.
//  2. Bandwidth here depends on the shape (rows x blocks-per-row): 48x160 (the
//     ssm_beta/ssm_alpha q8_0 matrices) can never fill 64 CUs with 24 CTAs,
//     while 17408x20 can. Timing every (type, shape) class of the real
//     inventory is the only way to know where the trunk matvec time goes.
//  3. A per-launch `hipEventRecord` + synchronize measures the DPM gap between
//     kernels as well as the kernel (this card drops to ~10 MHz SCLK when idle,
//     PLAN.md M2), which dominates the smallest tensors: a 261 KB q8_0 matrix
//     "costs" 10 us timed that way. The REPLAY mode below launches the whole
//     per-token set back-to-back with ONE event pair, i.e. exactly how the
//     graph calls them, and reports that number next to the sum of the
//     per-tensor ones. The difference is the measurement's own floor.
//
// usage:
//   bench-matvec-shapes-gpu <model.gguf> [--budget-mib N] [--reps N]
//                           [--rows a,b,c] [--minb a,b,c] [--no-ab]
//     --budget-mib  weight data to stage (default 12000 -> the whole inventory)
//     --rows        A/B the rows-per-CTA knob (bit-exact: it only moves which CTA
//                   computes which row)
//     --minb        A/B the __launch_bounds__ register budget (bit-exact: it
//                   only caps the register allocator)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"

namespace {

using rdna4::block_q8_1;

const char *dt_name(int dt) {
  switch (dt) {
    case 1: return "q8_0";    case 2: return "q2_k";   case 3: return "q3_k";
    case 4: return "q4_k";    case 5: return "q5_k";   case 6: return "q6_k";
    case 7: return "iq2_xxs"; case 8: return "iq2_xs"; case 9: return "iq3_xxs";
    case 10: return "iq1_s";  case 11: return "iq4_nl"; case 12: return "iq3_s";
    case 13: return "iq2_s";  case 14: return "iq4_xs"; default: return "?";
  }
}

struct Tensor {
  std::string name;
  int dt = 0;
  std::int64_t nrows = 0;  // output rows
  std::int64_t ncols = 0;  // reduction dim (== dims[0] of the GGUF tensor)
  std::int64_t bpr = 0;    // blocks per row (ncols / qk)
  double bytes = 0;
  void *d_w = nullptr;
};

void fill_x(std::vector<float> &x, std::int64_t n, unsigned seed) {
  x.resize(n);
  unsigned s = seed;
  for (std::int64_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 3.0f;
  }
}

// The launch used by every A/B variant:
//   kind 0 = shipping (per-type rows/WPR/ILP)
//   1 = forced rows, 2 = forced __launch_bounds__ min-blocks
//   3 = single-accumulator unroll (bit-exact MLP), 4 = L2 prefetch
//   5 = forced ILP (independent accumulators; changes the summation order)
//   6 = the shipping launch AGAIN, as its own variant -> the noise floor of this
//       very bench (same kernel, same shapes, measured independently)
//   7 = the LANDED tuning candidate: UNROLL=2 for the types whose shipping ILP is
//       already 1 (bit-exact: same ops, same order, more loads in flight) and the
//       shipping launch for every other type.
inline bool launch_variant(int kind, int value, const Tensor &t, const block_q8_1 *d_a, float *d_y) {
  if (kind == 0 || kind == 6)
    return rdna4::matvec_launch(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  if (kind == 1)
    return rdna4::matvec_launch_rows(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value);
  if (kind == 3)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value,
                                       /*pf=*/false);
  if (kind == 4)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, 1,
                                       /*pf=*/true);
  if (kind == 5)
    return rdna4::matvec_launch_tuned(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value,
                                      /*pf=*/false);
  if (kind == 8)  // LUT do tipo em LDS (mesma forma; ver matvec.cuh / journal §2)
    return rdna4::matvec_launch_lut_lds(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  if (kind == 9)  // LUT em LDS com UNROLL forcado (mede se a LDS muda o otimo de unroll)
    return rdna4::matvec_launch_lut_lds_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr,
                                               value);
  if (kind == 7)
    return rdna4::matvec_default_ilp(t.dt) == 1
               ? rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, 2,
                                             /*pf=*/false)
               : rdna4::matvec_launch(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  return rdna4::matvec_launch_minb(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value);
}

__global__ void null_kernel(float *p) {
  if (p) *p = 1.0f;
}

struct EvPair {
  hipEvent_t e0 = nullptr, e1 = nullptr;
  EvPair() {
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
  }
  ~EvPair() {
    if (e0) (void)hipEventDestroy(e0);
    if (e1) (void)hipEventDestroy(e1);
  }
  double operator()(const std::function<void()> &fn) {
    hipEventRecord(e0, nullptr);
    fn();
    hipEventRecord(e1, nullptr);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    return ms;
  }
};

struct Variant {
  const char *name = "ship";
  int kind = 0;   // 0 = ship, 1 = rows, 2 = minb, 3 = unroll, 4 = prefetch, 5 = ilp,
                  // 6 = ship again (noise floor), 7 = landed tuning candidate
  int value = 0;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bench-matvec-shapes-gpu <model.gguf> [--budget-mib N] [--reps N] "
                 "[--rows a,b,c] [--minb a,b,c] [--unroll 2,4] [--pf] [--ilp 1,2,4] "
                 "[--repeat-ship] [--land] [--lut-lds] [--lut-lds-unroll 1,2,4] [--no-ab]\n"
                 "       [--batch 1,2,4,8,16]  (sweep tokens-per-weight-pass; N=1 = GEMV)\n"
                 "       [--batch-unroll 2,4]  (bit-exact MLP knob inside the batch kernel)\n"
                 "       [--batch-cap]         (same pass on two streams: spare capacity?)\n");
    return 1;
  }
  double budget_mib = 12000.0;
  int reps = 4;
  bool do_ab = true;
  bool pf_only = false;
  bool repeat_ship = false;
  bool land = false;
  bool lut_lds = false;
  std::vector<int> lut_lds_unroll;
  std::vector<int> row_variants, minb_variants, unroll_variants, ilp_variants;
  std::vector<int> batch_list, unroll_list;
  bool cap_probe = false;
  for (int i = 2; i < argc; ++i) {
    auto list = [&](std::vector<int> &out) {
      const char *v = argv[++i];
      while (*v) {
        out.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    };
    if (std::strcmp(argv[i], "--budget-mib") == 0 && i + 1 < argc) {
      budget_mib = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
      list(row_variants);
    } else if (std::strcmp(argv[i], "--minb") == 0 && i + 1 < argc) {
      list(minb_variants);
    } else if (std::strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) {
      list(unroll_variants);
    } else if (std::strcmp(argv[i], "--ilp") == 0 && i + 1 < argc) {
      list(ilp_variants);
    } else if (std::strcmp(argv[i], "--pf") == 0) {
      unroll_variants.push_back(1);
      pf_only = true;
    } else if (std::strcmp(argv[i], "--repeat-ship") == 0) {
      repeat_ship = true;
    } else if (std::strcmp(argv[i], "--land") == 0) {
      land = true;
    } else if (std::strcmp(argv[i], "--lut-lds") == 0) {
      lut_lds = true;
    } else if (std::strcmp(argv[i], "--lut-lds-unroll") == 0 && i + 1 < argc) {
      list(lut_lds_unroll);  // aceita "1,2,4" (o atoi direto leria so o 1)
    } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
      // --batch a,b,c: sweep the number of tokens that share one weight pass
      // (prefill / MTP). N=1 in the list times the GEMV path as the reference.
      list(batch_list);
    } else if (std::strcmp(argv[i], "--batch-unroll") == 0 && i + 1 < argc) {
      // UNROLL forcado no lote (bit-exato), so' nos tipos com kMtIlp == 1.
      list(unroll_list);
    } else if (std::strcmp(argv[i], "--batch-cap") == 0) {
      // Sondagem de capacidade: a mesma passagem em duas streams (sobreposicao
      // possivel = latencia; 2x = throughput saturado).
      cap_probe = true;
    } else if (std::strcmp(argv[i], "--no-ab") == 0) {
      do_ab = false;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 1;
    }
  }
  if (reps < 1) reps = 1;
  if (!do_ab) {
    row_variants.clear();
    minb_variants.clear();
    unroll_variants.clear();
    ilp_variants.clear();
    repeat_ship = false;
    land = false;
    lut_lds = false;
    lut_lds_unroll.clear();
  }
  (void)pf_only;

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }

  // ---------------------------------------------------------------------
  // Inventory: exactly the tensors the decode path streams per token.
  // `token_embd` is skipped (one row per token), the MTP block blk.<block_count
  // - 1> is skipped (never executed by this engine), F32 has no matvec.
  // ---------------------------------------------------------------------
  std::vector<Tensor> all;
  double grand_total_bytes = 0, skipped_bytes = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    const auto &info = ld.meta().tensors[i];
    const std::string &name = info.name;
    rdna4::MatvecShape shape{};
    const int dt = (int)ld.dtype(i);
    if (name == "token_embd.weight" || name == "output_norm.weight" ||
        name.rfind("blk.64.", 0) == 0 || !rdna4::matvec_shape(dt, shape) ||
        info.dims.size() != 2 || (std::int64_t)info.dims[0] % shape.qk != 0) {
      skipped_bytes += (double)ld.tensor_bytes(i);
      continue;
    }
    Tensor t;
    t.name = name;
    t.dt = dt;
    t.ncols = (std::int64_t)info.dims[0];
    t.nrows = (std::int64_t)info.dims[1];
    t.bpr = t.ncols / shape.qk;
    t.bytes = (double)ld.tensor_bytes(i);
    grand_total_bytes += t.bytes;
    all.push_back(t);
  }
  std::printf("inventory: %zu tensors read per token, %.3f GB (%.3f GiB); skipped %.3f GB "
              "(token_embd / MTP / f32)\n",
              all.size(), grand_total_bytes / 1e9, grand_total_bytes / (1024.0 * 1024 * 1024),
              skipped_bytes / 1e9);

  std::sort(all.begin(), all.end(),
            [](const Tensor &a, const Tensor &b) { return a.bytes > b.bytes; });
  const double budget = budget_mib * 1024.0 * 1024.0;
  std::vector<Tensor> sel;
  double used = 0;
  for (const Tensor &t : all) {
    if (used + t.bytes > budget) continue;
    sel.push_back(t);
    used += t.bytes;
  }
  std::printf("staged: %zu tensors, %.3f GiB\n", sel.size(), used / (1024.0 * 1024 * 1024));

  float *d_y = nullptr;
  std::int64_t max_rows = 0;
  for (const Tensor &t : sel) max_rows = std::max(max_rows, t.nrows);
  if (hipMalloc(&d_y, (std::size_t)max_rows * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc d_y failed\n");
    return 1;
  }
  std::vector<std::int64_t> ncols_seen;
  for (const Tensor &t : sel)
    if (std::find(ncols_seen.begin(), ncols_seen.end(), t.ncols) == ncols_seen.end())
      ncols_seen.push_back(t.ncols);
  std::vector<std::pair<std::int64_t, block_q8_1 *>> acts;
  for (std::int64_t nc : ncols_seen) {
    std::vector<float> x;
    fill_x(x, nc, 999u);
    float *d_x = nullptr;
    block_q8_1 *d_a = nullptr;
    (void)hipMalloc(&d_x, (std::size_t)nc * sizeof(float));
    (void)hipMalloc(&d_a, (std::size_t)(nc / QK8_1) * sizeof(block_q8_1));
    (void)hipMemcpy(d_x, x.data(), (std::size_t)nc * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((nc / QK8_1 + 3) / 4), 128>>>(d_x, d_a, nc / QK8_1);
    hipDeviceSynchronize();
    (void)hipFree(d_x);
    acts.push_back({nc, d_a});
  }
  for (Tensor &t : sel) {
    std::vector<std::uint8_t> w;
    const auto *info = ld.find(t.name.c_str());
    if (!ld.load_tensor((std::size_t)(info - ld.meta().tensors.data()), w, err)) {
      std::fprintf(stderr, "%s: %s\n", t.name.c_str(), err.c_str());
      return 1;
    }
    if (hipMalloc(&t.d_w, w.size()) != hipSuccess) {
      std::fprintf(stderr, "hipMalloc %s (%.1f MB) failed\n", t.name.c_str(), w.size() / 1e6);
      return 1;
    }
    (void)hipMemcpy(t.d_w, w.data(), w.size(), hipMemcpyHostToDevice);
  }
  auto act_for = [&](std::int64_t nc) -> block_q8_1 * {
    for (auto &p : acts)
      if (p.first == nc) return p.second;
    return nullptr;
  };

  // Warm up until ~400 ms of GPU time has elapsed (DPM ramp; PLAN.md M2/M5).
  {
    double warm = 0;
    const int per_round = (int)std::min<std::size_t>(sel.size(), (std::size_t)64);
    EvPair ev;
    while (warm < 400.0) {
      warm += ev([&] {
        for (int i = 0; i < per_round; ++i) {
          const Tensor &t = sel[(std::size_t)i * sel.size() / per_round];
          (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols, nullptr);
        }
      });
    }
  }
  EvPair ev;

  // ---------------------------------------------------------------------
  // Launch floor: an empty kernel timed exactly like a tensor (record + sync),
  // and 1000 empty kernels launched back-to-back with one event pair. The first
  // is the floor of every per-tensor number below; the second is what a launch
  // costs when the queue is saturated (the engine's situation).
  // ---------------------------------------------------------------------
  float *d_dummy = nullptr;
  (void)hipMalloc(&d_dummy, sizeof(float));
  double null_once = 1e9, null_stream = 1e9;
  for (int r = 0; r < 5; ++r) {
    null_once = std::min(null_once, ev([&] { null_kernel<<<1, 32, 0, nullptr>>>(d_dummy); }));
    null_stream =
        std::min(null_stream, ev([&] {
                   for (int i = 0; i < 1000; ++i) null_kernel<<<1, 32, 0, nullptr>>>(d_dummy);
                 }) / 1000.0);
  }
  std::printf("launch floor: empty kernel %.3f ms timed with a sync, %.4f ms/launch "
              "back-to-back (1000 in one event pair)\n",
              null_once, null_stream);

  // ---------------------------------------------------------------------
  // REPLAY: one pass over the whole set with ONE event pair — this is how the
  // graph calls the matvecs (queued, no syncs), so it is the honest per-token
  // matvec number; the sum of the per-tensor measurements is inflated by the
  // sync floor above.
  // ---------------------------------------------------------------------
  double replay_ms = 1e9;
  for (int r = 0; r < reps; ++r) {
    replay_ms = std::min(replay_ms, ev([&] {
                           for (const Tensor &t : sel)
                             (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y,
                                                        t.nrows, t.ncols, nullptr);
                         }));
  }
  std::printf("replay (one queued pass over %zu tensors, min of %d): %.3f ms -> %.1f GB/s\n",
              sel.size(), reps, replay_ms, used / (replay_ms * 1e-3) / 1e9);

  // ---------------------------------------------------------------------
  // HIP GRAPH of the same 497-kernel pass. The engine issues ~2200 kernels per
  // token and the measured inter-kernel gap is ~3.6 us (null kernel, queued),
  // i.e. the launch tax is a large share of a 35 ms token -- so the question
  // "how much of that is avoidable?" needs an answer with the real sequence.
  // A captured graph replays the same kernels in the same order with the
  // driver's pre-built launch descriptors, so the difference IS the avoidable
  // part (llama.cpp ships with GGML_HIP_GRAPHS on).
  // ---------------------------------------------------------------------
  {
    hipStream_t cs = nullptr;
    if (hipStreamCreate(&cs) != hipSuccess) {
      std::printf("graph: stream create failed\n");
    } else {
      hipError_t ce = hipStreamBeginCapture(cs, hipStreamCaptureModeThreadLocal);
      if (ce != hipSuccess) {
        std::printf("graph: hipStreamBeginCapture failed (%s)\n", hipGetErrorString(ce));
      } else {
        for (const Tensor &t : sel)
          (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols, cs);
        hipGraph_t graph = nullptr;
        ce = hipStreamEndCapture(cs, &graph);
        if (ce != hipSuccess || graph == nullptr) {
          std::printf("graph: hipStreamEndCapture failed (%s)\n", hipGetErrorString(ce));
        } else {
          hipGraphExec_t gexec = nullptr;
          ce = hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
          if (ce != hipSuccess) {
            std::printf("graph: instantiate failed (%s)\n", hipGetErrorString(ce));
          } else {
            double gms = 1e9;
            for (int r = 0; r < reps; ++r) {
              gms = std::min(gms, ev([&] { (void)hipGraphLaunch(gexec, nullptr); }));
            }
            std::printf("graph replay of the same %zu kernels: %.3f ms (%.1f GB/s) = %.2fx the "
                        "streamed replay (%.3f ms) -> %.2f ms/token of launch tax recoverable\n",
                        sel.size(), gms, used / (gms * 1e-3) / 1e9, replay_ms / gms, replay_ms,
                        replay_ms - gms);
            (void)hipGraphExecDestroy(gexec);
          }
          (void)hipGraphDestroy(graph);
        }
      }
      (void)hipStreamDestroy(cs);
    }
  }

  // ---------------------------------------------------------------------
  // BATCH (prefill / MTP): N tokens share ONE read of the weights. The question
  // this section answers is *which* side runs out first.
  //   - weight side: the CTA pass over a row-block is amortized over N, so a
  //     pass whose cost is flat in N is weight-bound (per-token ms falls ~1/N);
  //   - token side: activation blocks + dp4a scale with N, and so does the
  //     activation working set per CTA (N * ncols bytes; 5 KB per token at
  //     ncols=5120, i.e. 82 KB at N=16) -- far past L1. If ms/token grows with
  //     N, the token side binds, and the growth rate sizes the activation
  //     re-read traffic (the hypothesis the previous session left open,
  //     journal-kernels §10: "a hipotese a medir e' o trafego de re-leitura da
  //     ativacao por linha").
  // The N=1 row is the GEMV pass again -- the same kernel the decode path uses --
  // so the rows of this table are directly comparable to each other.
  // ---------------------------------------------------------------------
  if (!batch_list.empty()) {
    std::int64_t max_rows_b = 0, max_bpr = 0;
    for (const Tensor &t : sel) {
      max_rows_b = std::max(max_rows_b, t.nrows);
      max_bpr = std::max(max_bpr, t.bpr);
    }
    const int n_max = *std::max_element(batch_list.begin(), batch_list.end());
    block_q8_1 *d_ab = nullptr;
    float *d_yb = nullptr;
    if (hipMalloc(&d_ab, (std::size_t)n_max * max_bpr * sizeof(block_q8_1)) != hipSuccess ||
        hipMalloc(&d_yb, (std::size_t)n_max * max_rows_b * sizeof(float)) != hipSuccess) {
      std::printf("batch: staging alloc failed\n");
      return 1;
    }
    // One quantized activation row per token slot, back to back: exactly the
    // layout the graph uses (act_stride = ncols/32).
    {
      std::vector<float> x;
      fill_x(x, max_bpr * QK8_1, 4242u);
      for (int n = 0; n < n_max; ++n) {
        float *d_x = nullptr;
        (void)hipMalloc(&d_x, x.size() * sizeof(float));
        (void)hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
        rdna4::quantize_q8_1_kernel<<<(unsigned)((max_bpr + 3) / 4), 128>>>(
            d_x, d_ab + (std::size_t)n * max_bpr, max_bpr);
        hipDeviceSynchronize();
        (void)hipFree(d_x);
      }
    }

    std::printf("\n=== BATCH === staged %.3f GiB, %zu tensors, launch cap %d\n",
                used / (1024.0 * 1024 * 1024), sel.size(), rdna4::matvec_batch_cap());
    std::printf("%5s %12s %12s %12s %14s %10s %12s %10s", "N", "pass_ms", "ms/token",
                "GB/s(pass)", "GB/s/token-eq", "vs GEMV", "reuse_ms", "reuse%");
    for (int U : unroll_list) std::printf(" %13s", U == 1 ? "unr1_ms" : (U == 2 ? "unr2_ms" : "unr4_ms"));
    std::printf("\n");
    double gemv_ms = 1e9, unroll_ref = 1e9;
    // `reuse` = act_stride 0: every token reads the SAME activation row. Same
    // kernel, same instruction count, same dp4a count, same weight traffic --
    // only the activation working set changes (5 KB instead of N * 5 KB), so
    // the difference against the dense pass is the cost of the activation
    // re-read traffic and nothing else. It is the control that separates
    // "issue-bound" from "cache/latency-bound" (journal-kernels §10).
    auto pass = [&](int N, bool reuse) {
      return ev([&] {
        for (const Tensor &t2 : sel) {
          if (N < 2) {
            (void)rdna4::matvec_launch(t2.dt, t2.d_w, act_for(t2.ncols), d_yb, t2.nrows, t2.ncols,
                                       nullptr);
          } else {
            (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                             reuse ? 0 : t2.bpr, N, nullptr);
          }
        }
      });
    };
    // Hybrid pass: the affected types (kMtIlp == 1) use the forced UNROLL, every
    // other type uses the shipping launch. Numbers stay bit-exact either way.
    auto pass_unr = [&](int N, int U) {
      return ev([&] {
        for (const Tensor &t2 : sel) {
          if (!rdna4::matvec_launch_batch_unroll(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                                 t2.bpr, N, U, nullptr)) {
            (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols, t2.bpr,
                                             N, nullptr);
          }
        }
      });
    };
    for (int N : batch_list) {
      // Two rounds: the first timed cell of the section is the one that pays the
      // clock ramp (the DPM gap this bench exists because of), so warm up first.
      (void)pass(N, false);
      double ms = 1e9, ms_reuse = 1e9;
      for (int r = 0; r < reps; ++r) ms = std::min(ms, pass(N, false));
      if (N >= 2) {
        (void)pass(N, true);
        for (int r = 0; r < reps; ++r) ms_reuse = std::min(ms_reuse, pass(N, true));
      }
      const int nn = N < 1 ? 1 : N;
      if (N < 2) gemv_ms = ms;
      const double gbs = used / (ms * 1e-3) / 1e9;
      std::printf("%5d %12.3f %12.3f %12.1f %14.1f %9.2fx %12.3f %10.1f", N, ms, ms / nn, gbs,
                  gbs * nn, gemv_ms / ms * nn, ms_reuse, 100.0 * ms_reuse / ms);
      for (int U : unroll_list) {
        if (N < 8) {
          std::printf(" %14s", "n/a");
          continue;
        }
        (void)pass_unr(N, U);
        double mu = 1e9;
        for (int r = 0; r < reps; ++r) mu = std::min(mu, pass_unr(N, U));
        std::printf(" %13.3f", mu);
        if (U == unroll_list.front()) unroll_ref = mu;
      }
      if (cap_probe) {
        // Capacidade livre: a MESMA passagem em DUAS streams. Se a maquina tem
        // folga (limitada por latencia), as duas se sobrepoem e o par custa bem
        // menos que 2x; se ja' esta' saturada (limitada por throughput), o par
        // custa 2x. Nao ha hazard: as duas escrevem no mesmo d_yb de proposito --
        // o que se mede aqui e' tempo, nao resultado.
        hipStream_t s1 = nullptr, s2 = nullptr;
        (void)hipStreamCreate(&s1);
        (void)hipStreamCreate(&s2);
        auto one = [&](hipStream_t st) {
          for (const Tensor &t2 : sel) {
            if (N < 2) {
              (void)rdna4::matvec_launch(t2.dt, t2.d_w, act_for(t2.ncols), d_yb, t2.nrows,
                                         t2.ncols, st);
            } else {
              (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                               t2.bpr, N, st);
            }
          }
        };
        (void)ev([&] { one(s1); });
        double two = 1e9;
        for (int r = 0; r < reps; ++r) {
          two = std::min(two, ev([&] {
                           one(s1);
                           one(s2);
                         }));
        }
        std::printf(" %14.3f (%.2fx de 2 passagens)", two, two / ms);
        (void)hipStreamDestroy(s1);
        (void)hipStreamDestroy(s2);
      }
      std::printf("\n");
    }

    // Per-type breakdown of the marginal cost. This is the experiment that
    // separates the two stories the totals cannot separate, and it works because
    // bits-per-weight differ: per byte, a 1.5 bpw type carries 2.3x the ELEMENTS
    // of a 3.4 bpw one, and the activation volume per token is
    // sum(nrows*ncols) -- the element count, not the byte count. So:
    //   - if the marginal cost per token is proportional to ELEMENTS, it is the
    //     activation-load side (L1/L2/IC traffic or its instruction count);
    //   - if it is proportional to BYTES, it is the weight side.
    // Slope from N=2 to N=16 (14 extra tokens), so the per-launch floor and the
    // shared weight pass land in the intercept and cancel out.
    std::printf("\nper-type marginal cost (slope of N=2 -> N=16, 14 extra tokens):\n");
    std::printf("%-10s %7s %9s %10s %12s %14s %14s\n", "type", "tensors", "bytes_MB", "bpw",
                "marg_ms/tok", "ms/tok per TB", "ms/tok per Telem");
    for (int dt = 1; dt <= 14; ++dt) {
      std::vector<const Tensor *> sub;
      double bytes = 0, elems = 0;
      for (const Tensor &t : sel)
        if (t.dt == dt) {
          sub.push_back(&t);
          bytes += t.bytes;
          elems += (double)t.nrows * (double)t.ncols;
        }
      if (sub.size() < 2) continue;
      auto spass = [&](int N) {
        (void)ev([&] {
          for (const Tensor *t2 : sub)
            (void)rdna4::matvec_launch_batch(t2->dt, t2->d_w, d_ab, d_yb, t2->nrows, t2->ncols,
                                             t2->bpr, N, nullptr);
        });
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
          best = std::min(best, ev([&] {
                             for (const Tensor *t2 : sub)
                               (void)rdna4::matvec_launch_batch(t2->dt, t2->d_w, d_ab, d_yb,
                                                                t2->nrows, t2->ncols, t2->bpr, N,
                                                                nullptr);
                           }));
        }
        return best;
      };
      const double m2 = spass(2), m16 = spass(16);
      const double slope = (m16 - m2) / 14.0;  // ms per extra token
      const double bpw = bytes > 0 && elems > 0 ? bytes * 8.0 / elems : 0.0;
      std::printf("%-10s %7zu %9.1f %10.2f %12.4f %14.2f %14.3f\n", dt_name(dt), sub.size(),
                  bytes / 1e6, bpw, slope, slope / (bytes / 1e12) / 1e3,
                  slope / (elems / 1e12));
    }

    // Register/local/shared usage per (type, N): the cheap way to rule out "the
    // batched kernel spills to scratch" before blaming traffic.
    std::printf("\nbatch kernel resources (hipFuncGetAttributes): type  N  regs  local  smem  threads\n");
    for (int dt = 1; dt <= 14; ++dt) {
      bool any = false;
      for (const Tensor &t : sel)
        if (t.dt == dt) any = true;
      if (!any) continue;
      for (int N : batch_list) {
        if (N < 2) continue;
        hipFuncAttributes a{};
        bool ok = false;
        switch (N) {
          case 2: ok = rdna4::matvec_batch_kernel_attrs<2>(dt, a); break;
          case 3: ok = rdna4::matvec_batch_kernel_attrs<3>(dt, a); break;
          case 4: ok = rdna4::matvec_batch_kernel_attrs<4>(dt, a); break;
          case 8: ok = rdna4::matvec_batch_kernel_attrs<8>(dt, a); break;
          case 16: ok = rdna4::matvec_batch_kernel_attrs<16>(dt, a); break;
          default: break;
        }
        if (!ok) continue;
        std::printf("  %-10s %3d %5d %6zu %5zu %8d\n", dt_name(dt), N, a.numRegs, a.localSizeBytes,
                    a.sharedSizeBytes, a.maxThreadsPerBlock);
      }
    }
    std::printf("\n");
    return 0;
  }

  // ---------------------------------------------------------------------
  // Per-tensor measurement, one launch per tensor per pass (never repeated
  // back-to-back, so no tensor stays in the 64 MB L2). Variant passes are
  // ROTATED per round: a fixed order gives the first variant of a round a
  // systematic penalty (measuring the shipping kernel and the rows=8 launch of
  // the same type — the same instantiation! — 1.5x apart is what exposed it).
  // ---------------------------------------------------------------------
  std::vector<Variant> variants;
  variants.push_back({"ship", 0, 0});
  if (repeat_ship) variants.push_back({"ship2", 6, 0});
  if (land) variants.push_back({"land", 7, 0});
  if (lut_lds) variants.push_back({"lut-lds", 8, 0});
  for (int v : lut_lds_unroll) variants.push_back({"lds-unrl", 9, v});
  for (int v : row_variants) variants.push_back({"rows", 1, v});
  for (int v : minb_variants) variants.push_back({"minb", 2, v});
  for (int v : unroll_variants) variants.push_back({v == 1 ? "prefetch" : "unroll", v == 1 ? 4 : 3, v});
  for (int v : ilp_variants) variants.push_back({"ilp", 5, v});

  struct Meas {
    double ship_ms = -1;
    std::vector<double> var_ms;
  };
  std::vector<Meas> m(sel.size());
  for (auto &mm : m) mm.var_ms.assign(variants.size(), -1.0);
  const int n_cfg = (int)variants.size();
  for (int r = 0; r < reps; ++r) {
    for (int c = 0; c < n_cfg; ++c) {
      const int cfg = (c + r) % n_cfg;
      for (std::size_t i = 0; i < sel.size(); ++i) {
        const double ms = ev([&] {
          (void)launch_variant(variants[cfg].kind, variants[cfg].value, sel[i],
                               act_for(sel[i].ncols), d_y);
        });
        if (ms < 0) continue;
        if (cfg == 0) {
          if (m[i].ship_ms < 0 || ms < m[i].ship_ms) m[i].ship_ms = ms;
        } else if (m[i].var_ms[cfg] < 0 || ms < m[i].var_ms[cfg]) {
          m[i].var_ms[cfg] = ms;
        }
      }
    }
  }

  // ---------------------------------------------------------------------
  // Report per (type, shape) class, then the extrapolated per-token matvec time
  // over ALL tensors of the inventory (not just the staged ones).
  // ---------------------------------------------------------------------
  struct Class {
    int dt;
    std::int64_t nrows, bpr;
    double bytes_staged = 0, bytes_all = 0, ms_staged = 0;
    int n_staged = 0;
  };
  std::vector<Class> cls;
  auto find_class = [&](int dt, std::int64_t nr, std::int64_t bpr) -> Class * {
    for (auto &c : cls)
      if (c.dt == dt && c.nrows == nr && c.bpr == bpr) return &c;
    cls.push_back({dt, nr, bpr, 0, 0, 0, 0});
    return &cls.back();
  };
  for (std::size_t i = 0; i < sel.size(); ++i) {
    Class *c = find_class(sel[i].dt, sel[i].nrows, sel[i].bpr);
    c->bytes_staged += sel[i].bytes;
    c->ms_staged += m[i].ship_ms;
    c->n_staged++;
  }
  for (const Tensor &t : all) find_class(t.dt, t.nrows, t.bpr)->bytes_all += t.bytes;

  struct Row {
    Class *c;
    double rate, ms_tok;
  };
  std::vector<Row> rows;
  double total_ms = 0;
  for (auto &c : cls) {
    const double rate = c.ms_staged > 0 ? c.bytes_staged / (c.ms_staged * 1e-3) / 1e9 : 0.0;
    const double ms_tok = rate > 0 ? c.bytes_all / (rate * 1e9) * 1e3 : 0.0;
    rows.push_back({&c, rate, ms_tok});
    total_ms += ms_tok;
  }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.ms_tok > b.ms_tok; });

  // ms(tok)-fix: the same per-class extrapolation with ONE sync floor removed per
  // tensor (`null_once` = an empty kernel timed with the same record+sync
  // protocol). The per-tensor measurement pays that floor on every launch, which
  // for a 261 KB q8_0 tensor is most of the number -- so the classification of
  // "this class is slow" has to be read off this column, not off ms/tens.
  std::printf("\n%-9s %9s %6s %6s %6s %10s %10s %10s %10s %8s\n", "type", "rows", "bpr", "n",
              "GB/s", "ms(tok)", "ms/tens", "ms-fix", "ms(tok)-fix", "share");
  for (const Row &r : rows) {
    const double per_fix = std::max(0.0, r.c->ms_staged / std::max(1, r.c->n_staged) - null_once);
    const double rate_fix = per_fix > 0 ? r.c->bytes_staged / (per_fix * 1e-3) / 1e9 : 0.0;
    const double ms_fix = rate_fix > 0 ? r.c->bytes_all / (rate_fix * 1e9) * 1e3 : 0.0;
    std::printf("%-9s %9lld %6lld %6d %6.0f %10.3f %10.4f %10.4f %10.3f %7.1f%%\n", dt_name(r.c->dt),
                (long long)r.c->nrows, (long long)r.c->bpr, r.c->n_staged, r.rate, r.ms_tok,
                r.c->ms_staged / std::max(1, r.c->n_staged), per_fix, ms_fix,
                100.0 * r.ms_tok / total_ms);
  }
  std::printf("per-tensor sum: %.3f GB in %.2f ms -> %.1f tok/s (matvec only)\n",
              grand_total_bytes / 1e9, total_ms, 1000.0 / total_ms);
  std::printf("REPLAY (honest, queued): %.2f ms -> %.1f tok/s (matvec only); the per-tensor sum "
              "carries the %.3f ms sync floor x %zu launches\n",
              replay_ms, 1000.0 / replay_ms, null_once, sel.size());
  {
    int unstaged = 0;
    double ub = 0;
    for (const auto &c : cls)
      if (c.bytes_staged == 0) {
        unstaged++;
        ub += c.bytes_all;
      }
    if (unstaged > 0)
      std::printf("WARNING: %d (type,shape) classes have no staged tensor; %.3f GB (%.1f%% of the "
                  "per-token traffic) is missing from the tables\n",
                  unstaged, ub / 1e9, 100.0 * ub / grand_total_bytes);
    else
      std::printf("coverage: every (type,shape) class is represented in the staged set\n");
  }
  std::printf("(real per-token weight traffic excluding token_embd/MTP: %.3f GB; bench reports "
              "%.3f GB per token using loader.total_bytes())\n",
              grand_total_bytes / 1e9, (double)ld.total_bytes() / 1e9);

  // Per-type roll-up (per-tensor sum and replay share the same rates).
  std::printf("\n%-9s %10s %8s %10s %8s\n", "type", "bytes(tok)", "share", "ms(tok)", "GB/s");
  double byt[16] = {0}, mst[16] = {0};
  for (const Row &r : rows) {
    byt[r.c->dt] += r.c->bytes_all;
    mst[r.c->dt] += r.ms_tok;
  }
  for (int dt = 1; dt <= 14; ++dt) {
    if (byt[dt] <= 0) continue;
    std::printf("%-9s %10.3f %7.1f%% %10.3f %8.0f\n", dt_name(dt), byt[dt] / 1e9,
                100.0 * byt[dt] / grand_total_bytes, mst[dt], byt[dt] / (mst[dt] * 1e-3) / 1e9);
  }

  // ---------------------------------------------------------------------
  // Variant A/B (rows / minb). Both axes are bit-exact: ROWS moves which CTA
  // computes which row, minb only caps the register allocator, so neither
  // touches the arithmetic of an output element (see matvec.cuh).
  // ---------------------------------------------------------------------
  if (variants.size() > 1) {
    std::printf("\nVARIANT A/B (min over %d rotated passes)\n", reps);
    std::printf("%-12s %10s %8s   %s\n", "variant", "agg GB/s", "vs ship", "per-class ms(tok)");
    // shipping baseline
    double ship_bytes = 0, ship_ms = 0;
    for (const Row &r : rows) {
      ship_bytes += r.c->bytes_all;
      ship_ms += r.ms_tok;
    }
    std::printf("%-12s %10.0f %7.3fx   %s\n", "ship", ship_bytes / (ship_ms * 1e-3) / 1e9, 1.0,
                "baseline");
    std::vector<double> var_tot(variants.size(), 0);
    for (std::size_t v = 1; v < variants.size(); ++v) {
      double vb = 0, vm = 0;
      for (const Row &r : rows) {
        const Class *c = r.c;
        double use_ms = 0;
        for (std::size_t i = 0; i < sel.size(); ++i) {
          if (sel[i].dt != c->dt || sel[i].nrows != c->nrows || sel[i].bpr != c->bpr) continue;
          if (m[i].var_ms[v] > 0) use_ms += m[i].var_ms[v];
        }
        if (use_ms <= 0) continue;
        const double rate = c->bytes_staged / (use_ms * 1e-3) / 1e9;
        vb += c->bytes_all;
        vm += c->bytes_all / (rate * 1e9) * 1e3;
      }
      var_tot[v] = vm;
      std::printf("%-7s%-4d %10.0f %7.3fx   %.2f ms vs %.2f ms over %.1f%% of the bytes\n",
                  variants[v].name, variants[v].value, vb / (vm * 1e-3) / 1e9,
                  vm > 0 ? ship_ms / vm : 0.0, vm, ship_ms, 100.0 * vb / ship_bytes);
    }

    // ------------------------------------------------------------------
    // Per-type matrix: the tuning table of this campaign (config -> measured,
    // per type). One row per (type, config); `ms(tok)` is the per-token cost
    // extrapolated over ALL tensors of that type in the inventory, so the rows of
    // one type are directly comparable and the ratios are what the tuning table
    // in matvec.cuh is chosen from.
    //
    // `same kernel?` marks the rows that are the SHIPPING kernel measured a
    // second time under a different label: those ratios are the noise floor of
    // this bench, and nothing below that floor is a win.
    // ------------------------------------------------------------------
    std::printf("\nPER-TYPE MATRIX (ms/token per type, min of %d rotated passes)\n", reps);
    std::printf("%-9s %-10s %6s %10s %10s %8s  %s\n", "type", "config", "knob", "ms(tok)",
                "vs ship", "ratio", "same kernel?");
    for (int dt = 1; dt <= 14; ++dt) {
      double ship_t = 0;
      for (std::size_t i = 0; i < sel.size(); ++i)
        if (sel[i].dt == dt && m[i].ship_ms > 0) ship_t += m[i].ship_ms;
      if (ship_t <= 0) continue;
      std::printf("%-9s %-10s %6s %10.3f %10s %8s  yes (baseline)\n", dt_name(dt), "ship", "-",
                  ship_t, "-", "1.00x");
      for (std::size_t v = 1; v < variants.size(); ++v) {
        const bool same = variants[v].kind == 6 ||
                          (variants[v].kind == 1 &&
                           variants[v].value == rdna4::matvec_default_rows(dt)) ||
                          (variants[v].kind == 5 &&
                           variants[v].value == rdna4::matvec_default_ilp(dt));
        double b = 0;
        for (std::size_t i = 0; i < sel.size(); ++i)
          if (sel[i].dt == dt && m[i].var_ms[v] > 0) b += m[i].var_ms[v];
        if (b <= 0) continue;
        std::printf("%-9s %-10s %6d %10.3f %9.3fx %7.2fx  %s\n", dt_name(dt), variants[v].name,
                    variants[v].value, b, ship_t / b, ship_t / b, same ? "yes" : "no");
      }
    }

    // Aggregate noise floor: the shipping kernel measured twice, over the whole
    // inventory (the per-type rows above are the same thing per type).
    for (std::size_t v = 1; v < variants.size(); ++v) {
      if (variants[v].kind != 6) continue;
      double a = 0, b = 0;
      for (std::size_t i = 0; i < sel.size(); ++i)
        if (m[i].ship_ms > 0 && m[i].var_ms[v] > 0) {
          a += m[i].ship_ms;
          b += m[i].var_ms[v];
        }
      std::printf("\nnoise floor (same kernel, whole inventory): ship %.3f ms vs ship2 %.3f ms = "
                  "%.3fx\n",
                  a, b, a / b);
    }
  }
  return 0;
}
