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

// The launch used by every A/B variant: `kind` 0 = shipping (per-type rows/ILP),
// 1 = forced rows, 2 = forced __launch_bounds__ min-blocks.
inline bool launch_variant(int kind, int value, const Tensor &t, const block_q8_1 *d_a, float *d_y) {
  if (kind == 0) return rdna4::matvec_launch(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  if (kind == 1)
    return rdna4::matvec_launch_rows(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value);
  if (kind == 3)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value,
                                       /*pf=*/false);
  if (kind == 4)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, 1,
                                       /*pf=*/true);
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
  int kind = 0;   // 0 = ship, 1 = rows, 2 = minb
  int value = 0;
};

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bench-matvec-shapes-gpu <model.gguf> [--budget-mib N] [--reps N] "
                 "[--rows a,b,c] [--minb a,b,c] [--no-ab]\n");
    return 1;
  }
  double budget_mib = 12000.0;
  int reps = 4;
  bool do_ab = true;
  bool pf_only = false;
  std::vector<int> row_variants, minb_variants, unroll_variants;
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
    } else if (std::strcmp(argv[i], "--pf") == 0) {
      unroll_variants.push_back(1);
      pf_only = true;
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
  // Per-tensor measurement, one launch per tensor per pass (never repeated
  // back-to-back, so no tensor stays in the 64 MB L2). Variant passes are
  // ROTATED per round: a fixed order gives the first variant of a round a
  // systematic penalty (measuring the shipping kernel and the rows=8 launch of
  // the same type — the same instantiation! — 1.5x apart is what exposed it).
  // ---------------------------------------------------------------------
  std::vector<Variant> variants;
  variants.push_back({"ship", 0, 0});
  for (int v : row_variants) variants.push_back({"rows", 1, v});
  for (int v : minb_variants) variants.push_back({"minb", 2, v});
  for (int v : unroll_variants) variants.push_back({v == 1 ? "prefetch" : "unroll", v == 1 ? 4 : 3, v});

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

  std::printf("\n%-9s %9s %6s %6s %6s %10s %10s %8s\n", "type", "rows", "bpr", "n", "GB/s",
              "ms(tok)", "ms/tens", "share");
  for (const Row &r : rows) {
    std::printf("%-9s %9lld %6lld %6d %6.0f %10.3f %10.4f %7.1f%%\n", dt_name(r.c->dt),
                (long long)r.c->nrows, (long long)r.c->bpr, r.c->n_staged, r.rate, r.ms_tok,
                r.c->ms_staged / std::max(1, r.c->n_staged), 100.0 * r.ms_tok / total_ms);
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

    // Noise floor: measuring the SHIPPING launch against the variant that equals
    // the type's own shipping configuration is the same kernel twice, so the
    // ratio must be 1.00. Anything else invalidates the smaller differences.
    std::printf("\nnoise floor (same kernel measured as 'ship' and as its own configuration):\n");
    std::printf("%-9s %12s %12s %12s %8s\n", "type", "knob=own", "ship ms", "same ms", "ratio");
    for (int dt = 1; dt <= 14; ++dt) {
      for (std::size_t v = 1; v < variants.size(); ++v) {
        const int own = variants[v].kind == 1 ? rdna4::matvec_default_rows(dt) : 0;
        if (variants[v].kind == 1 && variants[v].value != own) continue;
        if (variants[v].kind == 2 && variants[v].value != 0) continue;
        double a = 0, b = 0;
        int n = 0;
        for (std::size_t i = 0; i < sel.size(); ++i) {
          if (sel[i].dt != dt) continue;
          if (m[i].var_ms[v] <= 0) continue;
          a += m[i].ship_ms;
          b += m[i].var_ms[v];
          n++;
        }
        if (n == 0) continue;
        std::printf("%-9s %12d %12.3f %12.3f %7.2fx\n", dt_name(dt), variants[v].value, a, b,
                    b > 0 ? a / b : 0.0);
      }
    }
  }
  return 0;
}
