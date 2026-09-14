#pragma once
// Phase profiler for Graph::forward_run — ADDITIVE, OPT-IN diagnostic hook.
//
// Owner: the measurement agent (worktree rdna4-wt-medicoes-gpu). It is inert
// unless a test explicitly enables it (Graph::set_phase_prof), so the shipped
// engine path is unchanged: no extra kernel, no extra copy, no changed buffer.
//
// How it works
//   `mark(name)` records ONE hipEvent_t on the stream and stores the pair
//   (previous mark, this mark) that closes the previous bucket. Events only
//   timestamp the command queue — they read and write nothing — so no number of
//   the model can change. The device timeline of one token is partitioned into
//   consecutive buckets:
//
//       bucket(name) = [ event(name) .. next mark ] = that phase's kernels plus
//                      the dispatch gap in front of the following phase
//
//   Nothing is synchronised inside the token; `collect()` (called by the tool
//   once per token, right after the blocking logits readback that already drains
//   the stream) turns the recorded pairs into elapsed times.
//
// Cost, and why it must be reported
//   Every mark is one extra command in the queue (measured by
//   tests/bench_phases_gpu.hip with an empty-kernel chain). That cost lands
//   inside the bucket that owns it, so an instrumented run is SLOWER than the
//   clean run; the tool measures both and prints the inflation, so the reader can
//   subtract it. `level` selects the granularity: 1 marks the phase blocks
//   (attention, GDN, FFN...), 2 additionally splits every weight projection into
//   act_quant|matvec.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

namespace rdna4 {

struct PhaseProf {
  struct Bucket {
    std::string name;
    double ms = 0.0;      // accumulated device time of [mark .. next mark]
    std::int64_t marks = 0;
  };
  struct Pair {
    std::size_t slot;
    std::size_t open_ev;
    std::size_t close_ev;
  };

  int level = 1;
  bool on = true;
  // count_only: nenhum evento e criado/gravado — so conta marcas por bucket.
  // Custo zero no dispositivo, entao serve para contar lancamentos por fase sem
  // inflar a medicao (ver tests/bench_phases_gpu.hip --count-only).
  bool count_only = false;
  std::string tag;                                      // prefix ("head:")
  std::vector<hipEvent_t> ev;                           // one per mark (reused per token)
  std::vector<Pair> pairs;
  std::unordered_map<std::string, std::size_t> index;    // name -> bucket slot
  std::vector<Bucket> buckets;
  int pending = -1;                                      // event index that opened the bucket
  std::size_t pending_slot = 0;
  std::int64_t marks_total = 0;
  std::int64_t tokens = 0;
  double ev_us = 0.0;                                    // cost of one mark (set by the caller)

  bool fine() const { return level >= 2; }
  void set_tag(const char *t) { tag = t ? t : ""; }

  std::size_t slot_for(const std::string &name) {
    auto it = index.find(name);
    if (it != index.end()) return it->second;
    const std::size_t s = buckets.size();
    index.emplace(name, s);
    buckets.push_back(Bucket{name, 0.0, 0});
    return s;
  }

  void mark(const char *name) {
    if (!on) return;
    if (count_only) {
      const std::size_t s = slot_for(tag + name);
      buckets[s].marks += 1;
      ++marks_total;
      return;
    }
    // One fresh event per mark: reusing an event before its pair has been read
    // would overwrite the timestamp, so the pool only grows. The tool reads the
    // pairs ONCE, at the end of the measured pass (the per-token blocking logits
    // readback already drained the stream), which keeps the whole collection cost
    // out of the per-token wall time.
    hipEvent_t e{};
    if (hipEventCreate(&e) != hipSuccess) { on = false; return; }
    ev.push_back(e);
    const std::size_t idx = ev.size() - 1;
    if (hipEventRecord(e, nullptr) != hipSuccess) { on = false; return; }
    if (pending >= 0) pairs.push_back(Pair{pending_slot, (std::size_t)pending, idx});
    pending_slot = slot_for(tag + name);
    pending = (int)idx;
    ++marks_total;
  }

  // Call ONCE, after the queue has drained (hipDeviceSynchronize), at the end of
  // the measured pass: turns the recorded pairs into elapsed times. `collect()`
  // is the old name and does the same (kept so both spellings work).
  void finish() {
    if (!on) return;
    for (const Pair &p : pairs) {
      float ms = 0.0f;
      if (hipEventElapsedTime(&ms, ev[p.open_ev], ev[p.close_ev]) == hipSuccess) {
        buckets[p.slot].ms += (double)ms;
        buckets[p.slot].marks += 1;
      }
    }
    pairs.clear();
    pending = -1;
    marks_total = 0;
  }
  void collect() { finish(); }
  // Denominator of the per-token table (number of tokens in the measured pass).
  void set_tokens(std::int64_t n) { tokens = n; }

  double total_ms() const {
    double s = 0.0;
    for (const auto &b : buckets) s += b.ms;
    return s;
  }

  void print(const char *title) const {
    std::vector<const Bucket *> ordered;
    ordered.reserve(buckets.size());
    for (const auto &b : buckets) ordered.push_back(&b);
    std::sort(ordered.begin(), ordered.end(),
              [](const Bucket *a, const Bucket *b) { return a->ms > b->ms; });
    std::printf("  %s: %.3f ms sum, %lld buckets with time, %lld tokens (%.1f marks/token)\n", title,
                total_ms(), (long long)ordered.size(), (long long)tokens,
                tokens ? (double)marks_tot() / (double)tokens : 0.0);
    std::printf("  %-24s %10s %7s %9s %10s\n", "bucket", "ms/token", "%", "marks/tok",
                "ev_us/tok");
    for (const Bucket *b : ordered) {
      const double per_tok = tokens ? b->ms / (double)tokens : b->ms;
      const double pct = total_ms() > 0.0 ? 100.0 * b->ms / total_ms() : 0.0;
      std::printf("  %-24s %10.3f %6.1f%% %9.1f %10.1f\n", b->name.c_str(), per_tok, pct,
                  tokens ? (double)b->marks / (double)tokens : 0.0,
                  tokens ? ev_us * (double)b->marks / (double)tokens : 0.0);
    }
  }

  std::int64_t marks_tot() const {
    std::int64_t m = 0;
    for (const auto &b : buckets) m += b.marks;
    return m;
  }
};

// Insertion helper: `RD_PHASE(prof_, "name")` is a single statement and a no-op
// (one predictable branch) when the profiler is off.
#define RD_PHASE(p, name)                  \
  do {                                     \
    if ((p) != nullptr) (p)->mark(name);   \
  } while (0)

}  // namespace rdna4
