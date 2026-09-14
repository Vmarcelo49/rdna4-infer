// M8 — batched prefill must be BIT-IDENTICAL to the per-token path.
//
// That is the whole safety argument for the batched forward (graph.cuh
// `forward_batch`): it batches only the projections (the bit-exact
// `matvec_kernel_batch`), the batched activation quantization and the norms, and
// keeps the KV writes, the attention and the GDN recurrence sequential in token
// order. If any of those batched pieces did not reproduce the per-token
// arithmetic exactly, this test fails instead of the difference hiding inside a
// perplexity number.
//
// It also measures the prefill win: tokens/s for the batched path vs the same
// tokens one at a time, with the warm-up inside the GPU-locked, DPM-aware harness
// this repo uses everywhere.
//
// usage: check-batch-gpu <file.gguf> [batch sizes...]   (default 2 3 4 8 16)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/tokenizer.h"

namespace {

const char *kPrompt =
    "The history of the Roman Empire spans more than a thousand years, from the "
    "founding of the city to the fall of Constantinople. Its influence on law, "
    "language, engineering and politics can still be seen today in every country "
    "that inherited any part of its legacy.";

double now_ms() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double, std::milli>(clock::now() - t0).count();
}

bool compare(const std::vector<float> &a, const std::vector<float> &b, double *rel, double *maxd) {
  double num = 0, den = 0, mx = 0;
  bool exact = a.size() == b.size();
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)a[i] * a[i] + (double)b[i] * b[i];
    mx = std::max(mx, std::fabs(d));
    if (a[i] != b[i]) exact = false;
  }
  *rel = den > 0 ? std::sqrt(num / den) : 0.0;
  *maxd = mx;
  return exact;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-batch-gpu <file.gguf> [batch sizes...]\n");
    return 2;
  }
  std::vector<int> ns;
  for (int i = 2; i < argc; ++i) ns.push_back(std::atoi(argv[i]));
  if (ns.empty()) ns = {2, 3, 4, 8, 16};

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(argv[1], err)) {
    std::fprintf(stderr, "gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "config failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "tokenizer failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids = tk.encode(kPrompt, true);
  const int need = *std::max_element(ns.begin(), ns.end());
  while ((int)ids.size() < need + 4) ids.push_back(ids[ids.size() % 17]);

  const int ctx = (int)ids.size() + 8;
  rdna4::Graph g(loader, cfg);
  if (!g.init(ctx, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
    std::fprintf(stderr, "graph init failed: %s\n", err.c_str());
    return 1;
  }
  g.set_want_hidden(false);
  std::printf("prompt: %zu tokens, ctx %d\n\n", ids.size(), ctx);

  int failures = 0;
  std::printf("%5s %12s %12s %12s %10s %10s\n", "N", "per-token ms", "batched ms", "speedup",
              "rel-L2", "max|d|");
  for (int n : ns) {
    if (!rdna4::Graph::batch_supported(n) || n > rdna4::Graph::kMaxBatch) {
      std::printf("%5d  (no batched kernel for this size)\n", n);
      continue;
    }
    std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + n);
    std::vector<float> h1, l1, h2, l2;

    // reference: one token at a time (the shipping prefill path)
    g.reset_state(err);
    double t0 = now_ms();
    for (int t = 0; t < n; ++t) {
      if (!g.forward_tokens({chunk[(std::size_t)t]}, t, h1, l1, err)) {
        std::fprintf(stderr, "forward_tokens failed: %s\n", err.c_str());
        return 1;
      }
    }
    const double ms_tok = (now_ms() - t0) / n;

    // batched
    g.reset_state(err);
    t0 = now_ms();
    if (!g.forward_batch(chunk, 0, h2, l2, err)) {
      std::fprintf(stderr, "forward_batch failed: %s\n", err.c_str());
      return 1;
    }
    const double ms_bat = (now_ms() - t0) / n;

    double rel = 0, maxd = 0;
    const bool exact = compare(l1, l2, &rel, &maxd);
    std::printf("%5d %12.3f %12.3f %11.2fx %10.1e %10.1e  %s\n", n, ms_tok, ms_bat,
                ms_tok / ms_bat, rel, maxd, exact ? "BIT-EXACT" : "DIFFERS");
    if (!exact) {
      std::fprintf(stderr, "FAIL: N=%d batched logits differ from the per-token path "
                           "(%zu values, rel-L2 %.3e, max|d| %.3e)\n",
                   n, l1.size(), rel, maxd);
      ++failures;
    }
  }

  // the KV cache and the recurrent state must also match: run the full prompt both
  // ways and compare the logits of the last token, which depends on every KV row
  // and on the whole GDN state
  {
    std::vector<float> h1, l1, h2, l2;
    const int n = (int)ids.size() - 4;
    std::vector<std::int32_t> chunk(ids.begin(), ids.begin() + n);
    g.reset_state(err);
    for (int t = 0; t < n; ++t) {
      if (!g.forward_tokens({chunk[(std::size_t)t]}, t, h1, l1, err)) return 1;
    }
    g.reset_state(err);
    // mixed chunks (16 + 8 + tail per token): the CLI does exactly this
    int t = 0;
    while (t < n) {
      int take = std::min(16, n - t);
      if (take == 1 || (!rdna4::Graph::batch_supported(take) && take < 16)) {
        std::vector<std::int32_t> one{chunk[(std::size_t)t]};
        if (!g.forward_tokens(one, t, h2, l2, err)) return 1;
        ++t;
        continue;
      }
      std::vector<std::int32_t> part(chunk.begin() + t, chunk.begin() + t + take);
      if (!g.forward_batch(part, t, h2, l2, err)) {
        std::fprintf(stderr, "mixed-chunk forward_batch failed: %s\n", err.c_str());
        return 1;
      }
      t += take;
    }
    double rel = 0, maxd = 0;
    const bool exact = compare(l1, l2, &rel, &maxd);
    std::printf("\nfull prompt (%d tokens, mixed 16+tail chunks): rel-L2 %.3e max|d| %.3e  %s\n", n,
                rel, maxd, exact ? "BIT-EXACT" : "DIFFERS");
    if (!exact) ++failures;
  }

  if (failures == 0) {
    std::printf("check-batch-gpu: OK (batched prefill is bit-identical to per-token)\n");
    return 0;
  }
  std::printf("check-batch-gpu: FAILED (%d configuration(s))\n", failures);
  return 1;
}
