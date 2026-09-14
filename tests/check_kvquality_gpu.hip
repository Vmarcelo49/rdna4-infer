// KV cache quality probe — the measurement PPL cannot make.
//
// Why not perplexity: KV quantization is a *cache* error, and the aggregate NLL
// of natural text barely moves when the cache is destroyed (the upstream AIME25
// table cited in the briefing drops from 37.9% to 2.0% while PPL moves ~0.4%).
// A number that does not move cannot rank two formats. What this test measures
// instead is the divergence of the model's *output distribution* caused by the
// KV format, with the f16 cache of the SAME engine on the SAME tokens as the
// reference:
//
//   * mean KL( f16 || target ) in nats per position, over the full vocabulary;
//   * the fraction of positions whose argmax (the greedy token) changed;
//   * the paired per-position |dNLL| of the true next token, whose mean is the
//     PPL-style number (reported too, so the PPL argument is visible side by side
//     with the sensitive one).
//
// Everything runs in ONE process and one graph family, so the engine's own
// divergence from llama.cpp cancels: the only difference between two rows of the
// table is the cache type. Running f16 twice gives the noise floor of the whole
// harness (it must come out identical, not merely close).
//
// The token stream comes from `rdna4-infer tokenize -f <corpus>` (whitespace
// separated ids), so the test needs no tokenizer; the embeddings come from the
// graph's own device-side dequant row lookup (forward_tokens), so there is no
// second dequantization path involved.
//
// usage: check-kvquality-gpu <file.gguf> <ids.txt> [n_positions] [configs]
//   n_positions default 256
//   configs     default "k:v,k:v,..." -- see kDefaultConfigs below
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"

namespace {

// The comparison is designed around the night's question -- which format is the
// better K and which the better V -- so it holds one side fixed while the other
// moves, instead of only walking the "natural" pairs.
//
//   K axis at V=q4_1 fixed:  q8_0, q5_0, q4_1, q4_0
//   V axis at K=q5_0 fixed:  q8_0, q5_0, q4_1, q4_0
// plus f16/f16 at the top and the night's target q5_0/q4_1.
const char *kDefaultConfigs =
    "f16:f16,q8_0:q8_0,q4_0:q4_0,q5_0:q4_1,"
    "q8_0:q4_1,q4_1:q4_1,q4_0:q4_1,"
    "q5_0:q8_0,q5_0:q5_0,q5_0:q4_0";

struct Row {
  std::string tag;
  rdna4::KvType k = rdna4::KvType::F16, v = rdna4::KvType::F16;
  double ppl = 0.0;
  double mean_kl = 0.0;   // nats/position, KL(f16 || this)
  double max_kl = 0.0;
  double mean_dnll = 0.0; // paired |dNLL| vs f16
  double max_dnll = 0.0;
  int top1_diff = 0;
  int n = 0;
  bool have_kl = false;
};

}  // namespace

int main(int argc, char **argv) {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (argc < 3) {
    std::fprintf(stderr, "usage: check-kvquality-gpu <file.gguf> <ids.txt> [n_positions] "
                         "[k:v,k:v,...]\n");
    return 2;
  }
  const std::string model = argv[1];
  const char *ids_path = argv[2];
  const int n_pos = argc >= 4 ? std::atoi(argv[3]) : 256;
  const std::string cfg_s = argc >= 5 ? argv[4] : kDefaultConfigs;
  if (n_pos < 2 || n_pos > 4096) {
    std::fprintf(stderr, "n_positions must be in [2, 4096]\n");
    return 2;
  }

  // ---- token stream --------------------------------------------------------
  std::vector<std::int32_t> ids;
  {
    FILE *f = std::fopen(ids_path, "r");
    if (f == nullptr) {
      std::fprintf(stderr, "cannot open %s\n", ids_path);
      return 1;
    }
    long v = 0;
    while (std::fscanf(f, "%ld", &v) == 1) ids.push_back((std::int32_t)v);
    std::fclose(f);
  }
  if ((int)ids.size() < n_pos + 1) {
    std::fprintf(stderr, "ids file has %zu tokens, need %d\n", ids.size(), n_pos + 1);
    return 1;
  }
  ids.resize((std::size_t)n_pos + 1);

  // ---- config list ---------------------------------------------------------
  std::vector<Row> rows;
  {
    std::string s = cfg_s;
    std::size_t pos = 0;
    while (pos <= s.size()) {
      const std::size_t comma = s.find(',', pos);
      const std::string item = s.substr(pos, comma == std::string::npos ? std::string::npos
                                                                       : comma - pos);
      if (!item.empty()) {
        const std::size_t c = item.find(':');
        if (c == std::string::npos) {
          std::fprintf(stderr, "bad config %s (want k:v)\n", item.c_str());
          return 2;
        }
        Row r;
        r.tag = item;
        const std::string ks = item.substr(0, c), vs = item.substr(c + 1);
        if (rdna4::kv_type_parse(ks.c_str(), &r.k) != nullptr ||
            rdna4::kv_type_parse(vs.c_str(), &r.v) != nullptr) {
          std::fprintf(stderr, "bad config %s\n", item.c_str());
          return 2;
        }
        rows.push_back(r);
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
  }

  // ---- model ---------------------------------------------------------------
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

  // ---- reference pass (f16/f16) -------------------------------------------
  const int nv = 0;  // the real vocab comes from the graph (output_.dim1)
  std::vector<float> ref_ls;   // log-softmax of the reference, [n_pos][nv]
  std::vector<double> ref_nll;
  int ref_vocab = 0;
  {
    rdna4::Graph g(ld, cfg);
    if (!g.init(n_pos + 8, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
      std::fprintf(stderr, "reference graph init: %s\n", err.c_str());
      return 1;
    }
    ref_vocab = g.n_vocab();
    ref_ls.resize((std::size_t)n_pos * (std::size_t)ref_vocab);
    ref_nll.assign((std::size_t)n_pos, 0.0);
    std::vector<float> h, logits;
    std::vector<float> ls((std::size_t)ref_vocab);
    for (int t = 0; t < n_pos; ++t) {
      if (!g.forward_tokens({ids[(std::size_t)t]}, t, h, logits, err)) {
        std::fprintf(stderr, "reference forward at %d: %s\n", t, err.c_str());
        return 1;
      }
      double m = logits[0];
      for (float x : logits) m = std::max(m, (double)x);
      double sum = 0.0;
      for (float x : logits) sum += std::exp((double)x - m);
      const double logZ = m + std::log(sum);
      for (int v = 0; v < ref_vocab; ++v) {
        ls[(std::size_t)v] = (float)((double)logits[(std::size_t)v] - logZ);
      }
      std::memcpy(ref_ls.data() + (std::size_t)t * ref_vocab, ls.data(),
                  (std::size_t)ref_vocab * sizeof(float));
      ref_nll[(std::size_t)t] = -(double)ls[(std::size_t)ids[(std::size_t)t + 1]];
      if ((t + 1) % 64 == 0) std::printf("   reference %d/%d positions\r", t + 1, n_pos);
    }
    std::printf("reference (f16/f16) done: %d positions, vocab %d                \n", n_pos,
                ref_vocab);
  }
  (void)nv;

  // ---- one pass per config ------------------------------------------------
  double ref_ppl = 0.0;
  {
    double s = 0.0;
    for (double x : ref_nll) s += x;
    ref_ppl = std::exp(s / (double)ref_nll.size());
  }
  std::printf("\n%-12s %10s %12s %12s %12s %10s\n", "config", "PPL", "KL nats/pos",
              "KL max/pos", "mean|dNLL|", "top1 diff");
  std::printf("%-12s %10.5f %12s %12s %12s %10s\n", "f16:f16 (ref)", ref_ppl, "--", "--", "--",
              "--");

  for (const Row &cr : rows) {
    Row r = cr;
    if (r.k == rdna4::KvType::F16 && r.v == rdna4::KvType::F16) {
      // the f16 row of the list IS the reference: report its KL against itself,
      // which is the floor of the metric, not a measurement of the format.
      r.have_kl = true;
      r.mean_kl = 0.0;
      r.max_kl = 0.0;
      r.mean_dnll = 0.0;
      r.max_dnll = 0.0;
      r.top1_diff = 0;
      r.ppl = ref_ppl;
      r.n = n_pos;
      std::printf("%-12s %10.5f %12.6f %12.6f %12.6f %10d  (reference, must be exactly 0)\n",
                  r.tag.c_str(), r.ppl, r.mean_kl, r.max_kl, r.mean_dnll, r.top1_diff);
      continue;
    }
    rdna4::Graph g(ld, cfg);
    if (!g.init(n_pos + 8, r.k, r.v, err)) {
      std::printf("%-12s  graph init FAILED: %s\n", r.tag.c_str(), err.c_str());
      continue;
    }
    if (g.n_vocab() != ref_vocab) {
      std::fprintf(stderr, "%s: vocab changed\n", r.tag.c_str());
      return 1;
    }
    std::vector<float> h, logits;
    std::vector<float> ls((std::size_t)ref_vocab);
    double nll_sum = 0.0, kl_sum = 0.0, kl_max = 0.0, dnll_sum = 0.0, dnll_max = 0.0;
    int top1_diff = 0;
    for (int t = 0; t < n_pos; ++t) {
      if (!g.forward_tokens({ids[(std::size_t)t]}, t, h, logits, err)) {
        std::fprintf(stderr, "%s: forward at %d: %s\n", r.tag.c_str(), t, err.c_str());
        return 1;
      }
      double m = logits[0];
      for (float x : logits) m = std::max(m, (double)x);
      double sum = 0.0;
      for (float x : logits) sum += std::exp((double)x - m);
      const double logZ = m + std::log(sum);
      for (int v = 0; v < ref_vocab; ++v) {
        ls[(std::size_t)v] = (float)((double)logits[(std::size_t)v] - logZ);
      }
      const double nll = -(double)ls[(std::size_t)ids[(std::size_t)t + 1]];
      nll_sum += nll;

      const float *rl = ref_ls.data() + (std::size_t)t * ref_vocab;
      double kl = 0.0;
      int am_ref = 0, am_cur = 0;
      for (int v = 0; v < ref_vocab; ++v) {
        const double p = std::exp((double)rl[v]);
        kl += p * ((double)rl[v] - (double)ls[(std::size_t)v]);
        if (rl[v] > rl[am_ref]) am_ref = v;
        if (ls[(std::size_t)v] > ls[(std::size_t)am_cur]) am_cur = v;
      }
      kl_sum += kl;
      kl_max = std::max(kl_max, kl);
      if (am_cur != am_ref) ++top1_diff;

      const double dn = std::fabs(nll - ref_nll[(std::size_t)t]);
      dnll_sum += dn;
      dnll_max = std::max(dnll_max, dn);
      if ((t + 1) % 64 == 0) std::printf("   %-10s %d/%d positions\r", r.tag.c_str(), t + 1, n_pos);
    }
    r.n = n_pos;
    r.ppl = std::exp(nll_sum / (double)n_pos);
    r.mean_kl = kl_sum / (double)n_pos;
    r.max_kl = kl_max;
    r.mean_dnll = dnll_sum / (double)n_pos;
    r.max_dnll = dnll_max;
    r.top1_diff = top1_diff;
    r.have_kl = true;
    std::printf("%-12s %10.5f %12.6f %12.6f %12.6f %6d/%-3d\n", r.tag.c_str(), r.ppl, r.mean_kl,
                r.max_kl, r.mean_dnll, r.top1_diff, n_pos);
  }

  // ---- the two axes, spelled out -----------------------------------------
  auto find = [&](const std::string &tag) -> const Row * {
    for (const Row &r : rows) {
      if (r.tag == tag && r.have_kl) return &r;
    }
    return nullptr;
  };
  std::printf("\n-- K axis, V fixed at q4_1 (lower KL = better K) --\n");
  for (const char *t : {"q8_0:q4_1", "q5_0:q4_1", "q4_1:q4_1", "q4_0:q4_1"}) {
    const Row *r = find(t);
    if (r != nullptr) std::printf("   %-10s KL %9.6f  bits/token %7.5f  PPL %8.5f  top1 %d\n", t,
                                  r->mean_kl, r->mean_kl / std::log(2.0), r->ppl, r->top1_diff);
  }
  std::printf("-- V axis, K fixed at q5_0 (lower KL = better V) --\n");
  for (const char *t : {"q5_0:q8_0", "q5_0:q5_0", "q5_0:q4_1", "q5_0:q4_0"}) {
    const Row *r = find(t);
    if (r != nullptr) std::printf("   %-10s KL %9.6f  bits/token %7.5f  PPL %8.5f  top1 %d\n", t,
                                  r->mean_kl, r->mean_kl / std::log(2.0), r->ppl, r->top1_diff);
  }

  // A gate, not just a report: the night's default must not be worse than the
  // formats it replaces on the metric that is actually sensitive to the cache.
  {
    const Row *target = find("q5_0:q4_1");
    const Row *q40 = find("q4_0:q4_0");
    int failures = 0;
    if (target == nullptr) {
      std::printf("\nFAIL: q5_0:q4_1 was not measured\n");
      ++failures;
    } else {
      std::printf("\nq5_0/q4_1: KL %.6f nats (%.5f bits) per position, top-1 changed at %d/%d\n",
                  target->mean_kl, target->mean_kl / std::log(2.0), target->top1_diff, target->n);
      if (target->top1_diff > target->n / 20) {
        std::printf("FAIL: q5_0/q4_1 changes the greedy token at more than 5%% of positions\n");
        ++failures;
      }
      if (q40 != nullptr && target->mean_kl > q40->mean_kl) {
        std::printf("FAIL: q5_0/q4_1 has HIGHER KL than q4_0/q4_0 (%.6f vs %.6f) -- it costs "
                    "0.375 GiB more and is worse\n",
                    target->mean_kl, q40->mean_kl);
        ++failures;
      }
    }
    std::printf("check-kvquality-gpu: %s\n", failures ? "FAILED" : "OK");
    return failures ? 1 : 0;
  }
}
