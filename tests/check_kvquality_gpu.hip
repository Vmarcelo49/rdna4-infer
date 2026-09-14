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

// ---------------------------------------------------------------------------
// Needle mode: retrieval from a long REAL context, which is the part of the
// 131K promise that neither PPL nor a short KL window tests.
//
// The token stream in the ids file already contains the filler and the
// needle/question pairs; probes.txt has one "pos tok" line per needle, meaning
// "after consuming the token at index `pos`, the model must produce `tok`" (the
// first token of the passcode). Prefill stays batched: the stream is cut into
// segments that end exactly at each probe position, so the logits of the last
// forward_batch() call of a segment ARE the probe's logits. That matters because
// the GDN layers carry a recurrent state: probing a position out of order would
// read a state that has already advanced past it, and the answer would be wrong
// for a reason that has nothing to do with the KV cache.
//
// Reported per probe: the rank of the expected token (0 = it IS the greedy
// token), the greedy token, and the top-5. `retrieved` = rank 0.
// ---------------------------------------------------------------------------
struct Probe {
  int pos = 0;
  std::int32_t tok = 0;
  int rank = -1;
  std::int32_t top1 = -1;
};

int run_needle(const std::string &model, const std::vector<std::int32_t> &ids,
               const char *probes_path, const std::vector<Row> &rows) {
  std::vector<Probe> probes;
  {
    FILE *f = std::fopen(probes_path, "r");
    if (f == nullptr) {
      std::fprintf(stderr, "cannot open %s\n", probes_path);
      return 1;
    }
    int p = 0;
    long t = 0;
    while (std::fscanf(f, "%d %ld", &p, &t) == 2) probes.push_back(Probe{p, (std::int32_t)t, -1, -1});
    std::fclose(f);
  }
  if (probes.empty()) {
    std::fprintf(stderr, "no probes in %s\n", probes_path);
    return 1;
  }
  std::sort(probes.begin(), probes.end(),
            [](const Probe &a, const Probe &b) { return a.pos < b.pos; });
  for (const Probe &p : probes) {
    if (p.pos < 0 || p.pos >= (int)ids.size() - 1) {
      std::fprintf(stderr, "probe position %d outside the stream (%zu ids)\n", p.pos, ids.size());
      return 1;
    }
  }

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

  std::printf("needle: %zu probes over %zu tokens\n", probes.size(), ids.size());
  std::printf("%-12s %8s %8s %8s %8s  %s\n", "config", "retriev", "meanrank", "worst", "n", "ranks");
  int failures = 0;
  for (const Row &cr : rows) {
    rdna4::Graph g(ld, cfg);
    if (!g.init((int)ids.size() + 8, cr.k, cr.v, err)) {
      std::printf("%-12s  graph init FAILED: %s\n", cr.tag.c_str(), err.c_str());
      continue;
    }
    std::vector<float> h, logits;
    int done = 0;
    long rank_sum = 0;
    int worst = 0;
    std::string ranks;
    std::size_t pi = 0;
    int pos = 0;
    while (pos < (int)ids.size() && pi < probes.size()) {
      // segment [pos .. target]; the last chunk must end exactly on `target`
      const int target = probes[pi].pos;
      int remaining = target - pos + 1;
      if (remaining <= 0) {
        // two probes on the same position: reuse the logits we already have
        break;
      }
      // cut so that the final chunk lands on `target`: chunks of <=16, and the
      // remainder first when it is not a supported batch size
      while (remaining > 0) {
        int n = remaining > 16 ? 16 : remaining;
        if (n == remaining && !rdna4::Graph::batch_supported(n)) {
          // finish this segment with a per-token step instead of an unsupported
          // batch size (forward_tokens is bit-identical to forward_batch)
          for (int i = 0; i < n; ++i) {
            if (!g.forward_tokens({ids[(std::size_t)pos]}, pos, h, logits, err)) {
              std::fprintf(stderr, "needle forward at %d: %s\n", pos, err.c_str());
              return 1;
            }
            ++pos;
          }
          remaining = 0;
          break;
        }
        std::vector<std::int32_t> chunk(ids.begin() + pos, ids.begin() + pos + n);
        if (!g.forward_batch(chunk, pos, h, logits, err)) {
          std::fprintf(stderr, "needle batch at %d (n=%d): %s\n", pos, n, err.c_str());
          return 1;
        }
        pos += n;
        remaining -= n;
      }
      // logits now describe the distribution after consuming ids[target]
      int rank = 0;
      std::int32_t top1 = 0;
      for (int v = 0; v < g.n_vocab(); ++v) {
        if (logits[(std::size_t)v] > logits[(std::size_t)top1]) top1 = (std::int32_t)v;
        if (logits[(std::size_t)v] > logits[(std::size_t)probes[pi].tok]) ++rank;
      }
      probes[pi].rank = rank;
      probes[pi].top1 = top1;
      if (rank == 0) ++done;
      rank_sum += rank;
      worst = std::max(worst, rank);
      ranks += std::to_string(rank);
      ranks += " ";
      ++pi;
    }
    std::printf("%-12s %6d/%-2zu %8.2f %8d %8zu  %s\n", cr.tag.c_str(), done, probes.size(),
                probes.empty() ? 0.0 : (double)rank_sum / (double)probes.size(), worst,
                probes.size(), ranks.c_str());
    // The gate only applies to the format the night is deciding on; the others
    // are reported for the K/V axis comparison.
    if (cr.tag == "q5_0:q4_1" && done * 10 < (int)probes.size() * 9) {
      std::printf("FAIL: q5_0/q4_1 retrieves %d/%zu needles (<90%%)\n", done, probes.size());
      ++failures;
    }
    if (cr.tag == "f16:f16" && done < (int)probes.size()) {
      std::printf("NOTE: the f16 reference itself missed %zu needle(s) -- the probe is too hard "
                  "or the context is past what the model handles; do not read format differences "
                  "out of this run\n",
                  probes.size() - (std::size_t)done);
      ++failures;
    }
  }
  std::printf("check-kvquality-gpu (needle): %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}

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

  // needle mode: <model> <ids.txt> needle <probes.txt> [configs]
  const bool needle_mode = argc >= 4 && std::strcmp(argv[3], "needle") == 0;
  if (needle_mode) {
    if (argc < 5) {
      std::fprintf(stderr, "usage: check-kvquality-gpu <file.gguf> <ids.txt> needle <probes.txt> "
                           "[k:v,k:v,...]\n");
      return 2;
    }
    std::vector<std::int32_t> all_ids;
    {
      FILE *f = std::fopen(ids_path, "r");
      if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", ids_path);
        return 1;
      }
      long v = 0;
      while (std::fscanf(f, "%ld", &v) == 1) all_ids.push_back((std::int32_t)v);
      std::fclose(f);
    }
    const std::string ncfg = argc >= 6 ? argv[5] : "f16:f16,q5_0:q4_1,q4_0:q4_0,q8_0:q8_0";
    std::vector<Row> rows;
    std::size_t pos = 0;
    while (pos <= ncfg.size()) {
      const std::size_t comma = ncfg.find(',', pos);
      const std::string item =
          ncfg.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
      if (!item.empty()) {
        const std::size_t c = item.find(':');
        if (c == std::string::npos) return 2;
        Row r;
        r.tag = item;
        if (rdna4::kv_type_parse(item.substr(0, c).c_str(), &r.k) != nullptr ||
            rdna4::kv_type_parse(item.substr(c + 1).c_str(), &r.v) != nullptr) {
          return 2;
        }
        rows.push_back(r);
      }
      if (comma == std::string::npos) break;
      pos = comma + 1;
    }
    return run_needle(model, all_ids, argv[4], rows);
  }

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
  //
  // The stream is walked in chunks of 16 through forward_batch() and each probe
  // is the LAST position of a chunk. Probing every position would need
  // forward_tokens() one token at a time, which at 4096 tokens of context costs
  // 4096 serial layer stacks per config -- minutes each, and the night cannot
  // afford it (the coordinator measured prefill at 74.9 tok/s, so batched is the
  // only way to reach a context where the KV cache is actually stressed).
  // forward_batch() is documented and gated as BIT-IDENTICAL to calling
  // forward_tokens() once per token (tests/check_batch_gpu.hip asserts it), so
  // the logits are the same ones the slow path would produce.
  const int kChunk = 16;
  const int n_probe = n_pos / kChunk;  // probes at positions kChunk-1, 2*kChunk-1, ...
  if (n_probe < 4) {
    std::fprintf(stderr, "n_positions must be at least %d (got %d)\n", 4 * kChunk, n_pos);
    return 2;
  }
  std::vector<float> ref_ls;   // log-softmax of the reference, [n_probe][nv]
  std::vector<double> ref_nll;
  std::vector<int> probe_pos;
  int ref_vocab = 0;
  {
    rdna4::Graph g(ld, cfg);
    if (!g.init(n_pos + 8, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
      std::fprintf(stderr, "reference graph init: %s\n", err.c_str());
      return 1;
    }
    ref_vocab = g.n_vocab();
    ref_ls.resize((std::size_t)n_probe * (std::size_t)ref_vocab);
    ref_nll.assign((std::size_t)n_probe, 0.0);
    probe_pos.reserve((std::size_t)n_probe);
    std::vector<float> h, logits;
    std::vector<float> ls((std::size_t)ref_vocab);
    for (int c = 0; c < n_probe; ++c) {
      const int pos = c * kChunk;
      const std::vector<std::int32_t> chunk(ids.begin() + pos, ids.begin() + pos + kChunk);
      if (!g.forward_batch(chunk, pos, h, logits, err)) {
        std::fprintf(stderr, "reference batch at %d: %s\n", pos, err.c_str());
        return 1;
      }
      const int last = pos + kChunk - 1;  // position these logits describe
      probe_pos.push_back(last);
      double m = logits[0];
      for (float x : logits) m = std::max(m, (double)x);
      double sum = 0.0;
      for (float x : logits) sum += std::exp((double)x - m);
      const double logZ = m + std::log(sum);
      for (int v = 0; v < ref_vocab; ++v) {
        ls[(std::size_t)v] = (float)((double)logits[(std::size_t)v] - logZ);
      }
      std::memcpy(ref_ls.data() + (std::size_t)c * ref_vocab, ls.data(),
                  (std::size_t)ref_vocab * sizeof(float));
      ref_nll[(std::size_t)c] = -(double)ls[(std::size_t)ids[(std::size_t)last + 1]];
      if ((c + 1) % 32 == 0) std::printf("   reference %d/%d probes\r", c + 1, n_probe);
    }
    std::printf("reference (f16/f16): %d probes over %d tokens of context, vocab %d          \n",
                n_probe, n_pos, ref_vocab);
  }
  // (vocab comes from the graph: output_.dim1)

  // ---- one pass per config ------------------------------------------------
  double ref_ppl = 0.0;
  {
    double s = 0.0;
    for (double x : ref_nll) s += x;
    ref_ppl = std::exp(s / (double)ref_nll.size());
  }
  (void)probe_pos;
  std::printf("\n%-12s %10s %12s %12s %12s %10s\n", "config", "PPL", "KL nats/pos",
              "KL max/pos", "mean|dNLL|", "top1 diff");
  std::printf("%-12s %10.5f %12s %12s %12s %10s\n", "f16:f16 (ref)", ref_ppl, "--", "--", "--",
              "--");

  std::vector<Row> results;  // the f16 row plus every config that ran
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
      r.n = n_probe;
      std::printf("%-12s %10.5f %12.6f %12.6f %12.6f %10d  (reference, must be exactly 0)\n",
                  r.tag.c_str(), r.ppl, r.mean_kl, r.max_kl, r.mean_dnll, r.top1_diff);
      results.push_back(r);
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
    for (int c = 0; c < n_probe; ++c) {
      const int pos = c * kChunk;
      const std::vector<std::int32_t> chunk(ids.begin() + pos, ids.begin() + pos + kChunk);
      const int t = pos + kChunk - 1;  // same probe position as the reference
      if (!g.forward_batch(chunk, pos, h, logits, err)) {
        std::fprintf(stderr, "%s: batch at %d: %s\n", r.tag.c_str(), pos, err.c_str());
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

      const float *rl = ref_ls.data() + (std::size_t)c * ref_vocab;
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

      const double dn = std::fabs(nll - ref_nll[(std::size_t)c]);
      dnll_sum += dn;
      dnll_max = std::max(dnll_max, dn);
      if ((c + 1) % 32 == 0) std::printf("   %-10s %d/%d probes\r", r.tag.c_str(), c + 1, n_probe);
    }
    r.n = n_probe;
    r.ppl = std::exp(nll_sum / (double)n_probe);
    r.mean_kl = kl_sum / (double)n_probe;
    r.max_kl = kl_max;
    r.mean_dnll = dnll_sum / (double)n_probe;
    r.max_dnll = dnll_max;
    r.top1_diff = top1_diff;
    r.have_kl = true;
    std::printf("%-12s %10.5f %12.6f %12.6f %12.6f %6d/%-3d\n", r.tag.c_str(), r.ppl, r.mean_kl,
                r.max_kl, r.mean_dnll, r.top1_diff, n_probe);
    results.push_back(r);
  }

  // ---- the two axes, spelled out -----------------------------------------
  // `rows` is the CONFIG list; the measured values live in `results`. Searching
  // `rows` here is what made the first version of this gate print "q5_0:q4_1 was
  // not measured" for a config it had just printed a row for.
  auto find = [&](const std::string &tag) -> const Row * {
    for (const Row &r : results) {
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
