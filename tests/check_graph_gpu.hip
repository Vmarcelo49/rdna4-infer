// M3 step 3 — the assembled qwen35 forward graph vs the llama.cpp oracle.
//
// The eval-callback dump (reference/oracle_prompt6_cpu.txt) prints every named
// graph node with its first/last three values and its element sum, so the whole
// graph can be validated node by node without a second implementation: the
// Graph class emits the same node names through set_node_cb().
//
// Nodes are accumulated across the token loop (the reference evaluates all 8
// tokens of the prompt in one batch, so a node's sum covers every token).
// `state_predelta-*` is the state entering the batch (token 0 only) and
// `new_state-*` the state leaving it (last token).
//
// usage: check-graph-gpu <file.gguf> <oracle_dump.txt> [token ids...]
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);

namespace {

struct OracleNode {
  std::vector<float> sampled;  // v0,v1,v2, v[n-3],v[n-2],v[n-1]
  double sum = 0.0;
  std::string shape;
  bool found = false;
};

// One pass over the dump building every node at once (the file is ~93k lines).
std::map<std::string, OracleNode> parse_oracle(const char *path) {
  std::map<std::string, OracleNode> out;
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open oracle dump %s\n", path);
    return out;
  }
  const std::string needle = "common_debug_cb_eval:";
  std::string line;
  while (std::getline(f, line)) {
    if (line.find(needle) == std::string::npos) continue;
    const std::size_t eq = line.find(" = ");
    if (eq == std::string::npos) continue;
    std::string label = line.substr(needle.size(), eq - needle.size());
    while (!label.empty() && label.front() == ' ') label.erase(label.begin());
    while (!label.empty() && label.back() == ' ') label.pop_back();

    OracleNode node;
    node.found = true;
    // the result shape is the LAST {..} on the line (earlier ones belong to the
    // op description, e.g. the source tensor of a VIEW)
    const std::size_t rb = line.rfind('}');
    const std::size_t lb = rb == std::string::npos ? std::string::npos : line.rfind('{', rb);
    if (lb != std::string::npos && rb != std::string::npos) node.shape = line.substr(lb, rb - lb + 1);

    std::vector<float> vals;
    while (std::getline(f, line)) {
      const std::size_t sp = line.find("sum =");
      if (sp != std::string::npos) {
        node.sum = std::strtod(line.c_str() + sp + 5, nullptr);
        break;
      }
      const char *p = line.c_str();
      char *endp = nullptr;
      while (*p) {
        const double v = std::strtod(p, &endp);
        if (endp == p) {
          ++p;
          continue;
        }
        vals.push_back((float)v);
        p = endp;
      }
    }
    if (vals.size() >= 6) {
      node.sampled.assign(vals.begin(), vals.begin() + 3);
      node.sampled.insert(node.sampled.end(), vals.end() - 3, vals.end());
    } else {
      node.sampled = vals;
    }
    out[label] = std::move(node);
  }
  return out;
}

// Accumulates what the graph emits so the comparison sees oracle-shaped nodes.
struct Accum {
  const std::map<std::string, OracleNode> *oracle = nullptr;
  int n_tokens = 1;  // current token index arrives in operator()
  // insertion order, for reporting the first divergence in graph order
  std::vector<std::string> order;
  std::map<std::string, std::vector<float>> data;
  bool print_samples = false;
  bool last_token_only = false;
  int n_mismatch = 0;
  int n_bad_sample = 0;
  int n_bad_sum = 0;
  int n_compared = 0;
  double worst_rel = 0.0;
  std::string worst_name;
  bool verbose = false;

  std::string key(const char *name, int il) const {
    if (il < 0) return name;
    return std::string(name) + "-" + std::to_string(il);
  }

  void operator()(const char *name, int il, int t, const float *d_ptr, std::int64_t n) {
    std::vector<float> host((std::size_t)n);
    (void)hipMemcpy(host.data(), d_ptr, (std::size_t)n * sizeof(float), hipMemcpyDeviceToHost);
    const std::string k = key(name, il);

    // `last_token_only` matches a dump captured with -ub 1, where llama.cpp
    // evaluates one token per graph — the same shape of computation our engine
    // does (per-token matvec), as opposed to its batched prefill (which uses a
    // different MUL_MAT path and, measured, lands ~1-4% away from its own
    // single-token results). Then every label's last occurrence is the final
    // prompt token, so we only keep token n-1.
    if (last_token_only && t != n_tokens - 1) return;
    // Batch-level nodes: the state entering / leaving the batch.
    if (std::strcmp(name, "state_predelta") == 0) {
      const int want = last_token_only ? n_tokens - 1 : 0;
      if (t != want) return;
    }
    if (std::strcmp(name, "new_state") == 0 && t != n_tokens - 1) return;

    auto it = data.find(k);
    if (it == data.end()) {
      order.push_back(k);
      it = data.emplace(k, std::vector<float>()).first;
    }
    it->second.insert(it->second.end(), host.begin(), host.end());
  }

  // Comparison happens only once the whole token loop finished: a node is
  // complete then, and the oracle's samples are first-3-of-token-0 and
  // last-3-of-token-(n-1).
  void compare_all() {
    for (const std::string &k : order) compare(k, data[k]);
  }

  void compare(const std::string &k, const std::vector<float> &v) {
    if (!oracle) return;
    if (k.rfind("diag.", 0) == 0) {
      // pure diagnostics: no oracle counterpart, report the vector's statistics
      double ss = 0.0;
      float vmin = v[0], vmax = v[0];
      for (float x : v) {
        ss += (double)x * (double)x;
        vmin = std::min(vmin, x);
        vmax = std::max(vmax, x);
      }
      std::printf("diag      %-24s rms=%.4f min=%.4f max=%.4f\n", k.c_str(),
                  std::sqrt(ss / (double)v.size()), vmin, vmax);
      return;
    }
    auto o = oracle->find(k);
    if (o == oracle->end()) {
      std::printf("MISSING-ORACLE %s (gpu elems=%zu)\n", k.c_str(), v.size());
      return;
    }
    const OracleNode &r = o->second;
    ++n_compared;
    double sum = 0.0;
    for (float x : v) sum += (double)x;
    const double denom = std::max(1.0, std::fabs(r.sum));
    const double rel = std::fabs(sum - r.sum) / denom;

    // the dump prints the first three and the last three values of the node
    double max_adiff = 0.0;
    if (r.sampled.size() == 6 && v.size() >= 6) {
      for (std::size_t i = 0; i < 3; ++i) {
        max_adiff = std::max(max_adiff, (double)std::fabs((double)v[i] - (double)r.sampled[i]));
        max_adiff = std::max(
            max_adiff, (double)std::fabs((double)v[v.size() - 3 + i] - (double)r.sampled[3 + i]));
      }
    }
    double max_ref = 1e-3;
    for (float x : r.sampled) max_ref = std::max(max_ref, (double)std::fabs(x));
    // Two independent criteria, because neither works everywhere: the element
    // sum is meaningless for nodes whose values cancel (roughly half of them,
    // e.g. post-RoPE Q), while the sampled values are limited by the dump's 4
    // decimals for nodes whose samples are small.
    const double sample_rel = std::max(0.0, max_adiff - 1e-4) / max_ref;
    if (sample_rel > 0.2) ++n_bad_sample;
    if (rel > 1e-2) ++n_bad_sum;
    if (rel > worst_rel) {
      worst_rel = rel;
      worst_name = k;
    }
    const bool bad = rel > 1e-3 || max_adiff > 1e-2;
    if (bad && print_samples && r.sampled.size() == 6 && v.size() >= 6) {
      std::printf("            ref:");
      for (float x : r.sampled) std::printf(" %+.5f", x);
      std::printf("\n            gpu:");
      for (int i = 0; i < 3; ++i) std::printf(" %+.5f", v[i]);
      for (std::size_t i = v.size() - 3; i < v.size(); ++i) std::printf(" %+.5f", v[i]);
      std::printf("\n");
    }
    if (bad) {
      ++n_mismatch;
      std::printf("MISMATCH  %-28s %-24s sum=%+.6f ref=%+.6f rel=%.3e  max_adiff=%.3e (n=%zu)\n",
                  k.c_str(), r.shape.c_str(), sum, r.sum, rel, max_adiff, v.size());
    } else if (verbose) {
      std::printf("ok        %-28s %-24s sum=%+.6f ref=%+.6f rel=%.3e  max_adiff=%.3e\n",
                  k.c_str(), r.shape.c_str(), sum, r.sum, rel, max_adiff);
    }
  }
};

// Exact f32/double reference for layer 0's fused qkv projection:
//   y[r] = sum_c W[r][c] * x[c]
// W is dequantized on the host into f32 and x is the graph's own attn_norm-0
// output. The point is to separate two very different error sources: OUR
// activation quantization (q8_1, 32-element blocks) and the oracle's (q8_K,
// 256-element blocks for K-quants). Both are compared against the exact dot,
// so whichever is closer to `exact` is the more accurate engine.
bool exact_check(const rdna4::GgufLoader &ld, const rdna4::Qwen35Config &cfg,
                 const Accum &acc, int E, int n_tokens) {
  (void)cfg;
  const auto *wi = ld.find("blk.0.attn_qkv.weight");
  if (!wi) return false;
  const std::size_t ti = (std::size_t)(wi - ld.meta().tensors.data());
  const int C = (int)wi->dims[0];  // reduction dim
  const int R = (int)wi->dims[1];  // rows (output dim)
  if (C != E) return false;

  std::vector<std::uint8_t> raw;
  std::string err;
  if (!ld.load_tensor(ti, raw, err)) {
    std::fprintf(stderr, "exact: %s\n", err.c_str());
    return false;
  }
  std::vector<float> wf((std::size_t)R * C);
  if (!rdna4_cpu_dequant_row((int)ld.dtype(ti), raw.data(), wf.data(), (std::int64_t)R * C)) {
    std::fprintf(stderr, "exact: dequant failed\n");
    return false;
  }

  auto xit = acc.data.find("attn_norm-0");
  auto yit = acc.data.find("linear_attn_qkv_mixed-0");
  if (xit == acc.data.end() || yit == acc.data.end()) {
    std::fprintf(stderr, "exact: nodes missing\n");
    return false;
  }
  const std::vector<float> &x = xit->second;  // [t][E]
  const std::vector<float> &y = yit->second;  // [t][R]

  const std::size_t t_last = (std::size_t)n_tokens - 1;
  const float *xt = x.data() + t_last * (std::size_t)E;
  const float *yt = y.data() + t_last * (std::size_t)R;

  std::vector<double> exact(R);
  double l2_diff = 0.0, l2_ref = 0.0;
  for (int r = 0; r < R; ++r) {
    const float *wr = wf.data() + (std::size_t)r * C;
    double a = 0.0;
    for (int c = 0; c < C; ++c) a += (double)wr[c] * (double)xt[c];
    exact[r] = a;
    const double d = a - (double)yt[r];
    l2_diff += d * d;
    l2_ref += a * a;
  }
  double sum_gpu = 0.0, sum_exact = 0.0;
  for (int r = 0; r < R; ++r) {
    sum_gpu += (double)yt[r];
    sum_exact += exact[r];
  }

  std::printf("exact-f32 check: blk.0.attn_qkv.weight (%s) x attn_norm-0, last token %zu\n",
              rdna4::dtype_name(ld.dtype(ti)), t_last);
  std::printf("  rows=%d cols=%d  sum: gpu=%+.6f exact=%+.6f  rel=%.3e\n", R, C, sum_gpu, sum_exact,
              std::fabs(sum_gpu - sum_exact) / std::fabs(sum_exact));
  std::printf("  L2 error vs exact: ||gpu-exact||/||exact|| = %.3e\n",
              std::sqrt(l2_diff / l2_ref));

  auto o = acc.oracle ? acc.oracle->find("linear_attn_qkv_mixed-0")
                      : std::map<std::string, OracleNode>::const_iterator{};
  if (acc.oracle && o != acc.oracle->end()) {
    // first 3 rows belong to token 0, last 3 to the last token
    const float *x0 = x.data();
    const float *y0 = y.data();
    double o_sq = 0.0, g_sq = 0.0;
    for (int i = 0; i < 6; ++i) {
      const bool first = i < 3;
      const int r = first ? i : R - 3 + (i - 3);
      const std::size_t tk = first ? 0 : t_last;
      const float *xx = x.data() + tk * (std::size_t)E;
      double a = 0.0;
      const float *wr = wf.data() + (std::size_t)r * C;
      for (int c = 0; c < C; ++c) a += (double)wr[c] * (double)xx[c];
      const double g = (double)(first ? y0[r] : yt[r]);
      std::printf("  row %5d (tok %zu): exact=%+.6f gpu=%+.6f oracle=%+.6f  |gpu-exact|=%.2e |oracle-exact|=%.2e\n",
                  r, tk, a, g, (double)o->second.sampled[i], std::fabs(g - a),
                  std::fabs((double)o->second.sampled[i] - a));
      o_sq += ((double)o->second.sampled[i] - a) * ((double)o->second.sampled[i] - a);
      g_sq += (g - a) * (g - a);
    }
    std::printf("  over those 6 rows: rms(gpu-exact)=%.3e rms(oracle-exact)=%.3e\n", std::sqrt(g_sq / 6.0),
                std::sqrt(o_sq / 6.0));
  }
  return std::sqrt(l2_diff / l2_ref) < 1e-3;
}

double json_double(const std::string &s, const char *key) {
  const std::string k = std::string("\"") + key + "\"";
  const std::size_t p = s.find(k);
  if (p == std::string::npos) return 0.0;
  const std::size_t c = s.find(':', p + k.size());
  if (c == std::string::npos) return 0.0;
  return std::strtod(s.c_str() + c + 1, nullptr);
}

}  // namespace

// ---------------------------------------------------------------------------
// Structural checks. These are the M3 acceptance criteria: they pin the nodes
// whose inputs are still accurate (the first GDN layer, the first full-attention
// layer, the recurrent state) plus the end-to-end argmax, and they are the
// reason the test has a pass/fail exit code at all.
//
// They are deliberately NOT "every node matches": deeper nodes drift because
// our activation quantization (q8_1, 32-element blocks) differs from llama.cpp's
// CPU one (q8_K, 256-element blocks) and the deviation is amplified ~1.06x per
// layer over 63 layers. The exact-f32 cross-check (GRAPH_EXACT=1) shows which
// side is closer to the truth: ours is 2.4x more accurate than the CPU oracle
// on the very projections that dominate that noise.
// ---------------------------------------------------------------------------
struct Check {
  const char *name;
  double max_sample_rel;  // sample-wise, relative to the largest reference sample
  double max_sum_rel;     // relative error of the element sum (0 = not checked)
  double max_abs;         // absolute bound for the state nodes (0 = not a state)
  bool state = false;
};

// Thresholds are set so that the failures this validation actually caught are
// caught with margin, while the residual drift stays inside:
//   - the reference's own per-node method variance (its batched prefill and its
//     per-token decode differ by ~4% at layer 3),
//   - the activation-quantization difference (ours is q8_1/32-element blocks,
//     the CPU oracle's is q8_K/256-element blocks; measured against exact f32
//     ours is 2.4x closer),
// amplified over 63 layers. The two real bugs found here were a RoPE pair
// convention (Qcur-3 sample metric 1.57 -> 0.16) and the GQA head mapping plus
// a query-indexing bug (attn_pregate-3 sum rel 3.2e-02 -> 1.05e-04).
bool run_checks(const Accum &acc, const std::map<std::string, OracleNode> &oracle) {
  static const Check kChecks[] = {
      // deterministic prefix of the graph: a dequant and an RMS norm
      {"model.input_embed", 0.05, 1e-5, 0.0, false},
      {"attn_norm-0", 0.05, 1e-4, 0.0, false},
      // first GDN layer (block 0)
      {"linear_attn_qkv_mixed-0", 0.2, 1e-2, 0.0, false},
      {"beta-0", 0.05, 1e-4, 0.0, false},
      {"beta_sigmoid-0", 0.05, 1e-4, 0.0, false},
      {"a_softplus-0", 0.05, 1e-4, 0.0, false},
      {"gate-0", 0.05, 1e-4, 0.0, false},
      {"z-0", 0.2, 1e-2, 0.0, false},
      {"conv_output_silu-0", 0.2, 1e-2, 0.0, false},
      {"q_conv-0", 0.2, 1e-2, 0.0, false},
      {"k_conv-0", 0.2, 1e-2, 0.0, false},
      {"v_conv_predelta-0", 0.2, 1e-2, 0.0, false},
      {"final_output-0", 0.2, 1e-2, 0.0, false},
      {"linear_attn_out-0", 0.2, 1e-2, 0.0, false},
      {"l_out-0", 0.2, 1e-2, 0.0, false},
      // the recurrent state after the prompt: conv1d state, L2 norms, the delta
      // rule and the decay all have to be right for this to hold
      {"new_state-0", 0.0, 0.0, 1e-3, true},
      {"new_state-1", 0.0, 0.0, 1e-3, true},
      {"new_state-2", 0.0, 0.0, 1e-3, true},
      // first full-attention layer (block 3): Q/gate split, QK-norm, RoPE
      {"Qcur_full-3", 0.2, 1e-2, 0.0, false},
      {"gate_reshaped-3", 0.2, 1e-2, 0.0, false},
      {"gate_sigmoid-3", 0.2, 0.2, 0.0, false},
      {"Qcur_normed-3", 0.3, 0.3, 0.0, false},
      {"Qcur-3", 0.3, 0.3, 0.0, false},
      {"Kcur_normed-3", 0.3, 0.3, 0.0, false},
      {"Vcur-3", 0.3, 0.3, 0.0, false},
      // attention itself: the sharpest check in the graph (a wrong GQA mapping
      // or query indexing shows up here immediately)
      {"attn_pregate-3", 0.2, 5e-3, 0.0, false},
      {"attn_gated-3", 0.2, 0.2, 0.0, false},
      {"attn_output-3", 0.2, 0.2, 0.0, false},
      {"l_out-3", 0.2, 5e-2, 0.0, false},
      // last trunk layer and the end-to-end outputs. `l_out-63` existing at all
      // is itself a check: the trunk is 64 blocks (65 - 1 MTP), so a wrong
      // layer count shows up as CHECK MISSING here.
      {"attn_norm-62", 0.2, 0.1, 0.0, false},
      {"l_out-63", 0.2, 0.2, 0.0, false},
      {"result_norm", 0.1, 0.2, 0.0, false},
      {"result_output", 0.1, 5e-2, 0.0, false},
  };

  int failed = 0;
  for (const Check &c : kChecks) {
    auto g = acc.data.find(c.name);
    auto o = oracle.find(c.name);
    if (g == acc.data.end() || o == oracle.end()) {
      std::printf("CHECK MISSING %s (gpu=%d oracle=%d)\n", c.name, g != acc.data.end(),
                  o != oracle.end());
      ++failed;
      continue;
    }
    const std::vector<float> &v = g->second;
    double sum = 0.0;
    for (float x : v) sum += (double)x;
    const double rel_sum = std::fabs(sum - o->second.sum) / std::max(1.0, std::fabs(o->second.sum));
    double max_adiff = 0.0, max_ref = 1e-3;
    if (o->second.sampled.size() == 6 && v.size() >= 6) {
      for (std::size_t i = 0; i < 3; ++i) {
        max_adiff = std::max(max_adiff, (double)std::fabs((double)v[i] - (double)o->second.sampled[i]));
        max_adiff = std::max(max_adiff, (double)std::fabs((double)v[v.size() - 3 + i] -
                                                          (double)o->second.sampled[3 + i]));
      }
      for (float x : o->second.sampled) max_ref = std::max(max_ref, (double)std::fabs(x));
    }
    // The dump prints 4 decimals, so a node whose samples are small cannot be
    // validated better than ~1e-4 absolute; subtract that slack before
    // normalising, otherwise "small sample" nodes look like failures.
    const double slack = 1e-4;
    const double sample_metric = std::max(0.0, max_adiff - slack) / max_ref;
    bool ok = c.state ? (max_adiff <= c.max_abs) : (sample_metric <= c.max_sample_rel);
    if (!c.state && c.max_sum_rel > 0.0 && rel_sum > c.max_sum_rel) ok = false;
    if (!ok) ++failed;
    std::printf("%-5s %-24s sample=%.3e/%s sum=%.3e%s\n", ok ? "ok" : "FAIL", c.name, sample_metric,
                c.state ? "-" : std::to_string(c.max_sample_rel).c_str(), rel_sum,
                c.max_sum_rel > 0.0 ? ("/" + std::to_string(c.max_sum_rel)).c_str() : "");
  }
  return failed == 0;
}

// Reads the reference tool's output (scripts/capture_oracle.sh writes it next to
// the dump) and compares the argmax plus the top-5 token ids.
bool check_argmax(const char *path, const std::vector<float> &logits) {
  std::ifstream f(path);
  if (!f) {
    std::fprintf(stderr, "cannot open argmax reference %s\n", path);
    return false;
  }
  long ref_argmax = -1;
  std::vector<long> ref_top5;
  std::string line;
  while (std::getline(f, line)) {
    if (line.rfind("argmax =", 0) == 0) ref_argmax = std::strtol(line.c_str() + 8, nullptr, 10);
    if (line.rfind("top5:", 0) == 0) {
      const char *p = line.c_str() + 5;
      while (*p) {
        char *end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) {
          ++p;
          continue;
        }
        ref_top5.push_back(v);
        p = end;
      }
    }
  }
  if (ref_argmax < 0 || logits.empty()) {
    std::fprintf(stderr, "argmax reference %s has no argmax line\n", path);
    return false;
  }
  std::vector<std::size_t> idx(logits.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
  std::printf("argmax: reference=%ld ours=%zu (%s)  logit ours=%.4f\n", ref_argmax, idx[0],
              (long)idx[0] == ref_argmax ? "MATCH" : "DIFFERS", logits[idx[0]]);
  int overlap = 0;
  std::printf("top5 : reference:");
  for (long v : ref_top5) std::printf(" %ld", v);
  std::printf(" | ours:");
  for (int i = 0; i < 5; ++i) std::printf(" %zu", idx[i]);
  for (int i = 0; i < 5; ++i) {
    for (long v : ref_top5) {
      if ((long)idx[i] == v) ++overlap;
    }
  }
  std::printf("  (overlap %d/5)\n", overlap);
  return (long)idx[0] == ref_argmax;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: check-graph-gpu <file.gguf> <oracle_dump|-> <argmax_ref|-> [token ids...]\n");
    return 2;
  }
  const bool use_oracle = std::strcmp(argv[2], "-") != 0;
  const bool use_argmax = std::strcmp(argv[3], "-") != 0;
  std::vector<int> tokens;
  for (int i = 4; i < argc; ++i) tokens.push_back(std::atoi(argv[i]));
  if (tokens.empty()) tokens = {9419, 1814, 11, 411, 369, 264, 1228, 13};

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open %s: %s\n", argv[1], err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(ld.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }

  const std::size_t emb_i = (std::size_t)(ld.find("token_embd.weight") - ld.meta().tensors.data());
  const rdna4::DType emb_dt = ld.dtype(emb_i);
  const std::uint64_t row_bytes = rdna4::tensor_bytes(emb_dt, cfg.embedding_length);
  const int E = (int)cfg.embedding_length;

  std::vector<float> embeddings((std::size_t)tokens.size() * E, 0.0f);
  for (std::size_t t = 0; t < tokens.size(); ++t) {
    std::vector<std::uint8_t> raw;
    if (!ld.load_tensor_range(emb_i, (std::uint64_t)tokens[t] * row_bytes, row_bytes, raw, err)) {
      std::fprintf(stderr, "embedding row %d: %s\n", tokens[t], err.c_str());
      return 1;
    }
    if (!rdna4_cpu_dequant_row((int)emb_dt, raw.data(), embeddings.data() + t * (std::size_t)E, E)) {
      std::fprintf(stderr, "dequant embedding row %d failed\n", tokens[t]);
      return 1;
    }
  }
  std::printf("model: n_embd=%d n_layer=%u vocab=%llu emb_dtype=%s tokens=%zu\n", E,
              cfg.block_count - cfg.nextn_predict_layers, (unsigned long long)ld.find("output.weight")->dims[1],
              rdna4::dtype_name(emb_dt), tokens.size());

  // GRAPH_PERTURB=<rel>: multiply every embedding element by (1 +- rel) with a
  // pseudo-random sign pattern. This measures how much a per-layer-magnitude
  // perturbation is amplified by 63 layers — the yardstick for how closely two
  // different (but both valid) quantization paths can be expected to agree.
  if (const char *ps = std::getenv("GRAPH_PERTURB")) {
    const float rel = (float)std::atof(ps);
    for (std::size_t i = 0; i < embeddings.size(); ++i) {
      const std::uint32_t h = (std::uint32_t)i * 2654435761u;
      const float sgn = ((h >> 16) & 1u) ? 1.0f : -1.0f;
      embeddings[i] *= (1.0f + rel * sgn);
    }
    std::printf("perturbed embeddings by +-%.3g (rel)\n", (double)rel);
  }

  std::map<std::string, OracleNode> oracle;
  if (use_oracle) {
    oracle = parse_oracle(argv[2]);
    std::printf("oracle: %zu nodes\n", oracle.size());
  }

  rdna4::KvType kv_k = rdna4::KvType::F32, kv_v = rdna4::KvType::F32;
  if (const char *kv = std::getenv("GRAPH_KV")) {
    const char *e = rdna4::kv_type_parse(kv, &kv_k);
    kv_v = kv_k;
    if (e) {
      std::fprintf(stderr, "GRAPH_KV: %s\n", e);
      return 2;
    }
    std::printf("kv cache type: %s (%.2f bytes/elem)\n", rdna4::kv_type_name(kv_k),
                rdna4::kv_bytes_per_elem(kv_k));
  }
  rdna4::Graph g(ld, cfg);
  if (!g.init((int)tokens.size(), kv_k, kv_v, err)) {
    std::fprintf(stderr, "graph init: %s\n", err.c_str());
    return 1;
  }

  Accum acc;
  acc.oracle = use_oracle ? &oracle : nullptr;
  acc.n_tokens = (int)tokens.size();
  acc.verbose = std::getenv("GRAPH_VERBOSE") != nullptr;
  acc.print_samples = std::getenv("GRAPH_SAMPLES") != nullptr;
  acc.last_token_only = std::getenv("GRAPH_LAST_TOKEN") != nullptr;
  if (const char *nr = std::getenv("GRAPH_NOISE")) {
    const bool coherent = std::getenv("GRAPH_NOISE_COHERENT") != nullptr;
    g.set_layer_noise((float)std::atof(nr), coherent);
    std::printf("per-layer output perturbation: %s %.3g (rel)\n", coherent ? "scale" : "+-",
                std::atof(nr));
  }
  g.set_node_cb([&acc](const char *name, int il, int t, const float *d, std::int64_t n) {
    acc(name, il, t, d, n);
  });

  // One call, all 8 tokens: the reference evaluates the prompt as a single
  // batch and the recurrent layers walk it token by token internally, so the
  // node contents line up element for element.
  std::vector<float> hidden, logits;
  if (!g.forward(embeddings, hidden, logits, err)) {
    std::fprintf(stderr, "forward: %s\n", err.c_str());
    return 1;
  }

  // Optional exact-f32 check of the first matmul: the oracle's CPU kernels
  // quantize activations one way (q8_K, 256-element blocks for K-quants) and we
  // quantize them another (q8_1, 32-element blocks), so node values can never
  // match bit for bit. This measures OUR error against an exact reference, so
  // the residual difference from the oracle can be attributed.
  bool exact_ok = true;
  if (std::getenv("GRAPH_EXACT") != nullptr) {
    exact_ok = exact_check(ld, cfg, acc, E, (int)tokens.size());
  }

  acc.compare_all();
  std::printf("node comparison: %d compared | %d nodes with sampled values off by >20%% | "
              "%d nodes with a sum error >1e-2 | worst sum rel %.3e (%s)\n",
              acc.n_compared, acc.n_bad_sample, acc.n_bad_sum, acc.worst_rel,
              acc.worst_name.c_str());
  // first divergence in graph order: everything before it matches the oracle
  // within the reporting threshold, so a wrong layout would show up here.
  if (use_oracle) {
    for (const std::string &k : acc.order) {
      auto o = oracle.find(k);
      if (o == oracle.end()) continue;
      double sum = 0.0;
      for (float x : acc.data[k]) sum += (double)x;
      const double rel = std::fabs(sum - o->second.sum) / std::max(1.0, std::fabs(o->second.sum));
      if (rel > 1e-3) {
        std::printf("first node beyond threshold: %s (sum rel %.3e)\n", k.c_str(), rel);
        break;
      }
    }
  }

  if (!logits.empty()) {
    double sum = 0.0;
    for (float v : logits) sum += (double)v;
    const double mean = sum / (double)logits.size();
    double var = 0.0;
    float vmin = logits[0], vmax = logits[0];
    for (float v : logits) {
      const double d = (double)v - mean;
      var += d * d;
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
    std::vector<std::size_t> idx(logits.size());
    for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
    std::printf("logits: sum=%.6f mean=%.6f std=%.6f min=%.4f max=%.4f over %zu entries\n", sum, mean,
                std::sqrt(var / (double)logits.size()), vmin, vmax, logits.size());
    std::printf("argmax = %zu (%.6f)\n", idx[0], logits[idx[0]]);
    std::printf("top5:");
    for (int i = 0; i < 5; ++i) std::printf(" %zu(%.4f)", idx[i], logits[idx[i]]);
    std::printf("\n");
    std::printf("result_norm samples (first3, last3):");
    for (int i = 0; i < 3 && (std::size_t)i < hidden.size(); ++i) std::printf(" %.4f", hidden[i]);
    for (std::size_t i = hidden.size() > 3 ? hidden.size() - 3 : 0; i < hidden.size(); ++i)
      std::printf(" %.4f", hidden[i]);
    std::printf("\n");
  }
  bool ok = true;
  if (use_oracle) {
    // The structural checks are calibrated against a per-token (-ub 1) capture,
    // which is the same shape of computation this engine does. Against a
    // batched-prefill capture the reference itself lands a few percent away
    // from its own single-token results, so the checks would be measuring
    // llama.cpp's path difference instead of our engine.
    auto probe = oracle.find("attn_norm-0");
    const bool batched = probe != oracle.end() && probe->second.shape.find(", 8, 1, 1}") != std::string::npos;
    if (batched) {
      std::printf("\nstructural checks SKIPPED: dump is a batched-prefill capture; "
                  "re-capture with -ub 1 (scripts/capture_oracle.sh, see PLAN.md M3)\n");
    } else {
      std::printf("\nstructural checks:\n");
      ok = run_checks(acc, oracle) && ok;
    }
  }
  if (use_argmax) {
    std::printf("\nend-to-end check:\n");
    ok = check_argmax(argv[3], logits) && ok;
  }
  std::printf("\n%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
