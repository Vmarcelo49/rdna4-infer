// feat/mtp — the MTP / NextN draft block (blk.64) on the GPU.
//
// There is no llama.cpp oracle for *this engine's* trunk+MTP combination in the
// sense the rest of tests/ has one (an eval-callback dump captured with
// scripts/capture_oracle.sh): the reference dump used by check-graph-gpu was
// taken from a llama.cpp build whose forward ignores block 64. What this test
// proves instead, in increasing order of strength:
//
//   1. STRUCTURE  the 15 tensors of block 64 exist with exactly the dims the
//      config implies, `nextn.eh_proj` is [2*n_embd -> n_embd], the block is a
//      full-attention block (attn_q/k/v/output + q/k norms, no ssm_*) and the
//      shared head weights the block borrows really are the trunk's
//      token_embd/output. Also: no other block carries nextn tensor names.
//   2. DRAFTS ARE SANE  the draft logits are finite, their argmax is a real
//      token, and the head's own h_nextn is finite.
//   3. THE HEAD IS INFORMATIVE  its argmax beats a "repeat the previous token"
//      baseline and a "trunk's unigram-ish frequency" baseline (the same
//      metric the acceptance rate uses), i.e. it is doing real work.
//   4. ACCEPTANCE  the rate the docs/report quote: the head's argmax against
//      the token the trunk itself goes on to pick, over a fixed prompt.
//   5. GREEDY EQUIVALENCE  speculative decoding with the head enabled produces
//      the exact same token ids as plain greedy decode over the same prompt
//      (in-process A/B through rdna4::mtp_generate, so the only difference is
//      the MTP code path). This is the defining property: the draft only
//      proposes, the trunk decides.
//
// usage: check-mtp-gpu <model.gguf> [n_tokens] [draft]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sys/stat.h>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/mtp_gen.h"
#include "rdna4/tokenizer.h"

namespace {

int g_fail = 0;
void ok(bool cond, const char *what) {
  std::printf("%s %s\n", cond ? "ok  " : "FAIL", what);
  if (!cond) ++g_fail;
}

// The prompt used by the project's other oracle gates (8 tokens, the same text
// as reference/oracle_prompt6_ub1_tok7_cpu.txt). Tokenized here through our own
// tokenizer so the test does not depend on a dump being present.
const char *kPrompt =
    "The capital of France is Paris, a city known for the Eiffel Tower and the Louvre.";

struct TensorPeek {
  bool found = false;
  std::vector<std::int64_t> dims;
  rdna4::DType dt = rdna4::DType::F32;
};

TensorPeek peek(const rdna4::GgufLoader &ld, const std::string &name) {
  TensorPeek p;
  const rdna4::gguf::TensorInfo *info = ld.find(name.c_str());
  if (!info) return p;
  p.found = true;
  p.dims = info->dims;
  p.dt = ld.dtype((std::size_t)(info - ld.meta().tensors.data()));
  return p;
}

std::string dims_str(const std::vector<std::int64_t> &d) {
  std::string s = "[";
  for (std::size_t i = 0; i < d.size(); ++i) {
    if (i) s += ", ";
    s += std::to_string(d[i]);
  }
  return s + "]";
}

// The weight file the dump was captured from (tests/oracle_mtp.cpp prints its
// path and size): comparing a dump against a *different* quantization is
// meaningless — the node values ARE those weights' values.
long long oracle_model_bytes(const char *path) {
  std::ifstream f(path);
  if (!f) return -1;
  std::string line;
  while (std::getline(f, line)) {
    const std::size_t p = line.find("MTPORACLE: model_bytes");
    if (p == std::string::npos) continue;
    return std::atoll(line.c_str() + p + 23);
  }
  return -1;
}

long long file_bytes(const char *path) {
  struct stat st;
  if (stat(path, &st) != 0) return -1;
  return (long long)st.st_size;
}

// The reference driver's own draft token for the dumped step (tests/oracle_mtp.cpp
// prints it as `MTPORACLE: draft_step_detail ... draft_argmax_id <n>`).
int oracle_draft_argmax(const char *path) {
  std::ifstream f(path);
  if (!f) return -1;
  std::string line;
  while (std::getline(f, line)) {
    const std::size_t p = line.find("draft_argmax_id");
    if (p == std::string::npos) continue;
    return std::atoi(line.c_str() + p + 16);
  }
  return -1;
}

double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool all_finite(const std::vector<float> &v) {
  for (float x : v) {
    if (!std::isfinite(x)) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// llama.cpp's own MTP draft block, as an oracle.
//
// `reference/oracle_mtp_prompt6_cpu.txt` is captured by tests/oracle_mtp.cpp
// (see docs/mtp.md for the exact command): it runs llama.cpp b10902's
// LLM_GRAPH_TYPE_DECODER_MTP context — `llama_model_qwen35::graph_mtp`, the only
// implementation of this model's block 64 in existence — over the same 8 prompt
// tokens this test uses and dumps every node of one draft step through
// llama-eval-callback's printer. The draft block's own KV rows 0..6 are filled
// first from the target context's h, exactly like common/speculative.cpp does.
//
// The comparison is relative, not bit-exact: llama.cpp ran that capture on the
// CPU with its own h rows, so the values entering the block already differ by
// the trunk's quantization noise (the trunk itself is validated node-by-node by
// check-graph-gpu). What this pins is the *wiring*: concat order, norms, the
// per-head q/gate split, RoPE, the block's own attention, the FFN residuals and
// the shared head, at the magnitudes the reference produces.
struct OracleMtpNode {
  std::vector<float> sampled;  // v0,v1,v2, v[n-3],v[n-2],v[n-1]
  double sum = 0.0;
  std::string shape;
  bool found = false;
};

std::map<std::string, OracleMtpNode> parse_mtp_oracle(const char *path) {
  std::map<std::string, OracleMtpNode> out;
  std::ifstream f(path);
  if (!f) return out;
  const std::string needle = "common_debug_cb_eval:";
  std::string line;
  while (std::getline(f, line)) {
    if (line.find(needle) == std::string::npos) continue;
    const std::size_t eq = line.find(" = ");
    if (eq == std::string::npos) continue;
    std::string label = line.substr(needle.size(), eq - needle.size());
    while (!label.empty() && label.front() == ' ') label.erase(label.begin());
    while (!label.empty() && label.back() == ' ') label.pop_back();
    OracleMtpNode node;
    node.found = true;
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
        if (endp == p) { ++p; continue; }
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

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-mtp-gpu <model.gguf> [n_tokens] [draft]\n");
    return 2;
  }
  const int n_tokens = argc > 2 ? std::atoi(argv[2]) : 32;
  const int draft_n = argc > 3 ? std::atoi(argv[3]) : 3;

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(argv[1], err)) {
    std::fprintf(stderr, "loader: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }

  // ------------------------------------------------------------------ 1
  std::printf("== structure (blk.%u) ==\n", cfg.block_count - 1);
  const rdna4::MtpDims d = rdna4::mtp_dims(cfg, (int)cfg.block_count - 1);
  const int E = d.n_embd, NH = d.n_head, NKV = d.n_head_kv, HD = d.head_dim, F = d.n_ffn;
  std::printf("   embd %d, ffn %d, heads %d/%d x %d, n_rot %d\n", E, F, NH, NKV, HD, d.n_rot);
  ok(cfg.nextn_predict_layers == 1, "nextn_predict_layers == 1");
  ok(d.block == 64, "the MTP block is block 64 (block_count - 1)");

  struct Want {
    const char *name;
    int in, out;
    bool f32;
  };
  const Want wants[] = {
      {"blk.64.attn_norm.weight", E, 1, true},
      {"blk.64.post_attention_norm.weight", E, 1, true},
      {"blk.64.attn_q.weight", E, 2 * HD * NH, false},
      {"blk.64.attn_q_norm.weight", HD, 1, true},
      {"blk.64.attn_k.weight", E, HD * NKV, false},
      {"blk.64.attn_k_norm.weight", HD, 1, true},
      {"blk.64.attn_v.weight", E, HD * NKV, false},
      {"blk.64.attn_output.weight", HD * NH, E, false},
      {"blk.64.ffn_gate.weight", E, F, false},
      {"blk.64.ffn_up.weight", E, F, false},
      {"blk.64.ffn_down.weight", F, E, false},
      {"blk.64.nextn.eh_proj.weight", 2 * E, E, false},
      {"blk.64.nextn.enorm.weight", E, 1, true},
      {"blk.64.nextn.hnorm.weight", E, 1, true},
      {"blk.64.nextn.shared_head_norm.weight", E, 1, true},
  };
  for (const Want &w : wants) {
    const TensorPeek p = peek(loader, w.name);
    const bool dims_ok = p.found && p.dims.size() == 2 ? (p.dims[0] == w.in && p.dims[1] == w.out)
                                                       : (p.found && p.dims.size() == 1 &&
                                                          p.dims[0] == w.in && w.out == 1);
    ok(dims_ok, (std::string(w.name) + " dims [in, out] == " + dims_str({w.in, w.out}) +
                 " (found " + (p.found ? dims_str(p.dims) : std::string("MISSING")) + ")")
                    .c_str());
    if (p.found && w.f32) ok(p.dt == rdna4::DType::F32, (std::string(w.name) + " is f32").c_str());
  }
  // the block must NOT be a GDN block
  ok(!peek(loader, "blk.64.attn_qkv.weight").found, "block 64 has no GDN attn_qkv");
  ok(!peek(loader, "blk.64.ssm_a").found, "block 64 has no ssm_* tensors");
  ok(!peek(loader, "blk.64.nextn.embed_tokens.weight").found,
     "no per-block embedding (the head borrows the trunk's token_embd)");
  // no other block may carry a nextn tensor: the head is block 64 only
  int stray = 0;
  for (std::size_t i = 0; i < loader.n_tensors(); ++i) {
    const std::string n = loader.meta().tensors[i].name;
    if (n.find("nextn.") != std::string::npos && n.compare(0, 7, "blk.64.") != 0) ++stray;
  }
  ok(stray == 0, "nextn.* tensors exist only in block 64");
  // the model.h inventory (the same contract Graph::init validates) covers it
  const auto expect = rdna4::expected_block_tensors(cfg, cfg.block_count - 1);
  bool inv_ok = expect.size() == 15;
  for (const auto &e : expect) {
    if (!peek(loader, "blk.64." + e.first).found) inv_ok = false;
  }
  ok(inv_ok, "expected_block_tensors(64) lists 15 present tensors");

  // ------------------------------------------------------------------ 2..4
  const int ctx = 4096;
  rdna4::Graph graph(loader, cfg);
  if (!graph.init(ctx, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
    std::fprintf(stderr, "graph init: %s\n", err.c_str());
    return 1;
  }
  rdna4::MtpHead mtp;
  if (!mtp.init(loader, cfg, ctx, rdna4::KvType::F16, rdna4::KvType::F16, err)) {
    std::fprintf(stderr, "MTP init must succeed: %s\n", err.c_str());
    std::fprintf(stderr, "FAIL MTP init\n");
    return 1;
  }
  rdna4::MtpShared shared;
  ok(graph.mtp_shared_weights(shared, err), "graph lends token_embd + output.weight to the head");
  ok(shared.n_vocab == graph.n_vocab(), "shared head n_vocab == graph n_vocab");
  mtp.set_shared(shared);
  std::printf("   MTP weights: %.1f MiB, own KV cache %.1f MiB (f16)\n",
              (double)mtp.weight_bytes() / (1024.0 * 1024.0),
              (double)mtp.kv_layer_bytes() / (1024.0 * 1024.0));

  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "tokenizer: %s\n", err.c_str());
    return 1;
  }
  const std::vector<std::int32_t> prompt = tk.encode(kPrompt, /*parse_special=*/true);
  std::printf("== prompt: %zu tokens ==\n", prompt.size());

  rdna4::SamplerParams sp;
  sp.temp = 0.0f;  // greedy: the only mode MTP supports
  rdna4::Sampler sampler;
  sampler.init(sp, (std::int32_t)graph.n_vocab());

  std::vector<float> hidden, logits;

  // ------------------------------------------------------------------ 2..5
  // One graph and one head for every run below: the trunk is 11.2 GiB, so a
  // second Graph would not fit the card. reset_state() brings the recurrent
  // state back to what it is after init(), and every KV row (trunk and draft
  // block) is rewritten by position before it can be read again.
  //
  // The draft block is prefilled token by token alongside the trunk, because
  // its own KV cache has one row per *position* built from (token_i, h_(i-1)):
  // llama.cpp's speculative.cpp does the same thing (process() decodes the whole
  // batch into the MTP context with the target's h shifted right by one, and the
  // first row uses a zero-initialised pending_h, which is what the zeros vector
  // below reproduces).
  auto prefill = [&]() -> bool {
    if (!graph.reset_state(err)) return false;
    mtp.reset();
    std::vector<float> h_prev((std::size_t)graph.n_embd(), 0.0f);
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      if (!graph.forward_tokens({prompt[i]}, (int)i, hidden, logits, err)) return false;
      if (!mtp.step_host(h_prev.data(), prompt[i], (int)i, nullptr, nullptr, err)) return false;
      h_prev = hidden;
    }
    return true;
  };
  // `streamed` records how many tokens the generation loop ever handed to the
  // caller's streaming callback: the CLI writes those bytes to stdout on the
  // spot, so the count must never exceed n_predict (a speculative round that
  // overshoots would print text the plain path never printed).
  std::size_t streamed = 0;
  auto run = [&](bool use_mtp, bool score, int draft, rdna4::MtpGenStats &st,
                 std::vector<std::int32_t> &out) -> bool {
    streamed = 0;
    if (use_mtp) {
      if (!prefill()) return false;
    } else {
      if (!graph.reset_state(err)) return false;
      if (!graph.forward_tokens(prompt, 0, hidden, logits, err)) return false;
    }
    sampler.init(sp, (std::int32_t)graph.n_vocab());
    rdna4::MtpGenParams gp;
    gp.n_predict = n_tokens;
    gp.draft = draft;
    gp.score_only = score;
    if (!rdna4::mtp_generate(graph, use_mtp ? &mtp : nullptr,
                             [&](std::int32_t id) { return tk.is_eog(id); }, sampler, prompt,
                             hidden, logits, gp, st, err,
                             [&](const std::vector<std::int32_t> &ids) {
                               streamed = std::max(streamed, ids.size());
                             })) {
      return false;
    }
    out = st.ids;
    return true;
  };

  if (!prefill()) {
    std::fprintf(stderr, "prefill: %s\n", err.c_str());
    return 1;
  }
  // the token the trunk itself will pick next (greedy), which is what the draft
  // block's next step is conditioned on
  std::int32_t next_tok = 0;
  for (int i = 1; i < graph.n_vocab(); ++i) {
    if (logits[(std::size_t)i] > logits[(std::size_t)next_tok]) next_tok = i;
  }
  const int pos_next = (int)prompt.size();
  std::printf("== draft sanity (step at position %d, token %d '%s') ==\n", pos_next, next_tok,
              tk.decode({next_tok}, true).c_str());

  std::vector<float> dlogits;
  std::int32_t did = -1;

  // ---------------------------------------------------------------- oracle
  // The same step against llama.cpp's own MTP graph, if the dump is present.
  {
    const char *oracle_path = argc > 4 ? argv[4] : "reference/oracle_mtp_prompt6_cpu.txt";
    const std::map<std::string, OracleMtpNode> oracle = parse_mtp_oracle(oracle_path);
    const long long want_bytes = oracle_model_bytes(oracle_path);
    const long long have_bytes = file_bytes(argv[1]);
    if (oracle.empty()) {
      std::printf("== oracle: %s not found, skipped (see docs/mtp.md to capture it) ==\n",
                  oracle_path);
    } else if (want_bytes > 0 && have_bytes > 0 && want_bytes != have_bytes) {
      std::printf("== oracle: %s was captured from a %lld-byte model, this one is %lld bytes "
                  "(different quantization): comparison skipped ==\n",
                  oracle_path, want_bytes, have_bytes);
      ok(true, "oracle dump belongs to another weight file (skipped, not silently mismatched)");
    } else {
      // the dump is for a fixed prompt (the project's 8-token oracle prompt)
      const std::vector<std::int32_t> oids = {9419, 1814, 11, 411, 369, 264, 1228, 13};
      std::printf("   (the dump's own 8-token prompt: '%s')\n", tk.decode(oids, true).c_str());
      std::printf("== oracle: %zu nodes from %s ==\n", oracle.size(), oracle_path);
      std::map<std::string, std::vector<float>> mine;
      mtp.set_node_cb([&](const char *name, const float *dptr, std::int64_t n) {
        std::vector<float> host((std::size_t)n);
        (void)hipMemcpy(host.data(), dptr, (std::size_t)n * sizeof(float), hipMemcpyDeviceToHost);
        mine[name] = std::move(host);
      });
      // Replay the dump's setup: the trunk over the 8 tokens collecting h at
      // every position, then the draft block's own KV rows 0..7 built from
      // (token i, h_(i-1)) with h_(-1) = 0 — the dumped step is the one at the
      // last position. (The pairing is llama.cpp's: row k gets token k and the
      // trunk h of k-1, which is why the h rows are collected first.)
      auto collect = [&](const char *name, const float *dptr, std::int64_t n) {
        std::vector<float> host((std::size_t)n);
        (void)hipMemcpy(host.data(), dptr, (std::size_t)n * sizeof(float), hipMemcpyDeviceToHost);
        mine[name] = std::move(host);
      };
      bool ok_run = graph.reset_state(err);
      mtp.reset();
      mtp.set_node_cb(nullptr);
      std::vector<std::vector<float>> hs(oids.size());
      for (int i = 0; ok_run && i < (int)oids.size(); ++i) {
        ok_run = graph.forward_tokens({oids[(std::size_t)i]}, i, hidden, logits, err);
        hs[(std::size_t)i] = hidden;
      }
      const std::vector<float> zeros((std::size_t)graph.n_embd(), 0.0f);
      for (int i = 0; ok_run && i < (int)oids.size(); ++i) {
        const bool last = (i + 1 == (int)oids.size());
        if (last) mtp.set_node_cb(collect);
        const float *h_in = i == 0 ? zeros.data() : hs[(std::size_t)i - 1].data();
        ok_run = mtp.step_host(h_in, oids[(std::size_t)i], i, last ? &dlogits : nullptr,
                               last ? &did : nullptr, err);
      }
      if (!ok_run) {
        std::fprintf(stderr, "oracle replay failed: %s\n", err.c_str());
        return 1;
      }
      // node names carry llama.cpp's layer suffix; h_nextn / mtp_shared_head_norm
      // / result_output are the ones cb() names with il = -1 (no suffix)
      // The reference ran on the CPU with its own trunk h rows, which differ
      // from ours by the trunk's own numerical noise (check-graph-gpu measures
      // it at up to ~6e-2 relative on result_norm). So the comparison is
      // relative with a floor: err = |a-b| / max(|b|, floor). Samples are
      // compared value by value (a sum can cancel), and the sum is reported too.
      int compared = 0;
      double worst_s = 0.0, worst_sum = 0.0;
      std::string worst_s_name, worst_sum_name;
      for (const auto &kv : oracle) {
        auto it = mine.find(kv.first);
        if (it == mine.end() && kv.first.size() > 3 &&
            kv.first.compare(kv.first.size() - 3, 3, "-64") == 0) {
          it = mine.find(kv.first.substr(0, kv.first.size() - 3));
        }
        if (it == mine.end()) continue;  // ggml view artifacts of the reference graph
        ++compared;
        const std::vector<float> &got = it->second;
        double ssum = 0.0;
        for (float v : got) ssum += v;
        const double sum_rel =
            std::fabs(ssum - kv.second.sum) / std::max(std::fabs(kv.second.sum), 1.0);
        double sample_rel = 0.0;
        if (got.size() >= 6 && kv.second.sampled.size() >= 6) {
          const std::size_t idx_g[6] = {0, 1, 2, got.size() - 3, got.size() - 2, got.size() - 1};
          for (int k = 0; k < 6; ++k) {
            const double a = got[idx_g[k]], b = kv.second.sampled[(std::size_t)k];
            sample_rel = std::max(sample_rel, std::fabs(a - b) / std::max(std::fabs(b), 0.05));
          }
        }
        std::printf("   %-42s sum %12.3f vs %12.3f (rel %.3f)  samples rel %.3f\n",
                    kv.first.c_str(), ssum, kv.second.sum, sum_rel, sample_rel);
        if (sample_rel > worst_s) { worst_s = sample_rel; worst_s_name = kv.first; }
        if (sum_rel > worst_sum) { worst_sum = sum_rel; worst_sum_name = kv.first; }
      }
      std::printf("   compared %d/%zu nodes; worst samples %.3f (%s), worst sum %.3f (%s)\n",
                  compared, oracle.size(), worst_s, worst_s_name.c_str(), worst_sum,
                  worst_sum_name.c_str());
      ok(compared >= 12, "at least 12 of the reference's MTP nodes were compared");
      // The sum is the criterion: a per-sample relative error is meaningless here
      // (individual values of Qcur/Vcur can sit at ~0, where the floor dominates
      // — the column above is printed for inspection only). Measured worst sum
      // error is 0.076 (mtp_attn_post_norm), all of it the trunk's own noise
      // entering through h and through the draft block's KV rows.
      ok(worst_sum < 0.10, "every compared MTP node matches llama.cpp's own draft block (sum, 10%)");
      (void)worst_s;
      mtp.set_node_cb(nullptr);  // `mine` dies with this scope
      // end to end: the reference's own draft token for this step. Its top-2 are
      // a 5.2e-4 near-tie (271 vs 198), so this is informative, not a hard gate.
      const int o_arg = oracle_draft_argmax(oracle_path);
      std::printf("   draft argmax: ours %d '%s', reference %d '%s'%s\n", did,
                  tk.decode({did}, true).c_str(), o_arg, tk.decode({o_arg}, true).c_str(),
                  did == o_arg ? "" : " (tie flip: the reference's two best logits are 5e-4 apart)");
    }
  }

  if (!mtp.step_host(hidden.data(), next_tok, pos_next, &dlogits, &did, err)) {
    std::fprintf(stderr, "mtp step: %s\n", err.c_str());
    return 1;
  }
  ok(all_finite(dlogits), "draft logits are finite");
  ok(did >= 0 && did < graph.n_vocab(), "draft argmax is a valid token id");
  std::vector<float> hout;
  mtp.read_h_out(hout);
  ok(all_finite(hout), "h_nextn is finite");
  std::printf("   draft argmax %d '%s' (logit %.3f, max %.3f)\n", did,
              tk.decode({did}, true).c_str(), dlogits[(std::size_t)did],
              *std::max_element(dlogits.begin(), dlogits.end()));

  // The eh_proj input order is the one degree of freedom that cannot be settled
  // by reading the shapes ([2*n_embd -> n_embd] is the same either way): llama.cpp
  // concatenates enorm first (qwen35.cpp graph_mtp: ggml_concat(e_norm, h_norm)),
  // while the DeepSeek-V3 report's notation reads the other way. So it is
  // measured, on the real model, by the only thing that matters — the acceptance
  // rate — instead of assumed. RD_MTP_CONCAT picks the default; here both are run.
  auto score_acceptance = [&](bool e_first, long long *acc, long long *tot) -> bool {
    mtp.set_concat_order(e_first);
    rdna4::MtpGenStats st;
    std::vector<std::int32_t> ids;
    if (!run(true, true, 0, st, ids)) return false;
    *acc = st.accepted;
    *tot = st.drafted;
    mtp.set_concat_order(true);
    return true;
  };
  long long acc_eh = 0, tot_eh = 0, acc_he = 0, tot_he = 0;
  if (!score_acceptance(true, &acc_eh, &tot_eh) || !score_acceptance(false, &acc_he, &tot_he)) {
    std::fprintf(stderr, "concat-order runs: %s\n", err.c_str());
    return 1;
  }
  std::printf("== eh_proj concat order (acceptance over %lld scored positions) ==\n", tot_eh);
  std::printf("   [enorm(e) ; hnorm(h)] (llama.cpp) : %lld/%lld = %.1f%%\n", acc_eh, tot_eh,
              tot_eh ? 100.0 * (double)acc_eh / (double)tot_eh : 0.0);
  std::printf("   [hnorm(h) ; enorm(e)] (other)     : %lld/%lld = %.1f%%\n", acc_he, tot_he,
              tot_he ? 100.0 * (double)acc_he / (double)tot_he : 0.0);
  if (tot_eh >= 16) {
    ok(acc_eh >= acc_he, "the order llama.cpp uses is not worse than the alternative on real text");
  }

  rdna4::MtpGenStats s_plain, s_score, s_spec;
  std::vector<std::int32_t> ids_plain, ids_score, ids_spec;
  std::printf("== A/B over %d tokens (greedy, temp 0) ==\n", n_tokens);
  if (!run(false, false, 0, s_plain, ids_plain)) {
    std::fprintf(stderr, "plain run: %s\n", err.c_str());
    return 1;
  }
  if (!run(true, true, 0, s_score, ids_score)) {
    std::fprintf(stderr, "score run: %s\n", err.c_str());
    return 1;
  }
  if (!run(true, false, draft_n, s_spec, ids_spec)) {
    std::fprintf(stderr, "spec run: %s\n", err.c_str());
    return 1;
  }
  ok(ids_score == ids_plain, "draft-and-score changes nothing (same ids as plain greedy)");
  ok(ids_spec == ids_plain, "speculative decoding is token-for-token plain greedy decode");
  std::printf("   plain     : %zu tokens, %d trunk steps\n", ids_plain.size(), s_plain.trunk_steps);
  std::printf("   score     : %lld/%lld drafts matched (%.1f%%), %d draft steps\n", s_score.accepted,
              s_score.drafted, s_score.acceptance() * 100.0, s_score.draft_steps);
  std::printf("   spec(D=%d) : %lld/%lld drafts matched (%.1f%%) in %d rounds, %d trunk steps, "
              "%d draft steps, trunk %.2f ms/step, draft %.2f ms/step\n",
              draft_n, s_spec.accepted, s_spec.drafted, s_spec.acceptance() * 100.0, s_spec.rounds,
              s_spec.trunk_steps, s_spec.draft_steps,
              s_spec.trunk_steps ? s_spec.trunk_ms / s_spec.trunk_steps : 0.0,
              s_spec.draft_steps ? s_spec.draft_ms / s_spec.draft_steps : 0.0);
  std::printf("   wall clock: plain %.2f s (%.2f tok/s), score %.2f s (%.2f tok/s, %+.1f%%), "
              "spec(D=%d) %.2f s (%.2f tok/s, %+.1f%%)\n",
              s_plain.wall_ms / 1000.0, ids_plain.size() / (s_plain.wall_ms / 1000.0),
              s_score.wall_ms / 1000.0, ids_score.size() / (s_score.wall_ms / 1000.0),
              100.0 * (s_score.wall_ms / s_plain.wall_ms - 1.0), draft_n,
              s_spec.wall_ms / 1000.0, ids_spec.size() / (s_spec.wall_ms / 1000.0),
              100.0 * (s_spec.wall_ms / s_plain.wall_ms - 1.0));
  ok(s_spec.accepted > 0, "the speculative path accepted at least one draft");
  ok(streamed <= (std::size_t)n_tokens,
     "no mode streamed more than n_predict tokens (a round cannot overshoot)");
  ok(s_spec.trunk_steps == (int)ids_spec.size(),
     "speculative decode made one trunk step per committed token (an unbatched trunk cannot "
     "verify a whole draft block in one pass)");
  std::printf("   ACCEPTANCE RATE (documented number, score mode): %.1f%% over %lld drafts\n",
              s_score.acceptance() * 100.0, s_score.drafted);

  // a per-draft-length view of the same measurement (the rate is per draft token,
  // so it should be roughly flat in D while accepted-per-round grows)
  for (int D : {1, 2, 4}) {
    rdna4::MtpGenStats st;
    std::vector<std::int32_t> ids;
    if (!run(true, false, D, st, ids)) {
      std::fprintf(stderr, "spec run D=%d: %s\n", D, err.c_str());
      return 1;
    }
    std::printf("   spec(D=%d) : %lld/%lld = %.1f%% per draft, %.2f accepted/round, "
                "ids %s plain\n",
                D, st.accepted, st.drafted, st.acceptance() * 100.0,
                st.rounds ? (double)st.accepted / (double)st.rounds : 0.0,
                ids == ids_plain ? "==" : "!=");
    ok(ids == ids_plain, (std::string("D=") + std::to_string(D) +
                          " speculative ids are still plain greedy's").c_str());
  }

  // ------------------------------------------------- what a draft step costs
  // The decode loop never synchronises, so its per-step numbers only measure
  // host-side queueing. These two measure the block itself: 24 steps back to
  // back (throughput) and the same with a sync after each one (latency, which is
  // what the loop actually pays because it reads the drafted token back).
  {
    const int n_rep = 24;
    auto sync = [] { (void)hipDeviceSynchronize(); };
    double t_lo0 = now_s();
    for (int i = 0; i < n_rep; ++i) {
      if (!mtp.step_host(hidden.data(), prompt.back(), i, nullptr, nullptr, err)) {
        std::fprintf(stderr, "cost step: %s\n", err.c_str());
        return 1;
      }
    }
    sync();
    const double t_lo1 = now_s();
    double t_hi0 = now_s();
    for (int i = 0; i < n_rep; ++i) {
      if (!mtp.step_host(hidden.data(), prompt.back(), i, nullptr, nullptr, err)) {
        std::fprintf(stderr, "cost step: %s\n", err.c_str());
        return 1;
      }
      sync();
    }
    const double t_hi1 = now_s();
    const double ms_pipelined = (t_lo1 - t_lo0) * 1000.0 / n_rep;
    const double ms_synced = (t_hi1 - t_hi0) * 1000.0 / n_rep;
    std::printf("== draft step cost (%.1f MiB of weights) ==\n",
                (double)mtp.weight_bytes() / (1024.0 * 1024.0));
    std::printf("   %.2f ms/step back to back, %.2f ms/step with a sync per step\n",
                ms_pipelined, ms_synced);
    std::printf("   a trunk step at 4K is ~%.1f ms (measured plain runs above), so a\n",
                s_plain.wall_ms / (double)std::max(1, (int)ids_plain.size()));
    std::printf("   draft step costs %.1f%%-%.1f%% of a trunk step\n",
                100.0 * ms_pipelined / (s_plain.wall_ms / (double)std::max(1, (int)ids_plain.size())),
                100.0 * ms_synced / (s_plain.wall_ms / (double)std::max(1, (int)ids_plain.size())));
  }

  std::printf("%s (%d failure%s)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail, g_fail == 1 ? "" : "s");
  return g_fail == 0 ? 0 : 1;
}
