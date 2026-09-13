// M3 step 4 — KV cache behaviour at long context.
//
// Two things are checked here that the per-node oracle comparison cannot:
//
//  1. Continuation: a prompt split across several calls (prefill in chunks, as a
//     CLI does) must produce *bit-identical* logits to a single call. This is
//     what pins the position handling and the KV/recurrent state handoff
//     (Graph::forward(..., start_pos, ...)).
//
//  2. Long context within the 16 GB budget: the cache is allocated for
//     GRAPH_CTX tokens (default 65536) with each KV storage type, filled with a
//     deterministic pseudo-random pattern, and one decode step is run at the end
//     of the context. That exercises the full-length attention path (every
//     layer attends to every cached token) and reports the VRAM/time cost, which
//     is the M3 acceptance item "run de ctx longo (>= 64K, KV q4_0) dentro dos
//     16 GB".
//
// usage: check-kvctx-gpu <file.gguf> [ctx] [type]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);

namespace {

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

double now_ms() {
  return (double)std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
             .count() /
         1000.0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-kvctx-gpu <file.gguf> [ctx] [kv type]\n");
    return 2;
  }
  const int ctx = argc >= 3 ? std::atoi(argv[2]) : 65536;
  rdna4::KvType kv = rdna4::KvType::Q4_0;
  if (argc >= 4) {
    const char *e = rdna4::kv_type_parse(argv[3], &kv);
    if (e) {
      std::fprintf(stderr, "%s\n", e);
      return 2;
    }
  }

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(ld.meta(), cfg, err)) {
    std::fprintf(stderr, "config: %s\n", err.c_str());
    return 1;
  }

  int failures = 0;
  const std::vector<int> prompt = {9419, 1814, 11, 411, 369, 264, 1228, 13};
  std::vector<std::vector<float>> emb(prompt.size());
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    if (!load_embedding(ld, cfg, prompt[i], emb[i], err)) {
      std::fprintf(stderr, "embedding: %s\n", err.c_str());
      return 1;
    }
  }

  // ---------------- 1. continuation must be bit-identical ----------------
  {
    std::vector<float> all_emb, h_whole, l_whole, h_split, l_split;
    for (const auto &e : emb) all_emb.insert(all_emb.end(), e.begin(), e.end());

    {
      rdna4::Graph g(ld, cfg);
      const double t0 = now_ms();
      if (!g.init((int)prompt.size(), rdna4::KvType::F32, rdna4::KvType::F32, err)) {
        std::fprintf(stderr, "graph init: %s\n", err.c_str());
        return 1;
      }
      std::printf("graph init (weight upload + caches): %.1f ms\n", now_ms() - t0);
      if (!g.forward(all_emb, h_whole, l_whole, err)) {
        std::fprintf(stderr, "forward(whole): %s\n", err.c_str());
        return 1;
      }
    }  // the weights are ~12 GiB: only one graph may exist at a time
    {
      rdna4::Graph g2(ld, cfg);
      if (!g2.init((int)prompt.size(), rdna4::KvType::F32, rdna4::KvType::F32, err)) {
        std::fprintf(stderr, "graph init 2: %s\n", err.c_str());
        return 1;
      }
      // 1 + 3 + 4 tokens, feeding each chunk at its own start position
      const int chunks[] = {1, 3, 4};
      int pos = 0;
      for (int n : chunks) {
        std::vector<float> chunk;
        for (int i = 0; i < n; ++i) {
          chunk.insert(chunk.end(), emb[pos + i].begin(), emb[pos + i].end());
        }
        if (!g2.forward(chunk, pos, h_split, l_split, err)) {
          std::fprintf(stderr, "forward(chunk at %d): %s\n", pos, err.c_str());
          return 1;
        }
        pos += n;
      }
    }
    const bool same_h = h_split == h_whole;
    const bool same_l = l_split == l_whole;
    std::printf("continuation (1+3+4 vs 8 tokens): hidden %s, logits %s  %s\n",
                same_h ? "identical" : "DIFFER", same_l ? "identical" : "DIFFER",
                (same_h && same_l) ? "OK" : "MISMATCH");
    if (!same_h || !same_l) {
      ++failures;
      double worst = 0.0;
      for (std::size_t i = 0; i < l_split.size() && i < l_whole.size(); ++i) {
        worst = std::max(worst, (double)std::fabs(l_split[i] - l_whole[i]));
      }
      std::printf("  max logit difference: %.6e\n", worst);
    }
  }

  // ---------------- 2. long context ----------------
  {
    rdna4::Graph g(ld, cfg);
    const double t0 = now_ms();
    if (!g.init(ctx, kv, kv, err)) {
      std::fprintf(stderr, "graph init at ctx=%d: %s\n", ctx, err.c_str());
      return 1;
    }
    const double t_init = now_ms() - t0;
    const double layer_mib = (double)g.kv_layer_bytes() / (1024.0 * 1024.0);
    std::printf("ctx=%d kv=%s (%.2f B/elem): init %.1f ms, KV caches %.1f MiB "
                "(%d layers x K+V %.1f MiB)\n",
                ctx, rdna4::kv_type_name(kv), rdna4::kv_bytes_per_elem(kv), t_init,
                2.0 * layer_mib * g.n_full_attn(), g.n_full_attn(), 2.0 * layer_mib);

    // The embeddings are the same 8 rows; the cache/state are filled with a
    // deterministic pattern so the attention really walks the whole context.
    std::vector<float> all_emb;
    for (const auto &e : emb) all_emb.insert(all_emb.end(), e.begin(), e.end());
    if (!g.debug_fill_caches(1234u, err)) {
      std::fprintf(stderr, "fill caches: %s\n", err.c_str());
      return 1;
    }

    std::size_t free_before = 0, total = 0;
    hipMemGetInfo(&free_before, &total);
    std::vector<float> h, logits;
    const double t1 = now_ms();
    const int start = ctx - (int)prompt.size();
    if (!g.forward(all_emb, start, h, logits, err)) {
      std::fprintf(stderr, "forward at %d: %s\n", start, err.c_str());
      return 1;
    }
    const double t_fwd = now_ms() - t1;
    std::size_t free_after = 0;
    hipMemGetInfo(&free_after, &total);

    double lmin = logits[0], lmax = logits[0], lsum = 0.0;
    bool finite = true;
    for (float v : logits) {
      if (!std::isfinite(v)) finite = false;
      lmin = std::min(lmin, (double)v);
      lmax = std::max(lmax, (double)v);
      lsum += (double)v;
    }
    std::size_t best = 0;
    for (std::size_t i = 1; i < logits.size(); ++i) {
      if (logits[i] > logits[best]) best = i;
    }
    std::printf("decode %zu tokens at position %d..%d: %.1f ms (%.1f ms/token)\n", prompt.size(),
                start, ctx - 1, t_fwd, t_fwd / (double)prompt.size());
    std::printf("logits finite=%s min=%.3f max=%.3f mean=%.4f argmax=%zu\n", finite ? "yes" : "NO",
                lmin, lmax, lsum / (double)logits.size(), best);
    std::printf("VRAM in use after forward: %.2f GiB (free %.2f of %.2f GiB)\n",
                (double)(total - free_after) / (1024.0 * 1024.0 * 1024.0),
                (double)free_after / (1024.0 * 1024.0 * 1024.0),
                (double)total / (1024.0 * 1024.0 * 1024.0));
    if (!finite) {
      std::printf("long-context decode: FAILED (non-finite logits)\n");
      ++failures;
    } else {
      std::printf("long-context decode: OK\n");

    // M7: the split-KV attention sums the keys in a different order, so it is NOT
    // bit-identical -- but at the *same* context the logits must agree to fp32
    // rounding (measured ~1e-6 relative at the kernel level). Run the same step
    // twice, unsplit and split, and compare.
    {
      const int forced = (int)(ctx / 2048) < 1 ? 1 : ((int)(ctx / 2048) > 16 ? 16 : (int)(ctx / 2048));
      std::vector<float> l_ref, l_split, h_tmp;
      // the recurrent (GDN) state advances with every forward call, so both runs
      // must start from the same state; the KV rows are rewritten identically
      // localise the divergence: capture a few nodes from both runs and compare
      struct Cap {
        std::map<std::string, std::vector<float>> v;
      } cap1, cap2;
      auto make_cb = [](Cap *c) {
        return [c](const char *name, int il, int, const float *p, std::int64_t n) {
          std::string key = std::string(name) + "#" + std::to_string(il);
          if (key != "attn_pregate#3" && key != "attn_gated#3" && key != "l_out#3" &&
              key != "l_out#7" && key != "l_out#63" && key != "result_norm#-1") {
            return;
          }
          c->v[key].assign(p, p + n);
        };
      };
      g.set_attn_splits(1);
      g.set_node_cb(make_cb(&cap1));
      if (!g.reset_state(err) || !g.forward(all_emb, start, h_tmp, l_ref, err)) {
        std::printf("splits: reference forward failed: %s\n", err.c_str());
        return 1;
      }
      g.set_attn_splits(forced);
      g.set_node_cb(make_cb(&cap2));
      if (!g.reset_state(err) || !g.forward(all_emb, start, h_tmp, l_split, err)) {
        std::printf("splits: split forward failed: %s\n", err.c_str());
        return 1;
      }
      g.set_node_cb(nullptr);
      for (const auto &kv : cap1.v) {
        auto it = cap2.v.find(kv.first);
        if (it == cap2.v.end() || it->second.size() != kv.second.size()) continue;
        double n2 = 0, d2 = 0, mx = 0;
        for (std::size_t i = 0; i < kv.second.size(); ++i) {
          const double d = (double)kv.second[i] - (double)it->second[i];
          n2 += d * d;
          d2 += (double)kv.second[i] * kv.second[i] + (double)it->second[i] * it->second[i];
          mx = std::max(mx, std::fabs(d));
        }
        std::printf("   node %-18s rel-L2 %.3e max|d| %.3e\n", kv.first.c_str(),
                    std::sqrt(n2 / d2), mx);
      }
      double num = 0, den = 0, maxd = 0;
      for (std::size_t i = 0; i < l_ref.size(); ++i) {
        const double d = (double)l_ref[i] - (double)l_split[i];
        num += d * d;
        den += (double)l_ref[i] * l_ref[i] + (double)l_split[i] * l_split[i];
        maxd = std::max(maxd, std::fabs(d));
      }
      const double rel = std::sqrt(num / den);
      std::size_t am_ref = 0, am_split = 0;
      for (std::size_t i = 1; i < l_ref.size(); ++i) {
        if (l_ref[i] > l_ref[am_ref]) am_ref = i;
        if (l_split[i] > l_split[am_split]) am_split = i;
      }
      // The tolerance here is deliberately loose: this test runs against a
      // SYNTHETIC cache, so the attention output is a near-cancelling average of
      // 65K random V rows. In that regime a 1e-7 reordering difference in the
      // key sum is amplified by the cancellation (measured rel-L2 ~1.5e-2 on the
      // logits, with l_out#7 -- a GDN layer on the same residual -- bit-identical
      // and the argmax preserved), which is why the *gating* comparison uses real
      // text: scripts/check_attn_split.sh.
      const bool ok = rel < 5e-2 && am_ref == am_split;
      std::printf("attention splits (%d) vs unsplit at ctx %zu: logits rel-L2 %.3e max|d| %.3e, "
                  "argmax %zu vs %zu  %s (synthetic cache: informative, see the comment)\n",
                  forced, ctx, rel, maxd, am_ref, am_split, ok ? "OK" : "FAIL");
      if (!ok) {
        std::printf("attention split check: FAILED\n");
        return 1;
      }
      std::printf("attention split check: OK\n");
    }
    }
  }

  std::printf("check-kvctx-gpu: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
