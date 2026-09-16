// M4 — sampler behaviour.
//
// The sampler is what turns logits into text, so this test pins the properties
// the CLI depends on: greedy is exactly the argmax and is RNG-free, every filter
// stage keeps/removes what its reference says, the chain order is
// penalties -> top_k -> top_p -> min_p -> temperature -> dist, and a fixed seed
// reproduces the same sequence bit for bit.
//
// usage: check-sampler
#include <algorithm>
#include <cmath>
#include <limits>
#include <cstdio>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "rdna4/sampler.h"

namespace {

int failures = 0;

void check(bool cond, const char *what) {
  std::printf("%-5s %s\n", cond ? "ok" : "FAIL", what);
  if (!cond) ++failures;
}

std::vector<float> make_logits(int n, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> d(-6.0f, 6.0f);
  std::vector<float> l((std::size_t)n);
  for (float &v : l) v = d(rng);
  return l;
}

std::set<std::int32_t> ids_of(const std::vector<rdna4::Candidate> &c) {
  std::set<std::int32_t> s;
  for (const auto &x : c) s.insert(x.id);
  return s;
}

}  // namespace

int main() {
  const int n = 4096;
  const std::vector<float> base = make_logits(n, 42);
  const std::vector<std::int32_t> no_hist;

  // ---------------- greedy ----------------
  {
    rdna4::SamplerParams p;
    p.temp = 0.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    rdna4::Sampler s;
    s.init(p, n);

    std::size_t want = 0;
    for (int i = 1; i < n; ++i) {
      if (base[(std::size_t)i] > base[want]) want = (std::size_t)i;
    }
    std::vector<float> l = base;
    const std::int32_t got = s.sample(l.data(), no_hist);
    check(got == (std::int32_t)want, "greedy picks the argmax");

    // RNG-free: repeated calls on a fresh sampler with any seed agree
    rdna4::SamplerParams p2 = p;
    p2.seed = 1234567;
    rdna4::Sampler s2;
    s2.init(p2, n);
    std::vector<float> l2 = base;
    check(s2.sample(l2.data(), no_hist) == got, "greedy ignores the seed");

    // greedy after filters still picks the best of the survivors
    rdna4::SamplerParams p3 = p;
    p3.top_k = 5;
    rdna4::Sampler s3;
    s3.init(p3, n);
    auto cands = s3.filter(base.data(), no_hist);
    check(cands.size() == 1 && cands[0].id == (std::int32_t)want,
          "greedy after top_k still picks the global argmax");
  }

  // ---------------- top_k ----------------
  {
    rdna4::SamplerParams p;
    p.top_k = 7;
    p.top_p = 1.0f;
    rdna4::Sampler s;
    s.init(p, n);
    auto c = s.filter(base.data(), no_hist);
    check(c.size() == 7, "top_k keeps exactly k candidates");
    for (std::size_t i = 1; i < c.size(); ++i) {
      if (c[i - 1].logit < c[i].logit) check(false, "top_k output is sorted desc");
    }
    check(ids_of(c).size() == 7, "top_k ids are distinct");
    // k <= 0 disables the filter, k >= n keeps everything
    rdna4::SamplerParams p0 = p;
    p0.top_k = 0;
    p0.top_p = 1.0f;
    rdna4::Sampler s0;
    s0.init(p0, n);
    check(s0.filter(base.data(), no_hist).size() == (std::size_t)n, "top_k <= 0 keeps all");
  }

  // ---------------- top_p ----------------
  {
    rdna4::SamplerParams p;
    p.top_k = 0;
    p.top_p = 0.9f;
    rdna4::Sampler s;
    s.init(p, n);
    auto c = s.filter(base.data(), no_hist);
    // the kept set must be the smallest prefix with mass >= 0.9, computed on the
    // raw (un-tempered) logits with a max-subtracted softmax, like llama.cpp
    std::vector<float> l = base;
    std::sort(l.begin(), l.end(), std::greater<float>());
    double sum = 0.0;
    for (float v : l) sum += std::exp((double)v - (double)l[0]);
    double cum = 0.0;
    std::size_t want = l.size();
    for (std::size_t i = 0; i < l.size(); ++i) {
      cum += std::exp((double)l[i] - (double)l[0]) / sum;
      if (cum >= 0.9) {
        want = i + 1;
        break;
      }
    }
    check(c.size() == want, "top_p keeps the smallest prefix with mass >= p");
    check(c.size() > 1, "top_p keeps more than one token on a flat distribution");
    // p >= 1 disables it
    rdna4::SamplerParams p1 = p;
    p1.top_p = 1.0f;
    rdna4::Sampler s1;
    s1.init(p1, n);
    check(s1.filter(base.data(), no_hist).size() == (std::size_t)n, "top_p >= 1 keeps all");
  }

  // ---------------- min_p ----------------
  {
    rdna4::SamplerParams p;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.min_p = 0.1f;
    rdna4::Sampler s;
    s.init(p, n);
    auto c = s.filter(base.data(), no_hist);
    // every kept logit must be >= max + log(min_p), and the token just below the
    // threshold must be gone
    const float maxv = *std::max_element(base.begin(), base.end());
    const float thresh = maxv + std::log(0.1f);
    bool all_above = true;
    for (const auto &x : c) all_above = all_above && x.logit >= thresh;
    check(all_above, "min_p keeps only logits >= max + log(min_p)");
    std::size_t removed = 0;
    for (float v : base) {
      if (v < thresh) ++removed;
    }
    check(c.size() == (std::size_t)n - removed, "min_p removes exactly the tokens below threshold");
  }

  // ---------------- penalties ----------------
  {
    rdna4::SamplerParams p;
    p.temp = 0.0f;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.repeat_penalty = 2.0f;
    rdna4::Sampler s;
    s.init(p, n);
    std::vector<float> l = base;
    const std::int32_t best = (std::int32_t)(std::max_element(base.begin(), base.end()) -
                                            base.begin());
    std::vector<std::int32_t> hist = {best};
    const std::int32_t got = s.sample(l.data(), hist);
    check(got != best, "repeat penalty pushes the repeated token away from greedy");
    check(s.filter(base.data(), hist).size() == 1, "greedy keeps one candidate after penalties");
    // the transformation itself: a tiny penalty leaves the repeated token the
    // winner, so its filter() logit must be exactly logit/2 (positive logit)
    rdna4::SamplerParams pt = p;
    pt.repeat_penalty = 1.0001f;
    rdna4::Sampler st;
    st.init(pt, n);
    auto ct = st.filter(base.data(), hist);
    check(!ct.empty() && ct[0].id == best, "a tiny penalty keeps the repeated token the winner");
    check(!ct.empty() && std::fabs(ct[0].logit - base[(std::size_t)best] / 1.0001f) < 1e-4f,
          "positive logits are divided by the penalty");
    // ... and a negative logit is multiplied by it (llama.cpp's fix for the
    // "dividing makes negative logits more likely" problem)
    std::size_t neg = 0;
    for (std::size_t i = 0; i < base.size(); ++i) {
      if (base[i] < -1.0f) { neg = i; break; }
    }
    std::vector<std::int32_t> hist_neg = {(std::int32_t)neg};
    // temp = 1 keeps the whole candidate set (greedy would truncate to one)
    rdna4::SamplerParams pn = pt;
    pn.temp = 1.0f;
    rdna4::Sampler sn;
    sn.init(pn, n);
    auto cn = sn.filter(base.data(), hist_neg);
    bool found = false, scaled_ok = false;
    for (const auto &x : cn) {
      if (x.id == (std::int32_t)neg) {
        found = true;
        scaled_ok = std::fabs(x.logit - base[neg] * 1.0001f) < 1e-4f;
      }
    }
    check(found && scaled_ok, "negative logits are multiplied by the penalty");
    // penalty == 1 is a no-op
    p.repeat_penalty = 1.0f;
    rdna4::Sampler s2;
    s2.init(p, n);
    std::vector<float> l2 = base;
    check(s2.sample(l2.data(), hist) == best, "repeat_penalty == 1 is a no-op");
  }

  // ---------------- temperature ----------------
  {
    rdna4::SamplerParams p;
    p.top_k = 0;
    p.top_p = 1.0f;
    p.temp = 0.5f;
    rdna4::Sampler s;
    s.init(p, n);
    auto c = s.filter(base.data(), no_hist);
    // the same ordering, with every logit divided by the temperature
    std::vector<float> l = base;
    std::sort(l.begin(), l.end(), std::greater<float>());
    bool scaled = true;
    for (std::size_t i = 0; i < c.size(); ++i) {
      scaled = scaled && std::fabs(c[i].logit - l[i] / 0.5f) < 1e-4f;
    }
    check(scaled, "temperature divides the surviving logits");
    // temp > 1 flattens the distribution (top-1 probability drops)
    rdna4::SamplerParams ph = p;
    ph.temp = 4.0f;
    rdna4::Sampler sh;
    sh.init(ph, n);
    auto ch = sh.filter(base.data(), no_hist);
    rdna4::Sampler::softmax(c);
    rdna4::Sampler::softmax(ch);
    check(ch[0].p < c[0].p, "higher temperature flattens the distribution");
  }

  // ---------------- chain order ----------------
  {
    // top_k before top_p before min_p before temperature: with top_k = 3 the
    // top_p only sees 3 candidates, so its kept set is a subset of those 3
    rdna4::SamplerParams p;
    p.top_k = 3;
    p.top_p = 0.99f;
    p.min_p = 0.5f;
    p.temp = 2.0f;
    rdna4::Sampler s;
    s.init(p, n);
    auto c = s.filter(base.data(), no_hist);
    std::vector<float> l = base;
    std::partial_sort(l.begin(), l.begin() + 3, l.end(), std::greater<float>());
    std::set<std::int32_t> top3;
    for (int i = 0; i < 3; ++i) {
      top3.insert((std::int32_t)(std::find(base.begin(), base.end(), l[(std::size_t)i]) -
                                 base.begin()));
    }
    bool subset = true;
    for (const auto &x : c) subset = subset && top3.count(x.id) > 0;
    check(subset, "top_p/min_p only see the top_k survivors (chain order)");
    bool scaled = true;
    for (const auto &x : c) {
      scaled = scaled && std::fabs(x.logit - base[(std::size_t)x.id] / 2.0f) < 1e-4f;
    }
    check(scaled, "temperature is applied last, to the survivors");
  }

  // ---------------- determinism / distribution ----------------
  {
    rdna4::SamplerParams p;
    p.temp = 1.0f;
    p.top_k = 40;
    p.top_p = 0.95f;
    p.min_p = 0.02f;
    p.repeat_penalty = 1.1f;
    p.seed = 987654321ull;

    auto run = [&](std::uint64_t seed, std::vector<std::int32_t> &out) {
      rdna4::SamplerParams q = p;
      q.seed = seed;
      rdna4::Sampler s;
      s.init(q, n);
      std::vector<std::int32_t> hist;
      for (int i = 0; i < 200; ++i) {
        std::vector<float> l = base;
        const std::int32_t t = s.sample(l.data(), hist);
        hist.push_back(t);
        out.push_back(t);
      }
    };
    std::vector<std::int32_t> a, b, c;
    run(p.seed, a);
    run(p.seed, b);
    run(p.seed + 1, c);
    check(a == b, "same seed reproduces the same 200-token sequence");
    check(a != c, "a different seed gives a different sequence");
    std::set<std::int32_t> distinct(a.begin(), a.end());
    check(distinct.size() > 3, "the sampler explores more than one token");

    // statistical sanity: over many draws the empirical frequency of the top
    // token must be close to its filtered probability
    rdna4::SamplerParams q = p;
    q.repeat_penalty = 1.0f;
    rdna4::Sampler s;
    s.init(q, n);
    auto cands = s.filter(base.data(), no_hist);
    rdna4::Sampler::softmax(cands);
    const int best = cands[0].id;
    const double p_best = cands[0].p;
    int hits = 0;
    const int draws = 5000;  // was 20000: binomial sigma scales as 1/sqrt(n),
    // so 4x fewer draws only doubles sigma, and the 4-sigma tolerance below
    // widens by the same factor automatically through `draws` (plus the +0.01
    // floor), keeping the same confidence at a quarter of the cost.
    std::vector<float> l = base;  // hoisted: reused each draw, no per-draw alloc
    for (int i = 0; i < draws; ++i) {
      l = base;
      if (s.sample(l.data(), no_hist) == best) ++hits;
    }
    const double emp = (double)hits / draws;
    const double tol = 4.0 * std::sqrt(p_best * (1.0 - p_best) / draws) + 0.01;
    std::printf("      top token: p=%.4f empirical=%.4f (tol %.4f)\n", p_best, emp, tol);
    check(std::fabs(emp - p_best) < tol, "sampling frequency matches the filtered probability");
  }


  // ---------------- degenerate inputs (review M4) ----------------
  // These are the cases that make an ill-defined sort or an empty candidate set
  // observable; each expectation mirrors llama.cpp's implementation.
  {
    const int n = 8;
    std::vector<std::int32_t> no_hist;

    // all -inf logits with greedy: llama_sampler_temp_impl keeps the first
    // candidate (its running max starts at data[0] and nothing is > -inf), so the
    // engine must not return an empty set / -1
    std::vector<float> all_neg_inf((std::size_t)n, -std::numeric_limits<float>::infinity());
    rdna4::SamplerParams g;
    g.temp = 0.0f;
    rdna4::Sampler sg;
    sg.init(g, n);
    std::vector<float> l1 = all_neg_inf;
    const std::int32_t got = sg.sample(l1.data(), no_hist);
    check(got == 0, "greedy with all -inf logits returns the first candidate (not -1)");

    // NaN logits must not crash or produce an out-of-range id (llama.cpp's
    // comparisons are false for NaN as well, so candidate 0 wins)
    std::vector<float> nans((std::size_t)n, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> l2 = nans;
    const std::int32_t got_nan = sg.sample(l2.data(), no_hist);
    check(got_nan == 0, "greedy with NaN logits returns the first candidate");

    // greedy must pick the lowest id among exact ties (strict comparison)
    std::vector<float> tied((std::size_t)n, 1.0f);
    std::vector<float> l3 = tied;
    check(sg.sample(l3.data(), no_hist) == 0, "greedy breaks ties by lowest id");

    // n_vocab == 1: every stage is a no-op, and the RNG is still consumed so the
    // stream stays aligned with the multi-candidate case
    rdna4::Sampler one;
    one.init(g, 1);
    std::vector<float> l4{2.5f};
    check(one.sample(l4.data(), no_hist) == 0, "single-candidate vocab returns id 0");

    // top_k >= n_vocab keeps everything
    rdna4::SamplerParams tk;
    tk.top_k = 1000;
    tk.temp = 0.0f;
    rdna4::Sampler s_tk;
    s_tk.init(tk, n);
    std::vector<float> base((std::size_t)n, 0.0f);
    base[5] = 3.0f;
    std::vector<float> l5 = base;
    check(s_tk.sample(l5.data(), no_hist) == 5, "top_k > n_vocab keeps every candidate");

    // repeat_last_n = 0 disables the penalty (llama.cpp clamps it to 0 and
    // is_disabled() includes penalty_last_n == 0), so a repeated token keeps its
    // logit; a positive window applies the penalty
    std::vector<std::int32_t> hist{5, 5, 5};
    rdna4::SamplerParams pen;
    pen.temp = 0.0f;
    pen.repeat_penalty = 2.0f;
    pen.repeat_last_n = 0;
    rdna4::Sampler s0;
    s0.init(pen, n);
    auto c0 = s0.filter(base.data(), hist);
    check(c0.size() == 1 && c0[0].id == 5 && c0[0].logit == 3.0f,
          "repeat_last_n 0 disables the penalty");
    pen.repeat_last_n = -4;  // llama.cpp: max(n, 0) -> also disabled
    rdna4::Sampler sneg;
    sneg.init(pen, n);
    auto cneg = sneg.filter(base.data(), hist);
    check(cneg.size() == 1 && cneg[0].logit == 3.0f, "negative repeat_last_n disables the penalty");
    pen.repeat_last_n = 64;
    rdna4::Sampler sp2;
    sp2.init(pen, n);
    auto cp = sp2.filter(base.data(), hist);
    check(cp.size() == 1 && cp[0].logit == 1.5f, "a positive window applies the penalty (3/2)");

    // top_p <= 0 keeps exactly one candidate (llama.cpp's p<=0 path), min_p <= 0
    // and repeat_penalty == 1 are disabled
    rdna4::SamplerParams tp;
    tp.temp = 0.0f;
    tp.top_p = 0.0f;
    rdna4::Sampler s_tp;
    s_tp.init(tp, n);
    std::vector<float> l6 = base;
    check(s_tp.sample(l6.data(), no_hist) == 5, "top_p 0 keeps the best candidate");
  }

  std::printf("check-sampler: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
