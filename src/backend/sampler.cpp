// Sampler implementation (see include/rdna4/sampler.h for the semantics and the
// reference it copies).
#include "rdna4/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>

namespace rdna4 {

void Sampler::init(const SamplerParams &params, std::int32_t n_vocab) {
  params_ = params;
  n_vocab_ = n_vocab;
  rng_.seed(params.seed);
  // (a `cands_` member used to be reserved here and never touched again: ~512 KB
  // of idle heap per process, review finding M9. The working candidate list is a
  // local in sample(), which is the only place that needs it.)
}

void Sampler::softmax(std::vector<Candidate> &cands) {
  if (cands.empty()) return;
  float max_l = cands[0].logit;
  for (const Candidate &c : cands) max_l = std::max(max_l, c.logit);
  double sum = 0.0;
  for (Candidate &c : cands) {
    c.p = std::exp((double)c.logit - (double)max_l);
    sum += c.p;
  }
  if (sum <= 0.0) return;
  for (Candidate &c : cands) c.p /= sum;
}

std::vector<Candidate> Sampler::filter(const float *logits,
                                      const std::vector<std::int32_t> &history) const {
  std::vector<Candidate> cands;
  cands.reserve(1024);

  // --- penalties (llama_sampler_penalties_apply) ---------------------------
  // Count occurrences in the window, then for every candidate that appears:
  // logit <= 0 ? logit * penalty : logit / penalty. (The frequency and presence
  // penalties default to 0 in llama.cpp and are out of scope here.)
  // `repeat_last_n == 0` (or negative, which llama.cpp clamps to 0) disables the
  // penalty entirely: llama_sampler_init_penalties does penalty_last_n =
  // max(n, 0) and its is_disabled() includes penalty_last_n == 0, so there is no
  // "whole history" spelling in the reference (review M4).
  std::unordered_map<std::int32_t, int> counts;
  if (params_.repeat_penalty != 1.0f && params_.repeat_last_n > 0 && !history.empty()) {
    const std::size_t n = std::min<std::size_t>((std::size_t)params_.repeat_last_n,
                                                history.size());
    for (std::size_t i = history.size() - n; i < history.size(); ++i) {
      ++counts[history[i]];
    }
  }

  cands.resize((std::size_t)n_vocab_);
  for (std::int32_t i = 0; i < n_vocab_; ++i) {
    Candidate &c = cands[(std::size_t)i];
    c.id = i;
    c.logit = logits[i];
    if (!counts.empty()) {
      auto it = counts.find(i);
      if (it != counts.end()) {
        if (c.logit <= 0.0f) {
          c.logit *= params_.repeat_penalty;
        } else {
          c.logit /= params_.repeat_penalty;
        }
      }
    }
  }

  auto by_logit_desc = [](const Candidate &a, const Candidate &b) { return a.logit > b.logit; };

  // --- greedy (llama_sampler_temp_impl with temp <= 0) ---------------------
  // The reference does not sort here: it scans for the *first* strict maximum in
  // id order and sets everything else to -inf. Doing it before the other filters
  // keeps the same result (they can only remove candidates, and the maximum
  // always survives: top_k keeps the best, top_p keeps at least one, min_p has
  // min_keep = 1) while making ties and NaN/+-inf logits deterministic instead of
  // depending on an unstable sort's order (review M4).
  if (params_.temp <= 0.0f) {
    // same scan as llama_sampler_temp_impl: the running maximum starts at the
    // first candidate and only a *strictly* larger logit replaces it, so ties
    // keep the smaller id and all-(-inf)/NaN logits fall back to cands[0]
    // instead of yielding an empty set (review M4).
    std::size_t max_i = 0;
    for (std::size_t i = 1; i < cands.size(); ++i) {
      if (cands[i].logit > cands[max_i].logit) max_i = i;
    }
    const Candidate best = cands[max_i];
    cands.clear();
    cands.push_back(best);
    return cands;
  }

  // --- top_k (llama_sampler_top_k_impl) ------------------------------------
  if (params_.top_k > 0 && (std::size_t)params_.top_k < cands.size()) {
    std::partial_sort(cands.begin(), cands.begin() + params_.top_k, cands.end(), by_logit_desc);
    cands.resize((std::size_t)params_.top_k);
  } else {
    std::sort(cands.begin(), cands.end(), by_logit_desc);
  }

  // --- top_p (llama_sampler_top_p_apply) -----------------------------------
  if (params_.top_p < 1.0f) {
    softmax(cands);
    double cum = 0.0;
    std::size_t keep = cands.size();
    for (std::size_t i = 0; i < cands.size(); ++i) {
      cum += cands[i].p;
      if (cum >= (double)params_.top_p) {
        keep = i + 1;
        break;
      }
    }
    cands.resize(std::max<std::size_t>(keep, 1));
  }

  // --- min_p (llama_sampler_min_p_apply) -----------------------------------
  // threshold = max_logit + log(min_p): keep everything at or above it (min_keep
  // = 1, so the best token always survives).
  if (params_.min_p > 0.0f && !cands.empty()) {
    const float thresh = cands[0].logit + std::log(params_.min_p);
    std::size_t keep = 1;
    while (keep < cands.size() && cands[keep].logit >= thresh) ++keep;
    cands.resize(keep);
  }

  // --- temperature (llama_sampler_temp_impl) -------------------------------
  // (temp <= 0 was handled above: it short-circuits to the greedy candidate)
  if (params_.temp != 1.0f) {
    for (Candidate &c : cands) c.logit /= params_.temp;
  }

  return cands;
}

std::int32_t Sampler::sample(float *logits, const std::vector<std::int32_t> &history) {
  std::vector<Candidate> cands = filter(logits, history);
  if (cands.empty()) return -1;
  if (cands.size() == 1) {
    // keep the RNG stream aligned with the multi-candidate case (llama.cpp's
    // dist sampler draws once per output even for a single candidate)
    (void)rng_();
    return cands[0].id;
  }

  // --- dist (llama_sampler_dist_apply) -------------------------------------
  softmax(cands);
  // 53-bit uniform in [0, 1): the high 53 bits of one 64-bit draw
  const double rnd = (double)(rng_() >> 11) * (1.0 / 9007199254740992.0);
  double sum_run = 0.0;
  const double target = rnd;  // softmax() already normalised, so sum_cum == 1
  for (const Candidate &c : cands) {
    sum_run += c.p;
    if (sum_run >= target) return c.id;
  }
  return cands.back().id;
}

}  // namespace rdna4
