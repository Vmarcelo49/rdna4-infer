#pragma once
// feat/mtp — the CLI-side MTP modes (draft-and-score and speculative decoding).
//
// Kept out of src/main.hip so `cmd_run`'s own decode loop stays byte-identical
// when `--mtp` is not given: this is a separate entry point that takes over
// generation only when the flag is present.
//
// Three modes share one implementation:
//   * draft <= 0, !score_only : plain greedy decode (the baseline; identical to
//     cmd_run's loop, kept here so an A/B in one process cannot differ by
//     anything but the MTP code path);
//   * score_only              : plain greedy decode PLUS one MTP step per
//     committed token, scoring the head's argmax against the token the trunk
//     itself goes on to produce. Generation is unchanged, only the acceptance
//     rate is measured — the number that decides whether the head is worth
//     using at all;
//   * draft = N > 0           : speculative decoding. N tokens are drafted with
//     the MTP head (the first from the trunk's h, the rest chained through the
//     head's own h_nextn, as llama.cpp's speculative.cpp does), then verified by
//     the trunk. The trunk decides every committed token, so the output is
//     exactly plain greedy decode.
//
// Correctness contract: with temp <= 0 the committed sequence is bit-identical
// to plain greedy decode (the draft only proposes; every accepted token is one
// the trunk's own sampler produced). With temp > 0 the RNG stream would differ
// between the two paths, so MTP is refused — the caller checks this.
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "rdna4/graph.cuh"
#include "rdna4/mtp.cuh"
#include "rdna4/sampler.h"

#include <algorithm>
#include <chrono>
#include <cstdio>

namespace rdna4 {

struct MtpGenParams {
  int n_predict = 128;      // tokens to generate (already clipped to the context)
  int draft = 3;            // drafts per round in speculative mode
  bool score_only = false;  // draft-and-score instead of speculation
  bool verbose = false;
};

struct MtpGenStats {
  std::vector<std::int32_t> ids;  // generated tokens (= plain greedy's)
  bool hit_eos = false;
  int trunk_steps = 0;   // forward_tokens calls on the trunk
  int draft_steps = 0;   // MTP block forwards (drafts + cache rebuilds)
  int rounds = 0;        // speculative rounds
  long long drafted = 0; // draft tokens proposed (speculative) / scored (score)
  long long accepted = 0;  // drafts the trunk's own greedy token agreed with
  double draft_ms = 0.0;   // host time queueing the MTP head's kernels
  double trunk_ms = 0.0;   // host time queueing the trunk's kernels
  // True end-to-end wall time of the whole generation loop. The per-step numbers
  // above only measure host-side queueing (the graph is asynchronous, so they do
  // not attribute GPU time); this one is the number to compare between modes.
  double wall_ms = 0.0;
  double acceptance() const { return drafted > 0 ? (double)accepted / (double)drafted : 0.0; }
};

// Runs the loop. `hidden`/`logits` must hold the prefill result for
// `prompt_ids` (h at the prompt's last position and the logits for the next
// one); they are advanced in place.
//
// The tokenizer is not a dependency of this header on purpose: `is_eog` is the
// caller's end-of-generation predicate (llama.cpp's llama_vocab_is_eog, which
// the CLI takes from its own Tokenizer) and `on_generated` is called with the
// full prefix after every committed token so the caller can stream the new
// whole UTF-8 bytes (this header stays text-free).
//
// Header-only on purpose: CMakeLists.txt is append-only for the parallel-agent
// merge, so the CLI needs no new translation unit for its MTP modes.


inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// First maximum in id order — the same rule as Sampler's temp <= 0 branch, and
// equal to it whenever repeat_penalty == 1 (top_k/top_p/min_p only ever drop
// non-maximal candidates). That is what the score mode compares against.
inline std::int32_t argmax_id(const std::vector<float> &v) {
  std::int32_t best = 0;
  for (std::size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[(std::size_t)best]) best = (std::int32_t)i;
  }
  return best;
}

// The proposal for the position a draft step looked at: the same sampler chain
// as the trunk's own decision, applied to the draft's logits (llama.cpp samples
// its drafts with the target's sampler too). With temp <= 0 this consumes no RNG
// and is deterministic; a copy is used because the penalty stage rewrites the
// logits in place.
inline std::int32_t propose(Sampler &sm, const std::vector<float> &draft_logits,
                     const std::vector<std::int32_t> &history) {
  std::vector<float> tmp = draft_logits;
  return sm.sample(tmp.data(), history);
}


inline bool mtp_generate(Graph &g, MtpHead *mtp, const std::function<bool(std::int32_t)> &is_eog,
                  Sampler &sampler, const std::vector<std::int32_t> &prompt_ids,
                  std::vector<float> &hidden, std::vector<float> &logits, const MtpGenParams &p,
                  MtpGenStats &st, std::string &err,
                  const std::function<void(const std::vector<std::int32_t> &)> &on_generated) {
  const int ctx_size = g.max_ctx();
  const double wall0 = now_s();
  if (prompt_ids.empty()) {
    err = "mtp_generate: empty prompt";
    return false;
  }
  if (mtp != nullptr && !mtp->concat_order_e_first()) {
    std::fprintf(stderr, "rdna4-infer: MTP eh_proj input order = [hnorm(h) ; enorm(e)] "
                         "(RD_MTP_CONCAT=he)\n");
  }

  std::vector<std::int32_t> history = prompt_ids;
  std::vector<std::int32_t> &gen = st.ids;
  gen.clear();
  auto stream = [&]() {
    if (on_generated) on_generated(gen);
  };

  // Invariant at the top of every iteration: the trunk has committed tokens up
  // to position `pos`, `hidden` holds its h at that position and `logits` are
  // the distribution for position pos+1.
  int pos = (int)prompt_ids.size() - 1;
  std::vector<float> h_prev = hidden;

  auto trunk_step = [&](std::int32_t tok, int at) -> bool {
    const double t0 = now_s();
    if (!g.forward_tokens({tok}, at, hidden, logits, err)) return false;
    st.trunk_ms += now_s() - t0;
    ++st.trunk_steps;
    return true;
  };
  auto draft_step_host = [&](const float *h, std::int32_t tok, int at,
                             std::vector<float> *out) -> bool {
    const double t0 = now_s();
    if (!mtp->step_host(h, tok, at, out, nullptr, err)) return false;
    st.draft_ms += now_s() - t0;
    ++st.draft_steps;
    return true;
  };

  const int D = (mtp != nullptr && !p.score_only) ? std::max(0, p.draft) : 0;

  while ((long long)gen.size() < (long long)p.n_predict) {
    // ---- the trunk decides the next token (identical to plain greedy) ------
    const std::int32_t id = sampler.sample(logits.data(), history);
    if (id < 0) {
      err = "mtp_generate: sampler produced no token (empty candidate set)";
      return false;
    }
    if (is_eog(id)) {
      st.hit_eos = true;
      break;
    }
    gen.push_back(id);
    history.push_back(id);
    stream();
    // same stop rule as cmd_run's loop: the committed token fills the context
    if ((std::size_t)history.size() >= (std::size_t)ctx_size) break;

    // ---- MTP off: plain greedy baseline -----------------------------------
    if (mtp == nullptr) {
      if (!trunk_step(id, pos + 1)) return false;
      h_prev = hidden;
      ++pos;
      continue;
    }

    // ---- draft-and-score: one canonical step, generation untouched --------
    if (p.score_only) {
      std::vector<float> draft_logits;
      std::int32_t draft_id = 0;
      ++st.drafted;
      if (!draft_step_host(h_prev.data(), id, pos + 1, &draft_logits)) return false;
      draft_id = argmax_id(draft_logits);
      if (!trunk_step(id, pos + 1)) return false;
      h_prev = hidden;
      ++pos;
      // The trunk's own decision for the position the head just looked at. The
      // comparison reads `logits` without touching them, so the next iteration's
      // sampler call sees exactly what plain greedy would have seen.
      if (argmax_id(logits) == draft_id) ++st.accepted;
      continue;
    }

    // ---- speculative: draft up to D tokens, then verify with the trunk -----
    std::vector<std::int32_t> decisions;
    decisions.push_back(id);
    std::vector<std::vector<float>> hs;  // hs[i] = trunk h at position pos+i
    hs.push_back(h_prev);

    // A round commits 1..D_eff+1 tokens at positions pos+1..pos+1+len(decisions),
    // so it needs D_eff+2 free positions — and it must not commit past
    // n_predict, or the CLI would stream bytes the plain path never produces
    // (found by the wikitext A/B: the streaming callback fires on commit, so an
    // overshooting round cannot be fixed up by truncating `gen` afterwards).
    const int room = ctx_size - (pos + 1);
    const int room_pred = (int)((long long)p.n_predict - (long long)gen.size());
    const int D_eff =
        std::min(D, std::min(std::max(0, room - 1), std::max(0, room_pred - 1)));
    std::vector<std::int32_t> draft((std::size_t)std::max(1, D_eff), 0);
    if (D_eff == 0) {
      // no room left to verify a draft: keep the block's cache in step and
      // advance exactly like plain decode
      if (!draft_step_host(h_prev.data(), id, pos + 1, nullptr)) return false;
      if (!trunk_step(id, pos + 1)) return false;
      h_prev = hidden;
      ++pos;
      continue;
    }

    // (1) first draft: the trunk's h at `pos` with the token just committed at
    //     pos+1, i.e. exactly the canonical step llama.cpp's process() replays
    // (2) chained drafts: the head's own h_nextn becomes the next step's `h`,
    //     which is how llama.cpp drafts past the trunk (pending_h)
    std::vector<float> draft_logits;
    ++st.rounds;
    if (!draft_step_host(h_prev.data(), id, pos + 1, &draft_logits)) return false;
    draft[0] = propose(sampler, draft_logits, history);
    for (int k = 1; k < D_eff; ++k) {
      const double t0 = now_s();
      if (!mtp->step_chained(draft[(std::size_t)k - 1], pos + 1 + k, &draft_logits, nullptr, err)) {
        return false;
      }
      st.draft_ms += now_s() - t0;
      ++st.draft_steps;
      draft[(std::size_t)k] = propose(sampler, draft_logits, history);
    }
    st.drafted += D_eff;

    // (3) verify. Every trunk forward below is on an already committed token, so
    //     a rejected draft is simply never forwarded and the recurrent (GDN)
    //     state needs no rollback at all.
    if (!trunk_step(id, pos + 1)) return false;
    hs.push_back(hidden);
    int at = pos + 1;
    bool eog_next = false;
    for (int k = 0; k < D_eff; ++k) {
      const std::int32_t tid = sampler.sample(logits.data(), history);
      if (tid < 0) {
        err = "mtp_generate: sampler produced no token (empty candidate set)";
        return false;
      }
      if (is_eog(tid)) {
        decisions.push_back(tid);  // ends the sequence, never committed
        eog_next = true;
        break;
      }
      if (tid != draft[(std::size_t)k]) {
        decisions.push_back(tid);  // the trunk's own token replaces the draft
        history.push_back(tid);
        break;
      }
      decisions.push_back(draft[(std::size_t)k]);
      history.push_back(draft[(std::size_t)k]);
      ++st.accepted;
      if (k + 1 == D_eff) break;
      ++at;
      if (!trunk_step(draft[(std::size_t)k], at)) return false;
      hs.push_back(hidden);
    }

    for (std::size_t i = 1; i < decisions.size(); ++i) {
      if (eog_next && i + 1 == decisions.size()) break;
      gen.push_back(decisions[i]);
      stream();
    }
    if (eog_next) {
      st.hit_eos = true;
      break;
    }

    // the last decided token has not been forwarded yet
    ++at;
    if (!trunk_step(decisions.back(), at)) return false;
    hs.push_back(hidden);

    // (4) rebuild this block's KV rows for the positions just committed from the
    //     TRUNK's h rather than the chained one — llama.cpp's process() does the
    //     same (it clears the draft region and re-decodes it with the target's h
    //     rows), so a row's K/V never depends on which guesses preceded it
    for (std::size_t i = 1; i < decisions.size(); ++i) {
      if (!draft_step_host(hs[i].data(), decisions[i], pos + (int)i + 1, nullptr)) return false;
    }

    pos = at;
    h_prev = hidden;
  }

  st.wall_ms = (now_s() - wall0) * 1000.0;
  if ((long long)gen.size() > (long long)p.n_predict) {
    // The round clamp above makes this unreachable; if it ever fires, truncating
    // would silently hide a mismatch with plain greedy decode, so fail loudly.
    err = "mtp_generate: a round overshot n_predict (" + std::to_string(gen.size()) + " > " +
          std::to_string(p.n_predict) + ")";
    return false;
  }
  return true;
}

}  // namespace rdna4
