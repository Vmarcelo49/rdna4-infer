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
  // feat/noite-mtp: verify the whole draft block with ONE forward_batch call
  // (Graph::forward_batch_all) instead of one per-token trunk forward per draft.
  // false restores the pre-existing one-forward-per-committed-token path, which
  // is what the A/B in tests/check_mtp_gpu.hip measures against.
  bool batch_verify = true;
  // Diagnostics/A-B only: run the shared LM head on the MTP KV *rebuild* steps
  // too (they never read a logits row, so it is 874 MiB of pure cost). Off, which
  // is what the shipping path does.
  bool rebuild_logits = false;
};

struct MtpGenStats {
  std::vector<std::int32_t> ids;  // generated tokens (= plain greedy's)
  bool hit_eos = false;
  int trunk_steps = 0;   // forward_tokens calls on the trunk
  int draft_steps = 0;   // MTP block forwards (drafts + cache rebuilds)
  int rounds = 0;        // speculative rounds
  long long drafted = 0; // draft tokens proposed (speculative) / scored (score)
  long long accepted = 0;  // drafts the trunk's own greedy token agreed with
  // ATENCAO: os tres acumuladores abaixo estao em SEGUNDOS (now_s()), apesar do
  // sufixo `_ms`. Quem imprime tem de converter (main.hip faz x1e3). O sufixo ficou
  // por compatibilidade de nome; a alternativa e' renomear e tocar em 10 sitios.
  double draft_ms = 0.0;   // SEGUNDOS: host time queueing the MTP head's kernels
  double trunk_ms = 0.0;   // SEGUNDOS: host time queueing the trunk's kernels
  // True end-to-end wall time of the whole generation loop. The per-step numbers
  // above only measure host-side queueing (the graph is asynchronous, so they do
  // not attribute GPU time); this one is the number to compare between modes.
  double wall_ms = 0.0;
  // ---- feat/noite-mtp: the batched verification path ------------------------
  int verify_forwards = 0;  // forward_batch_all calls that verified a draft block
  int replay_forwards = 0;  // forward_batch_all calls that re-ran the committed
                            // prefix after a rejected draft (state rollback)
  int rollbacks = 0;        // rounds that needed that rollback
  long long rollback_tokens = 0;  // tokens re-forwarded by those replays
  double snapshot_ms = 0.0;  // SEGUNDOS: host time spent copying the recurrent state
  double acceptance() const { return drafted > 0 ? (double)accepted / (double)drafted : 0.0; }
  // ---- acceptance governor: skip drafting when drafts mostly reject ---------
  // A round below breakeven acceptance costs a full verify batch plus usually a
  // replay and commits fewer tokens than plain greedy's one forward (measured:
  // 49% acceptance ran 30.4 tok/s vs 37.1 plain). Spec rounds accumulate into
  // epochs of RD_MTP_GOV_EPOCH (default 12); an epoch under RD_MTP_GOV_MIN
  // (default 0.60) triggers RD_MTP_GOV_COOL (default 8) plain rounds, then a
  // fresh epoch re-probes. Epochs (not an EMA) so early-sample noise cannot
  // trip it: at 72% true acceptance a 12-round epoch fires with ~5% probability
  // while at 49% it still fires ~9 times in 10.
  double gov_acc_ema = -1.0;  // EMA of per-round accepted/drafted (<0 = no data)
  long long gov_ep_acc = 0;   // accepted drafts in the current epoch
  long long gov_ep_draft = 0;  // proposed drafts in the current epoch
  int gov_ep_rounds = 0;      // scored spec rounds in the current epoch
  int gov_plain_rounds = 0;   // rounds the governor forced plain
  int gov_cooldown_left = 0;  // plain rounds remaining in this cooldown
  int gov_cooldowns = 0;      // cooldowns triggered (report only)
};

// Acceptance governor observation: call once per scored speculative round with
// the accepted/drafted counters from before the round. Cooldown rounds (forced
// plain, no verify) carry no accept information and must not call this.
inline void gov_observe(MtpGenStats &st, long long acc0, long long dr0, double min_acc,
                        int cool, int epoch) {
  const long long dd = st.drafted - dr0;
  if (dd <= 0) return;
  const double r = (double)(st.accepted - acc0) / (double)dd;
  st.gov_acc_ema = (st.gov_acc_ema < 0.0) ? r : 0.75 * st.gov_acc_ema + 0.25 * r;
  st.gov_ep_acc += st.accepted - acc0;
  st.gov_ep_draft += dd;
  if (++st.gov_ep_rounds < epoch || st.gov_cooldown_left != 0) return;
  if ((double)st.gov_ep_acc < min_acc * (double)st.gov_ep_draft) {
    st.gov_cooldown_left = cool;
    ++st.gov_cooldowns;
  }
  st.gov_ep_acc = st.gov_ep_draft = 0;
  st.gov_ep_rounds = 0;
}

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

  // Governor config, read once (env only affects the run path, never gates):
  // RD_MTP_GOV=0 disables, RD_MTP_GOV_MIN sets the breakeven floor (default
  // 0.60, measured), RD_MTP_GOV_EPOCH the scored rounds per decision (default
  // 12), RD_MTP_GOV_COOL the plain rounds per cooldown (default 8).
  const char *gov_e = std::getenv("RD_MTP_GOV");
  const bool gov_on = (gov_e == nullptr || std::string(gov_e) != "0") && D > 0;
  double gov_min = 0.60;
  if (const char *v = std::getenv("RD_MTP_GOV_MIN")) {
    gov_min = std::atof(v);
  }
  int gov_epoch = 12;
  if (const char *v = std::getenv("RD_MTP_GOV_EPOCH")) {
    gov_epoch = std::max(1, std::atoi(v));
  }
  int gov_cool = 8;
  if (const char *v = std::getenv("RD_MTP_GOV_COOL")) {
    gov_cool = std::max(1, std::atoi(v));
  }

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
    // A round commits 1..D_eff+1 tokens at positions pos+1..pos+1+len(decisions),
    // so it needs D_eff+2 free positions — and it must not commit past
    // n_predict, or the CLI would stream bytes the plain path never produces
    // (found by the wikitext A/B: the streaming callback fires on commit, so an
    // overshooting round cannot be fixed up by truncating `gen` afterwards).
    const int room = ctx_size - (pos + 1);
    const int room_pred = (int)((long long)p.n_predict - (long long)gen.size());
    // The batched path runs [id, d1..dk] as one forward_batch call and, after a
    // rejection, re-runs [id, d1..d_a, t] as a second one: BOTH sizes need a
    // batched matvec instantiation (2/3/4/8/16), and 2..4 is the largest range
    // that covers every replay size, so the batched draft is capped at 3. A
    // larger D stays available on the serial path (MtpGenParams::batch_verify).
    const int D_max = p.batch_verify ? std::min(D, 3) : D;
    int D_eff =
        std::min(D_max, std::min(std::max(0, room - 1), std::max(0, room_pred - 1)));
    if (gov_on && D_eff > 0 && st.gov_cooldown_left > 0) {
      // governor cooldown: run this round plain (the D_eff == 0 branch below:
      // one draft step keeps the block cache in step, one trunk step advances
      // -- token-identical to verifying, minus the batch and any replay).
      --st.gov_cooldown_left;
      ++st.gov_plain_rounds;
      D_eff = 0;
    }
    if (D_eff == 0) {
      // no room left to verify a draft: keep the block's cache in step and
      // advance exactly like plain decode
      if (!draft_step_host(h_prev.data(), id, pos + 1, nullptr)) return false;
      if (!trunk_step(id, pos + 1)) return false;
      h_prev = hidden;
      ++pos;
      continue;
    }

    // ================= batched verification (feat/noite-mtp) ================
    // One trunk pass over the whole draft block instead of one per committed
    // token: the D_eff+1 tokens share a single read of every weight (the M8
    // matvec), so a round costs one weight pass plus the per-token scaffolding.
    if (p.batch_verify) {
      // (1) drafts. The first uses the trunk's h at `pos` with the token just
      //     committed at pos+1, exactly like llama.cpp's process(); the rest
      //     chain through the head's own h_nextn (pending_h), which is how it
      //     drafts past the trunk without another trunk pass.
      std::vector<float> draft_logits;
      std::vector<std::int32_t> draft((std::size_t)D_eff, 0);
      ++st.rounds;
      const long long gov_acc0 = st.accepted, gov_dr0 = st.drafted;
      if (!draft_step_host(h_prev.data(), id, pos + 1, &draft_logits)) return false;
      draft[0] = propose(sampler, draft_logits, history);
      for (int k = 1; k < D_eff; ++k) {
        const double t0 = now_s();
        if (!mtp->step_chained(draft[(std::size_t)k - 1], pos + 1 + k, &draft_logits, nullptr,
                               err)) {
          return false;
        }
        st.draft_ms += now_s() - t0;
        ++st.draft_steps;
        draft[(std::size_t)k] = propose(sampler, draft_logits, history);
      }
      st.drafted += D_eff;

      // (2) ONE batched trunk forward over [id, d1..dk] at pos+1. It advances the
      //     KV cache and the GDN state for every row it holds, so the recurrent
      //     state is snapshotted first: a rejected draft must not leave that
      //     advance behind. (The KV caches need no snapshot — attention is causal
      //     and every row is rewritten by position before it is read again.)
      std::vector<std::int32_t> vtoks;
      vtoks.reserve((std::size_t)D_eff + 1);
      vtoks.push_back(id);
      vtoks.insert(vtoks.end(), draft.begin(), draft.end());
      std::vector<float> h_rows, l_rows;
      const double ts0 = now_s();
      if (!g.state_snapshot(err)) return false;
      const double tv0 = now_s();
      if (!g.forward_batch_all(vtoks, pos + 1, h_rows, l_rows, err)) return false;
      const double tv1 = now_s();
      st.snapshot_ms += tv0 - ts0;
      st.trunk_ms += tv1 - tv0;
      ++st.verify_forwards;

      // (3) the trunk's own greedy token at each drafted position, read from row
      //     k of the verification logits: row j holds the distribution for
      //     position pos+2+j, which is where draft j guessed. Same sampler chain
      //     and same history as the serial path, so the decisions are the same.
      const int nv = mtp->n_vocab();
      std::vector<std::int32_t> committed;  // committed tokens beyond `id`
      bool rejected = false;
      for (int k = 0; k < D_eff; ++k) {
        const std::int32_t tid = sampler.sample(l_rows.data() + (std::size_t)k * nv, history);
        if (tid < 0) {
          err = "mtp_generate: sampler produced no token (empty candidate set)";
          return false;
        }
        if (tid != draft[(std::size_t)k]) {
          // The trunk's own token here (EOG included) is NOT committed in this
          // round: the round's state stops at the last accepted draft, so this
          // token opens the *next* round instead. The replayed logits reproduce
          // it exactly, so the committed sequence is unchanged — and the replay
          // stays one row shorter than it would be if the replacement were
          // committed and forwarded here.
          rejected = true;
          break;
        }
        committed.push_back(draft[(std::size_t)k]);
        history.push_back(draft[(std::size_t)k]);
        ++st.accepted;
      }
      if (gov_on) gov_observe(st, gov_acc0, gov_dr0, gov_min, gov_cool, gov_epoch);

      // Every token in `committed` is one the trunk itself produced (an accepted
      // draft is by definition the trunk's own greedy token), so committing them
      // is what keeps this path token-for-token plain greedy decode.
      for (std::int32_t t : committed) {
        gen.push_back(t);
        stream();
      }

      // (4) the authoritative token list of the round: all committed. Full
      //     acceptance keeps the verification batch (its state is already exactly
      //     at the last row); a rejection restores the snapshot and re-runs the
      //     committed prefix, whose last row then holds the state, the h and the
      //     logits of the next round. A prefix of one token (every draft
      //     rejected) takes the per-token path: it is the same arithmetic and it
      //     is cheaper than the smallest batch.
      std::vector<std::int32_t> ftoks;
      ftoks.reserve((std::size_t)D_eff + 1);
      ftoks.push_back(id);
      ftoks.insert(ftoks.end(), committed.begin(), committed.end());
      if (rejected) {
        const double tr0 = now_s();
        if (!g.state_restore(err)) return false;
        const double tr1 = now_s();
        bool ok_replay = false;
        if (ftoks.size() == 1) {
          ok_replay = g.forward_tokens(ftoks, pos + 1, hidden, logits, err);
          if (ok_replay) {
            h_rows = hidden;
            l_rows = logits;
          }
        } else {
          ok_replay = g.forward_batch_all(ftoks, pos + 1, h_rows, l_rows, err);
        }
        if (!ok_replay) return false;
        const double tr2 = now_s();
        st.snapshot_ms += tr1 - tr0;
        st.trunk_ms += tr2 - tr1;
        ++st.replay_forwards;
        ++st.rollbacks;
        st.rollback_tokens += (long long)ftoks.size();
      }

      // (5) rebuild this block's own KV rows for the committed positions from the
      //     TRUNK's h instead of the chained one — llama.cpp's process() does the
      //     same (it re-decodes the draft region with the target's h rows), so a
      //     row's K/V never depends on which guesses preceded it. h_rows[i-1] is
      //     the trunk h at the position of ftoks[i]; the row for `id` was written
      //     by the first draft step above and needs no rebuild. The shared LM head
      //     is skipped here (want_logits = false): nothing reads a logits row, and
      //     it is 874 MiB of the step's traffic.
      const int E = g.n_embd();
      for (std::size_t i = 1; i < ftoks.size(); ++i) {
        const double t0 = now_s();
        if (!mtp->step_host(h_rows.data() + (i - 1) * (std::size_t)E, ftoks[i], pos + 1 + (int)i,
                            nullptr, nullptr, err, /*want_logits=*/p.rebuild_logits)) {
          return false;
        }
        st.draft_ms += now_s() - t0;
        ++st.draft_steps;
      }

      pos += (int)ftoks.size();  // ftoks[0] is at pos+1, the last at pos+len
      h_prev.assign(h_rows.end() - E, h_rows.end());
      logits.assign(l_rows.end() - nv, l_rows.end());
      continue;
    }
    // =============== end batched verification ==============================

    std::vector<std::int32_t> decisions;
    decisions.push_back(id);
    std::vector<std::vector<float>> hs;  // hs[i] = trunk h at position pos+i
    hs.push_back(h_prev);
    std::vector<std::int32_t> draft((std::size_t)std::max(1, D_eff), 0);

    // (1) first draft: the trunk's h at `pos` with the token just committed at
    //     pos+1, i.e. exactly the canonical step llama.cpp's process() replays
    // (2) chained drafts: the head's own h_nextn becomes the next step's `h`,
    //     which is how llama.cpp drafts past the trunk (pending_h)
    std::vector<float> draft_logits;
    ++st.rounds;
    const long long gov_s_acc0 = st.accepted, gov_s_dr0 = st.drafted;
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
    if (gov_on) gov_observe(st, gov_s_acc0, gov_s_dr0, gov_min, gov_cool, gov_epoch);

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

// Batched MTP prefill (feat/noite-mtp). The prompt runs through the trunk in
// chunks of up to Graph::kMaxBatch tokens via forward_batch_all (one weight pass
// per chunk, the M8 kernels), and the draft block is stepped once per token with
// want_logits = false: its rows are (token i, trunk h of i-1), so the only thing
// the prompt pass needs from the head is the block's own KV row — never a logits
// row. Bit-identical to the per-token prefill: the trunk chunks are the same
// arithmetic (tests/check_batch_gpu.hip) and the block's steps are the same calls
// with the same h rows. On `hidden`/`logits` it leaves exactly what the per-token
// loop leaves: the h of the last prompt position and the distribution for the
// position after it.
inline bool mtp_prefill_batched(Graph &g, MtpHead &mtp, const std::vector<std::int32_t> &ids,
                                std::vector<float> &hidden, std::vector<float> &logits,
                                std::string &err) {
  static const int kSizes[] = {16, 8, 4, 3, 2};
  const int E = g.n_embd();
  std::vector<float> h_rows, l_rows;
  std::vector<float> h_prev((std::size_t)E, 0.0f);  // h at position -1 (pending_h)
  std::size_t pos = 0;
  while (pos < ids.size()) {
    std::size_t take = 1;
    for (int sz : kSizes) {
      if (ids.size() - pos >= (std::size_t)sz) {
        take = (std::size_t)sz;
        break;
      }
    }
    if (take == 1) {
      if (!g.forward_tokens({ids[pos]}, (int)pos, hidden, logits, err)) return false;
      if (!mtp.step_host(h_prev.data(), ids[pos], (int)pos, nullptr, nullptr, err,
                         /*want_logits=*/false)) {
        return false;
      }
      h_prev = hidden;
      pos += 1;
      continue;
    }
    const std::vector<std::int32_t> chunk(ids.begin() + (long)pos,
                                          ids.begin() + (long)(pos + take));
    if (!g.forward_batch_all(chunk, (int)pos, h_rows, l_rows, err)) return false;
    for (std::size_t t = 0; t < take; ++t) {
      const float *h_in = t == 0 ? h_prev.data() : h_rows.data() + (t - 1) * (std::size_t)E;
      if (!mtp.step_host(h_in, chunk[t], (int)(pos + t), nullptr, nullptr, err,
                         /*want_logits=*/false)) {
        return false;
      }
    }
    h_prev.assign(h_rows.end() - E, h_rows.end());
    hidden.assign(h_rows.end() - E, h_rows.end());
    logits.assign(l_rows.end() - g.n_vocab(), l_rows.end());
    pos += take;
  }
  return true;
}

}  // namespace rdna4
