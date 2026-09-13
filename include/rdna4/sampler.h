// Sampler for the `run` command (PLAN.md M4).
//
// The chain and the per-stage semantics mirror llama.cpp's samplers
// (src/llama-sampler.cpp): penalties -> top_k -> top_p -> min_p -> temperature ->
// distribution, which is the default order of common/sampling.cpp with the
// stages this engine implements (dry/typical/xtc/top_n_sigma are off by default
// and out of scope per SPEC.md §2). Each stage is documented with the reference
// behaviour it copies, so a future comparison against llama.cpp is meaningful.
//
// Determinism: the RNG is our own (std::mt19937_64 + a manual 53-bit uniform),
// so a given seed gives the same text on every run and platform — the acceptance
// criterion of M4. It is deliberately NOT the same stream as llama.cpp's
// std::mt19937 (matching that would require reimplementing its distribution
// internals); temp <= 0 (greedy) is RNG-independent and therefore directly
// comparable.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace rdna4 {

struct SamplerParams {
  float temp = 1.0f;            // <= 0 => greedy (argmax of the survivors)
  int top_k = 20;               // <= 0 => disabled
  float top_p = 0.95f;          // >= 1 => disabled
  float min_p = 0.0f;           // <= 0 => disabled
  float repeat_penalty = 1.0f;  // == 1 => disabled
  int repeat_last_n = 64;       // window for the penalty (<= 0 => whole history)
  std::uint64_t seed = 0;
};

// One candidate in the working set.
struct Candidate {
  std::int32_t id = 0;
  float logit = 0.0f;
  double p = 0.0;  // filled by the softmax stages
};

class Sampler {
 public:
  void init(const SamplerParams &params, std::int32_t n_vocab);

  const SamplerParams &params() const { return params_; }

  // Applies the chain to `logits` (n_vocab floats, modified in place for the
  // penalty stage) and returns the chosen token id. `history` holds the tokens
  // generated/consumed so far, most recent last (used by the penalties stage).
  std::int32_t sample(float *logits, const std::vector<std::int32_t> &history);

  // The surviving candidate set after the deterministic stages (penalties, top_k,
  // top_p, min_p, temperature), most likely first. No RNG is consumed. Exposed
  // for tests and for the CLI's verbose output.
  std::vector<Candidate> filter(const float *logits,
                                const std::vector<std::int32_t> &history) const;

  // Softmax probabilities over `cands` (max-subtracted), for inspection.
  static void softmax(std::vector<Candidate> &cands);

 private:
  SamplerParams params_;
  std::int32_t n_vocab_ = 0;
  std::mt19937_64 rng_;
  std::vector<Candidate> cands_;
};

}  // namespace rdna4
