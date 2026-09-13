// Test tool — llama.cpp's own next-token prediction for a fixed token sequence.
//
// The eval-callback dump (reference/oracle_*_cpu.txt) records every graph node
// but not the argmax of `result_output`, so the end-to-end acceptance check for
// M3 ("do we predict what llama.cpp predicts?") needs this: it feeds raw token
// ids through llama.cpp and prints the logits sum, the argmax and the top-5 of
// the last token.
//
// usage: oracle-next-token <file.gguf> <token id> [token id...]
//        (ids as printed by llama-tokenize, e.g. 9419 1814 11 411 369 264 1228 13)
#include <llama.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <numeric>
#include <vector>

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: oracle-next-token <file.gguf> <token id> [token id...]\n");
    return 2;
  }
  std::vector<llama_token> tokens;
  for (int i = 2; i < argc; ++i) tokens.push_back((llama_token)std::atoi(argv[i]));

  llama_backend_init();
  llama_model_params mparams = llama_model_default_params();
  // CPU by default (deterministic reference); ORACLE_NGL=99 measures the same
  // model on the GPU backend, which is how backend-to-backend variation — the
  // yardstick for what a from-scratch engine can be expected to match — is
  // quantified.
  const char *ngl_env = std::getenv("ORACLE_NGL");
  mparams.n_gpu_layers = ngl_env ? std::atoi(ngl_env) : 0;
  llama_model *model = llama_model_load_from_file(argv[1], mparams);
  if (!model) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }
  llama_context_params cparams = llama_context_default_params();
  cparams.n_ctx = 512;
  cparams.n_batch = 512;
  cparams.n_ubatch = 512;
  cparams.no_perf = true;
  llama_context *ctx = llama_init_from_model(model, cparams);
  if (!ctx) {
    std::fprintf(stderr, "context failed\n");
    return 1;
  }

  // ORACLE_STEP=1 decodes one token at a time (the decode path: per-token
  // matvecs, KV cache carried across calls) instead of one batched prefill.
  // Measuring both is how llama.cpp's own path-to-path variance — the yardstick
  // for what a from-scratch per-token engine can be expected to match — is
  // quantified: with a batched prefill the CPU uses a different MUL_MAT path, and
  // node values differ by a few percent from the single-token results.
  if (std::getenv("ORACLE_STEP")) {
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      llama_batch one = llama_batch_init(1, 0, 1);
      one.n_tokens = 1;
      one.token[0] = tokens[i];
      one.pos[0] = (llama_pos)i;
      one.seq_id[0][0] = 0;   // seq_id[i] is the pointer array, seq_id[i][j] the ids
      one.n_seq_id[0] = 1;
      one.logits[0] = (i + 1 == tokens.size());
      const int rc = llama_decode(ctx, one);
      llama_batch_free(one);
      if (rc != 0) {
        std::fprintf(stderr, "decode failed at token %zu\n", i);
        return 1;
      }
    }
  } else {
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t)tokens.size());
    if (llama_decode(ctx, batch) != 0) {
      std::fprintf(stderr, "decode failed\n");
      return 1;
    }
  }
  const int n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));
  // with per-token decoding the last call had a single token, so its output sits
  // at index 0 of that batch
  const bool stepped = std::getenv("ORACLE_STEP") != nullptr;
  const float *logits = llama_get_logits_ith(ctx, stepped ? 0 : (int32_t)tokens.size() - 1);
  if (!logits) {
    std::fprintf(stderr, "no logits\n");
    return 1;
  }

  double sum = 0.0;
  for (int i = 0; i < n_vocab; ++i) sum += (double)logits[i];
  std::vector<int> idx(n_vocab);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });

  std::printf("tokens=%zu vocab=%d\n", tokens.size(), n_vocab);
  double mean = sum / n_vocab;
  double var = 0.0;
  float vmin = logits[0], vmax = logits[0];
  for (int i = 0; i < n_vocab; ++i) {
    const double d = (double)logits[i] - mean;
    var += d * d;
    vmin = std::min(vmin, logits[i]);
    vmax = std::max(vmax, logits[i]);
  }
  std::printf("result_output sum = %.6f  mean = %.6f  std = %.6f  min = %.4f  max = %.4f\n", sum, mean,
              std::sqrt(var / n_vocab), vmin, vmax);
  std::printf("argmax = %d (logit %.6f)\n", idx[0], logits[idx[0]]);
  std::printf("top5:");
  for (int i = 0; i < 5; ++i) std::printf(" %d(%.4f)", idx[i], logits[idx[i]]);
  std::printf("\n");

  llama_free(ctx);
  llama_model_free(model);
  llama_backend_free();
  return 0;
}
