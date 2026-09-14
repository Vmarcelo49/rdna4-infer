// rdna4-infer CLI.
//
// Subcommands:
//   (none) / info  — M0 device gate + VRAM budget check, optionally --list-tensors
//   run            — M4: tokenize a prompt, run the graph, sample tokens, stream
//                    the decoded text to stdout (stats go to stderr)
//
// stdout carries *only* generated text for `run`, so a golden test can diff it
// byte for byte; every diagnostic goes to stderr.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "rdna4/chat.h"
#include "rdna4/cli_commands.h"
#include "rdna4/device.h"
#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/mtp_gen.h"
#include "rdna4/sampler.h"
#include "rdna4/stream.h"
#include "rdna4/tokenizer.h"

namespace {

void usage_info() {
  std::fprintf(stderr,
               "usage: rdna4-infer [info] [-m model.gguf] [--ctx-size N] [--cache-type-k TYPE] "
               "[--cache-type-v TYPE] [--list-tensors]\n  cache types: f32, f16 (default), "
               "q8_0, q4_0, q5_0, q4_1\n");
}

void usage_run() {
  std::fprintf(
      stderr,
      "usage: rdna4-infer run -m model.gguf [-p PROMPT] [options]\n"
      "  -m PATH                 GGUF model (required)\n"
      "  -p TEXT                 prompt text (default: \"Hello\")\n"
      "  -n N                    tokens to generate (default 128)\n"
      "  --chat                  render the prompt through the qwen35 chat template\n"
      "  --system TEXT           system message for --chat\n"
      "  --no-thinking           disable thinking (--chat)\n"
      "  --reasoning-effort E    xhigh (default) | medium | low\n"
      "  --temp F                temperature, <= 0 => greedy (default 1.0)\n"
      "  --top-k N               (metadata default 20; <= 0 disables)\n"
      "  --top-p F               (metadata default 0.95; >= 1 disables)\n"
      "  --min-p F               (default 0.0; <= 0 disables)\n"
      "  --repeat-penalty F      (default 1.0)\n"
      "  --repeat-last-n N       penalty window (default 64; 0 disables the penalty)\n"
      "  --seed N                RNG seed (default 0)\n"
      "  --greedy                same as --temp 0\n"
      "  --ctx-size N            context (default 4096)\n"
      "  --cache-type-k TYPE     f32, f16 (default), q8_0, q4_0, q5_0, q4_1\n"
      "  --cache-type-v TYPE     f32, f16 (default), q8_0, q4_0, q5_0, q4_1\n"
      // ---- feat/mtp ----
      "  --mtp                   use the blk.64 NextN/MTP head for speculative decoding\n"
      "  --draft N               tokens drafted per round with the MTP head (default 3)\n"
      "  --mtp-score             draft-and-score only: report the head's acceptance rate\n"
      "                          against the trunk's own greedy tokens (generation is\n"
      "                          unchanged); implies --mtp\n"
      // ---- end feat/mtp ----
      "  -v                      verbose: per-step timings and the token ids\n"
      "  --no-stats              suppress the stderr statistics\n");
}

bool parse_u64(const char *s, std::uint64_t *out) {
  if (s[0] == '-') return false;  // strtoull would wrap "-1" into a huge value
  char *end = nullptr;
  unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  *out = static_cast<std::uint64_t>(v);
  return true;
}

bool parse_i64(const char *s, long long *out) {
  char *end = nullptr;
  long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') {
    return false;
  }
  *out = v;
  return true;
}

bool parse_f32(const char *s, float *out) {
  char *end = nullptr;
  float v = std::strtof(s, &end);
  if (end == s || *end != '\0' || !std::isfinite(v)) {
    return false;
  }
  *out = v;
  return true;
}

// "f32" | "f16" | "q8_0" | "q4_0" | "q5_0" | "q4_1" -> KvType (kv.h). The names
// match llama.cpp's --cache-type-k/-v values for the types this engine
// implements; the parse table itself lives in kv.h so the CLI, the budget and the
// graph cannot drift apart (there used to be three copies of this list).
bool parse_kv_type(const char *s, rdna4::KvType *out) {
  return rdna4::kv_type_parse(s, out) == nullptr;
}

// The GGUF's own sampling defaults (general.sampling.*), which SPEC.md §1.4 and
// PLAN.md M4 item 1 require the CLI to honour. Any value the user passes on the
// command line wins; these are only the starting point, so a different file's
// metadata is respected instead of a hardcoded 20/0.95/1.0.
void apply_metadata_sampling(const rdna4::gguf::File &gf, rdna4::SamplerParams *sp,
                             bool set_temp, bool set_top_k, bool set_top_p, bool set_min_p) {
  auto num = [&](const char *key, double *out) {
    auto it = gf.kv.find(key);
    if (it == gf.kv.end()) return false;
    const rdna4::gguf::Value &v = it->second;
    const bool is_float = v.type == rdna4::gguf::F32 || v.type == rdna4::gguf::F64;
    if (!is_float && (std::int32_t)v.type > (std::int32_t)rdna4::gguf::U64) return false;
    *out = is_float ? v.f : (double)v.u;
    return true;
  };
  double d = 0.0;
  if (set_top_k && num("general.sampling.top_k", &d) && d > 0) sp->top_k = (int)d;
  if (set_top_p && num("general.sampling.top_p", &d) && d > 0.0 && d <= 1.0) sp->top_p = (float)d;
  if (set_temp && num("general.sampling.temp", &d) && d >= 0.0) sp->temp = (float)d;
  if (set_min_p && num("general.sampling.min_p", &d) && d >= 0.0) sp->min_p = (float)d;
}

const char *kv_type_cstr(rdna4::KvType t) {
  switch (t) {
    case rdna4::KvType::F32: return "f32";
    case rdna4::KvType::F16: return "f16";
    case rdna4::KvType::Q8_0: return "q8_0";
    case rdna4::KvType::Q4_0: return "q4_0";
    case rdna4::KvType::Q5_0: return "q5_0";
    case rdna4::KvType::Q4_1: return "q4_1";
  }
  return "?";
}

double now_s() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}


// ---------------------------------------------------------------------------
// M8: batched prefill. Splits the prompt into the chunk sizes the batched kernels
// have instantiations for (16/8/4/3/2), with single tokens for the tail. Every
// chunk is bit-identical to processing the tokens one at a time
// (tests/check_batch_gpu.hip asserts exactly that, including mixed chunks).
// ---------------------------------------------------------------------------
bool prefill_ids(rdna4::Graph &g, const std::vector<std::int32_t> &ids, std::vector<float> &hidden,
             std::vector<float> &logits, std::string &err) {
  static const int kSizes[] = {16, 8, 4, 3, 2};
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
    } else {
      const std::vector<std::int32_t> chunk(ids.begin() + (long)pos, ids.begin() + (long)(pos + take));
      if (!g.forward_batch(chunk, (int)pos, hidden, logits, err)) return false;
    }
    pos += take;
  }
  return true;
}

// ---------------------------------------------------------------------------
// bench: M5 measurement harness. Reports the numbers SPEC.md §3 and the README
// quote, for one (model, ctx, kv type) combination:
//   - load time and VRAM in use
//   - prefill: the per-token path (this engine has no batched prefill), so the
//     prompt is timed token by token
//   - decode: greedy loop, RNG-free, with a warm-up phase that is excluded —
//     gfx1201 drops to a deep DPM state between kernels (SCLK 9-16 MHz), so a
//     cold measurement is ~1.5x too low (see PLAN.md M2 "Passo 5")
//   - effective bandwidth = model bytes read per token / time
// `--fill-cache` seeds every KV cache and the GDN state (Graph::debug_fill_caches)
// so decode can be measured as if the context were full, without waiting for a
// real long prefill — that is a synthetic cache and is labelled as such.
// ---------------------------------------------------------------------------
int cmd_bench(int argc, char **argv) {
  const char *model_path = nullptr;
  std::string prompt = "The capital of France is";
  long long n_decode = 64;
  long long warmup = 8;
  long long reps = 3;
  long long prefill = 0;  // 0 => prompt length
  std::uint64_t ctx_size = 4096;
  rdna4::KvType kv_k = rdna4::KvType::F16, kv_v = rdna4::KvType::F16;
  bool fill_cache = false;
  std::uint64_t seed = 0;
  long long layers = -1;  // < 0 => all (diagnostic: attribute layer vs head cost)
  long long start_pos = -1;  // >= 0 => decode starts at this position (see --fill-cache)

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "rdna4-infer bench: %s needs a value\n", what);
        return nullptr;
      }
      return argv[++i];
    };
    if (std::strcmp(a, "-m") == 0) {
      const char *v = need("-m");
      if (!v) return 1;
      model_path = v;
    } else if (std::strcmp(a, "-p") == 0) {
      const char *v = need("-p");
      if (!v) return 1;
      prompt = v;
    } else if (std::strcmp(a, "-n") == 0 || std::strcmp(a, "--decode") == 0) {
      const char *v = need("-n");
      if (!v || !parse_i64(v, &n_decode) || n_decode < 1) {
        return 1;
      }
    } else if (std::strcmp(a, "--warmup") == 0) {
      const char *v = need("--warmup");
      if (!v || !parse_i64(v, &warmup) || warmup < 0) {
        return 1;
      }
    } else if (std::strcmp(a, "--reps") == 0) {
      const char *v = need("--reps");
      if (!v || !parse_i64(v, &reps) || reps < 1) {
        return 1;
      }
    } else if (std::strcmp(a, "--prefill") == 0) {
      const char *v = need("--prefill");
      if (!v || !parse_i64(v, &prefill) || prefill < 0) {
        return 1;
      }
    } else if (std::strcmp(a, "--ctx-size") == 0) {
      const char *v = need("--ctx-size");
      if (!v || !parse_u64(v, &ctx_size) || ctx_size == 0 ||
          ctx_size > rdna4::kMaxCtxSize) {
        return 1;
      }
    } else if (std::strcmp(a, "--cache-type-k") == 0) {
      const char *v = need("--cache-type-k");
      if (!v || !parse_kv_type(v, &kv_k)) return 1;
    } else if (std::strcmp(a, "--cache-type-v") == 0) {
      const char *v = need("--cache-type-v");
      if (!v || !parse_kv_type(v, &kv_v)) return 1;
    } else if (std::strcmp(a, "--fill-cache") == 0) {
      fill_cache = true;
    } else if (std::strcmp(a, "--start-pos") == 0) {
      const char *v = need("--start-pos");
      if (!v || !parse_i64(v, &start_pos) || start_pos < 0) return 1;
    } else if (std::strcmp(a, "--layers") == 0) {
      const char *v = need("--layers");
      if (!v || !parse_i64(v, &layers) || layers < 0) return 1;
    } else if (std::strcmp(a, "--seed") == 0) {
      const char *v = need("--seed");
      if (!v || !parse_u64(v, &seed)) return 1;
    } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      std::printf("usage: rdna4-infer bench -m model.gguf [-p TEXT] [-n DECODE] [--warmup N] "
                  "[--reps N] [--prefill N] [--start-pos N] [--ctx-size N] [--cache-type-k T] "
                  "[--cache-type-v T] [--fill-cache] [--seed N] [--layers N]\n");
      return 0;
    } else {
      std::fprintf(stderr, "rdna4-infer bench: unknown argument '%s'\n", a);
      return 1;
    }
  }
  if (model_path == nullptr) {
    std::fprintf(stderr, "rdna4-infer bench: -m model.gguf is required\n");
    return 1;
  }

  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0 ||
      !rdna4::is_gfx1201(arch)) {
    std::fprintf(stderr, "rdna4-infer: requires gfx1201, found %s\n", arch);
    return 2;
  }
  struct stat st;
  if (stat(model_path, &st) != 0) {
    std::fprintf(stderr, "rdna4-infer: cannot stat %s\n", model_path);
    return 1;
  }

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(model_path, err)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "rdna4-infer: config failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "rdna4-infer: tokenizer failed: %s\n", err.c_str());
    return 1;
  }
  std::vector<std::int32_t> ids = tk.encode(prompt, /*parse_special=*/true);
  if (ids.empty()) {
    std::fprintf(stderr, "rdna4-infer bench: the prompt produced no tokens\n");
    return 1;
  }
  // a longer timed prefill: repeat the prompt's own tokens
  while (prefill > 0 && (long long)ids.size() < prefill) {
    const std::size_t before = ids.size();
    for (std::size_t i = 0; i < before && (long long)ids.size() < prefill; ++i) {
      ids.push_back(ids[i]);
    }
  }
  if ((std::uint64_t)ids.size() + (std::uint64_t)n_decode + (std::uint64_t)warmup > ctx_size) {
    ctx_size = (std::uint64_t)ids.size() + (std::uint64_t)n_decode + (std::uint64_t)warmup;
  }

  const double t0 = now_s();
  rdna4::Graph graph(loader, cfg);
  if (!graph.init((int)ctx_size, kv_k, kv_v, err)) {
    std::fprintf(stderr, "rdna4-infer: graph init failed: %s\n", err.c_str());
    return 1;
  }
  const double t_load = now_s() - t0;

  graph.debug_set_layer_limit(layers < 0 ? graph.n_layer() : (int)layers);
  rdna4::SamplerParams sp;  // greedy: bench must be deterministic and RNG-free
  sp.temp = 0.0f;
  sp.seed = seed;
  rdna4::Sampler sampler;
  sampler.init(sp, (std::int32_t)graph.n_vocab());

  graph.set_want_hidden(false);  // the bench does not use `hidden`
  std::vector<float> hidden, logits;
  // Decoding at position N requires the cache to hold something: the trunk reads
  // positions 0..N-1, so seed it (the same synthetic-cache trick as
  // check-kvctx-gpu). It is a COST measurement, not a quality one.
  if (start_pos > 0 && !fill_cache) {
    fill_cache = true;
    std::fprintf(stderr, "rdna4-infer: --start-pos implies --fill-cache (seeding the cache)\n");
  }
  if (fill_cache && !graph.debug_fill_caches((unsigned)seed, err)) {
    std::fprintf(stderr, "rdna4-infer: fill cache failed: %s\n", err.c_str());
    return 1;
  }
  const double tp0 = now_s();
  if (!prefill_ids(graph, ids, hidden, logits, err)) {
    std::fprintf(stderr, "rdna4-infer: prefill failed: %s\n", err.c_str());
    return 1;
  }
  const double t_prefill = now_s() - tp0;

  const std::size_t bytes_per_token = loader.total_bytes();
  const double gib = 1024.0 * 1024.0 * 1024.0;

  // decode: warm up outside the timed window, then `reps` timed passes
  std::vector<std::int32_t> history = ids;
  double sample_ms = 0.0;
  graph.reset_host_timing();
  const std::size_t first_pos =
      start_pos > 0 ? (std::size_t)start_pos : ids.size();
  if (first_pos >= ctx_size) {
    std::fprintf(stderr, "rdna4-infer bench: --start-pos %zu beyond ctx %llu\n", first_pos,
                 (unsigned long long)ctx_size);
    return 1;
  }
  auto decode_pass = [&](long long n, double *ms_out, std::size_t *n_out) {
    const double a = now_s();
    std::size_t done = 0;
    std::size_t pos = first_pos;
    for (long long i = 0; i < n; ++i) {
      const double s0 = now_s();
      const std::int32_t id = sampler.sample(logits.data(), history);
      sample_ms += (now_s() - s0) * 1000.0;
      if (id < 0) return false;
      history.push_back(id);
      if (pos >= ctx_size) break;
      if (!graph.forward_tokens({id}, (int)pos, hidden, logits, err)) return false;
      ++pos;
      ++done;
    }
    *ms_out = (now_s() - a) * 1000.0;
    *n_out = done;
    return true;
  };
  if (warmup > 0) {
    double ms = 0;
    std::size_t done = 0;
    if (!decode_pass(warmup, &ms, &done)) {
      std::fprintf(stderr, "rdna4-infer: warmup failed: %s\n", err.c_str());
      return 1;
    }
  }
  double best_ms = 0, sum_ms = 0;
  std::size_t best_n = 0, total_tokens_timed = 0;
  double best_queue = 0, best_tail = 0, best_sample = 0;
  for (long long r = 0; r < reps; ++r) {
    double ms = 0;
    std::size_t done = 0;
    const double sample_before = sample_ms;
    graph.reset_host_timing();
    if (!decode_pass(n_decode, &ms, &done) || done == 0) {
      std::fprintf(stderr, "rdna4-infer: decode failed: %s\n", err.c_str());
      return 1;
    }
    if (best_ms == 0 || ms < best_ms) {
      best_ms = ms;
      best_n = done;
      best_queue = graph.host_launch_ms();
      best_tail = graph.host_readback_ms();
      best_sample = sample_ms - sample_before;
    }
    sum_ms += ms;
    total_tokens_timed += done;
    // rewind: the sequence length must stay in range across reps
    history.resize(ids.size());
    if (fill_cache && !graph.debug_fill_caches((unsigned)seed, err)) {
      // Without this the rep would time the attention over uninitialised
      // VRAM and print tok/s for a different experiment (review finding H4).
      std::fprintf(stderr, "rdna4-infer: cache fill failed: %s\n", err.c_str());
      return 1;
    }
    if (!prefill_ids(graph, ids, hidden, logits, err)) {
      std::fprintf(stderr, "rdna4-infer: reset failed: %s\n", err.c_str());
      return 1;
    }
  }
  const double tok_s = (double)best_n / (best_ms / 1000.0);
  // bytes read per *token* (all trunk weights once) divided by the per-token time
  const double sec_per_token = (best_ms / 1000.0) / (double)best_n;
  const double gb_s = (double)bytes_per_token / sec_per_token / 1e9;

  std::size_t free_b = 0, tot_b = 0;
  const bool have_vram = hipMemGetInfo(&free_b, &tot_b) == hipSuccess;
  std::printf("model: %s\n", model_path);
  std::printf("arch %s, %d layers (%d full-attn), embd %d, vocab %d\n", arch, graph.n_layer(),
              graph.n_full_attn(), graph.n_embd(), graph.n_vocab());
  if (layers >= 0) {
    std::printf("layers limited to %d (diagnostic: the output is meaningless)\n",
                graph.debug_layer_limit());
  }
  std::printf("ctx %llu, kv %s/%s%s, load %.2f s\n", (unsigned long long)ctx_size,
              kv_type_cstr(kv_k), kv_type_cstr(kv_v),
              fill_cache ? " (synthetic cache)" : "", t_load);
  std::printf("decode positions: %zu..%zu%s\n", first_pos, first_pos + (std::size_t)best_n,
              start_pos > 0 ? "  [synthetic keys/values: cost measurement, not quality]" : "");
  if (have_vram) {
    std::printf("vram in use: %.2f GiB (free %.2f of %.2f GiB)\n",
                (double)(tot_b - free_b) / gib, (double)free_b / gib, (double)tot_b / gib);
  } else {
    std::printf("vram in use: unavailable (hipMemGetInfo failed)\n");
  }
  std::printf("prefill %zu tokens: %.3f s (%.2f tok/s, batched N<=16)\n", ids.size(), t_prefill,
              (double)ids.size() / t_prefill);
  std::printf("decode %zu tokens x %lld reps (warmup %lld): best %.3f s (%.2f tok/s), "
              "mean %.2f tok/s\n",
              best_n, reps, warmup, best_ms / 1000.0, tok_s,
              (double)best_n * (double)reps / (sum_ms / 1000.0));
  std::printf("effective bandwidth: %.1f GB/s (%.2f GiB per token)\n", gb_s,
              (double)bytes_per_token / gib);
  // where the wall time goes, split host vs device (M5 diagnostics)
  const double tot_ms = best_ms / (double)best_n;
  const double q_ms = best_queue / (double)best_n;
  const double r_ms = best_tail / (double)best_n;
  const double s_ms = best_sample / (double)best_n;
  (void)total_tokens_timed;
  std::printf("per token: %.2f ms = launch loop %.2f (host thread; while the command queue is full "
              "this is back-pressure from the device, not host cost) + tail %.2f (trunk drain, LM "
              "head, logits copy) + sample %.2f\n",
              tot_ms, q_ms, r_ms, s_ms);
  return 0;
}

// ---------------------------------------------------------------------------
// tokenize: dump the ids of a file (one per line), CPU-only. M5 added it to
// compare the engine's BPE against llama.cpp over a real corpus
// (tests/oracle-tokenize --raw): a single different id shifts every perplexity
// chunk, so the perplexity comparison is only meaningful once the id streams are
// identical.
// ---------------------------------------------------------------------------
int cmd_tokenize(int argc, char **argv) {
  const char *model_path = nullptr;
  const char *file = nullptr;
  bool parse_special = false;  // common_tokenize's default (llama-perplexity's mode)
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "-f") == 0 && i + 1 < argc) {
      file = argv[++i];
    } else if (std::strcmp(argv[i], "--parse-special") == 0) {
      parse_special = true;
    } else if (std::strcmp(argv[i], "-h") == 0 || std::strcmp(argv[i], "--help") == 0) {
      std::printf("usage: rdna4-infer tokenize -m model.gguf -f file.txt [--parse-special]\n");
      return 0;
    } else {
      std::fprintf(stderr, "rdna4-infer tokenize: unknown argument '%s'\n", argv[i]);
      return 1;
    }
  }
  if (model_path == nullptr || file == nullptr) {
    std::fprintf(stderr, "rdna4-infer tokenize: -m and -f are required\n");
    return 1;
  }
  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(model_path, err)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "rdna4-infer: tokenizer failed: %s\n", err.c_str());
    return 1;
  }
  std::string text;
  {
    std::FILE *fp = std::fopen(file, "rb");
    if (fp == nullptr) {
      std::fprintf(stderr, "rdna4-infer tokenize: cannot open %s\n", file);
      return 1;
    }
    char buf[1 << 16];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, n);
    std::fclose(fp);
  }
  const std::vector<std::int32_t> ids = tk.encode(text, parse_special);
  for (std::int32_t id : ids) std::printf("%d\n", id);
  std::fprintf(stderr, "# %zu tokens from %zu bytes\n", ids.size(), text.size());
  return 0;
}

// ---------------------------------------------------------------------------
// ppl: M5 quality gate. Reproduces llama.cpp's `llama-perplexity` (strided mode,
// tools/perplexity/perplexity.cpp perplexity_v2) so the two numbers are directly
// comparable:
//   - the corpus is tokenized once with add_special/parse_special as upstream
//     does (this vocab adds neither BOS nor EOS);
//   - the evaluation window is `ctx + stride/2` tokens per chunk (upstream
//     enlarges n_ctx by stride/2 for the strided mode);
//   - only positions [window - stride - 1, window - 1) are scored, i.e. the last
//     `stride` tokens of the window, with the preceding tokens serving as
//     context; chunk i starts at i*stride, so the scored regions tile the corpus;
//   - NLL is -log softmax(logits)[next token] with the max-subtracted softmax of
//     the reference, and PPL = exp(mean NLL) with the same +/- std error.
// Each chunk restarts the sequence (Graph::reset_state), like llama.cpp clearing
// the KV cache per chunk.
// ---------------------------------------------------------------------------
int cmd_ppl(int argc, char **argv) {
  const char *model_path = nullptr;
  const char *corpus = nullptr;
  std::uint64_t ctx_size = 512, stride = 512;
  long long chunks = 20;
  rdna4::KvType kv_k = rdna4::KvType::F16, kv_v = rdna4::KvType::F16;
  bool stats = true, verbose = false;
  const char *nll_out = nullptr;

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "rdna4-infer ppl: %s needs a value\n", what);
        return nullptr;
      }
      return argv[++i];
    };
    if (std::strcmp(a, "-m") == 0) {
      const char *v = need("-m");
      if (!v) return 1;
      model_path = v;
    } else if (std::strcmp(a, "-f") == 0) {
      const char *v = need("-f");
      if (!v) return 1;
      corpus = v;
    } else if (std::strcmp(a, "--ctx-size") == 0 || std::strcmp(a, "-c") == 0) {
      const char *v = need("--ctx-size");
      if (!v || !parse_u64(v, &ctx_size) || ctx_size == 0 ||
          ctx_size > rdna4::kMaxCtxSize) {
        return 1;
      }
    } else if (std::strcmp(a, "--stride") == 0) {
      const char *v = need("--stride");
      if (!v || !parse_u64(v, &stride) || stride == 0 || stride % 2 != 0) return 1;
    } else if (std::strcmp(a, "--chunks") == 0) {
      const char *v = need("--chunks");
      if (!v || !parse_i64(v, &chunks) || chunks < 1) return 1;
    } else if (std::strcmp(a, "--cache-type-k") == 0) {
      const char *v = need("--cache-type-k");
      if (!v || !parse_kv_type(v, &kv_k)) return 1;
    } else if (std::strcmp(a, "--cache-type-v") == 0) {
      const char *v = need("--cache-type-v");
      if (!v || !parse_kv_type(v, &kv_v)) return 1;
    } else if (std::strcmp(a, "--no-stats") == 0) {
      stats = false;
    } else if (std::strcmp(a, "-v") == 0) {
      verbose = true;
    } else if (std::strcmp(a, "--nll-out") == 0) {
      const char *v = need("--nll-out");
      if (!v) return 1;
      nll_out = v;
    } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      std::printf("usage: rdna4-infer ppl -m model.gguf -f corpus.txt [--ctx-size 512] "
                  "[--stride 512] [--chunks 20] [--cache-type-k T] [--cache-type-v T]\n");
      return 0;
    } else {
      std::fprintf(stderr, "rdna4-infer ppl: unknown argument '%s'\n", a);
      return 1;
    }
  }
  if (model_path == nullptr || corpus == nullptr) {
    std::fprintf(stderr, "rdna4-infer ppl: -m model.gguf and -f corpus.txt are required\n");
    return 1;
  }

  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0 || !rdna4::is_gfx1201(arch)) {
    std::fprintf(stderr, "rdna4-infer: requires gfx1201, found %s\n", arch);
    return 2;
  }

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(model_path, err)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(loader.meta(), cfg, err)) {
    std::fprintf(stderr, "rdna4-infer: config failed: %s\n", err.c_str());
    return 1;
  }
  rdna4::Tokenizer tk;
  if (!tk.init(loader.meta(), err)) {
    std::fprintf(stderr, "rdna4-infer: tokenizer failed: %s\n", err.c_str());
    return 1;
  }

  std::string text;
  {
    std::FILE *fp = std::fopen(corpus, "rb");
    if (fp == nullptr) {
      std::fprintf(stderr, "rdna4-infer ppl: cannot open %s\n", corpus);
      return 1;
    }
    char buf[1 << 16];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), fp)) > 0) text.append(buf, n);
    std::fclose(fp);
  }
  const double t_tok0 = now_s();
  std::vector<std::int32_t> tokens = tk.encode(text, /*parse_special=*/false);
  const double t_tok1 = now_s();
  if (tokens.size() < 2 * ctx_size) {
    std::fprintf(stderr,
                 "rdna4-infer ppl: the corpus tokenizes to %zu tokens, need >= 2*ctx = %llu\n",
                 tokens.size(), (unsigned long long)(2 * ctx_size));
    return 1;
  }
  if (tk.add_bos() || tk.add_eos()) {
    std::fprintf(stderr, "rdna4-infer ppl: this vocab adds BOS/EOS; the reference's perplexity "
                         "harness assumes it does not\n");
    return 1;
  }

  const std::size_t window = (std::size_t)ctx_size + (std::size_t)(stride / 2);
  const std::size_t n_chunk_max = (tokens.size() - window + stride - 1) / stride;
  const std::size_t n_chunk = std::min<std::size_t>((std::size_t)chunks, n_chunk_max);
  if (n_chunk == 0) {
    std::fprintf(stderr, "rdna4-infer ppl: no chunks fit (corpus %zu tokens, window %zu)\n",
                 tokens.size(), window);
    return 1;
  }

  const double t0 = now_s();
  rdna4::Graph graph(loader, cfg);
  if (!graph.init((int)window, kv_k, kv_v, err)) {
    std::fprintf(stderr, "rdna4-infer: graph init failed: %s\n", err.c_str());
    return 1;
  }
  graph.set_want_hidden(false);
  const double t_load = now_s() - t0;

  if (stats) {
    std::fprintf(stderr,
                 "rdna4-infer ppl: %s, ctx %llu window %zu stride %llu chunks %zu, kv %s/%s\n"
                 "  corpus %s: %.2f MB, %zu tokens (tokenized in %.1f s), model loaded in %.1f s\n",
                 model_path, (unsigned long long)ctx_size, window, (unsigned long long)stride,
                 n_chunk, kv_type_cstr(kv_k), kv_type_cstr(kv_v), corpus,
                 (double)text.size() / 1048576.0, tokens.size(), t_tok1 - t_tok0, t_load);
  }

  const std::size_t first_scored = window - (std::size_t)stride - 1;
  const int n_vocab = graph.n_vocab();
  std::FILE *nll_fp = nullptr;
  if (nll_out != nullptr) {
    nll_fp = std::fopen(nll_out, "w");
    if (nll_fp == nullptr) {
      std::fprintf(stderr, "rdna4-infer ppl: cannot write %s\n", nll_out);
      return 1;
    }
  }
  std::vector<float> hidden, logits;
  double nll_sum = 0.0, nll_sq = 0.0;
  std::uint64_t scored = 0;
  std::printf("chunks:");
  for (std::size_t c = 0; c < n_chunk; ++c) {
    const std::size_t start = c * (std::size_t)stride;
    if (!graph.reset_state(err)) {
      std::fprintf(stderr, "rdna4-infer ppl: reset failed: %s\n", err.c_str());
      return 1;
    }
    double chunk_nll = 0.0;
    std::uint64_t chunk_n = 0;
    // diagnostics: the tokens this engine finds most surprising in the chunk.
    // A handful of extreme values means a specific construction is mishandled;
    // a uniformly worse chunk means something structural.
    std::vector<std::pair<double, std::pair<std::int32_t, std::size_t>>> worst;
    for (std::size_t t = 0; t < window; ++t) {
      if (!graph.forward_tokens({tokens[start + t]}, (int)t, hidden, logits, err)) {
        std::fprintf(stderr, "rdna4-infer ppl: forward failed at chunk %zu token %zu: %s\n", c, t,
                     err.c_str());
        return 1;
      }
      if (t < first_scored || t + 1 >= window) continue;
      const int next = tokens[start + t + 1];
      float max_logit = logits[0];
      for (int i = 1; i < n_vocab; ++i) max_logit = std::max(max_logit, logits[i]);
      float sum_exp = 0.0f;
      for (int i = 0; i < n_vocab; ++i) sum_exp += std::exp(logits[i] - max_logit);
      if (!(sum_exp > 0.0f)) {
        std::fprintf(stderr, "rdna4-infer ppl: degenerate softmax at chunk %zu token %zu\n", c, t);
        return 1;
      }
      const float prob = std::exp(logits[next] - max_logit) / sum_exp;
      if (!(prob > 0.0f)) {
        std::fprintf(stderr, "rdna4-infer ppl: zero probability for token %d (chunk %zu)\n", next,
                     c);
        return 1;
      }
      const double nll = -std::log((double)prob);
      if (nll_fp != nullptr) {
        std::fprintf(nll_fp, "%zu %.8f %d\n", start + t + 1, nll, next);
      }
      if (worst.size() < 5) {
        worst.push_back({nll, {next, start + t + 1}});
        std::sort(worst.begin(), worst.end(),
                  [](const auto &x, const auto &y) { return x.first > y.first; });
      } else if (nll > worst.back().first) {
        worst.back() = {nll, {next, start + t + 1}};
        std::sort(worst.begin(), worst.end(),
                  [](const auto &x, const auto &y) { return x.first > y.first; });
      }
      chunk_nll += nll;
      ++chunk_n;
      nll_sum += nll;
      nll_sq += nll * nll;
      ++scored;
    }
    std::printf("[%zu]%.4f,", c + 1, std::exp(chunk_nll / (double)chunk_n));
    std::fflush(stdout);
    if (verbose) {
      std::fprintf(stderr, "  chunk %zu worst tokens:", c + 1);
      for (const auto &w : worst) {
        std::fprintf(stderr, " [%zu] %d '%s' nll %.2f;", w.second.second, w.second.first,
                     tk.token_text(w.second.first).c_str(), w.first);
      }
      std::fprintf(stderr, "\n");
    }
  }
  std::printf("\n");

  if (nll_fp != nullptr) std::fclose(nll_fp);
  const double mean_nll = nll_sum / (double)scored;
  const double ppl = std::exp(mean_nll);
  double var = nll_sq / (double)scored - mean_nll * mean_nll;
  double ppl_std = 0.0;
  if (var > 0.0) {
    var = std::sqrt(var / (double)(scored - 1));
    ppl_std = var * ppl;
  }
  const double dt = now_s() - t0 - t_load;
  std::printf("Final estimate: PPL = %.4f +/- %.5f\n", ppl, ppl_std);
  if (stats) {
    std::fprintf(stderr,
                 "  scored %llu tokens over %zu chunks in %.1f s (%.2f tok/s, window %zu, kv "
                 "%s/%s)\n",
                 (unsigned long long)scored, n_chunk, dt,
                 dt > 0 ? (double)(scored + n_chunk * (window - (std::size_t)stride)) / dt : 0.0,
                 window, kv_type_cstr(kv_k), kv_type_cstr(kv_v));
  }
  return 0;
}

// ---------------------------------------------------------------------------
// info: the M0/M1/M2 device gate and VRAM budget check (unchanged behaviour).
// ---------------------------------------------------------------------------
int cmd_info(int argc, char **argv) {
  const char *model_path = nullptr;
  std::uint64_t ctx_size = 4096;
  const char *cache_k = "f16";
  const char *cache_v = "f16";
  bool list_tensors = false;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
      model_path = argv[++i];
    } else if (std::strcmp(argv[i], "--ctx-size") == 0 && i + 1 < argc) {
      if (!parse_u64(argv[++i], &ctx_size) || ctx_size == 0 ||
          ctx_size > rdna4::kMaxCtxSize) {
        usage_info();
        return 1;
      }
    } else if (std::strcmp(argv[i], "--cache-type-k") == 0 && i + 1 < argc) {
      cache_k = argv[++i];
    } else if (std::strcmp(argv[i], "--cache-type-v") == 0 && i + 1 < argc) {
      cache_v = argv[++i];
    } else if (std::strcmp(argv[i], "--list-tensors") == 0) {
      list_tensors = true;
    } else {
      usage_info();
      return 1;
    }
  }
  rdna4::KvType kt{}, vt{};
  if (!parse_kv_type(cache_k, &kt) || !parse_kv_type(cache_v, &vt)) {
    usage_info();
    return 1;
  }

  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0) {
    std::fprintf(stderr, "rdna4-infer: no HIP device found\n");
    return 1;
  }
  std::printf("device arch: %s\nvram total: %.2f GiB\n", arch,
              static_cast<double>(total_vram) / (1u << 30));
  if (!rdna4::is_gfx1201(arch)) {
    std::fprintf(stderr, "rdna4-infer: requires gfx1201, found %s\n", arch);
    return 2;
  }

  if (model_path == nullptr) {
    std::printf("rdna4-infer: gfx1201 OK (no model given, budget check skipped)\n");
    return 0;
  }

  struct stat st;
  if (stat(model_path, &st) != 0) {
    std::fprintf(stderr, "rdna4-infer: cannot stat %s\n", model_path);
    return 1;
  }
  const auto file_bytes = static_cast<std::uint64_t>(st.st_size);

  rdna4::GgufLoader loader;
  std::string gerr;
  if (!loader.open(model_path, gerr)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", gerr.c_str());
    return 1;
  }
  const rdna4::gguf::File &gf = loader.meta();
  std::printf("gguf v%u: arch=%s tensors=%llu kv=%llu data=%.2f GiB\n", gf.version,
              rdna4::gguf::kv_str(gf, "general.architecture", "?").c_str(),
              (unsigned long long)gf.tensors.size(), (unsigned long long)gf.kv.size(),
              static_cast<double>(loader.total_bytes()) / (1u << 30));
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(gf, cfg, gerr)) {
    std::fprintf(stderr, "rdna4-infer: config failed: %s\n", gerr.c_str());
    return 1;
  }
  if (!rdna4::validate_qwen35_layout(loader, cfg, gerr)) {
    std::fprintf(stderr, "rdna4-infer: layout failed: %s\n", gerr.c_str());
    return 1;
  }
  const unsigned main_blocks = cfg.block_count - 1;  // exclude the MTP block
  const unsigned full_attn = main_blocks / cfg.full_attention_interval;
  std::printf("qwen35: %u blocks (%u full-attn, %u GDN, 1 MTP) emb=%llu ffn=%llu "
              "ctx=%llu  layout OK\n",
              cfg.block_count, full_attn, main_blocks - full_attn,
              (unsigned long long)cfg.embedding_length,
              (unsigned long long)cfg.feed_forward_length,
              (unsigned long long)cfg.context_length);
  if (list_tensors) {
    for (std::size_t i = 0; i < gf.tensors.size(); ++i) {
      const auto &t = gf.tensors[i];
      std::printf("  %-40s %-8s [", t.name.c_str(), rdna4::dtype_name(loader.dtype(i)));
      for (std::size_t d = 0; d < t.dims.size(); ++d) {
        std::printf("%s%lld", d ? ", " : "", (long long)t.dims[d]);
      }
      std::printf("]\n");
    }
    return 0;
  }
  const double gib = static_cast<double>(1u << 30);
  // One source of truth for the K + V budget (device.h), the same call serve
  // makes before loading: this used to be two hand-rolled half-constant sums.
  //
  // The weights term is the tensors the engine actually uploads, NOT the file
  // size, and the buffers term is computed from the shape instead of the old flat
  // 1 GiB. Measured for this model: file 11.214 GiB, uploaded 10.877 GiB
  // (0.327 GiB of that file is the MTP block, which the trunk never binds), and
  // the fixed buffers 157.98 MiB. The old pair overestimated by ~1.18 GiB and
  // refused contexts that fit (docs/journal-kv.md §6).
  const std::uint64_t weights = rdna4::uploaded_weight_bytes(loader, cfg.block_count - 1, false);
  // vocab for the logits buffer: the graph takes it from output.weight's dim1
  // (model.h's config carries no vocab field).
  std::uint64_t vocab = 0;
  for (const auto &t : gf.tensors) {
    if (t.name == "output.weight" && t.dims.size() > 1) vocab = t.dims[1];
  }
  const rdna4::GraphBufferShape shape = rdna4::graph_buffer_shape(cfg, vocab);
  const std::uint64_t buffers_est = rdna4::graph_buffer_bytes(shape);
  const std::uint64_t kv_bytes = rdna4::kv_cache_bytes(ctx_size, kt, vt);
  const std::uint64_t need =
      rdna4::required_bytes(weights, ctx_size, kt, vt, buffers_est);
  std::printf("model file: %.2f GiB, uploaded weights: %.2f GiB, graph buffers: %.2f MiB\n",
              file_bytes / gib, weights / gib, buffers_est / (1024.0 * 1024.0));
  std::printf("kv(ctx=%llu, k=%s/v=%s): %.2f GiB + margin: %.2f MiB = need %.2f GiB "
              "(file-based estimate was %.2f GiB)\n",
              (unsigned long long)ctx_size, cache_k, cache_v, kv_bytes / gib,
              rdna4::kOverheadBytes / (1024.0 * 1024.0), need / gib,
              (file_bytes + kv_bytes + (1u << 30)) / gib);
  if (need > total_vram) {
    std::fprintf(stderr, "rdna4-infer: insufficient VRAM (need %.2f GiB, have %.2f GiB)\n",
                 need / gib, total_vram / gib);
    return 3;
  }
  std::printf("rdna4-infer: budget OK\n");
  return 0;
}

// ---------------------------------------------------------------------------
// run: M4 generation.
// ---------------------------------------------------------------------------
int cmd_run(int argc, char **argv) {
  const char *model_path = nullptr;
  std::string prompt = "Hello";
  std::string system_msg;
  long long n_predict = 128;
  std::uint64_t ctx_size = 4096;
  rdna4::KvType kv_k = rdna4::KvType::F16, kv_v = rdna4::KvType::F16;
  rdna4::SamplerParams sp;
  bool temp_set = false, top_k_set = false, top_p_set = false, min_p_set = false;
  bool chat = false, thinking = true, verbose = false, stats = true;
  std::string effort = "xhigh";
  // ---- feat/mtp ----
  bool mtp_on = false, mtp_score = false;
  long long mtp_draft = 3;
  // ---- end feat/mtp ----

  for (int i = 1; i < argc; ++i) {
    const char *a = argv[i];
    auto need_arg = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "rdna4-infer run: %s needs a value\n", what);
        return nullptr;
      }
      return argv[++i];
    };
    if (std::strcmp(a, "-m") == 0) {
      const char *v = need_arg("-m");
      if (!v) return 1;
      model_path = v;
    } else if (std::strcmp(a, "-p") == 0) {
      const char *v = need_arg("-p");
      if (!v) return 1;
      prompt = v;
    } else if (std::strcmp(a, "--system") == 0) {
      const char *v = need_arg("--system");
      if (!v) return 1;
      system_msg = v;
    } else if (std::strcmp(a, "-n") == 0) {
      const char *v = need_arg("-n");
      if (!v || !parse_i64(v, &n_predict) || n_predict < 0) {
        usage_run();
        return 1;
      }
    } else if (std::strcmp(a, "--chat") == 0) {
      chat = true;
    } else if (std::strcmp(a, "--no-thinking") == 0) {
      thinking = false;
    } else if (std::strcmp(a, "--reasoning-effort") == 0) {
      const char *v = need_arg("--reasoning-effort");
      if (!v) return 1;
      effort = v;
    } else if (std::strcmp(a, "--temp") == 0) {
      const char *v = need_arg("--temp");
      if (!v || !parse_f32(v, &sp.temp)) {
        usage_run();
        return 1;
      }
      temp_set = true;
    } else if (std::strcmp(a, "--greedy") == 0) {
      sp.temp = 0.0f;
      temp_set = true;
    } else if (std::strcmp(a, "--top-k") == 0) {
      long long v = 0;
      const char *s = need_arg("--top-k");
      if (!s || !parse_i64(s, &v) || v < -1 || v > 1000000) {
        usage_run();
        return 1;
      }
      sp.top_k = (int)v;
      top_k_set = true;
    } else if (std::strcmp(a, "--top-p") == 0) {
      const char *v = need_arg("--top-p");
      if (!v || !parse_f32(v, &sp.top_p)) {
        usage_run();
        return 1;
      }
      top_p_set = true;
    } else if (std::strcmp(a, "--min-p") == 0) {
      const char *v = need_arg("--min-p");
      if (!v || !parse_f32(v, &sp.min_p)) {
        usage_run();
        return 1;
      }
      min_p_set = true;
    } else if (std::strcmp(a, "--repeat-penalty") == 0) {
      const char *v = need_arg("--repeat-penalty");
      if (!v || !parse_f32(v, &sp.repeat_penalty)) {
        usage_run();
        return 1;
      }
    } else if (std::strcmp(a, "--repeat-last-n") == 0) {
      long long v = 0;
      const char *s = need_arg("--repeat-last-n");
      if (!s || !parse_i64(s, &v) || v < -1) {
        usage_run();
        return 1;
      }
      sp.repeat_last_n = (int)v;
    } else if (std::strcmp(a, "--seed") == 0) {
      std::uint64_t v = 0;
      const char *s = need_arg("--seed");
      if (!s || !parse_u64(s, &v)) {
        usage_run();
        return 1;
      }
      sp.seed = v;
    } else if (std::strcmp(a, "--ctx-size") == 0) {
      const char *v = need_arg("--ctx-size");
      if (!v || !parse_u64(v, &ctx_size) || ctx_size == 0 ||
          ctx_size > rdna4::kMaxCtxSize) {
        usage_run();
        return 1;
      }
    } else if (std::strcmp(a, "--cache-type-k") == 0) {
      const char *v = need_arg("--cache-type-k");
      if (!v || !parse_kv_type(v, &kv_k)) {
        usage_run();
        return 1;
      }
    } else if (std::strcmp(a, "--cache-type-v") == 0) {
      const char *v = need_arg("--cache-type-v");
      if (!v || !parse_kv_type(v, &kv_v)) {
        usage_run();
        return 1;
      }
    // ---- feat/mtp ---- (draft head / speculative decoding)
    } else if (std::strcmp(a, "--mtp") == 0) {
      mtp_on = true;
    } else if (std::strcmp(a, "--mtp-score") == 0) {
      mtp_on = true;
      mtp_score = true;
    } else if (std::strcmp(a, "--draft") == 0) {
      const char *v = need_arg("--draft");
      if (!v || !parse_i64(v, &mtp_draft) || mtp_draft < 1 || mtp_draft > 16) {
        usage_run();
        return 1;
      }
      mtp_on = true;
    // ---- end feat/mtp ----
    } else if (std::strcmp(a, "-v") == 0) {
      verbose = true;
    } else if (std::strcmp(a, "--no-stats") == 0) {
      stats = false;
    } else if (std::strcmp(a, "-h") == 0 || std::strcmp(a, "--help") == 0) {
      usage_run();
      return 0;
    } else {
      std::fprintf(stderr, "rdna4-infer run: unknown argument '%s'\n", a);
      usage_run();
      return 1;
    }
  }
  if (model_path == nullptr) {
    std::fprintf(stderr, "rdna4-infer run: -m model.gguf is required\n");
    usage_run();
    return 1;
  }

  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0) {
    std::fprintf(stderr, "rdna4-infer: no HIP device found\n");
    return 1;
  }
  if (!rdna4::is_gfx1201(arch)) {
    std::fprintf(stderr, "rdna4-infer: requires gfx1201, found %s\n", arch);
    return 2;
  }

  struct stat st;
  if (stat(model_path, &st) != 0) {
    std::fprintf(stderr, "rdna4-infer: cannot stat %s\n", model_path);
    return 1;
  }

  rdna4::GgufLoader loader;
  std::string err;
  if (!loader.open(model_path, err)) {
    std::fprintf(stderr, "rdna4-infer: gguf load failed: %s\n", err.c_str());
    return 1;
  }
  const rdna4::gguf::File &gf = loader.meta();
  rdna4::Qwen35Config cfg;
  if (!rdna4::parse_qwen35_config(gf, cfg, err)) {
    std::fprintf(stderr, "rdna4-infer: config failed: %s\n", err.c_str());
    return 1;
  }

  // sampler defaults from the metadata unless the user overrode them (see
  // SamplerParams; the flags that were *not* given keep their literal value)
  apply_metadata_sampling(gf, &sp, !temp_set, !top_k_set, !top_p_set, !min_p_set);

  rdna4::Tokenizer tk;
  if (!tk.init(gf, err)) {
    std::fprintf(stderr, "rdna4-infer: tokenizer failed: %s\n", err.c_str());
    return 1;
  }

  // ---- prompt ------------------------------------------------------------
  std::string text;
  if (chat) {
    std::vector<rdna4::ChatMessage> msgs;
    if (!system_msg.empty()) msgs.push_back({"system", system_msg, ""});
    msgs.push_back({"user", prompt, ""});
    rdna4::ChatOptions co;
    co.enable_thinking = thinking;
    co.reasoning_effort = effort;
    if (!rdna4::chat_render(msgs, co, text, err)) {
      std::fprintf(stderr, "rdna4-infer: chat render failed: %s\n", err.c_str());
      return 1;
    }
  } else {
    text = prompt;
  }
  // parse_special: the chat template (and any user text) may contain the
  // vocabulary's control tokens, which must map to their single ids just as
  // llama.cpp's tokenizer does.
  std::vector<std::int32_t> prompt_ids = tk.encode(text, /*parse_special=*/true);
  if (prompt_ids.empty()) {
    std::fprintf(stderr, "rdna4-infer: the prompt produced no tokens\n");
    return 1;
  }
  // no BOS insert here: Tokenizer::encode() already applies add_bos_token (doing
  // it twice would double the BOS for a vocab that sets it)
  if ((std::uint64_t)prompt_ids.size() >= ctx_size) {
    std::fprintf(stderr, "rdna4-infer: prompt (%zu tokens) does not fit ctx %llu\n",
                 prompt_ids.size(), (unsigned long long)ctx_size);
    return 1;
  }
  if ((std::uint64_t)(prompt_ids.size() + (std::size_t)n_predict) > ctx_size) {
    const long long room = (long long)(ctx_size - prompt_ids.size());
    if (stats) {
      std::fprintf(stderr, "rdna4-infer: -n %lld clipped to %lld (ctx %llu, prompt %zu)\n",
                   n_predict, room, (unsigned long long)ctx_size, prompt_ids.size());
    }
    n_predict = room;
  }

  // ---- graph -------------------------------------------------------------
  const double t_load0 = now_s();
  rdna4::Graph graph(loader, cfg);
  if (!graph.init((int)ctx_size, kv_k, kv_v, err)) {
    std::fprintf(stderr, "rdna4-infer: graph init failed: %s\n", err.c_str());
    return 1;
  }
  if (graph.n_vocab() != (int)tk.n_vocab()) {
    std::fprintf(stderr, "rdna4-infer: graph vocab (%d) != tokenizer vocab (%zu)\n",
                 graph.n_vocab(), tk.n_vocab());
    return 1;
  }
  if (tk.eos_id() < 0 || tk.eos_id() >= graph.n_vocab()) {
    std::fprintf(stderr, "rdna4-infer: eos token %d outside the graph vocab (%d)\n", tk.eos_id(),
                 graph.n_vocab());
    return 1;
  }
  rdna4::Sampler sampler;
  sampler.init(sp, (std::int32_t)graph.n_vocab());
  const double t_loaded = now_s();

  if (stats) {
    std::fprintf(stderr,
                 "rdna4-infer: %s: %d layers (%d full-attn), embd %d, vocab %d, ctx %llu, kv "
                 "%s/%s, loaded in %.2f s\n",
                 model_path, graph.n_layer(), graph.n_full_attn(), graph.n_embd(), graph.n_vocab(),
                 (unsigned long long)ctx_size, kv_type_cstr(kv_k), kv_type_cstr(kv_v),
                 t_loaded - t_load0);
    std::fprintf(stderr, "rdna4-infer: prompt %zu tokens%s, temp %.2f top_k %d top_p %.2f min_p %.2f%s%s\n",
                 prompt_ids.size(), chat ? " (chat template)" : "", sp.temp, sp.top_k, sp.top_p,
                 sp.min_p, sp.repeat_penalty != 1.0f ? " repeat_penalty" : "",
                 sp.temp <= 0.0f ? " [greedy]" : "");
  }

  // ===================== feat/mtp (delimited block) ==========================
  // Draft head / speculative decoding. The plain path is untouched: this block
  // only runs when --mtp / --draft / --mtp-score was given, and it takes over
  // prefill *and* decode (see include/rdna4/mtp_gen.h for the modes and the
  // correctness contract).
  //
  // It owns the prefill on purpose: the draft block keeps its own KV cache, and
  // every one of its rows is built from a trunk h and one token, so the whole
  // prompt has to be replayed through it (llama.cpp's speculative.cpp does the
  // same: process() decodes the entire batch into the MTP context, whose `h`
  // input is the target's hidden state shifted right by one). Prefilling one
  // token at a time here costs the same as the single forward_tokens() call the
  // plain path makes — that call is already a per-token loop internally.
  if (mtp_on) {
    if (sp.temp > 0.0f) {
      std::fprintf(stderr,
                   "rdna4-infer: MTP needs greedy decoding (--temp 0/--greedy): with "
                   "temperature > 0 the draft changes the RNG stream, so the output could "
                   "not be compared against plain sampling\n");
      return 1;
    }
    if (mtp_score && sp.repeat_penalty != 1.0f) {
      std::fprintf(stderr,
                   "rdna4-infer: warning: --mtp-score compares the draft against the raw "
                   "argmax of the trunk's logits, but repeat_penalty %.3f makes the trunk "
                   "pick a penalized token; the measured rate is then a lower bound\n",
                   sp.repeat_penalty);
    }
    rdna4::MtpHead mtp;
    std::string mtop_err;
    const double t_mtp0 = now_s();
    if (!mtp.init(loader, cfg, (int)ctx_size, kv_k, kv_v, mtop_err)) {
      std::fprintf(stderr, "rdna4-infer: MTP init failed: %s\n", mtop_err.c_str());
      return 1;
    }
    rdna4::MtpShared shared;
    if (!graph.mtp_shared_weights(shared, mtop_err)) {
      std::fprintf(stderr, "rdna4-infer: MTP shared weights failed: %s\n", mtop_err.c_str());
      return 1;
    }
    mtp.set_shared(shared);
    if (stats) {
      std::fprintf(stderr,
                   "rdna4-infer: MTP: block %d loaded (%.1f MiB, eh_proj [%d, %d], concat %s), "
                   "%.2f s\n",
                   mtp.dims().block, (double)mtp.weight_bytes() / (1024.0 * 1024.0),
                   2 * mtp.dims().n_embd, mtp.dims().n_embd,
                   mtp.concat_order_e_first() ? "[e ; h]" : "[h ; e]", now_s() - t_mtp0);
    }

    std::vector<float> hidden, logits;
    std::vector<float> h_prev((std::size_t)graph.n_embd(), 0.0f);  // h at position -1
    const double t_pre0 = now_s();
    for (std::size_t i = 0; i < prompt_ids.size(); ++i) {
      if (!graph.forward_tokens({prompt_ids[i]}, (int)i, hidden, logits, mtop_err)) {
        std::fprintf(stderr, "rdna4-infer: prefill failed: %s\n", mtop_err.c_str());
        return 1;
      }
      // row `i` of the draft block's cache: token i with the trunk h of i-1
      // (zeros for i == 0, which is what llama.cpp's zero-initialised pending_h
      // row holds)
      if (!mtp.step_host(h_prev.data(), prompt_ids[i], (int)i, nullptr, nullptr, mtop_err)) {
        std::fprintf(stderr, "rdna4-infer: MTP prefill step failed: %s\n", mtop_err.c_str());
        return 1;
      }
      h_prev = hidden;
    }
    const double t_pre1 = now_s();

    rdna4::MtpGenParams gp;
    gp.n_predict = (int)n_predict;
    gp.draft = (int)mtp_draft;
    gp.score_only = mtp_score;
    gp.verbose = verbose;
    rdna4::MtpGenStats gs;
    std::string emitted_mtp;
    const double t_gen0 = now_s();
    const bool ok = rdna4::mtp_generate(
        graph, &mtp, [&](std::int32_t id) { return tk.is_eog(id); }, sampler, prompt_ids, hidden,
        logits, gp, gs, mtop_err, [&](const std::vector<std::int32_t> &ids) {
          // same streaming rule as the plain loop: re-decode the prefix and emit
          // only what is new and ends on a UTF-8 boundary
          const std::string all = tk.decode(ids, /*skip_special=*/true);
          const std::size_t keep = rdna4::utf8_keep_len(all);
          if (keep > emitted_mtp.size()) {
            std::fwrite(all.data() + emitted_mtp.size(), 1, keep - emitted_mtp.size(), stdout);
            std::fflush(stdout);
            emitted_mtp.assign(all, 0, keep);
          }
        });
    const double t_gen1 = now_s();
    if (!ok) {
      std::fprintf(stderr, "rdna4-infer: MTP decode failed: %s\n", mtop_err.c_str());
      return 1;
    }
    const std::string all_mtp = tk.decode(gs.ids, /*skip_special=*/true);
    if (all_mtp.size() > emitted_mtp.size()) {
      std::fwrite(all_mtp.data() + emitted_mtp.size(), 1, all_mtp.size() - emitted_mtp.size(),
                  stdout);
    }
    std::fflush(stdout);
    if (stats) {
      const double pre = t_pre1 - t_pre0, gen = t_gen1 - t_gen0;
      std::fprintf(stderr, "rdna4-infer: prefill %zu tokens in %.2f s (%.2f tok/s)\n",
                   prompt_ids.size(), pre, pre > 0 ? (double)prompt_ids.size() / pre : 0.0);
      if (std::getenv("RD_MTP_IDS") != nullptr) {
        std::fprintf(stderr, "generated ids:");
        for (std::int32_t id : gs.ids) std::fprintf(stderr, " %d", id);
        std::fprintf(stderr, "\n");
      }
      if (mtp_score) {
        std::fprintf(stderr,
                     "rdna4-infer: MTP score: %lld/%lld drafts matched the trunk (%.1f%%), "
                     "%d trunk steps, %d draft steps\n",
                     gs.accepted, gs.drafted, gs.acceptance() * 100.0, gs.trunk_steps,
                     gs.draft_steps);
      } else {
        std::fprintf(stderr,
                     "rdna4-infer: MTP spec: %lld/%lld drafts matched the trunk (%.1f%%), "
                     "%d trunk steps, %d draft steps, %d rounds (%.2f tokens/round)\n",
                     gs.accepted, gs.drafted, gs.acceptance() * 100.0, gs.trunk_steps,
                     gs.draft_steps, gs.rounds,
                     gs.rounds > 0 ? (double)gs.ids.size() / (double)gs.rounds : 0.0);
      }
      std::fprintf(stderr, "rdna4-infer: generated %zu tokens in %.2f s (%.2f tok/s)%s\n",
                   gs.ids.size(), gen, gen > 0 ? (double)gs.ids.size() / gen : 0.0,
                   gs.hit_eos ? " [eos]" : "");
      std::fprintf(stderr, "rdna4-infer: trunk %.2f ms/step, draft total %.2f ms\n",
                   gs.trunk_steps > 0 ? gs.trunk_ms / gs.trunk_steps : 0.0, gs.draft_ms);
    }
    return 0;
  }
  // =================== end feat/mtp (delimited block) ========================

  // ---- prefill -----------------------------------------------------------
  std::vector<float> hidden, logits;
  const double t_pre0 = now_s();
  if (!prefill_ids(graph, prompt_ids, hidden, logits, err)) {
    std::fprintf(stderr, "rdna4-infer: prefill failed: %s\n", err.c_str());
    return 1;
  }
  const double t_pre1 = now_s();


  // ---- decode ------------------------------------------------------------
  std::vector<std::int32_t> history = prompt_ids;
  std::vector<std::int32_t> generated;
  std::string emitted;  // bytes already written to stdout
  bool hit_eos = false;
  const double t_gen0 = now_s();
  for (long long step = 0; step < n_predict; ++step) {
    if (verbose) {
      const std::vector<rdna4::Candidate> cands = sampler.filter(logits.data(), history);
      std::fprintf(stderr, "step %lld: %zu candidate(s):", step, cands.size());
      for (std::size_t c = 0; c < cands.size() && c < 5; ++c) {
        std::fprintf(stderr, " %d(%.3f)", cands[c].id, cands[c].logit);
      }
      std::fprintf(stderr, "\n");
    }
    const std::int32_t id = sampler.sample(logits.data(), history);
    if (id < 0) {
      std::fprintf(stderr, "rdna4-infer: sampler produced no token (empty candidate set)\n");
      return 1;
    }
    // stop on any end-of-generation token, exactly like llama.cpp's
    // llama_vocab_is_eog (248044 <|endoftext|>, 248046 <|im_end|>, the FIM ids):
    // stopping only on EOS would run past a generation the reference ends.
    if (tk.is_eog(id)) {
      hit_eos = true;
      break;
    }
    history.push_back(id);
    generated.push_back(id);

    // Stream: re-decode the generated prefix and emit whatever is new and ends
    // on a UTF-8 boundary (a multibyte character can span two tokens).
    const std::string all = tk.decode(generated, /*skip_special=*/true);
    const std::size_t keep = rdna4::utf8_keep_len(all);
    if (keep > emitted.size()) {
      std::fwrite(all.data() + emitted.size(), 1, keep - emitted.size(), stdout);
      std::fflush(stdout);
      emitted.assign(all, 0, keep);
    }

    if ((std::size_t)history.size() >= ctx_size) {
      if (stats) std::fprintf(stderr, "rdna4-infer: context full, stopping\n");
      break;
    }
    const double t0 = now_s();
    if (!graph.forward_tokens({id}, (int)history.size() - 1, hidden, logits, err)) {
      std::fprintf(stderr, "rdna4-infer: decode failed at step %lld: %s\n", step, err.c_str());
      return 1;
    }
    if (verbose) {
      std::fprintf(stderr, "step %lld: %.1f ms\n", step, (now_s() - t0) * 1000.0);
    }
  }
  const double t_gen1 = now_s();

  // Flush whatever is left: a trailing incomplete UTF-8 sequence can never be
  // completed once the loop ends, so it is written as-is rather than dropped.
  // No newline is appended (stdout is exactly the generated text).
  const std::string all = tk.decode(generated, /*skip_special=*/true);
  if (all.size() > emitted.size()) {
    std::fwrite(all.data() + emitted.size(), 1, all.size() - emitted.size(), stdout);
  }
  std::fflush(stdout);

  if (stats) {
    const double pre = t_pre1 - t_pre0, gen = t_gen1 - t_gen0;
    std::fprintf(stderr, "rdna4-infer: prefill %zu tokens in %.2f s (%.2f tok/s)\n",
                 prompt_ids.size(), pre, pre > 0 ? (double)prompt_ids.size() / pre : 0.0);
    std::fprintf(stderr, "rdna4-infer: generated %zu tokens in %.2f s (%.2f tok/s)%s\n",
                 generated.size(), gen, gen > 0 ? (double)generated.size() / gen : 0.0,
                 hit_eos ? " [eos]" : "");
  }
  if (verbose) {
    std::fprintf(stderr, "prompt ids:");
    for (std::int32_t id : prompt_ids) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\ngenerated ids:");
    for (std::int32_t id : generated) std::fprintf(stderr, " %d", id);
    std::fprintf(stderr, "\n");
  }
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc > 1 && std::strcmp(argv[1], "run") == 0) {
    return cmd_run(argc - 1, argv + 1);
  }
  if (argc > 1 && std::strcmp(argv[1], "bench") == 0) {
    return cmd_bench(argc - 1, argv + 1);
  }
  if (argc > 1 && std::strcmp(argv[1], "tokenize") == 0) {
    return cmd_tokenize(argc - 1, argv + 1);
  }
  if (argc > 1 && std::strcmp(argv[1], "ppl") == 0) {
    return cmd_ppl(argc - 1, argv + 1);
  }
  if (argc > 1 && std::strcmp(argv[1], "info") == 0) {
    return cmd_info(argc - 1, argv + 1);
  }
  if (argc > 1 && std::strcmp(argv[1], "serve") == 0) return cmd_serve(argc - 1, argv + 1);
  return cmd_info(argc, argv);
}
