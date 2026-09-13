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
#include "rdna4/device.h"
#include "rdna4/graph.cuh"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/sampler.h"
#include "rdna4/stream.h"
#include "rdna4/tokenizer.h"

namespace {

void usage_info() {
  std::fprintf(stderr,
               "usage: rdna4-infer [info] [-m model.gguf] [--ctx-size N] [--cache-type-k TYPE] "
               "[--cache-type-v TYPE] [--list-tensors]\n  cache types: f32, f16 (default), "
               "q8_0, q4_0\n");
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
      "  --top-k N               (default 20, <= 0 disables)\n"
      "  --top-p F               (default 0.95, >= 1 disables)\n"
      "  --min-p F               (default 0.0)\n"
      "  --repeat-penalty F      (default 1.0)\n"
      "  --repeat-last-n N       penalty window (default 64, <= 0 => whole history)\n"
      "  --seed N                RNG seed (default 0)\n"
      "  --greedy                same as --temp 0\n"
      "  --ctx-size N            context (default 4096)\n"
      "  --cache-type-k TYPE     f32, f16 (default), q8_0, q4_0\n"
      "  --cache-type-v TYPE     f32, f16 (default), q8_0, q4_0\n"
      "  -v                      verbose: per-step timings and the token ids\n"
      "  --no-stats              suppress the stderr statistics\n");
}

bool parse_u64(const char *s, std::uint64_t *out) {
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

// "f32" | "f16" | "q8_0" | "q4_0" -> KvType (kv.h). The names match llama.cpp's
// --cache-type-k/-v values for the types this engine implements.
bool parse_kv_type(const char *s, rdna4::KvType *out) {
  if (std::strcmp(s, "f32") == 0) {
    *out = rdna4::KvType::F32;
  } else if (std::strcmp(s, "f16") == 0) {
    *out = rdna4::KvType::F16;
  } else if (std::strcmp(s, "q8_0") == 0) {
    *out = rdna4::KvType::Q8_0;
  } else if (std::strcmp(s, "q4_0") == 0) {
    *out = rdna4::KvType::Q4_0;
  } else {
    return false;
  }
  return true;
}

const char *kv_type_cstr(rdna4::KvType t) {
  switch (t) {
    case rdna4::KvType::F32: return "f32";
    case rdna4::KvType::F16: return "f16";
    case rdna4::KvType::Q8_0: return "q8_0";
    case rdna4::KvType::Q4_0: return "q4_0";
  }
  return "?";
}

double now_s() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
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
      if (!parse_u64(argv[++i], &ctx_size) || ctx_size == 0) {
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
  const auto kv_k = static_cast<std::uint64_t>(ctx_size * rdna4::kQwen35KvElemsPerToken *
                                               rdna4::kv_bytes_per_elem(kt) / 2);
  const auto kv_v = static_cast<std::uint64_t>(ctx_size * rdna4::kQwen35KvElemsPerToken *
                                               rdna4::kv_bytes_per_elem(vt) / 2);
  const std::uint64_t need = file_bytes + kv_k + kv_v + rdna4::kOverheadBytes;
  std::printf("model: %.2f GiB + kv(ctx=%llu, k=%s/v=%s): %.2f GiB + overhead: %.2f GiB = need %.2f GiB\n",
              file_bytes / gib, (unsigned long long)ctx_size, cache_k, cache_v,
              (kv_k + kv_v) / gib, rdna4::kOverheadBytes / gib, need / gib);
  if (need > total_vram) {
    std::fprintf(stderr, "rdna4-infer: insufficient VRAM (need %.2f GiB)\n", need / gib);
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
  bool chat = false, thinking = true, verbose = false, stats = true;
  std::string effort = "xhigh";

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
    } else if (std::strcmp(a, "--greedy") == 0) {
      sp.temp = 0.0f;
    } else if (std::strcmp(a, "--top-k") == 0) {
      long long v = 0;
      const char *s = need_arg("--top-k");
      if (!s || !parse_i64(s, &v) || v < -1 || v > 1000000) {
        usage_run();
        return 1;
      }
      sp.top_k = (int)v;
    } else if (std::strcmp(a, "--top-p") == 0) {
      const char *v = need_arg("--top-p");
      if (!v || !parse_f32(v, &sp.top_p)) {
        usage_run();
        return 1;
      }
    } else if (std::strcmp(a, "--min-p") == 0) {
      const char *v = need_arg("--min-p");
      if (!v || !parse_f32(v, &sp.min_p)) {
        usage_run();
        return 1;
      }
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
      if (!v || !parse_u64(v, &ctx_size) || ctx_size == 0) {
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
  if (sp.top_k == 0 || sp.top_p <= 0.0f || sp.min_p < 0.0f || sp.repeat_penalty <= 0.0f) {
    std::fprintf(stderr,
                 "rdna4-infer run: invalid sampler parameters (top_k must not be 0, top_p > 0, "
                 "min_p >= 0, repeat_penalty > 0)\n");
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
  if (tk.add_bos()) prompt_ids.insert(prompt_ids.begin(), tk.bos_id());
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

  // ---- prefill -----------------------------------------------------------
  std::vector<float> hidden, logits;
  const double t_pre0 = now_s();
  if (!graph.forward_tokens(prompt_ids, 0, hidden, logits, err)) {
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
    if (id == tk.eos_id()) {
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

  // flush a trailing incomplete UTF-8 sequence (it can never be completed once
  // the loop ends) and make sure stdout ends with a newline for shell use
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
  if (argc > 1 && std::strcmp(argv[1], "info") == 0) {
    return cmd_info(argc - 1, argv + 1);
  }
  return cmd_info(argc, argv);
}
