// `rdna4-infer serve` — OpenAI-compatible HTTP server (branch feat/openai-server).
//
// Scope: the subset of the OpenAI API that real harnesses use. Documented in
// docs/servidor-openai.md, which is also where the deliberate omissions are
// listed (keep-alive, multiple choices, tool calling, prefix caching, ...).
//
//   GET  /health                    200 while the model is loaded
//   GET  /v1/models                 the loaded model, object: "list"
//   GET  /v1/models/{id}            that one model (404 for another id)
//   POST /v1/chat/completions       messages -> assistant message (SSE if stream)
//   POST /v1/completions            legacy prompt -> text (SSE if stream)
//
// Where the code lives and why:
//   src/server/http.cpp  — CPU-only HTTP/1.1 layer (sockets, router, errors)
//   src/server/json.cpp  — CPU-only JSON parser/serializer
//   src/server/serve.hip — this file: the HIP translation unit. It owns the
//                          model (GgufLoader + Graph + Tokenizer) and the
//                          generation loop, exactly the structure of cmd_run in
//                          src/main.hip, and plugs it into the CPU router
//                          through std::function callbacks. Keeping the sockets
//                          and the JSON out of the HIP TU means they can be unit
//                          tested on CPU (tests/check_server_http.cpp) and that
//                          a change to the kernels never recompiles them.
//
// Concurrency: one request at a time, single-threaded. The graph owns the KV
// cache and the recurrent (GDN) state, and there is exactly one of each, so a
// second concurrent request could not be served without a scheduler the engine
// does not have. Connections are accepted one by one and the SSE body is written
// straight from the generation loop.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <sys/stat.h>
#include <unordered_map>
#include <vector>

#include "rdna4/chat.h"
#include "rdna4/cli_commands.h"
#include "rdna4/device.h"
#include "rdna4/graph.cuh"
#include "rdna4/kv.h"
#include "rdna4/loader.h"
#include "rdna4/model.h"
#include "rdna4/sampler.h"
#include "rdna4/stream.h"
#include "rdna4/tokenizer.h"
#include "rdna4/server.h"
#include "rdna4/server_json.h"

namespace {

using rdna4::server::Headers;
using rdna4::server::Request;
using rdna4::server::Responder;
using rdna4::server::Router;
using rdna4::server::json::Value;

// The chat_template_kwargs accepted by the template (chat.cpp): with thinking
// enabled the template raises for anything else, so the server rejects it early
// with a 400 instead of failing inside the renderer.
constexpr const char *kEfforts[] = {"xhigh", "high", "medium", "low"};


// ---------------------------------------------------------------------------
// CLI (small local copies of main.hip's helpers: main.hip is shared with two
// other agents and the contract for this branch is that it only gains the two
// lines of dispatch, so the ~40 lines of parsing are duplicated instead of
// refactored).
// ---------------------------------------------------------------------------
bool parse_u64(const char *s, std::uint64_t *out) {
  if (s[0] == '-') return false;
  char *end = nullptr;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0') return false;
  *out = static_cast<std::uint64_t>(v);
  return true;
}

bool parse_i64(const char *s, long long *out) {
  char *end = nullptr;
  const long long v = std::strtoll(s, &end, 10);
  if (end == s || *end != '\0') return false;
  *out = v;
  return true;
}

// One table for the CLI, the budget and the graph: rdna4::kv_type_parse (kv.h).
// There used to be three copies of this list, and a type added to two of them
// would have been accepted by the CLI and then rejected by the dispatcher.
bool parse_kv_type(const char *s, rdna4::KvType *out) {
  return rdna4::kv_type_parse(s, out) == nullptr;
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

void usage_serve() {
  std::fprintf(
      stderr,
      "usage: rdna4-infer serve -m model.gguf [options]\n"
      "  -m PATH                  GGUF model (required)\n"
      "  --host HOST              bind address (default 127.0.0.1; 0.0.0.0 for the LAN)\n"
      "  --port N                 TCP port (default 8080; 0 = pick a free one)\n"
      "  --ctx-size N             context (default 8192)\n"
      "  --cache-type-k TYPE      f32, f16 (default), q8_0, q4_0, q5_0, q4_1\n"
      "  --cache-type-v TYPE      f32, f16 (default), q8_0, q4_0, q5_0, q4_1\n"
      "  --n-predict-default N    max_tokens when the request omits it (default 512; -1 = until\n"
      "                           the context is full)\n"
      "  --model-name NAME        id reported by /v1/models (default: the file name without .gguf)\n"
      "  -v, --verbose            per-request log: prompt/generated token ids and timings\n"
      "  -h, --help               this text\n"
      "\n"
      "Sampling defaults (temperature, top_k, top_p, min_p) come from the GGUF's\n"
      "general.sampling.* metadata, as in `run`; a request field always wins.\n"
      "The server handles one request at a time and closes each connection after it.\n");
}

struct ServeArgs {
  const char *model_path = nullptr;
  std::string model_name;  // empty => derived from the file name
  std::string host = "127.0.0.1";
  int port = 8080;
  std::uint64_t ctx_size = 8192;
  rdna4::KvType kv_k = rdna4::KvType::F16, kv_v = rdna4::KvType::F16;
  long long n_predict_default = 512;  // < 0 => as many as the context allows
  bool verbose = false;
};

int parse_serve_args(int argc, char **argv, ServeArgs *a) {
  for (int i = 1; i < argc; ++i) {
    const char *v = argv[i];
    auto need = [&](const char *what) -> const char * {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "rdna4-infer serve: %s needs a value\n", what);
        return nullptr;
      }
      return argv[++i];
    };
    if (std::strcmp(v, "-m") == 0) {
      const char *s = need("-m");
      if (s == nullptr) return 1;
      a->model_path = s;
    } else if (std::strcmp(v, "--host") == 0) {
      const char *s = need("--host");
      if (s == nullptr) return 1;
      a->host = s;
    } else if (std::strcmp(v, "--port") == 0) {
      long long p = 0;
      const char *s = need("--port");
      if (s == nullptr || !parse_i64(s, &p) || p < 0 || p > 65535) {
        std::fprintf(stderr, "rdna4-infer serve: --port must be 0..65535\n");
        return 1;
      }
      a->port = static_cast<int>(p);
    } else if (std::strcmp(v, "--ctx-size") == 0) {
      const char *s = need("--ctx-size");
      // Upper bound as well: ctx_size is carried as uint64_t but the graph takes
      // an int, so an unchecked huge value would be cast to a negative/odd
      // context before the VRAM budget ever saw it (review finding M5).
      if (s == nullptr || !parse_u64(s, &a->ctx_size) || a->ctx_size == 0 ||
          a->ctx_size > rdna4::kMaxCtxSize) {
        std::fprintf(stderr, "rdna4-infer serve: --ctx-size must be in 1..%llu\n",
                     (unsigned long long)rdna4::kMaxCtxSize);
        return 1;
      }
    } else if (std::strcmp(v, "--cache-type-k") == 0) {
      const char *s = need("--cache-type-k");
      if (s == nullptr || !parse_kv_type(s, &a->kv_k)) {
        std::fprintf(stderr, "rdna4-infer serve: --cache-type-k must be f32, f16, q8_0, q4_0, q5_0 or q4_1\n");
        return 1;
      }
    } else if (std::strcmp(v, "--cache-type-v") == 0) {
      const char *s = need("--cache-type-v");
      if (s == nullptr || !parse_kv_type(s, &a->kv_v)) {
        std::fprintf(stderr, "rdna4-infer serve: --cache-type-v must be f32, f16, q8_0, q4_0, q5_0 or q4_1\n");
        return 1;
      }
    } else if (std::strcmp(v, "--n-predict-default") == 0) {
      const char *s = need("--n-predict-default");
      if (s == nullptr || !parse_i64(s, &a->n_predict_default) || a->n_predict_default == 0 ||
          a->n_predict_default < -1) {
        std::fprintf(stderr, "rdna4-infer serve: --n-predict-default must be > 0 or -1\n");
        return 1;
      }
    } else if (std::strcmp(v, "--model-name") == 0) {
      const char *s = need("--model-name");
      if (s == nullptr || s[0] == '\0') return 1;
      a->model_name = s;
    } else if (std::strcmp(v, "-v") == 0 || std::strcmp(v, "--verbose") == 0) {
      a->verbose = true;
    } else if (std::strcmp(v, "-h") == 0 || std::strcmp(v, "--help") == 0) {
      usage_serve();
      return 2;  // caller turns this into exit 0
    } else {
      std::fprintf(stderr, "rdna4-infer serve: unknown argument '%s'\n", v);
      usage_serve();
      return 1;
    }
  }
  if (a->model_path == nullptr) {
    std::fprintf(stderr, "rdna4-infer serve: -m model.gguf is required\n");
    usage_serve();
    return 1;
  }
  return 0;
}

// "…/Qwen3.8-27B-UD-IQ3_S.gguf" -> "Qwen3.8-27B-UD-IQ3_S"
std::string default_model_name(const char *path) {
  std::string s = path;
  const std::size_t slash = s.find_last_of('/');
  if (slash != std::string::npos) s = s.substr(slash + 1);
  if (s.size() > 5 && s.compare(s.size() - 5, 5, ".gguf") == 0) s.resize(s.size() - 5);
  return s;
}

double now_s() {
  using clock = std::chrono::steady_clock;
  static const clock::time_point t0 = clock::now();
  return std::chrono::duration<double>(clock::now() - t0).count();
}

std::string make_completion_id(const char *prefix) {
  static std::mt19937_64 rng(static_cast<std::uint64_t>(std::time(nullptr)) ^ 0x9e3779b97f4a7c15ull);
  static const char *kHex = "0123456789abcdef";
  std::string s = prefix;
  for (int i = 0; i < 24; ++i) s.push_back(kHex[rng() & 0xF]);
  return s;
}

// ---------------------------------------------------------------------------
// Request/response plumbing.
// ---------------------------------------------------------------------------
struct ApiError {
  int status = 400;
  std::string message;
  std::string type = "invalid_request_error";
  std::string code;
  std::string param;
};

void send_api_error(Responder &out, const ApiError &e) {
  std::string msg = e.message;
  if (!e.param.empty()) msg += " (param: " + e.param + ")";
  rdna4::server::send_error(out, e.status, msg, e.type, e.code);
}

bool parse_json_body(const Request &req, Value &out, ApiError *err) {
  if (req.body.empty()) {
    *err = {400, "the request body is empty: send a JSON object", "invalid_request_error",
            "empty_body", ""};
    return false;
  }
  std::string parse_err;
  if (!rdna4::server::json::parse(req.body, out, parse_err)) {
    *err = {400, "invalid JSON in the request body: " + parse_err, "invalid_request_error",
            "invalid_json", ""};
    return false;
  }
  if (!out.is_object()) {
    *err = {400, "the request body must be a JSON object", "invalid_request_error", "invalid_body",
            ""};
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Engine: model + graph + one generation at a time.
// ---------------------------------------------------------------------------
struct GenRequest {
  bool chat = true;
  std::vector<rdna4::ChatMessage> messages;  // chat
  std::string prompt;                        // completions
  rdna4::SamplerParams sp;                   // temp/top_k/top_p/min_p/repeat/seed
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  long long max_tokens = 0;  // already clipped to the context by the caller
  std::vector<std::string> stops;
  bool ignore_eos = false;
  bool enable_thinking = true;
  std::string reasoning_effort = "xhigh";
};

struct GenResult {
  std::string text;
  std::vector<std::int32_t> prompt_ids, generated_ids;
  // "length" is the default because the generation loop can only end early by
  // setting one of the other reasons (EOG, stop string, context full, client
  // gone, Ctrl-C); running out of max_tokens leaves this value.
  std::string finish_reason = "length";  // stop | length | cancelled | client_disconnected
  std::string matched_stop;
  double prefill_s = 0.0, decode_s = 0.0;
  bool client_gone = false;
  bool hit_eog = false;
};

// Returns false when the client is gone (the caller stops generating).
using DeltaFn = std::function<bool(const std::string &)>;

class Engine {
 public:
  bool init(const ServeArgs &a, std::string &err);
  int ctx_size() const { return static_cast<int>(ctx_size_); }
  long long n_predict_default() const { return n_predict_default_; }
  const rdna4::SamplerParams &sampling_defaults() const { return defaults_; }
  const std::string &model_id() const { return model_id_; }
  int n_vocab() const { return graph_ == nullptr ? 0 : graph_->n_vocab(); }

  // Messages -> prompt tokens (chat_render + Tokenizer::encode, exactly as
  // cmd_run does it).
  bool build_prompt(const GenRequest &r, std::vector<std::int32_t> &ids, std::string &err) const;

  bool generate(const std::vector<std::int32_t> &prompt_ids, const GenRequest &r,
                const DeltaFn &on_delta, GenResult &res, std::string &err);

  // OpenAI presence/frequency penalties are applied to the raw logits before the
  // engine's own stages (llama.cpp applies its penalties first too), so they
  // stay independent of the sampler's repeat window.
  void apply_openai_penalties(float *logits, const std::vector<std::int32_t> &history) const;

 private:
  rdna4::GgufLoader loader_;
  rdna4::Qwen35Config cfg_;
  rdna4::Tokenizer tk_;
  std::unique_ptr<rdna4::Graph> graph_;
  rdna4::Sampler sampler_;
  rdna4::SamplerParams defaults_;
  std::string model_id_;
  std::uint64_t ctx_size_ = 8192;
  long long n_predict_default_ = 512;
  float presence_penalty_ = 0.0f;   // per request, used by the penalty pass
  float frequency_penalty_ = 0.0f;  // "
  mutable std::vector<std::int32_t> penalty_seen_;   // n_vocab counters
  mutable std::vector<std::int32_t> penalty_touch_;  // ids touched last step
};

bool Engine::init(const ServeArgs &a, std::string &err) {
  char arch[64] = {};
  std::size_t total_vram = 0;
  if (rdna4::query_device(arch, sizeof(arch), &total_vram) != 0) {
    err = "no HIP device found";
    return false;
  }
  if (!rdna4::is_gfx1201(arch)) {
    err = std::string("requires gfx1201, found ") + arch;
    err += "\n";  // caller maps this to exit code 2 by looking at the prefix
    return false;
  }
  struct stat st;
  if (::stat(a.model_path, &st) != 0) {
    err = std::string("cannot stat ") + a.model_path;
    return false;
  }
  // Budget check before the (slow) load, like `info` does: a context that does
  // not fit must fail with a message, not with hipMalloc. The K and V types are
  // counted separately (device.h) — using kv_k for both was exactly right only
  // while they were equal, and under-counted a K/V mix until the load died in
  // hipMalloc (review finding M4).
  const std::uint64_t kv_bytes =
      rdna4::kv_cache_bytes(a.ctx_size, a.kv_k, a.kv_v);
  // The budget needs the model's shape, so the header has to be read before the
  // check (it is a few MiB of metadata, not the weights). Counting the file size
  // and a flat 1 GiB of overhead instead overestimated by ~1.18 GiB and refused
  // contexts that fit -- see the note in include/rdna4/device.h and
  // docs/journal-kv.md §6. Same computation as `info` in src/main.hip.
  if (!loader_.open(a.model_path, err)) return false;
  if (!rdna4::parse_qwen35_config(loader_.meta(), cfg_, err)) return false;
  const std::uint64_t weights =
      rdna4::uploaded_weight_bytes(loader_, cfg_.block_count - 1, false);
  std::uint64_t vocab = 0;
  for (const auto &t : loader_.meta().tensors) {
    if (t.name == "output.weight" && t.dims.size() > 1) vocab = t.dims[1];
  }
  const std::uint64_t buffers_est =
      rdna4::graph_buffer_bytes(rdna4::graph_buffer_shape(cfg_, vocab));
  const std::uint64_t need =
      rdna4::required_bytes(weights, a.ctx_size, a.kv_k, a.kv_v, buffers_est);
  if (need > total_vram) {
    char buf[320];
    std::snprintf(buf, sizeof(buf),
                  "insufficient VRAM: weights %.2f GiB + graph buffers %.2f MiB + "
                  "kv(ctx=%llu, %s/%s) %.2f GiB + margin %.2f MiB = %.2f GiB of %.2f GiB",
                  (double)weights / (1u << 30), (double)buffers_est / (1024.0 * 1024.0),
                  (unsigned long long)a.ctx_size, kv_type_cstr(a.kv_k), kv_type_cstr(a.kv_v),
                  (double)kv_bytes / (1u << 30), (double)rdna4::kOverheadBytes / (1024.0 * 1024.0),
                  (double)need / (1u << 30), (double)total_vram / (1u << 30));
    err = buf;
    return false;
  }

  if (!tk_.init(loader_.meta(), err)) return false;
  if (!tk_.init(loader_.meta(), err)) return false;

  graph_.reset(new rdna4::Graph(loader_, cfg_));
  if (!graph_->init(static_cast<int>(a.ctx_size), a.kv_k, a.kv_v, err)) return false;
  graph_->set_want_hidden(false);  // the server never reads `hidden`
  if (graph_->n_vocab() != static_cast<int>(tk_.n_vocab())) {
    err = "graph vocab != tokenizer vocab";
    return false;
  }
  ctx_size_ = a.ctx_size;
  n_predict_default_ = a.n_predict_default;
  model_id_ = a.model_name.empty() ? default_model_name(a.model_path) : a.model_name;
  penalty_seen_.assign(static_cast<std::size_t>(graph_->n_vocab()), 0);
  penalty_touch_.reserve(1024);

  // Sampling defaults from general.sampling.* (same policy as cmd_run: the
  // metadata is the starting point, a request field overrides it). cmd_run reads
  // these per flag; here they are read once at startup.
  defaults_ = rdna4::SamplerParams();
  const rdna4::gguf::File &gf = loader_.meta();
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
  if (num("general.sampling.top_k", &d) && d > 0) defaults_.top_k = (int)d;
  if (num("general.sampling.top_p", &d) && d > 0.0 && d <= 1.0) defaults_.top_p = (float)d;
  if (num("general.sampling.temp", &d) && d >= 0.0) defaults_.temp = (float)d;
  if (num("general.sampling.min_p", &d) && d >= 0.0) defaults_.min_p = (float)d;
  return true;
}

bool Engine::build_prompt(const GenRequest &r, std::vector<std::int32_t> &ids,
                          std::string &err) const {
  std::string text;
  if (r.chat) {
    rdna4::ChatOptions co;
    co.add_generation_prompt = true;
    co.enable_thinking = r.enable_thinking;
    co.reasoning_effort = r.reasoning_effort;
    if (!rdna4::chat_render(r.messages, co, text, err)) return false;
  } else {
    text = r.prompt;
  }
  // parse_special=true: same as cmd_run — the rendered template contains the
  // vocabulary's control tokens and each must map to its own id.
  ids = tk_.encode(text, /*parse_special=*/true);
  if (ids.empty()) {
    err = "the request produced no tokens";
    return false;
  }
  return true;
}

void Engine::apply_openai_penalties(float *logits, const std::vector<std::int32_t> &history) const {
  if (presence_penalty_ == 0.0f && frequency_penalty_ == 0.0f) return;
  penalty_touch_.clear();
  for (const std::int32_t id : history) {
    if (id < 0 || static_cast<std::size_t>(id) >= penalty_seen_.size()) continue;
    std::int32_t &c = penalty_seen_[static_cast<std::size_t>(id)];
    if (c == 0) penalty_touch_.push_back(id);
    ++c;
  }
  for (const std::int32_t id : penalty_touch_) {
    const std::int32_t c = penalty_seen_[static_cast<std::size_t>(id)];
    const float pen = presence_penalty_ + frequency_penalty_ * static_cast<float>(c);
    logits[id] -= pen;
    penalty_seen_[static_cast<std::size_t>(id)] = 0;
  }
}

bool Engine::generate(const std::vector<std::int32_t> &prompt_ids, const GenRequest &r,
                      const DeltaFn &on_delta, GenResult &res, std::string &err) {
  res.prompt_ids = prompt_ids;
  presence_penalty_ = r.presence_penalty;
  frequency_penalty_ = r.frequency_penalty;

  // A fresh sequence: zeroes the recurrent (GDN) state. The KV cache needs no
  // clearing (attention is causal and every row is rewritten as the new
  // sequence advances from position 0) — see Graph::reset_state.
  if (!graph_->reset_state(err)) return false;

  sampler_.init(r.sp, graph_->n_vocab());

  std::vector<float> hidden, logits;
  const double t0 = now_s();
  if (!graph_->forward_tokens(prompt_ids, 0, hidden, logits, err)) return false;
  res.prefill_s = now_s() - t0;
  if (logits.size() != static_cast<std::size_t>(graph_->n_vocab())) {
    err = "the graph returned an unexpected number of logits";
    return false;
  }

  // Stop strings: bytes are held back so a stop sequence split across two
  // tokens is never emitted (and nothing has to be retracted).
  std::size_t max_stop = 0;
  for (const std::string &s : r.stops) max_stop = std::max(max_stop, s.size());
  const std::size_t hold = max_stop > 0 ? max_stop - 1 : 0;

  std::vector<std::int32_t> history = prompt_ids;
  std::string all;           // tk.decode(generated) so far
  std::size_t emitted = 0;   // bytes already handed to `on_delta`
  std::size_t stop_at = std::string::npos;
  std::size_t pos = prompt_ids.size();
  const std::size_t ctx = ctx_size_;
  const double t_dec0 = now_s();

  auto flush_upto = [&](std::size_t keep) -> bool {
    if (keep <= emitted) return true;
    const bool ok = on_delta ? on_delta(all.substr(emitted, keep - emitted)) : true;
    emitted = keep;
    return ok;
  };

  for (long long step = 0; step < r.max_tokens; ++step) {
    if (rdna4::server::stop_requested()) {
      res.finish_reason = "cancelled";
      break;
    }
    apply_openai_penalties(logits.data(), history);
    const std::int32_t id = sampler_.sample(logits.data(), history);
    if (id < 0) {
      err = "the sampler produced no token (empty candidate set)";
      return false;
    }
    if (!r.ignore_eos && tk_.is_eog(id)) {
      res.hit_eog = true;
      res.finish_reason = "stop";
      break;
    }
    history.push_back(id);
    res.generated_ids.push_back(id);

    // Re-decode the whole generated prefix and emit only what is new. Holding a
    // trailing incomplete UTF-8 sequence (utf8_keep_len) is what keeps a
    // multibyte character split across two tokens from being sent half-formed —
    // the same guard cmd_run uses for stdout.
    all = tk_.decode(res.generated_ids, /*skip_special=*/true);

    std::size_t hit = std::string::npos;
    std::string hit_str;
    for (const std::string &s : r.stops) {
      if (s.empty()) continue;
      const std::size_t p = all.find(s);
      if (p != std::string::npos && (hit == std::string::npos || p < hit)) {
        hit = p;
        hit_str = s;
      }
    }
    std::size_t keep = 0;
    if (hit != std::string::npos) {
      // The stop sequence is dropped from the output, like every OpenAI server.
      keep = hit;
      stop_at = hit;
      res.matched_stop = hit_str;
      res.finish_reason = "stop";
    } else {
      const std::size_t limit = all.size() > hold ? all.size() - hold : 0;
      keep = rdna4::utf8_keep_len(all.substr(0, limit));
    }
    if (!flush_upto(keep)) {
      res.client_gone = true;
      res.finish_reason = "client_disconnected";
      break;
    }
    if (hit != std::string::npos) break;

    if (pos >= ctx) {
      res.finish_reason = "length";
      break;
    }
    if (!graph_->forward_tokens({id}, static_cast<int>(pos), hidden, logits, err)) return false;
    ++pos;
  }
  res.decode_s = now_s() - t_dec0;

  if (stop_at != std::string::npos) {
    all.resize(stop_at);
  } else if (!res.client_gone) {
    // Nothing stopped the text: hand over the rest. Unlike cmd_run — which
    // flushes a trailing incomplete UTF-8 sequence to stdout as-is — a server
    // must not put invalid UTF-8 in a JSON body, so those 0..3 bytes are
    // dropped and reported. cmd_run's comment says the same bytes can never be
    // completed once the loop ends.
    const std::size_t keep = rdna4::utf8_keep_len(all);
    if (keep < all.size()) {
      std::fprintf(stderr,
                   "[serve] note: dropped %zu trailing byte(s) of an incomplete UTF-8 sequence\n",
                   all.size() - keep);
      all.resize(keep);
    }
    if (!flush_upto(all.size())) {
      res.client_gone = true;
      res.finish_reason = "client_disconnected";
    }
  }
  res.text = all;
  return true;
}

// ---------------------------------------------------------------------------
// The OpenAI surface.
// ---------------------------------------------------------------------------
class ServeApp {
 public:
  ServeApp(Engine *engine, const ServeArgs &a) : engine_(engine), args_(a) {}

  void register_routes(Router *router) {
    using rdna4::server::Request;
    router->add("GET", "/health", [this](const Request &q, Responder &o) { health(q, o); });
    router->add("GET", "/v1/models", [this](const Request &q, Responder &o) { models(q, o); });
    router->add_prefix("GET", "/v1/models/",
                       [this](const Request &q, Responder &o) { model_by_id(q, o); });
    router->add("POST", "/v1/chat/completions",
                [this](const Request &q, Responder &o) { chat_completions(q, o); });
    router->add("POST", "/v1/completions",
                [this](const Request &q, Responder &o) { completions(q, o); });
  }

 private:
  // ---- GET /health ------------------------------------------------------
  void health(const Request &, Responder &out) {
    Value root = Value::object();
    root.set("status", Value::string("ok"));
    root.set("model", Value::string(engine_->model_id()));
    root.set("ctx_size", Value::integer(engine_->ctx_size()));
    root.set("object", Value::string("health"));
    out.send(200, "application/json", root.dump(), Headers{});
  }

  // ---- GET /v1/models ---------------------------------------------------
  Value model_json() const {
    Value m = Value::object();
    m.set("id", Value::string(engine_->model_id()));
    m.set("object", Value::string("model"));
    m.set("created", Value::integer(static_cast<long long>(std::time(nullptr))));
    m.set("owned_by", Value::string("rdna4-infer"));
    return m;
  }

  void models(const Request &, Responder &out) {
    Value data = Value::array();
    data.push(model_json());
    Value root = Value::object();
    root.set("object", Value::string("list"));
    root.set("data", std::move(data));
    out.send(200, "application/json", root.dump(), Headers{});
  }

  void model_by_id(const Request &req, Responder &out) {
    const std::string id = req.path.substr(std::strlen("/v1/models/"));
    if (id != engine_->model_id()) {
      ApiError e{404, "model '" + id + "' not found (this server serves '" +
                          engine_->model_id() + "')",
                 "invalid_request_error", "model_not_found", ""};
      send_api_error(out, e);
      return;
    }
    out.send(200, "application/json", model_json().dump(), Headers{});
  }

  // ---- request parsing --------------------------------------------------
  // Every numeric field below is validated on *both* ends and for finiteness.
  // get_number() returns the raw strtod() result, so a request can deliver inf
  // or NaN: `1e400` used to be accepted as `temperature = inf` (every logit
  // becomes logit/inf = 0, so the softmax is uniform and the answer is
  // deterministic garbage) and as `seed = inf`, where the cast to uint64_t is
  // undefined behaviour. NaN slips through one-sided comparisons because every
  // comparison with NaN is false (review finding H1).
  bool parse_sampling(const Value &body, GenRequest *gr, ApiError *e) const {
    *gr = GenRequest();
    gr->sp = engine_->sampling_defaults();
    double d = 0.0;
    long long i = 0;

    if (body.get_number("temperature", &d)) {
      if (!std::isfinite(d) || d < 0.0 || d > 1e6) {
        *e = {400, "temperature must be a finite number in [0, 1e6]",
              "invalid_request_error", "invalid_temperature", "temperature"};
        return false;
      }
      gr->sp.temp = static_cast<float>(d);
    }
    if (body.get_number("top_p", &d)) {
      if (!std::isfinite(d) || d > 1.0) {
        *e = {400, "top_p must be a finite number <= 1", "invalid_request_error", "invalid_top_p",
              "top_p"};
        return false;
      }
      // <= 0 means "off" for several harnesses; the engine spells that as >= 1.
      gr->sp.top_p = d <= 0.0 ? 1.0f : static_cast<float>(d);
    }
    if (body.get_int("top_k", &i)) {
      if (i < -1) {
        *e = {400, "top_k must be >= -1 (0 disables it)", "invalid_request_error", "invalid_top_k",
              "top_k"};
        return false;
      }
      gr->sp.top_k = i <= 0 ? 0 : static_cast<int>(i);
    }
    if (body.get_number("min_p", &d)) {
      if (!std::isfinite(d) || d < 0.0 || d > 1e6) {
        *e = {400, "min_p must be a finite number in [0, 1e6]", "invalid_request_error",
              "invalid_min_p", "min_p"};
        return false;
      }
      gr->sp.min_p = static_cast<float>(d);
    }
    // llama.cpp's spelling first, then the OpenAI-ish one as an alias.
    const char *rp_key = body.has("repeat_penalty") ? "repeat_penalty" : "repetition_penalty";
    if (body.get_number(rp_key, &d)) {
      if (!std::isfinite(d) || d <= 0.0 || d > 1e6) {
        *e = {400, std::string(rp_key) + " must be a finite number in (0, 1e6]",
              "invalid_request_error", "invalid_repeat_penalty", rp_key};
        return false;
      }
      gr->sp.repeat_penalty = static_cast<float>(d);
    }
    if (body.get_int("repeat_last_n", &i)) {
      if (i < -1) {
        *e = {400, "repeat_last_n must be >= -1", "invalid_request_error", "invalid_repeat_last_n",
              "repeat_last_n"};
        return false;
      }
      gr->sp.repeat_last_n = static_cast<int>(i);
    }
    if (body.get_number("presence_penalty", &d)) {
      if (!std::isfinite(d) || d < -2.0 || d > 2.0) {
        *e = {400, "presence_penalty must be a finite number in [-2, 2]", "invalid_request_error",
              "invalid_presence_penalty", "presence_penalty"};
        return false;
      }
      gr->presence_penalty = static_cast<float>(d);
    }
    if (body.get_number("frequency_penalty", &d)) {
      if (!std::isfinite(d) || d < -2.0 || d > 2.0) {
        *e = {400, "frequency_penalty must be a finite number in [-2, 2]", "invalid_request_error",
              "invalid_frequency_penalty", "frequency_penalty"};
        return false;
      }
      gr->frequency_penalty = static_cast<float>(d);
    }
    if (body.get_number("seed", &d)) {
      // 1e15 < 2^53, so the value is exact in a double and the cast is defined.
      if (!std::isfinite(d) || d < 0.0 || d > 1e15) {
        *e = {400, "seed must be a finite number in [0, 1e15]", "invalid_request_error",
              "invalid_seed", "seed"};
        return false;
      }
      gr->sp.seed = static_cast<std::uint64_t>(d);
    }
    // max_tokens under its three common names; 0/absent means "the server's
    // --n-predict-default".
    i = -1;
    if (!body.get_int("max_tokens", &i) && !body.get_int("max_completion_tokens", &i)) {
      body.get_int("n_predict", &i);
    }
    if (i == 0 || i < -1) {
      *e = {400, "max_tokens must be > 0 (omit it to use the server default)",
            "invalid_request_error", "invalid_max_tokens", "max_tokens"};
      return false;
    }
    gr->max_tokens = i < 0 ? engine_->n_predict_default() : i;
    body.get_bool("ignore_eos", &gr->ignore_eos);

    long long n = 1;
    if (body.get_int("n", &n) && n != 1) {
      *e = {400, "n > 1 (multiple choices) is not supported: the engine generates one sequence "
                 "at a time",
            "invalid_request_error", "unsupported_n", "n"};
      return false;
    }
    long long best_of = 1;
    if (body.get_int("best_of", &best_of) && best_of != 1) {
      *e = {400, "best_of > 1 is not supported", "invalid_request_error", "unsupported_best_of",
            "best_of"};
      return false;
    }

    // stop: a string or an array of strings.
    const Value *stop = body.member("stop");
    if (stop != nullptr && !stop->is_null()) {
      if (stop->is_string()) {
        if (!stop->as_string().empty()) gr->stops.push_back(stop->as_string());
      } else if (stop->is_array()) {
        for (std::size_t k = 0; k < stop->size(); ++k) {
          const Value *s = stop->at(k);
          if (s != nullptr && s->is_string() && !s->as_string().empty()) {
            gr->stops.push_back(s->as_string());
          }
        }
      } else {
        *e = {400, "stop must be a string or an array of strings", "invalid_request_error",
              "invalid_stop", "stop"};
        return false;
      }
      if (gr->stops.size() > 16) {
        *e = {400, "at most 16 stop sequences are supported", "invalid_request_error",
              "invalid_stop", "stop"};
        return false;
      }
    }

    // Thinking controls: llama.cpp's chat_template_kwargs plus the
    // reasoning_effort shorthand ("none" turns thinking off).
    const Value *ctk = body.member("chat_template_kwargs");
    if (ctk != nullptr && ctk->is_object()) {
      bool think = true;
      if (ctk->get_bool("enable_thinking", &think)) gr->enable_thinking = think;
      std::string effort;
      if (ctk->get_string("reasoning_effort", &effort)) gr->reasoning_effort = effort;
    }
    std::string effort;
    if (body.get_string("reasoning_effort", &effort)) {
      if (effort == "none") {
        gr->enable_thinking = false;
      } else {
        gr->reasoning_effort = effort;
      }
    }
    if (gr->enable_thinking) {
      bool known = false;
      for (const char *k : kEfforts) known = known || gr->reasoning_effort == k;
      if (!known) {
        *e = {400, "reasoning_effort must be one of xhigh, high, medium, low or none",
              "invalid_request_error", "invalid_reasoning_effort", "reasoning_effort"};
        return false;
      }
    }

    if (args_.verbose) {
      const Value *ignored[] = {body.member("tools"), body.member("functions"),
                                body.member("logit_bias"), body.member("response_format")};
      const char *names[] = {"tools", "functions", "logit_bias", "response_format"};
      for (int k = 0; k < 4; ++k) {
        if (ignored[k] != nullptr && !ignored[k]->is_null()) {
          std::fprintf(stderr, "[serve] note: '%s' is accepted but ignored (not implemented)\n",
                       names[k]);
        }
      }
    }
    return true;
  }

  bool parse_common(const Value &body, GenRequest *gr, ApiError *e) const {
    if (!parse_sampling(body, gr, e)) return false;
    std::string model;
    if (body.get_string("model", &model) && !model.empty() && model != engine_->model_id()) {
      // Not fatal: harnesses (and OpenAI's own SDKs) send whatever name the user
      // configured, and there is exactly one model here.
      std::fprintf(stderr, "[serve] note: request asked for model '%s', serving '%s'\n",
                   model.c_str(), engine_->model_id().c_str());
    }
    return true;
  }

  bool parse_messages(const Value &body, GenRequest *gr, ApiError *e) const {
    const Value *msgs = body.member("messages");
    if (msgs == nullptr || !msgs->is_array() || msgs->size() == 0) {
      *e = {400, "messages must be a non-empty array", "invalid_request_error", "invalid_messages",
            "messages"};
      return false;
    }
    for (std::size_t i = 0; i < msgs->size(); ++i) {
      const Value *m = msgs->at(i);
      if (m == nullptr || !m->is_object()) {
        *e = {400, "every message must be an object with role and content",
              "invalid_request_error", "invalid_messages", "messages"};
        return false;
      }
      std::string role;
      if (!m->get_string("role", &role)) {
        *e = {400, "message " + std::to_string(i) + " has no role", "invalid_request_error",
              "invalid_messages", "messages[" + std::to_string(i) + "].role"};
        return false;
      }
      if (role != "system" && role != "developer" && role != "user" && role != "assistant" &&
          role != "tool") {
        *e = {400, "unsupported role '" + role +
                       "' (expected system, developer, user, assistant or tool)",
              "invalid_request_error", "invalid_role",
              "messages[" + std::to_string(i) + "].role"};
        return false;
      }
      std::string content;
      if (!content_text(*m, &content, e, i)) return false;
      rdna4::ChatMessage cm;
      cm.role = role;
      cm.content = content;
      if (role == "assistant") {
        const Value *rc = m->member("reasoning_content");
        if (rc != nullptr && rc->is_string()) cm.reasoning_content = rc->as_string();
      }
      gr->messages.push_back(std::move(cm));
    }
    return true;
  }

  // content is a string, or the OpenAI "content parts" array (vision style), in
  // which case the text parts are concatenated and the rest is ignored.
  bool content_text(const Value &m, std::string *out, ApiError *e, std::size_t i) const {
    const Value *c = m.member("content");
    if (c == nullptr || c->is_null()) {
      out->clear();
      return true;
    }
    if (c->is_string()) {
      *out = c->as_string();
      return true;
    }
    if (c->is_array()) {
      out->clear();
      for (std::size_t k = 0; k < c->size(); ++k) {
        const Value *part = c->at(k);
        if (part == nullptr || !part->is_object()) continue;
        const Value *txt = part->member("text");
        if (txt != nullptr && txt->is_string()) {
          *out += txt->as_string();
          continue;
        }
        if (args_.verbose) {
          std::fprintf(stderr,
                       "[serve] note: message %zu part %zu is not text; only text parts are used\n",
                       i, k);
        }
      }
      return true;
    }
    *e = {400, "message " + std::to_string(i) + ": content must be a string or an array",
          "invalid_request_error", "invalid_content",
          "messages[" + std::to_string(i) + "].content"};
    return false;
  }

  // Shared context handling: 400 when the prompt cannot fit at all, otherwise
  // max_tokens is clipped to the room left (logged, never silent — the same
  // policy cmd_run uses for -n).
  bool fit_context(const std::vector<std::int32_t> &prompt_ids, GenRequest *gr,
                   ApiError *e) const {
    const long long ctx = engine_->ctx_size();
    const long long prompt_len = static_cast<long long>(prompt_ids.size());
    if (prompt_len >= ctx) {
      *e = {400,
            "the prompt is " + std::to_string(prompt_len) + " tokens but this model's context is " +
                std::to_string(ctx) +
                " tokens: shorten the conversation or restart the server with a larger "
                "--ctx-size",
            "invalid_request_error", "context_length_exceeded", ""};
      return false;
    }
    const long long room = ctx - prompt_len;
    if (gr->max_tokens < 0 || gr->max_tokens > room) {
      if (gr->max_tokens > room) {
        std::fprintf(stderr,
                     "[serve] warning: max_tokens %lld clipped to %lld (context %lld, prompt %lld)\n",
                     gr->max_tokens, room, ctx, prompt_len);
      }
      gr->max_tokens = room;
    }
    return true;
  }

  // ---- generation driver ------------------------------------------------
  // `on_delta` is the endpoint's SSE writer (empty for a non-streaming
  // request); it returns false when the client is gone, which stops generation.
  bool run_generation(const std::vector<std::int32_t> &prompt_ids, const GenRequest &gr,
                      const char *endpoint, const DeltaFn &on_delta, Responder &out,
                      GenResult *res, ApiError *e) {
    const double t0 = now_s();
    std::string err;
    const bool ok = engine_->generate(prompt_ids, gr, on_delta, *res, err);
    if (!ok) {
      if (!on_delta) {
        *e = {500, "generation failed: " + err, "server_error", "generation_failed", ""};
        return false;
      }
      // The headers are already out, so the failure has to travel as a stream
      // event: the OpenAI SDKs raise on a chunk carrying `error`.
      Value root = Value::object();
      root.set("error", rdna4::server::openai_error(err, "server_error", "generation_failed"));
      out.stream_chunk("data: " + root.dump() + "\n\n");
      out.stream_chunk("data: [DONE]\n\n");
      out.end_stream();
      std::fprintf(stderr, "[serve] %s: generation failed: %s\n", endpoint, err.c_str());
      return false;
    }
    const double dt = now_s() - t0;
    std::fprintf(stderr,
                 "[serve] %s: prompt %zu tok (prefill %.2f s), %zu tok generated in %.2f s "
                 "(%.1f tok/s), finish=%s%s\n",
                 endpoint, prompt_ids.size(), res->prefill_s, res->generated_ids.size(),
                 res->decode_s,
                 res->decode_s > 0 ? (double)res->generated_ids.size() / res->decode_s : 0.0,
                 res->finish_reason.c_str(),
                 res->matched_stop.empty() ? "" : (" (stop string)"));
    if (args_.verbose) {
      std::fprintf(stderr, "prompt ids:");
      for (const std::int32_t id : prompt_ids) std::fprintf(stderr, " %d", id);
      std::fprintf(stderr, "\ngenerated ids:");
      for (const std::int32_t id : res->generated_ids) std::fprintf(stderr, " %d", id);
      std::fprintf(stderr, "\n[serve] temp %.2f%s top_k %d top_p %.2f min_p %.2f seed %llu "
                           "(total %.2f s)\n",
                   gr.sp.temp, gr.sp.temp <= 0.0f ? " [greedy]" : "", gr.sp.top_k, gr.sp.top_p,
                   gr.sp.min_p, (unsigned long long)gr.sp.seed, dt);
    }
    return true;
  }

  // The envelope every streaming chunk shares.
  static Value sse_root(const char *object, const std::string &id, long long created,
                        const std::string &model) {
    Value root = Value::object();
    root.set("id", Value::string(id));
    root.set("object", Value::string(object));
    root.set("created", Value::integer(created));
    root.set("model", Value::string(model));
    root.set("system_fingerprint", Value::string("rdna4-infer"));
    return root;
  }

  // One chat.completion.chunk carrying `delta` (finish_reason null).
  std::string chat_delta_chunk(const std::string &id, long long created,
                               const std::string &model, const char *kind,
                               const std::string &text) const {
    Value delta = Value::object();
    if (kind != nullptr) delta.set(kind, Value::string(text));
    Value choice = Value::object();
    choice.set("index", Value::integer(0));
    choice.set("delta", std::move(delta));
    choice.set("logprobs", Value::null());
    choice.set("finish_reason", Value::null());
    Value choices = Value::array();
    choices.push(std::move(choice));
    Value root = sse_root("chat.completion.chunk", id, created, model);
    root.set("choices", std::move(choices));
    return "data: " + root.dump() + "\n\n";
  }

  // One text_completion chunk carrying generated text for the legacy endpoint.
  std::string text_delta_chunk(const std::string &id, long long created,
                               const std::string &model, const std::string &text) const {
    Value choice = Value::object();
    choice.set("index", Value::integer(0));
    choice.set("text", Value::string(text));
    choice.set("logprobs", Value::null());
    choice.set("finish_reason", Value::null());
    Value choices = Value::array();
    choices.push(std::move(choice));
    Value root = sse_root("text_completion", id, created, model);
    root.set("choices", std::move(choices));
    return "data: " + root.dump() + "\n\n";
  }

  Value usage_json(const GenResult &res) const {
    Value u = Value::object();
    const long long p = static_cast<long long>(res.prompt_ids.size());
    const long long c = static_cast<long long>(res.generated_ids.size());
    u.set("prompt_tokens", Value::integer(p));
    u.set("completion_tokens", Value::integer(c));
    u.set("total_tokens", Value::integer(p + c));
    return u;
  }

  // ---- POST /v1/chat/completions ----------------------------------------
  void chat_completions(const Request &req, Responder &out) {
    Value body;
    ApiError e;
    if (!parse_json_body(req, body, &e) || !parse_common(body, &chat_req_, &e) ||
        !parse_messages(body, &chat_req_, &e)) {
      send_api_error(out, e);
      return;
    }
    GenRequest &gr = chat_req_;
    gr.chat = true;

    bool stream = false;
    body.get_bool("stream", &stream);
    bool include_usage = false;
    const Value *so = body.member("stream_options");
    if (so != nullptr && so->is_object()) so->get_bool("include_usage", &include_usage);

    std::vector<std::int32_t> prompt_ids;
    std::string perr;
    if (!engine_->build_prompt(gr, prompt_ids, perr)) {
      e = {400, "cannot render the conversation: " + perr, "invalid_request_error",
           "invalid_messages", ""};
      send_api_error(out, e);
      return;
    }
    if (!fit_context(prompt_ids, &gr, &e)) {
      send_api_error(out, e);
      return;
    }

    const std::string id = make_completion_id("chatcmpl-");
    const long long created = static_cast<long long>(std::time(nullptr));
    const std::string model = engine_->model_id();
    if (!stream) {
      GenResult res;
      if (!run_generation(prompt_ids, gr, "/v1/chat/completions", DeltaFn(), out, &res, &e)) {
        send_api_error(out, e);
        return;
      }
      Value message = Value::object();
      message.set("role", Value::string("assistant"));
      message.set("content", Value::string(res.text));
      Value choice = Value::object();
      choice.set("index", Value::integer(0));
      choice.set("message", std::move(message));
      choice.set("logprobs", Value::null());
      choice.set("finish_reason", Value::string(res.finish_reason));
      Value choices = Value::array();
      choices.push(std::move(choice));
      Value root = sse_root("chat.completion", id, created, model);
      root.set("choices", std::move(choices));
      root.set("usage", usage_json(res));
      out.send(200, "application/json", root.dump(), Headers{});
      return;
    }

    if (!out.begin_stream(200, "text/event-stream", Headers{})) return;
    // First chunk announces the role, as every OpenAI server does (clients rely
    // on it to build the message object before any content arrives).
    if (!out.stream_chunk(chat_delta_chunk(id, created, model, "role", "assistant"))) return;

    GenResult res;
    const DeltaFn on_delta = [&](const std::string &text) -> bool {
      return out.stream_chunk(chat_delta_chunk(id, created, model, "content", text));
    };
    if (!run_generation(prompt_ids, gr, "/v1/chat/completions", on_delta, out, &res, &e)) return;
    if (res.client_gone) return;  // the client is gone: nothing left to answer
    {
      Value choice = Value::object();
      choice.set("index", Value::integer(0));
      choice.set("delta", Value::object());
      choice.set("logprobs", Value::null());
      choice.set("finish_reason", Value::string(res.finish_reason));
      Value choices = Value::array();
      choices.push(std::move(choice));
      Value root = sse_root("chat.completion.chunk", id, created, model);
      root.set("choices", std::move(choices));
      out.stream_chunk("data: " + root.dump() + "\n\n");
    }
    if (include_usage) {
      Value root = sse_root("chat.completion.chunk", id, created, model);
      root.set("choices", Value::array());
      root.set("usage", usage_json(res));
      out.stream_chunk("data: " + root.dump() + "\n\n");
    }
    out.stream_chunk("data: [DONE]\n\n");
    out.end_stream();
  }

  // ---- POST /v1/completions --------------------------------------------
  void completions(const Request &req, Responder &out) {
    Value body;
    ApiError e;
    if (!parse_json_body(req, body, &e) || !parse_common(body, &chat_req_, &e)) {
      send_api_error(out, e);
      return;
    }
    GenRequest &gr = chat_req_;
    gr.chat = false;
    const Value *p = body.member("prompt");
    if (p == nullptr || p->is_null()) {
      e = {400, "prompt is required for /v1/completions", "invalid_request_error", "invalid_prompt",
           "prompt"};
      send_api_error(out, e);
      return;
    }
    if (p->is_string()) {
      gr.prompt = p->as_string();
    } else if (p->is_array() && p->size() == 1 && p->at(0) != nullptr && p->at(0)->is_string()) {
      gr.prompt = p->at(0)->as_string();
    } else {
      e = {400, "prompt must be a string (a single-element array is also accepted; batches are "
                "not supported)",
           "invalid_request_error", "invalid_prompt", "prompt"};
      send_api_error(out, e);
      return;
    }
    if (gr.prompt.empty()) {
      e = {400, "prompt must not be empty", "invalid_request_error", "invalid_prompt", "prompt"};
      send_api_error(out, e);
      return;
    }

    bool stream = false;
    body.get_bool("stream", &stream);
    bool include_usage = false;
    const Value *so = body.member("stream_options");
    if (so != nullptr && so->is_object()) so->get_bool("include_usage", &include_usage);

    std::vector<std::int32_t> prompt_ids;
    std::string perr;
    if (!engine_->build_prompt(gr, prompt_ids, perr)) {
      e = {400, perr, "invalid_request_error", "invalid_prompt", ""};
      send_api_error(out, e);
      return;
    }
    if (!fit_context(prompt_ids, &gr, &e)) {
      send_api_error(out, e);
      return;
    }

    const std::string id = make_completion_id("cmpl-");
    const long long created = static_cast<long long>(std::time(nullptr));
    const std::string model = engine_->model_id();
    if (!stream) {
      GenResult res;
      if (!run_generation(prompt_ids, gr, "/v1/completions", DeltaFn(), out, &res, &e)) {
        send_api_error(out, e);
        return;
      }
      Value choice = Value::object();
      choice.set("index", Value::integer(0));
      choice.set("text", Value::string(res.text));
      choice.set("logprobs", Value::null());
      choice.set("finish_reason", Value::string(res.finish_reason));
      Value choices = Value::array();
      choices.push(std::move(choice));
      Value root = sse_root("text_completion", id, created, model);
      root.set("choices", std::move(choices));
      root.set("usage", usage_json(res));
      out.send(200, "application/json", root.dump(), Headers{});
      return;
    }
    if (!out.begin_stream(200, "text/event-stream", Headers{})) return;
    GenResult res;
    const DeltaFn on_delta = [&](const std::string &text) -> bool {
      return out.stream_chunk(text_delta_chunk(id, created, model, text));
    };
    if (!run_generation(prompt_ids, gr, "/v1/completions", on_delta, out, &res, &e)) return;
    if (res.client_gone) return;
    {
      Value choice = Value::object();
      choice.set("index", Value::integer(0));
      choice.set("text", Value::string(""));
      choice.set("logprobs", Value::null());
      choice.set("finish_reason", Value::string(res.finish_reason));
      Value choices = Value::array();
      choices.push(std::move(choice));
      Value root = sse_root("text_completion", id, created, model);
      root.set("choices", std::move(choices));
      out.stream_chunk("data: " + root.dump() + "\n\n");
    }
    if (include_usage) {
      Value root = sse_root("text_completion", id, created, model);
      root.set("choices", Value::array());
      root.set("usage", usage_json(res));
      out.stream_chunk("data: " + root.dump() + "\n\n");
    }
    out.stream_chunk("data: [DONE]\n\n");
    out.end_stream();
  }

  Engine *engine_;
  ServeArgs args_;
  GenRequest chat_req_;  // scratch, reused by both endpoints (single-threaded)
};

}  // namespace

int cmd_serve(int argc, char **argv) {
  ServeArgs args;
  const int parsed = parse_serve_args(argc, argv, &args);
  if (parsed == 2) return 0;  // --help
  if (parsed != 0) return 1;

  // The socket comes first: "--port 8080 already in use" must be answered in
  // milliseconds, not after a 20+ s GGUF load. The routes are registered once
  // the engine exists (the server only holds a pointer to the router).
  rdna4::server::ServerOptions opts;
  opts.host = args.host;
  opts.port = args.port;
  opts.verbose = args.verbose;
  Router router;
  rdna4::server::HttpServer server(opts, &router);
  std::string err;
  if (!server.open(err)) {
    std::fprintf(stderr, "rdna4-infer serve: %s\n", err.c_str());
    return 1;
  }

  Engine engine;
  if (!engine.init(args, err)) {
    std::fprintf(stderr, "rdna4-infer serve: %s\n", err.c_str());
    if (err.find("requires gfx1201") != std::string::npos) return 2;
    if (err.find("insufficient VRAM") != std::string::npos) return 3;
    return 1;
  }

  ServeApp app(&engine, args);
  app.register_routes(&router);

  std::printf("rdna4-infer serve: listening on http://%s:%d (model %s, ctx %llu, kv %s/%s)\n",
              args.host.c_str(), server.port(), engine.model_id().c_str(),
              (unsigned long long)args.ctx_size, kv_type_cstr(args.kv_k),
              kv_type_cstr(args.kv_v));
  std::fflush(stdout);
  std::fprintf(stderr,
               "rdna4-infer serve: one request at a time, connection closed after each response; "
               "Ctrl-C stops the server\n");

  if (!server.serve(err)) {
    std::fprintf(stderr, "rdna4-infer serve: %s\n", err.c_str());
    return 1;
  }
  std::fprintf(stderr, "rdna4-infer serve: stopped\n");
  return 0;
}
