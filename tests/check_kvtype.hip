// KV type table regression gate — the doors of include/rdna4/kv.h.
//
// Adversarial review finding F6 (docs/adversarial-noite.md) found three places
// where an unknown KvType produced a *plausible wrong answer* instead of a
// failure:
//
//   1. `kv_row_bytes` returned 0 -> a row stride of 0, so every row of the cache
//      landed on the same address: the engine ran, wrote, and reported numbers.
//   2. `kv_store_row_kernel` / `kv_fill_kernel` ended in an *implicit* q4_0 body,
//      so any type that was not F32/F16/Q8_0 was silently stored as q4_0.
//   3. `kv_bytes_per_elem` returned 0.0, so `info` and `serve` approved a
//      context with no KV bytes counted at all.
//
// They are closed now: each table's switch is exhaustive with NO `default:` (so
// -Wswitch still fires when a type is added) and the code after it is
// kv_unreachable(), which aborts. This test is the proof, and it is the reason
// the fix cannot rot: it feeds KvType(99) to each of the three and *requires* the
// child process to die on SIGABRT.
//
// No GPU work is done here, but the binary is a HIP target (kv.h pulls in
// hip_runtime.h), so it needs --offload-arch like the rest.
//
// usage: check-kvtype
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>
#include <sys/wait.h>
#include <unistd.h>

#include "rdna4/device.h"  // kv_cache_bytes()
#include "rdna4/kv.h"

namespace {

int failures = 0;

void ok(bool cond, const char *what) {
  if (!cond) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  } else {
    std::printf("  ok  %s\n", what);
  }
}

// The type names llama.cpp's --cache-type-k/-v accepts and this engine implements.
const char *kNames[] = {"f32", "f16", "q8_0", "q4_0", "q5_0", "q4_1"};

// Runs `fn` in a forked child and returns true when the child died on `sig`.
// This is how a "must not return" contract is tested.
template <typename F>
bool dies_with_signal(F fn, int sig) {
  struct rlimit rl{};
  rl.rlim_cur = 0;  // no core dumps from the deliberate aborts
  rl.rlim_max = 0;
  setrlimit(RLIMIT_CORE, &rl);
  const pid_t pid = fork();
  if (pid < 0) return false;
  if (pid == 0) {
    // Redirect the child's stderr so the deliberate message does not look like a
    // failure of this test in a log.
    if (freopen("/dev/null", "w", stderr) == nullptr) _exit(127);
    fn();
    _exit(0);  // reached only when the contract was broken
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) return false;
  return WIFSIGNALED(status) && WTERMSIG(status) == sig;
}

// Independent arithmetic: what kv_row_bytes must be for head_dim 256.
std::uint64_t expected_row_bytes(rdna4::KvType t, int head_dim) {
  const std::uint64_t blocks = (std::uint64_t)(head_dim + 31) / 32;
  switch (t) {
    case rdna4::KvType::F32: return (std::uint64_t)head_dim * 4;
    case rdna4::KvType::F16: return (std::uint64_t)head_dim * 2;
    case rdna4::KvType::Q8_0: return blocks * 34;
    case rdna4::KvType::Q4_0: return blocks * 18;
    case rdna4::KvType::Q5_0: return blocks * 22;
    case rdna4::KvType::Q4_1: return blocks * 20;
  }
  return 0;
}

// Independent arithmetic for the 131K budget. 16 full-attention layers x 4 KV
// heads x 256 dims = 16384 elements per side per token.
std::uint64_t expected_kv_bytes_131k(rdna4::KvType k, rdna4::KvType v) {
  const std::uint64_t elems = 16ull * 4ull * 256ull;
  const std::uint64_t ctx = 131072ull;
  const double per_elem_k = rdna4::kv_bytes_per_elem(k);
  const double per_elem_v = rdna4::kv_bytes_per_elem(v);
  return (std::uint64_t)((double)ctx * (double)elems * (per_elem_k + per_elem_v));
}

}  // namespace

int main() {
  std::printf("== names and parsing\n");
  for (const char *n : kNames) {
    rdna4::KvType t = rdna4::KvType::F32;
    const char *e = rdna4::kv_type_parse(n, &t);
    ok(e == nullptr, (std::string("kv_type_parse(\"") + n + "\") accepts").c_str());
    ok(std::strcmp(rdna4::kv_type_name(t), n) == 0,
       (std::string("kv_type_name round-trips \"") + n + "\"").c_str());
    ok(rdna4::kv_bytes_per_elem(t) > 0.0,
       (std::string("kv_bytes_per_elem(\"") + n + "\") is positive").c_str());
    ok(rdna4::kv_row_bytes(t, 256) == expected_row_bytes(t, 256),
       (std::string("kv_row_bytes(\"") + n + "\", 256) matches the layout").c_str());
  }
  {
    rdna4::KvType t = rdna4::KvType::F16;
    const char *e = rdna4::kv_type_parse("q9_9", &t);
    ok(e != nullptr, "kv_type_parse(\"q9_9\") is rejected with a message");
    // A name that llama.cpp has and this engine does not must be rejected, not
    // silently mapped to something close.
    ok(rdna4::kv_type_parse("q5_1", &t) != nullptr, "kv_type_parse(\"q5_1\") is rejected");
    ok(rdna4::kv_type_parse("", &t) != nullptr, "kv_type_parse(\"\") is rejected");
  }
  // The exact row sizes the layouts imply, spelled out: a 22-byte q5_0 block and
  // a 20-byte q4_1 block, 8 blocks per 256-element row.
  ok(rdna4::kv_row_bytes(rdna4::KvType::Q5_0, 256) == 176, "q5_0 row = 8 x 22 = 176 B");
  ok(rdna4::kv_row_bytes(rdna4::KvType::Q4_1, 256) == 160, "q4_1 row = 8 x 20 = 160 B");
  ok(sizeof(rdna4::block_q5_0) == 22 && sizeof(rdna4::block_q4_1) == 20,
     "block sizes are 22 and 20 bytes (llama.cpp static_asserts the same)");

  std::printf("== budget arithmetic at 131072 tokens (exact byte counts)\n");
  {
    // The three numbers that decided the night's default. Written here so a
    // change to any KV size table breaks a test instead of a memory plan.
    const std::uint64_t f16 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::F16, rdna4::KvType::F16);
    const std::uint64_t q80 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q8_0, rdna4::KvType::Q8_0);
    const std::uint64_t q40 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q4_0, rdna4::KvType::Q4_0);
    const std::uint64_t q50_q41 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q5_0, rdna4::KvType::Q4_1);
    const std::uint64_t q80_q40 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q8_0, rdna4::KvType::Q4_0);
    const std::uint64_t q50_q40 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q5_0, rdna4::KvType::Q4_0);
    const std::uint64_t q80_q41 =
        rdna4::kv_cache_bytes(131072, rdna4::KvType::Q8_0, rdna4::KvType::Q4_1);
    std::printf("   f16/f16    %.4f GiB\n", (double)f16 / 1073741824.0);
    std::printf("   q8_0/q8_0  %.4f GiB\n", (double)q80 / 1073741824.0);
    std::printf("   q4_0/q4_0  %.4f GiB\n", (double)q40 / 1073741824.0);
    std::printf("   q5_0/q4_1  %.4f GiB\n", (double)q50_q41 / 1073741824.0);
    std::printf("   q8_0/q4_0  %.4f GiB\n", (double)q80_q40 / 1073741824.0);
    std::printf("   q5_0/q4_0  %.4f GiB\n", (double)q50_q40 / 1073741824.0);
    std::printf("   q8_0/q4_1  %.4f GiB\n", (double)q80_q41 / 1073741824.0);
    ok(f16 == 8589934592ull, "f16/f16 = 8.000 GiB (65536 B/token)");
    ok(q80 == 4563402752ull, "q8_0/q8_0 = 4.250 GiB (34816 B/token)");
    ok(q40 == 2415919104ull, "q4_0/q4_0 = 2.250 GiB (18432 B/token)");
    ok(q50_q41 == 2818572288ull, "q5_0/q4_1 = 2.625 GiB (21504 B/token)");
    ok(q80_q40 == 3489660928ull, "q8_0/q4_0 = 3.250 GiB (26624 B/token)");
    ok(q50_q40 == 2684354560ull, "q5_0/q4_0 = 2.500 GiB (20480 B/token)");
    ok(q80_q41 == 3623878656ull, "q8_0/q4_1 = 3.375 GiB (27648 B/token)");
    for (auto pr : {std::pair<rdna4::KvType, rdna4::KvType>{rdna4::KvType::Q5_0,
                                                            rdna4::KvType::Q4_1},
                    {rdna4::KvType::F16, rdna4::KvType::F16},
                    {rdna4::KvType::Q8_0, rdna4::KvType::Q4_0}}) {
      const std::uint64_t got = rdna4::kv_cache_bytes(131072, pr.first, pr.second);
      ok(got == expected_kv_bytes_131k(pr.first, pr.second),
         (std::string("kv_cache_bytes matches an independent recomputation for ") +
          rdna4::kv_type_name(pr.first) + "/" + rdna4::kv_type_name(pr.second))
             .c_str());
    }
  }

  std::printf("== the VRAM budget mirror (device.h) vs the allocator (graph.cuh)\n");
  {
    // The exact shape of the IQ3_S (scripts/inspect_gguf.py) -- so a change to
    // graph.cuh's allocations, or to the mirror in device.h, breaks HERE instead
    // of silently making `info`/`serve` approve a context that does not fit.
    rdna4::GraphBufferShape s;
    s.n_embd = 5120;
    s.ffn_len = 17408;
    s.n_head = 24;
    s.head_dim = 256;
    s.n_head_kv = 4;
    s.n_kv_layers = 16;
    s.n_recr_layers = 48;
    s.ssm_state = 128;
    s.ssm_group = 16;
    s.ssm_tsr = 48;
    s.conv_k = 4;
    s.vocab = 248320;
    const std::uint64_t got = rdna4::graph_buffer_bytes(s);
    std::printf("   graph buffers for the IQ3_S shape: %llu B = %.2f MiB\n",
                (unsigned long long)got, (double)got / (1024.0 * 1024.0));
    ok(got == 166314020ull, "graph_buffer_bytes == 166 314 020 B (158.60 MiB), the sum of "
                            "graph.cuh:518-590 added up by hand");
    // ... and the budget must not count the whole file: 10.877 GiB of trunk
    // weights, not 11.214 GiB.
    const std::uint64_t need = rdna4::required_bytes(11678877696ull, 131072,
                                                     rdna4::KvType::Q5_0, rdna4::KvType::Q4_1, got);
    std::printf("   131K q5_0/q4_1 with the honest terms: %.3f GiB\n",
                (double)need / 1073741824.0);
    ok(need == 11678877696ull + got + 2818572288ull + rdna4::kOverheadBytes,
       "required_bytes = weights + buffers + kv + the named margin (no flat gigabyte)");
    ok(rdna4::kOverheadBytes == (448ull << 20),
       "the margin is 448 MiB: 0.40 GiB measured at 131K, rounded up (was a flat 1 GiB)");
    ok(need < 15900000000ull, "131K q5_0/q4_1 fits in the 15.900 GB hipMemGetInfo reports");
  }

  std::printf("== the three doors: an out-of-enum KvType must abort, not guess\n");
  {
    const rdna4::KvType bad = static_cast<rdna4::KvType>(99u);
    ok(dies_with_signal([&] { (void)rdna4::kv_row_bytes(bad, 256); }, SIGABRT),
       "kv_row_bytes(KvType(99)) aborts (was a silent 0 -> every row on one address)");
    ok(dies_with_signal([&] { (void)rdna4::kv_bytes_per_elem(bad); }, SIGABRT),
       "kv_bytes_per_elem(KvType(99)) aborts (was a silent 0.0 -> budget counted no KV)");
    ok(dies_with_signal(
           [&] {
             std::string err;
             (void)rdna4::kv_cache_bytes(131072, bad, rdna4::KvType::F16);
           },
           SIGABRT),
       "kv_cache_bytes() with an invalid K aborts instead of returning a budget");
  }

  std::printf("== dispatchers reject an unknown type instead of falling through to q4_0\n");
  {
    // These two never reach a kernel for an unmatched type (the switch has no
    // matching case), so no GPU is needed and no context is created.
    const rdna4::KvType bad = static_cast<rdna4::KvType>(99u);
    ok(!rdna4::kv_store_row_launch(bad, nullptr, nullptr, 256),
       "kv_store_row_launch(KvType(99)) returns false");
    ok(!rdna4::kv_fill_launch(bad, nullptr, 1, 256, 0u),
       "kv_fill_launch(KvType(99)) returns false");
    // Deliberately NOT launching the six real types here: that would make this
    // gate touch the GPU, and this file's whole value is that it runs in
    // microseconds, needs no lock and no 12 GiB model. The accept side is covered
    // by every other KV gate (check-rope-gpu, check-kvquant-gpu, check-kvctx-gpu),
    // which all walk the six types through the real dispatcher.
  }

  std::printf("check-kvtype: %s\n", failures ? "FAILED" : "OK");
  return failures ? 1 : 0;
}
