// Device helpers (PLAN.md M0). Pure HIP queries + VRAM budget math, no model knowledge.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include <hip/hip_runtime.h>

#include "rdna4/kv.h"  // KvType + kv_bytes_per_elem (single source of truth)

namespace rdna4 {

// Returns 0 on success and fills total_vram + arch name; nonzero on failure.
inline int query_device(char *arch_out, std::size_t arch_cap, std::size_t *total_vram) {
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
    return 1;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    return 2;
  }
  std::strncpy(arch_out, prop.gcnArchName, arch_cap - 1);
  arch_out[arch_cap - 1] = '\0';
  *total_vram = prop.totalGlobalMem;
  return 0;
}

inline bool is_gfx1201(const char *arch) { return std::strstr(arch, "gfx1201") != nullptr; }

// KV estimate for the qwen35-27B hybrid (17 full-attention layers x 4 KV heads
// x (256+256) head dim). Bytes/token scale with the KV cache type:
// F16 2.0 B/elem, Q8_0 ~1.06, Q4_0 ~0.56. M1/M3 replace the layer/head/dim
// constants with real hparams from the GGUF.
// 16 KV-bearing full-attention layers (i%4==3 for i in 0..63). The 65th
// block is the MTP block, which v1 does not run, so it holds no KV.
// (M1 layout validation: 16 full-attn + 48 GDN + 1 MTP.)
inline constexpr std::uint64_t kQwen35KvElemsPerToken = 16u * 4u * (256u + 256u);

// Bytes per cache element for a cache-type name. Delegates to kv.h so the CLI
// budget and the graph cannot drift apart (the engine stores the rows itself).
inline double kv_bytes_per_elem(const char *kv_type) {
  KvType t = KvType::F16;  // llama.cpp's default
  if (kv_type_parse(kv_type, &t) != nullptr) return 0.0;
  return kv_bytes_per_elem(t);
}
inline constexpr std::uint64_t kOverheadBytes = 1u << 30;  // kernels, buffers, fragmentation

// Upper bound for any --ctx-size, enforced by every command that takes one
// (run/bench/ppl/serve). The graph takes the context as an int, so a larger
// value would be truncated before any budget check could reject it, and the
// KV cache would be sized from the truncated number (review finding M5;
// `serve` was capped first, the CLI commands had the same hole).
inline constexpr std::uint64_t kMaxCtxSize = 1u << 24;

// Bytes held by the KV cache at `ctx_size`. kQwen35KvElemsPerToken counts the K
// *and* the V elements of a token, so each side holds half of it. Review finding
// M4: `serve` sized the budget with kv_bytes_per_elem(kv_k) over the full
// constant, which is twice the K bytes — exactly right only while K and V have
// the same type, and wrong (too small, i.e. hipMalloc at load time) as soon as
// they differ.
inline std::uint64_t kv_cache_bytes(std::uint64_t ctx_size, KvType kv_k, KvType kv_v) {
  const double half = static_cast<double>(kQwen35KvElemsPerToken) / 2.0;
  return static_cast<std::uint64_t>(static_cast<double>(ctx_size) * half *
                                    (kv_bytes_per_elem(kv_k) + kv_bytes_per_elem(kv_v)));
}
inline std::uint64_t kv_cache_bytes(std::uint64_t ctx_size, const char *kv_k, const char *kv_v) {
  KvType k = KvType::F16;  // llama.cpp's default
  KvType v = KvType::F16;
  if (kv_type_parse(kv_k, &k) != nullptr || kv_type_parse(kv_v, &v) != nullptr) return 0;
  return kv_cache_bytes(ctx_size, k, v);
}

inline std::uint64_t required_bytes(std::uint64_t file_bytes, std::uint64_t ctx_size,
                                    const char *kv_type) {
  const auto kv =
      static_cast<std::uint64_t>(ctx_size * kQwen35KvElemsPerToken * kv_bytes_per_elem(kv_type));
  return file_bytes + kv + kOverheadBytes;
}

// Same, with independent K and V cache types.
inline std::uint64_t required_bytes(std::uint64_t file_bytes, std::uint64_t ctx_size,
                                    const char *kv_k, const char *kv_v) {
  return file_bytes + kv_cache_bytes(ctx_size, kv_k, kv_v) + kOverheadBytes;
}

}  // namespace rdna4
