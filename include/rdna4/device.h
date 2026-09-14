// Device helpers (PLAN.md M0). Pure HIP queries + VRAM budget math, no model knowledge.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>

#include <hip/hip_runtime.h>

#include "rdna4/dtype.h"  // tensor_bytes / dtype_block_bytes: what up() hipMallocs
#include "rdna4/kv.h"     // KvType + kv_bytes_per_elem (single source of truth)

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
// ---------------------------------------------------------------------------
// VRAM budget — rewritten for the 131K target (coordenador, 03:10).
//
// The old budget was `file_bytes + kv + kOverheadBytes` with a flat 1 GiB of
// "overhead", and it was wrong in BOTH terms:
//
//   * `file_bytes` is the GGUF *file*, 11.214 GiB for the IQ3_S, but the engine
//     uploads 10.877 GiB of trunk tensors: the 15 `blk.64.*` tensors (0.327 GiB)
//     are the MTP block and are only uploaded when `--mtp` is used
//     (`Graph::up()` is never called for them otherwise). There is no padding
//     between tensors — measured: the offsets are contiguous.
//   * 1 GiB of overhead does not exist. The graph's fixed allocations were
//     measured at 157.98 MiB for this model, and they are *computable* from the
//     shape — so they are computed here instead of guessed.
//
// Net effect of the old constants: `info`/`serve` overestimated by ~1.18 GiB and
// refused configurations that fit — 131K with q8_0/q8_0 (15.281 GiB) among them;
// 131K was reported as "q4_0 only" when it is not.
//
// What is left is an explicit, documented margin instead of a blanket gigabyte:
// kAllocatorMarginBytes, for the HIP allocator rounding every one of the ~150
// weight allocations up to its own granularity plus fragmentation, and
// kRuntimeReserveBytes for the HIP context/module/queue structures. Both are
// small and named; see docs/journal-kv.md §6 for the measurement that produced
// them.
// ---------------------------------------------------------------------------

// Shapes the graph's fixed allocations depend on, filled in by the caller (which
// is the only place that has the GGUF config). Keeping this a POD means device.h
// stays free of model/loader headers.
struct GraphBufferShape {
  std::uint64_t n_embd = 0;
  std::uint64_t ffn_len = 0;
  std::uint64_t n_head = 0;
  std::uint64_t head_dim = 0;
  std::uint64_t n_head_kv = 0;
  std::uint64_t n_kv_layers = 0;   // full-attention layers (16 here)
  std::uint64_t n_recr_layers = 0; // GDN layers (48 here)
  std::uint64_t ssm_state = 0;     // ssm.state_size (128)
  std::uint64_t ssm_group = 0;     // ssm.group_count (16)
  std::uint64_t ssm_tsr = 0;       // ssm.time_step_rank (48)
  std::uint64_t conv_k = 0;        // ssm.conv_kernel (4)
  std::uint64_t vocab = 0;
  std::uint64_t max_batch = 16;    // Graph::kMaxBatch
  std::uint64_t max_splits = 16;   // tuned::kAttnMaxSplits
};

// Bytes of everything Graph::init() allocates that is NOT weights and NOT the KV
// cache: the single-token activation block, the batch block (fixed at
// max_batch tokens), the split-KV partials, the GDN recurrent state and conv
// state, the q8 scratch and the logits. Mirrors graph.cuh:518-585 line for line;
// tests/check-kvtype.hip pins the total for the IQ3_S shape (165 658 660 B =
// 157.98 MiB) so the mirror cannot drift from the allocator in silence.
inline std::uint64_t graph_buffer_bytes(const GraphBufferShape &s) {
  const std::uint64_t E = s.n_embd, F = s.ffn_len;
  const std::uint64_t kv_elems = s.n_head_kv * s.head_dim;
  const std::uint64_t d_inner = s.ssm_tsr * s.ssm_state;
  const std::uint64_t chan = 2 * s.ssm_group * s.ssm_state + d_inner;
  const std::uint64_t q_dim = s.n_head * s.head_dim;
  // single-token activations, in FLOATS (graph.cuh:518-522): 3E is d_x_/d_xn_/
  // d_ffnout_, 2*chan is d_qkv_/d_conv_, 2*F is d_ffn_a_/d_ffn_b_, 2*q_dim is
  // d_attnout_/d_attngate_, 2*kv_elems is d_kstage_/d_vstage_, 3*ssm_tsr is
  // d_alpha_/d_beta_/d_gate_.
  const std::uint64_t q_gate_floats = s.n_head * 2 * s.head_dim;  // d_proj_ / d_qb_
  const std::uint64_t single_floats = 3 * E + q_gate_floats + 2 * q_dim + 2 * chan + d_inner +
                                      3 * s.ssm_tsr + 2 * F + 2 * kv_elems;
  // batch block per token, in FLOATS (graph.cuh:544-553): the same set minus
  // d_proj_ (the batch path writes E-wide into d_projb_) and plus d_qkvb_.
  // + d_qkb_ (2*key_dim) and d_vcb_ (d_inner): the batched GDN staging the prefill
  // front added in graph.cuh:590. Missing them under-counted the budget by 640 KiB
  // (16 x (4096+6144) x 4 B) after the preventive merge -- see the note in
  // docs/journal-kv.md §8: a merge that touches graph.cuh invalidates this mirror,
  // and the pinned total in tests/check_kvtype.hip is what forces the update.
  const std::uint64_t per_token_floats = 3 * E + q_gate_floats + 2 * kv_elems + 2 * q_dim +
                                         2 * F + 2 * chan + d_inner + 3 * s.ssm_tsr +
                                         2 * (s.ssm_group * s.ssm_state) + d_inner;
  const std::uint64_t aq_blocks = (F + 31) / 32;  // block_q8_1 (36 B) per 32 elems
  const std::uint64_t single = single_floats * 4;
  const std::uint64_t batch = s.max_batch * (per_token_floats * 4 + aq_blocks * 36 + 4);
  // Atencao: o alocador dimensiona as parciais por (token, cabeca, split)
  // -- `batch_max_ * attn_partial_bytes(NH, HD, kAttnMaxSplits)` em graph.cuh --
  // e este espelho esquecia o fator `max_batch`. Medido: 396 288 B/token, ou seja
  // o espelho sub-contava 5 944 320 B no default de 16 tokens (e 45,7 MB num chunk
  // de 128). O erro estava escondido dentro da margem de 384 MiB (kAllocatorMarginBytes),
  // que foi calibrada contra este numero -- ver o comentario dela.
  const std::uint64_t partial = s.max_batch * s.n_head * s.max_splits * (2 + s.head_dim) * 4;
  const std::uint64_t state = s.n_recr_layers * s.ssm_tsr * s.ssm_state * s.ssm_state * 4;
  const std::uint64_t convst = s.n_recr_layers * (s.conv_k - 1) * chan * 4;
  const std::uint64_t q8 = ((F / 32) + 8) * 36;
  const std::uint64_t logits = s.vocab * 4;
  return single + batch + partial + state + convst + q8 + logits + 4 /* d_pos_ */;
}

// HIP rounds every large hipMalloc up to its own granularity and the engine makes
// ~150 weight allocations plus the caches; the context/module/queue structures
// add a small fixed cost on top. 384 + 64 MiB is MEASURED, not guessed: at 131K
// the engine's real footprint minus (weights + graph buffers + KV) came out at a
// constant 0.40 GiB across four cache types —
//   q4_0/q4_0   measured 13.87 GiB vs 13.48 predicted (+0.39)
//   q5_0/q4_1   measured 14.25 GiB vs 13.85 predicted (+0.40)
//   q8_0/q4_1   measured 15.00 GiB vs 14.60 predicted (+0.40)
//   q8_0/q8_0   measured 15.87 GiB vs 15.48 predicted (+0.39)
// (docs/journal-kv.md §6.4; the desktop's ~0.19 GiB is inside the measured
// figure and in the predictions, so it cancels in the difference.)
// Deliberately NOT a gigabyte: a blanket GiB is what made the old budget refuse
// configurations that fit.
inline constexpr std::uint64_t kAllocatorMarginBytes = 384u << 20;
inline constexpr std::uint64_t kRuntimeReserveBytes = 64u << 20;
// Kept as a name so no caller silently loses the margin, but no longer 1 GiB.
inline constexpr std::uint64_t kOverheadBytes = kAllocatorMarginBytes + kRuntimeReserveBytes;

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

// Total VRAM the engine will hold for a run. `weights_bytes` must be the bytes
// of the tensors the engine actually uploads -- NOT the size of the GGUF file.
// Use uploaded_weight_bytes(loader, use_mtp) below, which walks the same tensor
// names Graph::init() binds.
inline std::uint64_t required_bytes(std::uint64_t weights_bytes, std::uint64_t ctx_size,
                                    KvType kv_k, KvType kv_v, std::uint64_t buffers_bytes) {
  return weights_bytes + buffers_bytes + kv_cache_bytes(ctx_size, kv_k, kv_v) + kOverheadBytes;
}

// Teto RUNTIME do chunk do prefill em lote. O default (16) e' o `Graph::kMaxBatch`,
// que e' o que o GEMV em lote tem instanciado; `RD_PREFILL_CHUNK` sobe esse teto
// para o GEMM tilejado (docs/plano-prefill.md, degrau D2). Esta funcao e' a fonte
// unica: o alocador (graph.cuh) e o orcamento de VRAM (graph_buffer_bytes) leem
// daqui, para nao poderem divergir.
//
// Custo medido dos buffers de lote: 0,46 MB por token (18 buffers) + 19,6 KB/token
// de blocos q8_1 + 396 KB/token de parciais de atencao = ~0,87 MB/token, ou seja
// ~109 MiB num chunk de 128 (7,2 MiB no default de 16).
inline int prefill_chunk_cap() {
  static const int v = [] {
    const char *e = std::getenv("RD_PREFILL_CHUNK");
    int n = e ? std::atoi(e) : 16;
    if (n < 16) n = 16;
    if (n > 512) n = 512;
    return n;
  }();
  return v;
}

// Fills GraphBufferShape from a parsed qwen35 config. Duck-typed on the config
// (`block_count`, `full_attention_interval`, ...) so this header still needs no
// model.h. `vocab` comes from the caller because the config carries no vocab —
// the graph takes it from `output.weight`'s dim1.
template <typename Cfg>
inline GraphBufferShape graph_buffer_shape(const Cfg &cfg, std::uint64_t vocab) {
  GraphBufferShape s;
  const std::uint64_t trunk = cfg.block_count - cfg.nextn_predict_layers;  // 64
  s.n_kv_layers = cfg.full_attention_interval ? trunk / cfg.full_attention_interval : 0;
  s.n_recr_layers = trunk - s.n_kv_layers;
  s.n_embd = cfg.embedding_length;
  s.ffn_len = cfg.feed_forward_length;
  s.n_head = cfg.head_count;
  s.head_dim = cfg.key_length;
  s.n_head_kv = cfg.head_count_kv;
  s.ssm_state = cfg.ssm_state_size;
  s.ssm_group = cfg.ssm_group_count;
  s.ssm_tsr = cfg.ssm_time_step_rank;
  s.conv_k = cfg.ssm_conv_kernel;
  s.vocab = vocab;
  s.max_batch = (std::uint64_t)prefill_chunk_cap();
  return s;
}

// Bytes of the tensors Graph::init() actually uploads, read from the GGUF's own
// tensor table: every tensor except the MTP block's (`blk.<mtp_layer>.*`), which
// only the `--mtp` path binds. Counting the whole file instead is what made the
// old budget overestimate: the file carries 0.327 GiB of MTP weights the trunk
// never touches, plus the header and the metadata.
//
// `dtype_block_bytes`/`tensor_bytes` (dtype.h) are the very functions `up()` sizes
// its hipMalloc with, so this number cannot disagree with the allocator.
//
// Duck-typed on the loader (`meta()`, `dtype(i)`) so this header keeps needing
// neither loader.h nor model.h; the callers in main.hip/serve.hip already have
// both.
template <typename Loader>
inline std::uint64_t uploaded_weight_bytes(const Loader &ld, std::uint64_t mtp_layer,
                                           bool use_mtp) {
  if (use_mtp) {
    // The --mtp path uploads the block too, so no tensor is excluded. (It also
    // allocates the block's own KV cache and buffers; that is NOT counted here,
    // it is a few MiB and the caller's margin covers it.)
    std::uint64_t all = 0;
    for (std::size_t i = 0; i < ld.meta().tensors.size(); ++i) {
      const auto &t = ld.meta().tensors[i];  // duck-typed: no gguf.h here
      std::uint64_t nelem = 1;
      for (std::uint64_t d : t.dims) nelem *= d;
      all += tensor_bytes(ld.dtype(i), nelem);
    }
    return all;
  }
  const std::string prefix = "blk." + std::to_string(mtp_layer) + ".";
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < ld.meta().tensors.size(); ++i) {
    const auto &t = ld.meta().tensors[i];  // duck-typed: no gguf.h here
    if (t.name.compare(0, prefix.size(), prefix) == 0) continue;  // MTP block
    std::uint64_t nelem = 1;
    for (std::uint64_t d : t.dims) nelem *= d;
    total += tensor_bytes(ld.dtype(i), nelem);
  }
  return total;
}

}  // namespace rdna4
