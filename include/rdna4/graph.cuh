#pragma once
// M3 — qwen35 forward graph on gfx1201 (correctness first: one token at a time).
//
// Structure transcribed from llama.cpp's reference graph (src/models/qwen35.cpp,
// llama_model_qwen35::graph):
//
//   per layer: x -> attn_norm (RMS) -> [GDN branch | full-attention branch]
//              -> + x -> attn_post_norm (RMS) -> FFN -> + residual
//   final:     output_norm (RMS) -> output (LM head)
//
// The FFN has NO residual on its input: the reference keeps the tensor from
// before attn_post_norm and adds the FFN output back to it (qwen35.cpp L184-196).
//
// Weights stay resident in VRAM as raw GGUF bytes (the matvec consumes them in
// place); activations are f32. Full-attention layers keep a f32 KV cache
// (quantized cache types are a later step).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "rdna4/attn.cuh"
#include "rdna4/dequant_row.cuh"
#include "rdna4/dtype.h"
#include "rdna4/gdn.cuh"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"
#include "rdna4/model.h"
#include "rdna4/mtp.cuh"
#include "rdna4/nn.cuh"

namespace rdna4 {

// Diagnostic knob (used by tests/check_graph_gpu.hip): perturb every layer's
// own output — the exact place the activation quantization injects its error —
// by +-rel, to measure how much such an error is amplified by the 63 layers.
// measuring this is what separates "our deviation from the oracle is inherent
// quantization noise" from "there is a bug left".
__global__ void perturb_kernel(float *__restrict__ x, int n, float rel, unsigned seed,
                               int coherent) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const unsigned h = (unsigned)i * 2654435761u + seed * 40503u;
  const bool up = coherent || ((h >> 16) & 1u);
  x[i] *= up ? (1.0f + rel) : (1.0f - rel);
}

// Test hook: fill a float buffer with a deterministic pseudo-random pattern
// (used by the long-context smoke test to give the attention a full cache).
__global__ void fill_random_kernel(float *__restrict__ x, std::int64_t n, unsigned seed) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned h = (unsigned)(i * 2654435761u) ^ (seed * 2246822519u);
  h ^= h >> 13;
  h *= 1274126177u;
  x[i] = ((float)(h >> 8) / 8388608.0f) - 1.0f;  // ~(-1, 1)
}

struct GpuTensor {
  void *ptr = nullptr;
  std::size_t bytes = 0;
  DType dt = DType::F32;
  bool valid = false;
  int dim0 = 0, dim1 = 0;
};

class Graph {
 public:
  Graph(const GgufLoader &ld, const Qwen35Config &cfg) : ld_(ld), cfg_(cfg) {}
  ~Graph() { release(); }
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;

  // kv_k / kv_v select the KV cache storage types (kv.h); F16 is llama.cpp's
  // default and Q4_0 is what makes a 64K+ context fit the 16 GB budget.
  bool init(int max_ctx, KvType kv_k, KvType kv_v, std::string &err);
  bool init(int max_ctx, std::string &err) { return init(max_ctx, KvType::F32, KvType::F32, err); }

  // Validation hook: llama.cpp's eval callback names every graph node the same
  // way, so a test can compare each intermediate against the oracle dump. The
  // callback runs synchronously on the calling thread with a device pointer.
  using NodeCb =
      std::function<void(const char *name, int il, int t, const float *d_ptr, std::int64_t n)>;
  void set_node_cb(NodeCb cb) { cb_ = std::move(cb); }
  // M5 diagnostics: where the per-token wall time goes, split into the calling
  // thread queueing work and the thread blocked on the readbacks. Purely
  // observational (two steady_clock reads per token), so it never changes the
  // arithmetic; `bench` prints it.
  double host_launch_ms() const { return host_launch_ms_; }
  double host_readback_ms() const { return host_readback_ms_; }
  void reset_host_timing() { host_launch_ms_ = 0.0; host_readback_ms_ = 0.0; }
  // The CLI/bench decode loop does not consume `hidden`; skipping that blocking
  // copy removes one device sync per token (it would otherwise be merged into
  // forward_run's logits copy). Default true: the oracle tests compare `hidden`.
  void set_want_hidden(bool want) { want_hidden_ = want; }
  // Force a split count (0 = automatic). tests/check_kvctx_gpu.hip compares the
  // split-KV path against the unsplit one at the same context through this.
  void set_attn_splits(int n) {
    attn_splits_env_read_ = true;
    attn_splits_env_ = n < 0 ? 0 : n;
  }
  // per-layer output perturbation, relative to each layer's own contribution
  void set_layer_noise(float rel, bool coherent = false) {
    noise_rel_ = rel;
    noise_coherent_ = coherent;
  }
  void emit(const char *name, int il, const float *d_ptr, std::int64_t n) {
    if (cb_) cb_(name, il, cur_token_, d_ptr, n);
  }

  // Runs `n_tokens` embedding rows (f32 [n_tokens][n_embd]) at positions
  // start_pos..start_pos+n-1. Repeated calls continue the same KV cache and the
  // same recurrent state, so prefill can be split arbitrarily (verified to be
  // bit-identical to a single call by tests/check_kvctx_gpu.hip).
  bool forward(const std::vector<float> &embeddings, std::vector<float> &hidden,
               std::vector<float> &logits, std::string &err);
  bool forward(const std::vector<float> &embeddings, int start_pos, std::vector<float> &hidden,
               std::vector<float> &logits, std::string &err);

  // M4: the CLI's entry point. `token_embd.weight` is quantized in the UD files
  // (q3_K in IQ3_S, q4_K in IQ4_XS), and llama.cpp's graph feeds the *dequantized*
  // row to the trunk (`ggml_get_rows` -> f32), so the row is dequantized on the
  // device with the same kernels validated bit-exact against llama.cpp's
  // dequantize_row_* in tests/check_dequant_gpu.hip (dequant_row.cuh) — no host
  // round trip and no second implementation to keep in sync.
  bool forward_tokens(const std::vector<std::int32_t> &tokens, int start_pos,
                      std::vector<float> &hidden, std::vector<float> &logits, std::string &err);

  // Diagnostic hook (M5): run only the first `n` trunk layers. The bench uses it
  // to attribute the per-token cost between the layer loop and the head/glue; it
  // produces wrong text by construction, so it is only reachable from the bench.
  void debug_set_layer_limit(int n) { layer_limit_ = n < 0 ? n_layer() : n; }
  int debug_layer_limit() const { return layer_limit_ < 0 ? n_layer() : layer_limit_; }

  // M8: batched prefill. Processes `tokens` (2..kMaxBatch) in ONE layer-major
  // pass: the projections of all N tokens share one read of each weight (the
  // bit-exact `matvec_kernel_batch`, docs/medicoes-m6.md), while the parts that
  // are inherently sequential -- the KV writes, the attention, the GDN recurrence
  // -- still run token by token in order. Every launch keeps the per-token
  // arithmetic, so the result is BIT-IDENTICAL to calling forward_tokens() once
  // per token (tests/check_batch_gpu.hip asserts exactly that).
  // `logits` receives the logits of the LAST token (what generation needs); the
  // LM head costs one weight read per call, not one per token.
  bool forward_batch(const std::vector<std::int32_t> &tokens, int start_pos,
                     std::vector<float> &hidden, std::vector<float> &logits, std::string &err);
  static constexpr int kMaxBatch = 16;
  // Batch sizes the matvec has instantiations for (2/3/4/8/16); anything else is
  // split into a supported chunk plus a per-token tail.
  static int batch_supported(int n) {
    return n == 2 || n == 3 || n == 4 || n == 8 || n == 16;
  }

  // M5: brings the graph back to the state it has after init(), for evaluating a
  // second independent sequence (perplexity chunks, chat turns). The recurrent
  // (GDN) state is zeroed — llama.cpp also starts every sequence from zeros — and
  // the KV caches do NOT need clearing: attention is causal and every row is
  // rewritten as the new sequence advances from position 0, so no stale row can
  // be read.
  bool reset_state(std::string &err);

  // Test hooks (diagnostics only; see PLAN.md M3).
  // Fills every KV cache and GDN state with a deterministic pseudo-random
  // pattern so a single decode step can be measured against a full cache.
  bool debug_fill_caches(unsigned seed, std::string &err);
  // Bytes per full-attention layer's cache (multiply by the number of
  // full-attention layers for the total).
  std::size_t kv_layer_bytes() const { return kv_bytes_; }
  int n_full_attn() const { return n_layer() - count_recr(); }
  int max_ctx() const { return max_ctx_; }

  // Returns false when the copy failed. Without the check the destination keeps
  // the zeros from resize() and the caller treats them as real hidden state (the
  // MTP draft then runs on noise, with no message) — review finding H2.
  bool readback(const float *d, std::size_t n, std::vector<float> &out) const {
    out.resize(n);
    return hipMemcpy(out.data(), d, n * sizeof(float), hipMemcpyDeviceToHost) == hipSuccess;
  }

  int n_embd() const { return (int)cfg_.embedding_length; }
  int n_vocab() const { return output_.dim1; }

  // ---- MTP (feat/mtp) -----------------------------------------------------
  // Read-only borrow of the two weights the draft block shares with the trunk:
  // `token_embd.weight` (the draft's input embedding) and `output.weight` (the
  // shared LM head). Additive: nothing in the trunk forward changes, and the MTP
  // tensors themselves are only ever loaded by MtpHead::init(), so a run without
  // MTP keeps the exact same allocations and numerics as before.
  bool mtp_shared_weights(MtpShared &out, std::string &err) const {
    if (!tok_embd_.valid || !output_.valid) {
      err = "mtp_shared_weights: the trunk weights are not loaded";
      return false;
    }
    if (tok_embd_.dim1 != output_.dim1) {
      err = "mtp_shared_weights: token_embd and output disagree on n_vocab";
      return false;
    }
    out.tok_embd = tok_embd_.ptr;
    out.tok_embd_dt = tok_embd_.dt;
    out.output = output_.ptr;
    out.output_dt = output_.dt;
    out.n_vocab = output_.dim1;
    return true;
  }
  // Executed trunk blocks. block_count covers the trunk *plus* the MTP block(s),
  // so the trunk is block_count - nextn_predict_layers: 65 - 1 = 64 blocks
  // (0..63), of which the full-attention ones are i = 3, 7, ..., 63 (16 layers).
  int n_layer() const { return (int)cfg_.block_count - (int)cfg_.nextn_predict_layers; }
  int head_dim() const { return (int)cfg_.key_length; }
  int n_head() const { return (int)cfg_.head_count; }
  int n_head_kv() const { return (int)cfg_.head_count_kv; }
  int n_rot() const { return (int)cfg_.rope_dim_count; }
  bool is_recr(int il) const {
    return ((il + 1) % (int)cfg_.full_attention_interval) != 0 && il < n_layer();
  }

 private:
  bool up(const char *name, GpuTensor &t, std::string &err);
  bool up_f32(const char *name, GpuTensor &t, std::string &err);
  bool alloc(float *&p, std::size_t n, std::string &err);
  // Body shared by both entry points. Exactly one of host_emb (n_tokens * n_embd
  // f32 rows) and toks (token ids) is non-null.
  bool forward_run(std::size_t n_tokens, int start_pos, const float *host_emb,
                   const std::vector<std::int32_t> *toks, std::vector<float> &hidden,
                   std::vector<float> &logits, std::string &err);
  // Quantize the n rows at d_x (row stride ncols) and multiply by w for all n at
  // once. `act_ready` skips the quantization when the caller already did it for
  // the same rows (the FFN's gate/up pair shares one activation -> one launch and
  // one pass less, bit-exactly).
  bool proj_batch(const GpuTensor &w, const float *d_x, float *d_y, int nrows, int ncols, int n,
                  bool act_ready, std::string &err);
  bool quantize_batch(const float *d_x, int ncols, int n, std::string &err);
  bool forward_batch_layer(int il, int n, int pos0, std::string &err);
  bool proj(const GpuTensor &w, const float *d_x, float *d_y, int nrows, int ncols,
            std::string &err);
  // Same as proj() but reuses the activation already quantized into the q8
  // scratch by the *previous* proj() call. Contract: the caller guarantees that
  // (a) the previous call quantized exactly `ncols` floats from the same source,
  // and (b) nothing wrote the q8 scratch since. Nothing writes d_xn_/d_q8_ between
  // the q/k/v projections of a layer, nor between the FFN's gate and up, which is
  // where the redundancy was measured (192 of the 305 activation quantizations per
  // token were re-quantizing byte-identical blocks; docs/rocm-estudo.md §A1).
  bool proj_qq(const GpuTensor &w, float *d_y, int nrows, int ncols, std::string &err);
  bool add_residual(float *d_src, float *d_dst, int n, std::string &err);
  bool full_attn(int il, int t, int pos, std::string &err);
  bool kv_write(int il, int t, const float *d_ksrc, const float *d_vsrc, std::string &err);
  int attn_slot(int il) const {
    int a = 0;
    for (int i = 0; i < il; ++i) a += is_recr(i) ? 0 : 1;
    return a;
  }
  bool gdn_layer(int il, int t, std::string &err);
  bool ffn(int il, std::string &err);
  void release();

  const GgufLoader &ld_;
  Qwen35Config cfg_;
  NodeCb cb_;
  float noise_rel_ = 0.0f;
  bool noise_coherent_ = false;
  unsigned noise_seed_ = 0;
  double host_launch_ms_ = 0.0;
  double host_readback_ms_ = 0.0;
  bool want_hidden_ = true;
  int layer_limit_ = -1;  // < 0 => all layers
  mutable bool attn_splits_env_read_ = false;
  mutable int attn_splits_env_ = 0;
  int attn_splits_last_ = 1;

  struct LayerW {
    GpuTensor attn_norm, attn_post_norm, ffn_gate, ffn_up, ffn_down;
    GpuTensor attn_q, attn_k, attn_v, attn_output, attn_q_norm, attn_k_norm;
    GpuTensor attn_qkv, attn_gate, ssm_conv1d, ssm_a, ssm_dt, ssm_beta, ssm_alpha, ssm_norm,
        ssm_out;
    bool recr = false;
  };
  std::vector<LayerW> w_;
  GpuTensor tok_embd_, output_norm_, output_;

  float *d_x_ = nullptr, *d_xn_ = nullptr, *d_proj_ = nullptr, *d_ffnout_ = nullptr;
  KvType kv_k_ = KvType::F32, kv_v_ = KvType::F32;
  float *d_kstage_ = nullptr, *d_vstage_ = nullptr;
  void *d_k_ = nullptr, *d_v_ = nullptr;
  std::size_t kv_bytes_ = 0;
  int count_recr() const {
    int n = 0;
    for (int i = 0; i < n_layer(); ++i) n += is_recr(i) ? 1 : 0;
    return n;
  }
  float *d_attnout_ = nullptr, *d_attngate_ = nullptr;
  // M8 batch buffers: [kMaxBatch][width] each, plus one N-row activation block
  // matrix shared by every projection in the batch.
  float *d_xb_ = nullptr, *d_xnb_ = nullptr, *d_qb_ = nullptr, *d_kb_ = nullptr, *d_vb_ = nullptr;
  float *d_attnb_ = nullptr, *d_gateb_ = nullptr, *d_ffnab_ = nullptr, *d_ffnbb_ = nullptr;
  float *d_projb_ = nullptr, *d_qkvb_ = nullptr, *d_zb_ = nullptr, *d_convb_ = nullptr;
  float *d_alphab_ = nullptr, *d_betab_ = nullptr, *d_gate2b_ = nullptr;
  block_q8_1 *d_aqb_ = nullptr;
  std::size_t aq_blocks_per_row_ = 0;
  int *d_posb_ = nullptr;
  // Split-KV attention scratch: [n_head][kAttnMaxSplits][2 + head_dim] (M7).
  float *d_attn_partial_ = nullptr;
  static constexpr int kAttnMaxSplits = 16;
  // Splits are chosen by context length: below kAttnSplitMin keys one CTA already
  // covers the range, and keeping the unsplit path there keeps the short-context
  // gates bit-identical to the pre-M7 numbers (the golden run, the oracle dumps).
  static constexpr int kAttnSplitMin = 512;  // M8: 2048 -> 512 (medido +14,7% a 4K, docs/rocm-estudo.md)
  int attn_splits_for(int keys) const {
    // RD_ATTN_SPLITS forces a split count (diagnostics/tests: 1 = the pre-M7 path,
    // which is what makes the split path checkable against it at any context).
    if (!attn_splits_env_read_) {
      attn_splits_env_read_ = true;
      const char *e = getenv("RD_ATTN_SPLITS");
      attn_splits_env_ = e ? atoi(e) : 0;
    }
    if (attn_splits_env_ > 0) {
      return attn_splits_env_ > kAttnMaxSplits ? kAttnMaxSplits : attn_splits_env_;
    }
    if (keys < kAttnSplitMin) return 1;
    const int sp = keys / kAttnSplitMin;
    return sp > kAttnMaxSplits ? kAttnMaxSplits : sp;
  }
  int attn_splits_last() const { return attn_splits_last_; }
  float *d_qkv_ = nullptr, *d_conv_ = nullptr, *d_z_ = nullptr;
  float *d_alpha_ = nullptr, *d_beta_ = nullptr, *d_gate_ = nullptr;
  float *d_state_ = nullptr, *d_convst_ = nullptr;
  float *d_ffn_a_ = nullptr, *d_ffn_b_ = nullptr;
  float *d_q8_ = nullptr;
  float *d_logits_ = nullptr;  // cached LM-head output (allocated on first use)
  std::size_t q8_blocks_ = 0;
  int cur_token_ = 0;
  int *d_pos_ = nullptr;
  int max_ctx_ = 0;
};

// ---------------------------------------------------------------------------
// Tensors the kernels read as f32 (norm weights, ssm_a/ssm_dt, conv1d weights)
// must really be F32: nothing else in the pipeline would notice an F16 weight,
// the values would just be wrong (review M4).
inline bool Graph::up_f32(const char *name, GpuTensor &t, std::string &err) {
  if (!up(name, t, err)) return false;
  if (t.dt != DType::F32) {
    err = std::string("tensor ") + name + " must be f32, got " + dtype_name(t.dt);
    return false;
  }
  return true;
}

inline bool Graph::up(const char *name, GpuTensor &t, std::string &err) {
  const auto *info = ld_.find(name);
  if (!info) {
    err = std::string("missing tensor: ") + name;
    return false;
  }
  const std::size_t i = (std::size_t)(info - ld_.meta().tensors.data());
  std::vector<std::uint8_t> bytes;
  if (!ld_.load_tensor(i, bytes, err)) return false;
  if (hipMalloc(&t.ptr, bytes.size()) != hipSuccess) {
    err = std::string("hipMalloc failed for ") + name;
    return false;
  }
  if (hipMemcpy(t.ptr, bytes.data(), bytes.size(), hipMemcpyHostToDevice) != hipSuccess) {
    err = std::string("hipMemcpy failed for ") + name;
    return false;
  }
  t.bytes = bytes.size();
  t.dt = ld_.dtype(i);
  t.dim0 = info->dims.size() > 0 ? (int)info->dims[0] : 0;
  t.dim1 = info->dims.size() > 1 ? (int)info->dims[1] : 0;
  t.valid = true;
  return true;
}

inline bool Graph::alloc(float *&p, std::size_t n, std::string &err) {
  if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    err = "hipMalloc failed";
    return false;
  }
  if (hipMemset(p, 0, n * sizeof(float)) != hipSuccess) {
    err = "hipMemset failed";
    return false;
  }
  return true;
}

inline bool Graph::init(int max_ctx, KvType kv_k, KvType kv_v, std::string &err) {
  max_ctx_ = max_ctx;
  kv_k_ = kv_k;
  kv_v_ = kv_v;
  const int L = n_layer();
  const int E = n_embd();
  const int HD = head_dim(), NH = n_head(), NKV = n_head_kv();
  const int S = (int)cfg_.ssm_state_size;
  const int nkh = (int)cfg_.ssm_group_count;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int d_inner = nvh * S;
  const int key_dim = nkh * S;
  const int chan = 2 * key_dim + d_inner;
  const int F = (int)cfg_.feed_forward_length;
  const int K = (int)cfg_.ssm_conv_kernel;

  // The graph trusts the config's shape fields, so validate the inventory first
  // (names *and* dims of all 866 tensors) and enforce the reference's own
  // assertions before touching any buffer (review M4/M5/m6/m7).
  if (!validate_qwen35_layout(ld_, cfg_, err)) return false;
  if (cfg_.key_length != cfg_.value_length) {
    err = "key_length != value_length is not supported (llama.cpp asserts equality)";
    return false;
  }
  if (NH <= 0 || NKV <= 0 || NH % NKV != 0) {
    err = "head_count must be a positive multiple of head_count_kv";
    return false;
  }
  if (n_rot() <= 0 || n_rot() > HD || n_rot() % 2 != 0) {
    err = "rope.dimension_count must be even and <= key_length";
    return false;
  }
  if (max_ctx <= 0) {
    err = "max_ctx must be positive";
    return false;
  }

  w_.resize(L);
  int n_recr = 0;
  for (int il = 0; il < L; ++il) {
    w_[il].recr = is_recr(il);
    char b[128];
    auto upn = [&](const char *fmt, GpuTensor &t) {
      std::snprintf(b, sizeof(b), fmt, il);
      if (!up(b, t, err)) {
        err = "layer " + std::to_string(il) + ": " + err;
        return false;
      }
      return true;
    };
    auto upn_f32 = [&](const char *fmt, GpuTensor &t) {
      std::snprintf(b, sizeof(b), fmt, il);
      if (!up_f32(b, t, err)) {
        err = "layer " + std::to_string(il) + ": " + err;
        return false;
      }
      return true;
    };
    if (!upn_f32("blk.%d.attn_norm.weight", w_[il].attn_norm)) return false;
    // the GGUF name is `post_attention_norm` (llama.cpp maps it to attn_post_norm)
    if (!upn_f32("blk.%d.post_attention_norm.weight", w_[il].attn_post_norm)) return false;
    if (!upn("blk.%d.ffn_gate.weight", w_[il].ffn_gate)) return false;
    if (!upn("blk.%d.ffn_up.weight", w_[il].ffn_up)) return false;
    if (!upn("blk.%d.ffn_down.weight", w_[il].ffn_down)) return false;
    if (w_[il].recr) {
      ++n_recr;
      if (!upn("blk.%d.attn_qkv.weight", w_[il].attn_qkv)) return false;
      if (!upn("blk.%d.attn_gate.weight", w_[il].attn_gate)) return false;
      if (!upn_f32("blk.%d.ssm_conv1d.weight", w_[il].ssm_conv1d)) return false;
      if (!upn_f32("blk.%d.ssm_a", w_[il].ssm_a)) return false;
      if (!upn_f32("blk.%d.ssm_dt.bias", w_[il].ssm_dt)) return false;
      if (!upn("blk.%d.ssm_beta.weight", w_[il].ssm_beta)) return false;
      if (!upn("blk.%d.ssm_alpha.weight", w_[il].ssm_alpha)) return false;
      if (!upn_f32("blk.%d.ssm_norm.weight", w_[il].ssm_norm)) return false;
      if (!upn("blk.%d.ssm_out.weight", w_[il].ssm_out)) return false;
    } else {
      if (!upn("blk.%d.attn_q.weight", w_[il].attn_q)) return false;
      if (!upn("blk.%d.attn_k.weight", w_[il].attn_k)) return false;
      if (!upn("blk.%d.attn_v.weight", w_[il].attn_v)) return false;
      if (!upn("blk.%d.attn_output.weight", w_[il].attn_output)) return false;
      if (!upn_f32("blk.%d.attn_q_norm.weight", w_[il].attn_q_norm)) return false;
      if (!upn_f32("blk.%d.attn_k_norm.weight", w_[il].attn_k_norm)) return false;
    }
  }
  if (!up("token_embd.weight", tok_embd_, err)) return false;
  if (!up_f32("output_norm.weight", output_norm_, err)) return false;
  // tied embeddings are legal in the reference (it reuses token_embd when
  // output.weight is absent); the UD files always carry it, so be loud instead
  // of silent (review n14)
  if (!up("output.weight", output_, err)) return false;
  // forward_tokens indexes token_embd rows by token id and forward() takes
  // n_vocab from output_, so a token_embd that does not cover the same rows
  // would read out of bounds (review M4)
  if (tok_embd_.dim0 != E || tok_embd_.dim1 != output_.dim1) {
    err = "token_embd.weight must be [n_embd, n_vocab], same n_vocab as output.weight";
    return false;
  }

  const int n_attn = L - n_recr;
  if (!alloc(d_x_, E, err) || !alloc(d_xn_, E, err) || !alloc(d_proj_, NH * 2 * HD, err) ||
      !alloc(d_ffnout_, E, err) || !alloc(d_attnout_, NH * HD, err) ||
      !alloc(d_attngate_, NH * HD, err) || !alloc(d_qkv_, chan, err) || !alloc(d_conv_, chan, err) ||
      !alloc(d_z_, d_inner, err) || !alloc(d_alpha_, nvh, err) || !alloc(d_beta_, nvh, err) ||
      !alloc(d_gate_, nvh, err) || !alloc(d_ffn_a_, F, err) || !alloc(d_ffn_b_, F, err)) {
    return false;
  }
  // KV cache: `n_attn` caches of max_ctx rows, each row NKV heads of HD elements
  // in the selected storage type.
  const std::size_t kv_bytes =
      (std::size_t)max_ctx * NKV * (std::size_t)kv_row_bytes(kv_k_, HD);
  kv_bytes_ = kv_bytes;
  if (hipMalloc(&d_k_, (std::size_t)n_attn * kv_bytes) != hipSuccess ||
      hipMalloc(&d_v_, (std::size_t)n_attn * kv_bytes) != hipSuccess) {
    err = "hipMalloc failed (kv cache)";
    return false;
  }
  if (!alloc(d_kstage_, NKV * HD, err) || !alloc(d_vstage_, NKV * HD, err)) return false;
  // M8 batch buffers. The activation block matrix is sized for the largest
  // reduction dimension in the model (ffn_down: F elements per row).
  const std::size_t aq_per_row = ((std::size_t)F + 31) / 32;
  aq_blocks_per_row_ = aq_per_row;
  {
    const int NH_ = NH, HD_ = HD, NKV_ = NKV;
    const auto balloc = [&](float *&p, std::size_t n) {
      return hipMalloc(&p, (std::size_t)kMaxBatch * n * sizeof(float)) == hipSuccess;
    };
    if (!balloc(d_xb_, E) || !balloc(d_xnb_, E) || !balloc(d_projb_, E) ||
        !balloc(d_qb_, (std::size_t)NH_ * 2 * HD_) || !balloc(d_kb_, (std::size_t)NKV_ * HD_) ||
        !balloc(d_vb_, (std::size_t)NKV_ * HD_) || !balloc(d_attnb_, (std::size_t)NH_ * HD_) ||
        !balloc(d_gateb_, (std::size_t)NH_ * HD_) || !balloc(d_ffnab_, F) ||
        !balloc(d_ffnbb_, F) || !balloc(d_qkvb_, chan) || !balloc(d_zb_, d_inner) ||
        !balloc(d_convb_, chan) || !balloc(d_alphab_, nvh) || !balloc(d_betab_, nvh) ||
        !balloc(d_gate2b_, nvh)) {
      err = "hipMalloc failed for the batch buffers";
      return false;
    }
    if (hipMalloc(&d_aqb_, (std::size_t)kMaxBatch * aq_per_row * sizeof(block_q8_1)) !=
        hipSuccess) {
      err = "hipMalloc failed for the batch activation blocks";
      return false;
    }
    if (hipMalloc(&d_posb_, (std::size_t)kMaxBatch * sizeof(int)) != hipSuccess) {
      err = "hipMalloc failed for the batch positions";
      return false;
    }
  }

  const std::size_t part_bytes = attn_partial_bytes(NH, HD, kAttnMaxSplits);
  if (hipMalloc(&d_attn_partial_, part_bytes) != hipSuccess) {
    err = "hipMalloc attn partial failed";
    return false;
  }
  if (!alloc(d_state_, (std::size_t)n_recr * nvh * S * S, err)) return false;
  if (!alloc(d_convst_, (std::size_t)n_recr * (K - 1) * chan, err)) return false;
  if (hipMalloc(&d_pos_, sizeof(int)) != hipSuccess) {
    err = "hipMalloc failed (pos)";
    return false;
  }

  // ssm_a is read with nvh floats (the reference broadcasts it), so a
  // single-element ssm_a would read out of bounds (review M4)
  if (w_[0].recr && w_[0].ssm_a.dim0 != nvh) {
    err = "ssm_a must have ssm_time_step_rank elements";
    return false;
  }
  q8_blocks_ = (std::size_t)(F / QK8_1) + 8;  // ffn_down has the largest reduction
  if (hipMalloc(&d_q8_, q8_blocks_ * sizeof(block_q8_1)) != hipSuccess) {
    err = "hipMalloc failed (q8 scratch)";
    return false;
  }
  return true;
}

inline bool Graph::proj(const GpuTensor &w, const float *d_x, float *d_y, int nrows, int ncols,
                        std::string &err) {
  // The caller passes the shape it expects; verify it against the tensor that
  // is actually in VRAM, otherwise a wrong width silently reads past the
  // weights (review M5).
  if (w.dim0 != ncols || w.dim1 != nrows ||
      w.bytes != tensor_bytes(w.dt, (std::uint64_t)nrows * (std::uint64_t)ncols)) {
    err = "weight shape mismatch (expected " + std::to_string(nrows) + "x" +
          std::to_string(ncols) + ", tensor is " + std::to_string(w.dim1) + "x" +
          std::to_string(w.dim0) + ")";
    return false;
  }
  const int nb = ncols / QK8_1;
  if (ncols % QK8_1 != 0 || (std::size_t)nb > q8_blocks_) {
    err = "bad reduction dim for q8 scratch";
    return false;
  }
  quantize_q8_1_kernel<<<(nb + 3) / 4, 128>>>(d_x, (block_q8_1 *)d_q8_, nb);
  if (hipGetLastError() != hipSuccess) {
    err = "quantize_q8_1 launch failed";
    return false;
  }
  if (!matvec_launch((int)w.dt, w.ptr, (const block_q8_1 *)d_q8_, d_y, nrows, ncols, nullptr)) {
    err = "matvec_launch failed";
    return false;
  }
  return true;
}

inline bool Graph::proj_qq(const GpuTensor &w, float *d_y, int nrows, int ncols,
                           std::string &err) {
  if (w.dim0 != ncols || w.dim1 != nrows ||
      w.bytes != tensor_bytes(w.dt, (std::uint64_t)nrows * (std::uint64_t)ncols)) {
    err = "weight shape mismatch in proj_qq";
    return false;
  }
  const int nb = ncols / QK8_1;
  if (ncols % QK8_1 != 0 || (std::size_t)nb > q8_blocks_) {
    err = "bad reduction dim for q8 scratch (proj_qq)";
    return false;
  }
  if (!matvec_launch((int)w.dt, w.ptr, (const block_q8_1 *)d_q8_, d_y, nrows, ncols, nullptr)) {
    err = "matvec_launch failed (proj_qq)";
    return false;
  }
  return true;
}

// Residual add, with the optional diagnostic perturbation of the layer output.
inline bool Graph::add_residual(float *d_src, float *d_dst, int n, std::string &err) {
  if (noise_rel_ > 0.0f) {
    perturb_kernel<<<(unsigned)((n + 255) / 256), 256>>>(d_src, n, noise_rel_, noise_seed_++,
                                                         noise_coherent_ ? 1 : 0);
    if (hipGetLastError() != hipSuccess) {
      err = "perturb launch failed";
      return false;
    }
  }
  return add_launch(d_src, d_dst, d_dst, n);
}

// ---------------------------------------------------------------------------
inline bool Graph::full_attn(int il, int t, int pos, std::string &err) {
  const int E = n_embd(), HD = head_dim(), NH = n_head(), NKV = n_head_kv();
  const LayerW &L = w_[il];
  if (!rms_norm_launch(d_x_, (const float *)L.attn_norm.ptr, d_xn_, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "attn_norm launch failed";
    return false;
  }
  emit("attn_norm", il, d_xn_, E);
  if (!proj(L.attn_q, d_xn_, d_proj_, NH * 2 * HD, E, err)) return false;
  // k and v read the same activation as q: reuse its q8_1 blocks
  if (!proj_qq(L.attn_k, d_kstage_, NKV * HD, E, err)) return false;
  if (!proj_qq(L.attn_v, d_vstage_, NKV * HD, E, err)) return false;
  emit("Qcur_full", il, d_proj_, (std::int64_t)NH * 2 * HD);
  emit("Vcur", il, d_vstage_, (std::int64_t)NKV * HD);

  if (!deinterleave_q_gate_launch(d_proj_, d_attnout_, d_attngate_, NH, HD)) {
    err = "deinterleave launch failed";
    return false;
  }
  if (!rms_norm_launch(d_attnout_, (const float *)L.attn_q_norm.ptr, d_attnout_, NH, HD,
                       (float)cfg_.rms_norm_eps)) {
    err = "q_norm launch failed";
    return false;
  }
  emit("Qcur_normed", il, d_attnout_, (std::int64_t)NH * HD);
  if (!rms_norm_launch(d_kstage_, (const float *)L.attn_k_norm.ptr, d_kstage_, NKV, HD,
                       (float)cfg_.rms_norm_eps)) {
    err = "k_norm launch failed";
    return false;
  }
  emit("Kcur_normed", il, d_kstage_, (std::int64_t)NKV * HD);
  if (hipMemcpy(d_pos_, &pos, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
    err = "position upload failed";
    return false;
  }
  const float base = (float)cfg_.rope_freq_base;
  if (!rope_launch(d_attnout_, 1, NH, HD, n_rot(), base, d_pos_)) return false;
  if (!rope_launch(d_kstage_, 1, NKV, HD, n_rot(), base, d_pos_)) return false;
  emit("Qcur", il, d_attnout_, (std::int64_t)NH * HD);
  emit("Kcur", il, d_kstage_, (std::int64_t)NKV * HD);
  emit("gate_reshaped", il, d_attngate_, (std::int64_t)NH * HD);

  // Store the rotated K and the raw V into the (possibly quantized) cache.
  if (!kv_write(il, pos, d_kstage_, d_vstage_, err)) return false;
  const char *kc = (const char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_;
  const char *vc = (const char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_;

  const float scale = 1.0f / std::sqrt((float)HD);
  const int n_keys = pos + 1;
  const int splits = attn_splits_for(n_keys);
  attn_splits_last_ = splits;
  if (splits > 1) {
    if (!attn_launch_split(d_attnout_, kc, vc, d_attnout_, d_attn_partial_, pos, NH, NKV, HD,
                           scale, kv_k_, kv_v_, splits)) {
      err = "attn split launch failed";
      return false;
    }
  } else if (!attn_launch(d_attnout_, kc, vc, d_attnout_, pos, NH, NKV, HD, scale, kv_k_, kv_v_)) {
    err = "attn launch failed";
    return false;
  }
  emit("attn_pregate", il, d_attnout_, (std::int64_t)NH * HD);
  if (!unary_launch(d_attngate_, d_attngate_, NH * HD, UnOp::Sigmoid)) return false;
  emit("gate_sigmoid", il, d_attngate_, (std::int64_t)NH * HD);
  if (!mul_launch(d_attnout_, d_attngate_, d_attnout_, NH * HD)) return false;
  emit("attn_gated", il, d_attnout_, (std::int64_t)NH * HD);

  if (!proj(L.attn_output, d_attnout_, d_ffnout_, E, NH * HD, err)) return false;
  emit("attn_output", il, d_ffnout_, E);
  if (!add_residual(d_ffnout_, d_x_, E, err)) return false;
  return true;
}

// Quantize the K row (already RoPE'd) and the V row into this layer's cache.
inline bool Graph::kv_write(int il, int t, const float *d_ksrc, const float *d_vsrc,
                            std::string &err) {
  const int HD = head_dim(), NKV = n_head_kv();
  const char *base = (const char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_;
  char *krow = (char *)base + (std::size_t)t * NKV * kv_row_bytes(kv_k_, HD);
  char *vrow = (char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_ +
               (std::size_t)t * NKV * kv_row_bytes(kv_v_, HD);
  for (int h = 0; h < NKV; ++h) {
    if (!kv_store_row_launch(kv_k_, d_ksrc + (std::size_t)h * HD,
                             krow + (std::size_t)h * kv_row_bytes(kv_k_, HD), HD) ||
        !kv_store_row_launch(kv_v_, d_vsrc + (std::size_t)h * HD,
                             vrow + (std::size_t)h * kv_row_bytes(kv_v_, HD), HD)) {
      err = "kv_store_row launch failed";
      return false;
    }
  }
  return true;
}

// ---------------------------------------------------------------------------
inline bool Graph::gdn_layer(int il, int t, std::string &err) {
  const int E = n_embd();
  const LayerW &L = w_[il];
  const int K = (int)cfg_.ssm_conv_kernel;
  const int S = (int)cfg_.ssm_state_size;
  const int nkh = (int)cfg_.ssm_group_count;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int d_inner = nvh * S;
  const int key_dim = nkh * S;
  const int chan = 2 * key_dim + d_inner;

  int slot = 0;
  for (int i = 0; i < il; ++i) slot += is_recr(i) ? 1 : 0;
  float *state = d_state_ + (std::size_t)slot * nvh * S * S;
  float *convst = d_convst_ + (std::size_t)slot * (K - 1) * chan;

  if (!rms_norm_launch(d_x_, (const float *)L.attn_norm.ptr, d_xn_, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "attn_norm launch failed";
    return false;
  }
  emit("attn_norm", il, d_xn_, E);
  if (!proj(L.attn_qkv, d_xn_, d_qkv_, chan, E, err)) return false;
  // all four GDN projections read d_xn_ unchanged: one quantization serves all
  if (!proj_qq(L.attn_gate, d_z_, d_inner, E, err)) return false;
  if (!proj_qq(L.ssm_beta, d_beta_, nvh, E, err)) return false;
  if (!proj_qq(L.ssm_alpha, d_alpha_, nvh, E, err)) return false;
  emit("linear_attn_qkv_mixed", il, d_qkv_, chan);
  emit("z", il, d_z_, d_inner);
  emit("beta", il, d_beta_, nvh);
  emit("alpha", il, d_alpha_, nvh);

  if (!unary_launch(d_beta_, d_beta_, nvh, UnOp::Sigmoid)) return false;
  emit("beta_sigmoid", il, d_beta_, nvh);
  if (!add_launch(d_alpha_, (const float *)L.ssm_dt.ptr, d_alpha_, nvh)) return false;
  if (!unary_launch(d_alpha_, d_gate_, nvh, UnOp::Softplus)) return false;
  emit("a_softplus", il, d_gate_, nvh);
  if (!mul_launch(d_gate_, (const float *)L.ssm_a.ptr, d_gate_, nvh)) return false;
  emit("gate", il, d_gate_, nvh);

  if (!conv1d_state_launch(d_qkv_, (const float *)L.ssm_conv1d.ptr, d_conv_, convst, chan, K)) {
    err = "conv1d launch failed";
    return false;
  }
  emit("conv_output_silu", il, d_conv_, chan);
  float *q_c = d_conv_;
  float *k_c = d_conv_ + key_dim;
  float *v_c = d_conv_ + 2 * key_dim;
  emit("q_conv", il, q_c, key_dim);
  emit("k_conv", il, k_c, key_dim);
  // the eval callback registers the same tensor under both names and keeps the
  // last one, so the dump only has `v_conv_predelta`
  emit("v_conv_predelta", il, v_c, d_inner);
  if (!l2_norm_launch(q_c, q_c, nkh, S, (float)cfg_.rms_norm_eps)) return false;
  if (!l2_norm_launch(k_c, k_c, nkh, S, (float)cfg_.rms_norm_eps)) return false;

  if (cb_) emit("state_predelta", il, state, (std::int64_t)nvh * S * S);
  if (!delta_rule_launch(q_c, k_c, v_c, d_gate_, d_beta_, state, v_c, nvh, nkh, S)) {
    err = "delta_rule launch failed";
    return false;
  }
  if (cb_) emit("new_state", il, state, (std::int64_t)nvh * S * S);
  if (!rms_norm_launch(v_c, (const float *)L.ssm_norm.ptr, v_c, nvh, S,
                       (float)cfg_.rms_norm_eps)) {
    err = "ssm_norm launch failed";
    return false;
  }
  if (!unary_launch(d_z_, d_z_, d_inner, UnOp::Silu)) return false;
  if (!mul_launch(v_c, d_z_, v_c, d_inner)) return false;
  emit("final_output", il, v_c, d_inner);

  if (!proj(L.ssm_out, v_c, d_ffnout_, E, d_inner, err)) return false;
  emit("linear_attn_out", il, d_ffnout_, E);
  if (!add_residual(d_ffnout_, d_x_, E, err)) return false;
  (void)t;
  return true;
}

// ---------------------------------------------------------------------------
inline bool Graph::ffn(int il, std::string &err) {
  const int E = n_embd();
  const int F = (int)cfg_.feed_forward_length;
  const LayerW &L = w_[il];
  if (!proj(L.ffn_gate, d_xn_, d_ffn_a_, F, E, err)) return false;  // d_xn_ holds attn_post_norm
  if (!unary_launch(d_ffn_a_, d_ffn_a_, F, UnOp::Silu)) return false;
  // up reads the same activation as gate (silu only touched the output)
  if (!proj_qq(L.ffn_up, d_ffn_b_, F, E, err)) return false;
  if (!mul_launch(d_ffn_a_, d_ffn_b_, d_ffn_a_, F)) return false;
  if (!proj(L.ffn_down, d_ffn_a_, d_ffnout_, E, F, err)) return false;
  emit("ffn_out", il, d_ffnout_, E);
  if (!add_residual(d_ffnout_, d_x_, E, err)) return false;
  return true;
}


// ---------------------------------------------------------------------------
// M8: batched prefill. See the declaration for the contract; the short version is
// "layer-major, projections batched, everything sequential stays sequential".
// ---------------------------------------------------------------------------
inline bool Graph::quantize_batch(const float *d_x, int ncols, int n, std::string &err) {
  if (ncols % QK8_1 != 0) {
    err = "bad reduction dim for the batch q8 scratch";
    return false;
  }
  const std::int64_t nb = ncols / QK8_1;
  if ((std::size_t)nb > aq_blocks_per_row_) {
    err = "batch q8 scratch too small";
    return false;
  }
  if (!quantize_q8_1_batch_launch(d_x, d_aqb_, nb, n, ncols)) {
    err = "batch quantize launch failed";
    return false;
  }
  return true;
}

inline bool Graph::proj_batch(const GpuTensor &w, const float *d_x, float *d_y, int nrows,
                              int ncols, int n, bool act_ready, std::string &err) {
  if (w.dim0 != ncols || w.dim1 != nrows ||
      w.bytes != tensor_bytes(w.dt, (std::uint64_t)nrows * (std::uint64_t)ncols)) {
    err = "batch weight shape mismatch";
    return false;
  }
  if (n < 2 || n > kMaxBatch || !batch_supported(n)) {
    err = "unsupported batch size " + std::to_string(n);
    return false;
  }
  if (!act_ready && !quantize_batch(d_x, ncols, n, err)) return false;
  const std::int64_t act_stride = ncols / QK8_1;
  if (!matvec_launch_batch((int)w.dt, w.ptr, d_aqb_, d_y, nrows, ncols, act_stride, n, nullptr)) {
    err = "matvec_launch_batch failed";
    return false;
  }
  return true;
}

// One layer for the whole batch. `d_xb_` holds the N token rows on entry and the
// updated rows on exit; the KV cache and the GDN state advance in token order.
inline bool Graph::forward_batch_layer(int il, int n, int pos0, std::string &err) {
  const int E = n_embd();
  const int F = (int)cfg_.feed_forward_length;
  const int HD = head_dim(), NH = n_head(), NKV = n_head_kv();
  const LayerW &L = w_[il];
  const float eps = (float)cfg_.rms_norm_eps;
  const float base = (float)cfg_.rope_freq_base;

  if (!rms_norm_launch(d_xb_, (const float *)L.attn_norm.ptr, d_xnb_, n, E, eps)) {
    err = "batch attn_norm launch failed";
    return false;
  }
  emit("attn_norm", il, d_xnb_, (std::int64_t)n * E);

  if (!L.recr) {
    if (!proj_batch(L.attn_q, d_xnb_, d_qb_, NH * 2 * HD, E, n, false, err)) return false;
    // k and v share the same activation: quantize once, reuse for both
    if (!quantize_batch(d_xnb_, E, n, err)) return false;
    if (!proj_batch(L.attn_k, d_xnb_, d_kb_, NKV * HD, E, n, true, err)) return false;
    if (!proj_batch(L.attn_v, d_xnb_, d_vb_, NKV * HD, E, n, true, err)) return false;
    // per token: split q/gate, QK-norm, RoPE, cache write, attention
    for (int t = 0; t < n; ++t) {
      float *q = d_qb_ + (std::size_t)t * NH * 2 * HD;
      float *k = d_kb_ + (std::size_t)t * NKV * HD;
      float *v = d_vb_ + (std::size_t)t * NKV * HD;
      float *an = d_attnb_ + (std::size_t)t * NH * HD;
      float *ag = d_gateb_ + (std::size_t)t * NH * HD;
      const int pos = pos0 + t;
      if (!deinterleave_q_gate_launch(q, an, ag, NH, HD)) {
        err = "batch deinterleave launch failed";
        return false;
      }
      if (!rms_norm_launch(an, (const float *)L.attn_q_norm.ptr, an, NH, HD, eps) ||
          !rms_norm_launch(k, (const float *)L.attn_k_norm.ptr, k, NKV, HD, eps)) {
        err = "batch qk norm launch failed";
        return false;
      }
      // positions were uploaded once for the whole batch by forward_batch()
      if (!rope_launch(an, 1, NH, HD, n_rot(), base, d_posb_ + t) ||
          !rope_launch(k, 1, NKV, HD, n_rot(), base, d_posb_ + t)) {
        err = "batch rope launch failed";
        return false;
      }
      if (!kv_write(il, pos, k, v, err)) return false;
      const char *kc = (const char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_;
      const char *vc = (const char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_;
      const int splits = attn_splits_for(pos + 1);
      attn_splits_last_ = splits;
      const float scale = 1.0f / std::sqrt((float)HD);
      if (splits > 1) {
        if (!attn_launch_split(an, kc, vc, an, d_attn_partial_, pos, NH, NKV, HD, scale, kv_k_,
                               kv_v_, splits)) {
          err = "batch attn split launch failed";
          return false;
        }
      } else if (!attn_launch(an, kc, vc, an, pos, NH, NKV, HD, scale, kv_k_, kv_v_)) {
        err = "batch attn launch failed";
        return false;
      }
      if (!unary_launch(ag, ag, NH * HD, UnOp::Sigmoid) ||
          !mul_launch(an, ag, an, NH * HD)) {
        err = "batch attn gate launch failed";
        return false;
      }
    }
    if (!proj_batch(L.attn_output, d_attnb_, d_projb_, E, NH * HD, n, false, err)) return false;
  } else {
    const int S = (int)cfg_.ssm_state_size;
    const int nkh = (int)cfg_.ssm_group_count;
    const int nvh = (int)cfg_.ssm_time_step_rank;
    const int d_inner = nvh * S;
    const int key_dim = nkh * S;
    const int chan = 2 * key_dim + d_inner;
    const int K = (int)cfg_.ssm_conv_kernel;
    int slot = 0;
    for (int i = 0; i < il; ++i) slot += is_recr(i) ? 1 : 0;
    float *state = d_state_ + (std::size_t)slot * nvh * S * S;
    float *convst = d_convst_ + (std::size_t)slot * (K - 1) * chan;

    if (!proj_batch(L.attn_qkv, d_xnb_, d_qkvb_, chan, E, n, false, err)) return false;
    if (!quantize_batch(d_xnb_, E, n, err)) return false;
    if (!proj_batch(L.attn_gate, d_xnb_, d_zb_, d_inner, E, n, true, err)) return false;
    if (!proj_batch(L.ssm_beta, d_xnb_, d_betab_, nvh, E, n, true, err)) return false;
    if (!proj_batch(L.ssm_alpha, d_xnb_, d_alphab_, nvh, E, n, true, err)) return false;
    // The recurrence itself is sequential: token order, same kernels as the
    // per-token path, one token's slices at a time.
    for (int t = 0; t < n; ++t) {
      float *qkv = d_qkvb_ + (std::size_t)t * chan;
      float *z = d_zb_ + (std::size_t)t * d_inner;
      float *beta = d_betab_ + (std::size_t)t * nvh;
      float *alpha = d_alphab_ + (std::size_t)t * nvh;
      float *conv = d_convb_ + (std::size_t)t * chan;
      float *g = d_gate2b_ + (std::size_t)t * nvh;
      if (!unary_launch(beta, beta, nvh, UnOp::Sigmoid)) return false;
      if (!add_launch(alpha, (const float *)L.ssm_dt.ptr, alpha, nvh)) return false;
      if (!unary_launch(alpha, g, nvh, UnOp::Softplus)) return false;
      if (!mul_launch(g, (const float *)L.ssm_a.ptr, g, nvh)) return false;
      if (!conv1d_state_launch(qkv, (const float *)L.ssm_conv1d.ptr, conv, convst, chan, K)) {
        err = "batch conv1d launch failed";
        return false;
      }
      float *q_c = conv;
      float *k_c = conv + key_dim;
      float *v_c = conv + 2 * key_dim;
      if (!l2_norm_launch(q_c, q_c, nkh, S, eps) || !l2_norm_launch(k_c, k_c, nkh, S, eps)) {
        err = "batch l2 norm launch failed";
        return false;
      }
      if (!delta_rule_launch(q_c, k_c, v_c, g, beta, state, v_c, nvh, nkh, S)) {
        err = "batch delta rule launch failed";
        return false;
      }
      if (!rms_norm_launch(v_c, (const float *)L.ssm_norm.ptr, v_c, nvh, S, eps)) {
        err = "batch ssm_norm launch failed";
        return false;
      }
      if (!unary_launch(z, z, d_inner, UnOp::Silu) || !mul_launch(v_c, z, v_c, d_inner)) {
        err = "batch gdn output launch failed";
        return false;
      }
      // v_c lives inside this token's conv buffer (not contiguous across the
      // batch), so copy it into d_zb_[t] -- free now that the gate was consumed --
      // which is what the batched ssm_out projection reads. Same values, so the
      // result stays bit-identical to the per-token path.
      if (hipMemcpy(z, v_c, (std::size_t)d_inner * sizeof(float),
                    hipMemcpyDeviceToDevice) != hipSuccess) {
        err = "batch gdn copy failed";
        return false;
      }
    }
    if (!proj_batch(L.ssm_out, d_zb_, d_projb_, E, d_inner, n, false, err)) return false;
  }

  // residual + post-attention norm
  if (!add_launch(d_projb_, d_xb_, d_xb_, (std::int64_t)n * E)) {
    err = "batch residual launch failed";
    return false;
  }
  if (!rms_norm_launch(d_xb_, (const float *)L.attn_post_norm.ptr, d_xnb_, n, E, eps)) {
    err = "batch post_attn_norm launch failed";
    return false;
  }

  // FFN (gate and up share the activation)
  if (!proj_batch(L.ffn_gate, d_xnb_, d_ffnab_, F, E, n, false, err)) return false;
  if (!proj_batch(L.ffn_up, d_xnb_, d_ffnbb_, F, E, n, true, err)) return false;
  if (!unary_launch(d_ffnab_, d_ffnab_, (std::int64_t)n * F, UnOp::Silu)) return false;
  if (!mul_launch(d_ffnab_, d_ffnbb_, d_ffnab_, (std::int64_t)n * F)) return false;
  if (!proj_batch(L.ffn_down, d_ffnab_, d_projb_, E, F, n, false, err)) return false;
  if (!add_launch(d_projb_, d_xb_, d_xb_, (std::int64_t)n * E)) {
    err = "batch ffn residual launch failed";
    return false;
  }
  emit("l_out", il, d_xb_, (std::int64_t)n * E);
  return true;
}

// ---------------------------------------------------------------------------

inline bool Graph::forward_batch(const std::vector<std::int32_t> &tokens, int start_pos,
                                 std::vector<float> &hidden, std::vector<float> &logits,
                                 std::string &err) {
  const int E = n_embd();
  const int n = (int)tokens.size();
  if (n < 2 || n > kMaxBatch) {
    err = "forward_batch needs 2.." + std::to_string(kMaxBatch) + " tokens";
    return false;
  }
  if (!batch_supported(n)) {
    err = "no batched kernel for n=" + std::to_string(n);
    return false;
  }
  const std::int32_t n_vocab = (std::int32_t)tok_embd_.dim1;
  for (std::int32_t tk : tokens) {
    if (tk < 0 || tk >= n_vocab) {
      err = "token id " + std::to_string(tk) + " out of range [0, " + std::to_string(n_vocab) + ")";
      return false;
    }
  }
  if (start_pos < 0 || (std::size_t)start_pos + (std::size_t)n > (std::size_t)max_ctx_) {
    err = "positions exceed the allocated context (" + std::to_string(max_ctx_) + ")";
    return false;
  }

  // positions for the whole batch, one upload (the per-token path does one 4-byte
  // synchronous copy per RoPE call)
  int pos_host[kMaxBatch];
  for (int t = 0; t < n; ++t) pos_host[t] = start_pos + t;
  if (hipMemcpy(d_posb_, pos_host, (std::size_t)n * sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess) {
    err = "batch position upload failed";
    return false;
  }

  // embeddings: one dequantized row per token
  const std::size_t emb_row = (std::size_t)tensor_bytes(tok_embd_.dt, (std::uint64_t)E);
  for (int t = 0; t < n; ++t) {
    const char *src = (const char *)tok_embd_.ptr + (std::size_t)tokens[t] * emb_row;
    if (!dequant_row_launch(tok_embd_.dt, src, d_xb_ + (std::size_t)t * E, E)) {
      err = "batch embedding dequant failed";
      return false;
    }
  }

  for (int il = 0; il < debug_layer_limit(); ++il) {
    if (!forward_batch_layer(il, n, start_pos, err)) {
      err = "layer " + std::to_string(il) + ": " + err;
      return false;
    }
  }

  // final norm + LM head for the LAST token only (that is what generation needs;
  // the head costs one full weight read per call)
  float *xlast = d_xb_ + (std::size_t)(n - 1) * E;
  if (!rms_norm_launch(xlast, (const float *)output_norm_.ptr, xlast, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "batch output_norm launch failed";
    return false;
  }
  if (!readback(xlast, E, hidden)) {
    err = "batch hidden readback failed";
    return false;
  }

  const int n_vocab_i = output_.dim1;
  if (d_logits_ == nullptr) {
    if (hipMalloc(&d_logits_, (std::size_t)n_vocab_i * sizeof(float)) != hipSuccess) {
      err = "hipMalloc logits failed";
      return false;
    }
  }
  if (!proj(output_, xlast, d_logits_, n_vocab_i, E, err)) return false;
  logits.resize(n_vocab_i);
  if (hipMemcpy(logits.data(), d_logits_, (std::size_t)n_vocab_i * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    err = "batch logits readback failed";
    return false;
  }
  return true;
}

inline bool Graph::forward(const std::vector<float> &embeddings, std::vector<float> &hidden,
                           std::vector<float> &logits, std::string &err) {
  return forward(embeddings, 0, hidden, logits, err);
}

inline bool Graph::forward(const std::vector<float> &embeddings, int start_pos,
                           std::vector<float> &hidden, std::vector<float> &logits,
                           std::string &err) {
  const int E = n_embd();
  if (embeddings.empty() || embeddings.size() % (std::size_t)E != 0) {
    err = "embeddings must hold whole rows of n_embd floats";
    return false;
  }
  return forward_run(embeddings.size() / (std::size_t)E, start_pos, embeddings.data(), nullptr,
                     hidden, logits, err);
}

inline bool Graph::forward_tokens(const std::vector<std::int32_t> &tokens, int start_pos,
                                  std::vector<float> &hidden, std::vector<float> &logits,
                                  std::string &err) {
  if (tokens.empty()) {
    err = "token list must not be empty";
    return false;
  }
  const std::int32_t n_vocab = (std::int32_t)tok_embd_.dim1;
  for (std::int32_t tk : tokens) {
    if (tk < 0 || tk >= n_vocab) {
      err = "token id " + std::to_string(tk) + " out of range [0, " + std::to_string(n_vocab) + ")";
      return false;
    }
  }
  return forward_run(tokens.size(), start_pos, nullptr, &tokens, hidden, logits, err);
}

inline bool Graph::forward_run(std::size_t n_tokens, int start_pos, const float *host_emb,
                               const std::vector<std::int32_t> *toks, std::vector<float> &hidden,
                               std::vector<float> &logits, std::string &err) {
  const int E = n_embd();
  if (n_tokens == 0 || (host_emb == nullptr) == (toks == nullptr)) {
    err = "forward_run needs exactly one source of embeddings";
    return false;
  }
  if (start_pos < 0 || (std::size_t)start_pos + n_tokens > (std::size_t)max_ctx_) {
    err = "positions exceed the allocated context (" + std::to_string(max_ctx_) + ")";
    return false;
  }
  // Row stride of one token_embd row in its own (possibly quantized) layout.
  const std::size_t emb_row = (std::size_t)tensor_bytes(tok_embd_.dt, (std::uint64_t)E);

  using clock = std::chrono::steady_clock;
  const clock::time_point t_queue0 = clock::now();
  for (std::size_t t = 0; t < n_tokens; ++t) {
    cur_token_ = (int)t;
    if (toks != nullptr) {
      const char *src = (const char *)tok_embd_.ptr + (std::size_t)(*toks)[t] * emb_row;
      if (!dequant_row_launch(tok_embd_.dt, src, d_x_, E)) {
        err = "embedding dequant failed (type " + std::string(dtype_name(tok_embd_.dt)) +
              ", row " + std::to_string((*toks)[t]) + ")";
        return false;
      }
    } else if (hipMemcpy(d_x_, host_emb + t * (std::size_t)E, (std::size_t)E * sizeof(float),
                         hipMemcpyHostToDevice) != hipSuccess) {
      err = "embedding upload failed";
      return false;
    }
    emit("model.input_embed", -1, d_x_, E);
    const int l_end = debug_layer_limit();
    for (int il = 0; il < l_end; ++il) {
      const bool ok = w_[il].recr ? gdn_layer(il, (int)t, err)
                                 : full_attn(il, (int)t, start_pos + (int)t, err);
      if (!ok) {
        err = "layer " + std::to_string(il) + ": " + err;
        return false;
      }
      emit("attn_residual", il, d_x_, E);
      if (!rms_norm_launch(d_x_, (const float *)w_[il].attn_post_norm.ptr, d_xn_, 1, E,
                           (float)cfg_.rms_norm_eps)) {
        err = "attn_post_norm launch failed";
        return false;
      }
      emit("attn_post_norm", il, d_xn_, E);
      if (!ffn(il, err)) {
        err = "layer " + std::to_string(il) + ": " + err;
        return false;
      }
      emit("l_out", il, d_x_, E);
    }
  }

  // diagnostic: the residual stream entering the final norm (its RMS is what
  // the RMS norm divides out, so a too-large value here compresses every
  // downstream logit)
  emit("diag.hidden_pre_norm", -1, d_x_, E);
  if (!rms_norm_launch(d_x_, (const float *)output_norm_.ptr, d_x_, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "output_norm launch failed";
    return false;
  }
  emit("result_norm", -1, d_x_, E);
  // (1) host time spent queueing the trunk (no sync yet)
  const clock::time_point t_queued = clock::now();
  host_launch_ms_ += std::chrono::duration<double, std::milli>(t_queued - t_queue0).count();
  // (2) the first blocking read: this is where the queued trunk is waited for.
  if (want_hidden_ && !readback(d_x_, E, hidden)) {
    err = "hidden readback failed";
    return false;
  }
  const clock::time_point t_drained = clock::now();

  const int n_vocab = output_.dim1;
  // The logits buffer is allocated once per Graph (1 MB): allocating and freeing
  // it every token is pure overhead in the decode loop.
  if (d_logits_ == nullptr) {
    if (hipMalloc(&d_logits_, (std::size_t)n_vocab * sizeof(float)) != hipSuccess) {
      err = "hipMalloc logits failed";
      return false;
    }
  }
  if (!proj(output_, d_x_, d_logits_, n_vocab, E, err)) {
    return false;
  }
  emit("result_output", -1, d_logits_, n_vocab);
  logits.resize(n_vocab);
  const bool copied = hipMemcpy(logits.data(), d_logits_, (std::size_t)n_vocab * sizeof(float),
                               hipMemcpyDeviceToHost) == hipSuccess;
  host_readback_ms_ += std::chrono::duration<double, std::milli>(clock::now() - t_drained).count();
  if (!copied) {
    err = "logits readback failed";
    return false;
  }
  return true;
}

inline bool Graph::reset_state(std::string &err) {
  const int n_recr = count_recr();
  if (n_recr == 0) return true;
  const int S = (int)cfg_.ssm_state_size;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int K = (int)cfg_.ssm_conv_kernel;
  const int chan = 2 * (int)cfg_.ssm_group_count * S + nvh * S;
  const std::size_t n_st = (std::size_t)n_recr * nvh * S * S;
  const std::size_t n_cv = (std::size_t)n_recr * (K - 1) * chan;
  if (hipMemset(d_state_, 0, n_st * sizeof(float)) != hipSuccess ||
      hipMemset(d_convst_, 0, n_cv * sizeof(float)) != hipSuccess) {
    err = "recurrent state reset failed";
    return false;
  }
  return true;
}

inline bool Graph::debug_fill_caches(unsigned seed, std::string &err) {
  const int NKV = n_head_kv(), HD = head_dim();
  const int n_recr = count_recr();
  // Every full-attention layer has its own cache; the buffers are contiguous
  // row arrays (n_layers * max_ctx * n_head_kv rows), so fill all of them —
  // leaving a layer uninitialised would feed the attention uninitialised VRAM.
  const std::int64_t rows =
      (std::int64_t)(n_layer() - n_recr) * (std::int64_t)max_ctx_ * NKV;
  if (!kv_fill_launch(kv_k_, d_k_, rows, HD, seed) ||
      !kv_fill_launch(kv_v_, d_v_, rows, HD, seed + 1)) {
    err = "kv_fill launch failed";
    return false;
  }
  const int S = (int)cfg_.ssm_state_size;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int K = (int)cfg_.ssm_conv_kernel;
  const int chan = 2 * (int)cfg_.ssm_group_count * S + nvh * S;
  const int threads = 256;
  const std::size_t n_st = (std::size_t)(n_recr > 0 ? n_recr : 1) * nvh * S * S;
  const std::size_t n_cv = (std::size_t)(n_recr > 0 ? n_recr : 1) * (K - 1) * chan;
  if (n_recr > 0) {
    fill_random_kernel<<<(unsigned)((n_st + threads - 1) / threads), threads>>>(
        d_state_, (std::int64_t)n_st, seed);
    fill_random_kernel<<<(unsigned)((n_cv + threads - 1) / threads), threads>>>(
        d_convst_, (std::int64_t)n_cv, seed + 2);
    if (hipGetLastError() != hipSuccess) {
      err = "fill_random launch failed";
      return false;
    }
  }
  return true;
}

inline void Graph::release() {
  auto fr = [](GpuTensor &t) {
    if (t.ptr) (void)hipFree(t.ptr);
    t.ptr = nullptr;
    t.valid = false;
  };
  for (auto &L : w_) {
    fr(L.attn_norm); fr(L.attn_post_norm); fr(L.ffn_gate); fr(L.ffn_up); fr(L.ffn_down);
    fr(L.attn_q); fr(L.attn_k); fr(L.attn_v); fr(L.attn_output); fr(L.attn_q_norm); fr(L.attn_k_norm);
    fr(L.attn_qkv); fr(L.attn_gate); fr(L.ssm_conv1d); fr(L.ssm_a); fr(L.ssm_dt); fr(L.ssm_beta);
    fr(L.ssm_alpha); fr(L.ssm_norm); fr(L.ssm_out);
  }
  fr(tok_embd_); fr(output_norm_); fr(output_);
  float *ptrs[] = {d_x_,     d_xn_,    d_proj_,   d_ffnout_, d_attnout_, d_attngate_,
                   d_qkv_,   d_conv_,  d_z_,      d_alpha_,  d_beta_,
                   d_gate_,  d_state_, d_convst_, d_ffn_a_,  d_ffn_b_,   d_kstage_,
                   d_vstage_};
  for (float *p : ptrs) {
    if (p) (void)hipFree(p);
  }
  // `ptrs` holds copies, so nulling the members is a separate step. Without it a
  // second release() freed the same 18 addresses again (the batch and the other
  // pointers below were already nulled) — review finding M1.
  d_x_ = d_xn_ = d_proj_ = d_ffnout_ = d_attnout_ = d_attngate_ = nullptr;
  d_qkv_ = d_conv_ = d_z_ = d_alpha_ = d_beta_ = nullptr;
  d_gate_ = d_state_ = d_convst_ = d_ffn_a_ = d_ffn_b_ = d_kstage_ = d_vstage_ = nullptr;
  if (d_k_) (void)hipFree(d_k_);
  if (d_v_) (void)hipFree(d_v_);
  d_k_ = nullptr;
  d_v_ = nullptr;
  float *batch_ptrs[] = {d_xb_,   d_xnb_,  d_qb_,    d_kb_,    d_vb_,     d_attnb_,
                         d_gateb_, d_ffnab_, d_ffnbb_, d_projb_, d_qkvb_,   d_zb_,
                         d_convb_, d_alphab_, d_betab_, d_gate2b_};
  for (float *p : batch_ptrs) {
    if (p) (void)hipFree(p);
  }
  d_xb_ = d_xnb_ = d_qb_ = d_kb_ = d_vb_ = d_attnb_ = d_gateb_ = nullptr;
  d_ffnab_ = d_ffnbb_ = d_projb_ = d_qkvb_ = d_zb_ = d_convb_ = nullptr;
  d_alphab_ = d_betab_ = d_gate2b_ = nullptr;
  if (d_aqb_) (void)hipFree(d_aqb_);
  if (d_posb_) (void)hipFree(d_posb_);
  d_aqb_ = nullptr;
  d_posb_ = nullptr;
  if (d_q8_) (void)hipFree(d_q8_);
  if (d_pos_) (void)hipFree(d_pos_);
  if (d_logits_) (void)hipFree(d_logits_);
  if (d_attn_partial_) (void)hipFree(d_attn_partial_);
  d_attn_partial_ = nullptr;
  d_q8_ = nullptr;
  d_pos_ = nullptr;
  d_logits_ = nullptr;
}

}  // namespace rdna4
