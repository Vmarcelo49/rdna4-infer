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

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "rdna4/attn.cuh"
#include "rdna4/dtype.h"
#include "rdna4/gdn.cuh"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"
#include "rdna4/model.h"
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

  // Test hooks (diagnostics only; see PLAN.md M3).
  // Fills every KV cache and GDN state with a deterministic pseudo-random
  // pattern so a single decode step can be measured against a full cache.
  bool debug_fill_caches(unsigned seed, std::string &err);
  // Bytes per full-attention layer's cache (multiply by the number of
  // full-attention layers for the total).
  std::size_t kv_layer_bytes() const { return kv_bytes_; }
  int n_full_attn() const { return n_layer() - count_recr(); }
  int max_ctx() const { return max_ctx_; }

  void readback(const float *d, std::size_t n, std::vector<float> &out) const {
    out.resize(n);
    (void)hipMemcpy(out.data(), d, n * sizeof(float), hipMemcpyDeviceToHost);
  }

  int n_embd() const { return (int)cfg_.embedding_length; }
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
  bool alloc(float *&p, std::size_t n, std::string &err);
  bool proj(const GpuTensor &w, const float *d_x, float *d_y, int nrows, int ncols,
            std::string &err);
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
  float *d_qkv_ = nullptr, *d_conv_ = nullptr, *d_z_ = nullptr;
  float *d_alpha_ = nullptr, *d_beta_ = nullptr, *d_gate_ = nullptr;
  float *d_state_ = nullptr, *d_convst_ = nullptr;
  float *d_ffn_a_ = nullptr, *d_ffn_b_ = nullptr;
  float *d_q8_ = nullptr;
  std::size_t q8_blocks_ = 0;
  int cur_token_ = 0;
  int *d_pos_ = nullptr;
  int max_ctx_ = 0;
};

// ---------------------------------------------------------------------------
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
    if (!upn("blk.%d.attn_norm.weight", w_[il].attn_norm)) return false;
    // the GGUF name is `post_attention_norm` (llama.cpp maps it to attn_post_norm)
    if (!upn("blk.%d.post_attention_norm.weight", w_[il].attn_post_norm)) return false;
    if (!upn("blk.%d.ffn_gate.weight", w_[il].ffn_gate)) return false;
    if (!upn("blk.%d.ffn_up.weight", w_[il].ffn_up)) return false;
    if (!upn("blk.%d.ffn_down.weight", w_[il].ffn_down)) return false;
    if (w_[il].recr) {
      ++n_recr;
      if (!upn("blk.%d.attn_qkv.weight", w_[il].attn_qkv)) return false;
      if (!upn("blk.%d.attn_gate.weight", w_[il].attn_gate)) return false;
      if (!upn("blk.%d.ssm_conv1d.weight", w_[il].ssm_conv1d)) return false;
      if (!upn("blk.%d.ssm_a", w_[il].ssm_a)) return false;
      if (!upn("blk.%d.ssm_dt.bias", w_[il].ssm_dt)) return false;
      if (!upn("blk.%d.ssm_beta.weight", w_[il].ssm_beta)) return false;
      if (!upn("blk.%d.ssm_alpha.weight", w_[il].ssm_alpha)) return false;
      if (!upn("blk.%d.ssm_norm.weight", w_[il].ssm_norm)) return false;
      if (!upn("blk.%d.ssm_out.weight", w_[il].ssm_out)) return false;
    } else {
      if (!upn("blk.%d.attn_q.weight", w_[il].attn_q)) return false;
      if (!upn("blk.%d.attn_k.weight", w_[il].attn_k)) return false;
      if (!upn("blk.%d.attn_v.weight", w_[il].attn_v)) return false;
      if (!upn("blk.%d.attn_output.weight", w_[il].attn_output)) return false;
      if (!upn("blk.%d.attn_q_norm.weight", w_[il].attn_q_norm)) return false;
      if (!upn("blk.%d.attn_k_norm.weight", w_[il].attn_k_norm)) return false;
    }
  }
  if (!up("token_embd.weight", tok_embd_, err)) return false;
  if (!up("output_norm.weight", output_norm_, err)) return false;
  if (!up("output.weight", output_, err)) return false;

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
  if (!alloc(d_state_, (std::size_t)n_recr * nvh * S * S, err)) return false;
  if (!alloc(d_convst_, (std::size_t)n_recr * (K - 1) * chan, err)) return false;
  if (hipMalloc(&d_pos_, sizeof(int)) != hipSuccess) return false;

  q8_blocks_ = (std::size_t)(F / QK8_1) + 8;  // ffn_down has the largest reduction
  if (hipMalloc(&d_q8_, q8_blocks_ * sizeof(block_q8_1)) != hipSuccess) {
    err = "hipMalloc failed (q8 scratch)";
    return false;
  }
  return true;
}

inline bool Graph::proj(const GpuTensor &w, const float *d_x, float *d_y, int nrows, int ncols,
                        std::string &err) {
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
  if (!proj(L.attn_k, d_xn_, d_kstage_, NKV * HD, E, err)) return false;
  if (!proj(L.attn_v, d_xn_, d_vstage_, NKV * HD, E, err)) return false;
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
  if (!attn_launch(d_attnout_, kc, vc, d_attnout_, pos, NH, NKV, HD, scale, kv_k_, kv_v_)) {
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
  if (!proj(L.attn_gate, d_xn_, d_z_, d_inner, E, err)) return false;
  if (!proj(L.ssm_beta, d_xn_, d_beta_, nvh, E, err)) return false;
  if (!proj(L.ssm_alpha, d_xn_, d_alpha_, nvh, E, err)) return false;
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
  if (!proj(L.ffn_up, d_xn_, d_ffn_b_, F, E, err)) return false;
  if (!mul_launch(d_ffn_a_, d_ffn_b_, d_ffn_a_, F)) return false;
  if (!proj(L.ffn_down, d_ffn_a_, d_ffnout_, E, F, err)) return false;
  emit("ffn_out", il, d_ffnout_, E);
  if (!add_residual(d_ffnout_, d_x_, E, err)) return false;
  return true;
}

// ---------------------------------------------------------------------------
inline bool Graph::forward(const std::vector<float> &embeddings, std::vector<float> &hidden,
                           std::vector<float> &logits, std::string &err) {
  return forward(embeddings, 0, hidden, logits, err);
}

inline bool Graph::forward(const std::vector<float> &embeddings, int start_pos,
                           std::vector<float> &hidden, std::vector<float> &logits,
                           std::string &err) {
  const int E = n_embd();
  const int L = n_layer();
  const std::size_t n_tokens = embeddings.size() / (std::size_t)E;
  if (start_pos < 0 || (std::size_t)start_pos + n_tokens > (std::size_t)max_ctx_) {
    err = "positions exceed the allocated context (" + std::to_string(max_ctx_) + ")";
    return false;
  }

  for (std::size_t t = 0; t < n_tokens; ++t) {
    cur_token_ = (int)t;
    if (hipMemcpy(d_x_, embeddings.data() + t * E, (std::size_t)E * sizeof(float),
                  hipMemcpyHostToDevice) != hipSuccess) {
      err = "embedding upload failed";
      return false;
    }
    emit("model.input_embed", -1, d_x_, E);
    for (int il = 0; il < L; ++il) {
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
  readback(d_x_, E, hidden);

  const int n_vocab = output_.dim1;
  float *d_logits = nullptr;
  if (hipMalloc(&d_logits, (std::size_t)n_vocab * sizeof(float)) != hipSuccess) {
    err = "hipMalloc logits failed";
    return false;
  }
  if (!proj(output_, d_x_, d_logits, n_vocab, E, err)) {
    (void)hipFree(d_logits);
    return false;
  }
  emit("result_output", -1, d_logits, n_vocab);
  logits.resize(n_vocab);
  const bool copied = hipMemcpy(logits.data(), d_logits, (std::size_t)n_vocab * sizeof(float),
                               hipMemcpyDeviceToHost) == hipSuccess;
  (void)hipFree(d_logits);
  if (!copied) {
    err = "logits readback failed";
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
  if (d_k_) (void)hipFree(d_k_);
  if (d_v_) (void)hipFree(d_v_);
  d_k_ = nullptr;
  d_v_ = nullptr;
  if (d_q8_) (void)hipFree(d_q8_);
  if (d_pos_) (void)hipFree(d_pos_);
  d_q8_ = nullptr;
  d_pos_ = nullptr;
}

}  // namespace rdna4
