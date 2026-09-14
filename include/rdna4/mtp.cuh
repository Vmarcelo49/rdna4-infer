#pragma once
// feat/mtp — the NextN / MTP draft block (`blk.64`) of Qwen3.8-27B.
//
// The GGUF has 65 blocks: 64 trunk layers plus one MTP block that the trunk
// forward never executes (`Graph::n_layer()` = block_count - nextn_predict_layers
// = 64). This header loads that block and runs the draft forward.
//
// Wiring transcribed from llama.cpp's own MTP graph for this exact architecture,
// `llama_model_qwen35::graph_mtp` (src/models/qwen35.cpp, the same file the trunk
// graph in graph.cuh was transcribed from; b10902 / commit df03399b8):
//
//   h_t   = trunk hidden state AFTER the final output_norm   (graph.cuh's
//           forward_tokens already returns exactly this: `hidden`)
//   x     = eh_proj( concat( enorm(emb(tok)), hnorm(h_t) ) )      <- e FIRST
//   x    += attn( rms_norm(x, attn_norm) ) gated by sigmoid(q gate), one
//           transformer block with the block's own Q/K/V/O and its OWN KV cache
//   x    += ffn( rms_norm(x, post_attention_norm) )
//   h'    = rms_norm(x, nextn.shared_head_norm)
//   logits= output.weight @ h'          (the LM head is SHARED with the trunk)
//
// `h'` is also what llama.cpp feeds back for chained drafting (common/
// speculative.cpp keeps it in `pending_h`), which is what step_chained() does.
//
// Every tensor is required and every dim is checked against the config: a
// missing or differently-shaped MTP tensor is a hard error, never a fallback.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
#include "rdna4/nn.cuh"

namespace rdna4 {

// The two trunk tensors the draft block shares (llama.cpp resolves them as
// `layer.nextn.embed_tokens ? ... : model.tok_embd` and `nextn.shared_head_head
// ? ... : model.output`; neither is present in this GGUF, so both come from the
// trunk). Graph::mtp_shared_weights() fills this in — the head itself never
// touches Graph internals.
struct MtpShared {
  const void *tok_embd = nullptr;      // token_embd.weight, raw quantized rows
  DType tok_embd_dt = DType::F32;
  const void *output = nullptr;        // output.weight = the shared LM head
  DType output_dt = DType::F32;
  int n_vocab = 0;
};

// Shapes of one MTP block, derived from the config and pinned by init().
struct MtpDims {
  int block = 0;      // GGUF block index (64 here)
  int n_embd = 0;     // 5120
  int n_head = 0;     // 24
  int n_head_kv = 0;  // 4
  int head_dim = 0;   // 256
  int n_rot = 0;      // 64 (rope.dimension_count)
  int n_ffn = 0;      // 17408
};

inline MtpDims mtp_dims(const Qwen35Config &cfg, int block) {
  MtpDims d;
  d.block = block;
  d.n_embd = (int)cfg.embedding_length;
  d.n_head = (int)cfg.head_count;
  d.n_head_kv = (int)cfg.head_count_kv;
  d.head_dim = (int)cfg.key_length;
  d.n_rot = (int)cfg.rope_dim_count;
  d.n_ffn = (int)cfg.feed_forward_length;
  return d;
}

class MtpHead {
 public:
  // Same contract as Graph::NodeCb: the node names are llama.cpp's own, so the
  // draft block can be diffed against an eval-callback dump of its
  // `LLM_GRAPH_TYPE_DECODER_MTP` graph (tests/check_mtp_gpu.hip).
  using NodeCb = std::function<void(const char *name, const float *d_ptr, std::int64_t n)>;

  MtpHead() = default;
  ~MtpHead() { release(); }
  MtpHead(const MtpHead &) = delete;
  MtpHead &operator=(const MtpHead &) = delete;

  // Loads blk.<dims.block>.* : the common block tensors plus the four nextn.*
  // ones, checking presence, dtype and dims. `max_ctx` sizes the block's own KV
  // cache; `kv_k`/`kv_v` should match the trunk's cache types.
  bool init(const GgufLoader &ld, const Qwen35Config &cfg, int max_ctx, KvType kv_k, KvType kv_v,
            std::string &err);

  void set_shared(const MtpShared &s) { shared_ = s; }
  void set_node_cb(NodeCb cb) { cb_ = std::move(cb); }
  // concat order of the eh_proj input: true = [enorm(emb) ; hnorm(h)] (what
  // llama.cpp's ggml_concat(e_norm, h_norm) does), false = the reverse. The
  // default comes from RD_MTP_CONCAT ("eh" | "he"), so the order can be measured
  // on the real model instead of assumed.
  void set_concat_order(bool e_first) { e_first_ = e_first; }
  bool concat_order_e_first() const { return e_first_; }

  // One draft step. `h_prev` (host, n_embd floats) is the trunk's hidden state at
  // position pos-1 — the plain post-final-norm state forward_tokens() returns;
  // `tok` is the token at position `pos`. Fills `logits` (n_vocab) with the
  // draft distribution for position pos+1 and sets `argmax`.
  bool step_host(const float *h_prev, std::int32_t tok, int pos, std::vector<float> *logits,
                 std::int32_t *argmax, std::string &err);
  // Chained step: the `h` input is this block's own previous output (h_nextn)
  // instead of a trunk row, which is how llama.cpp drafts more than one token
  // ahead without the trunk (common/speculative.cpp `pending_h`).
  bool step_chained(std::int32_t tok, int pos, std::vector<float> *logits,
                    std::int32_t *argmax, std::string &err);

  // New sequence: the KV cache needs no clearing (attention is causal and every
  // row is rewritten by position before it is read — same argument as
  // Graph::reset_state), only the chained-h state is dropped.
  void reset() { have_h_out_ = false; }

  // h_nextn of the last step (the post shared_head_norm hidden state), read back
  // to the host: tests check its finiteness, and it is what a chained step
  // feeds back as its `h` input.
  bool read_h_out(std::vector<float> &h) const;

  const MtpDims &dims() const { return d_; }
  int n_vocab() const { return shared_.n_vocab; }
  int max_ctx() const { return max_ctx_; }
  std::size_t weight_bytes() const { return weight_bytes_; }
  double last_step_ms() const { return last_ms_; }
  std::size_t kv_layer_bytes() const { return kv_bytes_; }

 private:
  struct W {
    void *ptr = nullptr;
    std::size_t bytes = 0;
    DType dt = DType::F32;
    int dim0 = 0, dim1 = 0;  // GGUF order: [in, out]
  };

  bool load(const GgufLoader &ld, const char *fmt, int block, W &w, bool must_f32,
            std::string &err);
  bool check_dims(const char *name, const W &w, int want_in, int want_out, std::string &err) const;
  bool alloc(float *&p, std::size_t n, std::string &err);
  bool alloc_kv(std::string &err);
  bool proj(const W &w, const float *d_x, float *d_y, int nrows, int ncols, std::string &err);
  bool run(std::int32_t tok, int pos, bool h_from_host, const float *h_host,
           std::vector<float> *logits, std::int32_t *argmax, std::string &err);
  void emit(const char *name, const float *d_ptr, std::int64_t n) const {
    if (cb_) cb_(name, d_ptr, n);
  }
  void release();

  MtpDims d_;
  Qwen35Config cfg_;
  MtpShared shared_;
  NodeCb cb_;
  bool e_first_ = true;

  W eh_proj_, enorm_, hnorm_, shared_head_norm_;
  W attn_norm_, attn_post_norm_, attn_q_, attn_k_, attn_v_, attn_output_, attn_q_norm_,
      attn_k_norm_, ffn_gate_, ffn_up_, ffn_down_;

  float *d_e_ = nullptr, *d_h_ = nullptr, *d_in_ = nullptr, *d_x_ = nullptr, *d_xn_ = nullptr;
  float *d_qf_ = nullptr, *d_q_ = nullptr, *d_gate_ = nullptr;
  float *d_kst_ = nullptr, *d_vst_ = nullptr, *d_attn_ = nullptr, *d_out_ = nullptr;
  float *d_ffn_a_ = nullptr, *d_ffn_b_ = nullptr, *d_hout_ = nullptr, *d_logits_ = nullptr;
  block_q8_1 *d_q8_ = nullptr;
  std::size_t q8_blocks_ = 0;
  int *d_pos_ = nullptr;

  KvType kv_k_ = KvType::F16, kv_v_ = KvType::F16;
  void *d_ck_ = nullptr, *d_cv_ = nullptr;
  std::size_t kv_bytes_ = 0;
  int max_ctx_ = 0;
  bool have_h_out_ = false;
  std::size_t weight_bytes_ = 0;
  double last_ms_ = 0.0;
};

// ---------------------------------------------------------------------------
inline bool MtpHead::load(const GgufLoader &ld, const char *fmt, int block, W &w, bool must_f32,
                          std::string &err) {
  char name[128];
  std::snprintf(name, sizeof(name), fmt, block);
  const auto *info = ld.find(name);
  if (!info) {
    err = std::string("MTP: missing tensor ") + name;
    return false;
  }
  const std::size_t i = (std::size_t)(info - ld.meta().tensors.data());
  std::vector<std::uint8_t> bytes;
  if (!ld.load_tensor(i, bytes, err)) return false;
  w.dt = ld.dtype(i);
  if (must_f32 && w.dt != DType::F32) {
    err = std::string("MTP: tensor ") + name + " must be f32, got " + dtype_name(w.dt);
    return false;
  }
  if (hipMalloc(&w.ptr, bytes.size()) != hipSuccess) {
    err = std::string("MTP: hipMalloc failed for ") + name;
    return false;
  }
  if (hipMemcpy(w.ptr, bytes.data(), bytes.size(), hipMemcpyHostToDevice) != hipSuccess) {
    err = std::string("MTP: hipMemcpy failed for ") + name;
    return false;
  }
  w.bytes = bytes.size();
  w.dim0 = info->dims.size() > 0 ? (int)info->dims[0] : 0;
  // a 1-D tensor is [n, 1] here: the checks below are all written as [in, out]
  w.dim1 = info->dims.size() > 1 ? (int)info->dims[1] : 1;
  weight_bytes_ += bytes.size();
  return true;
}

inline bool MtpHead::check_dims(const char *name, const W &w, int want_in, int want_out,
                                std::string &err) const {
  if (w.dim0 != want_in || w.dim1 != want_out) {
    err = "MTP: " + std::string(name) + " is [" + std::to_string(w.dim0) + ", " +
          std::to_string(w.dim1) + "], expected [" + std::to_string(want_in) + ", " +
          std::to_string(want_out) + "]";
    return false;
  }
  // the byte count must match the shape exactly (a quant type with a block size
  // that does not divide the row would otherwise read past the tensor)
  const std::uint64_t want = tensor_bytes(w.dt, (std::uint64_t)want_in * (std::uint64_t)want_out);
  if (w.bytes != want) {
    err = "MTP: " + std::string(name) + " has " + std::to_string(w.bytes) + " bytes, expected " +
          std::to_string(want) + " for " + std::to_string(want_in) + "x" +
          std::to_string(want_out) + " " + dtype_name(w.dt);
    return false;
  }
  return true;
}

inline bool MtpHead::alloc(float *&p, std::size_t n, std::string &err) {
  if (hipMalloc(&p, n * sizeof(float)) != hipSuccess) {
    err = "MTP: hipMalloc failed";
    return false;
  }
  return true;
}

inline bool MtpHead::init(const GgufLoader &ld, const Qwen35Config &cfg, int max_ctx, KvType kv_k,
                          KvType kv_v, std::string &err) {
  cfg_ = cfg;
  kv_k_ = kv_k;
  kv_v_ = kv_v;
  max_ctx_ = max_ctx;
  if (cfg.nextn_predict_layers == 0) {
    err = "MTP: the model declares no nextn_predict_layers, so there is no draft block";
    return false;
  }
  if (cfg.nextn_predict_layers != 1) {
    err = "MTP: only one NextN block is supported (nextn_predict_layers = " +
          std::to_string(cfg.nextn_predict_layers) + ")";
    return false;
  }
  if (max_ctx <= 0) {
    err = "MTP: max_ctx must be positive";
    return false;
  }
  // the block is the last one: llama.cpp stores it immediately after the trunk
  d_ = mtp_dims(cfg, (int)cfg.block_count - (int)cfg.nextn_predict_layers);
  const int E = d_.n_embd, NH = d_.n_head, NKV = d_.n_head_kv, HD = d_.head_dim, F = d_.n_ffn;
  if (E <= 0 || NH <= 0 || NKV <= 0 || HD <= 0 || F <= 0) {
    err = "MTP: degenerate config";
    return false;
  }
  if (NH % NKV != 0) {
    err = "MTP: head_count must be a multiple of head_count_kv";
    return false;
  }
  if (d_.n_rot <= 0 || d_.n_rot > HD || d_.n_rot % 2 != 0) {
    err = "MTP: rope.dimension_count must be even and <= key_length";
    return false;
  }

  // ---- the 15 tensors of the MTP block, in GGUF [in, out] order ------------
  // common to every block
  if (!load(ld, "blk.%d.attn_norm.weight", d_.block, attn_norm_, true, err)) return false;
  if (!load(ld, "blk.%d.post_attention_norm.weight", d_.block, attn_post_norm_, true, err))
    return false;
  if (!load(ld, "blk.%d.ffn_gate.weight", d_.block, ffn_gate_, false, err)) return false;
  if (!load(ld, "blk.%d.ffn_up.weight", d_.block, ffn_up_, false, err)) return false;
  if (!load(ld, "blk.%d.ffn_down.weight", d_.block, ffn_down_, false, err)) return false;
  // full-attention set (block 64 is a full-attention block, not a GDN one: it
  // carries attn_q/k/v/output + q/k norms and no ssm_* tensor)
  if (!load(ld, "blk.%d.attn_q.weight", d_.block, attn_q_, false, err)) return false;
  if (!load(ld, "blk.%d.attn_k.weight", d_.block, attn_k_, false, err)) return false;
  if (!load(ld, "blk.%d.attn_v.weight", d_.block, attn_v_, false, err)) return false;
  if (!load(ld, "blk.%d.attn_output.weight", d_.block, attn_output_, false, err)) return false;
  if (!load(ld, "blk.%d.attn_q_norm.weight", d_.block, attn_q_norm_, true, err)) return false;
  if (!load(ld, "blk.%d.attn_k_norm.weight", d_.block, attn_k_norm_, true, err)) return false;
  // NextN-specific
  if (!load(ld, "blk.%d.nextn.eh_proj.weight", d_.block, eh_proj_, false, err)) return false;
  if (!load(ld, "blk.%d.nextn.enorm.weight", d_.block, enorm_, true, err)) return false;
  if (!load(ld, "blk.%d.nextn.hnorm.weight", d_.block, hnorm_, true, err)) return false;
  if (!load(ld, "blk.%d.nextn.shared_head_norm.weight", d_.block, shared_head_norm_, true, err))
    return false;

  if (!check_dims("attn_norm.weight", attn_norm_, E, 1, err)) return false;
  if (!check_dims("post_attention_norm.weight", attn_post_norm_, E, 1, err)) return false;
  // attn_q emits [ q (head_dim) | gate (head_dim) ] per head, like every
  // full-attention block of the trunk (2*head_dim per head)
  if (!check_dims("attn_q.weight", attn_q_, E, 2 * HD * NH, err)) return false;
  if (!check_dims("attn_k.weight", attn_k_, E, HD * NKV, err)) return false;
  if (!check_dims("attn_v.weight", attn_v_, E, HD * NKV, err)) return false;
  if (!check_dims("attn_output.weight", attn_output_, HD * NH, E, err)) return false;
  if (!check_dims("attn_q_norm.weight", attn_q_norm_, HD, 1, err)) return false;
  if (!check_dims("attn_k_norm.weight", attn_k_norm_, HD, 1, err)) return false;
  if (!check_dims("ffn_gate.weight", ffn_gate_, E, F, err)) return false;
  if (!check_dims("ffn_up.weight", ffn_up_, E, F, err)) return false;
  if (!check_dims("ffn_down.weight", ffn_down_, F, E, err)) return false;
  // eh_proj maps the CONCATENATION of the two normalized vectors to the hidden
  // size: [2*n_embd -> n_embd]. This is the shape the concat order is pinned by.
  if (!check_dims("nextn.eh_proj.weight", eh_proj_, 2 * E, E, err)) return false;
  if (!check_dims("nextn.enorm.weight", enorm_, E, 1, err)) return false;
  if (!check_dims("nextn.hnorm.weight", hnorm_, E, 1, err)) return false;
  if (!check_dims("nextn.shared_head_norm.weight", shared_head_norm_, E, 1, err)) return false;

  if (const char *e = getenv("RD_MTP_CONCAT")) {
    if (std::strcmp(e, "he") == 0) {
      e_first_ = false;
    } else if (std::strcmp(e, "eh") == 0) {
      e_first_ = true;
    } else {
      err = "MTP: RD_MTP_CONCAT must be 'eh' or 'he'";
      return false;
    }
  }

  // ---- scratch ------------------------------------------------------------
  if (!alloc(d_e_, E, err) || !alloc(d_h_, E, err) || !alloc(d_in_, 2 * E, err) ||
      !alloc(d_x_, E, err) || !alloc(d_xn_, E, err) || !alloc(d_qf_, 2 * NH * HD, err) ||
      !alloc(d_q_, NH * HD, err) || !alloc(d_gate_, NH * HD, err) ||
      !alloc(d_kst_, NKV * HD, err) || !alloc(d_vst_, NKV * HD, err) ||
      !alloc(d_attn_, NH * HD, err) || !alloc(d_out_, E, err) || !alloc(d_ffn_a_, F, err) ||
      !alloc(d_ffn_b_, F, err) || !alloc(d_hout_, E, err)) {
    return false;
  }
  const int max_red = F > 2 * E ? F : 2 * E;  // eh_proj (2E) and ffn_down (F)
  q8_blocks_ = (std::size_t)(max_red / QK8_1) + 8;
  if (max_red % QK8_1 != 0) {
    err = "MTP: reduction dim is not a multiple of QK8_1";
    return false;
  }
  if (hipMalloc(&d_q8_, q8_blocks_ * sizeof(block_q8_1)) != hipSuccess) {
    err = "MTP: hipMalloc failed (q8 scratch)";
    return false;
  }
  if (hipMalloc(&d_pos_, sizeof(int)) != hipSuccess) {
    err = "MTP: hipMalloc failed (pos)";
    return false;
  }
  return alloc_kv(err);
}

inline bool MtpHead::alloc_kv(std::string &err) {
  const int NKV = d_.n_head_kv, HD = d_.head_dim;
  kv_bytes_ = (std::size_t)max_ctx_ * NKV * kv_row_bytes(kv_k_, HD);
  if (hipMalloc(&d_ck_, kv_bytes_) != hipSuccess || hipMalloc(&d_cv_, kv_bytes_) != hipSuccess) {
    err = "MTP: hipMalloc failed (own KV cache)";
    return false;
  }
  // The cache is written row by row as positions advance and attention only ever
  // reads rows <= pos, so no clearing is needed (see reset()).
  if (!kv_fill_launch(kv_k_, d_ck_, (std::int64_t)max_ctx_ * NKV, HD, 1u) ||
      !kv_fill_launch(kv_v_, d_cv_, (std::int64_t)max_ctx_ * NKV, HD, 2u)) {
    err = "MTP: kv_fill launch failed";
    return false;
  }
  return true;
}

inline bool MtpHead::proj(const W &w, const float *d_x, float *d_y, int nrows, int ncols,
                          std::string &err) {
  // Same check the trunk's Graph::proj makes: the dims alone do not pin the byte
  // size, so a blk.64.* tensor with the right dims but an inconsistent dtype or
  // block count would be read past its end — inside this process's own VRAM, so
  // without a fault, silently wrong (review finding M3).
  if (w.dim0 != ncols || w.dim1 != nrows ||
      w.bytes != tensor_bytes(w.dt, (std::uint64_t)nrows * (std::uint64_t)ncols)) {
    err = "MTP: proj shape mismatch";
    return false;
  }
  const int nb = ncols / QK8_1;
  if (ncols % QK8_1 != 0 || (std::size_t)nb > q8_blocks_) {
    err = "MTP: bad reduction dim for q8 scratch";
    return false;
  }
  quantize_q8_1_kernel<<<(nb + 3) / 4, 128>>>(d_x, d_q8_, nb);
  if (hipGetLastError() != hipSuccess) {
    err = "MTP: quantize_q8_1 launch failed";
    return false;
  }
  if (!matvec_launch((int)w.dt, w.ptr, d_q8_, d_y, nrows, ncols, nullptr)) {
    err = "MTP: matvec_launch failed";
    return false;
  }
  return true;
}

inline bool MtpHead::step_host(const float *h_prev, std::int32_t tok, int pos,
                               std::vector<float> *logits, std::int32_t *argmax,
                               std::string &err) {
  if (h_prev == nullptr) {
    err = "MTP: h_prev is null";
    return false;
  }
  return run(tok, pos, /*h_from_host=*/true, h_prev, logits, argmax, err);
}

inline bool MtpHead::step_chained(std::int32_t tok, int pos, std::vector<float> *logits,
                                  std::int32_t *argmax, std::string &err) {
  if (!have_h_out_) {
    err = "MTP: step_chained before any step (no h_nextn to chain from)";
    return false;
  }
  return run(tok, pos, /*h_from_host=*/false, nullptr, logits, argmax, err);
}

inline bool MtpHead::run(std::int32_t tok, int pos, bool h_from_host, const float *h_host,
                         std::vector<float> *logits, std::int32_t *argmax, std::string &err) {
  using clock = std::chrono::steady_clock;
  const clock::time_point t0 = clock::now();

  const int E = d_.n_embd, NH = d_.n_head, NKV = d_.n_head_kv, HD = d_.head_dim, F = d_.n_ffn;
  if (shared_.output == nullptr || shared_.tok_embd == nullptr) {
    err = "MTP: shared trunk weights were not provided (Graph::mtp_shared_weights)";
    return false;
  }
  if (pos < 0 || pos >= max_ctx_) {
    err = "MTP: position " + std::to_string(pos) + " outside the allocated context (" +
          std::to_string(max_ctx_) + ")";
    return false;
  }
  if (tok < 0 || tok >= shared_.n_vocab) {
    err = "MTP: token id " + std::to_string(tok) + " outside [0, " +
          std::to_string(shared_.n_vocab) + ")";
    return false;
  }
  const float eps = (float)cfg_.rms_norm_eps;

  // (1) embedding of the token at `pos` (dequantized on device, the same kernel
  // the trunk uses for its own rows)
  const std::size_t emb_row = (std::size_t)tensor_bytes(shared_.tok_embd_dt, (std::uint64_t)E);
  const char *src = (const char *)shared_.tok_embd + (std::size_t)tok * emb_row;
  if (!dequant_row_launch(shared_.tok_embd_dt, src, d_e_, E)) {
    err = "MTP: embedding dequant failed (type " + std::string(dtype_name(shared_.tok_embd_dt)) +
          ")";
    return false;
  }
  emit("mtp_tok_embd", d_e_, E);

  // (2) the h input: the trunk's h_(pos-1), or this block's own output for a
  // chained step
  if (h_from_host) {
    if (hipMemcpy(d_h_, h_host, (std::size_t)E * sizeof(float), hipMemcpyHostToDevice) !=
        hipSuccess) {
      err = "MTP: h upload failed";
      return false;
    }
  } else if (hipMemcpy(d_h_, d_hout_, (std::size_t)E * sizeof(float), hipMemcpyDeviceToDevice) !=
             hipSuccess) {
    err = "MTP: chained h copy failed";
    return false;
  }

  // (3) enorm / hnorm (RMS, no scaling beyond the weight)
  if (!rms_norm_launch(d_e_, (const float *)enorm_.ptr, d_e_, 1, E, eps) ||
      !rms_norm_launch(d_h_, (const float *)hnorm_.ptr, d_h_, 1, E, eps)) {
    err = "MTP: enorm/hnorm launch failed";
    return false;
  }
  emit("mtp_enorm", d_e_, E);
  emit("mtp_hnorm", d_h_, E);

  // (4) concat: e first (llama.cpp ggml_concat(e_norm, h_norm, 0)), h first if
  // RD_MTP_CONCAT=he
  float *dst_h = e_first_ ? d_in_ + E : d_in_;
  float *dst_e = e_first_ ? d_in_ : d_in_ + E;
  if (hipMemcpy(dst_e, d_e_, (std::size_t)E * sizeof(float), hipMemcpyDeviceToDevice) !=
          hipSuccess ||
      hipMemcpy(dst_h, d_h_, (std::size_t)E * sizeof(float), hipMemcpyDeviceToDevice) !=
          hipSuccess) {
    err = "MTP: concat copy failed";
    return false;
  }
  emit("mtp_concat", d_in_, 2 * E);

  // (5) eh_proj -> the block's residual stream (inpSA in the reference graph)
  if (!proj(eh_proj_, d_in_, d_x_, E, 2 * E, err)) return false;
  emit("mtp_eh_proj", d_x_, E);

  // (6) attn_norm + Q/K/V
  if (!rms_norm_launch(d_x_, (const float *)attn_norm_.ptr, d_xn_, 1, E, eps)) {
    err = "MTP: attn_norm launch failed";
    return false;
  }
  emit("mtp_attn_norm", d_xn_, E);
  if (!proj(attn_q_, d_xn_, d_qf_, 2 * NH * HD, E, err)) return false;
  if (!proj(attn_k_, d_xn_, d_kst_, NKV * HD, E, err)) return false;
  if (!proj(attn_v_, d_xn_, d_vst_, NKV * HD, E, err)) return false;
  emit("mtp_Qcur_full", d_qf_, (std::int64_t)2 * NH * HD);
  emit("mtp_Vcur", d_vst_, (std::int64_t)NKV * HD);

  // (7) q/gate split, per-head q/k RMS norms
  if (!deinterleave_q_gate_launch(d_qf_, d_q_, d_gate_, NH, HD)) {
    err = "MTP: deinterleave launch failed";
    return false;
  }
  if (!rms_norm_launch(d_q_, (const float *)attn_q_norm_.ptr, d_q_, NH, HD, eps)) {
    err = "MTP: q_norm launch failed";
    return false;
  }
  if (!rms_norm_launch(d_kst_, (const float *)attn_k_norm_.ptr, d_kst_, NKV, HD, eps)) {
    err = "MTP: k_norm launch failed";
    return false;
  }
  emit("mtp_Qcur_normed", d_q_, (std::int64_t)NH * HD);
  emit("mtp_Kcur_normed", d_kst_, (std::int64_t)NKV * HD);
  emit("mtp_gate", d_gate_, (std::int64_t)NH * HD);

  // (8) RoPE at the token's own position, exactly like a trunk full-attn layer
  if (hipMemcpy(d_pos_, &pos, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
    err = "MTP: position upload failed";
    return false;
  }
  const float base = (float)cfg_.rope_freq_base;
  if (!rope_launch(d_q_, 1, NH, HD, d_.n_rot, base, d_pos_) ||
      !rope_launch(d_kst_, 1, NKV, HD, d_.n_rot, base, d_pos_)) {
    err = "MTP: rope launch failed";
    return false;
  }

  // (9) the block's own KV cache (NOT the trunk's: the MTP block attends with
  // K/V projected from its own input, so it keeps its own rows)
  const std::size_t krow = (std::size_t)kv_row_bytes(kv_k_, HD);
  const std::size_t vrow = (std::size_t)kv_row_bytes(kv_v_, HD);
  char *kbase = (char *)d_ck_ + (std::size_t)pos * NKV * krow;
  char *vbase = (char *)d_cv_ + (std::size_t)pos * NKV * vrow;
  for (int h = 0; h < NKV; ++h) {
    if (!kv_store_row_launch(kv_k_, d_kst_ + (std::size_t)h * HD,
                             kbase + (std::size_t)h * krow, HD) ||
        !kv_store_row_launch(kv_v_, d_vst_ + (std::size_t)h * HD,
                             vbase + (std::size_t)h * vrow, HD)) {
      err = "MTP: kv_store_row launch failed";
      return false;
    }
  }

  // (10) causal attention over rows 0..pos, then the gate
  const float scale = 1.0f / std::sqrt((float)HD);
  if (!attn_launch(d_q_, d_ck_, d_cv_, d_attn_, pos, NH, NKV, HD, scale, kv_k_, kv_v_)) {
    err = "MTP: attn launch failed";
    return false;
  }
  emit("mtp_attn_pregate", d_attn_, (std::int64_t)NH * HD);
  if (!unary_launch(d_gate_, d_gate_, NH * HD, UnOp::Sigmoid) ||
      !mul_launch(d_attn_, d_gate_, d_attn_, NH * HD)) {
    err = "MTP: gate launch failed";
    return false;
  }
  if (!proj(attn_output_, d_attn_, d_out_, E, NH * HD, err)) return false;
  emit("mtp_attn_out", d_out_, E);
  if (!add_launch(d_out_, d_x_, d_x_, E)) {
    err = "MTP: attn residual failed";
    return false;
  }
  emit("mtp_attn_residual", d_x_, E);

  // (11) FFN on the post-attention norm, added to the same residual
  if (!rms_norm_launch(d_x_, (const float *)attn_post_norm_.ptr, d_xn_, 1, E, eps)) {
    err = "MTP: post_attention_norm launch failed";
    return false;
  }
  emit("mtp_attn_post_norm", d_xn_, E);
  if (!proj(ffn_gate_, d_xn_, d_ffn_a_, F, E, err)) return false;
  if (!unary_launch(d_ffn_a_, d_ffn_a_, F, UnOp::Silu)) {
    err = "MTP: silu launch failed";
    return false;
  }
  if (!proj(ffn_up_, d_xn_, d_ffn_b_, F, E, err)) return false;
  if (!mul_launch(d_ffn_a_, d_ffn_b_, d_ffn_a_, F)) {
    err = "MTP: ffn mul failed";
    return false;
  }
  if (!proj(ffn_down_, d_ffn_a_, d_out_, E, F, err)) return false;
  emit("mtp_ffn_out", d_out_, E);
  if (!add_launch(d_out_, d_x_, d_x_, E)) {
    err = "MTP: ffn residual failed";
    return false;
  }
  emit("mtp_post_ffn", d_x_, E);

  // (12) shared head norm + shared LM head (output.weight of the trunk)
  if (!rms_norm_launch(d_x_, (const float *)shared_head_norm_.ptr, d_hout_, 1, E, eps)) {
    err = "MTP: shared_head_norm launch failed";
    return false;
  }
  have_h_out_ = true;
  emit("h_nextn", d_hout_, E);
  emit("mtp_shared_head_norm", d_hout_, E);
  const int n_vocab = shared_.n_vocab;
  if (d_logits_ == nullptr) {
    if (hipMalloc(&d_logits_, (std::size_t)n_vocab * sizeof(float)) != hipSuccess) {
      err = "MTP: hipMalloc logits failed";
      return false;
    }
  }
  W head;
  head.ptr = const_cast<void *>(shared_.output);
  head.dt = shared_.output_dt;
  head.dim0 = E;
  head.dim1 = n_vocab;
  head.bytes = (std::size_t)tensor_bytes(shared_.output_dt, (std::uint64_t)E * n_vocab);
  if (!proj(head, d_hout_, d_logits_, n_vocab, E, err)) return false;
  emit("result_output", d_logits_, n_vocab);
  std::vector<float> tmp;
  std::vector<float> *dst = logits != nullptr ? logits : (argmax != nullptr ? &tmp : nullptr);
  if (dst != nullptr) {
    dst->resize(n_vocab);
    if (hipMemcpy(dst->data(), d_logits_, (std::size_t)n_vocab * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess) {
      err = "MTP: logits readback failed";
      return false;
    }
  }
  if (argmax != nullptr) {
    // same rule as the trunk's greedy path: the first maximum in id order
    // (Sampler's temp <= 0 branch). The logits readback above is the sync point,
    // so this costs nothing extra.
    const std::vector<float> &v = *dst;
    *argmax = 0;
    for (int i = 1; i < n_vocab; ++i) {
      if (v[(std::size_t)i] > v[(std::size_t)*argmax]) *argmax = i;
    }
  }
  last_ms_ = std::chrono::duration<double, std::milli>(clock::now() - t0).count();
  return true;
}

// h_nextn of the last step (the post shared_head_norm hidden state). Returns
// false when the copy failed: the caller would otherwise feed zeros back as the
// previous hidden state (review finding H2). `have_h_out_ == false` is not a
// failure, it is "no draft ran yet" and leaves `h` as it was.
inline bool MtpHead::read_h_out(std::vector<float> &h) const {
  if (!have_h_out_) return true;
  h.resize((std::size_t)d_.n_embd);
  return hipMemcpy(h.data(), d_hout_, (std::size_t)d_.n_embd * sizeof(float),
                   hipMemcpyDeviceToHost) == hipSuccess;
}

inline void MtpHead::release() {
  auto fr = [](W &w) {
    if (w.ptr) (void)hipFree(w.ptr);
    w.ptr = nullptr;
  };
  fr(eh_proj_); fr(enorm_); fr(hnorm_); fr(shared_head_norm_);
  fr(attn_norm_); fr(attn_post_norm_); fr(attn_q_); fr(attn_k_); fr(attn_v_); fr(attn_output_);
  fr(attn_q_norm_); fr(attn_k_norm_); fr(ffn_gate_); fr(ffn_up_); fr(ffn_down_);
  float *ptrs[] = {d_e_, d_h_, d_in_, d_x_, d_xn_, d_qf_, d_q_, d_gate_, d_kst_, d_vst_,
                   d_attn_, d_out_, d_ffn_a_, d_ffn_b_, d_hout_, d_logits_};
  for (float *p : ptrs) {
    if (p) (void)hipFree(p);
  }
  // `ptrs` holds copies: nulling the members is what makes release() idempotent
  // (same defect as Graph::release, review finding M1).
  d_e_ = d_h_ = d_in_ = d_x_ = d_xn_ = d_qf_ = d_q_ = d_gate_ = nullptr;
  d_kst_ = d_vst_ = d_attn_ = d_out_ = d_ffn_a_ = d_ffn_b_ = d_hout_ = nullptr;
  d_logits_ = nullptr;
  if (d_q8_) (void)hipFree(d_q8_);
  if (d_pos_) (void)hipFree(d_pos_);
  if (d_ck_) (void)hipFree(d_ck_);
  if (d_cv_) (void)hipFree(d_cv_);
  d_q8_ = nullptr;
  d_pos_ = nullptr;
  d_ck_ = nullptr;
  d_cv_ = nullptr;
}

}  // namespace rdna4
