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

#include <algorithm>
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
#include "rdna4/device.h"  // prefill_chunk_cap(): dependencia explicita, nao transitiva
#include "rdna4/dtype.h"
#include "rdna4/gdn.cuh"
#include "rdna4/gemm.cuh"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"
#include "rdna4/model.h"
#include "rdna4/mtp.cuh"
#include "rdna4/nn.cuh"
#include "rdna4/phase_prof.cuh"  // RD_PHASE_PROF: diagnostico opt-in (agente de medicoes)

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
  // Troca o tipo de KV reutilizando TUDO o mais (pesos, buffers, ctx): libera e
  // realoca SOMENTE os caches K/V. Existe para o `--pairs` do check-batch-gpu
  // percorrer varios pares (K,V) num unico processo com um unico upload dos
  // pesos (~11 GiB); so' pode ser chamado depois de init().
  bool reinit_kv(KvType kv_k, KvType kv_v, std::string &err);

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
  // Greedy fast path: argmax on the device instead of copying the whole 993 KB
  // logits vector back every token. The caller may only enable it when the
  // sampler's pre-argmax chain is the identity (temp <= 0 and repeat_penalty == 1
  // or repeat_last_n <= 0), and then it must read last_argmax() instead of
  // `logits`. Ties and NaN behave exactly like the host scan (first strict
  // maximum in id order), which is what makes the greedy ids identical.
  void set_want_argmax(bool want) { want_argmax_ = want; }
  std::int32_t last_argmax() const { return last_argmax_; }
  // Shared by forward_run and forward_batch: both must leave last_argmax_
  // consistent, because either can be the forward that produced the logits the
  // caller is about to sample (review finding R9).
  bool finish_argmax(int n_vocab, std::string &err);
  // --- RD_PHASE_PROF (diagnostico opt-in, agente de medicoes) ---------------
  // Ativa o cronometro de fases por eventos HIP (include/rdna4/phase_prof.cuh).
  // Puramente aditivo: sem profiler o caminho e identico (RD_PHASE e um no-op).
  // Nao altera numeros: os eventos so marcam a fila, nenhum buffer e tocado.
  void set_phase_prof(PhaseProf *p) { prof_ = p; }
  // splits usados na ultima atencao (para o relatorio de trafego de KV)
  int debug_attn_splits() const { return attn_splits_last_; }
  PhaseProf *phase_prof() const { return prof_; }
  // nivel 2 (separar act_quant de matvec dentro de cada proj)
  PhaseProf *pfine() { return (prof_ != nullptr && prof_->fine()) ? prof_ : nullptr; }
  // --- fim RD_PHASE_PROF ---------------------------------------------------
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

  // feat/noite-prefill: RD_ATTN_SPLIT_BATCH=0 forces the per-token split kernel even
// when a batched instantiation exists. Two uses: (a) the gate that proves the
// fallback path -- taken by any (K,V) pair without a batched instantiation, e.g. a
// future KV type -- is bit-exact, (b) an in-binary A/B of the batched split
// attention. Default 1 (batched).
inline bool split_batch_enabled() {
  static const bool v = [] {
    const char *e = getenv("RD_ATTN_SPLIT_BATCH");
    return !(e != nullptr && e[0] == '0');
  }();
  return v;
}

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
  // Teto HOST do chunk: os arrays de pilha do caminho em lote (posicoes) sao deste
  // tamanho. Tem de acompanhar `prefill_chunk_cap()`.
  static constexpr int kMaxChunkHost = 512;
  // Batch sizes the matvec has instantiations for (2/3/4/8/16); anything else is
  // split into a supported chunk plus a per-token tail.
  static int batch_supported(int n) {
    return n == 2 || n == 3 || n == 4 || n == 8 || n == 16;
  }
  // Tamanho de chunk ACEITO pelo caminho em lote. Ate 16 e' o GEMV (com as
  // instanciacoes acima); acima disso quem decide e' o despacho do `proj_batch`:
  // GEMM tilejado onde ele cobre o tipo, GEMV em sub-lotes de 16 onde nao cobre.
  // O teto e' `batch_max_` (RD_PREFILL_CHUNK).
  bool chunk_ok(int n) const {
    return n >= 2 && n <= batch_max_ && (n <= 16 ? batch_supported(n) != 0 : true);
  }

  // ---------------------------------------------------------------------------
  // POLITICA UNICA DE CHUNK DO PREFILL (frente prefill-chunk, 14/09).
  //
  // Ate' hoje CADA consumidor cortava o prompt por conta propria: o CLI em
  // `src/main.hip::prefill_ids` (chunks de 16/8/4/3/2) e o servidor com
  // `forward_tokens(prompt, 0, ...)` -- ou seja, TOKEN A TOKEN, sem lote nenhum.
  // Dois consumidores do mesmo modelo com duas aritmeticas diferentes: com o chunk
  // em 16 a diferenca era invisivel (o GEMV em lote e' bit-exato contra o caminho
  // por token), mas com o GEMM tilejado ligado (`RD_PREFILL_CHUNK`, D2 do
  // docs/plano-prefill.md) o CLI passava a usar o GEMM em parte das projecoes e o
  // servidor continuava no caminho por token -- e ai os dois divergiam no ultimo
  // bit e, as vezes, no ultimo token (gate do `serve`: 2 falhas com chunk 128, 0
  // com 16). Alem disso o servidor nao tinha prefill em lote nenhum: ele
  // prefillava a ~28 tok/s enquanto o CLI ja' fazia ~230.
  //
  // Agora existe UMA funcao, esta, e os dois a chamam. Todo consumidor de prefill
  // de prompt (CLI `run`/`chat`, `bench`, servidor) corta o prompt identicamente
  // por construcao, porque a lista de tamanhos vem de `prefill_chunk_sizes()`, que
  // e' a mesma para todos.
  //
  // Contrato numerico (declarado, nao implicito):
  //   - todo chunk de n <= 16 e' o GEMV em lote e continua BIT-EXATO contra o
  //     caminho por token (contrato do M8, inegociavel; tests/check_batch_gpu.hip
  //     exige bit-exatidao nessa faixa);
  //   - chunks com n > 16 usam o GEMM tilejado, que e' bit-exato contra o
  //     `vec_dot_*` do motor mas NAO contra o `matvec_launch_batch` (particao de k
  //     diferente): tolerancia declarada em tests/check_batch_gpu.hip
  //     (kTolChunkRelL2 / kTolChunkMaxAbs), com a medicao que a justifica.
  // `hidden`/`logits` sao os do ULTIMO token do prompt.
  bool prefill(const std::vector<std::int32_t> &ids, int start_pos, std::vector<float> &hidden,
               std::vector<float> &logits, std::string &err);
  // A lista de tamanhos que `prefill()` tenta, do maior para o menor. Exposta
  // porque e' a POLITICA (e o gate a imprime): qualquer consumidor que queira
  // cortar o prompt "do mesmo jeito" tem de chamar `prefill()`, nao recopiar isto.
  std::vector<int> prefill_chunk_sizes() const;
  // O corte do prompt em chunks: DP sobre o total que minimiza (n. de chunks,
  // n. de chunks de 1 token), so' com tamanhos chunk_ok. E' o que evita a cauda
  // 2+1 do guloso (resto 3 saia como 2 + 1 token) e alinha as caudas aos
  // tamanhos suportados {16,8,4,3,2} sem mudar nenhum caso que ja' era otimo.
  std::vector<int> prefill_plan(std::size_t total) const;

  // feat/noite-prefill: `RD_PREFILL_BATCH=0` runs forward_batch with the
  // per-token scaffolding the M8 path shipped (the attention and the GDN
  // recurrence launched once per token) instead of the batched kernels. It is the
  // in-binary A/B switch for the measurement — same window, same DPM state, same
  // weights — and the fallback if a gate ever disagrees with the batched path.
  bool batch_ops() const {
    if (!batch_env_read_) {
      batch_env_read_ = true;
      const char *e = getenv("RD_PREFILL_BATCH");
      batch_env_ = !(e != nullptr && e[0] == '0');
    }
    return batch_env_;
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
  // V only differs from K's size when the two cache types do; both accessors are
  // needed to report the cache footprint honestly (docs/journal-kv.md §VRAM).
  std::size_t kv_layer_bytes_v() const { return kv_bytes_v_; }
  // Teto do chunk do prefill em lote (RD_PREFILL_CHUNK; default kMaxBatch).
  int batch_cap() const { return batch_max_; }
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
  // Batched *verification* (feat/noite-mtp). Same layer pass as forward_batch(),
  // but it returns the post-output_norm hidden state of EVERY token in the batch
  // (h_all, n*n_embd) and the LM head's logits for EVERY token (logits_all,
  // n*n_vocab). Verification needs both: the trunk's own greedy token at each
  // drafted position is read from rows 0..n-2 of logits_all, and the hidden rows
  // are what the draft block's own KV rows are rebuilt from.
  //
  // Row t of both outputs is the value the per-token path produces for `tokens[t]`
  // at position start_pos+t: the layer pass is forward_batch_layer() (the function
  // tests/check_batch_gpu.hip proves bit-identical to per-token forwards), the
  // final norm is the same rms_norm_launch, and the head is the batched matvec —
  // the same kernel family the M6/M8 gates hold bit-exact against the GEMV path.
  // forward_batch() itself is untouched, so the prefill gates keep their numbers.
  bool forward_batch_all(const std::vector<std::int32_t> &tokens, int start_pos,
                         std::vector<float> &h_all, std::vector<float> &logits_all,
                         std::string &err);

  // Rollback of the recurrent (GDN) state for speculative decoding: a batched
  // verification advances d_state_/d_convst_ for every row it processes, and a
  // rejected draft must not leave that advance behind. These copy the two state
  // buffers device-to-device (156.4 MiB for this model, ~0.5 ms at the measured
  // 633 GB/s) — the KV caches need no snapshot, since attention is causal and
  // every row is rewritten by position before it is read again.
  bool state_snapshot(std::string &err);
  bool state_restore(std::string &err);
  std::size_t state_bytes() const;
  // Byte cost of one snapshot + one restore (the number the journal reports).
  bool state_snapshot_allocated() const { return d_state_snap_ != nullptr; }
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
  // Gather+dequantize the embedding rows for `tokens` into d_dst (n x E f32) in
  // ONE launch. Same per-(row, unit) arithmetic as n dequant_row_launch calls,
  // so bit-identical; only the n-1 extra launches (and their drain) go away.
  bool embed_batch(const std::vector<std::int32_t> &tokens, float *d_dst, int E,
                   std::string &err);
  bool forward_batch_layer(int il, int n, int pos0, std::string &err);
  // One launch for the whole chunk's K and V rows (feat/noite-prefill). The rows
  // of a chunk are contiguous BOTH in the cache ((t*NKV + h) * row_bytes) and in
  // the staging buffers (t*NKV*head_dim + h*head_dim), so a single call of the
  // per-row kernel with `n_tok*NKV*head_dim` as its width reproduces the per-row
  // calls element for element: F32/F16 are elementwise, and Q8_0/Q4_0 walk
  // blocks of 32 in the same packed order.
  bool kv_write_batch(int il, int pos0, const float *d_ksrc, const float *d_vsrc, int n_tok,
                      std::string &err);
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
  bool want_argmax_ = false;
  std::int32_t last_argmax_ = 0;
  int *d_argmax_idx_ = nullptr;
  float *d_argmax_val_ = nullptr;
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
  std::size_t kv_bytes_v_ = 0;  // per-layer V bytes; == kv_bytes_ when kt row == vt row
  // Teto RUNTIME do chunk do prefill em lote (RD_PREFILL_CHUNK, default kMaxBatch).
  // Os buffers de lote sao dimensionados por ele; com o default (16) nada muda.
  int batch_max_ = kMaxBatch;
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
  // feat/noite-prefill: the batched GDN recurrence's staging — `d_qkb_` is
  // [n][q(nkh*S) | k(nkh*S)] so both l2_norm and the delta rule see plain rows,
  // `d_vcb_` is [n][d_inner] in place of the per-token conv slice.
  float *d_qkb_ = nullptr, *d_vcb_ = nullptr;
  mutable bool batch_env_read_ = false;
  mutable bool batch_env_ = true;
  float *d_alphab_ = nullptr, *d_betab_ = nullptr, *d_gate2b_ = nullptr;
  block_q8_1 *d_aqb_ = nullptr;
  std::size_t aq_blocks_per_row_ = 0;
  int *d_posb_ = nullptr;
  // Device-side token ids for the batched embedding gather (embed_batch):
  // [kMaxChunkHost] ints, uploaded once per chunk. Lazy-allocated on first use
  // so a run without the batch path pays nothing for it.
  int *d_emb_ids_ = nullptr;
  // Split-KV attention scratch: [n_head][kAttnMaxSplits][2 + head_dim] (M7).
  float *d_attn_partial_ = nullptr;
  // Valores medidos em include/rdna4/tuning.h (tabela unica); os nomes e o
  // resto da politica de splits seguem exatamente como estavam.
  static constexpr int kAttnMaxSplits = tuned::kAttnMaxSplits;
  // Splits are chosen by context length: below kAttnSplitMin keys one CTA already
  // covers the range, and keeping the unsplit path there keeps the short-context
  // gates bit-identical to the pre-M7 numbers (the golden run, the oracle dumps).
  static constexpr int kAttnSplitMin = tuned::kAttnSplitMin;  // M8: 2048 -> 512 (medido +14,7% a 4K, docs/rocm-estudo.md)
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
    // N-POL (medido, docs/autotuning-gfx1201.md §Atencao): com KV SEM dequant no
    // laco interno (f16/f32) o otimo e um numero FIXO de CTAs, nao uma fracao das
    // chaves. A politica antiga (`keys/kAttnSplitMin`, teto 16) dobrava a grade com
    // o contexto e chegava a 384 CTAs a 16K, onde a celula medida e 4 splits x 16
    // warps = 96 CTAs (kAttnSplitCtasDense/n_head): 1,047x a 4K, 1,089x a 8K,
    // 1,030x a 16K, 1,023x a 32K, 1,020x a 64K -- A/B intercalado de 15 rodadas
    // contra a celula antiga, 15/15 rodadas, piso de ruido 1,001-1,005x. Com KV
    // quantizado o mesmo A/B mede o oposto (4 splits perde 6-13%, 0/15), entao a
    // regra por chaves fica exatamente para ele: o que separa os dois casos e o
    // dequant, nao o numero de chaves.
    if ((kv_k_ == KvType::F16 || kv_k_ == KvType::F32) &&
        (kv_v_ == KvType::F16 || kv_v_ == KvType::F32)) {
      const int nh = n_head();
      int sp = nh > 0 ? (tuned::kAttnSplitCtasDense + nh - 1) / nh : 1;
      if (sp < 1) sp = 1;
      return sp > kAttnMaxSplits ? kAttnMaxSplits : sp;
    }
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
  // feat/noite-mtp: one logits row per batch row (forward_batch_all) and the
  // recurrent-state snapshot (state_snapshot/state_restore). Both are allocated
  // on first use, so a run without --mtp pays nothing for them.
  float *d_logitsb_ = nullptr;
  std::size_t logitsb_rows_ = 0;
  float *d_state_snap_ = nullptr, *d_convst_snap_ = nullptr;
  std::size_t q8_blocks_ = 0;
  int cur_token_ = 0;
  int *d_pos_ = nullptr;
  // Last position uploaded to d_pos_ (-1 = none). full_attn runs once per layer
  // with the same pos, so the 4-byte synchronous H2D only re-issues when pos
  // actually changes -- once per decode token instead of once per layer.
  int pos_cached_ = -1;
  int max_ctx_ = 0;
  // RD_PHASE_PROF: nullptr = desligado (comportamento de producao).
  PhaseProf *prof_ = nullptr;
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
  // in the selected storage type. K and V are sized *separately*: they were
  // allocated with kv_row_bytes(kv_k_) for both, which is only right while the two
  // types have the same row size (the M4 finding in device.h, same shape, in the
  // allocator instead of the budget). With K=q5_0 (22 B/32) and V=q4_1 (20 B/32)
  // that over-allocated every V layer by 10% -- 128 MiB at 131K, see
  // docs/journal-kv.md §VRAM. Every V stride below uses kv_bytes_v_.
  const std::size_t kv_k_bytes =
      (std::size_t)max_ctx * NKV * (std::size_t)kv_row_bytes(kv_k_, HD);
  const std::size_t kv_v_bytes =
      (std::size_t)max_ctx * NKV * (std::size_t)kv_row_bytes(kv_v_, HD);
  kv_bytes_ = kv_k_bytes;
  kv_bytes_v_ = kv_v_bytes;
  if (hipMalloc(&d_k_, (std::size_t)n_attn * kv_k_bytes) != hipSuccess ||
      hipMalloc(&d_v_, (std::size_t)n_attn * kv_v_bytes) != hipSuccess) {
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
    // Zeroed on purpose, like alloc() below/above. hipMalloc returns whatever was
    // in those pages: for the FIRST Graph of a process that is fresh (zeroed) VRAM,
    // but a second Graph in the same process gets pages recycled from the first one
    // and its batch buffers start with the previous graph's activations. 18 x
    // kMaxBatch buffers, ~7.6 MiB of memset per init -- measured irrelevant next to
    // the 10.9 GiB weight upload, and it removes the only place where the engine's
    // behaviour depended on the process's allocation history.
    // Found by the KV front's check-kvquality-gpu, which is the only tool that
    // builds several Graphs in one process: with the un-zeroed version the SECOND
    // graph produced NaN logits at a 64-token context in a window where the first
    // one was fine (docs/journal-kv.md §8.2).
    batch_max_ = prefill_chunk_cap();
    const auto balloc = [&](float *&p, std::size_t n) {
      const std::size_t bytes = (std::size_t)batch_max_ * n * sizeof(float);
      if (hipMalloc(&p, bytes) != hipSuccess) return false;
      return hipMemset(p, 0, bytes) == hipSuccess;
    };
    if (!balloc(d_xb_, E) || !balloc(d_xnb_, E) || !balloc(d_projb_, E) ||
        !balloc(d_qb_, (std::size_t)NH_ * 2 * HD_) || !balloc(d_kb_, (std::size_t)NKV_ * HD_) ||
        !balloc(d_vb_, (std::size_t)NKV_ * HD_) || !balloc(d_attnb_, (std::size_t)NH_ * HD_) ||
        !balloc(d_gateb_, (std::size_t)NH_ * HD_) || !balloc(d_ffnab_, F) ||
        !balloc(d_ffnbb_, F) || !balloc(d_qkvb_, chan) || !balloc(d_zb_, d_inner) ||
        !balloc(d_convb_, chan) || !balloc(d_alphab_, nvh) || !balloc(d_betab_, nvh) ||
        !balloc(d_gate2b_, nvh) || !balloc(d_qkb_, 2 * key_dim) || !balloc(d_vcb_, d_inner)) {
      err = "hipMalloc failed for the batch buffers";
      return false;
    }
    if (hipMalloc(&d_aqb_, (std::size_t)batch_max_ * aq_per_row * sizeof(block_q8_1)) !=
        hipSuccess) {
      err = "hipMalloc failed for the batch activation blocks";
      return false;
    }
    if (hipMalloc(&d_posb_, (std::size_t)batch_max_ * sizeof(int)) != hipSuccess) {
      err = "hipMalloc failed for the batch positions";
      return false;
    }
  }

  // Sized for the whole batch: the batched split attention keeps one partial per
  // (token, head, split) instead of reusing one token's buffer (feat/noite-prefill).
  const std::size_t part_bytes =
      (std::size_t)batch_max_ * attn_partial_bytes(NH, HD, kAttnMaxSplits);
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

// Ve' o comentario na declaracao: troca o tipo de KV sem reenviar os pesos.
// Aloca os novos caches ANTES de liberar os antigos, para um OOM nao deixar o
// grafo sem cache nenhum. Nada mais depende do tipo de KV no init (o staging
// d_kstage_/d_vstage_ e' f32, o scratch de atencao parcial e' por (token,head,
// split) e `attn_splits_for` le kv_k_/kv_v_ em tempo de execucao), entao trocar
// os dois buffers + os tres campos e' a re-inicializacao completa.
inline bool Graph::reinit_kv(KvType kv_k, KvType kv_v, std::string &err) {
  if (max_ctx_ <= 0 || w_.empty()) {
    err = "reinit_kv before init";
    return false;
  }
  const int HD = head_dim(), NKV = n_head_kv();
  const int n_attn = n_layer() - count_recr();
  const std::size_t kv_k_bytes =
      (std::size_t)max_ctx_ * NKV * (std::size_t)kv_row_bytes(kv_k, HD);
  const std::size_t kv_v_bytes =
      (std::size_t)max_ctx_ * NKV * (std::size_t)kv_row_bytes(kv_v, HD);
  void *nk = nullptr, *nv = nullptr;
  if (hipMalloc(&nk, (std::size_t)n_attn * kv_k_bytes) != hipSuccess ||
      hipMalloc(&nv, (std::size_t)n_attn * kv_v_bytes) != hipSuccess) {
    if (nk) (void)hipFree(nk);
    if (nv) (void)hipFree(nv);
    err = "hipMalloc failed (kv cache)";
    return false;
  }
  if (d_k_) (void)hipFree(d_k_);
  if (d_v_) (void)hipFree(d_v_);
  d_k_ = nk;
  d_v_ = nv;
  kv_k_ = kv_k;
  kv_v_ = kv_v;
  kv_bytes_ = kv_k_bytes;
  kv_bytes_v_ = kv_v_bytes;
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
  RD_PHASE(pfine(), "act_quant");  // RD_PHASE_PROF
  quantize_q8_1_kernel<<<(nb + 3) / 4, 128>>>(d_x, (block_q8_1 *)d_q8_, nb);
  if (hipGetLastError() != hipSuccess) {
    err = "quantize_q8_1 launch failed";
    return false;
  }
  RD_PHASE(pfine(), "matvec");  // RD_PHASE_PROF
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
  RD_PHASE(pfine(), "matvec");  // RD_PHASE_PROF
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
inline bool Graph::full_attn(int il, [[maybe_unused]] int t, int pos, std::string &err) {
  const int E = n_embd(), HD = head_dim(), NH = n_head(), NKV = n_head_kv();
  const LayerW &L = w_[il];
  RD_PHASE(prof_, "attn_norm");  // RD_PHASE_PROF
  if (!rms_norm_launch(d_x_, (const float *)L.attn_norm.ptr, d_xn_, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "attn_norm launch failed";
    return false;
  }
  emit("attn_norm", il, d_xn_, E);
  RD_PHASE(prof_, "qkv_proj");  // RD_PHASE_PROF
  if (!proj(L.attn_q, d_xn_, d_proj_, NH * 2 * HD, E, err)) return false;
  // k and v read the same activation as q: reuse its q8_1 blocks
  if (!proj_qq(L.attn_k, d_kstage_, NKV * HD, E, err)) return false;
  if (!proj_qq(L.attn_v, d_vstage_, NKV * HD, E, err)) return false;
  emit("Qcur_full", il, d_proj_, (std::int64_t)NH * 2 * HD);
  emit("Vcur", il, d_vstage_, (std::int64_t)NKV * HD);

  RD_PHASE(prof_, "qk_norm_rope_kv");  // RD_PHASE_PROF
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
  // One 4-byte synchronous H2D per TOKEN, not per layer: forward_run walks the
  // layers with a fixed pos, so re-upload only on change (attn already takes
  // pos by value; rope reads d_pos_[0], which then holds exactly this pos).
  if (pos != pos_cached_) {
    if (hipMemcpy(d_pos_, &pos, sizeof(int), hipMemcpyHostToDevice) != hipSuccess) {
      err = "position upload failed";
      return false;
    }
    pos_cached_ = pos;
  }
  const float base = (float)cfg_.rope_freq_base;
  if (!rope_launch(d_attnout_, 1, NH, HD, n_rot(), base, d_pos_)) return false;
  if (!rope_launch(d_kstage_, 1, NKV, HD, n_rot(), base, d_pos_)) return false;
  emit("Qcur", il, d_attnout_, (std::int64_t)NH * HD);
  emit("Kcur", il, d_kstage_, (std::int64_t)NKV * HD);
  emit("gate_reshaped", il, d_attngate_, (std::int64_t)NH * HD);

  // Decode (n_tok=1): one token's staging rows (d_kstage_/d_vstage_, NKV*HD
  // contiguous floats) and cache rows (NKV contiguous rows at token pos) are
  // contiguous, and kv_store_row_kernel is block-local (one 32-elem block per
  // thread, F32/F16 per element), so a single width=NKV*HD launch stores
  // exactly the bytes the per-head loop below stored -- 2 launches instead of
  // 2*NKV. Same call the batch path uses (there with n_tok=n).
  if (!kv_write_batch(il, pos, d_kstage_, d_vstage_, 1, err)) return false;
  const char *kc = (const char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_;
  const char *vc = (const char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_v_;

  const float scale = 1.0f / std::sqrt((float)HD);
  const int n_keys = pos + 1;
  const int splits = attn_splits_for(n_keys);
  attn_splits_last_ = splits;
  RD_PHASE(prof_, "attention");  // RD_PHASE_PROF
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
  RD_PHASE(prof_, "attn_gate_out");  // RD_PHASE_PROF
  // Fused sigmoid+mul: d_attngate_ still ends up holding the sigmoid output, so
  // the gate_sigmoid dump below reads the same values (see nn.cuh).
  if (!sigmoid_gate_launch(d_attngate_, d_attnout_, NH * HD)) {
    err = "attn gate launch failed";
    return false;
  }
  emit("gate_sigmoid", il, d_attngate_, (std::int64_t)NH * HD);
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
  char *vrow = (char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_v_ +
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

// Same rows, one launch for the whole chunk (see the declaration).
inline bool Graph::kv_write_batch(int il, int pos0, const float *d_ksrc, const float *d_vsrc,
                                  int n_tok, std::string &err) {
  const int HD = head_dim(), NKV = n_head_kv();
  const std::size_t krow_bytes = kv_row_bytes(kv_k_, HD);
  const std::size_t vrow_bytes = kv_row_bytes(kv_v_, HD);
  // O stride POR CAMADA tem de ser o da alocacao de cada cache: d_k_ tem
  // n_attn * kv_k_bytes e d_v_ tem n_attn * kv_v_bytes. Usar kv_bytes_ (o de K) na V
  // enderecava a camada il fora do slot dela sempre que a linha de K fosse MAIOR que
  // a de V -- overflow de heap silencioso, e PAGE FAULT quando a memoria logo depois
  // de d_v_ nao estava mapeada. Medido: K=f16/V=q4_1 (86,5 MB alem do fim da
  // alocacao) e K=q5_0/V=q4_1 (3,9 MB) davam `Memory access fault`; K=q8_0/V=q4_1
  // (27,5 MB) passava *corrompendo* memoria de outra alocacao, que e' pior.
  // Ver docs/estudo-prefill.md 3h.
  char *krow = (char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_ +
               (std::size_t)pos0 * NKV * krow_bytes;
  char *vrow = (char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_v_ +
               (std::size_t)pos0 * NKV * vrow_bytes;
  const int width = n_tok * NKV * HD;
  if (!kv_store_row_launch(kv_k_, d_ksrc, krow, width) ||
      !kv_store_row_launch(kv_v_, d_vsrc, vrow, width)) {
    err = "batched kv_store_row launch failed";
    return false;
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

  RD_PHASE(prof_, "attn_norm");  // RD_PHASE_PROF
  if (!rms_norm_launch(d_x_, (const float *)L.attn_norm.ptr, d_xn_, 1, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "attn_norm launch failed";
    return false;
  }
  emit("attn_norm", il, d_xn_, E);
  RD_PHASE(prof_, "gdn_proj");  // RD_PHASE_PROF
  if (!proj(L.attn_qkv, d_xn_, d_qkv_, chan, E, err)) return false;
  // all four GDN projections read d_xn_ unchanged: one quantization serves all
  if (!proj_qq(L.attn_gate, d_z_, d_inner, E, err)) return false;
  if (!proj_qq(L.ssm_beta, d_beta_, nvh, E, err)) return false;
  if (!proj_qq(L.ssm_alpha, d_alpha_, nvh, E, err)) return false;
  emit("linear_attn_qkv_mixed", il, d_qkv_, chan);
  emit("z", il, d_z_, d_inner);
  emit("beta", il, d_beta_, nvh);
  emit("alpha", il, d_alpha_, nvh);

  RD_PHASE(prof_, "gdn_scalars");  // RD_PHASE_PROF
  // 4 scalar launches -> 1 (see gdn.cuh): d_beta_/d_gate_ keep their old values;
  // d_alpha_ now holds softplus(a) instead of the dead a+dt, so the a_softplus
  // dump reads from there.
  if (!gdn_scalars_launch(d_beta_, d_alpha_, d_gate_, (const float *)L.ssm_dt.ptr,
                          (const float *)L.ssm_a.ptr, nvh)) {
    err = "gdn scalars launch failed";
    return false;
  }
  emit("beta_sigmoid", il, d_beta_, nvh);
  emit("a_softplus", il, d_alpha_, nvh);
  emit("gate", il, d_gate_, nvh);

  RD_PHASE(prof_, "gdn_conv");  // RD_PHASE_PROF
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
  RD_PHASE(prof_, "gdn_l2norm");  // RD_PHASE_PROF
  if (!l2_norm_launch(q_c, q_c, nkh, S, (float)cfg_.rms_norm_eps)) return false;
  if (!l2_norm_launch(k_c, k_c, nkh, S, (float)cfg_.rms_norm_eps)) return false;

  if (cb_) emit("state_predelta", il, state, (std::int64_t)nvh * S * S);
  RD_PHASE(prof_, "gdn_delta");  // RD_PHASE_PROF
  if (!delta_rule_launch(q_c, k_c, v_c, d_gate_, d_beta_, state, v_c, nvh, nkh, S)) {
    err = "delta_rule launch failed";
    return false;
  }
  if (cb_) emit("new_state", il, state, (std::int64_t)nvh * S * S);
  RD_PHASE(prof_, "gdn_norm_silu");  // RD_PHASE_PROF
  if (!rms_norm_launch(v_c, (const float *)L.ssm_norm.ptr, v_c, nvh, S,
                       (float)cfg_.rms_norm_eps)) {
    err = "ssm_norm launch failed";
    return false;
  }
  // Fused silu+mul: d_z_ still ends up holding silu(z) (see nn.cuh).
  if (!silu_gate_launch(d_z_, v_c, v_c, d_inner)) {
    err = "gdn output launch failed";
    return false;
  }
  emit("final_output", il, v_c, d_inner);

  RD_PHASE(prof_, "gdn_out_proj");  // RD_PHASE_PROF
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
  RD_PHASE(prof_, "ffn_gate_up");  // RD_PHASE_PROF
  if (!proj(L.ffn_gate, d_xn_, d_ffn_a_, F, E, err)) return false;  // d_xn_ holds attn_post_norm
  // up reads the same activation as gate (silu only touched the output)
  if (!proj_qq(L.ffn_up, d_ffn_b_, F, E, err)) return false;
  // Fused silu+mul: d_ffn_a_ still ends up holding silu(gate)*up (see nn.cuh).
  if (!silu_gate_launch(d_ffn_a_, d_ffn_b_, d_ffn_a_, F)) {
    err = "ffn gate launch failed";
    return false;
  }
  RD_PHASE(prof_, "ffn_down");  // RD_PHASE_PROF
  if (!proj(L.ffn_down, d_ffn_a_, d_ffnout_, E, F, err)) return false;
  emit("ffn_out", il, d_ffnout_, E);
  if (!add_residual(d_ffnout_, d_x_, E, err)) return false;
  return true;
}


// ---------------------------------------------------------------------------
// M8: batched prefill. See the declaration for the contract; the short version is
// "layer-major, projections batched, everything sequential stays sequential".
// ---------------------------------------------------------------------------
// Batched embedding gather: one launch dequantizes the rows ids[0..n) into the
// contiguous d_dst rows. Each (row, unit) pair runs exactly the same device
// call the per-row dequant_row_launch kernels run for that row (same Fn::apply
// arguments, same blockDim per dtype -- see dequant_row.cuh), so the bytes are
// identical and only the n-1 extra launches go away. Grid-strided over
// row*units+unit, so the grid can be capped like the per-row kernels'.
template <class Fn>
__global__ void dequant_gather_kernel_256(const char *__restrict__ base,
                                          std::int64_t row_bytes, const int *__restrict__ ids,
                                          float *__restrict__ dst, std::int64_t E,
                                          std::int64_t units_per_row, std::int64_t n) {
  const std::int64_t total = n * units_per_row;
  for (std::int64_t j = blockIdx.x; j < total; j += gridDim.x) {
    const std::int64_t row = j / units_per_row;
    const std::int64_t u = j % units_per_row;
    Fn::apply(base + (std::int64_t)ids[row] * row_bytes, u, dst + row * E + u * 256,
              threadIdx.x);
  }
}

__global__ void dequant_gather_kernel_q8_0(const char *__restrict__ base,
                                           std::int64_t row_bytes, const int *__restrict__ ids,
                                           float *__restrict__ dst, std::int64_t E,
                                           std::int64_t units_per_row, std::int64_t n) {
  const std::int64_t total = n * units_per_row;
  for (std::int64_t j = blockIdx.x; j < total; j += gridDim.x) {
    const std::int64_t row = j / units_per_row;
    const std::int64_t u = j % units_per_row;
    float2 v;
    // Same call as dequant_kernel_q8_0 (16 threads x 2 elements, the float2 form
    // llama.cpp's get_rows uses for q8_0).
    rdna4::dequantize_q8_0(base + (std::int64_t)ids[row] * row_bytes, u,
                           2 * (int)threadIdx.x, v);
    float *y = dst + row * E + u * 32;
    y[2 * threadIdx.x + 0] = v.x;
    y[2 * threadIdx.x + 1] = v.y;
  }
}

// F32 embedding needs no dequant: plain gather copy, trivially bit-identical.
__global__ void embed_gather_f32_kernel(const float *__restrict__ base,
                                        const int *__restrict__ ids, float *__restrict__ dst,
                                        std::int64_t E, std::int64_t n) {
  const std::int64_t total = n * E;
  for (std::int64_t j = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x; j < total;
       j += (std::int64_t)gridDim.x * blockDim.x) {
    dst[j] = base[(std::int64_t)ids[j / E] * E + j % E];
  }
}

// Same dtype -> blockDim mapping as dequant_row_launch (dequant_row.cuh); anything
// it rejects is rejected here too.
inline bool dequant_gather_launch(DType dt, const void *d_base, std::int64_t row_bytes,
                                  const int *d_ids, float *d_dst, std::int64_t E, int n,
                                  hipStream_t stream = nullptr) {
  if (n <= 0 || E <= 0) return false;
  if (dt == DType::F32) {
    const std::int64_t total = (std::int64_t)n * E;
    const int threads = 256;
    const unsigned grid = (unsigned)((total + threads - 1) / threads);
    embed_gather_f32_kernel<<<grid, threads, 0, stream>>>((const float *)d_base, d_ids, d_dst,
                                                          E, n);
    return hipGetLastError() == hipSuccess;
  }
  const std::int64_t unit = dt == DType::Q8_0 ? 32 : 256;
  if (E % unit != 0) return false;
  const std::int64_t units = E / unit;
  const std::int64_t total = (std::int64_t)n * units;
  const unsigned grid = (unsigned)(total < 4096 ? total : 4096);
  const char *base = (const char *)d_base;
  if (dt == DType::Q8_0) {
    dequant_gather_kernel_q8_0<<<grid, 16, 0, stream>>>(base, row_bytes, d_ids, d_dst, E, units,
                                                        n);
    return hipGetLastError() == hipSuccess;
  }
  if (dt == DType::IQ4_NL) {
    dequant_gather_kernel_256<FnIQ4NL><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
    return hipGetLastError() == hipSuccess;
  }
  switch (dt) {
    case DType::Q2_K:
      dequant_gather_kernel_256<FnQ2K><<<grid, 64, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
      break;
    case DType::Q3_K:
      dequant_gather_kernel_256<FnQ3K><<<grid, 64, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
      break;
    case DType::Q4_K:
      dequant_gather_kernel_256<FnQ4K><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
      break;
    case DType::Q5_K:
      dequant_gather_kernel_256<FnQ5K><<<grid, 64, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
      break;
    case DType::Q6_K:
      dequant_gather_kernel_256<FnQ6K><<<grid, 64, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                E, units, n);
      break;
    case DType::IQ2_XXS:
      dequant_gather_kernel_256<FnIQ2XXS><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids,
                                                                   d_dst, E, units, n);
      break;
    case DType::IQ2_XS:
      dequant_gather_kernel_256<FnIQ2XS><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                  E, units, n);
      break;
    case DType::IQ3_XXS:
      dequant_gather_kernel_256<FnIQ3XXS><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids,
                                                                   d_dst, E, units, n);
      break;
    case DType::IQ1_S:
      dequant_gather_kernel_256<FnIQ1S><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                 E, units, n);
      break;
    case DType::IQ3_S:
      dequant_gather_kernel_256<FnIQ3S><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                 E, units, n);
      break;
    case DType::IQ2_S:
      dequant_gather_kernel_256<FnIQ2S><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids, d_dst,
                                                                 E, units, n);
      break;
    case DType::IQ4_XS:
      dequant_gather_kernel_256<FnIQ4XS><<<grid, 32, 0, stream>>>(base, row_bytes, d_ids,
                                                                  d_dst, E, units, n);
      break;
    default: return false;
  }
  return hipGetLastError() == hipSuccess;
}

inline bool Graph::embed_batch(const std::vector<std::int32_t> &tokens, float *d_dst, int E,
                               std::string &err) {
  const int n = (int)tokens.size();
  if (n <= 0 || n > kMaxChunkHost) {
    err = "embed batch size out of range";
    return false;
  }
  if (d_emb_ids_ == nullptr) {
    if (hipMalloc(&d_emb_ids_, (std::size_t)kMaxChunkHost * sizeof(int)) != hipSuccess) {
      err = "embed batch ids alloc failed";
      return false;
    }
  }
  // int32_t token ids are 32-bit values; the kernel indexes rows with them.
  if (hipMemcpy(d_emb_ids_, tokens.data(), (std::size_t)n * sizeof(std::int32_t),
                hipMemcpyHostToDevice) != hipSuccess) {
    err = "embed batch ids upload failed";
    return false;
  }
  const std::int64_t emb_row =
      (std::int64_t)tensor_bytes(tok_embd_.dt, (std::uint64_t)E);
  if (!dequant_gather_launch(tok_embd_.dt, tok_embd_.ptr, emb_row, d_emb_ids_, d_dst, E, n)) {
    err = "batch embedding gather failed";
    return false;
  }
  return true;
}

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
  RD_PHASE(pfine(), "act_quant");  // RD_PHASE_PROF (frente prefill: caminho em lote)
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
  if (n < 2 || n > batch_max_) {
    err = "unsupported batch size " + std::to_string(n);
    return false;
  }
  if (!act_ready && !quantize_batch(d_x, ncols, n, err)) return false;
  const std::int64_t act_stride = ncols / QK8_1;
  RD_PHASE(pfine(), "matvec");  // RD_PHASE_PROF (frente prefill: caminho em lote)

  // ---------------------------------------------------------------------
  // Despacho GEMV x GEMM (D2 do plano do prefill, docs/plano-prefill.md).
  //
  // Ate 16 tokens o GEMV em lote e' o caminho certo -- medido: o GEMM tilejado
  // rende 0,96x ali, porque o tile nao amortiza o staging da dequantizacao. Acima
  // disso o GEMV regride (0,82x em N=32, 0,29x em N=64, em iq3_s) e o GEMM e' o
  // unico caminho: 2,31x em M=64, 2,69x em M=128 e 3,42x em M=512 contra o chunk
  // de 16 (bench-gemm-engine-gpu, min de 5, medido antes de cronometrar).
  //
  // `gemm_launch` reproduz a aritmetica do `vec_dot_*` do motor bit a bit (0 de
  // 4096 elementos, max ulp 0, nos tres tipos) e devolve false para tudo o que nao
  // cobre (68,0 % dos bytes deste modelo: iq3_s, iq3_xxs, iq4_xs). Para os tipos
  // fora da cobertura, com n > 16, o caminho e' o GEMV em SUB-LOTES de 16: o kernel
  // escreve `d_o[n*nrows + row]` e le `d_a + n*act_stride`, entao deslocar os dois
  // por t0 resolve -- e' GEMV puro, sem kernel novo, e mantem o resto do chunk
  // correto ainda que mais lento.
  // ---------------------------------------------------------------------
  if (n <= 16 && batch_supported(n)) {
    if (!matvec_launch_batch((int)w.dt, w.ptr, d_aqb_, d_y, nrows, ncols, act_stride, n, nullptr)) {
      err = "matvec_launch_batch failed";
      return false;
    }
    return true;
  }
  if (rdna4::gemm_launch((int)w.dt, w.ptr, d_aqb_, d_y, nrows, ncols, act_stride, n, nullptr)) {
    return true;
  }
  for (int t0 = 0; t0 < n; t0 += 16) {
    const int nb = n - t0 < 16 ? n - t0 : 16;
    if (nb < 2) {
      // Cauda de 1 token: o GEMV POR TOKEN (o caminho do decode) e' a resposta --
      // o em lote nao tem instanciacao para n=1. Achado com chunk 17 (RD_PREFILL_CHUNK=17
      // em camada cujo tipo nao esta' coberto pelo GEMM, ex.: blk.0.ffn_up e' iq1_s).
      if (!matvec_launch((int)w.dt, w.ptr, d_aqb_ + (std::size_t)t0 * act_stride,
                         d_y + (std::size_t)t0 * nrows, nrows, ncols, nullptr)) {
        err = "matvec_launch (single-token tail) failed";
        return false;
      }
      continue;
    }
    if (!matvec_launch_batch((int)w.dt, w.ptr, d_aqb_ + (std::size_t)t0 * act_stride,
                             d_y + (std::size_t)t0 * nrows, nrows, ncols, act_stride, nb,
                             nullptr)) {
      err = "matvec_launch_batch (sub-batch) failed";
      return false;
    }
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
  // RD_PHASE_PROF: os mesmos nomes de bucket do caminho por token, para as duas
  // tabelas (decode e prefill em lote) serem comparáveis fase a fase. Marcas
  // inertes sem profiler armado (include/rdna4/phase_prof.cuh).
  RD_PHASE(prof_, L.recr ? "gdn_proj" : "qkv_proj");

  if (!L.recr) {
    // q, k and v share the same activation: quantize once up front and reuse
    // for all three (was: Q quantized inside proj_batch, then quantize_batch
    // re-quantized the same rows before K -- one full batched quantize wasted
    // per layer per chunk). Same kernel, same input, so d_aqb_ is identical.
    if (!quantize_batch(d_xnb_, E, n, err)) return false;
    if (!proj_batch(L.attn_q, d_xnb_, d_qb_, NH * 2 * HD, E, n, true, err)) return false;
    if (!proj_batch(L.attn_k, d_xnb_, d_kb_, NKV * HD, E, n, true, err)) return false;
    if (!proj_batch(L.attn_v, d_xnb_, d_vb_, NKV * HD, E, n, true, err)) return false;
    if (batch_ops()) {
      // ---- BATCHED (feat/noite-prefill): one launch per stage for the chunk ---
      // Every one of these is element- or row-local, so folding the N tokens into
      // one launch reproduces the per-token arithmetic exactly; only the ATTENTION
      // is per token when its key range would be split across CTAs (different
      // summation order -- see attn.cuh).
      RD_PHASE(prof_, "qk_norm_rope_kv");  // RD_PHASE_PROF
      if (!deinterleave_q_gate_batch_launch(d_qb_, d_attnb_, d_gateb_, NH, HD, n)) {
        err = "batch deinterleave launch failed";
        return false;
      }
      if (!rms_norm_launch(d_attnb_, (const float *)L.attn_q_norm.ptr, d_attnb_,
                           (std::int64_t)n * NH, HD, eps) ||
          !rms_norm_launch(d_kb_, (const float *)L.attn_k_norm.ptr, d_kb_,
                           (std::int64_t)n * NKV, HD, eps)) {
        err = "batch qk norm launch failed";
        return false;
      }
      // rope_kernel is already multi-token: pos[] holds this chunk's positions
      if (!rope_launch(d_attnb_, n, NH, HD, n_rot(), base, d_posb_) ||
          !rope_launch(d_kb_, n, NKV, HD, n_rot(), base, d_posb_)) {
        err = "batch rope launch failed";
        return false;
      }
      if (!kv_write_batch(il, pos0, d_kb_, d_vb_, n, err)) return false;
      RD_PHASE(prof_, "attention");  // RD_PHASE_PROF
      const char *kc = (const char *)d_k_ + (std::size_t)attn_slot(il) * kv_bytes_;
      // V usa o stride da alocacao de V (ver kv_write_batch): com K maior que V o
      // kv_bytes_ levava a leitura para fora do slot desta camada.
      const char *vc = (const char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_v_;
      // splits is non-decreasing in pos, so the last token's key count decides
      // whether the whole chunk can use the batched (unsplit) kernel.
      const int splits = attn_splits_for(pos0 + n);
      attn_splits_last_ = splits;
      const float scale = 1.0f / std::sqrt((float)HD);
      if (splits > 1) {
        // Long context: the key range is split across CTAs and the split COUNT
        // depends on the token's position. splits is non-decreasing in pos, so the
        // chunk is a sequence of contiguous runs of equal split count; inside a run
        // every token's key assignment and merge order is exactly what the
        // per-token split kernel does (same WPB, same j = w + WPB*s + k*WPB*sp,
        // same merge), so the batched launch is bit-identical and only removes
        // N-1 (kernel, merge) pairs per layer.
        int t0 = 0;
        while (t0 < n) {
          const int sp = attn_splits_for(pos0 + t0 + 1);
          int t1 = t0 + 1;
          while (t1 < n && attn_splits_for(pos0 + t1 + 1) == sp) ++t1;
          const int cnt = t1 - t0;
          float *an0 = d_attnb_ + (std::size_t)t0 * NH * HD;
          if (sp > 1) {
            float *part0 = d_attn_partial_ +
                           (std::size_t)t0 * attn_partial_bytes(NH, HD, kAttnMaxSplits) /
                               sizeof(float);
            // A (K,V) pair with no batched instantiation must cost SPEED, not
            // correctness: fall back to the per-token split kernel, which covers
            // every pair kv.h defines (review finding R11). RD_ATTN_SPLIT_BATCH=0
            // forces this path, which is how the gate exercises it.
            if (!split_batch_enabled() ||
                !attn_split_batch_launch(an0, kc, vc, an0, part0, d_posb_ + t0, cnt, NH, NKV, HD,
                                         scale, kv_k_, kv_v_, sp)) {
              for (int t = t0; t < t1; ++t) {
                float *an = d_attnb_ + (std::size_t)t * NH * HD;
                if (!attn_launch_split(an, kc, vc, an, d_attn_partial_, pos0 + t, NH, NKV, HD,
                                       scale, kv_k_, kv_v_, sp)) {
                  err = "batch attn split fallback launch failed";
                  return false;
                }
              }
            }
          } else if (!attn_batch_launch(an0, kc, vc, an0, d_posb_ + t0, cnt, NH, NKV, HD, scale,
                                        kv_k_, kv_v_)) {
            err = "batch attn launch failed";
            return false;
          }
          t0 = t1;
        }
      } else if (!attn_batch_launch(d_attnb_, kc, vc, d_attnb_, d_posb_, n, NH, NKV, HD, scale,
                                    kv_k_, kv_v_)) {
        err = "batch attn (batched) launch failed";
        return false;
      }
      RD_PHASE(prof_, "attn_gate_out");  // RD_PHASE_PROF
      if (!unary_launch(d_gateb_, d_gateb_, (std::int64_t)n * NH * HD, UnOp::Sigmoid) ||
          !mul_launch(d_attnb_, d_gateb_, d_attnb_, (std::int64_t)n * NH * HD)) {
        err = "batch attn gate launch failed";
        return false;
      }
      RD_PHASE(prof_, "attn_out_proj");  // RD_PHASE_PROF
      if (!proj_batch(L.attn_output, d_attnb_, d_projb_, E, NH * HD, n, false, err)) return false;
    } else {
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
      const char *vc = (const char *)d_v_ + (std::size_t)attn_slot(il) * kv_bytes_v_;
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
    }
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

    // qkv, gate, beta and alpha share the same activation: quantize once up
    // front and reuse for all four (was: qkv quantized inside proj_batch, then
    // quantize_batch re-quantized the same rows -- one full batched quantize
    // wasted per layer per chunk). Same kernel, same input, so d_aqb_ is identical.
    if (!quantize_batch(d_xnb_, E, n, err)) return false;
    if (!proj_batch(L.attn_qkv, d_xnb_, d_qkvb_, chan, E, n, true, err)) return false;
    if (!proj_batch(L.attn_gate, d_xnb_, d_zb_, d_inner, E, n, true, err)) return false;
    if (!proj_batch(L.ssm_beta, d_xnb_, d_betab_, nvh, E, n, true, err)) return false;
    if (!proj_batch(L.ssm_alpha, d_xnb_, d_alphab_, nvh, E, n, true, err)) return false;
    if (batch_ops()) {
      // ---- BATCHED (feat/noite-prefill) ------------------------------------
      // The recurrence stays SEQUENTIAL -- the state of token t+1 is built from
      // the state of token t -- but the whole chunk now runs inside one launch per
      // stage, walking the tokens in order internally. Same operations, same
      // order, same rounding: bit-identical (tests/check_batch_gpu.hip). What
      // changes is that the state of the layer is read once per CHUNK instead of
      // once per token (measured 152 GB/s and 69.6 us per delta_rule launch
      // against a 20 us memory floor, docs/medicoes-banda-e-gargalos.md 4.2).
      const std::int64_t nvh_tot = (std::int64_t)n * nvh;
      RD_PHASE(prof_, "gdn_scalars");  // RD_PHASE_PROF
      if (!unary_launch(d_betab_, d_betab_, nvh_tot, UnOp::Sigmoid)) return false;
      if (!add_bcast_launch(d_alphab_, (const float *)L.ssm_dt.ptr, d_alphab_, nvh_tot, nvh)) {
        return false;
      }
      if (!unary_launch(d_alphab_, d_gate2b_, nvh_tot, UnOp::Softplus)) return false;
      if (!mul_bcast_launch(d_gate2b_, (const float *)L.ssm_a.ptr, d_gate2b_, nvh_tot, nvh)) {
        return false;
      }
      // conv1d: q|k go to d_qkb_ as contiguous rows (so l2_norm and the delta
      // rule see plain row arrays), v goes to d_vcb_ as n rows of d_inner
      RD_PHASE(prof_, "gdn_conv");  // RD_PHASE_PROF
      if (!conv1d_state_batch_launch(d_qkvb_, (const float *)L.ssm_conv1d.ptr, d_qkb_, d_vcb_,
                                     convst, chan, K, key_dim, d_inner, n)) {
        err = "batch conv1d launch failed";
        return false;
      }
      RD_PHASE(prof_, "gdn_l2norm");  // RD_PHASE_PROF
      if (!l2_norm_launch(d_qkb_, d_qkb_, (std::int64_t)n * 2 * nkh, S, eps)) {
        err = "batch l2 norm launch failed";
        return false;
      }
      RD_PHASE(prof_, "gdn_delta");  // RD_PHASE_PROF
      // Blocked multi-token scan (gdn.cuh): the chunk's n tokens still run in
      // order inside ONE launch -- same operations, same order, same rounding
      // per (token, row), so bit-identical to the per-token loop below -- but
      // each CTA keeps its 32 state rows resident in LDS across the tokens
      // instead of round-tripping them through DRAM twice per token.
      // RD_GDN_RESIDENT=0 restores delta_rule_batch_launch (same contract);
      // odd S also falls back inside the launcher. The per-token loop in the
      // else branch stays the fallback for RD_PREFILL_BATCH=0.
      static const bool gdn_resident = [] {
        const char *e = std::getenv("RD_GDN_RESIDENT");
        return !(e != nullptr && e[0] == '0');
      }();
      const bool gdn_ok =
          gdn_resident ? delta_rule_batch_resident_launch(d_qkb_, d_vcb_, d_gate2b_, d_betab_, state,
                                                          d_vcb_, nvh, nkh, S, nvh, key_dim,
                                                          d_inner, n)
                       : delta_rule_batch_launch(d_qkb_, d_vcb_, d_gate2b_, d_betab_, state, d_vcb_,
                                                 nvh, nkh, S, nvh, key_dim, d_inner, n);
      if (!gdn_ok) {
        err = "batch delta rule launch failed";
        return false;
      }
      RD_PHASE(prof_, "gdn_norm_silu");  // RD_PHASE_PROF
      if (!rms_norm_launch(d_vcb_, (const float *)L.ssm_norm.ptr, d_vcb_,
                           (std::int64_t)n * nvh, S, eps)) {
        err = "batch ssm_norm launch failed";
        return false;
      }
      if (!unary_launch(d_zb_, d_zb_, (std::int64_t)n * d_inner, UnOp::Silu) ||
          !mul_launch(d_vcb_, d_zb_, d_vcb_, (std::int64_t)n * d_inner)) {
        err = "batch gdn output launch failed";
        return false;
      }
      // no copy: the delta rule wrote into d_vcb_ in place, which is exactly the
      // buffer the batched ssm_out projection reads
      RD_PHASE(prof_, "gdn_out_proj");  // RD_PHASE_PROF
      if (!proj_batch(L.ssm_out, d_vcb_, d_projb_, E, d_inner, n, false, err)) return false;
    } else {
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
  }

  // residual + post-attention norm
  RD_PHASE(prof_, "post_norm");  // RD_PHASE_PROF
  if (!add_launch(d_projb_, d_xb_, d_xb_, (std::int64_t)n * E)) {
    err = "batch residual launch failed";
    return false;
  }
  if (!rms_norm_launch(d_xb_, (const float *)L.attn_post_norm.ptr, d_xnb_, n, E, eps)) {
    err = "batch post_attn_norm launch failed";
    return false;
  }

  // FFN (gate and up share the activation)
  RD_PHASE(prof_, "ffn_gate_up");  // RD_PHASE_PROF
  if (!proj_batch(L.ffn_gate, d_xnb_, d_ffnab_, F, E, n, false, err)) return false;
  if (!proj_batch(L.ffn_up, d_xnb_, d_ffnbb_, F, E, n, true, err)) return false;
  if (!unary_launch(d_ffnab_, d_ffnab_, (std::int64_t)n * F, UnOp::Silu)) return false;
  if (!mul_launch(d_ffnab_, d_ffnbb_, d_ffnab_, (std::int64_t)n * F)) return false;
  RD_PHASE(prof_, "ffn_down");  // RD_PHASE_PROF
  if (!proj_batch(L.ffn_down, d_ffnab_, d_projb_, E, F, n, false, err)) return false;
  RD_PHASE(prof_, "ffn_residual");  // RD_PHASE_PROF
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
  if (!chunk_ok(n)) {
    err = "forward_batch needs 2.." + std::to_string(batch_max_) + " tokens";
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
  // kMaxChunkHost, nao kMaxBatch: este array era de 16 ints e recebia `n` posicoes,
  // entao com o chunk elevado (RD_PREFILL_CHUNK=64) ele corrompia a PILHA -- o
  // sintoma aparecia depois, como SIGSEGV no readback do fim do forward_batch.
  int pos_host[kMaxChunkHost];
  for (int t = 0; t < n; ++t) pos_host[t] = start_pos + t;
  if (hipMemcpy(d_posb_, pos_host, (std::size_t)n * sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess) {
    err = "batch position upload failed";
    return false;
  }

  // embeddings: one batched gather+dequant launch for the whole chunk
  // (bit-identical to one dequant_row_launch per token row, see embed_batch).
  if (!embed_batch(tokens, d_xb_, E, err)) return false;

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
  if (want_argmax_ && !finish_argmax(n_vocab_i, err)) return false;
  logits.resize(n_vocab_i);
  if (hipMemcpy(logits.data(), d_logits_, (std::size_t)n_vocab_i * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    err = "batch logits readback failed";
    return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// A POLITICA UNICA DE CHUNK DO PREFILL -- ver a declaracao, no topo da classe,
// para o porque' (CLI e servidor cortavam o prompt de dois jeitos diferentes).
//
// Esta funcao e' a extracao LITERAL do `prefill_ids` que vivia em
// `src/main.hip:162-195`: a mesma lista de tamanhos, a mesma escolha do maior que
// cabe (`for (int sz : sizes) if (resta >= sz) { take = sz; break; }`) e a mesma
// cauda. Com `RD_PREFILL_CHUNK=16` a lista e' exatamente {16,8,4,3,2} e o
// comportamento e' byte a byte o que embarcava (era o unico caminho, entao os
// numeros de 104,86 tok/s e o golden atual foram medidos com ele).
//
// Detalhe da cauda (frente prefill, 16/09): com cap > 16 a lista descia
// {cap, cap/2, ..., 2} e so' DEPOIS recebia os tamanhos que faltam de {16,8,4,3,2},
// entao o 3 caia no fim da lista e uma cauda de 3 tokens saia como 2 + 1 token (com
// cap = 16, onde a lista e' {16,8,4,3,2}, ela saia como um chunk de 3). Corrigido
// em dois pontos: (a) os tamanhos que faltam entram na posicao ordenada, nao no
// fim -- a lista sai em ordem decrescente e o guloso ja' escolheria 3 antes de 2;
// (b) o corte deixou de ser guloso: `prefill_plan()` (abaixo) planeja o prompt
// inteiro por DP, minimizando (n. de chunks, n. de chunks de 1 token), entao
// restos como 5 saem como 3+2 (nao 4+1) e 13 como 8+3+2 (nao 8+4+1), com o mesmo
// numero de leituras de peso. Casos que ja' eram otimos (1024 = 8x128, 7 = 4+3)
// saem identicos -- so' a cauda burra mudou.
// ---------------------------------------------------------------------------
inline std::vector<int> Graph::prefill_chunk_sizes() const {
  const int cap = batch_max_;
  std::vector<int> sizes;
  for (int sz = cap; sz >= 2; sz >>= 1) sizes.push_back(sz);
  if (cap > 16) {
    for (int sz : {16, 8, 4, 3, 2}) {
      bool tem = false;
      for (int x : sizes) tem = tem || x == sz;
      if (!tem) sizes.push_back(sz);
    }
    // Ordem decrescente: sem isto o 2 precedia o 3 e o resto 3 caia no 2+1.
    std::sort(sizes.begin(), sizes.end(), std::greater<int>());
  } else {
    sizes = {16, 8, 4, 3, 2};
  }
  return sizes;
}

inline std::vector<int> Graph::prefill_plan(std::size_t total) const {
  const std::vector<int> sizes = prefill_chunk_sizes();
  // Candidatos: os tamanhos da politica que passam no chunk_ok (o chain de
  // halving de um cap nao-potencia-de-2 pode trazer 6/12, que o GEMV em lote
  // nao instancia -- o guloso antigo os aceitaria e o forward_batch falharia),
  // do maior para o menor, mais o fallback de 1 token por ultimo.
  std::vector<int> cand;
  for (int sz : sizes) {
    if (chunk_ok(sz)) cand.push_back(sz);
  }
  cand.push_back(1);
  const int INF = (int)total + 1000;
  std::vector<int> cost(total + 1, INF), ones(total + 1, INF);
  cost[0] = 0;
  ones[0] = 0;
  for (std::size_t r = 1; r <= total; ++r) {
    for (int p : cand) {
      if ((std::size_t)p > r) continue;
      const int cc = cost[r - (std::size_t)p] + 1;
      const int oo = ones[r - (std::size_t)p] + (p == 1 ? 1 : 0);
      // Empate desempatado pelo MAIOR p (cand esta' em ordem decrescente e a
      // melhora e' estrita): mantem a forma larger-first do guloso antigo onde
      // ele ja' era otimo.
      if (cc < cost[r] || (cc == cost[r] && oo < ones[r])) {
        cost[r] = cc;
        ones[r] = oo;
      }
    }
  }
  std::vector<int> plan;
  // Reconstrucao da FRENTE para tras (maior p valido primeiro): onde o guloso
  // antigo ja' era otimo o plano sai IDENTICO a ele (ex.: 7 = 4+3, 1024 = 8x128,
  // 129 = 128+1) -- so' a cauda burra muda (5 = 3+2, 13 = 8+3+2). Em particular
  // a cauda continua no FIM, como antes.
  std::size_t r = total;
  while (r > 0) {
    int take = 1;  // inalcancavel (cand sempre tem o 1); cai no guloso seguro
    for (int p : cand) {
      if ((std::size_t)p > r) continue;
      if (cost[r - (std::size_t)p] + 1 == cost[r] &&
          ones[r - (std::size_t)p] + (p == 1 ? 1 : 0) == ones[r]) {
        take = p;
        break;
      }
    }
    plan.push_back(take);
    r -= (std::size_t)take;
  }
  return plan;
}

inline bool Graph::prefill(const std::vector<std::int32_t> &ids, int start_pos,
                           std::vector<float> &hidden, std::vector<float> &logits,
                           std::string &err) {
  if (ids.empty()) {
    err = "the prompt must not be empty";
    return false;
  }
  const std::vector<int> plan = prefill_plan(ids.size());
  std::size_t pos = 0;
  for (int take_i : plan) {
    const std::size_t take = (std::size_t)take_i;
    const int p = start_pos + (int)pos;
    if (take == 1) {
      // Cauda de 1 token: o caminho por token. O em lote nao tem instanciacao
      // para n = 1 e este caminho e' o proprio caminho por token, entao nao ha'
      // nada a tolerar aqui.
      const std::vector<std::int32_t> one{ids[pos]};
      if (!forward_tokens(one, p, hidden, logits, err)) return false;
    } else {
      const std::vector<std::int32_t> chunk(ids.begin() + (long)pos,
                                            ids.begin() + (long)(pos + take));
      if (!forward_batch(chunk, p, hidden, logits, err)) return false;
    }
    pos += take;
  }
  return true;
}

// feat/noite-mtp: batched verification. Everything up to the layer loop is the
// same sequence forward_batch() runs (same checks, same one-shot position upload,
// same per-row embedding dequant), then the final norm is applied to EVERY row and
// the shared LM head runs once for the whole batch through the batched matvec.
inline bool Graph::forward_batch_all(const std::vector<std::int32_t> &tokens, int start_pos,
                                     std::vector<float> &h_all, std::vector<float> &logits_all,
                                     std::string &err) {
  const int E = n_embd();
  const int n = (int)tokens.size();
  if (!chunk_ok(n)) {
    err = "forward_batch_all needs 2.." + std::to_string(batch_max_) + " tokens";
    return false;
  }
  const std::int32_t n_vocab_in = (std::int32_t)tok_embd_.dim1;
  for (std::int32_t tk : tokens) {
    if (tk < 0 || tk >= n_vocab_in) {
      err = "token id " + std::to_string(tk) + " out of range [0, " + std::to_string(n_vocab_in) +
            ")";
      return false;
    }
  }
  if (start_pos < 0 || (std::size_t)start_pos + (std::size_t)n > (std::size_t)max_ctx_) {
    err = "positions exceed the allocated context (" + std::to_string(max_ctx_) + ")";
    return false;
  }

  // kMaxChunkHost, nao kMaxBatch: este array era de 16 ints e recebia `n` posicoes,
  // entao com o chunk elevado (RD_PREFILL_CHUNK=64) ele corrompia a PILHA -- o
  // sintoma aparecia depois, como SIGSEGV no readback do fim do forward_batch.
  int pos_host[kMaxChunkHost];
  for (int t = 0; t < n; ++t) pos_host[t] = start_pos + t;
  if (hipMemcpy(d_posb_, pos_host, (std::size_t)n * sizeof(int), hipMemcpyHostToDevice) !=
      hipSuccess) {
    err = "batch position upload failed";
    return false;
  }
  // embeddings: one batched gather+dequant launch for the whole chunk
  // (bit-identical to one dequant_row_launch per token row, see embed_batch).
  if (!embed_batch(tokens, d_xb_, E, err)) return false;

  for (int il = 0; il < debug_layer_limit(); ++il) {
    if (!forward_batch_layer(il, n, start_pos, err)) {
      err = "layer " + std::to_string(il) + ": " + err;
      return false;
    }
  }

  // final norm over every row, in place: row t becomes the post-output_norm h of
  // position start_pos+t, i.e. exactly what forward_tokens() returns for it
  if (!rms_norm_launch(d_xb_, (const float *)output_norm_.ptr, d_xb_, n, E,
                       (float)cfg_.rms_norm_eps)) {
    err = "batch output_norm launch failed";
    return false;
  }
  h_all.resize((std::size_t)n * E);
  if (hipMemcpy(h_all.data(), d_xb_, (std::size_t)n * E * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    err = "batch hidden readback failed";
    return false;
  }

  const int n_vocab = output_.dim1;
  if (logitsb_rows_ < (std::size_t)n) {
    if (d_logitsb_ != nullptr) {
      (void)hipFree(d_logitsb_);
      d_logitsb_ = nullptr;
      logitsb_rows_ = 0;
    }
    if (hipMalloc(&d_logitsb_, (std::size_t)n * (std::size_t)n_vocab * sizeof(float)) !=
        hipSuccess) {
      err = "hipMalloc batch logits failed";
      return false;
    }
    logitsb_rows_ = (std::size_t)n;
  }
  if (!proj_batch(output_, d_xb_, d_logitsb_, n_vocab, E, n, false, err)) return false;
  logits_all.resize((std::size_t)n * (std::size_t)n_vocab);
  if (hipMemcpy(logits_all.data(), d_logitsb_,
                (std::size_t)n * (std::size_t)n_vocab * sizeof(float),
                hipMemcpyDeviceToHost) != hipSuccess) {
    err = "batch logits readback failed";
    return false;
  }
  return true;
}

// Row layout of the two recurrent buffers (d_state_ is [layer][nvh][S][S],
// d_convst_ is [layer][K-1][chan]); both are contiguous allocations, so one
// hipMemcpy each is the whole snapshot.
inline std::size_t Graph::state_bytes() const {
  const int n_recr = count_recr();
  if (n_recr == 0) return 0;
  const int S = (int)cfg_.ssm_state_size;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int K = (int)cfg_.ssm_conv_kernel;
  const int chan = 2 * (int)cfg_.ssm_group_count * S + nvh * S;
  return ((std::size_t)n_recr * (std::size_t)nvh * (std::size_t)S * (std::size_t)S +
          (std::size_t)n_recr * (std::size_t)(K - 1) * (std::size_t)chan) *
         sizeof(float);
}

inline bool Graph::state_snapshot(std::string &err) {
  const int n_recr = count_recr();
  if (n_recr == 0) return true;  // no recurrent layer: nothing to roll back
  const int S = (int)cfg_.ssm_state_size;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int K = (int)cfg_.ssm_conv_kernel;
  const int chan = 2 * (int)cfg_.ssm_group_count * S + nvh * S;
  const std::size_t n_st = (std::size_t)n_recr * (std::size_t)nvh * (std::size_t)S * (std::size_t)S;
  const std::size_t n_cv = (std::size_t)n_recr * (std::size_t)(K - 1) * (std::size_t)chan;
  if (d_state_snap_ == nullptr) {
    if (hipMalloc(&d_state_snap_, n_st * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_convst_snap_, n_cv * sizeof(float)) != hipSuccess) {
      err = "state snapshot allocation failed";
      return false;
    }
  }
  // Async on the same (default) stream the kernels use: ordered against them
  // without draining the queue, which is what keeps the snapshot off the
  // critical path (it is 0.5 ms of traffic, not a pipeline flush).
  if (hipMemcpyAsync(d_state_snap_, d_state_, n_st * sizeof(float), hipMemcpyDeviceToDevice,
                     nullptr) != hipSuccess ||
      hipMemcpyAsync(d_convst_snap_, d_convst_, n_cv * sizeof(float), hipMemcpyDeviceToDevice,
                     nullptr) != hipSuccess) {
    err = "state snapshot copy failed";
    return false;
  }
  return true;
}

inline bool Graph::state_restore(std::string &err) {
  const int n_recr = count_recr();
  if (n_recr == 0) return true;
  if (d_state_snap_ == nullptr) {
    err = "state_restore without a snapshot";
    return false;
  }
  const int S = (int)cfg_.ssm_state_size;
  const int nvh = (int)cfg_.ssm_time_step_rank;
  const int K = (int)cfg_.ssm_conv_kernel;
  const int chan = 2 * (int)cfg_.ssm_group_count * S + nvh * S;
  const std::size_t n_st = (std::size_t)n_recr * (std::size_t)nvh * (std::size_t)S * (std::size_t)S;
  const std::size_t n_cv = (std::size_t)n_recr * (std::size_t)(K - 1) * (std::size_t)chan;
  if (hipMemcpyAsync(d_state_, d_state_snap_, n_st * sizeof(float), hipMemcpyDeviceToDevice,
                     nullptr) != hipSuccess ||
      hipMemcpyAsync(d_convst_, d_convst_snap_, n_cv * sizeof(float), hipMemcpyDeviceToDevice,
                     nullptr) != hipSuccess) {
    err = "state restore copy failed";
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

// Device-side greedy argmax of the logits buffer: 64 partial candidates come back
// (512 bytes) instead of the whole 993 KB vector. Same rule as the host scan in
// Sampler::filter (greatest value wins, lower index on ties, index 0 when nothing
// qualified -- every logit NaN/-inf), which is what makes the greedy ids identical.
inline bool Graph::finish_argmax(int n_vocab, std::string &err) {
  if (d_argmax_idx_ == nullptr) {
    if (hipMalloc(&d_argmax_idx_, kArgmaxBlocks * sizeof(int)) != hipSuccess ||
        hipMalloc(&d_argmax_val_, kArgmaxBlocks * sizeof(float)) != hipSuccess) {
      err = "hipMalloc argmax scratch failed";
      return false;
    }
  }
  if (!argmax_launch(d_logits_, n_vocab, d_argmax_val_, d_argmax_idx_)) {
    err = "argmax launch failed";
    return false;
  }
  int h_idx[kArgmaxBlocks];
  float h_val[kArgmaxBlocks];
  if (hipMemcpy(h_idx, d_argmax_idx_, sizeof(h_idx), hipMemcpyDeviceToHost) != hipSuccess ||
      hipMemcpy(h_val, d_argmax_val_, sizeof(h_val), hipMemcpyDeviceToHost) != hipSuccess) {
    err = "argmax readback failed";
    return false;
  }
  bool have = false;
  float bv = -INFINITY;
  std::int32_t best = 0;
  for (int b = 0; b < kArgmaxBlocks; ++b) {
    if (h_idx[b] < 0) continue;
    if (!have || h_val[b] > bv || (h_val[b] == bv && h_idx[b] < best)) {
      bv = h_val[b];
      best = h_idx[b];
      have = true;
    }
  }
  last_argmax_ = best;
  return true;
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
    RD_PHASE(prof_, "embed");  // RD_PHASE_PROF
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
      RD_PHASE(prof_, "post_norm");  // RD_PHASE_PROF
      if (!rms_norm_launch(d_x_, (const float *)w_[il].attn_post_norm.ptr, d_xn_, 1, E,
                           (float)cfg_.rms_norm_eps)) {
        err = "attn_post_norm launch failed";
        return false;
      }
      emit("attn_post_norm", il, d_xn_, E);
      RD_PHASE(prof_, "ffn");  // RD_PHASE_PROF
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
  RD_PHASE(prof_, "out_norm");  // RD_PHASE_PROF
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
  if (prof_ != nullptr) prof_->set_tag("head:");  // RD_PHASE_PROF
  RD_PHASE(prof_, "head");                        // RD_PHASE_PROF
  if (!proj(output_, d_x_, d_logits_, n_vocab, E, err)) {
    return false;
  }
  if (prof_ != nullptr) prof_->set_tag("");       // RD_PHASE_PROF
  emit("result_output", -1, d_logits_, n_vocab);
  RD_PHASE(prof_, "logits_copy");  // RD_PHASE_PROF
  if (want_argmax_) {
    if (!finish_argmax(n_vocab, err)) return false;
    host_readback_ms_ +=
        std::chrono::duration<double, std::milli>(clock::now() - t_drained).count();
    RD_PHASE(prof_, "token_end");  // RD_PHASE_PROF
    return true;
  }
  logits.resize(n_vocab);
  const bool copied = hipMemcpy(logits.data(), d_logits_, (std::size_t)n_vocab * sizeof(float),
                               hipMemcpyDeviceToHost) == hipSuccess;
  host_readback_ms_ += std::chrono::duration<double, std::milli>(clock::now() - t_drained).count();
  if (!copied) {
    err = "logits readback failed";
    return false;
  }
  RD_PHASE(prof_, "token_end");  // RD_PHASE_PROF (fecha a ultima fase do token)
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
                         d_convb_, d_alphab_, d_betab_, d_gate2b_, d_qkb_,   d_vcb_};
  for (float *p : batch_ptrs) {
    if (p) (void)hipFree(p);
  }
  d_xb_ = d_xnb_ = d_qb_ = d_kb_ = d_vb_ = d_attnb_ = d_gateb_ = nullptr;
  d_ffnab_ = d_ffnbb_ = d_projb_ = d_qkvb_ = d_zb_ = d_convb_ = nullptr;
  d_alphab_ = d_betab_ = d_gate2b_ = d_qkb_ = d_vcb_ = nullptr;
  if (d_aqb_) (void)hipFree(d_aqb_);
  if (d_posb_) (void)hipFree(d_posb_);
  if (d_emb_ids_) (void)hipFree(d_emb_ids_);
  d_aqb_ = nullptr;
  d_posb_ = nullptr;
  d_emb_ids_ = nullptr;
  if (d_q8_) (void)hipFree(d_q8_);
  if (d_pos_) (void)hipFree(d_pos_);
  if (d_logits_) (void)hipFree(d_logits_);
  if (d_attn_partial_) (void)hipFree(d_attn_partial_);
  if (d_argmax_idx_) (void)hipFree(d_argmax_idx_);
  if (d_argmax_val_) (void)hipFree(d_argmax_val_);
  d_argmax_idx_ = nullptr;
  d_argmax_val_ = nullptr;
  d_attn_partial_ = nullptr;
  d_q8_ = nullptr;
  d_pos_ = nullptr;
  d_logits_ = nullptr;
  // feat/noite-mtp: the batch logits rows and the recurrent-state snapshot
  if (d_logitsb_) (void)hipFree(d_logitsb_);
  if (d_state_snap_) (void)hipFree(d_state_snap_);
  if (d_convst_snap_) (void)hipFree(d_convst_snap_);
  d_logitsb_ = nullptr;
  d_state_snap_ = nullptr;
  d_convst_snap_ = nullptr;
  logitsb_rows_ = 0;
}

}  // namespace rdna4
