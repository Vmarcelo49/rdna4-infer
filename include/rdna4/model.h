#pragma once
// Qwen3.8-27B dense ("qwen35") model config + layer-layout validation
// (PLAN.md M1 step 2 + step 3).
//
// The layout is derived from the two UD files (see scripts/ + the llama.cpp
// b10902 reference dump). There are exactly 3 layer signatures over the 65
// blocks (block_count = 65, blocks 0..64):
//
//   * GDN linear  (48 layers): i in 0..63 with (i+1) % full_attention_interval
//                               != 0  -> attn_qkv + attn_gate + ssm_*
//   * full attn   (16 layers): i in 0..63 with (i+1) % full_attention_interval
//                               == 0  -> separate attn_q/k/v + q/k norms
//   * MTP         (1 layer)   : block 64  -> full-attn set + nextn.*
//
// Every block also carries the common attn_norm / post_attention_norm /
// ffn_gate/up/down. Top-level: token_embd, output_norm, output.
//
// validate_qwen35_layout() fail-fasts if the loaded inventory deviates from
// this spec (missing/extra tensor, wrong dims, wrong partition).
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "rdna4/loader.h"

namespace rdna4 {

struct Qwen35Config {
  std::uint32_t block_count = 0;
  std::uint32_t full_attention_interval = 0;
  std::uint64_t embedding_length = 0;   // emb
  std::uint64_t feed_forward_length = 0;  // ffn
  std::uint64_t context_length = 0;
  std::uint32_t head_count = 0;         // hc
  std::uint32_t head_count_kv = 0;      // hckv
  std::uint64_t key_length = 0;         // klen
  std::uint64_t value_length = 0;       // vlen
  std::uint32_t nextn_predict_layers = 0;
  std::uint32_t ssm_conv_kernel = 0;    // conv
  std::uint32_t ssm_state_size = 0;     // state
  std::uint32_t ssm_group_count = 0;
  std::uint32_t ssm_time_step_rank = 0; // tsr
  std::uint32_t ssm_inner_size = 0;     // inner
  std::uint32_t rope_dim_count = 0;
  std::vector<std::uint64_t> rope_dim_sections;
  double rms_norm_eps = 0.0;            // attention.layer_norm_rms_epsilon
  double rope_freq_base = 0.0;          // rope.freq_base
  std::uint32_t bos_token_id = 0;
  std::uint32_t eos_token_id = 0;
  std::uint32_t pad_token_id = 0;
};

// Parses + validates the required qwen35 KVs (M1 item 2). Fails if
// general.architecture != "qwen35" or any required KV is missing/wrong type.
bool parse_qwen35_config(const gguf::File &f, Qwen35Config &cfg, std::string &err);

// True for full-attention layers i in 0..63 (block 64 is always MTP).
inline bool is_full_attention_layer(std::uint32_t i, std::uint32_t interval) {
  return interval != 0 && (i + 1) % interval == 0;
}

// Expected (name, dims) for one block, in a deterministic order. `i` is the
// block index 0..block_count-1. Returns empty on an out-of-range i.
std::vector<std::pair<std::string, std::vector<std::int64_t>>>
expected_block_tensors(const Qwen35Config &cfg, std::uint32_t i);

// Validates the whole inventory: every block matches its expected set exactly
// (no missing, no extras, dims match) and the top-level tensors are present.
bool validate_qwen35_layout(const GgufLoader &ld, const Qwen35Config &cfg,
                            std::string &err);

}  // namespace rdna4
