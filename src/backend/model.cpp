// Qwen35 config parsing + layer-layout validation (PLAN.md M1 items 2-3).
#include "rdna4/model.h"

#include <algorithm>

namespace rdna4 {
namespace {

// Integer array KV (I32/U32/I64/U64 element types) -> vector<uint64>.
bool kv_u64_vec(const gguf::File &f, const char *key,
                std::vector<std::uint64_t> &out) {
  auto it = f.kv.find(key);
  if (it == f.kv.end() || it->second.type != gguf::ValueType::ARRAY) {
    return false;
  }
  const auto &v = it->second;
  switch (v.arr_type) {
    case gguf::ValueType::U32:
    case gguf::ValueType::I32:
    case gguf::ValueType::U64:
    case gguf::ValueType::I64:
      break;
    default:
      return false;
  }
  out.clear();
  out.reserve(v.arr.size());
  for (const auto &e : v.arr) {
    out.push_back(e.u);
  }
  return true;
}

bool kv_f64(const gguf::File &f, const char *key, double &out) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) {
    return false;
  }
  const auto &v = it->second;
  if (v.type == gguf::ValueType::F32) {
    out = v.f;
    return true;
  }
  if (v.type == gguf::ValueType::F64) {
    out = v.f;
    return true;
  }
  return false;
}

// (an unchecked `prod()` used to live here: the checked product is
// loader.cpp's prod_dims(), which is the one the geometry guards rely on.)

using Expect = std::pair<std::string, std::vector<std::int64_t>>;

// Full attention block (any of 3,7,...,63): 11 tensors.
std::vector<Expect> full_attn_tensors(const Qwen35Config &c) {
  const std::int64_t emb = static_cast<std::int64_t>(c.embedding_length);
  const std::int64_t ffn = static_cast<std::int64_t>(c.feed_forward_length);
  const std::int64_t k_width =
      static_cast<std::int64_t>(c.head_count_kv) *
      static_cast<std::int64_t>(c.key_length);
  const std::int64_t v_width =
      static_cast<std::int64_t>(c.head_count_kv) *
      static_cast<std::int64_t>(c.value_length);
  // q per head = 2 x key_length (nope + rope), per the qwen35 MRoPE layout.
  const std::int64_t q_width = static_cast<std::int64_t>(c.head_count) *
                               2 * static_cast<std::int64_t>(c.key_length);
  const std::int64_t out_width =
      static_cast<std::int64_t>(c.head_count) *
      static_cast<std::int64_t>(c.value_length);
  const std::int64_t klen = static_cast<std::int64_t>(c.key_length);
  return {
      {"attn_k.weight", {emb, k_width}},
      {"attn_k_norm.weight", {klen}},
      {"attn_norm.weight", {emb}},
      {"attn_output.weight", {out_width, emb}},
      {"attn_q.weight", {emb, q_width}},
      {"attn_q_norm.weight", {klen}},
      {"attn_v.weight", {emb, v_width}},
      {"ffn_down.weight", {ffn, emb}},
      {"ffn_gate.weight", {emb, ffn}},
      {"ffn_up.weight", {emb, ffn}},
      {"post_attention_norm.weight", {emb}},
  };
}

// GDN linear block: common + attn_qkv/attn_gate + ssm_* (14 tensors total).
std::vector<Expect> gdn_tensors(const Qwen35Config &c) {
  const std::int64_t emb = static_cast<std::int64_t>(c.embedding_length);
  const std::int64_t ffn = static_cast<std::int64_t>(c.feed_forward_length);
  const std::int64_t inner = static_cast<std::int64_t>(c.ssm_inner_size);
  const std::int64_t tsr = static_cast<std::int64_t>(c.ssm_time_step_rank);
  const std::int64_t state = static_cast<std::int64_t>(c.ssm_state_size);
  const std::int64_t conv = static_cast<std::int64_t>(c.ssm_conv_kernel);
  // GDN qkv width = q(2*g*d) + k(g*d) + v(g*d) + z(g*d) = 5*g*d, where
  // g = ssm.group_count, d = ssm.state_size (16*128*5 = 10240, verified in
  // the UD files; ssm_conv1d has the same width).
  const std::int64_t gdn_qkv = 5 * static_cast<std::int64_t>(c.ssm_group_count) *
                               static_cast<std::int64_t>(c.ssm_state_size);
  std::vector<Expect> v;
  v.push_back({"attn_gate.weight", {emb, inner}});
  v.push_back({"attn_norm.weight", {emb}});
  v.push_back({"attn_qkv.weight", {emb, gdn_qkv}});
  v.push_back({"ffn_down.weight", {ffn, emb}});
  v.push_back({"ffn_gate.weight", {emb, ffn}});
  v.push_back({"ffn_up.weight", {emb, ffn}});
  v.push_back({"post_attention_norm.weight", {emb}});
  v.push_back({"ssm_a", {tsr}});
  v.push_back({"ssm_alpha.weight", {emb, tsr}});
  v.push_back({"ssm_beta.weight", {emb, tsr}});
  v.push_back({"ssm_conv1d.weight", {conv, gdn_qkv}});
  v.push_back({"ssm_dt.bias", {tsr}});
  v.push_back({"ssm_norm.weight", {state}});
  v.push_back({"ssm_out.weight", {inner, emb}});
  return v;
}

}  // namespace

std::vector<std::pair<std::string, std::vector<std::int64_t>>>
expected_block_tensors(const Qwen35Config &cfg, std::uint32_t i) {
  if (i >= cfg.block_count) {
    return {};
  }
  std::vector<Expect> out;
  if (i == cfg.block_count - 1) {
    // MTP block: full-attention set + nextn.* (ignored by the v1 forward,
    // but present in the files; validated here for completeness).
    for (auto &e : full_attn_tensors(cfg)) {
      out.push_back(std::move(e));
    }
    const std::int64_t emb = static_cast<std::int64_t>(cfg.embedding_length);
    // eh_proj maps concat(hidden, embed) -> hidden, i.e. [2*emb, emb]. For this
    // model 2*emb == 5*group*state == 10240 by coincidence; use the real
    // definition so a differently-shaped qwen35 MTP block fails for the right
    // reason (review finding L4).
    out.push_back({"nextn.eh_proj.weight", {2 * emb, emb}});
    out.push_back({"nextn.enorm.weight", {emb}});
    out.push_back({"nextn.hnorm.weight", {emb}});
    out.push_back({"nextn.shared_head_norm.weight", {emb}});
  } else if (is_full_attention_layer(i, cfg.full_attention_interval)) {
    for (auto &e : full_attn_tensors(cfg)) {
      out.push_back(std::move(e));
    }
  } else {
    for (auto &e : gdn_tensors(cfg)) {
      out.push_back(std::move(e));
    }
  }
  return out;
}

bool parse_qwen35_config(const gguf::File &f, Qwen35Config &cfg, std::string &err) {
  if (gguf::kv_str(f, "general.architecture") != "qwen35") {
    err = "general.architecture is not qwen35";
    return false;
  }
  auto req_u32 = [&](const char *key, std::uint32_t &out) -> bool {
    auto it = f.kv.find(key);
    if (it == f.kv.end() || it->second.type != gguf::ValueType::U32) {
      err = std::string("missing or wrong-type KV: ") + key;
      return false;
    }
    out = static_cast<std::uint32_t>(it->second.u);
    return true;
  };
  cfg = Qwen35Config{};
  if (!req_u32("qwen35.block_count", cfg.block_count)) {
    return false;
  }
  if (!req_u32("qwen35.full_attention_interval", cfg.full_attention_interval)) {
    return false;
  }
  if (!req_u32("qwen35.nextn_predict_layers", cfg.nextn_predict_layers)) {
    return false;
  }
  auto req_u64 = [&](const char *key, std::uint64_t &out) -> bool {
    auto it = f.kv.find(key);
    if (it == f.kv.end() || it->second.type != gguf::ValueType::U32) {
      err = std::string("missing or wrong-type KV: ") + key;
      return false;
    }
    out = it->second.u;
    return true;
  };
  if (!req_u64("qwen35.context_length", cfg.context_length)) {
    return false;
  }
  if (!req_u64("qwen35.embedding_length", cfg.embedding_length)) {
    return false;
  }
  if (!req_u64("qwen35.feed_forward_length", cfg.feed_forward_length)) {
    return false;
  }
  if (!req_u32("qwen35.attention.head_count", cfg.head_count)) {
    return false;
  }
  if (!req_u32("qwen35.attention.head_count_kv", cfg.head_count_kv)) {
    return false;
  }
  if (!req_u64("qwen35.attention.key_length", cfg.key_length)) {
    return false;
  }
  if (!req_u64("qwen35.attention.value_length", cfg.value_length)) {
    return false;
  }
  if (!req_u32("qwen35.ssm.conv_kernel", cfg.ssm_conv_kernel)) {
    return false;
  }
  if (!req_u32("qwen35.ssm.state_size", cfg.ssm_state_size)) {
    return false;
  }
  if (!req_u32("qwen35.ssm.group_count", cfg.ssm_group_count)) {
    return false;
  }
  if (!req_u32("qwen35.ssm.time_step_rank", cfg.ssm_time_step_rank)) {
    return false;
  }
  if (!req_u32("qwen35.ssm.inner_size", cfg.ssm_inner_size)) {
    return false;
  }
  if (!req_u32("qwen35.rope.dimension_count", cfg.rope_dim_count)) {
    return false;
  }
  if (!kv_u64_vec(f, "qwen35.rope.dimension_sections", cfg.rope_dim_sections)) {
    err = "missing or wrong-type KV: qwen35.rope.dimension_sections";
    return false;
  }
  // Value validation (PLAN M1 item 2). Presence/type alone is not enough: a
  // wrong block_count or MRoPE section split is accepted by the layout check
  // (which only keys off the parsed values) and would silently break M3.
  if (cfg.block_count < 2) {
    err = "qwen35.block_count must be >= 2 (got " + std::to_string(cfg.block_count) + ")";
    return false;
  }
  if (cfg.full_attention_interval < 1) {
    err = "qwen35.full_attention_interval must be >= 1 (got " +
          std::to_string(cfg.full_attention_interval) + ")";
    return false;
  }
  if (cfg.nextn_predict_layers >= cfg.block_count) {
    err = "qwen35.nextn_predict_layers (" + std::to_string(cfg.nextn_predict_layers) +
          ") must be < block_count (" + std::to_string(cfg.block_count) + ")";
    return false;
  }
  {
    std::uint64_t sec_sum = 0;
    for (std::uint64_t v : cfg.rope_dim_sections) sec_sum += v;
    // MRoPE sections cover HALF the rotary dimensions (HF mrope_section
    // semantics, applied per half): for this model 11+11+10+0 = 32 and
    // dimension_count = 64. Asserting sum == dimension_count here would reject
    // the real files, so the invariant is 2*sum == dimension_count.
    if (cfg.rope_dim_count % 2 != 0 || sec_sum * 2 != cfg.rope_dim_count) {
      err = "qwen35.rope.dimension_sections sums to " + std::to_string(sec_sum) +
            " but dimension_count is " + std::to_string(cfg.rope_dim_count) +
            " (expected 2*sections == dimension_count)";
      return false;
    }
    // This model's MRoPE split (t, h, w, extra) as shipped in both UD files.
    // NOTE: dimension_count is the rotary width (64 here), NOT the number of
    // sections; the split has 4 sections [t,h,w,extra].
    // The exact values are asserted only for the real 65-block qwen35 layout
    // (the one validate_qwen35_layout is written for); synthetic/other shapes
    // only have to satisfy the structural invariants.
    if (cfg.block_count == 65 &&
        (cfg.rope_dim_sections.size() != 4 || cfg.rope_dim_sections[0] != 11 ||
         cfg.rope_dim_sections[1] != 11 || cfg.rope_dim_sections[2] != 10 ||
         cfg.rope_dim_sections[3] != 0)) {
      err = "unexpected qwen35 MRoPE layout: dimension_count=" + std::to_string(cfg.rope_dim_count) +
            " sections=[" ;
      for (std::size_t i = 0; i < cfg.rope_dim_sections.size(); ++i) {
        err += std::to_string(cfg.rope_dim_sections[i]) + (i + 1 < cfg.rope_dim_sections.size() ? "," : "");
      }
      err += "] (expected 4 sections [11,11,10,0])";
      return false;
    }
  }
  double eps = 0.0, freq_base = 0.0;
  if (!kv_f64(f, "qwen35.attention.layer_norm_rms_epsilon", eps)) {
    err = "missing or wrong-type KV: qwen35.attention.layer_norm_rms_epsilon";
    return false;
  }
  if (!kv_f64(f, "qwen35.rope.freq_base", freq_base)) {
    err = "missing or wrong-type KV: qwen35.rope.freq_base";
    return false;
  }
  cfg.rms_norm_eps = eps;
  cfg.rope_freq_base = freq_base;

  auto req_tok = [&](const char *key, std::uint32_t &out) -> bool {
    auto it = f.kv.find(key);
    if (it == f.kv.end() || it->second.type != gguf::ValueType::U32) {
      err = std::string("missing or wrong-type KV: ") + key;
      return false;
    }
    out = static_cast<std::uint32_t>(it->second.u);
    return true;
  };
  if (!req_tok("tokenizer.ggml.bos_token_id", cfg.bos_token_id)) {
    return false;
  }
  if (!req_tok("tokenizer.ggml.eos_token_id", cfg.eos_token_id)) {
    return false;
  }
  if (!req_tok("tokenizer.ggml.padding_token_id", cfg.pad_token_id)) {
    return false;
  }
  return true;
}

namespace {

std::string dims_str(const std::vector<std::int64_t> &dims) {
  std::string s = "[";
  for (std::size_t d = 0; d < dims.size(); ++d) {
    if (d) {
      s += ",";
    }
    s += std::to_string(dims[d]);
  }
  s += "]";
  return s;
}

}  // namespace

bool validate_qwen35_layout(const GgufLoader &ld, const Qwen35Config &cfg,
                            std::string &err) {
  const rdna4::gguf::File &f = ld.meta();
  // Group tensors by block. block_tensors[i] = name -> dims (within block i).
  std::vector<std::map<std::string, std::vector<std::int64_t>>>
      block_tensors(cfg.block_count);
  std::vector<std::string> top_level;
  for (const auto &t : f.tensors) {
    const std::string &n = t.name;
    if (n.compare(0, 4, "blk.") != 0) {
      top_level.push_back(n);
      continue;
    }
    const std::size_t dot2 = n.find('.', 4);
    if (dot2 == std::string::npos) {
      err = "malformed tensor name: " + n;
      return false;
    }
    const std::string idx = n.substr(4, dot2 - 4);
    std::size_t pos = 0;
    unsigned long li = 0;
    try {
      li = std::stoul(idx, &pos, 10);
    } catch (...) {
      err = "malformed block index in tensor name: " + n;
      return false;
    }
    if (pos != idx.size() || li >= cfg.block_count) {
      err = "malformed or out-of-range block index in tensor name: " + n;
      return false;
    }
    const std::string rest = n.substr(dot2 + 1);
    if (block_tensors[li].count(rest)) {
      err = "duplicate tensor name: " + n;
      return false;
    }
    block_tensors[li][rest] = t.dims;
  }
  // Per block: exact match of the expected (name, dims) set.
  for (std::uint32_t i = 0; i < cfg.block_count; ++i) {
    const std::string prefix = "blk." + std::to_string(i) + ".";
    const auto expect = expected_block_tensors(cfg, i);
    std::map<std::string, std::vector<std::int64_t>> expect_map(expect.begin(),
                                                                expect.end());
    // Missing or dim-mismatched expected tensors.
    for (const auto &kv : expect_map) {
      auto it = block_tensors[i].find(kv.first);
      if (it == block_tensors[i].end()) {
        err = "missing tensor: " + prefix + kv.first;
        return false;
      }
      if (it->second != kv.second) {
        err = "tensor " + prefix + kv.first + ": dims " + dims_str(it->second) +
              " != expected " + dims_str(kv.second);
        return false;
      }
    }
    // Extra (unexpected) tensors in this block.
    for (const auto &kv : block_tensors[i]) {
      if (!expect_map.count(kv.first)) {
        err = "unexpected tensor: " + prefix + kv.first;
        return false;
      }
    }
  }
  // Top level: exactly token_embd / output_norm / output, with the right dims.
  const std::int64_t emb = static_cast<std::int64_t>(cfg.embedding_length);
  std::map<std::string, std::vector<std::int64_t>> top_map;
  for (const std::string &n : top_level) {
    if (top_map.count(n)) {
      err = "duplicate top-level tensor: " + n;
      return false;
    }
    top_map[n] = ld.find(n.c_str())->dims;
  }
  const auto need = [&](const char *name, const std::vector<std::int64_t> &dims) {
    auto it = top_map.find(name);
    if (it == top_map.end()) {
      err = std::string("missing top-level tensor: ") + name;
      return false;
    }
    if (it->second != dims) {
      err = std::string(name) + ": dims " + dims_str(it->second) + " != expected " +
            dims_str(dims);
      return false;
    }
    return true;
  };
  // token_embd and output share [emb, vocab]; vocab is a model property we do
  // not pin here, so validate the shape and mutual consistency instead.
  auto tok = top_map.find("token_embd.weight");
  auto out = top_map.find("output.weight");
  if (tok == top_map.end()) {
    err = "missing top-level tensor: token_embd.weight";
    return false;
  }
  if (out == top_map.end()) {
    err = "missing top-level tensor: output.weight";
    return false;
  }
  if (tok->second.size() != 2 || tok->second[0] != emb ||
      out->second.size() != 2 || out->second[0] != emb ||
      out->second[1] != tok->second[1] || tok->second[1] <= 0) {
    err = "token_embd/output: bad dims (" + dims_str(tok->second) + " vs " +
          dims_str(out->second) + ")";
    return false;
  }
  if (!need("output_norm.weight", {emb})) {
    return false;
  }
  for (const auto &kv : top_map) {
    if (kv.first != "token_embd.weight" && kv.first != "output.weight" &&
        kv.first != "output_norm.weight") {
      err = "unexpected top-level tensor: " + kv.first;
      return false;
    }
  }
  (void)need;
  return true;
}

}  // namespace rdna4
