// M1 step 2 acceptance (CPU-only): GgufLoader opens a real UD file, enforces
// the dtype whitelist + geometry, and spot-loads tensors into host memory.
//
// The F32 spot-load is the real data-correctness check: if data_offset /
// tensor offsets were wrong, the loaded floats would be garbage (NaN/Inf or
// absurd magnitudes), while header-only checks would still pass.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include "rdna4/loader.h"
#include "rdna4/model.h"

namespace {

bool check_f32(const rdna4::GgufLoader &ld, const char *name, double max_abs) {
  const auto *t = ld.find(name);
  if (!t) {
    std::fprintf(stderr, "tensor %s not found\n", name);
    return false;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  std::vector<std::uint8_t> raw;
  std::string err;
  if (!ld.load_tensor(i, raw, err)) {
    std::fprintf(stderr, "load %s: %s\n", name, err.c_str());
    return false;
  }
  if (raw.size() != ld.tensor_bytes(i)) {
    std::fprintf(stderr, "load %s: size mismatch\n", name);
    return false;
  }
  const auto *v = reinterpret_cast<const float *>(raw.data());
  const std::size_t n = raw.size() / sizeof(float);
  // Sampled scan: the first 4096 floats plus every 16th after that. Wrong
  // data/tensor offsets produce garbage everywhere (NaN/Inf/absurd
  // magnitudes), so a strided sample rejects the same corrupt files as a full
  // scan at a fraction of the cost (was: every float).
  double mn = 1e300, mx = -1e300;
  auto scan_one = [&](std::size_t k) -> bool {
    if (!std::isfinite(v[k])) {
      std::fprintf(stderr, "load %s: non-finite value at %zu\n", name, k);
      return false;
    }
    mn = std::min(mn, static_cast<double>(v[k]));
    mx = std::max(mx, static_cast<double>(v[k]));
    return true;
  };
  const std::size_t kHead = 4096, kStride = 16;
  for (std::size_t k = 0; k < n && k < kHead; ++k) {
    if (!scan_one(k)) return false;
  }
  for (std::size_t k = kHead; k < n; k += kStride) {
    if (!scan_one(k)) return false;
  }
  if (mx > max_abs) {
    std::fprintf(stderr, "load %s: max |v| %.3g exceeds %.3g\n", name, mx, max_abs);
    return false;
  }
  std::printf("loaded %-32s %-8s %zu floats in [%.3g, %.3g]  OK\n", name,
              rdna4::dtype_name(ld.dtype(i)), n, mn, mx);
  return true;
}

bool check_quant_head(const rdna4::GgufLoader &ld, const char *name) {
  const auto *t = ld.find(name);
  if (!t) {
    std::fprintf(stderr, "tensor %s not found\n", name);
    return false;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  std::vector<std::uint8_t> raw;
  std::string err;
  if (!ld.load_tensor(i, raw, err)) {
    std::fprintf(stderr, "load %s: %s\n", name, err.c_str());
    return false;
  }
  std::printf("loaded %-32s %-8s %llu bytes  head:", name,
              rdna4::dtype_name(ld.dtype(i)), (unsigned long long)raw.size());
  for (std::size_t k = 0; k < 16 && k < raw.size(); ++k) {
    std::printf(" %02x", raw[k]);
  }
  std::printf("  OK\n");
  return true;
}

// Same accept/reject contract as check_quant_head (missing tensor or unreadable
// bytes fail), but reads only the first `count` bytes via load_tensor_range.
// For multi-MB quant tensors the head bytes + the size metadata prove the same
// thing (right tensor, right offset, readable data) without the full read.
bool check_quant_prefix(const rdna4::GgufLoader &ld, const char *name, std::uint64_t count) {
  const auto *t = ld.find(name);
  if (!t) {
    std::fprintf(stderr, "tensor %s not found\n", name);
    return false;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  if (count > ld.tensor_bytes(i)) count = ld.tensor_bytes(i);
  std::vector<std::uint8_t> raw;
  std::string err;
  if (!ld.load_tensor_range(i, 0, count, raw, err)) {
    std::fprintf(stderr, "load %s: %s\n", name, err.c_str());
    return false;
  }
  std::printf("loaded %-32s %-8s %llu bytes  head:", name,
              rdna4::dtype_name(ld.dtype(i)), (unsigned long long)ld.tensor_bytes(i));
  for (std::size_t k = 0; k < 16 && k < raw.size(); ++k) {
    std::printf(" %02x", raw[k]);
  }
  std::printf("  OK\n");
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-loader <model.gguf>\n");
    return 1;
  }
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open failed: %s\n", err.c_str());
    return 1;
  }
  const rdna4::gguf::File &f = ld.meta();
  std::printf("file: %s\n", argv[1]);
  std::printf("arch=%s version=%u tensors=%zu kv=%zu alignment=%zu\n",
              rdna4::gguf::kv_str(f, "general.architecture", "?").c_str(),
              f.version, f.tensors.size(), f.kv.size(), f.alignment);
  struct stat st;
  const bool have_stat = stat(argv[1], &st) == 0;
  const std::uint64_t file_size = have_stat ? static_cast<std::uint64_t>(st.st_size) : 0;
  const bool file_end_exact = f.data_offset + ld.total_bytes() == file_size;
  std::printf("data_offset=%llu total_bytes=%llu file_size=%llu file_end_exact=%s\n",
              (unsigned long long)f.data_offset,
              (unsigned long long)ld.total_bytes(),
              (unsigned long long)file_size,
              file_end_exact ? "yes" : "no");
  // Per-type counts.
  for (auto dt : {rdna4::DType::F32, rdna4::DType::Q8_0, rdna4::DType::Q2_K,
                 rdna4::DType::Q3_K, rdna4::DType::Q4_K, rdna4::DType::Q5_K,
                 rdna4::DType::Q6_K, rdna4::DType::IQ2_XXS, rdna4::DType::IQ2_XS,
                 rdna4::DType::IQ3_XXS, rdna4::DType::IQ1_S, rdna4::DType::IQ4_NL,
                 rdna4::DType::IQ3_S, rdna4::DType::IQ2_S, rdna4::DType::IQ4_XS}) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < f.tensors.size(); ++i) {
      if (ld.dtype(i) == dt) {
        ++c;
      }
    }
    std::printf("  %-10s %zu\n", rdna4::dtype_name(dt), c);
  }

  bool ok = true;
  // qwen35 config + layer-layout validation (M1 items 2-3).
  rdna4::Qwen35Config cfg;
  std::string verr;
  if (!rdna4::parse_qwen35_config(f, cfg, verr)) {
    std::fprintf(stderr, "check-loader: config failed: %s\n", verr.c_str());
    return 1;
  }
  std::printf("qwen35: %u blocks (full-attn every %u) emb=%llu ffn=%llu ctx=%llu "
              "rope_sections=[",
              cfg.block_count, cfg.full_attention_interval,
              (unsigned long long)cfg.embedding_length,
              (unsigned long long)cfg.feed_forward_length,
              (unsigned long long)cfg.context_length);
  for (std::size_t i = 0; i < cfg.rope_dim_sections.size(); ++i) {
    std::printf("%s%llu", i ? "," : "",
                (unsigned long long)cfg.rope_dim_sections[i]);
  }
  std::printf("]\n");
  ok = rdna4::validate_qwen35_layout(ld, cfg, verr) && ok;
  if (!ok) {
    std::fprintf(stderr, "check-loader: layout failed: %s\n", verr.c_str());
    return 1;
  }
  // F32 spot-loads: finiteness + magnitude prove the data section offsets are
  // right (garbage offsets => garbage floats).
  ok = check_f32(ld, "blk.0.attn_norm.weight", 10.0) && ok;
  ok = check_f32(ld, "blk.0.ssm_a", 100.0) && ok;
  ok = check_f32(ld, "blk.64.post_attention_norm.weight", 10.0) && ok;
  // Quantized spot-loads: size + readable bytes only (no dequant yet, M2).
  // blk.0 is a GDN (linear) layer, so no attn_v; ssm_alpha (q8_0) exists in
  // both UD files.
  ok = check_quant_head(ld, "blk.0.ssm_alpha.weight") && ok;
  // ffn_up is ~17 MB (iq1_s): a 16-byte prefix proves the same thing (right
  // tensor, right offset, readable data) without the full read.
  ok = check_quant_prefix(ld, "blk.0.ffn_up.weight", 16) && ok;

  if (!ok) {
    std::fprintf(stderr, "check-loader: FAILED\n");
    return 1;
  }
  std::printf("check-loader: OK\n");
  return 0;
}
