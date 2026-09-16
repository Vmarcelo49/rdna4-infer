// CPU-only mmproj (CLIP vision) loading acceptance: VisionLoader opens a real
// mmproj file, parse_clip_vision_config pins the clip.vision.* KV, and
// validate_clip_vision_layout pins the exact 334-tensor inventory (27 ViT
// blocks x 12 + 10 top-level incl. the qwen3vl_merger pair). Spot reads prove
// the data-section offsets land on sane values (F32 finiteness scan, F16
// decode scan, quant prefix reads), mirroring tests/check_loader.cpp.
//
//   check-vision <mmproj.gguf> [--expect-fail]
//   Both shipped files pass: mmproj-F16.gguf and mmproj-q8_0.gguf (weights
//   differ in dtype: F16 vs Q8_0; the layout check accepts both).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "rdna4/vision.h"

namespace {

// IEEE-754 binary16 -> float, for the F16 spot scan.
float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = (h >> 15) & 1;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign << 31;
    } else {
      // subnormal: normalize
      std::uint32_t m = mant;
      std::uint32_t e = 0;
      while ((m & 0x400) == 0) {
        m <<= 1;
        ++e;
      }
      m &= 0x3FF;
      bits = (sign << 31) | ((127 - 14 - e) << 23) | (m << 13);
    }
  } else if (exp == 31) {
    bits = (sign << 31) | (0xFF << 23) | (mant << 13);  // inf/nan
  } else {
    bits = (sign << 31) | ((exp + 112) << 23) | (mant << 13);
  }
  float out = 0.0F;
  __builtin_memcpy(&out, &bits, sizeof(out));
  return out;
}

bool check_f32(const rdna4::VisionLoader &ld, const char *name, double max_abs) {
  const auto *t = ld.find(name);
  if (!t) {
    std::fprintf(stderr, "tensor %s not found\n", name);
    return false;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  if (ld.dtype(i) != rdna4::VisionDType::F32) {
    std::fprintf(stderr, "tensor %s is not f32\n", name);
    return false;
  }
  std::vector<std::uint8_t> raw;
  std::string err;
  if (!ld.load_tensor(i, raw, err)) {
    std::fprintf(stderr, "load %s: %s\n", name, err.c_str());
    return false;
  }
  const auto *v = reinterpret_cast<const float *>(raw.data());
  const std::size_t n = raw.size() / sizeof(float);
  double mn = 1e300, mx = -1e300;
  for (std::size_t k = 0; k < n; ++k) {
    if (!std::isfinite(v[k])) {
      std::fprintf(stderr, "load %s: non-finite value at %zu\n", name, k);
      return false;
    }
    mn = std::min(mn, static_cast<double>(v[k]));
    mx = std::max(mx, static_cast<double>(v[k]));
  }
  if (mx > max_abs || mn < -max_abs) {
    std::fprintf(stderr, "load %s: |v| exceeds %.3g ([%.3g, %.3g])\n", name, max_abs, mn, mx);
    return false;
  }
  std::printf("loaded %-28s f32 %zu floats in [%.3g, %.3g]  OK\n", name, n, mn, mx);
  return true;
}

// Finiteness scan over the first `count` F16 values (decoded). Wrong data
// offsets decode to NaN/Inf with overwhelming probability.
bool check_f16_head(const rdna4::VisionLoader &ld, const char *name, std::size_t count,
                    double max_abs) {
  const auto *t = ld.find(name);
  if (!t) {
    std::fprintf(stderr, "tensor %s not found\n", name);
    return false;
  }
  const std::size_t i = static_cast<std::size_t>(t - ld.meta().tensors.data());
  if (ld.dtype(i) != rdna4::VisionDType::F16) {
    std::printf("skip   %-28s (not f16 here)\n", name);
    return true;
  }
  std::vector<std::uint8_t> raw;
  std::string err;
  const std::uint64_t want = count * 2;
  if (!ld.load_tensor_range(i, 0, want > ld.tensor_bytes(i) ? ld.tensor_bytes(i) : want, raw,
                            err)) {
    std::fprintf(stderr, "load %s: %s\n", name, err.c_str());
    return false;
  }
  const std::size_t n = raw.size() / 2;
  double mx = 0.0;
  for (std::size_t k = 0; k < n; ++k) {
    std::uint16_t h = static_cast<std::uint16_t>(raw[2 * k] | (raw[2 * k + 1] << 8));
    const float v = f16_to_f32(h);
    if (!std::isfinite(v)) {
      std::fprintf(stderr, "load %s: non-finite f16 at %zu\n", name, k);
      return false;
    }
    mx = std::max(mx, static_cast<double>(v < 0 ? -v : v));
  }
  if (mx > max_abs) {
    std::fprintf(stderr, "load %s: max |v| %.3g exceeds %.3g\n", name, mx, max_abs);
    return false;
  }
  std::printf("loaded %-28s f16 %zu-value head finite, max|v|=%.3g  OK\n", name, n, mx);
  return true;
}

bool check_prefix(const rdna4::VisionLoader &ld, const char *name, std::uint64_t count) {
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
  std::printf("loaded %-28s %-4s %llu bytes  head:", name,
              rdna4::vision_dtype_name(ld.dtype(i)), (unsigned long long)ld.tensor_bytes(i));
  for (std::size_t k = 0; k < 16 && k < raw.size(); ++k) std::printf(" %02x", raw[k]);
  std::printf("  OK\n");
  return true;
}

// Synthetic negative tests: no file needed, pure validator behavior.
bool check_negatives() {
  using rdna4::gguf::File;
  using rdna4::gguf::Value;
  using rdna4::gguf::ValueType;
  bool ok = true;
  auto str_val = [](const char *s) {
    Value v;
    v.type = ValueType::STR;
    v.s = s;
    return v;
  };
  // 1. wrong architecture
  {
    File f;
    f.kv["general.architecture"] = str_val("qwen35");
    rdna4::ClipVisionConfig cfg;
    std::string err;
    const bool good = rdna4::parse_clip_vision_config(f, cfg, err);
    std::printf("%-5s non-clip arch rejected (%s)\n", !good ? "ok" : "FAIL", err.c_str());
    ok = ok && !good;
  }
  // 2. clip arch but missing vision KV
  {
    File f;
    f.kv["general.architecture"] = str_val("clip");
    rdna4::ClipVisionConfig cfg;
    std::string err;
    const bool good = rdna4::parse_clip_vision_config(f, cfg, err);
    std::printf("%-5s missing vision KV rejected (%s)\n", !good ? "ok" : "FAIL", err.c_str());
    ok = ok && !good;
  }
  // 3. layout validator catches a missing tensor (built from a real file's
  //    inventory with one entry dropped); needs a valid config, exercised by
  //    the caller with the real cfg. Placeholder here is trivially false-safe:
  //    empty inventory can never validate.
  {
    File f;
    f.kv["general.architecture"] = str_val("clip");
    rdna4::ClipVisionConfig cfg;
    std::string err;
    const bool good = rdna4::parse_clip_vision_config(f, cfg, err);
    ok = ok && !good;
  }
  // 4+. synthetic valid clip KV (mirrors the real mmproj values): parse must
  // accept it, and each single-field corruption must reject with a naming msg.
  {
    auto u32 = [](std::uint32_t v) {
      Value x;
      x.type = ValueType::U32;
      x.u = v;
      return x;
    };
    auto boolean = [](bool v) {
      Value x;
      x.type = ValueType::BOOL;
      x.u = v ? 1 : 0;
      return x;
    };
    auto f32 = [](double v) {
      Value x;
      x.type = ValueType::F32;
      x.f = v;
      return x;
    };
    auto f32x3 = [&](double a) {
      Value x;
      x.type = ValueType::ARRAY;
      x.arr_type = ValueType::F32;
      for (int i = 0; i < 3; ++i) {
        Value e;
        e.type = ValueType::F32;
        e.f = a;
        x.arr.push_back(e);
      }
      return x;
    };
    auto deepstack = [&](std::uint64_t hot) {
      Value x;
      x.type = ValueType::ARRAY;
      x.arr_type = ValueType::BOOL;
      for (int i = 0; i < 27; ++i) {
        Value e;
        e.type = ValueType::BOOL;
        e.u = (hot == static_cast<std::uint64_t>(i)) ? 1 : 0;
        x.arr.push_back(e);
      }
      return x;
    };
    auto base = [&]() {
      File f;
      f.kv["general.architecture"] = str_val("clip");
      f.kv["clip.vision.block_count"] = u32(27);
      f.kv["clip.vision.embedding_length"] = u32(1152);
      f.kv["clip.vision.feed_forward_length"] = u32(4304);
      f.kv["clip.vision.attention.head_count"] = u32(16);
      f.kv["clip.vision.image_size"] = u32(768);
      f.kv["clip.vision.patch_size"] = u32(16);
      f.kv["clip.vision.projection_dim"] = u32(5120);
      f.kv["clip.vision.spatial_merge_size"] = u32(2);
      f.kv["clip.vision.attention.layer_norm_epsilon"] = f32(1e-6);
      f.kv["clip.vision.image_mean"] = f32x3(0.5);
      f.kv["clip.vision.image_std"] = f32x3(0.5);
      f.kv["clip.has_vision_encoder"] = boolean(true);
      f.kv["clip.use_gelu"] = boolean(true);
      f.kv["clip.projector_type"] = str_val("qwen3vl_merger");
      f.kv["clip.vision.is_deepstack_layers"] = deepstack(27);  // 27 == none hot
      return f;
    };
    auto expect = [&](const char *what, File f, bool want_ok, const char *want_sub) {
      rdna4::ClipVisionConfig cfg;
      std::string err;
      const bool good = rdna4::parse_clip_vision_config(f, cfg, err);
      const bool named =
          good == want_ok && (want_ok || std::string(err).find(want_sub) != std::string::npos);
      std::printf("%-5s synthetic %s (%s)\n", named ? "ok" : "FAIL", what, err.c_str());
      ok = ok && named;
    };
    expect("valid", base(), true, "");
    {
      File f = base();
      f.kv["clip.vision.is_deepstack_layers"] = deepstack(5);
      expect("deepstack", f, false, "deepstack is set on block 5");
    }
    {
      File f = base();
      f.kv["clip.projector_type"] = str_val("mlp");
      expect("projector", f, false, "only qwen3vl_merger is supported");
    }
    {
      File f = base();
      f.kv["clip.vision.attention.head_count"] = u32(7);
      expect("heads", f, false, "must divide embedding_length");
    }
    {
      File f = base();
      f.kv["clip.vision.block_count"] = u32(0);
      expect("blocks", f, false, "block_count");
    }
  }
  return ok;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-vision <mmproj.gguf> [--expect-fail]\n");
    return 2;
  }
  bool expect_fail = false;
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "--expect-fail") expect_fail = true;
  }
  bool ok = check_negatives();

  rdna4::VisionLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    if (expect_fail) {
      std::printf("check-vision: open failed as expected (%s)\n", err.c_str());
      return ok ? 0 : 1;
    }
    std::fprintf(stderr, "check-vision: open failed: %s\n", err.c_str());
    return 1;
  }
  if (expect_fail) {
    std::fprintf(stderr, "check-vision: UNEXPECTED: opened but failure expected\n");
    return 1;
  }
  const rdna4::gguf::File &f = ld.meta();
  std::printf("file: %s\n", argv[1]);
  std::printf("arch=%s version=%u tensors=%zu kv=%zu alignment=%zu\n",
              rdna4::gguf::kv_str(f, "general.architecture", "?").c_str(), f.version,
              f.tensors.size(), f.kv.size(), f.alignment);
  for (auto dt :
       {rdna4::VisionDType::F32, rdna4::VisionDType::F16, rdna4::VisionDType::Q8_0}) {
    std::size_t c = 0;
    for (std::size_t i = 0; i < f.tensors.size(); ++i) {
      if (ld.dtype(i) == dt) ++c;
    }
    std::printf("  %-4s %zu\n", rdna4::vision_dtype_name(dt), c);
  }

  rdna4::ClipVisionConfig cfg;
  if (!rdna4::parse_clip_vision_config(f, cfg, err)) {
    std::fprintf(stderr, "check-vision: config failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("clip vision: %u blocks emb=%u ffn=%u heads=%u image=%u patch=%u proj=%u "
              "merge=%ux%u gelu=%d mean=[%.2g,%.2g,%.2g]\n",
              cfg.block_count, cfg.embedding_length, cfg.feed_forward_length, cfg.head_count,
              cfg.image_size, cfg.patch_size, cfg.projection_dim, cfg.spatial_merge_size,
              cfg.spatial_merge_size, (int)cfg.use_gelu, cfg.image_mean[0], cfg.image_mean[1],
              cfg.image_mean[2]);
  if (!rdna4::validate_clip_vision_layout(f, cfg, err)) {
    std::fprintf(stderr, "check-vision: layout failed: %s\n", err.c_str());
    return 1;
  }
  std::printf("layout: %zu tensors validated  OK\n", f.tensors.size());

  // Forged-inventory negatives against the REAL tensor list: each must fail
  // with the forged tensor named in the message.
  {
    bool fok = true;
    auto forged_check = [&](const char *what, const rdna4::gguf::File &forged,
                            const char *want_sub) {
      std::string ferr;
      const bool good = rdna4::validate_clip_vision_layout(forged, cfg, ferr);
      const bool named =
          !good && std::string(ferr).find(want_sub) != std::string::npos;
      std::printf("%-5s forged %s rejected (%s)\n", named ? "ok" : "FAIL", what, ferr.c_str());
      fok = fok && named;
    };
    {
      rdna4::gguf::File forged = f;
      for (auto it = forged.tensors.begin(); it != forged.tensors.end(); ++it) {
        if (it->name == "v.blk.26.ffn_down.bias") {
          forged.tensors.erase(it);
          break;
        }
      }
      forged_check("missing-tensor", forged, "missing tensor: v.blk.26.ffn_down.bias");
    }
    {
      rdna4::gguf::File forged = f;
      for (auto &t : forged.tensors) {
        if (t.name == "mm.2.weight") t.dims[1] += 1;
      }
      forged_check("dims-mismatch", forged, "mm.2.weight");
    }
    {
      rdna4::gguf::File forged = f;
      rdna4::gguf::TensorInfo extra;
      extra.name = "v.blk.0.evil";
      extra.dims = {1};
      forged.tensors.push_back(extra);
      forged_check("extra-tensor", forged, "unexpected tensor: v.blk.0.evil");
    }
    ok = ok && fok;
  }

  // Spot reads: F32 finiteness (offsets land on real data), F16 decode head,
  // quant prefix.
  ok = check_f32(ld, "v.post_ln.weight", 10.0) && ok;
  ok = check_f32(ld, "mm.2.bias", 10.0) && ok;
  ok = check_f32(ld, "v.patch_embd.bias", 10.0) && ok;
  ok = check_f16_head(ld, "mm.0.weight", 4096, 1e4) && ok;
  ok = check_f16_head(ld, "v.patch_embd.weight", 4096, 1e4) && ok;
  ok = check_prefix(ld, "mm.2.weight", 64) && ok;
  ok = check_prefix(ld, "v.blk.0.attn_qkv.weight", 64) && ok;

  if (!ok) {
    std::fprintf(stderr, "check-vision: FAILED\n");
    return 1;
  }
  std::printf("check-vision: OK\n");
  return 0;
}
