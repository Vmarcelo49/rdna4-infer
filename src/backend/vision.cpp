// CLIP/mmproj vision loading: config parsing, layout validation, tensor reads.
// See include/rdna4/vision.h. Validation style mirrors model.cpp (fail-fast,
// every rejection names the key or tensor); the file-geometry guards mirror
// loader.cpp (same M1/M3/M6/M7 rationale, applied to the vision dtype set).
#include "rdna4/vision.h"

#include <sys/stat.h>

#include <exception>
#include <map>

namespace rdna4 {
namespace {

bool vision_dtype_from_ggml(std::uint32_t id, VisionDType &out) {
  switch (id) {
    case 0: out = VisionDType::F32; return true;
    case 1: out = VisionDType::F16; return true;
    case 8: out = VisionDType::Q8_0; return true;
    default: return false;
  }
}

std::uint32_t vision_block_elems(VisionDType t) {
  return t == VisionDType::Q8_0 ? 32 : 1;
}

std::uint32_t vision_block_bytes(VisionDType t) {
  switch (t) {
    case VisionDType::F32: return 4;
    case VisionDType::F16: return 2;
    case VisionDType::Q8_0: return 34;
  }
  return 0;
}

// Overflow-checked sum (same role as loader.cpp's add_checked).
bool add_checked(std::uint64_t a, std::uint64_t b, std::uint64_t &out) {
  if (a > UINT64_MAX - b) return false;
  out = a + b;
  return true;
}

std::uint64_t prod_dims(const std::vector<std::int64_t> &dims) {
  std::uint64_t n = 1;
  for (std::int64_t d : dims) {
    if (d <= 0) return 0;
    if (n > UINT64_MAX / static_cast<std::uint64_t>(d)) return 0;
    n *= static_cast<std::uint64_t>(d);
  }
  return n;
}

bool kv_f64(const gguf::File &f, const char *key, double &out) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) return false;
  const auto &v = it->second;
  if (v.type == gguf::ValueType::F32 || v.type == gguf::ValueType::F64) {
    out = v.f;
    return true;
  }
  return false;
}

// F32 array KV of exactly 3 (image_mean / image_std) -> out[3].
bool kv_f32x3(const gguf::File &f, const char *key, double *out) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) return false;
  const auto &v = it->second;
  if (v.type != gguf::ValueType::ARRAY || v.arr_type != gguf::ValueType::F32 ||
      v.arr.size() != 3) {
    return false;
  }
  for (int i = 0; i < 3; ++i) out[i] = v.arr[static_cast<std::size_t>(i)].f;
  return true;
}

bool kv_bool_vec(const gguf::File &f, const char *key, std::vector<std::uint64_t> &out) {
  auto it = f.kv.find(key);
  if (it == f.kv.end() || it->second.type != gguf::ValueType::ARRAY ||
      it->second.arr_type != gguf::ValueType::BOOL) {
    return false;
  }
  out.clear();
  out.reserve(it->second.arr.size());
  for (const auto &e : it->second.arr) out.push_back(e.u);
  return true;
}

std::string dims_str(const std::vector<std::int64_t> &dims) {
  std::string s = "[";
  for (std::size_t d = 0; d < dims.size(); ++d) {
    if (d) s += ",";
    s += std::to_string(dims[d]);
  }
  s += "]";
  return s;
}

struct Expect {
  std::string name;
  std::vector<std::int64_t> dims;
  // Allowed ggml type ids for this tensor (weights: F16|Q8_0 across the two
  // shipped files; biases/norms/embeddings: F32; patch embed: F16|F32).
  std::vector<std::uint32_t> dtypes;
};

}  // namespace

const char *vision_dtype_name(VisionDType t) {
  switch (t) {
    case VisionDType::F32: return "f32";
    case VisionDType::F16: return "f16";
    case VisionDType::Q8_0: return "q8_0";
  }
  return "?";
}

std::uint64_t vision_tensor_bytes(VisionDType t, std::uint64_t nelem) {
  const std::uint64_t be = vision_block_elems(t);
  const std::uint64_t blocks = (nelem + be - 1) / be;
  return blocks * vision_block_bytes(t);
}

bool parse_clip_vision_config(const gguf::File &f, ClipVisionConfig &cfg,
                              std::string &err) {
  if (gguf::kv_str(f, "general.architecture") != "clip") {
    err = "general.architecture is not clip";
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
  cfg = ClipVisionConfig{};
  if (!req_u32("clip.vision.block_count", cfg.block_count)) return false;
  if (!req_u32("clip.vision.embedding_length", cfg.embedding_length)) return false;
  if (!req_u32("clip.vision.feed_forward_length", cfg.feed_forward_length)) return false;
  if (!req_u32("clip.vision.attention.head_count", cfg.head_count)) return false;
  if (!req_u32("clip.vision.image_size", cfg.image_size)) return false;
  if (!req_u32("clip.vision.patch_size", cfg.patch_size)) return false;
  if (!req_u32("clip.vision.projection_dim", cfg.projection_dim)) return false;
  if (!req_u32("clip.vision.spatial_merge_size", cfg.spatial_merge_size)) return false;
  if (!kv_f64(f, "clip.vision.attention.layer_norm_epsilon", cfg.layer_norm_eps)) {
    err = "missing or wrong-type KV: clip.vision.attention.layer_norm_epsilon";
    return false;
  }
  if (!kv_f32x3(f, "clip.vision.image_mean", cfg.image_mean)) {
    err = "missing or wrong-type KV: clip.vision.image_mean (want 3 floats)";
    return false;
  }
  if (!kv_f32x3(f, "clip.vision.image_std", cfg.image_std)) {
    err = "missing or wrong-type KV: clip.vision.image_std (want 3 floats)";
    return false;
  }
  {
    auto it = f.kv.find("clip.has_vision_encoder");
    if (it == f.kv.end() || it->second.type != gguf::ValueType::BOOL) {
      err = "missing or wrong-type KV: clip.has_vision_encoder";
      return false;
    }
    cfg.has_vision_encoder = it->second.u != 0;
  }
  {
    auto it = f.kv.find("clip.use_gelu");
    if (it == f.kv.end() || it->second.type != gguf::ValueType::BOOL) {
      err = "missing or wrong-type KV: clip.use_gelu";
      return false;
    }
    cfg.use_gelu = it->second.u != 0;
  }
  cfg.projector_type = gguf::kv_str(f, "clip.projector_type");
  if (cfg.projector_type != "qwen3vl_merger") {
    err = "clip.projector_type is \"" + cfg.projector_type +
          "\", only qwen3vl_merger is supported";
    return false;
  }
  if (!kv_bool_vec(f, "clip.vision.is_deepstack_layers", cfg.deepstack_flags)) {
    err = "missing or wrong-type KV: clip.vision.is_deepstack_layers (want bool array)";
    return false;
  }
  // --- value validation (ranges first: the products below are file-driven) ---
  if (cfg.block_count < 1 || cfg.block_count > 1024) {
    err = "clip.vision.block_count must be in [1,1024] (got " +
          std::to_string(cfg.block_count) + ")";
    return false;
  }
  if (cfg.embedding_length < 1 || cfg.embedding_length > 65536) {
    err = "clip.vision.embedding_length out of range (got " +
          std::to_string(cfg.embedding_length) + ")";
    return false;
  }
  if (cfg.feed_forward_length < 1 || cfg.feed_forward_length > 1048576) {
    err = "clip.vision.feed_forward_length out of range (got " +
          std::to_string(cfg.feed_forward_length) + ")";
    return false;
  }
  if (cfg.head_count < 1 || cfg.head_count > 1024 ||
      cfg.embedding_length % cfg.head_count != 0) {
    err = "clip.vision.attention.head_count must divide embedding_length (got heads=" +
          std::to_string(cfg.head_count) + " emb=" + std::to_string(cfg.embedding_length) + ")";
    return false;
  }
  if (cfg.patch_size < 1 || cfg.patch_size > 256 || cfg.image_size < cfg.patch_size ||
      cfg.image_size > 8192 || cfg.image_size % cfg.patch_size != 0) {
    err = "clip.vision.image_size must be a multiple of patch_size (got image=" +
          std::to_string(cfg.image_size) + " patch=" + std::to_string(cfg.patch_size) + ")";
    return false;
  }
  if (cfg.spatial_merge_size < 1 || cfg.spatial_merge_size > 8) {
    err = "clip.vision.spatial_merge_size out of range (got " +
          std::to_string(cfg.spatial_merge_size) + ")";
    return false;
  }
  if (cfg.projection_dim < 1 || cfg.projection_dim > 65536) {
    err = "clip.vision.projection_dim out of range (got " +
          std::to_string(cfg.projection_dim) + ")";
    return false;
  }
  if (!(cfg.layer_norm_eps > 0.0)) {
    err = "clip.vision.attention.layer_norm_epsilon must be positive";
    return false;
  }
  for (int i = 0; i < 3; ++i) {
    if (!(cfg.image_std[i] > 0.0)) {
      err = "clip.vision.image_std must be positive in all channels";
      return false;
    }
  }
  if (!cfg.has_vision_encoder) {
    err = "clip.has_vision_encoder is false";
    return false;
  }
  if (cfg.deepstack_flags.size() != cfg.block_count) {
    err = "clip.vision.is_deepstack_layers has " + std::to_string(cfg.deepstack_flags.size()) +
          " entries for " + std::to_string(cfg.block_count) + " blocks";
    return false;
  }
  for (std::size_t i = 0; i < cfg.deepstack_flags.size(); ++i) {
    if (cfg.deepstack_flags[i]) {
      err = "deepstack is set on block " + std::to_string(i) +
            ": deepstack layers are not supported (no reference layout)";
      return false;
    }
  }
  return true;
}

std::vector<std::pair<std::string, std::vector<std::int64_t>>>
expected_vision_block_tensors(const ClipVisionConfig &cfg, std::uint32_t i) {
  if (i >= cfg.block_count) return {};
  const std::int64_t emb = static_cast<std::int64_t>(cfg.embedding_length);
  const std::int64_t ffn = static_cast<std::int64_t>(cfg.feed_forward_length);
  return {
      {"attn_qkv.weight", {emb, 3 * emb}},
      {"attn_qkv.bias", {3 * emb}},
      {"attn_out.weight", {emb, emb}},
      {"attn_out.bias", {emb}},
      {"ffn_up.weight", {emb, ffn}},
      {"ffn_up.bias", {ffn}},
      {"ffn_down.weight", {ffn, emb}},
      {"ffn_down.bias", {emb}},
      {"ln1.weight", {emb}},
      {"ln1.bias", {emb}},
      {"ln2.weight", {emb}},
      {"ln2.bias", {emb}},
  };
}

namespace {

// Allowed dtypes per tensor class (ggml ids): weights differ between the two
// shipped files (F16 vs Q8_0); biases/norms/embeddings do not.
const std::vector<std::uint32_t> kWeightTypes = {1, 8};    // F16 | Q8_0
const std::vector<std::uint32_t> kNormTypes = {0};         // F32
const std::vector<std::uint32_t> kPatchTypes = {1, 0};     // F16 | F32

std::vector<Expect> top_level_expects(const ClipVisionConfig &cfg) {
  const std::int64_t emb = static_cast<std::int64_t>(cfg.embedding_length);
  const std::int64_t p = static_cast<std::int64_t>(cfg.patch_size);
  const std::int64_t grid = static_cast<std::int64_t>(cfg.image_size) / p;
  const std::int64_t merge = static_cast<std::int64_t>(clip_merger_width(cfg));
  const std::int64_t proj = static_cast<std::int64_t>(cfg.projection_dim);
  return {
      {"v.patch_embd.weight", {p, p, 3, emb}, kPatchTypes},
      {"v.patch_embd.weight.1", {p, p, 3, emb}, kPatchTypes},
      {"v.patch_embd.bias", {emb}, kNormTypes},
      {"v.position_embd.weight", {emb, grid * grid}, kNormTypes},
      {"v.post_ln.weight", {emb}, kNormTypes},
      {"v.post_ln.bias", {emb}, kNormTypes},
      {"mm.0.weight", {merge, merge}, kWeightTypes},
      {"mm.0.bias", {merge}, kNormTypes},
      {"mm.2.weight", {merge, proj}, kWeightTypes},
      {"mm.2.bias", {proj}, kNormTypes},
  };
}

}  // namespace

VisionLoader::~VisionLoader() {
  if (fp_) std::fclose(fp_);
}

bool VisionLoader::open(const char *path, std::string &err) {
  if (fp_) {
    std::fclose(fp_);
    fp_ = nullptr;
  }
  fp_ = std::fopen(path, "rb");
  if (!fp_) {
    err = "cannot open file";
    return false;
  }
  path_ = path;
  bool parsed = false;
  try {
    parsed = gguf::read(fp_, f_, err);
  } catch (const std::exception &ex) {
    err = std::string("cannot parse GGUF header: ") + ex.what();
    parsed = false;
  }
  if (!parsed) {
    std::fclose(fp_);
    fp_ = nullptr;
    return false;
  }
  // Architecture first: a non-mmproj file (e.g. the text model) is rejected as
  // "not clip" here instead of tripping the dtype whitelist below with a
  // confusing message.
  if (gguf::kv_str(f_, "general.architecture") != "clip") {
    err = "general.architecture is not clip";
    std::fclose(fp_);
    fp_ = nullptr;
    return false;
  }
  struct stat st;
  if (stat(path, &st) != 0) {
    err = "cannot stat file";
    std::fclose(fp_);
    fp_ = nullptr;
    return false;
  }
  const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);

  const std::size_t n = f_.tensors.size();
  dtypes_.resize(n);
  elems_.resize(n);
  bytes_.resize(n);
  index_.clear();
  index_.reserve(n);

  for (std::size_t i = 0; i < n; ++i) {
    const gguf::TensorInfo &t = f_.tensors[i];
    VisionDType dt;
    if (!vision_dtype_from_ggml(t.dtype, dt)) {
      err = "tensor '" + t.name + "': ggml_type id " + std::to_string(t.dtype) +
            " (" + gguf::ggml_type_name(t.dtype) + ") outside the vision set {f32,f16,q8_0}";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    for (std::int64_t d : t.dims) {
      if (d <= 0) {
        err = "tensor '" + t.name + "': non-positive dimension " + std::to_string(d);
        std::fclose(fp_);
        fp_ = nullptr;
        return false;
      }
    }
    const std::uint32_t be = vision_block_elems(dt);
    if (!t.dims.empty() && be > 1 && (t.dims[0] % static_cast<std::int64_t>(be)) != 0) {
      err = "tensor '" + t.name + "': ne0=" + std::to_string(t.dims[0]) +
            " is not a multiple of the " + vision_dtype_name(dt) + " block size " +
            std::to_string(be);
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    dtypes_[i] = dt;
    elems_[i] = prod_dims(t.dims);
    bytes_[i] = vision_tensor_bytes(dt, elems_[i]);
    if (bytes_[i] == 0) {
      err = "tensor '" + t.name + "': computed size is 0 (overflow or empty)";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    if (bytes_[i] > file_size) {
      err = "tensor '" + t.name + "': computed size " + std::to_string(bytes_[i]) +
            " exceeds the " + std::to_string(file_size) + "-byte file";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    total_bytes_ += bytes_[i];
    if (index_.count(t.name)) {
      err = "duplicate tensor name '" + t.name + "'";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    index_[t.name] = i;
  }

  const std::uint64_t doff = f_.data_offset;
  const std::size_t align = f_.alignment;
  if (doff > file_size) {
    err = "data section offset " + std::to_string(doff) + " is past the end of the " +
          std::to_string(file_size) + "-byte file";
    std::fclose(fp_);
    fp_ = nullptr;
    return false;
  }
  for (std::size_t i = 0; i < n; ++i) {
    const gguf::TensorInfo &t = f_.tensors[i];
    std::uint64_t start = 0, abs_end = 0, bound = file_size;
    const bool sums_ok =
        add_checked(doff, t.offset, start) && add_checked(start, bytes_[i], abs_end) &&
        (i + 1 >= n || add_checked(doff, f_.tensors[i + 1].offset, bound));
    if (!sums_ok) {
      err = "tensor '" + t.name + "': offset/size sum overflows 64 bits";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    if (abs_end > bound) {
      err = "tensor '" + f_.tensors[i].name + "': size overflows into next tensor / EOF";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    if (align != 0 && abs_end % align != 0) {
      err = "tensor '" + f_.tensors[i].name + "': end not aligned";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
  }
  return true;
}

const gguf::TensorInfo *VisionLoader::find(const char *name) const {
  auto it = index_.find(name);
  if (it == index_.end()) return nullptr;
  return &f_.tensors[it->second];
}

bool VisionLoader::load_tensor_range(std::size_t i, std::uint64_t off,
                                     std::uint64_t count, std::vector<std::uint8_t> &out,
                                     std::string &err) const {
  if (!fp_ || i >= f_.tensors.size()) {
    err = "bad tensor index";
    return false;
  }
  const gguf::TensorInfo &t = f_.tensors[i];
  std::uint64_t end = 0;
  if (!add_checked(off, count, end) || end > bytes_[i]) {
    err = "range out of tensor bounds";
    return false;
  }
  std::uint64_t abs = 0;
  if (!add_checked(f_.data_offset, t.offset, abs) || !add_checked(abs, off, abs)) {
    err = "tensor offset overflows 64 bits";
    return false;
  }
  const long where = static_cast<long>(abs);
  if (std::fseek(fp_, where, SEEK_SET) != 0) {
    err = "seek to tensor failed";
    return false;
  }
  out.resize(count);
  if (count > 0 && std::fread(out.data(), 1, count, fp_) != count) {
    err = "short read of tensor data";
    return false;
  }
  return true;
}

bool VisionLoader::load_tensor(std::size_t i, std::vector<std::uint8_t> &out,
                               std::string &err) const {
  return load_tensor_range(i, 0, bytes_[i], out, err);
}

bool validate_clip_vision_layout(const gguf::File &f, const ClipVisionConfig &cfg,
                                 std::string &err) {
  const std::string blk_prefix = "v.blk.";
  std::vector<std::map<std::string, const gguf::TensorInfo *>> blocks(cfg.block_count);
  std::map<std::string, const gguf::TensorInfo *> top;
  for (const auto &t : f.tensors) {
    const std::string &n = t.name;
    if (n.compare(0, blk_prefix.size(), blk_prefix) != 0) {
      if (top.count(n)) {
        err = "duplicate top-level tensor: " + n;
        return false;
      }
      top[n] = &t;
      continue;
    }
    const std::size_t dot2 = n.find('.', blk_prefix.size());
    if (dot2 == std::string::npos) {
      err = "malformed tensor name: " + n;
      return false;
    }
    const std::string idx = n.substr(blk_prefix.size(), dot2 - blk_prefix.size());
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
    if (blocks[li].count(rest)) {
      err = "duplicate tensor name: " + n;
      return false;
    }
    blocks[li][rest] = &t;
  }
  for (std::uint32_t i = 0; i < cfg.block_count; ++i) {
    const std::string prefix = blk_prefix + std::to_string(i) + ".";
    const auto expect = expected_vision_block_tensors(cfg, i);
    std::map<std::string, std::vector<std::int64_t>> expect_map(expect.begin(),
                                                                expect.end());
    for (const auto &kv : expect_map) {
      auto it = blocks[i].find(kv.first);
      if (it == blocks[i].end()) {
        err = "missing tensor: " + prefix + kv.first;
        return false;
      }
      if (it->second->dims != kv.second) {
        err = "tensor " + prefix + kv.first + ": dims " + dims_str(it->second->dims) +
              " != expected " + dims_str(kv.second);
        return false;
      }
      const std::uint32_t dt = it->second->dtype;
      // Only attn/ffn weights are quantized (F16|Q8_0 across the two shipped
      // files); norm weights, like all biases, are F32.
      const bool is_qweight =
          kv.first.size() >= 6 && kv.first.compare(kv.first.size() - 6, 6, "weight") == 0 &&
          kv.first.compare(0, 2, "ln") != 0;
      const bool dt_ok = is_qweight ? (dt == 1 || dt == 8) : (dt == 0);
      if (!dt_ok) {
        err = "tensor " + prefix + kv.first + ": dtype " + gguf::ggml_type_name(dt) +
              " not allowed here (want " + (is_qweight ? "f16|q8_0" : "f32") + ")";
        return false;
      }
    }
    for (const auto &kv : blocks[i]) {
      if (!expect_map.count(kv.first)) {
        err = "unexpected tensor: " + prefix + kv.first;
        return false;
      }
    }
  }
  const auto top_expect = top_level_expects(cfg);
  for (const auto &e : top_expect) {
    auto it = top.find(e.name);
    if (it == top.end()) {
      err = "missing top-level tensor: " + e.name;
      return false;
    }
    if (it->second->dims != e.dims) {
      err = e.name + ": dims " + dims_str(it->second->dims) + " != expected " +
            dims_str(e.dims);
      return false;
    }
    bool dt_ok = false;
    for (std::uint32_t allowed : e.dtypes) {
      if (it->second->dtype == allowed) {
        dt_ok = true;
        break;
      }
    }
    if (!dt_ok) {
      err = e.name + ": dtype " + gguf::ggml_type_name(it->second->dtype) + " not allowed";
      return false;
    }
  }
  for (const auto &kv : top) {
    bool known = false;
    for (const auto &e : top_expect) {
      if (kv.first == e.name) {
        known = true;
        break;
      }
    }
    if (!known) {
      err = "unexpected top-level tensor: " + kv.first;
      return false;
    }
  }
  return true;
}

}  // namespace rdna4
