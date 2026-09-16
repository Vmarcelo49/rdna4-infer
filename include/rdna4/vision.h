#pragma once
// Vision (mmproj) file loading for the qwen35 companion encoder (M-vision).
//
// The project's text model (`general.architecture == "qwen35"`) ships next to
// `mmproj-*` files carrying its vision tower: `general.architecture == "clip"`,
// a 27-block ViT (1152 emb, 16 heads, patch 16) plus the `qwen3vl_merger`
// projector into the text embedding width (llama.cpp: tools/mtmd/clip.cpp,
// PROJECTOR_TYPE_QWEN3VL path; tensor names below match its TN_* lookups).
// SPEC.md keeps vision *inference* out of v1; this module ports *loading*:
// open + fail-fast validate + read raw tensor bytes, so a future encoder has
// a trusted inventory to build on.
//
// What is validated (mirrors model.h/model.cpp for the text side):
//   - parse_clip_vision_config(): arch == "clip", the required clip.vision.*
//     KV, and projector_type == "qwen3vl_merger" (any other projector has a
//     different tensor layout and is rejected, not silently misread).
//   - validate_clip_vision_layout(): the exact tensor inventory — 12 tensors
//     per v.blk.i with config-derived shapes, plus the top-level patch/pos
//     embeddings, post norm and mm.0/mm.2 merger pair. No missing, no extras,
//     dims match. Weights may be F16 or Q8_0 (the shipped mmproj-F16 and
//     mmproj-q8_0 files); biases/norms/embeddings are F32, patch embeddings
//     F16 or F32.
//
// Why a separate loader class: GgufLoader (loader.h) hard-rejects any dtype
// outside the text-engine whitelist, and F16 is deliberately not in it
// (dtype.h). Vision weights ARE F16/Q8_0, so VisionLoader owns its own open
// file and applies the same header/geometry guards (unique names, block
// divisibility, checked offset sums, alignment) for the three ggml types the
// shipped mmproj files contain (F32=0, F16=1, Q8_0=8). Anything else fails
// with the tensor name in the message.
//
// Out of scope on purpose (rejected loudly, not half-supported):
//   - deepstack layers (clip.vision.is_deepstack_layers all-false in both
//     shipped files; a file with any set is rejected — its extra tensor
//     names have no on-disk example to pin down);
//   - the ViT forward itself, image decode/preprocess, and multimodal prompt
//     assembly (needs new HIP kernels + MRoPE-vision positions; SPEC non-goal).
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "rdna4/gguf.h"

namespace rdna4 {

// ggml type ids accepted in mmproj files (the shipped files use only these).
enum class VisionDType : std::uint32_t {
  F32 = 0,
  F16 = 1,
  Q8_0 = 8,
};

const char *vision_dtype_name(VisionDType t);
// Byte size of a tensor with `nelem` elements (ggml row-size semantics).
std::uint64_t vision_tensor_bytes(VisionDType t, std::uint64_t nelem);

struct ClipVisionConfig {
  std::uint32_t block_count = 0;
  std::uint32_t embedding_length = 0;    // ViT width (1152)
  std::uint32_t feed_forward_length = 0;  // ViT FFN width (4304)
  std::uint32_t head_count = 0;           // ViT heads (16)
  std::uint32_t image_size = 0;           // 768
  std::uint32_t patch_size = 0;           // 16
  std::uint32_t projection_dim = 0;       // merger output = text emb (5120)
  std::uint32_t spatial_merge_size = 0;   // 2 (2x2 patch concat)
  double layer_norm_eps = 0.0;
  double image_mean[3] = {0.0, 0.0, 0.0};
  double image_std[3] = {0.0, 0.0, 0.0};
  bool has_vision_encoder = false;
  bool use_gelu = false;
  std::string projector_type;  // must be "qwen3vl_merger"
  // Raw is_deepstack_layers flags, one per block. Any set bit rejects the
  // file (see header note): deepstack tensor naming has no on-disk example.
  std::vector<std::uint64_t> deepstack_flags;
};

// Parses + validates the required clip KV. Fails if general.architecture !=
// "clip", any required KV is missing/wrong type, values are out of range, the
// projector is not qwen3vl_merger, or any deepstack flag is set.
bool parse_clip_vision_config(const gguf::File &f, ClipVisionConfig &cfg,
                              std::string &err);

// Merged patch width into the merger: emb * spatial_merge^2 (1152*4 = 4608).
inline std::uint64_t clip_merger_width(const ClipVisionConfig &cfg) {
  return static_cast<std::uint64_t>(cfg.embedding_length) * cfg.spatial_merge_size *
         cfg.spatial_merge_size;
}

// Expected (name, dims) for vision block i, in a deterministic order.
// Returns empty on an out-of-range i.
std::vector<std::pair<std::string, std::vector<std::int64_t>>>
expected_vision_block_tensors(const ClipVisionConfig &cfg, std::uint32_t i);

// Minimal GGUF reader for mmproj files. Same contract as GgufLoader (header +
// geometry guards, data stays on disk, load_tensor() copies raw bytes), but
// over the {F32, F16, Q8_0} vision dtype set instead of the text whitelist.
class VisionLoader {
 public:
  VisionLoader() = default;
  ~VisionLoader();
  VisionLoader(const VisionLoader &) = delete;
  VisionLoader &operator=(const VisionLoader &) = delete;

  bool open(const char *path, std::string &err);

  const gguf::File &meta() const { return f_; }
  std::size_t n_tensors() const { return f_.tensors.size(); }
  const gguf::TensorInfo *find(const char *name) const;  // nullptr if missing
  VisionDType dtype(std::size_t i) const { return dtypes_[i]; }
  std::uint64_t tensor_elems(std::size_t i) const { return elems_[i]; }
  std::uint64_t tensor_bytes(std::size_t i) const { return bytes_[i]; }
  std::uint64_t total_bytes() const { return total_bytes_; }

  bool load_tensor(std::size_t i, std::vector<std::uint8_t> &out,
                   std::string &err) const;
  bool load_tensor_range(std::size_t i, std::uint64_t off, std::uint64_t count,
                         std::vector<std::uint8_t> &out, std::string &err) const;

 private:
  std::string path_;
  mutable FILE *fp_ = nullptr;
  gguf::File f_;
  std::vector<VisionDType> dtypes_;       // parallel to f_.tensors
  std::vector<std::uint64_t> elems_;      // parallel
  std::vector<std::uint64_t> bytes_;      // parallel
  std::uint64_t total_bytes_ = 0;
  std::unordered_map<std::string, std::size_t> index_;
};

// Validates the whole inventory against the config: every v.blk.i matches
// its expected set exactly and the top-level tensors are present with the
// right dims and dtypes (weights F16|Q8_0, rest per header note). Takes the
// GGUF File (not the loader) so forged inventories are unit-testable.
bool validate_clip_vision_layout(const gguf::File &f, const ClipVisionConfig &cfg,
                                 std::string &err);

}  // namespace rdna4
