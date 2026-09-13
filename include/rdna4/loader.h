#pragma once
// GGUF tensor loader (PLAN.md M1 step 2).
//
// GgufLoader owns an open UD GGUF file. open() parses the header and
// fail-fast validates:
//   - every tensor dtype is in the DType whitelist (M1 step 4: "resto é erro");
//   - tensor names are unique;
//   - tensor geometry: each tensor's computed size fits the span until the
//     next tensor (no overlaps, fits the file), and data_offset + offset +
//     size is aligned. Both UD files pass this exactly (scripts/
//     check_geometry.py mirrors these checks in Python).
//
// Tensor DATA stays on disk; load_tensor() copies one tensor's raw
// (still-quantized) bytes into host memory on demand. GPU staging is M2+.
#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/gguf.h"

namespace rdna4 {

class GgufLoader {
 public:
  GgufLoader() = default;
  ~GgufLoader();
  GgufLoader(const GgufLoader &) = delete;
  GgufLoader &operator=(const GgufLoader &) = delete;

  // Parses the header and runs all validations. The file is opened read-only
  // and kept open for load_tensor().
  bool open(const char *path, std::string &err);

  const gguf::File &meta() const { return f_; }
  std::size_t n_tensors() const { return f_.tensors.size(); }
  const gguf::TensorInfo *find(const char *name) const;  // nullptr if missing
  DType dtype(std::size_t i) const { return dtypes_[i]; }
  std::uint64_t tensor_elems(std::size_t i) const { return elems_[i]; }
  std::uint64_t tensor_bytes(std::size_t i) const { return bytes_[i]; }
  std::uint64_t total_bytes() const { return total_bytes_; }

  // Copies the raw quantized bytes of tensor i into `out` (resized to
  // tensor_bytes(i)).
  bool load_tensor(std::size_t i, std::vector<std::uint8_t> &out, std::string &err) const;

 private:
  std::string path_;
  mutable FILE *fp_ = nullptr;
  gguf::File f_;
  std::vector<DType> dtypes_;       // parallel to f_.tensors
  std::vector<std::uint64_t> elems_;  // parallel
  std::vector<std::uint64_t> bytes_;  // parallel
  std::uint64_t total_bytes_ = 0;
  std::unordered_map<std::string, std::size_t> index_;
};

}  // namespace rdna4
