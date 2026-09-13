// Minimal GGUF reader for rdna4-infer (PLAN.md M1).
// Reads header, KV pairs and tensor entries; tensor DATA stays on disk
// (offsets recorded, mapped in M2). No ggml dependency by design.
#pragma once

#include <cstdio>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace rdna4 {
namespace gguf {

// Mirrors gguf_type in ggml/include/gguf.h (enums stored as int32).
enum ValueType : std::int32_t {
  U8 = 0,
  I8 = 1,
  U16 = 2,
  I16 = 3,
  U32 = 4,
  I32 = 5,
  F32 = 6,
  BOOL = 7,
  STR = 8,
  ARRAY = 9,
  U64 = 10,
  I64 = 11,
  F64 = 12,
};

struct Value {
  ValueType type = U8;
  std::uint64_t u = 0;  // ints, bool
  double f = 0.0;       // floats
  std::string s;        // strings
  ValueType arr_type = U8;
  std::vector<Value> arr;  // arrays
};

struct TensorInfo {
  std::string name;
  std::vector<std::int64_t> dims;
  std::uint32_t dtype = 0;  // ggml_type id, see ggml_type_name()
  std::uint64_t offset = 0;  // offset within the data section (see File::data_offset)
};

struct File {
  std::uint32_t version = 0;
  std::size_t alignment = 32;
  // Start of the tensor data section: header end padded to `alignment`
  // (GGML_PAD), per the GGUF v3 reader. Tensor offsets are relative to this.
  std::uint64_t data_offset = 0;
  std::map<std::string, Value> kv;
  std::vector<TensorInfo> tensors;
};

// Returns true on success; on failure returns false with err set.
// Opens (and closes) the file. The second overload reads from an already
// open file and leaves it open, positioned at the start of the data section.
bool read(const char *path, File &out, std::string &err);
bool read(FILE *fp, File &out, std::string &err);

// Numeric ggml_type id (ggml/include/ggml.h) to name; "?" if unknown.
const char *ggml_type_name(std::uint32_t dtype);

// Convenience accessors (abort-free: return default when missing/wrong type).
std::string kv_str(const File &f, const char *key, const char *dflt = "");
std::int64_t kv_int(const File &f, const char *key, std::int64_t dflt = 0);

}  // namespace gguf
}  // namespace rdna4
