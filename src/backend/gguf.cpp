// Minimal GGUF reader implementation. Sequential parse of the header section
// (magic, version, counts, KV pairs, tensor entries) per ggml/include/gguf.h.
#include "rdna4/gguf.h"

#include <cstdio>

namespace rdna4 {
namespace gguf {
namespace {

struct Cursor {
  FILE *fp = nullptr;
  bool ok = true;

  void read(void *dst, std::size_t n) {
    if (!ok || std::fread(dst, 1, n, fp) != n) {
      ok = false;
    }
  }
  template <typename T>
  T get() {
    T v = {};
    read(&v, sizeof(v));
    return v;
  }
  std::string get_str() {
    const std::uint64_t n = get<std::uint64_t>();
    std::string s;
    if (!ok) {
      return s;
    }
    s.resize(n);
    if (n > 0) {
      read(s.data(), n);
    }
    return s;
  }
};

bool read_value(Cursor &c, ValueType type, Value &out) {
  out.type = type;
  switch (type) {
    case U8:
      out.u = c.get<std::uint8_t>();
      break;
    case I8:
      out.u = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.get<std::int8_t>()));
      break;
    case U16:
      out.u = c.get<std::uint16_t>();
      break;
    case I16:
      out.u = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.get<std::int16_t>()));
      break;
    case U32:
      out.u = c.get<std::uint32_t>();
      break;
    case I32:
      out.u = static_cast<std::uint64_t>(static_cast<std::int64_t>(c.get<std::int32_t>()));
      break;
    case F32:
      out.f = c.get<float>();
      break;
    case BOOL:
      out.u = c.get<std::int8_t>() != 0;
      break;
    case STR:
      out.s = c.get_str();
      break;
    case U64:
      out.u = c.get<std::uint64_t>();
      break;
    case I64:
      out.u = static_cast<std::uint64_t>(c.get<std::int64_t>());
      break;
    case F64:
      out.f = c.get<double>();
      break;
    case ARRAY: {
      out.arr_type = static_cast<ValueType>(c.get<std::int32_t>());
      const std::uint64_t n = c.get<std::uint64_t>();
      if (n > (1u << 28)) {
        c.ok = false;  // sanity cap: 256M elements
        break;
      }
      out.arr.resize(n);
      for (std::uint64_t i = 0; i < n && c.ok; ++i) {
        if (!read_value(c, out.arr_type, out.arr[i])) {
          break;
        }
      }
      break;
    }
    default:
      c.ok = false;
      break;
  }
  return c.ok;
}

}  // namespace

const char *ggml_type_name(std::uint32_t dtype) {
  // Matches enum ggml_type in ggml/include/ggml.h.
  static const char *names[] = {
      "f32",     "f16",     "q4_0",    "q4_1",    "?",       "?",       "q5_0",
      "q5_1",    "q8_0",    "q8_1",    "q2_k",    "q3_k",    "q4_k",    "q5_k",
      "q6_k",    "q8_k",    "iq2_xxs", "iq2_xs",  "iq3_xxs", "iq1_s",   "iq4_nl",
      "iq3_s",   "iq2_s",   "iq4_xs",  "i8",      "i16",     "i32",     "i64",
      "f64",     "iq1_m",   "bf16",    "?",       "?",       "?",       "tq1_0",
      "tq2_0",   "?",       "?",       "?",       "mxfp4",   "nvfp4",   "q1_0",
      "q2_0",
  };
  const std::size_t n = sizeof(names) / sizeof(names[0]);
  return dtype < n ? names[dtype] : "?";
}

bool read(const char *path, File &out, std::string &err) {
  File f;
  FILE *fp = std::fopen(path, "rb");
  if (!fp) {
    err = "cannot open file";
    return false;
  }
  Cursor c{fp, true};

  char magic[4] = {};
  c.read(magic, 4);
  if (!c.ok || magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F') {
    err = "bad GGUF magic";
    std::fclose(fp);
    return false;
  }
  f.version = c.get<std::uint32_t>();
  if (f.version != 3) {
    err = "unsupported GGUF version (want 3)";
    std::fclose(fp);
    return false;
  }
  const std::int64_t n_tensors = c.get<std::int64_t>();
  const std::int64_t n_kv = c.get<std::int64_t>();
  if (!c.ok || n_tensors < 0 || n_tensors > (1 << 24) || n_kv < 0 || n_kv > (1 << 20)) {
    err = "insane header counts";
    std::fclose(fp);
    return false;
  }

  for (std::int64_t i = 0; i < n_kv && c.ok; ++i) {
    const std::string key = c.get_str();
    const auto type = static_cast<ValueType>(c.get<std::int32_t>());
    Value v;
    if (!read_value(c, type, v)) {
      break;
    }
    f.kv[key] = v;
  }

  f.tensors.reserve(n_tensors);
  for (std::int64_t i = 0; i < n_tensors && c.ok; ++i) {
    TensorInfo t;
    t.name = c.get_str();
    const std::uint32_t n_dims = c.get<std::uint32_t>();
    if (n_dims > 4) {
      c.ok = false;
      break;
    }
    t.dims.resize(n_dims);
    for (std::uint32_t d = 0; d < n_dims && c.ok; ++d) {
      t.dims[d] = c.get<std::int64_t>();
    }
    t.dtype = c.get<std::uint32_t>();
    t.offset = c.get<std::uint64_t>();
    f.tensors.push_back(t);
  }

  std::fclose(fp);
  if (!c.ok) {
    err = "truncated or corrupt header";
    return false;
  }
  auto it = f.kv.find("general.alignment");
  if (it != f.kv.end() && it->second.type == U32) {
    f.alignment = static_cast<std::size_t>(it->second.u);
  }
  out = f;
  return true;
}

std::string kv_str(const File &f, const char *key, const char *dflt) {
  auto it = f.kv.find(key);
  if (it == f.kv.end() || it->second.type != STR) {
    return dflt;
  }
  return it->second.s;
}

std::int64_t kv_int(const File &f, const char *key, std::int64_t dflt) {
  auto it = f.kv.find(key);
  if (it == f.kv.end()) {
    return dflt;
  }
  const Value &v = it->second;
  switch (v.type) {
    case U8:
    case U16:
    case U32:
    case U64:
    case BOOL:
      return static_cast<std::int64_t>(v.u);
    case I8:
    case I16:
    case I32:
    case I64:
      return static_cast<std::int64_t>(v.u);
    default:
      return dflt;
  }
}

}  // namespace gguf
}  // namespace rdna4
