// GgufLoader implementation (PLAN.md M1 step 2).
#include "rdna4/loader.h"

#include <sys/stat.h>

#include <exception>

namespace rdna4 {
namespace {

// Overflow-checked product: a wrapped product could make the geometry check
// below accept a corrupt tensor (review finding M3).
std::uint64_t prod_dims(const std::vector<std::int64_t> &dims) {
  std::uint64_t n = 1;
  for (std::int64_t d : dims) {
    if (d <= 0) return 0;  // rejected by the caller with a clear error
    if (n > UINT64_MAX / static_cast<std::uint64_t>(d)) return 0;  // overflow
    n *= static_cast<std::uint64_t>(d);
  }
  return n;
}

// Overflow-checked sum (review finding M7).
bool add_checked(std::uint64_t a, std::uint64_t b, std::uint64_t &out) {
  if (a > UINT64_MAX - b) return false;
  out = a + b;
  return true;
}

}  // namespace

GgufLoader::~GgufLoader() {
  if (fp_) {
    std::fclose(fp_);
  }
}

bool GgufLoader::open(const char *path, std::string &err) {
  if (fp_) {  // reopening: a second open() used to leak the previous FILE*
    std::fclose(fp_);
    fp_ = nullptr;
  }
  fp_ = std::fopen(path, "rb");
  if (!fp_) {
    err = "cannot open file";
    return false;
  }
  path_ = path;

  // The header is file-controlled, so its parse can throw (std::bad_alloc /
  // std::length_error from the string and vector allocations). Translate that
  // into the normal error path instead of letting the process abort: the
  // per-field caps in gguf.cpp bound each length, not their sum (review finding
  // M8).
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
    // Whitelist (M1 step 4).
    auto dt = dtype_from_ggml(t.dtype);
    if (!dt) {
      err = "tensor '" + t.name + "': ggml_type id " + std::to_string(t.dtype) +
            " (" + gguf::ggml_type_name(t.dtype) + ") outside whitelist";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    // Dimension sanity + per-row block divisibility (review finding M7/M3):
    // ggml sizes quantized tensors per row, so ne[0] must be a whole number of
    // blocks. A non-multiple would make tensor_bytes() undercount and the
    // geometry check pass while the dequant reads a wrong range. Also reject
    // non-positive dims before the unsigned product below.
    for (std::int64_t d : t.dims) {
      if (d <= 0) {
        err = "tensor '" + t.name + "': non-positive dimension " + std::to_string(d);
        std::fclose(fp_);
        fp_ = nullptr;
        return false;
      }
    }
    const std::uint32_t be = rdna4::dtype_block_elems(*dt);
    if (!t.dims.empty() && be > 1 && (t.dims[0] % static_cast<std::int64_t>(be)) != 0) {
      err = "tensor '" + t.name + "': ne0=" + std::to_string(t.dims[0]) +
            " is not a multiple of the " + rdna4::dtype_name(*dt) + " block size " +
            std::to_string(be);
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    dtypes_[i] = *dt;
    elems_[i] = prod_dims(t.dims);
    bytes_[i] = rdna4::tensor_bytes(*dt, elems_[i]);
    if (bytes_[i] == 0) {
      err = "tensor '" + t.name + "': computed size is 0 (overflow or empty)";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    // tensor_bytes() rounds up and multiplies without an overflow check, so a
    // wrapped product can land on a small non-zero value. A tensor can never be
    // larger than the file that holds it, which makes file_size a sound and
    // cheap upper bound (review finding M7).
    if (bytes_[i] > file_size) {
      err = "tensor '" + t.name + "': computed size " + std::to_string(bytes_[i]) +
            " exceeds the " + std::to_string(file_size) + "-byte file";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    total_bytes_ += bytes_[i];
    // Unique names.
    if (index_.count(t.name)) {
      err = "duplicate tensor name '" + t.name + "'";
      std::fclose(fp_);
      fp_ = nullptr;
      return false;
    }
    index_[t.name] = i;
  }

  // Geometry: no tensor may overflow the span until the next tensor (or EOF),
  // and each tensor end must be aligned. Mirrors scripts/check_geometry.py.
  //
  // All three operands of the sums below come from the file, so the sums are
  // checked: unsigned wrap would make a forged offset pass both guards and the
  // later fseek would land in the GGUF header, loading header bytes as weights
  // (review finding M7).
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
    std::uint64_t start = 0;
    std::uint64_t abs_end = 0;
    std::uint64_t bound = file_size;
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
      err = "tensor '" + f_.tensors[i].name +
            "': size overflows into next tensor / EOF";
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

const gguf::TensorInfo *GgufLoader::find(const char *name) const {
  auto it = index_.find(name);
  if (it == index_.end()) {
    return nullptr;
  }
  return &f_.tensors[it->second];
}

bool GgufLoader::load_tensor_range(std::size_t i, std::uint64_t off,
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

bool GgufLoader::load_tensor(std::size_t i, std::vector<std::uint8_t> &out,
                             std::string &err) const {
  return load_tensor_range(i, 0, bytes_[i], out, err);
}

}  // namespace rdna4
