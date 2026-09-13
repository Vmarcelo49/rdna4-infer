// M2 step 1 — CPU dequant oracle + real-tensor validation (no GPU, no VRAM).
//
// Links the reference dequant (llama.cpp libggml-base.so, MIT — extern "C")
// and validates the pipeline: GGUF raw bytes -> block structs -> dequantize.
//   (1) struct bridge: sizeof(block_X) == rdna4::dtype_block_bytes(X) for all
//       14 quantized types (M1 block table matches the reference layouts);
//   (2) per type: first tensor of the type dequantizes to finite, plausible
//       values (would be garbage if byte layout/offsets were wrong);
//   (3) Q8_0 pipeline: bit-exact vs the trivial inline formula d*qs[i]
//       (proves the raw-bytes -> struct cast and alignment);
//   (4) cross-file: rel-L2 between two quantizations of the same F32 weights
//       (file1 vs file2) — must be small (loose cap, informational+assert).
//
//   check-dequant <file1.gguf> [file2.gguf]
#include "ggml-quants.h"  // extern "C" dequantize_row_* + block structs (ggml-common.h)

#include "rdna4/dtype.h"
#include "rdna4/loader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

using DType = rdna4::DType;

struct TypeRef {
  DType dt;
  const char *name;
};

const TypeRef kTypes[] = {
    {DType::Q8_0, "q8_0"},
    {DType::Q2_K, "q2_k"},
    {DType::Q3_K, "q3_k"},
    {DType::Q4_K, "q4_k"},
    {DType::Q5_K, "q5_k"},
    {DType::Q6_K, "q6_k"},
    {DType::IQ2_XXS, "iq2_xxs"},
    {DType::IQ2_XS, "iq2_xs"},
    {DType::IQ3_XXS, "iq3_xxs"},
    {DType::IQ1_S, "iq1_s"},
    {DType::IQ4_NL, "iq4_nl"},
    {DType::IQ3_S, "iq3_s"},
    {DType::IQ2_S, "iq2_s"},
    {DType::IQ4_XS, "iq4_xs"},
};

// Reference dequant of `nelem` elements (multiple of the type's block elems).
void dequant_row(DType dt, const std::uint8_t *src, float *dst, std::int64_t nelem) {
  switch (dt) {
    case DType::Q8_0:
      dequantize_row_q8_0(reinterpret_cast<const block_q8_0 *>(src), dst, nelem);
      return;
    case DType::Q2_K:
      dequantize_row_q2_K(reinterpret_cast<const block_q2_K *>(src), dst, nelem);
      return;
    case DType::Q3_K:
      dequantize_row_q3_K(reinterpret_cast<const block_q3_K *>(src), dst, nelem);
      return;
    case DType::Q4_K:
      dequantize_row_q4_K(reinterpret_cast<const block_q4_K *>(src), dst, nelem);
      return;
    case DType::Q5_K:
      dequantize_row_q5_K(reinterpret_cast<const block_q5_K *>(src), dst, nelem);
      return;
    case DType::Q6_K:
      dequantize_row_q6_K(reinterpret_cast<const block_q6_K *>(src), dst, nelem);
      return;
    case DType::IQ2_XXS:
      dequantize_row_iq2_xxs(reinterpret_cast<const block_iq2_xxs *>(src), dst, nelem);
      return;
    case DType::IQ2_XS:
      dequantize_row_iq2_xs(reinterpret_cast<const block_iq2_xs *>(src), dst, nelem);
      return;
    case DType::IQ3_XXS:
      dequantize_row_iq3_xxs(reinterpret_cast<const block_iq3_xxs *>(src), dst, nelem);
      return;
    case DType::IQ1_S:
      dequantize_row_iq1_s(reinterpret_cast<const block_iq1_s *>(src), dst, nelem);
      return;
    case DType::IQ4_NL:
      dequantize_row_iq4_nl(reinterpret_cast<const block_iq4_nl *>(src), dst, nelem);
      return;
    case DType::IQ3_S:
      dequantize_row_iq3_s(reinterpret_cast<const block_iq3_s *>(src), dst, nelem);
      return;
    case DType::IQ2_S:
      dequantize_row_iq2_s(reinterpret_cast<const block_iq2_s *>(src), dst, nelem);
      return;
    case DType::IQ4_XS:
      dequantize_row_iq4_xs(reinterpret_cast<const block_iq4_xs *>(src), dst, nelem);
      return;
    default:
      std::fprintf(stderr, "dequant_row: unsupported dtype\n");
      std::exit(2);
  }
}

// (1) Reference struct size per type — must equal the M1 block table.
std::size_t struct_size_of(DType dt) {
  switch (dt) {
    case DType::Q8_0:
      return sizeof(block_q8_0);
    case DType::Q2_K:
      return sizeof(block_q2_K);
    case DType::Q3_K:
      return sizeof(block_q3_K);
    case DType::Q4_K:
      return sizeof(block_q4_K);
    case DType::Q5_K:
      return sizeof(block_q5_K);
    case DType::Q6_K:
      return sizeof(block_q6_K);
    case DType::IQ2_XXS:
      return sizeof(block_iq2_xxs);
    case DType::IQ2_XS:
      return sizeof(block_iq2_xs);
    case DType::IQ3_XXS:
      return sizeof(block_iq3_xxs);
    case DType::IQ1_S:
      return sizeof(block_iq1_s);
    case DType::IQ4_NL:
      return sizeof(block_iq4_nl);
    case DType::IQ3_S:
      return sizeof(block_iq3_s);
    case DType::IQ2_S:
      return sizeof(block_iq2_s);
    case DType::IQ4_XS:
      return sizeof(block_iq4_xs);
    default:
      return 0;
  }
}

std::int64_t prod_dims(const std::vector<std::int64_t> &dims, std::size_t from) {
  std::int64_t n = 1;
  for (std::size_t i = from; i < dims.size(); ++i) {
    n *= dims[i];
  }
  return n;
}

struct Stats {
  bool all_finite = true;
  bool in_range = true;
  float vmin = 0, vmax = 0, mean = 0;
};

Stats check_stats(const float *y, std::int64_t nelem, float bound) {
  Stats s;
  double sum = 0.0;
  for (std::int64_t i = 0; i < nelem; ++i) {
    const float v = y[i];
    if (!std::isfinite(v)) {
      s.all_finite = false;
      continue;
    }
    const float a = v < 0 ? -v : v;
    if (a > s.vmax) {
      s.vmax = a;
    }
    if (a > bound) {
      s.in_range = false;
    }
    if (i == 0 || v < s.vmin) {
      s.vmin = v;
    }
    sum += v;
  }
  s.mean = static_cast<float>(sum / nelem);
  return s;
}

// Dequantize the first `max_blocks` blocks of tensor `i`; returns false on load error.
bool dequant_prefix(const rdna4::GgufLoader &ld, std::size_t i, std::uint32_t max_blocks,
                    std::vector<std::uint8_t> &raw, std::vector<float> &f32,
                    std::string &err) {
  const auto dt = ld.dtype(i);
  const std::uint32_t be = rdna4::dtype_block_elems(dt);
  const std::uint32_t bb = rdna4::dtype_block_bytes(dt);
  const std::uint64_t total_blocks = ld.tensor_elems(i) / be;
  const std::uint64_t nblocks =
      std::min<std::uint64_t>(max_blocks, total_blocks ? total_blocks : 1);
  const std::uint64_t nbytes = nblocks * bb;
  if (!ld.load_tensor_range(i, 0, nbytes, raw, err)) {
    return false;
  }
  f32.resize(nblocks * be);
  dequant_row(dt, raw.data(), f32.data(), static_cast<std::int64_t>(nblocks * be));
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-dequant <file1.gguf> [file2.gguf]\n");
    return 2;
  }
  bool ok = true;

  // (1) Struct bridge: reference block structs == M1 block table.
  for (const auto &t : kTypes) {
    const std::size_t ss = struct_size_of(t.dt);
    const std::uint32_t bb = rdna4::dtype_block_bytes(t.dt);
    if (ss != bb) {
      std::fprintf(stderr, "struct bridge: %s sizeof(block)=%zu != M1 table %u\n", t.name, ss,
                   (unsigned)bb);
      ok = false;
    }
  }
  if (!ok) {
    std::fprintf(stderr, "check-dequant: FAILED (struct bridge)\n");
    return 1;
  }

  rdna4::GgufLoader ld1;
  std::string err;
  if (!ld1.open(argv[1], err)) {
    std::fprintf(stderr, "check-dequant: open %s failed: %s\n", argv[1], err.c_str());
    return 1;
  }
  const rdna4::gguf::File &f1 = ld1.meta();

  // (2)+(3) Per-type dequant of real tensors (first tensor of each type).
  for (const auto &t : kTypes) {
    std::size_t idx = std::size_t(-1);
    for (std::size_t i = 0; i < f1.tensors.size(); ++i) {
      if (ld1.dtype(i) == t.dt) {
        idx = i;
        break;
      }
    }
    if (idx == std::size_t(-1)) {
      std::printf("%-10s (absent in file)\n", t.name);
      continue;
    }
    std::vector<std::uint8_t> raw;
    std::vector<float> f32;
    if (!dequant_prefix(ld1, idx, 16, raw, f32, err)) {
      std::fprintf(stderr, "%-10s %s: load failed: %s\n", t.name, f1.tensors[idx].name.c_str(),
                   err.c_str());
      ok = false;
      continue;
    }
    const Stats s = check_stats(f32.data(), static_cast<std::int64_t>(f32.size()), 32.0f);
    const bool good = s.all_finite && s.in_range;
    std::printf("%-10s %-40s %5llu elems  min=%+.4g max=%+.4g mean=%+.4g  %s\n", t.name,
                f1.tensors[idx].name.c_str(), (unsigned long long)f32.size(), s.vmin, s.vmax,
                s.mean, good ? "OK" : "BAD");
    ok = ok && good;
    // (3) Q8_0: bit-exact vs inline d*qs[i].
    if (t.dt == DType::Q8_0) {
      std::vector<float> ref(f32.size());
      std::size_t off = 0;
      for (std::int64_t i = 0; i < static_cast<std::int64_t>(f32.size()); i += 32) {
        const std::uint16_t d16 =
            static_cast<std::uint16_t>(raw[off] | (raw[off + 1] << 8));
        off += 2;
        const float d = ggml_fp16_to_fp32(static_cast<ggml_fp16_t>(d16));
        for (int j = 0; j < 32; ++j) {
          ref[i + j] = d * static_cast<float>(static_cast<std::int8_t>(raw[off + j]));
        }
        off += 32;
      }
      const bool exact = std::memcmp(ref.data(), f32.data(), f32.size() * sizeof(float)) == 0;
      std::printf("q8_0 exact pipeline (inline d*qs vs oracle): %s\n", exact ? "OK" : "BAD");
      ok = ok && exact;
    }
  }

  // (4) Cross-file: same weights, two quantizations -> small rel-L2.
  if (argc >= 3) {
    rdna4::GgufLoader ld2;
    if (!ld2.open(argv[2], err)) {
      std::fprintf(stderr, "check-dequant: open %s failed: %s\n", argv[2], err.c_str());
      return 1;
    }
    const rdna4::gguf::File &f2 = ld2.meta();
    const char *pairs[] = {"blk.3.attn_q.weight", "blk.0.ffn_down.weight", "output.weight"};
    constexpr std::uint64_t kMaxElems = 8 * 1024 * 1024;  // ~32 MB f32 per side
    for (const char *name : pairs) {
      const auto *t1 = ld1.find(name);
      const auto *t2 = ld2.find(name);
      if (!t1 || !t2) {
        std::fprintf(stderr, "cross-file: %s missing in one of the files\n", name);
        ok = false;
        continue;
      }
      if (t1->dims != t2->dims) {
        std::fprintf(stderr, "cross-file: %s dims differ\n", name);
        ok = false;
        continue;
      }
      const std::uint64_t rowlen = static_cast<std::uint64_t>(prod_dims(t1->dims, 1));
      const std::uint64_t rows = std::min<std::uint64_t>(512, kMaxElems / rowlen);
      const std::uint64_t nelem = rows * rowlen;
      const std::size_t i1 = static_cast<std::size_t>(t1 - f1.tensors.data());
      const std::size_t i2 = static_cast<std::size_t>(t2 - f2.tensors.data());
      const rdna4::DType dt1 = ld1.dtype(i1);
      const rdna4::DType dt2 = ld2.dtype(i2);
      const std::uint64_t nb1 =
          nelem * rdna4::dtype_block_bytes(dt1) / rdna4::dtype_block_elems(dt1);
      const std::uint64_t nb2 =
          nelem * rdna4::dtype_block_bytes(dt2) / rdna4::dtype_block_elems(dt2);
      std::vector<std::uint8_t> raw1, raw2;
      std::vector<float> a, b;
      if (!ld1.load_tensor_range(i1, 0, nb1, raw1, err) ||
          !ld2.load_tensor_range(i2, 0, nb2, raw2, err)) {
        std::fprintf(stderr, "cross-file: %s load failed: %s\n", name, err.c_str());
        ok = false;
        continue;
      }
      a.resize(nelem);
      b.resize(nelem);
      dequant_row(dt1, raw1.data(), a.data(), static_cast<std::int64_t>(nelem));
      dequant_row(dt2, raw2.data(), b.data(), static_cast<std::int64_t>(nelem));
      double num = 0.0, da = 0.0, db = 0.0;
      for (std::uint64_t i = 0; i < nelem; ++i) {
        const double d = a[i] - b[i];
        num += d * d;
        da += static_cast<double>(a[i]) * a[i];
        db += static_cast<double>(b[i]) * b[i];
      }
      const double rel = std::sqrt(num / (da + db));
      const bool good = rel < 0.3;
      std::printf("cross-file %-40s rel-L2=%6.4f  %s\n", name, rel, good ? "OK" : "BAD");
      ok = ok && good;
    }
  }

  if (!ok) {
    std::fprintf(stderr, "check-dequant: FAILED\n");
    return 1;
  }
  std::printf("check-dequant: OK\n");
  return 0;
}
