// M2 step 2/3 — GPU dequant vs CPU oracle, per quantization type.
//
// For each type present in the UD GGUF:
//   1. take the first tensor of that type (real bytes, real offsets);
//   2. CPU reference: llama.cpp's dequantize_row_<type> (libggml-base, MIT);
//   3. GPU: rdna4 device dequant (include/rdna4/dequant.cuh) on gfx1201;
//   4. compare element-wise (report bit-exact count + max abs diff).
//
// usage: check-dequant-gpu <file.gguf> [nblocks]
#include "rdna4/dequant.cuh"
#include "rdna4/dequant_row.cuh"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rdna4/dtype.h"
#include "rdna4/loader.h"

extern "C" bool rdna4_cpu_dequant_row(int dt, const void *src, float *dst, std::int64_t nelem);
extern "C" int rdna4_audit_layout(char *buf, int bufsize);

namespace {

using rdna4::DType;

struct TypeCase {
  DType dt;
  const char *name;
  int nthreads;  // threads per CTA (llama.cpp getrows.cu mapping)
};

// q8_0 is handled separately (float2 form, 16 threads = 32 elems/block).
const TypeCase kCases[] = {
    {DType::Q2_K, "q2_k", 64},
    {DType::Q3_K, "q3_k", 64},
    {DType::Q4_K, "q4_k", 32},
    {DType::Q5_K, "q5_k", 64},
    {DType::Q6_K, "q6_k", 64},
    {DType::IQ2_XXS, "iq2_xxs", 32},
    {DType::IQ2_XS, "iq2_xs", 32},
    {DType::IQ3_XXS, "iq3_xxs", 32},
    {DType::IQ1_S, "iq1_s", 32},
    {DType::IQ4_NL, "iq4_nl", 32},
    {DType::IQ3_S, "iq3_s", 32},
    {DType::IQ2_S, "iq2_s", 32},
    {DType::IQ4_XS, "iq4_xs", 32},
};

// (the per-type dequant kernels + the 14-type dispatch now live in
// include/rdna4/dequant_row.cuh, shared with the engine)
// --- host -----------------------------------------------------------------

struct Stats {
  double rel_l2 = 0;
  double max_abs = 0;
  long long exact = 0;
  long long nelem = 0;
};

Stats compare(const float *a, const float *b, std::int64_t n) {
  Stats s;
  s.nelem = n;
  double num = 0, da = 0, db = 0;
  for (std::int64_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    da += (double)a[i] * a[i];
    db += (double)b[i] * b[i];
    const double ad = std::fabs(d);
    if (ad > s.max_abs) s.max_abs = ad;
    if (a[i] == b[i]) ++s.exact;
  }
  s.rel_l2 = std::sqrt(num / (da + db));
  return s;
}

// Poison byte for the destination guard band: 0xAB is not a value any
// dequantizer produces, so "still 0xABABABAB" is a reliable "never written".
constexpr int kGuardByte = 0xAB;
inline bool guard_is(float v) {
  std::uint32_t bits = 0;
  std::memcpy(&bits, &v, sizeof(bits));
  return bits == 0xABABABABu;
}

int run_case(const char *label, DType dt, const std::uint8_t *src, std::int64_t nblocks,
             int nthreads, bool q8_variant) {
  const std::uint32_t be = rdna4::dtype_block_elems(dt);
  const std::uint32_t bb = rdna4::dtype_block_bytes(dt);
  // nblocks counts 256-element units for the q8_0=false path; the q8_0 kernel
  // uses 32-element blocks with 16 threads.
  const std::int64_t nelem = q8_variant ? nblocks * 32 : nblocks * 256;
  const std::size_t nbytes = (std::size_t)(nelem / be) * bb;

  std::vector<float> cpu(nelem);
  if (!rdna4_cpu_dequant_row((int)dt, src, cpu.data(), nelem)) {
    std::fprintf(stderr, "%-9s no CPU oracle\n", label);
    return 1;
  }

  void *d_src = nullptr;
  float *d_dst = nullptr;
  // Guard band: the destination is over-allocated and poisoned before the
  // launch, so a kernel that writes past `nelem` (a block count that is too
  // large for the dtype) trips the pattern check below instead of silently
  // corrupting VRAM while the compared prefix stays bit-exact — exactly how the
  // IQ4_NL 8x overrun slipped through (review M4, CRITICAL). The poison also
  // catches the opposite: elements the kernel never wrote.
  const std::size_t guard = 4096;
  const std::size_t dst_elems = (std::size_t)nelem + guard;
  if (hipMalloc(&d_src, nbytes) != hipSuccess ||
      hipMalloc(&d_dst, dst_elems * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "%-9s hipMalloc failed\n", label);
    return 1;
  }
  hipMemcpy(d_src, src, nbytes, hipMemcpyHostToDevice);
  if (hipMemset(d_dst, kGuardByte, dst_elems * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "%-9s hipMemset failed\n", label);
    hipFree(d_src);
    hipFree(d_dst);
    return 1;
  }

  if (!rdna4::dequant_row_launch(dt, d_src, d_dst, nelem)) {
    std::fprintf(stderr, "%-9s unsupported or launch failed\n", label);
    hipFree(d_src);
    hipFree(d_dst);
    return 1;
  }
  const hipError_t le = hipGetLastError();
  if (le != hipSuccess) {
    std::fprintf(stderr, "%-9s launch failed: %s\n", label, hipGetErrorString(le));
    hipFree(d_src);
    hipFree(d_dst);
    return 1;
  }
  std::vector<float> gpu(dst_elems);
  hipMemcpy(gpu.data(), d_dst, dst_elems * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_src);
  hipFree(d_dst);

  std::int64_t unwritten = 0, overflow = 0;
  for (std::size_t i = 0; i < dst_elems; ++i) {
    const bool poisoned = guard_is(gpu[i]);
    if (i < (std::size_t)nelem) {
      if (poisoned) ++unwritten;
    } else if (!poisoned) {
      ++overflow;
    }
  }

  const Stats s = compare(cpu.data(), gpu.data(), nelem);
  const bool bit_exact = (s.exact == s.nelem);
  const bool ok = (bit_exact || s.rel_l2 < 1e-6) && unwritten == 0 && overflow == 0;
  std::printf("%-9s %5lld blocks %8lld elems  exact=%7lld/%lld  max|d|=%.3e  rel-L2=%.3e  guard=%s  %s\n",
              label, (long long)nblocks, (long long)nelem, s.exact, s.nelem, s.max_abs, s.rel_l2,
              overflow ? "OVERFLOW" : (unwritten ? "GAPS" : "ok"),
              bit_exact ? "BIT-EXACT" : (ok ? "OK" : "MISMATCH"));
  if (overflow) {
    std::fprintf(stderr, "%-9s wrote %lld element(s) past the %lld-element destination\n", label,
                 (long long)overflow, (long long)nelem);
  }
  if (unwritten) {
    std::fprintf(stderr, "%-9s left %lld element(s) unwritten (the destination is poisoned)\n",
                 label, (long long)unwritten);
  }
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: check-dequant-gpu <file.gguf> [nblocks]\n");
    return 2;
  }
  const std::int64_t nblocks_req = argc >= 3 ? std::atoll(argv[2]) : 16;

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    std::fprintf(stderr, "no HIP device\n");
    return 1;
  }
  std::printf("device: %s  gcnArchName=%s  VRAM=%.2f GiB\n", prop.name, prop.gcnArchName,
              prop.totalGlobalMem / 1073741824.0);

  // Struct-layout audit: our block structs vs llama.cpp's originals
  // (catches same-size/reordered-field bugs, e.g. block_q5_K qh/qs).
  {
    char buf[4096];
    const int bad = rdna4_audit_layout(buf, (int)sizeof(buf));
    if (bad) {
      std::fprintf(stderr, "struct layout audit FAILED (%d mismatch(es)):\n%s", bad, buf);
      return 1;
    }
    std::printf("struct layout audit: OK (all offsets match llama.cpp)\n");
  }

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open %s: %s\n", argv[1], err.c_str());
    return 1;
  }

  int failures = 0;
  int tested = 0;

  // q8_0 first (separate kernel form, 32 elems/block).
  {
    std::size_t idx = std::size_t(-1);
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if (ld.dtype(i) == DType::Q8_0) {
        idx = i;
        break;
      }
    }
    if (idx != std::size_t(-1)) {
      const std::int64_t total = (std::int64_t)(ld.tensor_elems(idx) / rdna4::dtype_block_elems(DType::Q8_0));
      const std::int64_t nb = total < nblocks_req ? total : nblocks_req;
      std::vector<std::uint8_t> raw;
      if (ld.load_tensor_range(idx, 0, (std::uint64_t)nb * 34, raw, err)) {
        failures += run_case("q8_0", DType::Q8_0, raw.data(), nb, 16, true);
        ++tested;
      }
    }
  }

  for (const TypeCase &tc : kCases) {
    std::size_t idx = std::size_t(-1);
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if (ld.dtype(i) == tc.dt) {
        idx = i;
        break;
      }
    }
    if (idx == std::size_t(-1)) {
      std::printf("%-9s (absent in file)\n", tc.name);
      continue;
    }
    const std::uint32_t be = rdna4::dtype_block_elems(tc.dt);
    const std::uint32_t bb = rdna4::dtype_block_bytes(tc.dt);
    // dequant_kernel_256 processes ONE 256-element unit per iteration for every
    // type in kCases. For IQ4_NL that unit is 8 ggml blocks (32 elems each) --
    // sizing the buffer by the ggml block (as this test used to) makes the
    // kernel read/write 8x past the end of both allocations.
    constexpr std::int64_t kUnit = 256;
    const std::int64_t total_units = (std::int64_t)(ld.tensor_elems(idx) / kUnit);
    const std::int64_t nb = total_units < nblocks_req ? total_units : nblocks_req;
    const std::int64_t nelem = nb * kUnit;                       // floats the kernel writes
    const std::uint64_t nbytes = (std::uint64_t)(nelem / be) * bb;  // bytes it reads
    std::vector<std::uint8_t> raw;
    if (!ld.load_tensor_range(idx, 0, nbytes, raw, err)) {
      std::fprintf(stderr, "%-9s load failed: %s\n", tc.name, err.c_str());
      ++failures;
      continue;
    }
    failures += run_case(tc.name, tc.dt, raw.data(), nb, tc.nthreads, false);
    ++tested;
  }

  std::printf("check-dequant-gpu: %s (%d types tested)\n", failures ? "FAILED" : "OK", tested);
  return failures ? 1 : 0;
}
