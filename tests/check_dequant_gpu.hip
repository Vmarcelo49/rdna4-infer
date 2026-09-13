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

// --- device kernels -------------------------------------------------------

template <class Fn>
__global__ void dequant_kernel_256(const void *vx, float *yy, int64_t nblocks) {
  for (int64_t i = blockIdx.x; i < nblocks; i += gridDim.x) {
    Fn::apply(vx, i, yy + i * 256, threadIdx.x);
  }
}

__global__ void dequant_kernel_q8_0(const void *vx, float *yy, int64_t nblocks) {
  for (int64_t i = blockIdx.x; i < nblocks; i += gridDim.x) {
    float2 v;
    rdna4::dequantize_q8_0(vx, i, 2 * (int)threadIdx.x, v);
    yy[i * 32 + 2 * threadIdx.x + 0] = v.x;
    yy[i * 32 + 2 * threadIdx.x + 1] = v.y;
  }
}

#define FN(Name, FnName)                                                                        \
  struct Name {                                                                                 \
    static __device__ __forceinline__ void apply(const void *vx, int64_t i, float *y, int tid) { \
      rdna4::FnName(vx, i, y, tid);                                                             \
    }                                                                                           \
  }

FN(FnQ2K, dequantize_q2_K);
FN(FnQ3K, dequantize_q3_K);
FN(FnQ4K, dequantize_q4_K);
FN(FnQ5K, dequantize_q5_K);
FN(FnQ6K, dequantize_q6_K);
FN(FnIQ2XXS, dequantize_iq2_xxs);
FN(FnIQ2XS, dequantize_iq2_xs);
FN(FnIQ3XXS, dequantize_iq3_xxs);
FN(FnIQ1S, dequantize_iq1_s);
FN(FnIQ4NL, dequantize_iq4_nl);
FN(FnIQ3S, dequantize_iq3_s);
FN(FnIQ2S, dequantize_iq2_s);
FN(FnIQ4XS, dequantize_iq4_xs);

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

int run_case(const char *label, DType dt, const std::uint8_t *src, std::int64_t nblocks,
             int nthreads, bool q8_variant) {
  const std::uint32_t be = rdna4::dtype_block_elems(dt);
  const std::uint32_t bb = rdna4::dtype_block_bytes(dt);
  const std::int64_t nelem = nblocks * (std::int64_t)be;
  const std::size_t nbytes = (std::size_t)nblocks * bb;

  std::vector<float> cpu(nelem);
  if (!rdna4_cpu_dequant_row((int)dt, src, cpu.data(), nelem)) {
    std::fprintf(stderr, "%-9s no CPU oracle\n", label);
    return 1;
  }

  void *d_src = nullptr;
  float *d_dst = nullptr;
  if (hipMalloc(&d_src, nbytes) != hipSuccess ||
      hipMalloc(&d_dst, (std::size_t)nelem * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "%-9s hipMalloc failed\n", label);
    return 1;
  }
  hipMemcpy(d_src, src, nbytes, hipMemcpyHostToDevice);

  const int grid = (int)(nblocks < 4096 ? nblocks : 4096);
  if (q8_variant) {
    dequant_kernel_q8_0<<<grid, 16>>>(d_src, d_dst, nblocks);
  } else {
    switch (dt) {
      case DType::Q2_K:    dequant_kernel_256<FnQ2K><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::Q3_K:    dequant_kernel_256<FnQ3K><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::Q4_K:    dequant_kernel_256<FnQ4K><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::Q5_K:    dequant_kernel_256<FnQ5K><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::Q6_K:    dequant_kernel_256<FnQ6K><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ2_XXS: dequant_kernel_256<FnIQ2XXS><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ2_XS:  dequant_kernel_256<FnIQ2XS><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ3_XXS: dequant_kernel_256<FnIQ3XXS><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ1_S:   dequant_kernel_256<FnIQ1S><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ4_NL:  dequant_kernel_256<FnIQ4NL><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ3_S:   dequant_kernel_256<FnIQ3S><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ2_S:   dequant_kernel_256<FnIQ2S><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      case DType::IQ4_XS:  dequant_kernel_256<FnIQ4XS><<<grid, nthreads>>>(d_src, d_dst, nblocks); break;
      default:
        std::fprintf(stderr, "%-9s unsupported\n", label);
        return 1;
    }
  }
  const hipError_t le = hipGetLastError();
  if (le != hipSuccess) {
    std::fprintf(stderr, "%-9s launch failed: %s\n", label, hipGetErrorString(le));
    hipFree(d_src);
    hipFree(d_dst);
    return 1;
  }
  std::vector<float> gpu(nelem);
  hipMemcpy(gpu.data(), d_dst, (std::size_t)nelem * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(d_src);
  hipFree(d_dst);

  const Stats s = compare(cpu.data(), gpu.data(), nelem);
  const bool bit_exact = (s.exact == s.nelem);
  const bool ok = bit_exact || s.rel_l2 < 1e-6;
  std::printf("%-9s %5lld blocks %8lld elems  exact=%7lld/%lld  max|d|=%.3e  rel-L2=%.3e  %s\n",
              label, (long long)nblocks, (long long)nelem, s.exact, s.nelem, s.max_abs, s.rel_l2,
              bit_exact ? "BIT-EXACT" : (ok ? "OK" : "MISMATCH"));
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
    const std::int64_t total = (std::int64_t)(ld.tensor_elems(idx) / be);
    const std::int64_t nb = total < nblocks_req ? total : nblocks_req;
    std::vector<std::uint8_t> raw;
    if (!ld.load_tensor_range(idx, 0, (std::uint64_t)nb * bb, raw, err)) {
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
