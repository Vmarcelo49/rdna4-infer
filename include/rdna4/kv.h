#pragma once
// M3 — KV cache storage types (F32 / F16 / Q8_0 / Q4_0), mirroring llama.cpp's
// `--cache-type-k/v`: the rows are *stored* quantized and the attention
// dequantizes on the fly, so the cache stays small at long context without a
// second copy of the weights.
//
// Layout per row (one head of one token): `head_dim` elements, quantized in
// 32-element blocks exactly like llama.cpp's `quantize_row_q8_0_ref` /
// `quantize_row_q4_0_ref` (ggml/src/ggml-quants.c), so a block is byte-identical
// to what llama.cpp would store. head_dim (256) is a multiple of 32 for
// qwen35, which makes `row_bytes` exact.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

#include "rdna4/fp16.h"
#include "rdna4/quants.h"

namespace rdna4 {

enum class KvType : std::uint32_t {
  F32 = 0,
  F16 = 1,   // llama.cpp's default
  Q8_0 = 2,  // 8.5 bpw
  Q4_0 = 3,  // 4.5 bpw
};

inline const char *kv_type_name(KvType t) {
  switch (t) {
    case KvType::F32: return "f32";
    case KvType::F16: return "f16";
    case KvType::Q8_0: return "q8_0";
    case KvType::Q4_0: return "q4_0";
  }
  return "?";
}

// nullptr for an unknown name (the caller reports the error).
inline const char *kv_type_parse(const char *name, KvType *out) {
  static const struct {
    const char *name;
    KvType t;
  } kTable[] = {{"f32", KvType::F32}, {"f16", KvType::F16}, {"q8_0", KvType::Q8_0},
                {"q4_0", KvType::Q4_0}};
  for (const auto &e : kTable) {
    if (std::strcmp(e.name, name) == 0) {
      *out = e.t;
      return nullptr;
    }
  }
  return "kv type must be one of f32, f16, q8_0, q4_0";
}

inline double kv_bytes_per_elem(KvType t) {
  switch (t) {
    case KvType::F32: return 4.0;
    case KvType::F16: return 2.0;
    case KvType::Q8_0: return 34.0 / 32.0;
    case KvType::Q4_0: return 18.0 / 32.0;
  }
  return 0.0;
}

// Bytes of one row (head_dim elements).
__host__ __device__ __forceinline__ std::uint64_t kv_row_bytes(KvType t, int head_dim) {
  const std::uint64_t blocks = (std::uint64_t)(head_dim + 31) / 32;
  switch (t) {
    case KvType::F32: return (std::uint64_t)head_dim * 4;
    case KvType::F16: return (std::uint64_t)head_dim * 2;
    case KvType::Q8_0: return blocks * sizeof(block_q8_0);
    case KvType::Q4_0: return blocks * sizeof(block_q4_0);
  }
  return 0;
}

// ---------------------------------------------------------------------------
// Device: read element `d` of a stored row.
// ---------------------------------------------------------------------------
template <KvType CT>
__device__ __forceinline__ float kv_load(const void *row, int d);

template <>
__device__ __forceinline__ float kv_load<KvType::F32>(const void *row, int d) {
  return ((const float *)row)[d];
}

template <>
__device__ __forceinline__ float kv_load<KvType::F16>(const void *row, int d) {
  return __half2float(__ushort_as_half(((const std::uint16_t *)row)[d]));
}

template <>
__device__ __forceinline__ float kv_load<KvType::Q8_0>(const void *row, int d) {
  const block_q8_0 *b = (const block_q8_0 *)row + (d >> 5);
  return fp16_to_float(b->d) * (float)b->qs[d & 31];
}

template <>
__device__ __forceinline__ float kv_load<KvType::Q4_0>(const void *row, int d) {
  const block_q4_0 *b = (const block_q4_0 *)row + (d >> 5);
  const int j = d & 31;
  // llama.cpp packs element j in the low nibble and element j+16 in the high one
  const std::uint8_t byte = b->qs[j & 15];
  const int q = (j < 16) ? (byte & 0x0F) : (byte >> 4);
  return fp16_to_float(b->d) * (float)(q - 8);
}

// ---------------------------------------------------------------------------
// Device: quantize one f32 row (head_dim elements) into the cache layout.
// Q8_0 and Q4_0 reproduce ggml's reference quantizers byte for byte
// (quantize_row_q8_0_ref: amax/127 + roundf + an fp16 scale; quantize_row_q4_0_ref:
// the signed max, d = max/-8, the (int8_t)(x*id + 8.5) rounding and the nibble
// split). check-rope-gpu checks the stored rows against a host mirror of those
// formulas for every type.
// One thread per block of 32 (Q4_0/Q8_0) or per element (F16/F32).
// ---------------------------------------------------------------------------
template <KvType CT>
__global__ void kv_store_row_kernel(const float *__restrict__ src, void *__restrict__ dst,
                                   int head_dim) {
  const int blk = blockIdx.x * blockDim.x + threadIdx.x;
  if (CT == KvType::F32) {
    if (blk >= head_dim) return;
    ((float *)dst)[blk] = src[blk];
    return;
  }
  if (CT == KvType::F16) {
    if (blk >= head_dim) return;
    ((std::uint16_t *)dst)[blk] = float_to_fp16(src[blk]);
    return;
  }
  if (blk >= head_dim / 32) return;
  const float *x = src + (std::size_t)blk * 32;
  if (CT == KvType::Q8_0) {
    block_q8_0 *y = (block_q8_0 *)dst + blk;
    float amax = 0.0f;
    for (int j = 0; j < 32; ++j) amax = fmaxf(amax, fabsf(x[j]));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y->d = float_to_fp16(d);
    for (int j = 0; j < 32; ++j) y->qs[j] = (std::int8_t)roundf(x[j] * id);
    return;
  }
  // Q4_0 — transcribe of ggml's quantize_row_q4_0_ref (ggml-quants.c): the scale
  // is built from the *signed* value with the largest magnitude,
  // `d = max / -8`, so the grid is [-8, 8] with d carrying the sign. (An
  // amax/7 symmetric grid looks equivalent but stores different bytes.)
  block_q4_0 *y = (block_q4_0 *)dst + blk;
  float amax = 0.0f, vmax = 0.0f;
  for (int j = 0; j < 32; ++j) {
    const float v = x[j];
    if (amax < fabsf(v)) {
      amax = fabsf(v);
      vmax = v;
    }
  }
  const float d = vmax / -8.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  y->d = float_to_fp16(d);
  for (int j = 0; j < 16; ++j) {
    const float x0 = x[j] * id;
    const float x1 = x[j + 16] * id;
    const int q0 = (int)(std::int8_t)(x0 + 8.5f);  // (int8_t) cast, then MIN(15, ..)
    const int q1 = (int)(std::int8_t)(x1 + 8.5f);
    const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
    const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
    y->qs[j] = (std::uint8_t)(xi0 | (xi1 << 4));
  }
}

// ---------------------------------------------------------------------------
// Device: fill whole caches with a deterministic pseudo-random pattern (test
// hook — gives a long-context smoke test a realistic, non-degenerate cache
// without materialising an f32 copy of it). One warp per 32-element block.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float kv_rand(unsigned idx, unsigned seed) {
  unsigned h = idx * 2654435761u ^ seed * 2246822519u;
  h ^= h >> 13;
  h *= 1274126177u;
  return ((float)(h >> 8) / 8388608.0f) - 1.0f;  // ~(-1, 1)
}

template <KvType CT>
__global__ void kv_fill_kernel(void *__restrict__ cache, std::int64_t n_rows, int head_dim,
                               unsigned seed) {
  const int lane = threadIdx.x & 31;
  const std::int64_t wid = ((std::int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int blocks_per_row = head_dim / 32;
  const std::int64_t row = wid / blocks_per_row;
  const int blk = (int)(wid % blocks_per_row);
  if (row >= n_rows) return;
  const float x = kv_rand((unsigned)(row * head_dim + blk * 32 + lane), seed);
  char *dst = (char *)cache + row * (std::int64_t)kv_row_bytes(CT, head_dim);
  if (CT == KvType::F32) {
    ((float *)dst)[blk * 32 + lane] = x;
    return;
  }
  if (CT == KvType::F16) {
    ((std::uint16_t *)dst)[blk * 32 + lane] = float_to_fp16(x);
    return;
  }
  float amax = fabsf(x);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffull, amax, off));
  if (CT == KvType::Q8_0) {
    block_q8_0 *b = (block_q8_0 *)dst + blk;
    const float d = amax / 127.0f;  // amax is the warp-reduced |x| over the block
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    b->d = float_to_fp16(d);
    b->qs[lane] = (std::int8_t)roundf(x * id);
    return;
  }
  block_q4_0 *b = (block_q4_0 *)dst + blk;
  if (lane >= 16) return;  // one thread per byte: both nibbles are computed
                           // together, so there is no read-modify-write race
  // half the warp computed amax from its own element above; the pair value must
  // come from the same element that contributed it, so redo the signed max over
  // the two elements this byte covers.
  float a0 = kv_rand((unsigned)(row * head_dim + blk * 32 + lane), seed);
  float a1 = kv_rand((unsigned)(row * head_dim + blk * 32 + lane + 16), seed);
  const float amx = fmaxf(fabsf(a0), fabsf(a1));
  const float signed_max = (fabsf(a0) >= fabsf(a1)) ? a0 : a1;
  const float d = signed_max / -8.0f;
  const float id = d != 0.0f ? 1.0f / d : 0.0f;
  b->d = float_to_fp16(d);
  const int q0 = (int)(std::int8_t)(a0 * id + 8.5f);
  const int q1 = (int)(std::int8_t)(a1 * id + 8.5f);
  (void)amx;
  const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
  const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
  b->qs[lane] = (std::uint8_t)(xi0 | (xi1 << 4));
}

inline bool kv_fill_launch(KvType t, void *d_cache, std::int64_t n_rows, int head_dim,
                           unsigned seed, hipStream_t stream = nullptr) {
  const int warps_per_block = 4;
  const std::int64_t blocks_per_row = (head_dim + 31) / 32;
  const std::int64_t total_warps = n_rows * blocks_per_row;
  const unsigned blocks = (unsigned)((total_warps + warps_per_block - 1) / warps_per_block);
  switch (t) {
    case KvType::F32:
      kv_fill_kernel<KvType::F32><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                              head_dim, seed);
      break;
    case KvType::F16:
      kv_fill_kernel<KvType::F16><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                              head_dim, seed);
      break;
    case KvType::Q8_0:
      kv_fill_kernel<KvType::Q8_0><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      break;
    case KvType::Q4_0:
      kv_fill_kernel<KvType::Q4_0><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      break;
  }
  return hipGetLastError() == hipSuccess;
}

template <KvType CT>
inline bool kv_store_row_launch(const float *d_src, void *d_dst, int head_dim,
                                hipStream_t stream = nullptr) {
  const int units = (CT == KvType::F32 || CT == KvType::F16) ? head_dim : head_dim / 32;
  const int threads = 128;
  kv_store_row_kernel<CT><<<(units + threads - 1) / threads, threads, 0, stream>>>(d_src, d_dst,
                                                                                   head_dim);
  return hipGetLastError() == hipSuccess;
}

inline bool kv_store_row_launch(KvType t, const float *d_src, void *d_dst, int head_dim,
                               hipStream_t stream = nullptr) {
  switch (t) {
    case KvType::F32: return kv_store_row_launch<KvType::F32>(d_src, d_dst, head_dim, stream);
    case KvType::F16: return kv_store_row_launch<KvType::F16>(d_src, d_dst, head_dim, stream);
    case KvType::Q8_0: return kv_store_row_launch<KvType::Q8_0>(d_src, d_dst, head_dim, stream);
    case KvType::Q4_0: return kv_store_row_launch<KvType::Q4_0>(d_src, d_dst, head_dim, stream);
  }
  return false;
}

}  // namespace rdna4
