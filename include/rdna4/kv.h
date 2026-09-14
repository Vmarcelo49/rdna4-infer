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
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "rdna4/fp16.h"
#include "rdna4/quants.h"

// ---------------------------------------------------------------------------
// The two 32-element blocks this file adds on top of quants.h (llama.cpp's
// ggml-common.h, MIT: `block_q5_0` = half d + uint8_t qh[4] + uint8_t qs[16]
// = 22 B / 32 elems, and `block_q4_1` = half d + half m + uint8_t qs[16]
// = 20 B / 32). Declared here, not in quants.h, because the engine never stores
// them as tensor types -- only the KV cache uses them -- and the #ifndef guards
// keep a future move to quants.h from redefining anything.
// ---------------------------------------------------------------------------
#ifndef QK5_0
#define QK5_0 32
#endif
#ifndef QK4_1
#define QK4_1 32
#endif

namespace rdna4 {

// ---------------------------------------------------------------------------
// Adversarial-review finding F6 (docs/adversarial-noite.md): the KV tables used
// to answer "I do not know this type" with a *silent* 0 -- `kv_row_bytes`
// returned 0 bytes per row, so every row of a cache landed on the same address
// and the engine ran, wrote, and returned wrong numbers without a word;
// `kv_store_row_kernel`/`kv_fill_kernel` fell through to their q4_0 body for any
// type that was not F32/F16/Q8_0 (the trailing branch was an implicit `else`);
// and `kv_bytes_per_elem` returned 0.0, which made `info`/`serve` approve a
// configuration with NO KV bytes counted.
//
// The fix has two halves, and both are needed:
//   * every switch stays EXHAUSTIVE WITH NO `default:` so `-Wswitch` still fires
//     the moment a type is added (a `default:` would silence the compiler, which
//     is how this class of bug survives);
//   * the code after each switch is kv_unreachable(), which aborts on the host
//     and traps on the device instead of degrading into a plausible answer.
// tests/check_kvtype.hip is the regression gate: it forks a child, feeds it
// KvType(99), and requires it to die on SIGABRT for each of those three tables.
// ---------------------------------------------------------------------------
__host__ __device__ __forceinline__ void kv_unreachable(const char *what) {
#if defined(__HIP_DEVICE_COMPILE__)
  (void)what;
  __builtin_trap();
#else
  std::fprintf(stderr,
               "rdna4: %s: unknown KvType -- the value is outside the enum and the KV "
               "path refuses to guess (see include/rdna4/kv.h)\n",
               what);
  std::abort();
#endif
}

// llama.cpp layout (ggml-common.h): the 5th bit plane `qh` lives *between* the
// scale and the nibbles, so the struct is not 4-byte aligned and a 32-bit read
// of qh would be a misaligned access on the GPU -- the load paths below read it
// byte-wise on purpose.
typedef struct {
  std::uint16_t d;             // delta (fp16)
  std::uint8_t qh[QK5_0 / 8];  // 5th bit of the quants, one bit per element
  std::uint8_t qs[QK5_0 / 2];  // nibbles: element j low, element j+16 high
} block_q5_0;
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 must be 22 bytes");

typedef struct {
  std::uint16_t d;             // delta (fp16)
  std::uint16_t m;             // min   (fp16)
  std::uint8_t qs[QK4_1 / 2];  // nibbles: element j low, element j+16 high
} block_q4_1;
static_assert(sizeof(block_q4_1) == 20, "block_q4_1 must be 20 bytes");

enum class KvType : std::uint32_t {
  F32 = 0,
  F16 = 1,   // llama.cpp's default
  Q8_0 = 2,  // 8.5 bpw
  Q4_0 = 3,  // 4.5 bpw
  // Appended (never renumbered): these were the night's hypothesis for 131K --
  // 5 bits with no min for K, 4 bits + a per-block min for V.
  Q5_0 = 4,  // 5.5 bpw, symmetric-ish (d from the signed max, grid [-16, 16])
  Q4_1 = 5,  // 4.5 bpw + fp16 min per block (asymmetric: d = (max-min)/15)
};

inline const char *kv_type_name(KvType t) {
  switch (t) {
    case KvType::F32: return "f32";
    case KvType::F16: return "f16";
    case KvType::Q8_0: return "q8_0";
    case KvType::Q4_0: return "q4_0";
    case KvType::Q5_0: return "q5_0";
    case KvType::Q4_1: return "q4_1";
  }
  // Not an abort: this one only feeds printed output, and dying while trying to
  // report a bad type is worse than printing a visibly wrong name.
  return "<unknown-kv-type>";
}

// nullptr for an unknown name (the caller reports the error).
// Namespace-scope, not a function-local static: this header is included by HIP
// translation units and a function-local static here is referenced from device code,
// which breaks the link of the shared library with "relocation R_X86_64_PC32 cannot
// be used against symbol ... -- recompile with -fPIC". It linked until the new types
// changed which TU instantiates it; the same trap bit the RD_ATTN_WPB knob earlier
// tonight, so the rule is now: no function-local statics in device-visible headers.
struct KvTypeName {
  const char *name;
  KvType t;
};
inline constexpr KvTypeName kKvTypeNames[] = {
    {"f32", KvType::F32},   {"f16", KvType::F16},   {"q8_0", KvType::Q8_0},
    {"q4_0", KvType::Q4_0}, {"q5_0", KvType::Q5_0}, {"q4_1", KvType::Q4_1}};

inline const char *kv_type_parse(const char *name, KvType *out) {
  for (const auto &e : kKvTypeNames) {
    if (std::strcmp(e.name, name) == 0) {
      *out = e.t;
      return nullptr;
    }
  }
  return "kv type must be one of f32, f16, q8_0, q4_0, q5_0, q4_1";
}

inline double kv_bytes_per_elem(KvType t) {
  switch (t) {
    case KvType::F32: return 4.0;
    case KvType::F16: return 2.0;
    case KvType::Q8_0: return 34.0 / 32.0;
    case KvType::Q4_0: return 18.0 / 32.0;
    case KvType::Q5_0: return 22.0 / 32.0;  // 0.6875 B/elem
    case KvType::Q4_1: return 20.0 / 32.0;  // 0.625  B/elem
  }
  // A silent 0.0 here is what let `info`/`serve` approve a context with no KV
  // counted at all (F6).
  kv_unreachable("kv_bytes_per_elem");
  return 0.0;  // not reached: kv_unreachable() aborts/traps
}

// Bytes of one row (head_dim elements).
__host__ __device__ __forceinline__ std::uint64_t kv_row_bytes(KvType t, int head_dim) {
  const std::uint64_t blocks = (std::uint64_t)(head_dim + 31) / 32;
  switch (t) {
    case KvType::F32: return (std::uint64_t)head_dim * 4;
    case KvType::F16: return (std::uint64_t)head_dim * 2;
    case KvType::Q8_0: return blocks * sizeof(block_q8_0);
    case KvType::Q4_0: return blocks * sizeof(block_q4_0);
    case KvType::Q5_0: return blocks * sizeof(block_q5_0);
    case KvType::Q4_1: return blocks * sizeof(block_q4_1);
  }
  // A silent 0 here is the worst of the three: the row stride becomes 0 and the
  // whole cache collapses onto one address (F6).
  kv_unreachable("kv_row_bytes");
  return 0;  // not reached: kv_unreachable() aborts/traps
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

// q5_0: same nibble packing as q4_0 plus a 5th bit plane in `qh`. Element e's
// 5th bit is bit e of the little-endian uint32 built from qh[0..3], so it is
// read here as bit (e&7) of qh[e>>3] -- a byte access, never a 32-bit one, since
// `qh` sits at offset 2 inside a 22-byte struct. The value is
// d*((low4 | (hi<<4)) - 16), i.e. the signed max lands on -16.
template <>
__device__ __forceinline__ float kv_load<KvType::Q5_0>(const void *row, int d) {
  const block_q5_0 *b = (const block_q5_0 *)row + (d >> 5);
  const int j = d & 31;
  const std::uint8_t byte = b->qs[j & 15];
  const int nib = (j < 16) ? (byte & 0x0F) : (byte >> 4);
  const int hi = (b->qh[j >> 3] >> (j & 7)) & 1;
  return fp16_to_float(b->d) * (float)((nib | (hi << 4)) - 16);
}

// q4_1: 4 bits plus a per-block fp16 min. dequantize_row_q4_1 (ggml-quants.c:479)
// is `q*d + m`, so this is the one storage type whose reconstruction is affine
// with a non-zero offset -- which is exactly why it can represent an asymmetric
// (e.g. all-positive) V block that q4_0/q5_0 lose half their range on.
template <>
__device__ __forceinline__ float kv_load<KvType::Q4_1>(const void *row, int d) {
  const block_q4_1 *b = (const block_q4_1 *)row + (d >> 5);
  const int j = d & 31;
  const std::uint8_t byte = b->qs[j & 15];
  const int q = (j < 16) ? (byte & 0x0F) : (byte >> 4);
  return fp16_to_float(b->d) * (float)q + fp16_to_float(b->m);
}

// ---------------------------------------------------------------------------
// Device: load 8 CONSECUTIVE dims for one lane in a single (vectorized,
// coalesced) access.
//
// The attention kernel gives lane L the dims [L*8, L*8+8). Loading those with the
// scalar kv_load() above costs 8 separate 2-byte (F16) or 1-byte (Q4_0) loads per
// lane per row, and across a warp those addresses are strided by 8 elements — so
// every load instruction fans out over several cache lines. Measured effect
// (tests/bench_attn_gpu.hip): the kernel moved the cache at **27 GB/s** (f16) and
// 5.5 GB/s (q4_0) at 64K, i.e. ~20x below DRAM, purely on load inefficiency. With
// the chunk loaded as one 16/8-byte access the warp's 32 lanes cover exactly one
// 512-byte row contiguously, which is what the hardware wants.
//
// The returned values are identical to 8 kv_load() calls, in the same order, so
// every kernel that switches to this keeps its arithmetic exactly.
// ---------------------------------------------------------------------------
template <KvType CT>
__device__ __forceinline__ void kv_load8(const void *row, int lane, float out[8]);

template <>
__device__ __forceinline__ void kv_load8<KvType::F32>(const void *row, int lane, float out[8]) {
  // 8 dims = 32 bytes = two 16-byte chunks; lane L covers bytes [L*32, L*32+32)
  const float *p = (const float *)row + lane * 8;
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = p[i];
}

template <>
__device__ __forceinline__ void kv_load8<KvType::F16>(const void *row, int lane, float out[8]) {
  // lane L covers 16 bytes = 8 halves, i.e. exactly one uint4
  const uint4 v = ((const uint4 *)row)[lane];
  const std::uint16_t h[8] = {(std::uint16_t)(v.x & 0xFFFF), (std::uint16_t)(v.x >> 16),
                              (std::uint16_t)(v.y & 0xFFFF), (std::uint16_t)(v.y >> 16),
                              (std::uint16_t)(v.z & 0xFFFF), (std::uint16_t)(v.z >> 16),
                              (std::uint16_t)(v.w & 0xFFFF), (std::uint16_t)(v.w >> 16)};
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = __half2float(__ushort_as_half(h[i]));
}

template <>
__device__ __forceinline__ void kv_load8<KvType::Q8_0>(const void *row, int lane, float out[8]) {
  // 8 dims sit inside one 32-element block: qs[(lane%4)*8 .. +8), scale shared
  const block_q8_0 *b = (const block_q8_0 *)row + (lane >> 2);
  const std::uint64_t q = *(const std::uint64_t *)(b->qs + (lane & 3) * 8);
  const float d = fp16_to_float(b->d);
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = d * (float)(std::int8_t)((q >> (8 * i)) & 0xFF);
}

template <>
__device__ __forceinline__ void kv_load8<KvType::Q4_0>(const void *row, int lane, float out[8]) {
  // lanes 0-3 cover one 32-element block: 8 low nibbles (lanes 0,1) or 8 high
  // nibbles (lanes 2,3) of the same 8 qs bytes
  // lane L covers dims [L*8, L*8+8): L&1 selects the low-nibble half (dims 0-15)
  // or the high-nibble half (dims 16-31) of the same 16 qs bytes, L&2 the half
  const block_q4_0 *b = (const block_q4_0 *)row + (lane >> 2);
  const std::uint64_t q = *(const std::uint64_t *)(b->qs + (lane & 1) * 8);
  const bool high = (lane & 2) != 0;
  const float d = fp16_to_float(b->d);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int byte = (int)((q >> (8 * i)) & 0xFF);
    const int nib = high ? (byte >> 4) : (byte & 0x0F);
    out[i] = d * (float)(nib - 8);
  }
}

template <>
__device__ __forceinline__ void kv_load8<KvType::Q5_0>(const void *row, int lane, float out[8]) {
  // Same nibble addressing as Q4_0 above (lane&1 picks the qs half, lane&2 the
  // nibble half); the extra 5th bit is simpler than in the scalar path: the 8
  // dims of lane L are bits [8*(lane&3), 8*(lane&3)+8) of the qh plane, which is
  // exactly qh[lane&3] -- so one byte load, no shift gymnastics, and the whole
  // warp still touches each 22-byte block from 4 lanes only.
  const block_q5_0 *b = (const block_q5_0 *)row + (lane >> 2);
  const std::uint64_t q = *(const std::uint64_t *)(b->qs + (lane & 1) * 8);
  const std::uint8_t hb = b->qh[lane & 3];
  const bool high = (lane & 2) != 0;
  const float d = fp16_to_float(b->d);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int byte = (int)((q >> (8 * i)) & 0xFF);
    const int nib = high ? (byte >> 4) : (byte & 0x0F);
    const int hi = (hb >> i) & 1;
    out[i] = d * (float)((nib | (hi << 4)) - 16);
  }
}

template <>
__device__ __forceinline__ void kv_load8<KvType::Q4_1>(const void *row, int lane, float out[8]) {
  // Identical nibble addressing to Q4_0; the only difference is the affine
  // reconstruction q*d + m, with both d and m read once per lane.
  const block_q4_1 *b = (const block_q4_1 *)row + (lane >> 2);
  const std::uint64_t q = *(const std::uint64_t *)(b->qs + (lane & 1) * 8);
  const bool high = (lane & 2) != 0;
  const float d = fp16_to_float(b->d);
  const float m = fp16_to_float(b->m);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int byte = (int)((q >> (8 * i)) & 0xFF);
    const int nib = high ? (byte >> 4) : (byte & 0x0F);
    out[i] = d * (float)nib + m;
  }
}

// ---------------------------------------------------------------------------
// Device: quantize one f32 row (head_dim elements) into the cache layout.
// Q8_0 and Q4_0 reproduce ggml's reference quantizers byte for byte
// (quantize_row_q8_0_ref: amax/127 + roundf + an fp16 scale; quantize_row_q4_0_ref:
// the signed max, d = max/-8, the (int8_t)(x*id + 8.5) rounding and the nibble
// split). Q5_0 and Q4_1 likewise transcribe quantize_row_q5_0_ref
// (ggml-quants.c:187) and quantize_row_q4_1_ref (ggml-quants.c:150):
//   Q5_0: amax/signed-max -> d = max/-16, (int8_t)(x*id + 16.5) clamped to [0,31],
//         low 4 bits into qs, bit 4 into the `qh` plane at index j resp. j+16;
//   Q4_1: min/max -> d = (max-min)/15, m = min, (int8_t)(x*id + 0.5) clamped to
//         [0,15] with x = (x[j]-min)*id.
// check-rope-gpu checks the stored rows against a host mirror of those formulas
// for every type, and tests/check_kvquant_gpu.hip checks them against llama.cpp's
// own quantize_row_q5_0_ref / quantize_row_q4_1_ref out of libggml-base.
// One thread per block of 32 (Q4_0/Q5_0/Q4_1/Q8_0) or per element (F16/F32).
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
  if (CT == KvType::Q5_0) {
    block_q5_0 *y = (block_q5_0 *)dst + blk;
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < 32; ++j) {
      const float v = x[j];
      if (amax < fabsf(v)) {
        amax = fabsf(v);
        vmax = v;
      }
    }
    const float d = vmax / -16.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y->d = float_to_fp16(d);
    std::uint32_t qh = 0;
    for (int j = 0; j < 16; ++j) {
      const float x0 = x[j] * id;
      const float x1 = x[j + 16] * id;
      const int q0 = (int)(std::int8_t)(x0 + 16.5f);
      const int q1 = (int)(std::int8_t)(x1 + 16.5f);
      const std::uint8_t xi0 = (std::uint8_t)(q0 > 31 ? 31 : (q0 < 0 ? 0 : q0));
      const std::uint8_t xi1 = (std::uint8_t)(q1 > 31 ? 31 : (q1 < 0 ? 0 : q1));
      y->qs[j] = (std::uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
      qh |= (std::uint32_t)((xi0 & 0x10u) >> 4) << (j + 0);
      qh |= (std::uint32_t)((xi1 & 0x10u) >> 4) << (j + 16);
    }
    // little-endian bytes of qh, written out instead of memcpy: std::memcpy is a
    // __host__ function here, and this also removes the (mis)alignment question
    // that a 32-bit store into a uint8_t[4] at offset 2 would raise.
    y->qh[0] = (std::uint8_t)(qh & 0xFFu);
    y->qh[1] = (std::uint8_t)((qh >> 8) & 0xFFu);
    y->qh[2] = (std::uint8_t)((qh >> 16) & 0xFFu);
    y->qh[3] = (std::uint8_t)((qh >> 24) & 0xFFu);
    return;
  }
  if (CT == KvType::Q4_1) {
    block_q4_1 *y = (block_q4_1 *)dst + blk;
    float vmin = x[0], vmax = x[0];
    for (int j = 1; j < 32; ++j) {
      vmin = fminf(vmin, x[j]);
      vmax = fmaxf(vmax, x[j]);
    }
    const float d = (vmax - vmin) / 15.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    y->d = float_to_fp16(d);
    y->m = float_to_fp16(vmin);
    for (int j = 0; j < 16; ++j) {
      const float x0 = (x[j] - vmin) * id;
      const float x1 = (x[j + 16] - vmin) * id;
      const int q0 = (int)(std::int8_t)(x0 + 0.5f);
      const int q1 = (int)(std::int8_t)(x1 + 0.5f);
      const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
      const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
      y->qs[j] = (std::uint8_t)(xi0 | (xi1 << 4));
    }
    return;
  }
  // Q4_0 — transcribe of ggml's quantize_row_q4_0_ref (ggml-quants.c): the scale
  // is built from the *signed* value with the largest magnitude,
  // `d = max / -8`, so the grid is [-8, 8] with d carrying the sign. (An
  // amax/7 symmetric grid looks equivalent but stores different bytes.)
  //
  // This used to be the *implicit* tail of the if-chain, so any CT that matched
  // none of the branches above would silently be stored as q4_0 (F6). It is an
  // explicit branch now and the chain ends in kv_unreachable().
  if (CT == KvType::Q4_0) {
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
  return;
  }
  kv_unreachable("kv_store_row_kernel");
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
  // Q5_0 and Q4_1 are block-serial (lane 0 quantizes all 32 elements). Their
  // scale/min depend on the whole block, and the Q5_0 signed max with its exact
  // first-wins tie-break is not reproducible from a warp reduction without
  // carrying the value alongside the magnitude -- the store kernel does it
  // serially too, so the fill stays byte-identical to a store of the same row
  // (which is what check-kvctx-gpu's synthetic cache relies on). The lanes that
  // are not 0 still call kv_rand above so the kernel's divergence is uniform.
  if (CT == KvType::Q5_0 || CT == KvType::Q4_1) {
    if (lane != 0) return;
    float v[32];
    for (int j = 0; j < 32; ++j) {
      v[j] = kv_rand((unsigned)(row * head_dim + blk * 32 + j), seed);
    }
    if (CT == KvType::Q5_0) {
      block_q5_0 *b = (block_q5_0 *)dst + blk;
      float amax = 0.0f, vmax = 0.0f;
      for (int j = 0; j < 32; ++j) {
        if (amax < fabsf(v[j])) {
          amax = fabsf(v[j]);
          vmax = v[j];
        }
      }
      const float d = vmax / -16.0f;
      const float id = d != 0.0f ? 1.0f / d : 0.0f;
      b->d = float_to_fp16(d);
      std::uint32_t qh = 0;
      for (int j = 0; j < 16; ++j) {
        const int q0 = (int)(std::int8_t)(v[j] * id + 16.5f);
        const int q1 = (int)(std::int8_t)(v[j + 16] * id + 16.5f);
        const std::uint8_t xi0 = (std::uint8_t)(q0 > 31 ? 31 : (q0 < 0 ? 0 : q0));
        const std::uint8_t xi1 = (std::uint8_t)(q1 > 31 ? 31 : (q1 < 0 ? 0 : q1));
        b->qs[j] = (std::uint8_t)((xi0 & 0x0F) | ((xi1 & 0x0F) << 4));
        qh |= (std::uint32_t)((xi0 & 0x10u) >> 4) << (j + 0);
        qh |= (std::uint32_t)((xi1 & 0x10u) >> 4) << (j + 16);
      }
      b->qh[0] = (std::uint8_t)(qh & 0xFFu);
      b->qh[1] = (std::uint8_t)((qh >> 8) & 0xFFu);
      b->qh[2] = (std::uint8_t)((qh >> 16) & 0xFFu);
      b->qh[3] = (std::uint8_t)((qh >> 24) & 0xFFu);
      return;
    }
    block_q4_1 *b = (block_q4_1 *)dst + blk;
    float vmin = v[0], vmax = v[0];
    for (int j = 1; j < 32; ++j) {
      vmin = fminf(vmin, v[j]);
      vmax = fmaxf(vmax, v[j]);
    }
    const float d = (vmax - vmin) / 15.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    b->d = float_to_fp16(d);
    b->m = float_to_fp16(vmin);
    for (int j = 0; j < 16; ++j) {
      const int q0 = (int)(std::int8_t)((v[j] - vmin) * id + 0.5f);
      const int q1 = (int)(std::int8_t)((v[j + 16] - vmin) * id + 0.5f);
      const std::uint8_t xi0 = (std::uint8_t)(q0 > 15 ? 15 : (q0 < 0 ? 0 : q0));
      const std::uint8_t xi1 = (std::uint8_t)(q1 > 15 ? 15 : (q1 < 0 ? 0 : q1));
      b->qs[j] = (std::uint8_t)(xi0 | (xi1 << 4));
    }
    return;
  }
  // Explicit q4_0 branch, and the chain ends in kv_unreachable(): this used to be
  // the implicit tail, i.e. any unknown type was filled as q4_0 (F6).
  if (CT == KvType::Q4_0) {
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
  return;
  }
  kv_unreachable("kv_fill_kernel");
}

inline bool kv_fill_launch(KvType t, void *d_cache, std::int64_t n_rows, int head_dim,
                           unsigned seed, hipStream_t stream = nullptr) {
  const int warps_per_block = 4;
  const std::int64_t blocks_per_row = (head_dim + 31) / 32;
  const std::int64_t total_warps = n_rows * blocks_per_row;
  const unsigned blocks = (unsigned)((total_warps + warps_per_block - 1) / warps_per_block);
  // Each case returns its OWN launch status, and an unknown type falls out of the
  // switch to `return false`. The tail used to be a single
  // `return hipGetLastError() == hipSuccess;` AFTER the switch, which reported the
  // status of whatever ran last -- for an unmatched type that is "no error", i.e.
  // the dispatcher said "filled OK" for a cache it never touched (found by
  // tests/check-kvtype.hip, the same F6 shape as the tables above).
  switch (t) {
    case KvType::F32:
      kv_fill_kernel<KvType::F32><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                              head_dim, seed);
      return hipGetLastError() == hipSuccess;
    case KvType::F16:
      kv_fill_kernel<KvType::F16><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                              head_dim, seed);
      return hipGetLastError() == hipSuccess;
    case KvType::Q8_0:
      kv_fill_kernel<KvType::Q8_0><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      return hipGetLastError() == hipSuccess;
    case KvType::Q4_0:
      kv_fill_kernel<KvType::Q4_0><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      return hipGetLastError() == hipSuccess;
    case KvType::Q5_0:
      kv_fill_kernel<KvType::Q5_0><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      return hipGetLastError() == hipSuccess;
    case KvType::Q4_1:
      kv_fill_kernel<KvType::Q4_1><<<blocks, warps_per_block * 32, 0, stream>>>(d_cache, n_rows,
                                                                               head_dim, seed);
      return hipGetLastError() == hipSuccess;
  }
  return false;  // unknown type: refuse, do not report the previous launch's status
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
    case KvType::Q5_0: return kv_store_row_launch<KvType::Q5_0>(d_src, d_dst, head_dim, stream);
    case KvType::Q4_1: return kv_store_row_launch<KvType::Q4_1>(d_src, d_dst, head_dim, stream);
  }
  return false;
}

}  // namespace rdna4
