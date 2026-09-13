#pragma once
// Block struct layouts for the qwen35 quantization types (M2 GPU dequant).
// Extracted verbatim from llama.cpp's ggml-common.h (master @ df03399b8, MIT),
// rewritten to plain C++ types (no ggml macro/vendor dependencies).
// The field order matches the on-disk layout (validated by M1 step 2's
// empirically-verified dtype_block_bytes table).
//
// Device-side dequant (src/backend/dequant.cuh) casts a block pointer to
// the matching struct and reads the fields directly.
#include <stdint.h>

#define QK_K  256
#define QK8_0 32
#define K_SCALE_SIZE 12
#define IQ3S_N_SCALE 4
#define QK4_NL 32

namespace rdna4 {

struct half2 { uint16_t d; uint16_t dmin; };

#define QK8_1 32

// QI/QR: "quants per int" / "quants per row" per type (from ggml-common.h).
#define QI1_0 (QK1_0 / 32)
#define QR1_0 1
#define QI2_0 (QK2_0 / 32)
#define QR2_0 1
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2
#define QI4_1 (QK4_1 / (4 * QR4_1))
#define QR4_1 2
#define QI_MXFP4 (QK_MXFP4 / (4 * QR_MXFP4))
#define QR_MXFP4 2
#define QI_NVFP4 (QK_NVFP4 / (4 * QR_NVFP4))
#define QR_NVFP4 2
#define QI5_0 (QK5_0 / (4 * QR5_0))
#define QR5_0 2
#define QI5_1 (QK5_1 / (4 * QR5_1))
#define QR5_1 2
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1
#define QI2_K (QK_K / (4*QR2_K))
#define QR2_K 4
#define QI3_K (QK_K / (4*QR3_K))
#define QR3_K 4
#define QI4_K (QK_K / (4*QR4_K))
#define QR4_K 2
#define QI5_K (QK_K / (4*QR5_K))
#define QR5_K 2
#define QI6_K (QK_K / (4*QR6_K))
#define QR6_K 2
#define QI2_XXS (QK_K / (4*QR2_XXS))
#define QR2_XXS 4
#define QI2_XS (QK_K / (4*QR2_XS))
#define QR2_XS 4
#define QI2_S (QK_K / (4*QR2_S))
#define QR2_S 4
#define QI3_XXS (QK_K / (4*QR3_XXS))
#define QR3_XXS 4
#define QI3_XS (QK_K / (4*QR3_XS))
#define QR3_XS 4
#define QI1_S (QK_K / (4*QR1_S))
#define QR1_S 8
#define QI1_M (QK_K / (4*QR1_M))
#define QR1_M 8
#define QI4_NL (QK4_NL / (4*QR4_NL))
#define QR4_NL 2
#define QI4_XS (QK_K / (4*QR4_XS))
#define QR4_XS 2
#define QI3_S (QK_K / (4*QR3_S))
#define QR3_S 4

// 8-bit activation block (matvec input side): ds packs {d, s} as two fp16
// halves (low = d, high = s) exactly like llama.cpp's anonymous union.
struct block_q8_1 {
    uint32_t ds;          // low 16 bits = d (scale), high 16 = s (d*sum(qs))
    int8_t   qs[QK8_1];
};


// 4-bit (KV cache type in M3; not a tensor type in the UD files, so it lives
// here only for the cache — see include/rdna4/kv.h)
#define QK4_0 32
typedef struct {
    uint16_t d;           // delta
    uint8_t  qs[QK4_0/2]; // nibbles: element j in the low nibble, j+16 in the high
} block_q4_0;

// 8-bit
typedef struct {
    uint16_t d;       // delta
    int8_t   qs[QK8_0];
} block_q8_0;

// 2-bit (super-block)
typedef struct {
    uint8_t  scales[QK_K/16]; // scales and mins, 4-bit quantized
    uint8_t  qs[QK_K/4];      // quants
    half2    dm;
} block_q2_K;

// 3-bit
typedef struct {
    uint8_t  hmask[QK_K/8];   // high-bit of quants
    uint8_t  qs[QK_K/4];      // low 2 bits
    uint8_t  scales[12];      // scales, 6-bit quantized
    uint16_t d;
} block_q3_K;

// 4-bit
typedef struct {
    half2    dm;              // d (scales), dmin (mins)
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K/2];
} block_q4_K;

// 5-bit
typedef struct {
    half2    dm;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qh[QK_K/8];      // quants, high bit
    uint8_t  qs[QK_K/2];      // quants, low 4 bits
} block_q5_K;

// 6-bit
typedef struct {
    uint8_t  ql[QK_K/2];      // quants, lower 4 bits
    uint8_t  qh[QK_K/4];      // quants, upper 2 bits
    int8_t   scales[QK_K/16];
    uint16_t d;
} block_q6_K;

// 2.0625 bpw
typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
} block_iq2_xxs;

// 2.25 bpw
typedef struct {
    uint16_t d;
    uint16_t qs[QK_K/8];
    uint8_t  scales[QK_K/32];
} block_iq2_xs;

// 2.5 bpw
typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/32];
    uint8_t  scales[QK_K/32];
} block_iq2_s;

// 2.75 bpw
typedef struct {
    uint16_t d;
    uint8_t  qs[3*QK_K/8];
} block_iq3_xxs;

// 3 bpw
typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K/4];
    uint8_t  qh[QK_K/32];
    uint8_t  signs[QK_K/8];
    uint8_t  scales[IQ3S_N_SCALE];
} block_iq3_s;

// 1.5625 bpw
typedef struct {
    uint16_t d;
    uint8_t  qs[QK_K/8];
    uint16_t qh[QK_K/32];
} block_iq1_s;

// 4-bit nonlinear
typedef struct {
    uint16_t d;
    uint8_t  qs[QK4_NL/2];
} block_iq4_nl;

// 4-bit extra
typedef struct {
    uint16_t d;
    uint16_t scales_h;
    uint8_t  scales_l[QK_K/64];
    uint8_t  qs[QK_K/2];
} block_iq4_xs;

}  // namespace rdna4
