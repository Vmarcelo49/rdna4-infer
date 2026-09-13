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
