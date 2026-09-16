// M2 step 3 — fused vec_dot (dequant + dot product) for the qwen35 types.
//
// Vendored from llama.cpp's ggml/src/ggml-cuda/vecdotq.cuh (master @ df03399b8,
// MIT), subset to the 14 quantized types of the M1 union, with mechanical
// adaptations only:
//   - block structs / IQ tables from rdna4/{quants,quant_tables}.h
//   - ggml_cuda_dp4a (RDNA4: __builtin_amdgcn_sudot4) vendored below
//   - __half2half/__half2float on our uint16_t/uint32_t fields -> rdna4::fp16_to_float
//   - QI*/QR* constants moved to quants.h
//
// Activation side is block_q8_1 (see quants.h); the matvec kernel quantizes the
// activations with quantize_q8_1_block() (matvec.cuh).
#pragma once
#include <hip/hip_runtime.h>

#include "rdna4/quants.h"
#include "rdna4/fp16.h"
#include "rdna4/quant_tables.h"

namespace rdna4 {


// --- vendored helpers (vecdotq.cuh L97-104 + ggml-cuda/vendors/hip.h, MIT) ---
static __device__ __forceinline__ uint32_t unpack_ksigns(const uint8_t v) {
    // v is a 7 bit int, with the 8th sign being encodable as popcnt
    const uint32_t p = __popc(v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}

typedef int8_t  int8x4_t  __attribute__((ext_vector_type(4)));
typedef uint8_t uint8x4_t __attribute__((ext_vector_type(4)));

static __device__ __forceinline__ int __vsubss4(const int a, const int b) {
    const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
    const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
#if __has_builtin(__builtin_elementwise_sub_sat)
    const int8x4_t c = __builtin_elementwise_sub_sat(va, vb);
    return reinterpret_cast<const int &>(c);
#else
    int8x4_t c;
    int16_t tmp;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        tmp = va[i] - vb[i];
        if (tmp > 127) tmp = 127;
        if (tmp < -128) tmp = -128;
        c[i] = tmp;
    }
    return reinterpret_cast<int &>(c);
#endif
}

static __device__ __forceinline__ int __vsub4(const int a, const int b) {
    return __vsubss4(a, b);
}

static __device__ __forceinline__ unsigned int __vcmpeq4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0xff : 0x00;
    }
    return c;
}

static __device__ __forceinline__ unsigned int __vcmpne4(unsigned int a, unsigned int b) {
    const uint8x4_t& va = reinterpret_cast<const uint8x4_t&>(a);
    const uint8x4_t& vb = reinterpret_cast<const uint8x4_t&>(b);
    unsigned int c;
    uint8x4_t& vc = reinterpret_cast<uint8x4_t&>(c);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        vc[i] = va[i] == vb[i] ? 0x00 : 0xff;
    }
    return c;
}

// ggml_cuda_dp4a: 4-way int8 dot product (from llama.cpp common.cuh, MIT).
// gfx1201 (RDNA4) uses the signed x unsigned dot instruction.
static __device__ __forceinline__ int ggml_cuda_dp4a(const int a, const int b, int c) {
#if defined(__HIP_DEVICE_COMPILE__)
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__)
    c = __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#else
    const signed char * a8 = (const signed char *) &a;
    const signed char * b8 = (const signed char *) &b;
    c += a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
#else
    const signed char * a8 = (const signed char *) &a;
    const signed char * b8 = (const signed char *) &b;
    c += a8[0]*b8[0] + a8[1]*b8[1] + a8[2]*b8[2] + a8[3]*b8[3];
#endif
    return c;
}

static __device__ __forceinline__ int get_int_b1(const void * x, const int & i32) {
    const uint8_t * x8 = (const uint8_t *) x;

    int x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    return x32;
}

static __device__ __forceinline__ int get_int_b2(const void * x, const int & i32) {
    const uint16_t * x16 = (const uint16_t *) x; // assume at least 2 byte alignment

    int x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    return x32;
}

static __device__ __forceinline__ int get_int_b4(const void * x, const int & i32) {
    return ((const int *) x)[i32]; // assume at least 4 byte alignment
}

// q4 contains 8 indices with 4 bit each.
// This function selects those bytes from table that are at those indices and returns them as int2.
// The first int contains the bytes with even indices in q4, the second int contains the bytes with odd indices in q4.
static __device__ __forceinline__ int2 get_int_from_table_16(const int & q4, const int8_t * table) {
    // HIP/gfx1201 only (this engine targets a single AMD arch): llama.cpp's
    // upstream body is guarded by #if defined(GGML_USE_HIP) with a generic
    // byte-lookup fallback. That macro is NOT defined in this build, so the
    // fallback was compiled and the ISA showed 72 ds_load_u8 per kernel with
    // the table spilled to LDS. Keeping only the perm path removes the trap.
    const uint32_t * values = (const uint32_t *) table;

    const uint32_t q_even = q4;
    const uint32_t q_odd  = (q4 >> 4);

    const uint32_t v_even_low  = __builtin_amdgcn_perm(values[1], values[0], q_even & 0x07070707);
    const uint32_t v_odd_low   = __builtin_amdgcn_perm(values[1], values[0], q_odd  & 0x07070707);
    const uint32_t v_even_high = __builtin_amdgcn_perm(values[3], values[2], q_even & 0x07070707);
    const uint32_t v_odd_high  = __builtin_amdgcn_perm(values[3], values[2], q_odd  & 0x07070707);

    const uint32_t mask_even = 0x03020100 | ((q_even & 0x08080808) >> 1);
    const uint32_t mask_odd  = 0x03020100 | ((q_odd  & 0x08080808) >> 1);

    return make_int2(__builtin_amdgcn_perm(v_even_high, v_even_low, mask_even),
                     __builtin_amdgcn_perm(v_odd_high,  v_odd_low,  mask_odd));
}

#define VDR_Q1_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism

#define VDR_Q2_0_Q8_1_MMVQ 1  // Process one 32-element chunk at a time for parallelism

#define VDR_Q4_0_Q8_1_MMVQ 2

#define VDR_Q4_1_Q8_1_MMVQ 2

#define VDR_Q5_0_Q8_1_MMVQ 2

#define VDR_Q5_1_Q8_1_MMVQ 2

#define VDR_Q8_0_Q8_1_MMVQ 2

#define VDR_MXFP4_Q8_1_MMVQ 2

#define VDR_NVFP4_Q8_1_MMVQ 4

#define VDR_Q2_K_Q8_1_MMVQ 1

#define VDR_Q3_K_Q8_1_MMVQ 1

#define VDR_Q4_K_Q8_1_MMVQ 2

#define VDR_Q5_K_Q8_1_MMVQ 2

#define VDR_Q6_K_Q8_1_MMVQ 1

#define VDR_IQ2_XXS_Q8_1_MMVQ 2

#define VDR_IQ2_XS_Q8_1_MMVQ 2

#define VDR_IQ2_S_Q8_1_MMVQ 2

#define VDR_IQ3_XXS_Q8_1_MMVQ 2

#define VDR_IQ3_S_Q8_1_MMVQ 2

#define VDR_IQ1_S_Q8_1_MMVQ 1

#define VDR_IQ1_M_Q8_1_MMVQ 1

#define VDR_IQ4_NL_Q8_1_MMVQ 2

#define VDR_IQ4_XS_Q8_1_MMVQ 4

template <typename T, int vdr> static __device__ __forceinline__ T vec_dot_q8_0_q8_1_impl(
    const int * v, const int * u, const T & d8_0, const T & d8_1) {

    int sumi = 0;

#pragma unroll
    for (int i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_cuda_dp4a(v[i], u[i], sumi);
    }

    return d8_0*d8_1 * ((T) sumi);
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1_impl_mmvq(
    const int & v, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const half2 & dm2, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR2_K; ++i) {
        const int sc = scales[2*i];

        const int vi = (v >> (2*i)) & 0x03030303;

        sumf_d += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        int m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] * ggml_cuda_dp4a(m, u[i], 0); // multiply constant q2_K part with sum of q8_1 values
    }

    const float2 dm2f = make_float2(rdna4::fp16_to_float(dm2.d), rdna4::fp16_to_float(dm2.dmin));

    return dm2f.x*sumf_d - dm2f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const uint8_t * __restrict__ scales,
    const int & scale_offset, const float & d3, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        const int isc = scale_offset + 2*i;

        const int isc_low = isc % (QK_K/32);
        const int sc_shift_low = 4 * (isc / (QK_K/32));
        const int sc_low  = (scales[isc_low] >> sc_shift_low) & 0xF;

        const int isc_high = isc % (QK_K/64);
        const int sc_shift_high = 2 * (isc / (QK_K/64));
        const int sc_high = ((scales[(QK_K/32) + isc_high] >> sc_shift_high) & 3) << 4;

        const int sc = (sc_low | sc_high) - 32;

        const int vil = (vl >> (2*i)) & 0x03030303;

        const int vih = ((vh >> i) << 2) & 0x04040404;

        // dp4a-linearity (bit-exact): vil bytes are 2-bit fields (0..3) and
        // vih bytes are 0/4, so vil-vih in [-4,3] never saturates int8 and
        //   dp4a(vil-vih, u) == dp4a(vil, u) - dp4a(vih, u)
        // holds exactly in int32 (ggml_cuda_dp4a accumulates without
        // saturation). Two single-instruction v_dot4_i32 replace the
        // multi-instruction per-byte saturating subtract (__vsubss4
        // emulation, 208 instr/call on this path). The float tail
        // (dot*sc, d8[i]*, sumf order) is untouched, hence bit-identical.
        const int dot = ggml_cuda_dp4a(vil, u[i], 0) - ggml_cuda_dp4a(vih, u[i], 0);

        sumf += d8[i] * (dot * sc); // SIMD dot product
    }

    return d3 * sumf;
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1_impl_vmmq(
    const int * __restrict__ v, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm4, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        const int dot1 = ggml_cuda_dp4a(v1i, u[2*i+1], ggml_cuda_dp4a(v0i, u[2*i+0], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+1], ggml_cuda_dp4a(0x01010101, u[2*i+0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    const float2 dm4f = make_float2(rdna4::fp16_to_float(dm4.d), rdna4::fp16_to_float(dm4.dmin));

    return dm4f.x*sumf_d - dm4f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1_impl_vmmq(
    const int * __restrict__ vl, const int * __restrict__ vh, const int * __restrict__ u, const uint8_t * __restrict__ sc,
    const uint8_t * __restrict__ m, const half2 & dm5, const float * __restrict__ d8) {

    float sumf_d = 0.0f;
    float sumf_m = 0.0f;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;

        const int dot1 = ggml_cuda_dp4a(v0i, u[2*i+0], ggml_cuda_dp4a(v1i, u[2*i+1], 0)); // SIMD dot product
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2*i+0], ggml_cuda_dp4a(0x01010101, u[2*i+1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    const float2 dm5f = make_float2(rdna4::fp16_to_float(dm5.d), rdna4::fp16_to_float(dm5.dmin));

    return dm5f.x*sumf_d - dm5f.y*sumf_m;
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1_impl_mmvq(
    const int & vl, const int & vh, const int * __restrict__ u, const int8_t * __restrict__ scales,
    const float & d, const float * __restrict__ d8) {

    float sumf = 0.0f;

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        const int sc = scales[4*i];

        const int vil = (vl >> (4*i)) & 0x0F0F0F0F;

        const int vih = ((vh >> (4*i)) << 4) & 0x30303030;

        const int vi = __vsubss4((vil | vih), 0x20202020); // vi = (vil | vih) - 32

        sumf += d8[i] * (ggml_cuda_dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    return d*sumf;
}

static __device__ __forceinline__ float vec_dot_q8_0_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q8_0 * bq8_0 = (const block_q8_0 *) vbq + kbx;

    int v[VDR_Q8_0_Q8_1_MMVQ];
    int u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    for (int i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_b2(bq8_0->qs, iqs + i);
        u[i] = get_int_b4(bq8_1->qs, iqs + i);
    }

    return vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMVQ>(v, u, rdna4::fp16_to_float(bq8_0->d), rdna4::fp16_to_float((uint16_t)(bq8_1->ds & 0xFFFFu)));
}

static __device__ __forceinline__ float vec_dot_q2_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q2_K * bq2_K = (const block_q2_K *) vbq + kbx;

    const int bq8_offset = QR2_K * (iqs / QI8_1);
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const uint8_t * scales = bq2_K->scales + scale_offset;

    const int v = get_int_b4(bq2_K->qs, iqs);
    int    u[QR2_K];
    float d8[QR2_K];

#pragma unroll
    for (int i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = rdna4::fp16_to_float((uint16_t)((bq8_1[bq8_offset + i].ds) & 0xFFFFu));
    }

    return vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q3_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q3_K * bq3_K = (const block_q3_K *) vbq + kbx;

    const int bq8_offset = QR3_K * (iqs / (QI3_K/2));
    const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    const float d = rdna4::fp16_to_float(bq3_K->d);

    const int vl = get_int_b2(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    const int vh = ~get_int_b2(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    int    u[QR3_K];
    float d8[QR3_K];

#pragma unroll
    for (int i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = rdna4::fp16_to_float((uint16_t)((bq8_1[bq8_offset + i].ds) & 0xFFFFu));
    }

    return vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

static __device__ __forceinline__ float vec_dot_q4_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q4_K * bq4_K = (const block_q4_K *) vbq + kbx;

    int    v[2];
    int    u[2*QR4_K];
    float d8[QR4_K];

    // iqs is in 0,2..30. bq8_offset = iqs/4 -> bq8_offset = 0, 2, 4, 6
    const int bq8_offset = QR4_K * ((iqs/2) / (QI8_1/2));

    // iqs = 0....3 -> bq8_offset = 0, want q4_offset = 0, 4, 8, 12
    // iqs = 4....7 -> bq8_offset = 2, want q4_offset = 32, 36, 40, 44
    // iqs = 8...11 -> bq8_offset = 4, want q4_offset = 64, 68, 72, 76
    // iqs = 12..15 -> bq8_offset = 6, want q4_offset = 96, 100, 104, 108

    const int * q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    v[0] = q4[0];
    v[1] = q4[4];

    // branchless so nvcc can hoist this out of the ncols_dst loop
    const uint16_t * scales = (const uint16_t *)bq4_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = rdna4::fp16_to_float((uint16_t)(bq8i->ds & 0xFFFFu));

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q5_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q5_K * bq5_K = (const block_q5_K *) vbq + kbx;

    int   vl[2];
    int   vh[2];
    int    u[2*QR5_K];
    float d8[QR5_K];

    const int bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    const int * ql = (const int *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    const int * qh = (const int *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    // same as q4_K
    const uint16_t * scales = (const uint16_t *)bq5_K->scales;
    const int j  = bq8_offset/2;
    const int jm = j & 1;

    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];

    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);

    uint16_t aux[2];
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));

    const uint8_t * sc = (const uint8_t *)aux;
    const uint8_t * m  = sc + 2;

#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = rdna4::fp16_to_float((uint16_t)(bq8i->ds & 0xFFFFu));

        const int * q8 = (const int *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}

static __device__ __forceinline__ float vec_dot_q6_K_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_q6_K * bq6_K = (const block_q6_K *) vbq + kbx;

    const int bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    const int scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    const int vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    const int vl = get_int_b2(bq6_K->ql, iqs);
    const int vh = get_int_b2(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    const int8_t * scales = bq6_K->scales + scale_offset;

    int    u[QR6_K];
    float d8[QR6_K];

#pragma unroll
    for (int i = 0; i < QR6_K; ++i) {
        u[i]  = get_int_b4(bq8_1[bq8_offset + 2*i].qs, iqs % QI8_1);
        d8[i] = rdna4::fp16_to_float((uint16_t)((bq8_1[bq8_offset + 2*i].ds) & 0xFFFFu));
    }

    return vec_dot_q6_K_q8_1_impl_mmvq(vl, vh, u, scales, rdna4::fp16_to_float(bq6_K->d), d8);
}

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid0 = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid1 = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;
    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0 = 0;
    int sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos[0] ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos[1] ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;

    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const int signs0 = __vcmpne4(signs & 0x08040201, 0);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);

        const int signs1 = __vcmpne4(signs & 0x80402010, 0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq1_s_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {
    const block_iq1_s * bq1 = (const block_iq1_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq1->qs, iqs);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq1->qh[iqs];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int grid = iq1s_grid_gpu[qs[l0/2] | (((qh >> 3*(l0/2)) & 0x07) << 8)];

        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;

        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }

    const float  d1q   = rdna4::fp16_to_float(bq1->d) * (((qh >> 11) & 0x0E) + 1);
    const float  delta = -1.0f + IQ1S_DELTA - (qh & 0x8000) * (2.0f*IQ1S_DELTA/0x8000);
    const float2 ds    = make_float2(rdna4::fp16_to_float((uint16_t)(bq8_1[iqs].ds & 0xFFFFu)),
                                     rdna4::fp16_to_float((uint16_t)(bq8_1[iqs].ds >> 16)));
    return d1q * (ds.x*sumi + ds.y*delta);
}

static __device__ __forceinline__ float vec_dot_iq4_nl_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_nl * bq4 = (const block_iq4_nl *) vbq + kbx;

    const int * q8 = (const int *) bq8_1->qs + iqs;

    int sumi = 0;
#pragma unroll
    for (int l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }

    const float d = rdna4::fp16_to_float(bq4->d) * rdna4::fp16_to_float((uint16_t)(bq8_1->ds & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq4_xs_q8_1(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq4_xs * bq4 = (const block_iq4_xs *) vbq + kbx;

    int sumi = 0;
#pragma unroll 1
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);

        const int u0 = get_int_b4(bq8_1[iqs/4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs/4].qs, j + 4);

        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }

    const int ls = ((bq4->scales_l[iqs/8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs/2)) & 0x03) << 4);
    sumi *= ls - 32;

    const float d = rdna4::fp16_to_float(bq4->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/4].ds) & 0xFFFFu));
    return d * sumi;
}


// ---------------------------------------------------------------------------
// iq3_s sign-handling variants (bench A/B, see check-matvec-gpu --bench-ab).
//
// The shipping body computes, per 4-byte group:
//     grid_l = __vsub4(grid_pos.x ^ m, m)        // per-byte conditional negate
// What that costs on gfx1201 is not obvious (llama.cpp's HIP __vsub4/__vcmpne4
// are per-byte emulations), so the bench can also run DIAGNOSTIC variants that
// are deliberately WRONG and exist only to attribute time:
//   DIAG_NOSIGN   - skips the sign application entirely
//   DIAG_NOLOOKUP - uses a constant instead of the iq3s_grid lookup
// and one CORRECT candidate:
//   LIN - uses dp4a linearity to remove the saturating packed-byte subtract:
//         sum (+-g)*u = 2 * sum (g & ~m)*u - sum g*u
//         (m is 0xFF on negative-sign bytes, so ~m selects the positive ones;
//          exact in int32, and identical to the shipping form for |g| <= 127).
// ---------------------------------------------------------------------------

// CORRECT candidate: dp4a-linearity form.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_lin(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;

    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi_pos = 0;  // dot over the positive-sign elements
    int sumi_all = 0;  // dot over every element (unsigned grid)
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);

        const int m0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int m1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi_pos = ggml_cuda_dp4a(grid_pos.x & ~m0, u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(grid_pos.y & ~m1, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a(grid_pos.x, u0, sumi_all);
        sumi_all = ggml_cuda_dp4a(grid_pos.y, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}


// CORRECT candidate 2: perm-based sign masking + dp4a linearity.
// Instead of materialising a 0x00/0xFF byte mask (__vcmpne4) and then applying
// it, a single V_PERM_B32 per group selects either the grid byte or a zero byte
// (indices 0-3 = grid bytes, 4-7 = the zero dword), so the mask costs no
// separate generation: spread the 4 sign bits into byte positions 2 of each
// byte (one multiply by 0x00810204 masked with 0x04040404), OR in the byte
// indices 0x03020100, and permute.
// ..._lut: mesmo corpo, mas a tabela (((const uint32_t *)lut), 512 x 4 B = 2 KB) vem de um ponteiro do
// chamador (LDS em matvec.cuh; docs/journal-kernels.md). Bit-exata: mesma
// tabela, mesmos valores, mesmas operacoes, mesma ordem.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_perm_lut(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, const void * __restrict__ lut) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * sp              = (const uint8_t *) &signs_packed_32;

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = ((const uint32_t *)lut)[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t gy = ((const uint32_t *)lut)[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const uint8_t sb = sp[l0/2];
        const uint32_t sel_x = (((uint32_t)(sb & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel_y = (((uint32_t)((sb >> 4) & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        // index 0-3 -> grid byte i ; index 4-7 -> zero byte (first operand is 0)
        const int gx_pos = (int)__builtin_amdgcn_perm(0u, gx, sel_x);
        const int gy_pos = (int)__builtin_amdgcn_perm(0u, gy, sel_y);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi_pos = ggml_cuda_dp4a(gx_pos, u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(gy_pos, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a((int)gx, u0, sumi_all);
        sumi_all = ggml_cuda_dp4a((int)gy, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// Entrada de producao: mesmo corpo lendo a tabela global. Separada para um
// bench poder A/B as duas fontes com aritmetica identica.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_perm(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs) {
    return vec_dot_iq3_s_q8_1_perm_lut(vbq, bq8_1, kbx, iqs, (const void *)iq3s_grid);
}


// ---------------------------------------------------------------------------
// perm + linearity variants for the other sign-using IQ types (same transform
// validated on iq3_s: ~2x, bit-identical). Each group replaces
//     g_signed = __vsub4(g ^ m, m)            (m = 0x00/0xFF per byte)
// with a single V_PERM_B32 that selects either the grid byte or a zero byte
// (selector byte = i for positive, 4+i for negative), accumulating two exact
// integer dots and combining them as sumi = 2*sumi_pos - sumi_all:
//     sum (+-g)*u = 2*sum(g & ~m)*u - sum g*u
// ---------------------------------------------------------------------------

static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_perm(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xxs_grid)[aux8[k0/2]];
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));

        const uint32_t sel0 = ((uint32_t)__vcmpne4(signs & 0x08040201, 0) & 0x04040404u) | 0x03020100u;
        const int g0_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.x, sel0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        sumi_pos = ggml_cuda_dp4a(g0_pos, u0, sumi_pos);
        sumi_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi_all);

        const uint32_t sel1 = ((uint32_t)__vcmpne4(signs & 0x80402010, 0) & 0x04040404u) | 0x03020100u;
        const int g1_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.y, sel1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi_pos = ggml_cuda_dp4a(g1_pos, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_perm(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0_pos = 0, sumi0_all = 0;
    int sumi1_pos = 0, sumi1_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)iq2xs_grid)[q2[l0/2] & 0x1FF];
        const uint32_t signs = unpack_ksigns(q2[l0/2] >> 9);

        const uint32_t sel0 = ((uint32_t)__vcmpne4(signs & 0x08040201, 0) & 0x04040404u) | 0x03020100u;
        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.x, sel0);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const uint32_t sel1 = ((uint32_t)__vcmpne4(signs & 0x80402010, 0) & 0x04040404u) | 0x03020100u;
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.y, sel1);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi0_pos);
            sumi0_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi0_pos);
            sumi0_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi0_all);
            sumi0_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi0_all);
        } else {
            sumi1_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi1_pos);
            sumi1_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi1_pos);
            sumi1_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi1_all);
            sumi1_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi1_all);
        }
    }
    const int sumi0 = 2*sumi0_pos - sumi0_all;
    const int sumi1 = 2*sumi1_pos - sumi1_all;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_perm(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0_pos = 0, sumi0_all = 0;
    int sumi1_pos = 0, sumi1_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(iq2s_grid + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const uint32_t sel0 = ((uint32_t)__vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = ((uint32_t)__vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000) & 0x04040404u) | 0x03020100u;
        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos[0], sel0);
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos[1], sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi0_pos);
            sumi0_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi0_pos);
            sumi0_all = ggml_cuda_dp4a(grid_pos[0], u0, sumi0_all);
            sumi0_all = ggml_cuda_dp4a(grid_pos[1], u1, sumi0_all);
        } else {
            sumi1_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi1_pos);
            sumi1_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi1_pos);
            sumi1_all = ggml_cuda_dp4a(grid_pos[0], u0, sumi1_all);
            sumi1_all = ggml_cuda_dp4a(grid_pos[1], u1, sumi1_all);
        }
    }
    const int sumi0 = 2*sumi0_pos - sumi0_all;
    const int sumi1 = 2*sumi1_pos - sumi1_all;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_perm(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;

    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7*l0/2));

        const uint32_t sel0 = ((uint32_t)__vcmpne4(signs & 0x08040201, 0) & 0x04040404u) | 0x03020100u;
        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.x, sel0);
        const uint32_t sel1 = ((uint32_t)__vcmpne4(signs & 0x80402010, 0) & 0x04040404u) | 0x03020100u;
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.y, sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a(grid_pos.x, u0, sumi_all);
        sumi_all = ggml_cuda_dp4a(grid_pos.y, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}


// iq3_xxs: leaner selector. `unpack_ksigns` broadcasts the sign byte to all four
// bytes (a multiply) and then __vcmpne4 turns the relevant bits into a mask; we
// only need 4 bits per group, so compute the byte once and take a nibble:
//   unpack_ksigns(v) = (v ^ ((popc(v)&1) << 7)) * 0x01010101, and the masks
//   0x08040201 / 0x80402010 pick bits 0-3 / 4-7 of that byte.
// ..._lut: mesmo corpo, mas a tabela (((const uint32_t *)lut), 256 x 4 B = 1 KB) vem de um ponteiro do
// chamador (LDS em matvec.cuh; docs/journal-kernels.md). Bit-exata: mesma
// tabela, mesmos valores, mesmas operacoes, mesma ordem.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_perm2_lut(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, const void * __restrict__ lut) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;

    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(((const uint32_t *)lut)[q3[l0 + 0]], ((const uint32_t *)lut)[q3[l0 + 1]]);

        uint32_t sv = (uint32_t)(uint8_t)(aux32 >> (7*l0/2));
        sv ^= (uint32_t)(__popc(sv) & 1u) << 7;
        const uint32_t sel0 = (((sv       & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = ((((sv >> 4) & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.x, sel0);
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.y, sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a(grid_pos.x, u0, sumi_all);
        sumi_all = ggml_cuda_dp4a(grid_pos.y, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// Entrada de producao: mesmo corpo lendo a tabela global. Separada para um
// bench poder A/B as duas fontes com aritmetica identica.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_perm2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs) {
    return vec_dot_iq3_xxs_q8_1_perm2_lut(vbq, bq8_1, kbx, iqs, (const void *)iq3xxs_grid);
}

// DIAGNOSTIC (wrong on purpose) for iq3_xxs: skip the sign application.
static __device__ __forceinline__ float vec_dot_iq3_xxs_q8_1_diag_nosign(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(iq3xxs_grid[q3[l0 + 0]], iq3xxs_grid[q3[l0 + 1]]);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_pos.x, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_pos.y, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}


// ---------------------------------------------------------------------------
// Lean-selector (perm2) forms for the remaining sign types: the sign nibble is
// taken directly from the packed sign byte (no unpack_ksigns broadcast and no
// __vcmpne4 mask), and a single V_PERM_B32 per group picks the grid byte or a
// zero byte. Same exact-integer linearity as the other perm variants.
// ---------------------------------------------------------------------------
// ..._lut: mesmo corpo, mas a tabela (((const uint64_t *)lut), 256 x 8 B = 2 KB) vem de um ponteiro do
// chamador (LDS em matvec.cuh; docs/journal-kernels.md). Bit-exata: mesma
// tabela, mesmos valores, mesmas operacoes, mesma ordem.
static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_perm2_lut(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, const void * __restrict__ lut) {

    const block_iq2_xxs * bq2 = (const block_iq2_xxs *) vbq + kbx;

    const int q2 = get_int_b2(bq2->qs, iqs);
    const uint8_t * aux8 = (const uint8_t *) &q2;
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = ((const uint2*)((const uint64_t *)lut))[aux8[k0/2]];

        uint32_t sv = (uint32_t)(uint8_t)(aux32 >> (7 * k0 / 2));
        sv ^= (uint32_t)(__popc(sv) & 1u) << 7;
        const uint32_t sel0 = (((sv       & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = ((((sv >> 4) & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        const int g0_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.x, sel0);
        const int g1_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.y, sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, k0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, k0 + 1);
        sumi_pos = ggml_cuda_dp4a(g0_pos, u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(g1_pos, u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi_all);
        sumi_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// Entrada de producao: mesmo corpo lendo a tabela global. Separada para um
// bench poder A/B as duas fontes com aritmetica identica.
static __device__ __forceinline__ float vec_dot_iq2_xxs_q8_1_perm2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs) {
    return vec_dot_iq2_xxs_q8_1_perm2_lut(vbq, bq8_1, kbx, iqs, (const void *)iq2xxs_grid);
}

// ..._lut: mesmo corpo, mas a tabela (((const uint64_t *)lut), 512 x 8 B = 4 KB) vem de um ponteiro do
// chamador (LDS em matvec.cuh; docs/journal-kernels.md). Bit-exata: mesma
// tabela, mesmos valores, mesmas operacoes, mesma ordem.
static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_perm2_lut(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, const void * __restrict__ lut) {

    const block_iq2_xs * bq2 = (const block_iq2_xs *) vbq + kbx;

    const int2 q2_packed = make_int2(get_int_b2(bq2->qs, iqs + 0), get_int_b2(bq2->qs, iqs + 1));
    const uint16_t * q2 = (const uint16_t *) &q2_packed;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0_pos = 0, sumi0_all = 0;
    int sumi1_pos = 0, sumi1_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = ((const uint2*)((const uint64_t *)lut))[q2[l0/2] & 0x1FF];

        uint32_t sv = (uint32_t)(uint8_t)(q2[l0/2] >> 9);
        sv ^= (uint32_t)(__popc(sv) & 1u) << 7;
        const uint32_t sel0 = (((sv       & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = ((((sv >> 4) & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.x, sel0);
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, grid_pos.y, sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi0_pos);
            sumi0_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi0_pos);
            sumi0_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi0_all);
            sumi0_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi0_all);
        } else {
            sumi1_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi1_pos);
            sumi1_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi1_pos);
            sumi1_all = ggml_cuda_dp4a((int)grid_pos.x, u0, sumi1_all);
            sumi1_all = ggml_cuda_dp4a((int)grid_pos.y, u1, sumi1_all);
        }
    }
    const int sumi0 = 2*sumi0_pos - sumi0_all;
    const int sumi1 = 2*sumi1_pos - sumi1_all;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// Entrada de producao: mesmo corpo lendo a tabela global. Separada para um
// bench poder A/B as duas fontes com aritmetica identica.
static __device__ __forceinline__ float vec_dot_iq2_xs_q8_1_perm2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs) {
    return vec_dot_iq2_xs_q8_1_perm2_lut(vbq, bq8_1, kbx, iqs, (const void *)iq2xs_grid);
}

// ..._lut: mesmo corpo, mas a tabela (((const uint64_t *)lut), 1024 x 8 B = 8 KB) vem de um ponteiro do
// chamador (LDS em matvec.cuh; docs/journal-kernels.md). Bit-exata: mesma
// tabela, mesmos valores, mesmas operacoes, mesma ordem.
static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_perm2_lut(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs, const void * __restrict__ lut) {

    const block_iq2_s * bq2 = (const block_iq2_s *) vbq + kbx;

    const int       qs_packed = get_int_b2(bq2->qs, iqs/2);
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq2->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq2->qs, QK_K/32 + iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    const int ls0 = bq2->scales[iqs/2] & 0x0F;
    const int ls1 = bq2->scales[iqs/2] >> 4;

    int sumi0_pos = 0, sumi0_all = 0;
    int sumi1_pos = 0, sumi1_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int * grid_pos = (const int *)(((const uint64_t *)lut) + (qs[l0/2] | ((qh << (8-l0)) & 0x300)));

        const uint8_t sb = signs_packed_8[l0/2];
        const uint32_t sel0 = (((uint32_t)(sb & 0x0Fu)          * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = (((uint32_t)((sb >> 4) & 0x0Fu)   * 0x00810204u) & 0x04040404u) | 0x03020100u;

        const int g_l_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos[0], sel0);
        const int g_h_pos = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos[1], sel1);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        if (l0 < 4) {
            sumi0_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi0_pos);
            sumi0_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi0_pos);
            sumi0_all = ggml_cuda_dp4a(grid_pos[0], u0, sumi0_all);
            sumi0_all = ggml_cuda_dp4a(grid_pos[1], u1, sumi0_all);
        } else {
            sumi1_pos = ggml_cuda_dp4a(g_l_pos, u0, sumi1_pos);
            sumi1_pos = ggml_cuda_dp4a(g_h_pos, u1, sumi1_pos);
            sumi1_all = ggml_cuda_dp4a(grid_pos[0], u0, sumi1_all);
            sumi1_all = ggml_cuda_dp4a(grid_pos[1], u1, sumi1_all);
        }
    }
    const int sumi0 = 2*sumi0_pos - sumi0_all;
    const int sumi1 = 2*sumi1_pos - sumi1_all;
    const int sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2)/4;
    const float d = rdna4::fp16_to_float(bq2->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// Entrada de producao: mesmo corpo lendo a tabela global. Separada para um
// bench poder A/B as duas fontes com aritmetica identica.
static __device__ __forceinline__ float vec_dot_iq2_s_q8_1_perm2(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx,
    const int & iqs) {
    return vec_dot_iq2_s_q8_1_perm2_lut(vbq, bq8_1, kbx, iqs, (const void *)iq2s_grid);
}


// iq3_s: direct 6-op sign application (research-agent proposal, independently
// verified exhaustively here: 0 mismatches over all 512 iq3s_grid entries x all
// 16 sign nibbles). Per 4-byte group:
//   t_b = 1 if sign bit b else 0      (nib * 0x00204081) & 0x01010101
//   m_b = 0xFF if sign bit b else 0   t * 0xFF
//   signed = (g ^ m) + t              == +g when t=0, == -g when t=1
// The add cannot carry between bytes because every iq3s_grid byte is odd and
// >= 1, so (~g) + 1 <= 0xFF (checked over the whole table).
// Keeps ONE dp4a per group instead of the two the linearity form needs.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_xoradd(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * sp              = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t gy = iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const uint32_t sb = sp[l0/2];
        const uint32_t tx = (((sb)       & 0x0Fu) * 0x00204081u) & 0x01010101u;
        const uint32_t ty = (((sb >> 4)  & 0x0Fu) * 0x00204081u) & 0x01010101u;

        const int grid_l = (int)((gx ^ (tx * 0xFFu)) + tx);
        const int grid_h = (int)((gy ^ (ty * 0xFFu)) + ty);

        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// DIAGNOSTIC (wrong on purpose): no sign application.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_diag_nosign(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(
            iq3s_grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)],
            iq3s_grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)]);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_pos.x, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_pos.y, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// DIAGNOSTIC (wrong on purpose): no grid lookup.
static __device__ __forceinline__ float vec_dot_iq3_s_q8_1_diag_nolookup(
    const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1, const int & kbx, const int & iqs) {

    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];
    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * signs_packed_8  = (const uint8_t *) &signs_packed_32;

    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        // keep the index computation (so the same ALU is spent) but skip the load
        const uint32_t g0 = 0x01010101u * (uint32_t)(qs[l0 + 0] | ((qh << (8 - l0)) & 0x100));
        const uint32_t g1 = 0x01010101u * (uint32_t)(qs[l0 + 1] | ((qh << (7 - l0)) & 0x100));
        const int2 grid_pos = make_int2((int)g0, (int)g1);
        const int signs0 = __vcmpne4(((signs_packed_8[l0/2] & 0x03) << 7) | ((signs_packed_8[l0/2] & 0x0C) << 21), 0x00000000);
        const int signs1 = __vcmpne4(((signs_packed_8[l0/2] & 0x30) << 3) | ((signs_packed_8[l0/2] & 0xC0) << 17), 0x00000000);
        const int grid_l = __vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = __vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

// ---------------------------------------------------------------------------
// Formas "prep" para o caminho em LOTE (matvec_kernel_batch): a dequantizacao do
// peso -- gather da LUT, montagem da mascara de sinal e o V_PERM_B32 -- e' feita
// UMA vez por bloco, e o resultado (os 4 int32 por grupo de 4 valores, na forma
// "positivo" e "cru" que a linearidade exata usa) fica em registrador para as N
// linhas de ativacao. So' o dp4a e' por token.
//
// Por que existe (medido pela frente de prefill, janela limpa): o matvec em lote
// lia 12,0 GB por chunk de 16 tokens em 110 ms = 109 GB/s, contra 446 GB/s do
// MESMO matvec no caminho por token -- 3,6x mais trabalho de ALU por byte,
// porque a dequantizacao era refeita N vezes.
//
// Bit-exatidao: mesmas operacoes, mesma ordem. O que sai do laco por token e'
// exatamente o que nao depende do token; a sequencia de dp4a e' identica a de
// `vec_dot_iq3_s_q8_1_perm_lut`. Gate: `check-matmul-gpu` (lote == N GEMVs) e
// `check-batch-gpu` (prefill em lote == caminho por token), os dois bit-exatos.
// ---------------------------------------------------------------------------
struct iq3s_prep_t {
  int v[4][4];  // [l0/2] = {gx_pos, gy_pos, gx, gy}
};

static __device__ __forceinline__ iq3s_prep_t vec_prep_iq3_s_q8_1_perm(
    const void * __restrict__ vbq, const int & kbx, const int & iqs,
    const void * __restrict__ lut) {
    const uint32_t * __restrict__ grid = (const uint32_t *) lut;
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    const int2      qs_packed = make_int2(get_int_b2(bq3->qs, iqs + 0), get_int_b2(bq3->qs, iqs + 1));
    const uint8_t * qs        = (const uint8_t *) &qs_packed;
    const int qh = bq3->qh[iqs/2];

    const int       signs_packed_32 = get_int_b2(bq3->signs, iqs/2);
    const uint8_t * sp              = (const uint8_t *) &signs_packed_32;

    iq3s_prep_t out;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t gx = grid[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
        const uint32_t gy = grid[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];

        const uint8_t sb = sp[l0/2];
        const uint32_t sel_x = (((uint32_t)(sb & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel_y = (((uint32_t)((sb >> 4) & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        out.v[l0/2][0] = (int)__builtin_amdgcn_perm(0u, gx, sel_x);
        out.v[l0/2][1] = (int)__builtin_amdgcn_perm(0u, gy, sel_y);
        out.v[l0/2][2] = (int)gx;
        out.v[l0/2][3] = (int)gy;
    }
    return out;
}

static __device__ __forceinline__ float vec_dot_prep_iq3_s_q8_1_perm(
    const iq3s_prep_t & w, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs, const void * __restrict__ /*lut*/) {
    const block_iq3_s * bq3 = (const block_iq3_s *) vbq + kbx;

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);

        sumi_pos = ggml_cuda_dp4a(w.v[l0/2][0], u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(w.v[l0/2][1], u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a(w.v[l0/2][2], u0, sumi_all);
        sumi_all = ggml_cuda_dp4a(w.v[l0/2][3], u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    sumi *= 1 + 2*((bq3->scales[iqs/4] >> ((iqs << 1) & 0x04)) & 0x0F);

    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

struct iq3xxs_prep_t {
  int v[4][4];  // [l0/2] = {g_l_pos, g_h_pos, grid_pos.x, grid_pos.y}
};

static __device__ __forceinline__ iq3xxs_prep_t vec_prep_iq3_xxs_q8_1_perm2(
    const void * __restrict__ vbq, const int & kbx, const int & iqs,
    const void * __restrict__ lut) {
    const uint32_t * __restrict__ grid = (const uint32_t *) lut;
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;

    const int2 q3_packed = make_int2(get_int_b2(bq3->qs, iqs), get_int_b2(bq3->qs, iqs+1));
    const uint8_t * q3 = (const uint8_t *) &q3_packed;

    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    iq3xxs_prep_t out;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int2 grid_pos = make_int2(grid[q3[l0 + 0]], grid[q3[l0 + 1]]);

        uint32_t sv = (uint32_t)(uint8_t)(aux32 >> (7*l0/2));
        sv ^= (uint32_t)(__popc(sv) & 1u) << 7;
        const uint32_t sel0 = (((sv       & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;
        const uint32_t sel1 = ((((sv >> 4) & 0x0Fu) * 0x00810204u) & 0x04040404u) | 0x03020100u;

        out.v[l0/2][0] = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.x, sel0);
        out.v[l0/2][1] = (int)__builtin_amdgcn_perm(0u, (uint32_t)grid_pos.y, sel1);
        out.v[l0/2][2] = grid_pos.x;
        out.v[l0/2][3] = grid_pos.y;
    }
    return out;
}

static __device__ __forceinline__ float vec_dot_prep_iq3_xxs_q8_1_perm2(
    const iq3xxs_prep_t & w, const void * __restrict__ vbq, const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs, const void * __restrict__ /*lut*/) {
    const block_iq3_xxs * bq3 = (const block_iq3_xxs *) vbq + kbx;
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K/16 + iqs/2);

    int sumi_pos = 0;
    int sumi_all = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int u0 = get_int_b4(bq8_1[iqs/2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs/2].qs, l0 + 1);
        sumi_pos = ggml_cuda_dp4a(w.v[l0/2][0], u0, sumi_pos);
        sumi_pos = ggml_cuda_dp4a(w.v[l0/2][1], u1, sumi_pos);
        sumi_all = ggml_cuda_dp4a(w.v[l0/2][2], u0, sumi_all);
        sumi_all = ggml_cuda_dp4a(w.v[l0/2][3], u1, sumi_all);
    }
    int sumi = 2*sumi_pos - sumi_all;

    const int ls = aux32 >> 28;
    sumi = (ls*sumi + sumi/2)/2;
    const float d = rdna4::fp16_to_float(bq3->d) * rdna4::fp16_to_float((uint16_t)((bq8_1[iqs/2].ds) & 0xFFFFu));
    return d * sumi;
}

}  // namespace rdna4
