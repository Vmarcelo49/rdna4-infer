// M2 — device-side dequant (quantized block -> f32), HIP gfx1201.
//
// Vendored from llama.cpp's ggml/src/ggml-cuda/dequantize.cuh (master @ df03399b8, MIT)
// with mechanical adaptations only:
//   - block structs come from rdna4/quants.h (plain C++ types, same layout)
//   - ggml_half -> uint16_t, __low2half/__high2half -> rdna4::fp16_to_float
//   - dst_t template parameter -> float
//   - ggml_cuda_cast<dst_t>(v) -> static_cast<float>(v)
//
// Thread mapping per type (from llama.cpp getrows.cu get_rows_cuda_kq<N,...>):
//   64 threads: q2_K, q3_K, q5_K, q6_K
//   32 threads: q4_K, iq2_xxs, iq2_xs, iq2_s, iq3_xxs, iq3_s, iq1_s, iq1_m, iq4_nl, iq4_xs
//   q8_0 uses the float2 form: one call covers 2 elements (iqs, iqs+1).
#pragma once
#include <hip/hip_runtime.h>

#include "rdna4/quants.h"
#include "rdna4/fp16.h"
#include "rdna4/quant_tables.h"

namespace rdna4 {

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int64_t ib, const int iqs, float2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const float d = rdna4::fp16_to_float(x[ib].d);

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

    v.x *= d;
    v.y *= d;
}

//================================== k-quants

// Each call dequantizes one super-block of QK_K values into y using the
// thread layout of the caller: 32 threads for q4_K, 64 threads otherwise.

static __device__ __forceinline__ void dequantize_q2_K(const void * vx, const int64_t ib, float * yy, const int tid) {
    const block_q2_K * x = (const block_q2_K *) vx;

    const int64_t n   = tid/32;
    const int64_t l   = tid - 32*n;
    const int64_t is  = 8*n + l/16;

    const uint8_t q = x[ib].qs[32*n + l];
    float * y = yy + 128*n;

    float dall = rdna4::fp16_to_float(x[ib].dm.d);
    float dmin = rdna4::fp16_to_float(x[ib].dm.dmin);
    y[l+ 0] = static_cast<float>(dall * (x[ib].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[ib].scales[is+0] >> 4));
    y[l+32] = static_cast<float>(dall * (x[ib].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[ib].scales[is+2] >> 4));
    y[l+64] = static_cast<float>(dall * (x[ib].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[ib].scales[is+4] >> 4));
    y[l+96] = static_cast<float>(dall * (x[ib].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[ib].scales[is+6] >> 4));
}

static __device__ __forceinline__ void dequantize_q3_K(const void * vx, const int64_t ib, float * yy, const int tid) {
    const block_q3_K * x = (const block_q3_K *) vx;

    const int64_t r = tid/4;
    const int64_t t = r/2;
    const int64_t is0 = r%2;
    const int64_t l0 = 16*is0 + 4*(tid%4);
    const int64_t n = t / 4;
    const int64_t j = t - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[ib].scales[is-0] & 0xF) | (((x[ib].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is+0] >> 4) & 3) << 4) :
                          (x[ib].scales[is-8] >>  4) | (((x[ib].scales[is-4] >> 6) & 3) << 4);
    float d_all = rdna4::fp16_to_float(x[ib].d);
    float dl = d_all * (us - 32);

    float * y = yy + 128*n + 32*j;
    const uint8_t * q = x[ib].qs + 32*n;
    const uint8_t * hm = x[ib].hmask;

    for (int l = l0; l < l0+4; ++l) {
        y[l] = static_cast<float>(dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}

static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

static __device__ __forceinline__ void dequantize_q4_K(const void * vx, const int64_t ib, float * yy, const int tid) {
    const block_q4_K * x = (const block_q4_K *) vx;

    // assume 32 threads
    const int64_t il  = tid/8;
    const int64_t ir  = tid%8;
    const int64_t is  = 2*il;
    const int64_t n   = 4;

    float * y = yy + 64*il + n*ir;

    const float dall = rdna4::fp16_to_float(x[ib].dm.d);
    const float dmin = rdna4::fp16_to_float(x[ib].dm.dmin);

    const uint8_t * q = x[ib].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = static_cast<float>(d1 * (q[l] & 0xF) - m1);
        y[l +32] = static_cast<float>(d2 * (q[l] >>  4) - m2);
    }
}

static __device__ __forceinline__ void dequantize_q5_K(const void * vx, const int64_t ib, float * yy, const int tid) {
    const block_q5_K * x = (const block_q5_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t il  = tid/16;   // il is in 0...3
    const int64_t ir  = tid%16;   // ir is in 0...15
    const int64_t is  = 2*il;     // is is in 0...6

    float * y = yy + 64*il + 2*ir;

    const float dall = rdna4::fp16_to_float(x[ib].dm.d);
    const float dmin = rdna4::fp16_to_float(x[ib].dm.dmin);

    const uint8_t * ql = x[ib].qs + 32*il + 2*ir;
    const uint8_t * qh = x[ib].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[ib].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[ib].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = static_cast<float>(d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1);
    y[ 1] = static_cast<float>(d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1);
    hm <<= 1;
    y[32] = static_cast<float>(d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2);
    y[33] = static_cast<float>(d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2);
}

static __device__ __forceinline__ void dequantize_q6_K(const void * vx, const int64_t ib, float * yy, const int tid) {
    const block_q6_K * x = (const block_q6_K *) vx;

    // assume 64 threads - this is very slightly better than the one below
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    float * y = yy + 128*ip + il;

    const float d = rdna4::fp16_to_float(x[ib].d);

    const uint8_t * ql = x[ib].ql + 64*ip + il;
    const uint8_t   qh = x[ib].qh[32*ip + il];
    const int8_t  * sc = x[ib].scales + is;

    y[ 0] = static_cast<float>(d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32));
    y[32] = static_cast<float>(d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32));
    y[64] = static_cast<float>(d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32));
    y[96] = static_cast<float>(d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32));
}

//================================== i-quants

// Each call dequantizes one super-block of QK_K values into y with 32
// threads; iq4_nl packs QK_K/QK4_NL sub-blocks per super-block.

static __device__ __forceinline__ void dequantize_iq2_xxs(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq2_xxs * x = (const block_iq2_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * aux8 = (const uint8_t *)q2;
    const uint8_t  * grid = (const uint8_t *)(iq2xxs_grid + aux8[il]);
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = rdna4::fp16_to_float(x[ibs].d) * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<float>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

static __device__ __forceinline__ void dequantize_iq2_xs(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq2_xs * x = (const block_iq2_xs *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const uint16_t * q2 = x[ibs].qs + 4*ib;
    const uint8_t  * grid = (const uint8_t *)(iq2xs_grid + (q2[il] & 511));
    const float d = rdna4::fp16_to_float(x[ibs].d) * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<float>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

static __device__ __forceinline__ void dequantize_iq2_s(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq2_s * x = (const block_iq2_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const uint8_t * grid = (const uint8_t *)(iq2s_grid + (x[ibs].qs[4*ib+il] | ((x[ibs].qh[ib] << (8-2*il)) & 0x300)));
    const float d = rdna4::fp16_to_float(x[ibs].d) * (0.5f + ((x[ibs].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K/8+4*ib+il];
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<float>(d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
    }
}

static __device__ __forceinline__ void dequantize_iq3_xxs(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq3_xxs * x = (const block_iq3_xxs  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const uint8_t  * q3 = x[ibs].qs + 8*ib;
    const uint16_t * gas = (const uint16_t *)(x[ibs].qs + QK_K/4) + 2*ib;
    const uint8_t  * grid1 = (const uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    const uint8_t  * grid2 = (const uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = rdna4::fp16_to_float(x[ibs].d) * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = static_cast<float>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = static_cast<float>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

static __device__ __forceinline__ void dequantize_iq3_s(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq3_s * x = (const block_iq3_s *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const uint8_t * qs = x[ibs].qs + 8*ib;
    const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[ibs].qh[ib] << (8-2*il)) & 256)));
    const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[ibs].qh[ib] << (7-2*il)) & 256)));
    const float d = rdna4::fp16_to_float(x[ibs].d) * (1 + 2*((x[ibs].scales[ib/2] >> 4*(ib%2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4*ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j+0] = static_cast<float>(d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f));
        y[j+4] = static_cast<float>(d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f));
    }
}

static __device__ __forceinline__ void dequantize_iq1_s(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq1_s * x = (const block_iq1_s  *) vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 8*il;
    const float delta = x[ibs].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    const float d = rdna4::fp16_to_float(x[ibs].d) * (2*((x[ibs].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; const int8_t * q = (const int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[ibs].qs[4*ib+il] | (((x[ibs].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
    for (int j = 0; j < 8; ++j) {
        y[j] = static_cast<float>(d * (q[j] + delta));
    }
}

static __device__ __forceinline__ void dequantize_iq4_nl(const void * vx, const int64_t ibs, float * yy, const int tid) {

    const block_iq4_nl * x = (const block_iq4_nl *) vx + ibs*(QK_K/QK4_NL);

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ib].qs + 4*il;
    const float d = rdna4::fp16_to_float(x[ib].d);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = static_cast<float>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = static_cast<float>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

static __device__ __forceinline__ void dequantize_iq4_xs(const void * vx, const int64_t ibs, float * yy, const int tid) {
    const block_iq4_xs * x = (const block_iq4_xs *)vx;

    const int64_t il = tid/8; // 0...3
    const int64_t ib = tid%8; // 0...7
    float * y = yy + 32*ib + 4*il;
    const uint8_t  * q4 = x[ibs].qs + 16*ib + 4*il;
    const float d = rdna4::fp16_to_float(x[ibs].d) * ((((x[ibs].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[ibs].scales_h >> 2*ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j+ 0] = static_cast<float>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j+16] = static_cast<float>(d * kvalues_iq4nl[q4[j] >>  4]);
    }
}

}  // namespace rdna4
