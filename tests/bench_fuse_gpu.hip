// H3 launch-fusion prototypes (PROTOTYPE-ONLY, decode path, gfx1201).
//
// Triage (read, not re-measured):
//   - H3.2 rms_norm+quantize: EXCLUDED, measured 0.817x DEAD last round
//     (1-CTA norm reduction serializes the 40-CTA quantize grid). This file
//     contains NO norm+quantize prototype. Fusions below keep per-launch
//     parallelism: every fused kernel runs the SAME CTA count as the widest
//     member of its pair/chain (the lesson of the 0.817x kill).
//   - H3.6 argmax device: LANDED (graph.cuh:111-122,2133-2163, 2x512B D2H).
//   - Already fused in production, verified by reading (not re-prototyped):
//     sigmoid+mul    -> nn.cuh:329 sigmoid_gate_kernel (graph.cuh:981)
//     silu+mul (FFN) -> nn.cuh:309 silu_gate_kernel   (graph.cuh:1145)
//     silu+mul (GDN) -> nn.cuh:309                    (graph.cuh:1121)
//     kv 8->2        -> graph.cuh:1015 kv_write_batch (decode path, :958)
//     gdn 4 scalars->1 -> gdn.cuh:796 gdn_scalars_launch (graph.cuh:1080)
//     conv+silu      -> gdn.cuh:36 out[c]=silu_f(acc)
//     pos memcpy     -> graph.cuh:938 (1x4B per TOKEN, cached)
//   Live in this file (fuse the already-fused elementwise kernels onwards,
//   plus small-kernel chains sharing CTA shape):
//     F1 kvK+kvV 2->1 (f16 default path; Q-types need a block-parallel variant)
//     F3a deinterleave+q_norm+k_norm 3->1 (28 CTAs, one per norm row)
//     F3b silu_gate+act_quant 2->1 (quantize grid; CTA-local, syncs inside CTA)
//     F3c sigmoid_gate+act_quant(attn_out) 2->1 (same pattern, n=6144)
//     F4 add_residual+attn_norm 2->1 (1 CTA, the norm's grid; add first, exact)
//     F5 ropeQ+ropeK+kvK+kvV 4->1 (f16; single-CTA aggressive + 2+2 fallback)
//     GDN l2q+l2k 2->1 (32 CTAs) ; GDN ssm_norm+silu+quant 3->1 (24x256)
//     COMP chains measuring composition directly (tuning.h:48-51 precedent).
//
// Bit-identity: every fused kernel does the same per-element fp ops in the same
// order as the unfused sequence (same fma chains, same reduction trees, same
// writebacks to the same addresses). memcmp over ALL outputs proves it.
//
// Method: pristine masters + work buffers; every timed iteration re-copies
// master->work INSIDE the event pair on BOTH sides (common-mode copy cost, so
// `saved` is unbiased). Warmup >=320ms, 3 rotated passes, best+mean, 1% floor
// vs the 26.4 ms token. stdout + /tmp/fuse_h3.txt dump.
//
// usage: bench-fuse-gpu [--rounds N] [--chain N]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "rdna4/nn.cuh"
#include "rdna4/matvec.cuh"
#include "rdna4/kv.h"
#include "rdna4/attn.cuh"
#include "rdna4/gdn.cuh"

namespace {

// Model geometry (Qwen3.8-27B UD-IQ3_S): E=5120, 64 layers (16 attn + 48 GDN),
// NH=24, NKV=4, HD=256, n_rot=64, FF=17408, d_inner=6144, key_dim=2048.
constexpr int E = 5120, NH = 24, NKV = 4, HD = 256, NROT = 64;
constexpr float FREQ = 10000000.0f, EPS = 1e-6f;
constexpr int FF = 17408, DIN = 6144, NKH = 16, SS = 128, NVH = 48;
constexpr int N_ATTN = 16, N_GDN = 48, N_ALL = 64;
constexpr double TOKEN_MS = 26.4;

__global__ void fill_kernel(float *__restrict__ x, long long n, unsigned seed) {
  const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned s = seed * 1664525u + 1013904223u + (unsigned)i * 2654435761u;
  s ^= s >> 13;
  s *= 1274126177u;
  x[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 2.0f;
}

double now_ms() {
  using namespace std::chrono;
  return (double)duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count() / 1000.0;
}

// ---------------------------------------------------------------------------
// F4: add_residual + attn_norm rms(1x5120,w) -> 1 CTA x 256 (the norm's grid).
// Replicates rms_norm_kernel's exact fma order over t=src+dst while writing
// back dst (the add's in-place update); values kept in registers for the epilog.
// ---------------------------------------------------------------------------
template <int N>
__global__ void fz_add_rms_kernel(const float *__restrict__ src, float *__restrict__ dst_io,
                                  const float *__restrict__ w, float *__restrict__ y, float eps) {
  __shared__ float partial[256];
  constexpr int T = 256;
  float tmp[N / T];
  float acc = 0.0f;
  std::int64_t i = threadIdx.x;
  int k = 0;
  for (; i + 3 * T < N; i += 4 * T) {
    const float t0 = src[i] + dst_io[i];
    const float t1 = src[i + T] + dst_io[i + T];
    const float t2 = src[i + 2 * T] + dst_io[i + 2 * T];
    const float t3 = src[i + 3 * T] + dst_io[i + 3 * T];
    dst_io[i] = t0;
    dst_io[i + T] = t1;
    dst_io[i + 2 * T] = t2;
    dst_io[i + 3 * T] = t3;
    tmp[k] = t0;
    tmp[k + 1] = t1;
    tmp[k + 2] = t2;
    tmp[k + 3] = t3;
    k += 4;
    acc = fmaf(t0, t0, acc);
    acc = fmaf(t1, t1, acc);
    acc = fmaf(t2, t2, acc);
    acc = fmaf(t3, t3, acc);
  }
  for (; i < N; i += T) {
    const float t = src[i] + dst_io[i];
    dst_io[i] = t;
    tmp[k++] = t;
    acc = fmaf(t, t, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)N + eps);
  i = threadIdx.x;
  k = 0;
  for (; i < N; i += T) y[i] = tmp[k++] * scale * w[i];
}

// ---------------------------------------------------------------------------
// F3a: deinterleave + q_norm(24x256,wq) + k_norm(4x256,wk) -> 28 CTAs x 256.
// One CTA per norm row; Q rows also emit their gate element (the deinterleave
// writeback). ncols=256 => 1 element/thread, same single fmaf as production.
// ---------------------------------------------------------------------------
__global__ void fz_deint_qknorm_kernel(const float *__restrict__ yq, float *__restrict__ q,
                                       float *__restrict__ gate, float *__restrict__ kio,
                                       const float *__restrict__ wq, const float *__restrict__ wk,
                                       float eps) {
  __shared__ float partial[256];
  const int b = blockIdx.x;
  const int t = threadIdx.x;
  float rv;
  const float *wrow;
  float *yrow;
  if (b < NH) {
    rv = yq[(std::int64_t)b * 2 * HD + t];
    gate[(std::int64_t)b * HD + t] = yq[(std::int64_t)b * 2 * HD + HD + t];
    wrow = wq;
    yrow = q + (std::int64_t)b * HD;
  } else {
    const int h = b - NH;
    rv = kio[(std::int64_t)h * HD + t];
    wrow = wk;
    yrow = kio + (std::int64_t)h * HD;
  }
  float acc = fmaf(rv, rv, 0.0f);
  partial[t] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) partial[t] += partial[t + s];
    __syncthreads();
  }
  const float sc = rsqrtf(partial[0] / (float)HD + eps);
  yrow[t] = rv * sc * wrow[t];
}

// ---------------------------------------------------------------------------
// F5a: ropeQ + ropeK -> one kernel, pair-indexed exactly like rope_kernel.
// No cross-thread reads => multi-CTA, zero sync, full parallelism.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void fz_rope_pair(float *__restrict__ x, int idx, int n_heads,
                                            int n_rot, float base, int pos) {
  const int pairs = n_rot / 2;
  const int p = idx % pairs;
  const int h = (idx / pairs) % n_heads;
  const float theta = (float)pos * powf(base, -2.0f * (float)p / (float)n_rot);
  const float c = cosf(theta);
  const float s = sinf(theta);
  float *bse = x + ((std::int64_t)h) * HD;
  const int jc = p + n_rot / 2;
  const float x0 = bse[p];
  const float x1 = bse[jc];
  bse[p] = x0 * c - x1 * s;
  bse[jc] = x0 * s + x1 * c;
}

__global__ void fz_rope_qk_kernel(float *__restrict__ Q, float *__restrict__ K, int pos) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j < NH * (NROT / 2)) fz_rope_pair(Q, j, NH, NROT, FREQ, pos);
  if (j < NKV * (NROT / 2)) fz_rope_pair(K, j, NKV, NROT, FREQ, pos);
}

// ---------------------------------------------------------------------------
// F1: kvK + kvV (f16) -> 1 kernel, elementwise, same op as kv_store_row F16.
// ---------------------------------------------------------------------------
__global__ void fz_kvkv_f16_kernel(const float *__restrict__ ks, const float *__restrict__ vs,
                                   void *__restrict__ kd, void *__restrict__ vd, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  ((std::uint16_t *)kd)[i] = rdna4::float_to_fp16(ks[i]);
  ((std::uint16_t *)vd)[i] = rdna4::float_to_fp16(vs[i]);
}

// ---------------------------------------------------------------------------
// F5: ropeQ + ropeK + kvK + kvV (f16) -> SINGLE CTA x 1024. Rope phases use
// 768/128 threads, store phase 1024; one __syncthreads between (same CTA).
// ---------------------------------------------------------------------------
__global__ void fz_prep_kv4_f16_kernel(float *__restrict__ Q, float *__restrict__ K,
                                       const float *__restrict__ Vs, void *__restrict__ Kd,
                                       void *__restrict__ Vd, int pos) {
  const int t = threadIdx.x;
  if (t < NH * (NROT / 2)) fz_rope_pair(Q, t, NH, NROT, FREQ, pos);
  if (t < NKV * (NROT / 2)) fz_rope_pair(K, t, NKV, NROT, FREQ, pos);
  __syncthreads();
  if (t < NKV * HD) {
    ((std::uint16_t *)Kd)[t] = rdna4::float_to_fp16(K[t]);
    ((std::uint16_t *)Vd)[t] = rdna4::float_to_fp16(Vs[t]);
  }
}

// ---------------------------------------------------------------------------
// F3b: silu_gate + act_quant -> the QUANTIZE grid ((nb+3)/4 x 128). Phase 1 is
// elementwise; phase 2 calls quantize_q8_1_block verbatim. CTA c owns elems
// [128c,128c+128) = blocks 4c..4c+3, so the barrier is CTA-local.
// a_io==v_out allowed (FFN single buffer, same write order as silu_gate).
// ---------------------------------------------------------------------------
__global__ void fz_silu_quant_kernel(float *__restrict__ a_io, const float *__restrict__ b,
                                     float *__restrict__ v_out,
                                     rdna4::block_q8_1 *__restrict__ y, std::int64_t n,
                                     std::int64_t nb) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float s = rdna4::silu_f(a_io[i]);
    a_io[i] = s;
    v_out[i] = s * b[i];
  }
  __syncthreads();
  const std::int64_t W = ((std::int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (W < nb) rdna4::quantize_q8_1_block(v_out + W * 32, y + W);
}

// ---------------------------------------------------------------------------
// F3c: sigmoid_gate + act_quant(attn_out input, n=6144) -> same pattern.
// ---------------------------------------------------------------------------
__global__ void fz_sig_quant_kernel(float *__restrict__ g_io, float *__restrict__ y_io,
                                    rdna4::block_q8_1 *__restrict__ y, std::int64_t n,
                                    std::int64_t nb) {
  const std::int64_t i = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float s = rdna4::sigmoid_f(g_io[i]);
    g_io[i] = s;
    y_io[i] = y_io[i] * s;
  }
  __syncthreads();
  const std::int64_t W = ((std::int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (W < nb) rdna4::quantize_q8_1_block(y_io + W * 32, y + W);
}

// ---------------------------------------------------------------------------
// GDN l2fuse: l2q(16x128) + l2k(16x128) -> 32 CTAs x 256, exact l2 loop
// (threads >=128 contribute 0 to the same tree; OOB read guarded).
// ---------------------------------------------------------------------------
__global__ void fz_l2qk_kernel(float *__restrict__ qio, float *__restrict__ kio, float eps) {
  __shared__ float partial[256];
  const int b = blockIdx.x;
  const int t = threadIdx.x;
  float *row = (b < NKH) ? qio + (std::int64_t)b * SS : kio + (std::int64_t)(b - NKH) * SS;
  const float v = (t < SS) ? row[t] : 0.0f;
  float acc = fmaf(v, v, 0.0f);
  partial[t] = acc;
  __syncthreads();
  for (int s = 128; s > 0; s >>= 1) {
    if (t < s) partial[t] += partial[t + s];
    __syncthreads();
  }
  const float sc = rsqrtf(partial[0] + eps);
  if (t < SS) row[t] = v * sc;
}

// ---------------------------------------------------------------------------
// GDN 3in1: ssm_norm rms(48x128,w) + silu_gate(z,v) + act_quant(v,192 blocks)
// -> 24 CTAs x 256. CTA b owns norm rows 2b,2b+1 AND elems [256b,256b+256)
// AND quant warps 8b..8b+7 (24*8=192=nb). All barriers CTA-local.
// ---------------------------------------------------------------------------
__global__ void fz_norm_silu_quant_kernel(float *__restrict__ v_io, float *__restrict__ z_io,
                                          const float *__restrict__ w,
                                          rdna4::block_q8_1 *__restrict__ q, float eps) {
  __shared__ float partial[256];
  const int b = blockIdx.x;
  const int t = threadIdx.x;
  for (int r = 0; r < 2; ++r) {
    const int row = 2 * b + r;
    float *rr = v_io + (std::int64_t)row * SS;
    float acc = 0.0f;
    std::int64_t i = t;
    for (; i + 3 * 256 < SS; i += 4 * 256) {  // skipped for SS=128 (as in production)
      const float v0 = rr[i];
      const float v1 = rr[i + 256];
      const float v2 = rr[i + 512];
      const float v3 = rr[i + 768];
      acc = fmaf(v0, v0, acc);
      acc = fmaf(v1, v1, acc);
      acc = fmaf(v2, v2, acc);
      acc = fmaf(v3, v3, acc);
    }
    for (; i < SS; i += 256) {
      const float v = rr[i];
      acc = fmaf(v, v, acc);
    }
    partial[t] = acc;
    __syncthreads();
    for (int s = 128; s > 0; s >>= 1) {
      if (t < s) partial[t] += partial[t + s];
      __syncthreads();
    }
    const float sc = rsqrtf(partial[0] / (float)SS + eps);
    // ssm_norm.weight is {state}=[128] SHARED across the 48 rows (model.cpp),
    // and rms_norm_kernel indexes w[i] per row -- replicate exactly.
    if (t < SS) rr[t] = rr[t] * sc * w[t];
  }
  __syncthreads();
  {
    const std::int64_t e = (std::int64_t)b * 256 + t;
    const float s = rdna4::silu_f(z_io[e]);
    z_io[e] = s;
    v_io[e] = s * v_io[e];
  }
  __syncthreads();
  {
    const std::int64_t W = (std::int64_t)b * 8 + (t >> 5);
    if (W < (DIN / 32)) rdna4::quantize_q8_1_block(v_io + W * 32, q + W);
  }
}

struct Stats {
  double best = 1e30, mean = 0;
};

Stats measure(const std::function<void()> &fn, int chain, int rounds, hipEvent_t e0,
              hipEvent_t e1) {
  const double w0 = now_ms();
  while (now_ms() - w0 < 320.0) {
    for (int i = 0; i < chain; ++i) fn();
    hipDeviceSynchronize();
  }
  double sum = 0, best = 1e30;
  for (int r = 0; r < rounds; ++r) {
    hipEventRecord(e0, nullptr);
    for (int i = 0; i < chain; ++i) fn();
    hipEventRecord(e1, nullptr);
    hipDeviceSynchronize();
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    const double us = (double)ms * 1000.0 / chain;
    best = std::min(best, us);
    sum += us;
  }
  return {best, sum / rounds};
}

}  // namespace

int main(int argc, char **argv) {
  int rounds = 12, chain = 300;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "unknown %s\n", argv[i]);
      return 1;
    }
  }
  FILE *dump = std::fopen("/tmp/fuse_h3.txt", "w");
  auto out = [&](const char *fmt, ...) {
    va_list ap1, ap2;
    va_start(ap1, fmt);
    va_copy(ap2, ap1);
    std::vprintf(fmt, ap1);
    if (dump) std::vfprintf(dump, fmt, ap2);
    va_end(ap1);
    va_end(ap2);
  };

  auto MA = [&](void **d, std::size_t n) {
    if (hipMalloc(d, n) != hipSuccess) {
      std::fprintf(stderr, "hipMalloc failed\n");
      std::exit(1);
    }
  };
  // masters (pristine inputs) + work buffers
  float *m_x, *m_res, *m_w, *m_yq, *m_ks, *m_wq, *m_wk, *m_Q, *m_K, *m_Vs;
  float *m_fa, *m_fb, *m_z, *m_v, *m_w128, *m_y, *m_g, *m_l2q, *m_l2k;
  float *w_x, *w_xn, *w_q, *w_g, *w_ks, *w_Q, *w_K;
  float *w_fa, *w_z, *w_v, *w_y, *w_l2q, *w_l2k;
  std::uint16_t *w_Kd, *w_Vd;
  rdna4::block_q8_1 *w_q544, *w_q192;
  int *dpos;
  MA((void **)&m_x, E * 4);
  MA((void **)&m_res, E * 4);
  MA((void **)&m_w, E * 4);
  MA((void **)&m_yq, (std::size_t)NH * 2 * HD * 4);
  MA((void **)&m_ks, (std::size_t)NKV * HD * 4);
  MA((void **)&m_wq, HD * 4);
  MA((void **)&m_wk, HD * 4);
  MA((void **)&m_Q, (std::size_t)NH * HD * 4);
  MA((void **)&m_K, (std::size_t)NKV * HD * 4);
  MA((void **)&m_Vs, (std::size_t)NKV * HD * 4);
  MA((void **)&m_fa, FF * 4);
  MA((void **)&m_fb, FF * 4);
  MA((void **)&m_z, DIN * 4);
  MA((void **)&m_v, DIN * 4);
  MA((void **)&m_w128, DIN * 4);
  MA((void **)&m_y, DIN * 4);
  MA((void **)&m_g, DIN * 4);
  MA((void **)&m_l2q, NKH * SS * 4);
  MA((void **)&m_l2k, NKH * SS * 4);
  MA((void **)&w_x, E * 4);
  MA((void **)&w_xn, E * 4);
  MA((void **)&w_q, (std::size_t)NH * HD * 4);
  MA((void **)&w_g, (std::size_t)NH * HD * 4);
  MA((void **)&w_ks, (std::size_t)NKV * HD * 4);
  MA((void **)&w_Q, (std::size_t)NH * HD * 4);
  MA((void **)&w_K, (std::size_t)NKV * HD * 4);
  MA((void **)&w_Kd, (std::size_t)NKV * HD * 2);
  MA((void **)&w_Vd, (std::size_t)NKV * HD * 2);
  MA((void **)&w_fa, FF * 4);
  MA((void **)&w_z, DIN * 4);
  MA((void **)&w_v, DIN * 4);
  MA((void **)&w_y, DIN * 4);
  MA((void **)&w_l2q, NKH * SS * 4);
  MA((void **)&w_l2k, NKH * SS * 4);
  MA((void **)&w_q544, (FF / 32) * sizeof(rdna4::block_q8_1));
  MA((void **)&w_q192, (DIN / 32) * sizeof(rdna4::block_q8_1));
  MA((void **)&dpos, sizeof(int));

  const unsigned bl = 256;
  auto fill = [&](float *d, long long n, unsigned s) {
    fill_kernel<<<(unsigned)((n + bl - 1) / bl), bl>>>(d, n, s);
  };
  fill(m_x, E, 21u);
  fill(m_res, E, 22u);
  fill(m_w, E, 23u);
  fill(m_yq, (std::size_t)NH * 2 * HD, 24u);
  fill(m_ks, (std::size_t)NKV * HD, 25u);
  fill(m_wq, HD, 26u);
  fill(m_wk, HD, 27u);
  fill(m_Q, (std::size_t)NH * HD, 28u);
  fill(m_K, (std::size_t)NKV * HD, 29u);
  fill(m_Vs, (std::size_t)NKV * HD, 30u);
  fill(m_fa, FF, 31u);
  fill(m_fb, FF, 32u);
  fill(m_z, DIN, 33u);
  fill(m_v, DIN, 34u);
  fill(m_w128, DIN, 35u);
  fill(m_y, DIN, 38u);
  fill(m_g, DIN, 39u);
  fill(m_l2q, (std::size_t)NKH * SS, 36u);
  fill(m_l2k, (std::size_t)NKH * SS, 37u);
  const int hpos = 129;
  hipMemcpy(dpos, &hpos, sizeof(int), hipMemcpyHostToDevice);
  hipDeviceSynchronize();

  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);

  out("H3 fusions bench (gfx1201, rounds=%d chain=%d, warmup>=320ms, token=%.1fms)\n", rounds,
      chain, TOKEN_MS);
  out("copies master->work inside the timed region on BOTH sides (common-mode).\n");
  out("%-20s %9s %9s %9s %9s %7s %7s  proj_ms proj_%%  verdict\n", "fusion", "pair", "fused",
      "fused2", "saved", "p-mean", "f-mean");

  int nship = 0, ndead = 0, ibad = 0;
  auto report = [&](const char *name, const std::function<void()> &pair,
                    const std::function<void()> &fused, int count) {
    Stats a{1e30, 0}, b{1e30, 0}, b2{1e30, 0};
    double sa = 0, sb = 0;
    for (int p = 0; p < 3; ++p) {
      Stats ta, tb, tc;
      if (p % 2 == 0) {
        ta = measure(pair, chain, rounds, e0, e1);
        tb = measure(fused, chain, rounds, e0, e1);
        tc = measure(fused, chain, rounds, e0, e1);
      } else {
        tb = measure(fused, chain, rounds, e0, e1);
        ta = measure(pair, chain, rounds, e0, e1);
        tc = measure(fused, chain, rounds, e0, e1);
      }
      a.best = std::min(a.best, ta.best);
      b.best = std::min(b.best, tb.best);
      b2.best = std::min(b2.best, tc.best);
      sa += ta.mean;
      sb += tb.mean;
    }
    const double saved = a.best - b.best;
    const double pms = saved * count / 1000.0;
    const double pct = 100.0 * pms / TOKEN_MS;
    const bool ship = pct >= 1.0;
    if (ship) ++nship;
    else ++ndead;
    out("%-20s %9.2f %9.2f %9.2f %9.2f %7.2f %7.2f  %7.3f %6.2f%%  %s\n", name, a.best, b.best,
        b2.best, saved, sa / 3, sb / 3, pms, pct, ship ? "SHIP-IT" : "dead");
    if (dump) std::fflush(dump);
  };

  std::vector<unsigned> ha, hb;
  auto get = [&](void *d, std::size_t bytes, std::vector<unsigned> &o) {
    o.resize((bytes + 3) / 4);
    hipMemcpy(o.data(), d, bytes, hipMemcpyDeviceToHost);
  };
  // outs: (work_ptr, bytes). Runs setup() then checks work bufs vs host gold.
  auto identity = [&](const char *name, const std::function<void()> &run_unfused,
                      const std::function<void()> &run_fused,
                      const std::vector<std::pair<void *, std::size_t>> &outs) {
    std::vector<std::vector<unsigned>> gold;
    run_unfused();
    hipDeviceSynchronize();
    for (auto &pr : outs) {
      get(pr.first, pr.second, ha);
      gold.push_back(ha);
    }
    run_fused();
    hipDeviceSynchronize();
    bool ok = true;
    for (std::size_t k = 0; k < outs.size(); ++k) {
      get(outs[k].first, outs[k].second, hb);
      if (hb.size() != gold[k].size() ||
          std::memcmp(hb.data(), gold[k].data(), outs[k].second) != 0)
        ok = false;
    }
    out("  %-22s %s\n", name, ok ? "BIT-IDENTICO" : "DIFERE");
    if (!ok) ++ibad;
    if (dump) std::fflush(dump);
  };

  auto cp = [&](void *d, void *s, std::size_t n) {
    hipMemcpy(d, s, n, hipMemcpyDeviceToDevice);
  };
  const std::size_t bE = E * 4, bQ = (std::size_t)NH * HD * 4, bKV = (std::size_t)NKV * HD * 4;
  const std::size_t bFF = FF * 4, bDIN = DIN * 4, bK = NKH * SS * 4;
  const std::size_t bQ544 = (FF / 32) * sizeof(rdna4::block_q8_1);
  const std::size_t bQ192 = (DIN / 32) * sizeof(rdna4::block_q8_1);
  const std::int64_t nb_ffn = FF / 32, nb_din = DIN / 32;
  const unsigned qg_ffn = (unsigned)((nb_ffn + 3) / 4), qg_din = (unsigned)((nb_din + 3) / 4);
  const int KVW = NKV * HD;

  out("\nidentity (memcmp, unfused vs fused):\n");

  // ---- F4 ----
  auto f4_u = [&] {
    cp(w_x, m_x, bE);
    (void)rdna4::add_launch(m_res, w_x, w_x, E);
    (void)rdna4::rms_norm_launch(w_x, m_w, w_xn, 1, E, EPS);
  };
  auto f4_f = [&] {
    cp(w_x, m_x, bE);
    fz_add_rms_kernel<E><<<1, 256>>>(m_res, w_x, m_w, w_xn, EPS);
  };

  // ---- F3a ----
  auto f3a_u = [&] {
    cp(w_ks, m_ks, bKV);
    (void)rdna4::deinterleave_q_gate_launch(m_yq, w_q, w_g, NH, HD);
    (void)rdna4::rms_norm_launch(w_q, m_wq, w_q, NH, HD, EPS);
    (void)rdna4::rms_norm_launch(w_ks, m_wk, w_ks, NKV, HD, EPS);
  };
  auto f3a_f = [&] {
    cp(w_ks, m_ks, bKV);
    fz_deint_qknorm_kernel<<<NH + NKV, 256>>>(m_yq, w_q, w_g, w_ks, m_wq, m_wk, EPS);
  };

  // ---- F5a ----
  auto f5a_u = [&] {
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    (void)rdna4::rope_launch(w_Q, 1, NH, HD, NROT, FREQ, dpos);
    (void)rdna4::rope_launch(w_K, 1, NKV, HD, NROT, FREQ, dpos);
  };
  auto f5a_f = [&] {
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    fz_rope_qk_kernel<<<3, 256>>>(w_Q, w_K, hpos);
  };

  // ---- F1 ----
  auto f1_u = [&] {
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, m_K, w_Kd, KVW);
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, m_Vs, w_Vd, KVW);
  };
  auto f1_f = [&] { fz_kvkv_f16_kernel<<<4, 256>>>(m_K, m_Vs, w_Kd, w_Vd, KVW); };

  // ---- F5 ----
  auto f5_u = [&] {
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    (void)rdna4::rope_launch(w_Q, 1, NH, HD, NROT, FREQ, dpos);
    (void)rdna4::rope_launch(w_K, 1, NKV, HD, NROT, FREQ, dpos);
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, w_K, w_Kd, KVW);
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, m_Vs, w_Vd, KVW);
  };
  auto f5_f = [&] {
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    fz_prep_kv4_f16_kernel<<<1, 1024>>>(w_Q, w_K, m_Vs, w_Kd, w_Vd, hpos);
  };

  // ---- F3b FFN ----
  auto f3b_u = [&] {
    cp(w_fa, m_fa, bFF);
    (void)rdna4::silu_gate_launch(w_fa, m_fb, w_fa, FF);
    rdna4::quantize_q8_1_kernel<<<qg_ffn, 128>>>(w_fa, w_q544, nb_ffn);
  };
  auto f3b_f = [&] {
    cp(w_fa, m_fa, bFF);
    fz_silu_quant_kernel<<<qg_ffn, 128>>>(w_fa, m_fb, w_fa, w_q544, FF, nb_ffn);
  };

  // ---- F3b GDN ----
  auto f3bg_u = [&] {
    cp(w_z, m_z, bDIN);
    cp(w_v, m_v, bDIN);
    (void)rdna4::silu_gate_launch(w_z, w_v, w_v, DIN);
    rdna4::quantize_q8_1_kernel<<<qg_din, 128>>>(w_v, w_q192, nb_din);
  };
  auto f3bg_f = [&] {
    cp(w_z, m_z, bDIN);
    cp(w_v, m_v, bDIN);
    fz_silu_quant_kernel<<<qg_din, 128>>>(w_z, w_v, w_v, w_q192, DIN, nb_din);
  };

  // ---- F3c ----
  auto f3c_u = [&] {
    cp(w_g, m_g, bDIN);
    cp(w_y, m_y, bDIN);
    (void)rdna4::sigmoid_gate_launch(w_g, w_y, DIN);
    rdna4::quantize_q8_1_kernel<<<qg_din, 128>>>(w_y, w_q192, nb_din);
  };
  auto f3c_f = [&] {
    cp(w_g, m_g, bDIN);
    cp(w_y, m_y, bDIN);
    fz_sig_quant_kernel<<<qg_din, 128>>>(w_g, w_y, w_q192, DIN, nb_din);
  };

  // ---- l2fuse ----
  auto l2_u = [&] {
    cp(w_l2q, m_l2q, bK);
    cp(w_l2k, m_l2k, bK);
    (void)rdna4::l2_norm_launch(w_l2q, w_l2q, NKH, SS, EPS);
    (void)rdna4::l2_norm_launch(w_l2k, w_l2k, NKH, SS, EPS);
  };
  auto l2_f = [&] {
    cp(w_l2q, m_l2q, bK);
    cp(w_l2k, m_l2k, bK);
    fz_l2qk_kernel<<<2 * NKH, 256>>>(w_l2q, w_l2k, EPS);
  };

  // ---- 3in1 ----
  auto t3_u = [&] {
    cp(w_v, m_v, bDIN);
    cp(w_z, m_z, bDIN);
    (void)rdna4::rms_norm_launch(w_v, m_w128, w_v, NVH, SS, EPS);
    (void)rdna4::silu_gate_launch(w_z, w_v, w_v, DIN);
    rdna4::quantize_q8_1_kernel<<<qg_din, 128>>>(w_v, w_q192, nb_din);
  };
  auto t3_f = [&] {
    cp(w_v, m_v, bDIN);
    cp(w_z, m_z, bDIN);
    fz_norm_silu_quant_kernel<<<NVH / 2, 256>>>(w_v, w_z, m_w128, w_q192, EPS);
  };

  // ---- COMP attn7 = F3a + F5 ----
  auto ca_u = [&] {
    cp(w_ks, m_ks, bKV);
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    (void)rdna4::deinterleave_q_gate_launch(m_yq, w_q, w_g, NH, HD);
    (void)rdna4::rms_norm_launch(w_q, m_wq, w_q, NH, HD, EPS);
    (void)rdna4::rms_norm_launch(w_ks, m_wk, w_ks, NKV, HD, EPS);
    (void)rdna4::rope_launch(w_Q, 1, NH, HD, NROT, FREQ, dpos);
    (void)rdna4::rope_launch(w_K, 1, NKV, HD, NROT, FREQ, dpos);
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, w_K, w_Kd, KVW);
    (void)rdna4::kv_store_row_launch(rdna4::KvType::F16, m_Vs, w_Vd, KVW);
  };
  auto ca_f = [&] {
    cp(w_ks, m_ks, bKV);
    cp(w_Q, m_Q, bQ);
    cp(w_K, m_K, bKV);
    fz_deint_qknorm_kernel<<<NH + NKV, 256>>>(m_yq, w_q, w_g, w_ks, m_wq, m_wk, EPS);
    fz_prep_kv4_f16_kernel<<<1, 1024>>>(w_Q, w_K, m_Vs, w_Kd, w_Vd, hpos);
  };

  // ---- COMP gdn5 = l2fuse + 3in1 ----
  auto cg_u = [&] {
    cp(w_l2q, m_l2q, bK);
    cp(w_l2k, m_l2k, bK);
    cp(w_v, m_v, bDIN);
    cp(w_z, m_z, bDIN);
    (void)rdna4::l2_norm_launch(w_l2q, w_l2q, NKH, SS, EPS);
    (void)rdna4::l2_norm_launch(w_l2k, w_l2k, NKH, SS, EPS);
    (void)rdna4::rms_norm_launch(w_v, m_w128, w_v, NVH, SS, EPS);
    (void)rdna4::silu_gate_launch(w_z, w_v, w_v, DIN);
    rdna4::quantize_q8_1_kernel<<<qg_din, 128>>>(w_v, w_q192, nb_din);
  };
  auto cg_f = [&] {
    cp(w_l2q, m_l2q, bK);
    cp(w_l2k, m_l2k, bK);
    cp(w_v, m_v, bDIN);
    cp(w_z, m_z, bDIN);
    fz_l2qk_kernel<<<2 * NKH, 256>>>(w_l2q, w_l2k, EPS);
    fz_norm_silu_quant_kernel<<<NVH / 2, 256>>>(w_v, w_z, m_w128, w_q192, EPS);
  };

  identity("F4 add+norm", f4_u, f4_f, {{w_x, bE}, {w_xn, bE}});
  identity("F3a deint+q+k", f3a_u, f3a_f, {{w_q, bQ}, {w_g, bQ}, {w_ks, bKV}});
  identity("F5a ropeQ+ropeK", f5a_u, f5a_f, {{w_Q, bQ}, {w_K, bKV}});
  identity("F1 kvK+kvV f16", f1_u, f1_f, {{w_Kd, bKV / 2}, {w_Vd, bKV / 2}});
  identity("F5 prep+kv 4in1", f5_u, f5_f, {{w_Q, bQ}, {w_K, bKV}, {w_Kd, bKV / 2}, {w_Vd, bKV / 2}});
  identity("F3b silu+quant FFN", f3b_u, f3b_f, {{w_fa, bFF}, {w_q544, bQ544}});
  identity("F3b silu+quant GDN", f3bg_u, f3bg_f, {{w_z, bDIN}, {w_v, bDIN}, {w_q192, bQ192}});
  identity("F3c sig+quant", f3c_u, f3c_f, {{w_g, bDIN}, {w_y, bDIN}, {w_q192, bQ192}});
  identity("l2q+l2k", l2_u, l2_f, {{w_l2q, bK}, {w_l2k, bK}});
  identity("3in1 nrm+silu+q", t3_u, t3_f, {{w_v, bDIN}, {w_z, bDIN}, {w_q192, bQ192}});
  identity("COMP attn7", ca_u, ca_f,
           {{w_q, bQ}, {w_g, bQ}, {w_ks, bKV}, {w_Q, bQ}, {w_K, bKV}, {w_Kd, bKV / 2}, {w_Vd, bKV / 2}});
  identity("COMP gdn5", cg_u, cg_f,
           {{w_l2q, bK}, {w_l2k, bK}, {w_v, bDIN}, {w_z, bDIN}, {w_q192, bQ192}});

  out("\ntiming: pair vs fused, us per launch-group (chain=%d):\n", chain);
  report("F4 add+norm", f4_u, f4_f, N_ALL);
  report("F3a deint+qk", f3a_u, f3a_f, N_ATTN);
  report("F5a ropeQ+K", f5a_u, f5a_f, N_ATTN);
  report("F1 kvK+kvV", f1_u, f1_f, N_ATTN);
  report("F5 prep+kv4", f5_u, f5_f, N_ATTN);
  report("F3b silu+q FFN", f3b_u, f3b_f, N_ATTN);
  report("F3b silu+q GDN", f3bg_u, f3bg_f, N_GDN);  // info: covered by 3in1 on GDN layers
  report("F3c sig+q", f3c_u, f3c_f, N_ATTN);
  report("l2q+l2k", l2_u, l2_f, N_GDN);
  report("3in1 nrm+silu+q", t3_u, t3_f, N_GDN);
  report("COMP attn7", ca_u, ca_f, N_ATTN);
  report("COMP gdn5", cg_u, cg_f, N_GDN);

  out("\nship-it: %d  dead: %d  identity-mismatch: %d\n", nship, ndead, ibad);
  if (dump) std::fclose(dump);
  return ibad != 0 ? 1 : 0;
}
