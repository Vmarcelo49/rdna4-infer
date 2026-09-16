// M7 step 0 — where does long-context attention time go, and which fix works?
//
// M6's corrected measurement (bench --start-pos) showed a decode step costs 45 ms
// at 4K, 174 ms at 64K (f16 KV) and 441 ms at 131K, i.e. long context is
// attention-bound, not weight-bound. This bench isolates the attention kernel and
// tests the two candidate fixes for the same problem before rewriting it:
//
//   A. shipping kernel (attn_kernel): grid = n_head, 8 warps splitting KEYS
//      (j = w; j <= t; j += 8), one 5-shuffle reduction + 2 expf per key per warp.
//      GQA: 6 query heads of a kv group each re-read the same K/V rows.
//   B. prototype "gqa" kernel: grid = n_head_kv, 8 warps splitting keys, but each
//      warp keeps ALL 6 query heads of its kv group in registers and walks the
//      same rows for them (6 dots per loaded row). The 6 query heads then share
//      the row through L1 instead of streaming it 6 times, and the FMAs are free
//      (the kernel uses ~1% of the fp32 throughput).
//
// usage: bench-attn-gpu [kvtype] [t...] [--smax N] [--splits a,b,c] [--wpb a,b,c]
//        (kvtype: f16 | q4_0 | q8_0 | f32)
//
// AUTOTUNING (task 6, docs/autotuning-gfx1201.md): the split-count x warps-per-CTA
// table is the search space of the attention kernel, and the engine's policy is
// `attn_splits_for(keys)` (graph.cuh: a fixed kAttnSplitCtasDense/n_head CTAs when
// the KV has no dequant -- f16/f32 -- and keys/kAttnSplitMin capped at
// kAttnMaxSplits for the quantized types) plus `attn_split_wpb(splits)` (attn.cuh:
// 16 warps up to kAttnSplitWpbLimit splits, 8 in between, 16 again from
// kAttnSplitWpbWide). --smax raises the split cap so the sweep can ask whether the
// shipped kAttnMaxSplits=16 is still the optimum at long context, and every `t`
// row prints the best cell, the cell the shipping policy would pick, and the
// same-kernel-twice noise floor that makes the comparison meaningful.
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "rdna4/attn.cuh"
#include "rdna4/kv.h"

namespace {

constexpr int kHeadDim = 256;
constexpr int kHeads = 24;
constexpr int kHeadsKv = 4;
constexpr float kScale = 1.0f / 16.0f;
// Attention layers of the shipped model (qwen35 dense, block_count 65 with the
// MTP block: 16 layers carry a KV cache). Used only to turn a per-layer kernel
// number into the ms/token at stake.
constexpr int kLayers = 16;

void fill_random(std::vector<float> &v, std::size_t n, unsigned seed) {
  v.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    unsigned h = (unsigned)(i * 2654435761u) ^ (seed * 2246822519u);
    h ^= h >> 13;
    h *= 1274126177u;
    v[i] = ((float)(h >> 8) / 8388608.0f) - 1.0f;
  }
}

// Event-pair timer: one pair around `reps` launches, no per-launch sync (the
// pattern the matvec bench uses). A per-cell warm-up is NOT usable here -- the
// per-t benchmark showed the first cell measured after the cache fill running
// 15% slower than the same kernel measured later.
struct EvPair {
  hipEvent_t e0 = nullptr, e1 = nullptr;
  EvPair() {
    (void)hipEventCreate(&e0);
    (void)hipEventCreate(&e1);
  }
  ~EvPair() {
    if (e0) (void)hipEventDestroy(e0);
    if (e1) (void)hipEventDestroy(e1);
  }
  double operator()(const std::function<void()> &fn) {
    hipEventRecord(e0, nullptr);
    fn();
    hipEventRecord(e1, nullptr);
    hipEventSynchronize(e1);
    float ms = 0;
    hipEventElapsedTime(&ms, e0, e1);
    return ms;
  }
};

double time_ms(const std::function<void()> &fn, int reps, double warm_ms) {
  const auto w0 = std::chrono::steady_clock::now();
  do {
    fn();
    hipDeviceSynchronize();
  } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() <
           warm_ms);
  const auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < reps; ++i) fn();
  hipDeviceSynchronize();
  const auto t1 = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(t1 - t0).count() / reps;
}

// ---------------------------------------------------------------------------
// Prototype B: one CTA per KV head; each warp walks a key slice and computes the
// scores of every query head in that kv group for the same row.
// ---------------------------------------------------------------------------
template <rdna4::KvType KT, rdna4::KvType VT>
__global__ void attn_gqa_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                const void *__restrict__ v, float *__restrict__ out, int t,
                                int n_head, int n_head_kv, int head_dim, float dscale) {
  extern __shared__ float smem[];
  const int kvh = blockIdx.x;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = n_head / n_head_kv;   // query heads per kv head (6 here)
  const int dpw = head_dim / 32;        // dims per lane (8 here)

  const std::uint64_t krow = rdna4::kv_row_bytes(KT, head_dim);
  const std::uint64_t vrow = rdna4::kv_row_bytes(VT, head_dim);

  // per-query-head query vector and accumulator, all in registers
  float qv[8][rdna4::kAttnMaxDimsPerLane];
  float acc[8][rdna4::kAttnMaxDimsPerLane];
  float m[8], l[8];
  for (int hh = 0; hh < grp; ++hh) {
    const int h = kvh * grp + hh;
    for (int i = 0; i < dpw; ++i) qv[hh][i] = q[h * head_dim + lane * dpw + i];
    for (int i = 0; i < dpw; ++i) acc[hh][i] = 0.0f;
    m[hh] = -INFINITY;
    l[hh] = 0.0f;
  }

  for (int j = w; j <= t; j += rdna4::kAttnWarpsPerBlock) {
    const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krow;
    const char *vr = (const char *)v + ((std::int64_t)j * n_head_kv + kvh) * vrow;
    // load the row ONCE per lane, reuse it for every query head of the group
    float kk[rdna4::kAttnMaxDimsPerLane];
    float vv[rdna4::kAttnMaxDimsPerLane];
    for (int i = 0; i < dpw; ++i) {
      kk[i] = rdna4::kv_load<KT>(kr, lane * dpw + i);
      vv[i] = rdna4::kv_load<VT>(vr, lane * dpw + i);
    }
    for (int hh = 0; hh < grp; ++hh) {
      float partial = 0.0f;
      for (int i = 0; i < dpw; ++i) partial = fmaf(qv[hh][i], kk[i], partial);
#pragma unroll
      for (int off = 16; off > 0; off >>= 1) partial += __shfl_xor_sync(0xffffffffull, partial, off);
      const float score = partial * dscale;
      if (score > m[hh]) {
        const float corr = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - score);
        l[hh] *= corr;
        for (int i = 0; i < dpw; ++i) acc[hh][i] *= corr;
        m[hh] = score;
      }
      const float p = (m[hh] == -INFINITY) ? 0.0f : expf(score - m[hh]);
      l[hh] += p;
      for (int i = 0; i < dpw; ++i) acc[hh][i] = fmaf(p, vv[i], acc[hh][i]);
    }
  }

  // merge the 8 key slices: one shared slice per (query head)
  __syncthreads();
  for (int hh = 0; hh < grp; ++hh) {
    float *pm = smem + ((std::int64_t)hh * rdna4::kAttnWarpsPerBlock + w) * (2 + head_dim);
    if (lane == 0) {
      pm[0] = m[hh];
      pm[1] = l[hh];
    }
    for (int i = 0; i < dpw; ++i) pm[2 + lane * dpw + i] = acc[hh][i];
  }
  __syncthreads();
  for (int hh = 0; hh < grp; ++hh) {
    const int h = kvh * grp + hh;
    float mm = -INFINITY;
    for (int s = 0; s < rdna4::kAttnWarpsPerBlock; ++s)
      mm = fmaxf(mm, smem[((std::int64_t)hh * rdna4::kAttnWarpsPerBlock + s) * (2 + head_dim)]);
    float ll = 0.0f;
    for (int s = 0; s < rdna4::kAttnWarpsPerBlock; ++s) {
      const float *ps = smem + ((std::int64_t)hh * rdna4::kAttnWarpsPerBlock + s) * (2 + head_dim);
      ll += ps[1] * expf(ps[0] - mm);
    }
    const float inv = (ll > 0.0f) ? 1.0f / ll : 0.0f;
    for (int i = 0; i < dpw; ++i) {
      float a = 0.0f;
      for (int s = 0; s < rdna4::kAttnWarpsPerBlock; ++s) {
        const float *ps = smem + ((std::int64_t)hh * rdna4::kAttnWarpsPerBlock + s) * (2 + head_dim);
        a += ps[2 + lane * dpw + i] * expf(ps[0] - mm);
      }
      out[h * head_dim + lane * dpw + i] = a * inv;
    }
  }
}

template <rdna4::KvType KT, rdna4::KvType VT>
bool launch_gqa(const float *q, const void *k, const void *v, float *out, int t, int n_head,
                int n_head_kv, int head_dim, float scale, hipStream_t stream) {
  const int grp = n_head / n_head_kv;
  const std::size_t smem = (std::size_t)grp * rdna4::kAttnWarpsPerBlock * (2 + (std::size_t)head_dim) *
                           sizeof(float);
  if (smem > 60 * 1024) return false;
  attn_gqa_kernel<KT, VT><<<n_head_kv, rdna4::kAttnWarpsPerBlock * 32, smem, stream>>>(
      q, k, v, out, t, n_head, n_head_kv, head_dim, scale);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// Prototype C (frente longctx): GQA sharing WITH the wide grid that prototype B
// is missing. Grid = (n_head_kv, n_splits) -- the same split-over-keys structure
// the shipping kernel uses, but one CTA serves a whole GQA group instead of one
// query head, so a K/V row is fetched once per group instead of 6 times.
//
// The HG template parameter is the head-chunk: the CTA computes HG query heads of
// its group per pass over its key slice and loops 6/HG passes. It separates the
// two effects the M7 prototype confounded:
//   HG = 1  -> 6 passes, same K/V traffic as the shipping kernel, but grid (4,S)
//   HG = 3  -> 2 passes, half the traffic, same grid
//   HG = 6  -> 1 pass,  1/6 of the traffic, same grid
// Comparing HG=1 with HG=6 at the same grid says how much of the win is the
// traffic and how much is the parallelism; comparing HG=6 with the shipping
// kernel says whether the wide grid is enough to pay for the lost CTAs.
//
// Registers are the reason HG is a knob: q and acc are HG*8 floats per lane each
// (head_dim 256 -> dpw 8). HG=6 costs ~110 registers per lane, HG=1 ~30.
// dpw != 8 is refused: this is a prototype for the model's head_dim, not a
// general kernel (the shipped one stays general).
//
// The partials are written into the SAME [h][s] layout the shipping split kernel
// uses, so attn_merge_kernel combines them unchanged.
// ---------------------------------------------------------------------------
template <rdna4::KvType KT, rdna4::KvType VT, int HG, int WPB = rdna4::kAttnWarpsPerBlock>
__global__ void attn_gqa_split_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                     const void *__restrict__ v, float *__restrict__ partial, int t,
                                     int n_head, int n_head_kv, int head_dim, float dscale,
                                     int n_splits) {
  extern __shared__ float smem[];
  const int kvh = blockIdx.x;
  const int s = blockIdx.y;
  const int w = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int grp = n_head / n_head_kv;
  const int dpw = head_dim / 32;
  if (dpw != 8) return;  // prototype: head_dim 256 only

  const std::uint64_t krow = rdna4::kv_row_bytes(KT, head_dim);
  const std::uint64_t vrow = rdna4::kv_row_bytes(VT, head_dim);
  const int step = WPB * n_splits;
  float *pm = smem + (std::int64_t)w * (2 + head_dim);
  __shared__ float wts[WPB];
  __shared__ float cmax;

  for (int c = 0; c * HG < grp; ++c) {
    const int h0 = kvh * grp + c * HG;
    const int nh = (grp - c * HG < HG) ? (grp - c * HG) : HG;
    float qv[HG][8];
    float acc[HG][8];
    float m[HG], l[HG];
    for (int hh = 0; hh < nh; ++hh) {
      const float *qp = q + (std::int64_t)(h0 + hh) * head_dim + lane * dpw;
      for (int i = 0; i < dpw; ++i) qv[hh][i] = qp[i];
      for (int i = 0; i < dpw; ++i) acc[hh][i] = 0.0f;
      m[hh] = -INFINITY;
      l[hh] = 0.0f;
    }

    for (int j = w + WPB * s; j <= t; j += step) {
      const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krow;
      const char *vr = (const char *)v + ((std::int64_t)j * n_head_kv + kvh) * vrow;
      float kk[8], vv[8];
      rdna4::kv_load8<KT>(kr, lane, kk);
      rdna4::kv_load8<VT>(vr, lane, vv);
      for (int hh = 0; hh < nh; ++hh) {
        float dot = 0.0f;
        for (int i = 0; i < dpw; ++i) dot = fmaf(qv[hh][i], kk[i], dot);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
          dot += __shfl_xor_sync(0xffffffffull, dot, off);
        const float score = dot * dscale;
        if (score > m[hh]) {
          const float corr = (m[hh] == -INFINITY) ? 0.0f : expf(m[hh] - score);
          l[hh] *= corr;
          for (int i = 0; i < dpw; ++i) acc[hh][i] *= corr;
          m[hh] = score;
        }
        const float p = (m[hh] == -INFINITY) ? 0.0f : expf(score - m[hh]);
        l[hh] += p;
        for (int i = 0; i < dpw; ++i) acc[hh][i] = fmaf(p, vv[i], acc[hh][i]);
      }
    }

    // merge this CTA's WPB warp slices once per head, reusing the same smem slot
    // (HG heads sequentially: HG barriers per chunk, WPB*(2+head_dim) floats).
    for (int hh = 0; hh < nh; ++hh) {
      __syncthreads();
      if (lane == 0) {
        pm[0] = m[hh];
        pm[1] = l[hh];
      }
      for (int i = 0; i < dpw; ++i) pm[2 + lane * dpw + i] = acc[hh][i];
      __syncthreads();
      if (threadIdx.x == 0) {
        float mm = -INFINITY;
        for (int i = 0; i < WPB; ++i) mm = fmaxf(mm, smem[(std::int64_t)i * (2 + head_dim)]);
        cmax = mm;
      }
      __syncthreads();
      if (threadIdx.x < WPB)
        wts[threadIdx.x] = (cmax == -INFINITY)
                               ? 0.0f
                               : expf(smem[(std::int64_t)threadIdx.x * (2 + head_dim)] - cmax);
      __syncthreads();
      float *out = partial + ((std::int64_t)(h0 + hh) * n_splits + s) * (2 + head_dim);
      if (threadIdx.x < 32) {
        float lsum = 0.0f;
        if (threadIdx.x < WPB)
          lsum = wts[threadIdx.x] * smem[(std::int64_t)threadIdx.x * (2 + head_dim) + 1];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) lsum += __shfl_xor_sync(0xffffffffull, lsum, off);
        if (threadIdx.x == 0) {
          out[0] = cmax;
          out[1] = lsum;
        }
      }
      for (int d = (int)threadIdx.x; d < head_dim; d += (int)blockDim.x) {
        float a = 0.0f;
        for (int i = 0; i < WPB; ++i) a += smem[(std::int64_t)i * (2 + head_dim) + 2 + d] * wts[i];
        out[2 + d] = a;
      }
    }
  }
}

template <rdna4::KvType KT, rdna4::KvType VT, int HG>
bool launch_gqa_split_hg(const float *q, const void *k, const void *v, float *out, float *partial,
                         int t, int n_head, int n_head_kv, int head_dim, float scale, int n_splits,
                         hipStream_t stream) {
  if (head_dim != 256 || n_splits < 1) return false;
  const int threads = rdna4::kAttnWarpsPerBlock * 32;
  const std::size_t smem =
      (std::size_t)rdna4::kAttnWarpsPerBlock * (2 + (std::size_t)head_dim) * sizeof(float);
  dim3 grid((unsigned)n_head_kv, (unsigned)n_splits);
  attn_gqa_split_kernel<KT, VT, HG><<<grid, threads, smem, stream>>>(
      q, k, v, partial, t, n_head, n_head_kv, head_dim, scale, n_splits);
  if (hipGetLastError() != hipSuccess) return false;
  rdna4::attn_merge_kernel<<<n_head, 32, 0, stream>>>(partial, out, n_head, head_dim, n_splits);
  return hipGetLastError() == hipSuccess;
}

template <rdna4::KvType KT, rdna4::KvType VT>
bool launch_gqa_split(const float *q, const void *k, const void *v, float *out, float *partial, int t,
                      int n_head, int n_head_kv, int head_dim, float scale, int n_splits, int hg,
                      hipStream_t stream) {
  switch (hg) {
    case 1: return launch_gqa_split_hg<KT, VT, 1>(q, k, v, out, partial, t, n_head, n_head_kv,
                                                  head_dim, scale, n_splits, stream);
    case 2: return launch_gqa_split_hg<KT, VT, 2>(q, k, v, out, partial, t, n_head, n_head_kv,
                                                  head_dim, scale, n_splits, stream);
    case 3: return launch_gqa_split_hg<KT, VT, 3>(q, k, v, out, partial, t, n_head, n_head_kv,
                                                  head_dim, scale, n_splits, stream);
    case 6: return launch_gqa_split_hg<KT, VT, 6>(q, k, v, out, partial, t, n_head, n_head_kv,
                                                  head_dim, scale, n_splits, stream);
    default: return false;
  }
}

// Prototype D / E1 (tile de linhas): one CTA per (kv head, split), 6 warps, one
// warp per query head of the group. Each key's K/V row is decoded ONCE into LDS
// (warp 0 via kv_load8) and shared by all 6 warps; every warp keeps ONLY its own
// head state (q:8, acc:8, m, l) -- the partitioned accumulator. This is the
// structural difference vs prototype C: C splits KEYS across warps, so each warp
// carries all HG heads' state (18x replication at HG=6 -> 68 VGPR + 400 B
// spill); D splits HEADS across warps (~30 VGPR, no spill expected).
// Same [h][s] partials layout + attn_merge_kernel as the split kernels, same
// per-head op order as attn_kernel (per-lane fmaf chain, same shuffle tree),
// so the numerics should be BIT-EXACT vs the unsplit kernel, not just close.
// Bench-only (docs/plano-ninfer-pendente.md E1). Pre-registered targets: >=1.02x
// vs the ship-policy split cell, i.e. 4K <= 0.0709, 16K <= 0.2381 ms/launch.
template <rdna4::KvType KT, rdna4::KvType VT>
__global__ void attn_gqa_rowtile_kernel(const float *__restrict__ q, const void *__restrict__ k,
                                        const void *__restrict__ v, float *__restrict__ partial,
                                        int t, int n_head, int n_head_kv, int head_dim,
                                        float dscale, int n_splits) {
  const int kvh = blockIdx.x;
  const int s = blockIdx.y;
  const int w = threadIdx.x >> 5;  // 0..5: query head inside the group
  const int lane = threadIdx.x & 31;
  const int grp = n_head / n_head_kv;
  const int dpw = head_dim / 32;
  if (dpw != 8 || grp != 6) return;  // prototype: this model's shape only
  __shared__ float krow[256];
  __shared__ float vrow[256];

  const int h = kvh * grp + w;
  float qv[8], acc[8];
  {
    const float *qp = q + (std::int64_t)h * head_dim + lane * dpw;
    for (int i = 0; i < dpw; ++i) {
      qv[i] = qp[i];
      acc[i] = 0.0f;
    }
  }
  float m = -INFINITY, l = 0.0f;
  const std::uint64_t krowb = rdna4::kv_row_bytes(KT, head_dim);
  const std::uint64_t vrowb = rdna4::kv_row_bytes(VT, head_dim);

  for (int j = s; j <= t; j += n_splits) {
    // warp 0 decodes this key's rows once for the whole group
    if (w == 0) {
      const char *kr = (const char *)k + ((std::int64_t)j * n_head_kv + kvh) * krowb;
      const char *vr = (const char *)v + ((std::int64_t)j * n_head_kv + kvh) * vrowb;
      float kk[8], vv[8];
      rdna4::kv_load8<KT>(kr, lane, kk);
      rdna4::kv_load8<VT>(vr, lane, vv);
      for (int i = 0; i < dpw; ++i) {
        krow[lane * dpw + i] = kk[i];
        vrow[lane * dpw + i] = vv[i];
      }
    }
    __syncthreads();
    float dot = 0.0f;
    for (int i = 0; i < dpw; ++i) dot = fmaf(qv[i], krow[lane * dpw + i], dot);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffull, dot, off);
    const float score = dot * dscale;
    if (score > m) {
      const float corr = (m == -INFINITY) ? 0.0f : expf(m - score);
      l *= corr;
      for (int i = 0; i < dpw; ++i) acc[i] *= corr;
      m = score;
    }
    const float p = (m == -INFINITY) ? 0.0f : expf(score - m);
    l += p;
    for (int i = 0; i < dpw; ++i) acc[i] = fmaf(p, vrow[lane * dpw + i], acc[i]);
  }

  // disjoint heads per warp: no barrier needed, same [h][s] partials as C
  float *out = partial + ((std::int64_t)h * n_splits + s) * (2 + head_dim);
  if (lane == 0) {
    out[0] = m;
    out[1] = l;
  }
  for (int i = 0; i < dpw; ++i) out[2 + lane * dpw + i] = acc[i];
}

template <rdna4::KvType KT, rdna4::KvType VT>
bool launch_gqa_rowtile(const float *q, const void *k, const void *v, float *out, float *partial,
                        int t, int n_head, int n_head_kv, int head_dim, float scale, int n_splits,
                        hipStream_t stream) {
  if (head_dim != 256 || n_head / n_head_kv != 6 || n_splits < 1) return false;
  constexpr int kRowtileWarps = 6;
  dim3 grid((unsigned)n_head_kv, (unsigned)n_splits);
  attn_gqa_rowtile_kernel<KT, VT>
      <<<grid, kRowtileWarps * 32, 2 * 256 * sizeof(float), stream>>>(
          q, k, v, partial, t, n_head, n_head_kv, head_dim, scale, n_splits);
  if (hipGetLastError() != hipSuccess) return false;
  rdna4::attn_merge_kernel<<<n_head, 32, 0, stream>>>(partial, out, n_head, head_dim, n_splits);
  return hipGetLastError() == hipSuccess;
}

// Kernel resource report: registers and spill bytes decide whether a prototype is
// slow because of its algorithm or because of its occupancy. (The M7 prototype's
// 8-12x was attributed to the narrow grid without this number; measure it.)
template <typename F>
void report_resources(const char *name, F *fn) {  hipFuncAttributes a{};
  if (hipFuncGetAttributes(&a, (const void *)fn) != hipSuccess) {
    std::printf("  %-22s (attributes unavailable)\n", name);
    return;
  }
  std::printf("  %-22s regs/lane %3d  spill %4zu B  maxThreads %4d  shared %zu B\n", name,
              a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock, a.sharedSizeBytes);
}


// Kill-test (pendente §3): ONE fused GQA variant (HG) vs production at ONE t,
// interleaved rounds, best+mean+wins. No sweep: a decision duel, nothing else.
template <rdna4::KvType KT, rdna4::KvType VT>
static bool gqa_duel_launch(const float *q, const void *k, const void *v, float *out, float *partial,
                            int t, int hg, int n_splits, hipStream_t stream) {
  return launch_gqa_split<KT, VT>(q, k, v, out, partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                                  n_splits, hg, stream);
}

static int run_gqa_duel(rdna4::KvType kt, int t, int hg, int rounds) {
  const std::uint64_t rowb = rdna4::kv_row_bytes(kt, kHeadDim);
  const std::size_t bytes = (std::size_t)(t + 1) * kHeadsKv * rowb;
  float *d_q = nullptr, *d_out = nullptr;
  void *d_k = nullptr, *d_v = nullptr;
  std::vector<float> qh;
  fill_random(qh, (std::size_t)kHeads * kHeadDim, 3u);
  hipMalloc(&d_q, qh.size() * sizeof(float));
  hipMalloc(&d_out, (std::size_t)kHeads * kHeadDim * sizeof(float));
  hipMemcpy(d_q, qh.data(), qh.size() * sizeof(float), hipMemcpyHostToDevice);
  if (hipMalloc(&d_k, bytes) != hipSuccess || hipMalloc(&d_v, bytes) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed for the %zu-byte cache\n", bytes);
    return 1;
  }
  const std::int64_t rows = (std::int64_t)(t + 1) * kHeadsKv;
  if (!rdna4::kv_fill_launch(kt, d_k, rows, kHeadDim, 11u) ||
      !rdna4::kv_fill_launch(kt, d_v, rows, kHeadDim, 22u)) {
    std::fprintf(stderr, "kv_fill_launch failed: %s\n", hipGetErrorString(hipGetLastError()));
    return 1;
  }
  const int keys = t + 1;
  const bool kv_dense = (kt == rdna4::KvType::F16 || kt == rdna4::KvType::F32);
  int n_splits;
  if (keys < rdna4::tuned::kAttnSplitMin) n_splits = 1;
  else if (kv_dense) n_splits = (rdna4::tuned::kAttnSplitCtasDense + kHeads - 1) / kHeads;
  else n_splits = keys / rdna4::tuned::kAttnSplitMin;
  if (n_splits < 1) n_splits = 1;
  if (n_splits > rdna4::tuned::kAttnMaxSplits) n_splits = rdna4::tuned::kAttnMaxSplits;
  const int wpb = rdna4::attn_split_wpb(n_splits);
  float *d_partial = nullptr;
  if (hipMalloc(&d_partial, rdna4::attn_partial_bytes(kHeads, kHeadDim, n_splits)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc partial failed\n");
    return 1;
  }
  auto run_ref = [&] {
    if (n_splits <= 1)
      (void)rdna4::attn_launch(d_q, d_k, d_v, d_out, t, kHeads, kHeadsKv, kHeadDim, kScale, kt, kt,
                               nullptr);
    else
      (void)rdna4::attn_launch_split_wpb(d_q, d_k, d_v, d_out, d_partial, t, kHeads, kHeadsKv,
                                         kHeadDim, kScale, kt, kt, n_splits, wpb, nullptr);
  };
  auto run_cand = [&] {
    if (kt == rdna4::KvType::Q4_0)
      (void)gqa_duel_launch<rdna4::KvType::Q4_0, rdna4::KvType::Q4_0>(
          d_q, d_k, d_v, d_out, d_partial, t, hg, n_splits, nullptr);
    else if (kt == rdna4::KvType::Q8_0)
      (void)gqa_duel_launch<rdna4::KvType::Q8_0, rdna4::KvType::Q8_0>(
          d_q, d_k, d_v, d_out, d_partial, t, hg, n_splits, nullptr);
    else if (kt == rdna4::KvType::F32)
      (void)gqa_duel_launch<rdna4::KvType::F32, rdna4::KvType::F32>(
          d_q, d_k, d_v, d_out, d_partial, t, hg, n_splits, nullptr);
    else
      (void)gqa_duel_launch<rdna4::KvType::F16, rdna4::KvType::F16>(
          d_q, d_k, d_v, d_out, d_partial, t, hg, n_splits, nullptr);
  };
  {  // warmup >= 300 ms, both configs
    const auto w0 = std::chrono::steady_clock::now();
    do {
      run_ref();
      run_cand();
      hipDeviceSynchronize();
    } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0)
                 .count() < 300.0);
  }
  constexpr int kReps = 3;
  EvPair ev;
  double best_ref = -1, best_cand = -1, sum_ref = 0, sum_cand = 0;
  int wins = 0;
  for (int r = 0; r < rounds; ++r) {
    const bool fwd = (r % 2) == 0;
    double got_ref = -1, got_cand = -1;
    for (int k = 0; k < 2; ++k) {
      const bool is_ref = fwd ? (k == 0) : (k == 1);
      const double ms = ev([&] {
                        for (int j = 0; j < kReps; ++j) (is_ref ? run_ref() : run_cand());
                      }) / kReps;
      if (is_ref) got_ref = ms;
      else got_cand = ms;
    }
    if (best_ref < 0 || got_ref < best_ref) best_ref = got_ref;
    if (best_cand < 0 || got_cand < best_cand) best_cand = got_cand;
    sum_ref += got_ref;
    sum_cand += got_cand;
    if (got_cand < got_ref) ++wins;
  }
  // numerics vs unsplit reference
  std::vector<float> a((std::size_t)kHeads * kHeadDim), b((std::size_t)kHeads * kHeadDim);
  (void)rdna4::attn_launch(d_q, d_k, d_v, d_out, t, kHeads, kHeadsKv, kHeadDim, kScale, kt, kt,
                           nullptr);
  hipDeviceSynchronize();
  hipMemcpy(a.data(), d_out, a.size() * sizeof(float), hipMemcpyDeviceToHost);
  run_cand();
  hipDeviceSynchronize();
  hipMemcpy(b.data(), d_out, b.size() * sizeof(float), hipMemcpyDeviceToHost);
  double n2 = 0, d2 = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    n2 += d * d;
    d2 += (double)a[i] * a[i] + (double)b[i] * b[i];
  }
  std::printf("GQA-DUEL t=%d keys=%d splits=%d (wpb %d) HG=%d, %d interleaved rounds x %d reps:\n",
              t, keys, n_splits, wpb, hg, rounds, kReps);
  std::printf("  production: best %.4f ms  mean %.4f ms\n", best_ref, sum_ref / rounds);
  std::printf("  HG=%d fused: best %.4f ms  mean %.4f ms  rel-L2 %.1e\n", hg, best_cand,
              sum_cand / rounds, std::sqrt(n2 / d2));
  std::printf("  ratio best %.4fx  mean %.4fx  wins %d/%d  (kill bar: <1.02x => DEAD)\n",
              best_ref / best_cand, (sum_ref / rounds) / (sum_cand / rounds), wins, rounds);
  std::printf("  x%d layers: %.3f ms/token at stake\n", kLayers, (best_ref - best_cand) * kLayers);
  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_partial);
  return 0;
}

// ---------------------------------------------------------------------------
// H6 step 1 — merge audit (hipfire PR #760 pattern).
//
// Their stable form (CK-style, new in #760): one LSE plane + normalized Oacc,
//   m = max_s LSE_s;  w_s = isfinite(LSE_s) ? exp(LSE_s - m) : 0
//   out = sum_s(w_s * Oacc_s) / sum_s(w_s)      (zeros if no finite split)
// Our shipped form (attn_merge_kernel, plus the CTA-local WPB reduce inside
// attn_split_kernel — same statements in attn_merge_batch_kernel):
//   mm = max_s m_s;  w'_s = exp(m_s - mm)
//   out = sum_s(w'_s * acc_s) / sum_s(w'_s * l_s)   (zeros if mm == -inf)
// The two are the same max-subtracted reduction up to the l-folding
// (LSE_s = m_s + log l_s in exact arithmetic). What killed THEIR first attempt
// was not the formula but the partial O-write: lanes ml>=8 of the multi-query
// group wrote partial O to the wrong record offset, i.e. SOME OUTPUT DIMS
// WERE NEVER WRITTEN and the merge read stale VRAM. The analogue here is the
// [h][s] partial layout + the strided O-write
// `for (d = threadIdx.x; d < head_dim; d += blockDim.x)` (split kernel) and
// the 32-lane dpw merge: with WPB=16/32 most threads write NO dim, with
// head_dim > blockDim some write several, and an over-read past n_splits
// would silently consume garbage. Coverage is checked with NaN poisoning:
// out AND the unused partial tail are pre-filled with NaN, so any unwritten
// dim or any over-read survives as NaN in the merge output.
//
// Shapes: tail/ragged (n_splits not dividing the keys), sentinel (splits with
// zero keys: m=-inf, l=0 — the expf(-inf - -inf) = NaN trap; and ALL splits
// empty, which must emit exact zeros), extremal LSE spread (+/-20, so only
// max-subtraction avoids overflow), and the ml>=8-lane class as splits>=8
// with WPB 8/16/32. Gate: all finite, zero NaN, rel-L2 vs host f64 <= 2e-6
// (fp32 rounding over <= 16 terms), all-empty bit-exact zeros.
// ---------------------------------------------------------------------------
static int run_merge_audit(rdna4::KvType kt, const char *tname) {
  int failures = 0, cases = 0;
  auto check = [&](const char *name, bool ok, double rel, double maxd) {
    ++cases;
    char detail[128];
    std::snprintf(detail, sizeof(detail), "rel-L2 %.2e max|d| %.2e", rel, maxd);
    std::printf("  [%s] %-40s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok) ++failures;
  };
  std::printf("H6 merge audit vs max-LSE-stable form (hipfire PR #760 pattern)\n");
  std::printf("  ours: mm=max m_s, out=sum(acc_s*e^(m_s-mm))/sum(l_s*e^(m_s-mm)); "
              "theirs: m=max LSE_s, out=sum(Oacc_s*e^(LSE_s-m))/sum(e^(LSE_s-m))\n");

  // ---- Part A: attn_merge_kernel vs host f64 on synthetic partials ----
  const int n_head = 4, head_dim = 256, smax = 16;
  float *d_partial = nullptr, *d_out = nullptr;
  if (hipMalloc(&d_partial, rdna4::attn_partial_bytes(n_head, head_dim, smax)) != hipSuccess ||
      hipMalloc(&d_out, (std::size_t)n_head * head_dim * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "merge-audit: hipMalloc failed\n");
    return 1;
  }
  std::vector<float> h_partial((std::size_t)n_head * smax * (2 + head_dim));
  std::vector<float> h_out((std::size_t)n_head * head_dim);
  unsigned rng = 0x243F6A88u;
  auto rnd = [&]() {
    rng = rng * 1664525u + 1013904223u;
    return (float)(rng >> 8) * (1.0f / 8388608.0f) - 1.0f;
  };
  const int splits_a[] = {1, 2, 4, 8, 16};
  const char *modes_a[] = {"finite", "tail-empty", "all-empty", "extremal"};
  for (int ns : splits_a) {
    for (const char *mode : modes_a) {
      if (ns == 1 && std::strcmp(mode, "tail-empty") == 0) continue;  // degenerate: == all-empty
      for (int h = 0; h < n_head; ++h) {
        for (int s = 0; s < smax; ++s) {
          float *rec = h_partial.data() + ((std::size_t)h * smax + s) * (2 + head_dim);
          const bool active = s < ns;
          const bool empty =
              active && (std::strcmp(mode, "all-empty") == 0 ||
                         (std::strcmp(mode, "tail-empty") == 0 && s >= ns - ns / 2));
          if (!active) {
            for (int d = 0; d < 2 + head_dim; ++d) rec[d] = NAN;  // over-read tripwire
            continue;
          }
          if (empty) {  // production-faithful: an empty CTA writes (m=-inf, l=0, acc=0)
            rec[0] = -INFINITY;
            rec[1] = 0.0f;
            for (int d = 0; d < head_dim; ++d) rec[2 + d] = 0.0f;
            continue;
          }
          rec[0] = (std::strcmp(mode, "extremal") == 0) ? ((s % 2) ? 20.0f + rnd() : -20.0f + rnd())
                                                        : 4.0f * rnd();
          rec[1] = 1.0f + std::fabs(rnd()) * 10.0f;
          for (int d = 0; d < head_dim; ++d) rec[2 + d] = 2.0f * rnd();
        }
      }
      hipMemcpy(d_partial, h_partial.data(), h_partial.size() * sizeof(float),
                hipMemcpyHostToDevice);
      hipMemset(d_out, 0xFF, h_out.size() * sizeof(float));  // NaN: unwritten dims survive
      // The merge addresses partials with stride n_splits ([h][ns]), exactly
      // like production (attn_partial_bytes(n_head, head_dim, n_splits)), so
      // compact the active records; the NaN tail stays HOST-side as the
      // over-read tripwire only if the kernel strayed past ns (it cannot be
      // placed device-side at stride 16: the kernel would read it as data).
      std::vector<float> compact((std::size_t)n_head * ns * (2 + head_dim));
      for (int h = 0; h < n_head; ++h)
        for (int s = 0; s < ns; ++s) {
          const float *src = h_partial.data() + ((std::size_t)h * smax + s) * (2 + head_dim);
          float *dst = compact.data() + ((std::size_t)h * ns + s) * (2 + head_dim);
          for (int d = 0; d < 2 + head_dim; ++d) dst[d] = src[d];
        }
      hipMemcpy(d_partial, compact.data(), compact.size() * sizeof(float), hipMemcpyHostToDevice);
      // Over-read tripwire: poison the slack past the [h][ns] image (only the
      // last head's tail is unambiguously out-of-bounds, but any stray read
      // past ns lands either here or in a neighbour head's records and shows
      // up as NaN/rel failure downstream).
      {
        const std::size_t cbytes = compact.size() * sizeof(float);
        const std::size_t tbytes =
            rdna4::attn_partial_bytes(n_head, head_dim, smax) - cbytes;
        if (tbytes > 0)
          hipMemset((char *)d_partial + cbytes, 0xFF, tbytes);
      }
      rdna4::attn_merge_kernel<<<n_head, 32, 0, nullptr>>>(d_partial, d_out, n_head, head_dim,
                                                           ns);
      hipDeviceSynchronize();
      if (hipGetLastError() != hipSuccess) {
        char nm[96];
        std::snprintf(nm, sizeof(nm), "A S=%d %s (launch)", ns, mode);
        check(nm, false, -1.0, -1.0);
        continue;
      }
      hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), hipMemcpyDeviceToHost);
      // host f64 reference over the ACTIVE splits only
      bool ok = true;
      double worst_rel = 0, worst_maxd = 0;
      for (int h = 0; h < n_head && ok; ++h) {
        double mm = -INFINITY;
        for (int s = 0; s < ns; ++s) {
          const float *rec = h_partial.data() + ((std::size_t)h * smax + s) * (2 + head_dim);
          mm = std::fmax(mm, (double)rec[0]);
        }
        const float *got = h_out.data() + (std::size_t)h * head_dim;
        for (int d = 0; d < head_dim; ++d)
          if (!std::isfinite(got[d])) ok = false;  // NaN = unwritten dim / over-read
        if (mm == -INFINITY) {
          for (int d = 0; d < head_dim; ++d) {
            worst_maxd = std::fmax(worst_maxd, std::fabs((double)got[d]));
            if (got[d] != 0.0f) ok = false;
          }
          continue;
        }
        double ll = 0;
        for (int s = 0; s < ns; ++s) {
          const float *rec = h_partial.data() + ((std::size_t)h * smax + s) * (2 + head_dim);
          if (rec[0] == -INFINITY) continue;
          ll += (double)rec[1] * std::exp((double)rec[0] - mm);
        }
        double num = 0, den = 0, md = 0;
        for (int d = 0; d < head_dim; ++d) {
          double a = 0;
          for (int s = 0; s < ns; ++s) {
            const float *rec = h_partial.data() + ((std::size_t)h * smax + s) * (2 + head_dim);
            if (rec[0] == -INFINITY) continue;
            a += (double)rec[2 + d] * std::exp((double)rec[0] - mm);
          }
          const double ref = (ll > 0) ? a / ll : 0.0;
          const double df = (double)got[d] - ref;
          num += df * df;
          den += ref * ref + (double)got[d] * got[d];
          md = std::fmax(md, std::fabs(df));
        }
        worst_rel = std::fmax(worst_rel, std::sqrt(num / den));
        worst_maxd = std::fmax(worst_maxd, md);
        if (worst_rel > 2e-6) ok = false;
      }
      char nm[96];
      std::snprintf(nm, sizeof(nm), "A merge-S=%d %s vs f64", ns, mode);
      check(nm, ok, worst_rel, worst_maxd);
    }
  }

  // ---- Part B: split-vs-unsplit on ragged/tail/sentinel key counts ----
  // (forced splits even where the engine would run unsplit < 512 keys: the
  // merge must be right at every shape, not just the admitted ones.)
  std::vector<std::pair<rdna4::KvType, const char *>> kb{{rdna4::KvType::F32, "f32"}};
  if (kt != rdna4::KvType::F32) kb.push_back({kt, tname});
  const int keys_b[] = {1, 2, 3, 5, 7, 8, 9, 15, 16, 17, 24, 25, 31, 33,
                        100, 511, 512, 513, 520, 1023, 1025, 2048, 2050};
  const int splits_b[] = {2, 4, 8, 16};
  const int maxkeys = 2050;
  float *b_q = nullptr, *b_o1 = nullptr, *b_o2 = nullptr;
  void *b_k = nullptr, *b_v = nullptr;
  float *b_partial = nullptr;
  {
    std::uint64_t maxb = 0;
    for (auto &kv : kb) maxb = std::max<std::uint64_t>(maxb, rdna4::kv_row_bytes(kv.first, 256));
    const std::size_t cb = (std::size_t)maxkeys * kHeadsKv * maxb;
    if (hipMalloc(&b_q, (std::size_t)kHeads * 256 * sizeof(float)) != hipSuccess ||
        hipMalloc(&b_o1, (std::size_t)kHeads * 256 * sizeof(float)) != hipSuccess ||
        hipMalloc(&b_o2, (std::size_t)kHeads * 256 * sizeof(float)) != hipSuccess ||
        hipMalloc(&b_k, cb) != hipSuccess || hipMalloc(&b_v, cb) != hipSuccess ||
        hipMalloc(&b_partial, rdna4::attn_partial_bytes(kHeads, 256, 16)) != hipSuccess) {
      std::fprintf(stderr, "merge-audit: hipMalloc (part B) failed\n");
      return 1;
    }
  }
  std::vector<float> qh;
  fill_random(qh, (std::size_t)kHeads * 256, 3u);
  hipMemcpy(b_q, qh.data(), qh.size() * sizeof(float), hipMemcpyHostToDevice);
  std::vector<float> o1((std::size_t)kHeads * 256), o2((std::size_t)kHeads * 256);
  for (auto &kv : kb) {
    const std::uint64_t rowb = rdna4::kv_row_bytes(kv.first, 256);
    const std::size_t cb = (std::size_t)maxkeys * kHeadsKv * rowb;
    // separate fills per type would be ideal; kv_fill_launch is deterministic
    // per (type, rows, seed), so re-fill per type keeps the comparison honest
    double wrel = 0, wmaxd = 0;
    bool ok = true;
    int ncase = 0;
    for (int keys : keys_b) {
      const int t = keys - 1;
      const std::int64_t rows = (std::int64_t)keys * kHeadsKv;
      if (!rdna4::kv_fill_launch(kv.first, b_k, rows, 256, 11u) ||
          !rdna4::kv_fill_launch(kv.first, b_v, rows, 256, 22u)) {
        ok = false;
        break;
      }
      hipDeviceSynchronize();
      for (int ns : splits_b) {
        ++ncase;
        const int wpb = rdna4::attn_split_wpb(ns);
        hipMemset(b_o1, 0xFF, o1.size() * sizeof(float));
        hipMemset(b_o2, 0xFF, o2.size() * sizeof(float));
        hipMemset(b_partial, 0xFF, rdna4::attn_partial_bytes(kHeads, 256, 16));
        (void)rdna4::attn_launch(b_q, b_k, b_v, b_o1, t, kHeads, kHeadsKv, 256, kScale, kv.first,
                                 kv.first, nullptr);
        (void)rdna4::attn_launch_split_wpb(b_q, b_k, b_v, b_o2, b_partial, t, kHeads, kHeadsKv,
                                           256, kScale, kv.first, kv.first, ns, wpb, nullptr);
        hipDeviceSynchronize();
        if (hipGetLastError() != hipSuccess) {
          ok = false;
          continue;
        }
        hipMemcpy(o1.data(), b_o1, o1.size() * sizeof(float), hipMemcpyDeviceToHost);
        hipMemcpy(o2.data(), b_o2, o2.size() * sizeof(float), hipMemcpyDeviceToHost);
        double num = 0, den = 0, md = 0;
        for (std::size_t i = 0; i < o1.size(); ++i) {
          if (!std::isfinite(o1[i]) || !std::isfinite(o2[i])) ok = false;
          const double d = (double)o1[i] - (double)o2[i];
          num += d * d;
          den += (double)o1[i] * o1[i] + (double)o2[i] * o2[i];
          md = std::fmax(md, std::fabs(d));
        }
        const double rel = std::sqrt(num / den);
        wrel = std::fmax(wrel, rel);
        wmaxd = std::fmax(wmaxd, md);
        if (!(rel <= 1e-6)) ok = false;
      }
    }
    char nm[96];
    std::snprintf(nm, sizeof(nm), "B ragged/sentinel split-vs-unsplit kv=%s (%d shapes)", kv.second,
                  ncase);
    check(nm, ok, wrel, wmaxd);
    (void)cb;
  }
  std::printf("  merge-audit: %d/%d cases passed (kv under test: %s + f32 anchor)\n", cases - failures,
              cases, tname);
  hipFree(d_partial);
  hipFree(d_out);
  hipFree(b_q);
  hipFree(b_o1);
  hipFree(b_o2);
  hipFree(b_k);
  hipFree(b_v);
  hipFree(b_partial);
  return failures ? 1 : 0;
}

// ---------------------------------------------------------------------------
// H6 step 2 — split sweep {2,4,8} (+16 where asked) vs the ship policy, one
// batched gpu-lock hold: every ctx row is an INTERLEAVED duel (rounds >= 8,
// order alternated each round, 3 launches per measurement), reporting
// best+mean+wins and the kernel->token stake (x16 layers). Reference per row
// is attn_splits_for/attn_split_wpb mirrored from graph.cuh/attn.cuh (the
// same mirror the sweep table uses), candidates are --splits x --wpb.
// ---------------------------------------------------------------------------
static int run_h6(rdna4::KvType kt, const char *tname, const std::vector<int> &ts,
                  const std::vector<int> &split_list, const std::vector<int> &wpb_list,
                  int rounds) {
  constexpr int kReps = 3;
  std::printf("H6 split sweep kv=%s, %d interleaved rounds x %d reps (order alternated)\n", tname,
              rounds, kReps);
  float *d_q = nullptr, *d_out = nullptr;
  void *d_k = nullptr, *d_v = nullptr;
  std::vector<float> qh;
  fill_random(qh, (std::size_t)kHeads * kHeadDim, 3u);
  hipMalloc(&d_q, qh.size() * sizeof(float));
  hipMalloc(&d_out, (std::size_t)kHeads * kHeadDim * sizeof(float));
  hipMemcpy(d_q, qh.data(), qh.size() * sizeof(float), hipMemcpyHostToDevice);
  int maxt = ts.back();
  for (int t : ts) maxt = std::max(maxt, t);
  const std::uint64_t rowb = rdna4::kv_row_bytes(kt, kHeadDim);
  const std::size_t max_bytes = (std::size_t)(maxt + 1) * kHeadsKv * rowb;
  int smax = 16;
  for (int sp : split_list) smax = std::max(smax, sp);
  float *d_partial = nullptr;
  if (hipMalloc(&d_k, max_bytes) != hipSuccess || hipMalloc(&d_v, max_bytes) != hipSuccess ||
      hipMalloc(&d_partial, rdna4::attn_partial_bytes(kHeads, kHeadDim, smax)) != hipSuccess) {
    std::fprintf(stderr, "H6: hipMalloc failed\n");
    return 1;
  }
  struct Cell {
    int sp, wpb;
    bool ref;
  };
  for (int t : ts) {
    const int keys = t + 1;
    const std::int64_t rows = (std::int64_t)keys * kHeadsKv;
    if (!rdna4::kv_fill_launch(kt, d_k, rows, kHeadDim, 11u) ||
        !rdna4::kv_fill_launch(kt, d_v, rows, kHeadDim, 22u)) {
      std::fprintf(stderr, "H6: kv_fill_launch failed at t=%d\n", t);
      return 1;
    }
    const bool kv_dense = (kt == rdna4::KvType::F16 || kt == rdna4::KvType::F32);
    int ship_sp;
    if (keys < rdna4::tuned::kAttnSplitMin) ship_sp = 1;
    else if (kv_dense) ship_sp = (rdna4::tuned::kAttnSplitCtasDense + kHeads - 1) / kHeads;
    else ship_sp = keys / rdna4::tuned::kAttnSplitMin;
    ship_sp = std::min(std::max(ship_sp, 1), rdna4::tuned::kAttnMaxSplits);
    const int ship_wpb = rdna4::attn_split_wpb(ship_sp);
    std::vector<Cell> cells{{ship_sp, ship_wpb, true}};
    for (int sp : split_list)
      for (int wpb : wpb_list) {
        if (sp == ship_sp && wpb == ship_wpb) continue;
        if (wpb != 8 && wpb != 16 && wpb != 32) continue;
        cells.push_back({sp, wpb, false});
      }
    auto run_cell = [&](const Cell &c) {
      if (c.sp <= 1)
        (void)rdna4::attn_launch(d_q, d_k, d_v, d_out, t, kHeads, kHeadsKv, kHeadDim, kScale, kt,
                                 kt, nullptr);
      else
        (void)rdna4::attn_launch_split_wpb(d_q, d_k, d_v, d_out, d_partial, t, kHeads, kHeadsKv,
                                           kHeadDim, kScale, kt, kt, c.sp, c.wpb, nullptr);
    };
    {
      const auto w0 = std::chrono::steady_clock::now();
      do {
        for (const Cell &c : cells) run_cell(c);
        hipDeviceSynchronize();
      } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0)
                   .count() < 300.0);
    }
    EvPair ev;
    std::vector<double> best(cells.size(), -1.0), sum(cells.size(), 0.0);
    std::vector<int> wins(cells.size(), 0);
    std::vector<std::vector<double>> hist(cells.size());
    for (int r = 0; r < rounds; ++r) {
      const bool fwd = (r % 2) == 0;
      std::vector<double> got(cells.size(), -1.0);
      for (std::size_t k = 0; k < cells.size(); ++k) {
        const std::size_t idx = fwd ? k : cells.size() - 1 - k;
        const double ms = ev([&] {
          for (int j = 0; j < kReps; ++j) run_cell(cells[idx]);
        }) / kReps;
        got[idx] = ms;
        hist[idx].push_back(ms);
        if (best[idx] < 0 || ms < best[idx]) best[idx] = ms;
        sum[idx] += ms;
      }
      for (std::size_t k = 1; k < cells.size(); ++k)
        if (got[k] > 0 && got[0] > 0 && got[k] < got[0]) wins[k]++;
    }
    std::printf("t=%d (%d keys) kv=%s ship=%d:%d\n", t, keys, tname, ship_sp, ship_wpb);
    std::printf("  %-8s %-8s %10s %10s %9s %9s %9s  %s\n", "splits", "warps", "best ms",
                "mean ms", "best/ref", "mean/ref", "wins", "note");
    for (std::size_t k = 0; k < cells.size(); ++k) {
      const double mean = sum[k] / rounds;
      std::printf("  %-8d %-8d %10.4f %10.4f %8.3fx %8.3fx %5d/%-4d  %s\n", cells[k].sp,
                  cells[k].wpb, best[k], mean, best[0] / best[k], (sum[0] / rounds) / mean,
                  wins[k], rounds, k == 0 ? "reference (ship policy)" : "candidate");
    }
    // numerics of every cell vs the unsplit output (one-shot, same convention
    // as the sweep table: rel-L2 + max|d|)
    std::vector<float> ref((std::size_t)kHeads * kHeadDim), gotv((std::size_t)kHeads * kHeadDim);
    {
      // unsplit reference even where the engine ships splits
      (void)rdna4::attn_launch(d_q, d_k, d_v, d_out, t, kHeads, kHeadsKv, kHeadDim, kScale, kt, kt,
                               nullptr);
      hipDeviceSynchronize();
      hipMemcpy(ref.data(), d_out, ref.size() * sizeof(float), hipMemcpyDeviceToHost);
    }
    for (std::size_t k = 0; k < cells.size(); ++k) {
      run_cell(cells[k]);
      hipDeviceSynchronize();
      hipMemcpy(gotv.data(), d_out, gotv.size() * sizeof(float), hipMemcpyDeviceToHost);
      double num = 0, den = 0, md = 0;
      for (std::size_t i = 0; i < gotv.size(); ++i) {
        const double d = (double)ref[i] - (double)gotv[i];
        num += d * d;
        den += (double)ref[i] * ref[i] + (double)gotv[i] * gotv[i];
        md = std::max(md, std::fabs(d));
      }
      std::printf("    cell %d:%d numerics vs unsplit: rel-L2 %.2e max|d| %.2e%s\n", cells[k].sp,
                  cells[k].wpb, std::sqrt(num / den), md,
                  (k == 0 && ship_sp <= 1) ? " (identical path)" : "");
    }
    double best_cand = -1;
    for (std::size_t k = 1; k < cells.size(); ++k)
      if (best_cand < 0 || best[k] < best_cand) best_cand = best[k];
    std::printf("  x%d layers: ref-best %.4f ms -> cand-best %.4f ms = %+.4f ms/token\n", kLayers,
                best[0], best_cand, (best[0] - best_cand) * kLayers);
  }
  hipFree(d_q);
  hipFree(d_out);
  hipFree(d_k);
  hipFree(d_v);
  hipFree(d_partial);
  return 0;
}

}  // namespace

int main(int argc, char **argv) {
  const char *tname = argc > 1 ? argv[1] : "f16";
  std::vector<int> ts;
  std::vector<int> split_list{2, 4, 8, 16};
  std::vector<int> wpb_list{8, 16, 32};
  int smax = 16;
  bool have_ab = false;
  bool no_sweep = false;
  int gqa_duel_hg = 0;  // >0: kill-test duel only (HG fused vs production), then exit
  int duel_rounds = 8;
  bool merge_audit = false;  // H6 step 1: merge reduction audit, then exit
  bool h6_mode = false;      // H6 step 2: batched interleaved split sweep, then exit
  int h6_rounds = 12;        // >= 8 per the H6 brief (64K decode variance +-10%)
  bool have_ts = false, have_splits = false, have_wpb = false;
  int ab_sp = 0, ab_wpb = 0, ab_rounds = 15, ab_reps = 3;
  std::vector<std::pair<int, int>> cands;  // (splits, warps/CTA)
  std::vector<int> gqa_hg{6, 3, 1};        // heads per pass of the GQA-split prototype
  int gqa_splits = 0;                      // 0 = the shipping split policy for that t
  auto cell_spec = [](const char *v, int *sp, int *wpb) {
    *sp = std::atoi(v);
    while (*v && *v != ':') ++v;
    *wpb = (*v == ':') ? std::atoi(v + 1) : 0;
  };
  for (int i = 2; i < argc; ++i) {
    auto list = [&](std::vector<int> &out) {
      const char *v = argv[++i];
      while (*v) {
        out.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    };
    if (std::strcmp(argv[i], "--smax") == 0 && i + 1 < argc) {
      smax = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--splits") == 0 && i + 1 < argc) {
      split_list.clear();
      list(split_list);
      have_splits = true;
    } else if (std::strcmp(argv[i], "--wpb") == 0 && i + 1 < argc) {
      wpb_list.clear();
      list(wpb_list);
      have_wpb = true;
    } else if (std::strcmp(argv[i], "--ab") == 0 && i + 1 < argc) {
      cell_spec(argv[++i], &ab_sp, &ab_wpb);
      have_ab = true;
    } else if (std::strcmp(argv[i], "--cand") == 0 && i + 1 < argc) {
      int sp = 0, wpb = 0;
      cell_spec(argv[++i], &sp, &wpb);
      cands.push_back({sp, wpb});
    } else if (std::strcmp(argv[i], "--gqa-hg") == 0 && i + 1 < argc) {
      gqa_hg.clear();
      list(gqa_hg);
    } else if (std::strcmp(argv[i], "--gqa-splits") == 0 && i + 1 < argc) {
      gqa_splits = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--no-sweep") == 0) {
      no_sweep = true;
    } else if (std::strcmp(argv[i], "--gqa-duel") == 0 && i + 1 < argc) {
      gqa_duel_hg = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--duel-rounds") == 0 && i + 1 < argc) {
      duel_rounds = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--ab-rounds") == 0 && i + 1 < argc) {
      ab_rounds = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--merge-audit") == 0) {
      merge_audit = true;
    } else if (std::strcmp(argv[i], "--h6") == 0) {
      h6_mode = true;
    } else if (std::strcmp(argv[i], "--h6-rounds") == 0 && i + 1 < argc) {
      h6_rounds = std::atoi(argv[++i]);
    } else {
      ts.push_back(std::atoi(argv[i]));
      have_ts = true;
    }
  }
  if (ts.empty()) ts = {4096, 16384, 65536};
  if (smax < 1) smax = 1;
  for (int sp : split_list) smax = std::max(smax, sp);
  // The duel runs (`--ab`/`--cand`) also need their split counts in the partial
  // buffer's size: a candidate above smax writes past d_partial. Measured the hard
  // way -- `--cand 24:16` without `--splits 24` left smax at 16 and every duel in
  // a 9-run battery died with "Memory access fault ... Page not present", which
  // looks like a broken card instead of an under-sized buffer.
  for (const auto &c : cands) smax = std::max(smax, c.first);

  rdna4::KvType kt = rdna4::KvType::F16;
  // The WPB/split entry points are only instantiated for the diagonal pairs (this
  // bench always benches K and V at the same type), so the new formats are added
  // here as diagonal entries -- the shipping attn_launch carries the full 6x6.
  if (std::strcmp(tname, "q4_0") == 0) kt = rdna4::KvType::Q4_0;
  else if (std::strcmp(tname, "q8_0") == 0) kt = rdna4::KvType::Q8_0;
  else if (std::strcmp(tname, "f32") == 0) kt = rdna4::KvType::F32;
  else if (std::strcmp(tname, "q5_0") == 0) kt = rdna4::KvType::Q5_0;
  else if (std::strcmp(tname, "q4_1") == 0) kt = rdna4::KvType::Q4_1;

  if (gqa_duel_hg > 0) {
    if (ts.empty()) ts = {65536};
    for (int t : ts) {
      const int rc = run_gqa_duel(kt, t, gqa_duel_hg, duel_rounds);
      if (rc) return rc;
    }
    return 0;
  }

  if (merge_audit) return run_merge_audit(kt, tname);

  if (h6_mode) {
    if (!have_ts) ts = {8191, 16383, 32767, 65535};  // 8K/16K/32K/64K keys
    if (!have_splits) split_list = {2, 4, 8};
    if (!have_wpb) wpb_list = {8, 16};
    if (h6_rounds < 8) h6_rounds = 8;
    return run_h6(kt, tname, ts, split_list, wpb_list, h6_rounds);
  }

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 1;
  std::printf("device: %s\nkey/value cache type: %s, head_dim %d, %d heads / %d kv heads "
              "(GQA %dx)\n\n",
              prop.name, tname, kHeadDim, kHeads, kHeadsKv, kHeads / kHeadsKv);
  // Registers/spills of every attention kernel this bench measures: at 8 warps/CTA
  // and head_dim 256 the GQA prototypes are register-hungry, and a prototype that
  // spills explains its own slowness without any theory about grids.
  {
    std::printf("kernel resources (hipFuncGetAttributes):\n");
    report_resources("attn_kernel<f16>", rdna4::attn_kernel<rdna4::KvType::F16, rdna4::KvType::F16>);
    report_resources("attn_split_kernel<f16>",
                     rdna4::attn_split_kernel<rdna4::KvType::F16, rdna4::KvType::F16>);
    report_resources("attn_gqa_kernel<f16>", attn_gqa_kernel<rdna4::KvType::F16, rdna4::KvType::F16>);
    report_resources("gqa_split HG=6", attn_gqa_split_kernel<rdna4::KvType::F16, rdna4::KvType::F16, 6>);
    report_resources("gqa_split HG=3", attn_gqa_split_kernel<rdna4::KvType::F16, rdna4::KvType::F16, 3>);
    report_resources("gqa_split HG=1", attn_gqa_split_kernel<rdna4::KvType::F16, rdna4::KvType::F16, 1>);
    report_resources("rowtile D", attn_gqa_rowtile_kernel<rdna4::KvType::F16, rdna4::KvType::F16>);
    report_resources("attn_merge_kernel", rdna4::attn_merge_kernel);
    int cu = 0;
    (void)hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, 0);
    std::printf("  (device: %d CU)\n\n", cu);
  }

  const int maxt = ts.empty() ? 65536 : ts.back();
  std::vector<float> qh, kh;
  fill_random(qh, (std::size_t)kHeads * kHeadDim, 3u);
  const std::size_t kelem = (std::size_t)maxt * kHeadsKv * kHeadDim;
  fill_random(kh, kelem, 5u);

  float *d_q = nullptr, *d_out = nullptr, *d_out2 = nullptr;
  void *d_k = nullptr, *d_v = nullptr;
  hipMalloc(&d_q, qh.size() * sizeof(float));
  hipMalloc(&d_out, (std::size_t)kHeads * kHeadDim * sizeof(float));
  hipMalloc(&d_out2, (std::size_t)kHeads * kHeadDim * sizeof(float));
  hipMemcpy(d_q, qh.data(), qh.size() * sizeof(float), hipMemcpyHostToDevice);
  // the cache must be sized for the LARGEST t up front (it is reused across rows)
  const std::uint64_t max_rowb = rdna4::kv_row_bytes(kt, kHeadDim);
  const std::size_t max_bytes = (std::size_t)(maxt + 1) * kHeadsKv * max_rowb;
  if (hipMalloc(&d_k, max_bytes) != hipSuccess || hipMalloc(&d_v, max_bytes) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed for the %zu-byte cache\n", max_bytes);
    return 1;
  }

  std::printf("%8s %10s %12s %12s %12s %10s %10s\n", "t", "kv MiB", "ship ms", "split ms",
              "gqa ms", "split GB/s", "speedup");
  for (int t : ts) {
    // fill the cache with the deterministic pattern the engine's tests use (the
    // values only need to be finite; this is a cost bench)
    const std::uint64_t rowb = rdna4::kv_row_bytes(kt, kHeadDim);
    const std::size_t bytes = (std::size_t)(t + 1) * kHeadsKv * rowb;
    const std::int64_t rows = (std::int64_t)(t + 1) * kHeadsKv;
    (void)hipGetLastError();  // a stale error from an earlier phase must not be
                              // misattributed to the fill (it was, in run 3)
    if (!rdna4::kv_fill_launch(kt, d_k, rows, kHeadDim, 11u) ||
        !rdna4::kv_fill_launch(kt, d_v, rows, kHeadDim, 22u)) {
      std::fprintf(stderr, "kv_fill_launch failed at t=%d: %s\n", t,
                   hipGetErrorString(hipGetLastError()));
      return 1;
    }
    // ------------------------------------------------------------------
    // The search space of the attention kernel: split count x warps per CTA,
    // measured the way this repo measures everything (PLAN.md M2 "Passo 5b"):
    // ONE warm-up until ~300 ms of GPU time, then every cell timed by an event
    // pair around `reps` launches, in ROTATED order over `rounds` rounds, keeping
    // the MINIMUM per cell. The first cell measured after the cache fill is
    // systematically slower (measured 0.85x on the same unsplit kernel when it
    // was timed first and last), so a sequential sweep with a per-cell warm-up is
    // not trustworthy at the few-percent level this table is used at.
    //
    // `ship` is the same unsplit kernel as `ship2` and `ship3` measured again as
    // their own cells: their spread IS this bench's noise floor.
    //
    // WPB is a template knob (attn.cuh); changing it changes how many partial
    // slices are merged per CTA (rel-L2 column), i.e. it is a numeric-equivalence
    // change, not a bit-exact one. The shipping policy picks the warps
    // (attn_split_wpb: 16 warps for n_splits <= kAttnSplitWpbLimit, 8 above).
    // ------------------------------------------------------------------
    constexpr int kReps = 5;
    constexpr int kRounds = 3;
    float *d_partial = nullptr;
    if (hipMalloc(&d_partial, rdna4::attn_partial_bytes(kHeads, kHeadDim, smax)) != hipSuccess) {
      std::fprintf(stderr, "hipMalloc partial failed\n");
      return 1;
    }
    const int keys = t + 1;
    // The cell the SHIPPING policy picks, mirrored from Graph::attn_splits_for
    // (graph.cuh) -- it is the reference every table below is judged against, so it
    // has to be that function and not a paraphrase of it. Two bugs lived here: the
    // WPB came from a stale copy of attn_split_wpb (16 warps only for splits <=
    // kAttnSplitWpbLimit, which missed kAttnSplitWpbWide and so labelled 16:8 -- a
    // cell the engine never runs -- as "ship policy" for every keys >= 8192), and
    // the split count ignored the KV-type split of the rule (f16/f32 want a fixed
    // CTA count, quantized KV keeps keys/kAttnSplitMin).
    const bool kv_dense = (kt == rdna4::KvType::F16 || kt == rdna4::KvType::F32);
    int n_splits;
    if (keys < rdna4::tuned::kAttnSplitMin) {
      n_splits = 1;
    } else if (kv_dense) {
      n_splits = (rdna4::tuned::kAttnSplitCtasDense + kHeads - 1) / kHeads;
    } else {
      n_splits = keys / rdna4::tuned::kAttnSplitMin;
    }
    if (n_splits < 1) n_splits = 1;
    if (n_splits > rdna4::tuned::kAttnMaxSplits) n_splits = rdna4::tuned::kAttnMaxSplits;
    const int n_splits_wpb = rdna4::attn_split_wpb(n_splits);
    const int gqa_splits_t = gqa_splits > 0 ? gqa_splits : n_splits;

    if (no_sweep) {
      // duel only: the sweep (and the GQA prototype) is skipped so a run measures
      // exactly the two configurations being decided, nothing else
      struct DCell { int sp, wpb; double ms; };
      std::vector<DCell> duel;
      duel.push_back({ab_sp, ab_wpb, -1.0});
      for (auto &c : cands) duel.push_back({c.first, c.second, -1.0});
      auto run_d = [&](const DCell &c) {
        if (c.sp == 0)
          (void)rdna4::attn_launch(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, kt,
                                   kt, nullptr);
        else
          (void)rdna4::attn_launch_split_wpb(d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv,
                                             kHeadDim, kScale, kt, kt, c.sp, c.wpb, nullptr);
      };
      {
        const auto w0 = std::chrono::steady_clock::now();
        do {
          for (const DCell &c : duel) run_d(c);
          hipDeviceSynchronize();
        } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0)
                     .count() < 300.0);
      }
      std::vector<int> wins(duel.size(), 0);
      EvPair ev2;
      for (int r = 0; r < ab_rounds; ++r) {
        const bool fwd = (r % 2) == 0;
        std::vector<double> got(duel.size(), -1.0);
        for (std::size_t k = 0; k < duel.size(); ++k) {
          const std::size_t idx = fwd ? k : duel.size() - 1 - k;
          const double ms = ev2([&] {
            for (int j = 0; j < ab_reps; ++j) run_d(duel[idx]);
          }) / ab_reps;
          got[idx] = ms;
          if (duel[idx].ms < 0 || ms < duel[idx].ms) duel[idx].ms = ms;
        }
        for (std::size_t k = 1; k < duel.size(); ++k)
          if (got[k] > 0 && got[0] > 0 && got[k] < got[0]) wins[k]++;
      }
      std::printf("t=%d (%d keys) INTERLEAVED A/B (%d rounds, %d launches each, order swapped)\n",
                  t, t + 1, ab_rounds, ab_reps);
      std::printf("  %-8s %-8s %10s %8s %10s  %s\n", "splits", "warps", "min ms", "vs ref",
                  "wins", "note");
      for (std::size_t k = 0; k < duel.size(); ++k)
        std::printf("  %-8d %-8d %10.4f %7.3fx %5d/%-4d  %s\n", duel[k].sp, duel[k].wpb,
                    duel[k].ms, duel[0].ms / duel[k].ms, wins[k], ab_rounds,
                    k == 0 ? "reference" : "candidate");
      std::printf("  (%d layers x 24 heads: the difference is worth %.3f ms/token)\n\n",
                  kLayers, (duel[0].ms - duel[1].ms) * kLayers);
      hipFree(d_partial);
      continue;
    }

    struct Cell {
      int sp;    // 0 = the shipping unsplit kernel; >0 = split count (split path
                 // keeps its functional WPB knob). The unsplit forced-warp cells
                 // are gone (H0): the unsplit kernel ignores WPB by construction.
      int wpb;
      double ms;
      double rel;
      double maxd;
    };
    std::vector<Cell> cells;
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship (attn_launch, unsplit)
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship2: identical -> noise floor
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship3: identical -> noise floor
    for (int sp : split_list)
      for (int wpb : wpb_list) cells.push_back({sp, wpb, -1.0, 0.0, 0.0});

    auto run_cell = [&](const Cell &c) {
      if (c.sp == 0) {
        (void)rdna4::attn_launch(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, kt,
                                 kt, nullptr);
      } else {
        (void)rdna4::attn_launch_split_wpb(d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv,
                                           kHeadDim, kScale, kt, kt, c.sp, c.wpb, nullptr);
      }
    };
    // one global warm-up: the DPM state is what it is, and every cell below is
    // measured in the same (hot) state
    {
      const auto w0 = std::chrono::steady_clock::now();
      do {
        run_cell(cells[0]);
        run_cell(cells[cells.size() - 1]);
        hipDeviceSynchronize();
      } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0)
                   .count() < 300.0);
    }
    EvPair ev;

    // ------------------------------------------------------------------
    // Interleaved A/B ("duel"): the sweep above can place two cells several
    // seconds apart, and on this card the SAME kernel measured as three
    // independent cells spread by up to 20% (see the noise-floor line). For a
    // decision that ships, the two configurations are measured ALTERNATELY,
    // `ab_rounds` times each way, and the table reports the min, the ratio and
    // how many rounds the candidate won -- a difference that does not win most
    // rounds is not a win, whatever the min says.
    // ------------------------------------------------------------------
    if (have_ab || !cands.empty()) {
      std::vector<Cell> duel;
      duel.push_back({ab_sp, ab_wpb, -1.0, 0.0, 0.0});
      for (auto &c : cands) duel.push_back({c.first, c.second, -1.0, 0.0, 0.0});
      std::vector<int> wins(duel.size(), 0);
      for (int r = 0; r < ab_rounds; ++r) {
        const bool fwd = (r % 2) == 0;
        std::vector<double> got(duel.size(), -1.0);
        for (std::size_t k = 0; k < duel.size(); ++k) {
          const std::size_t idx = fwd ? k : duel.size() - 1 - k;
          const double ms = ev([&] {
            for (int j = 0; j < ab_reps; ++j) run_cell(duel[idx]);
          }) / ab_reps;
          got[idx] = ms;
          if (duel[idx].ms < 0 || ms < duel[idx].ms) duel[idx].ms = ms;
        }
        for (std::size_t k = 1; k < duel.size(); ++k)
          if (got[k] > 0 && got[0] > 0 && got[k] < got[0]) wins[k]++;
      }
      std::printf("  INTERLEAVED A/B (%d rounds, %d launches per measurement, order swapped each "
                  "round)\n", ab_rounds, ab_reps);
      std::printf("  %-10s %-10s %10s %8s %10s  %s\n", "splits", "warps/CTA", "min ms", "vs A",
                  "wins", "note");
      for (std::size_t k = 0; k < duel.size(); ++k) {
        std::printf("  %-10d %-10d %10.4f %7.3fx %5d/%-4d  %s\n", duel[k].sp, duel[k].wpb,
                    duel[k].ms, duel[0].ms / duel[k].ms, wins[k], ab_rounds,
                    k == 0 ? "reference" : "candidate");
      }
      std::printf("  A/B noise floor is the reference row against itself: run --cand %d:%d to "
                  "measure it.\n", ab_sp, ab_wpb);
    }

    for (int r = 0; r < kRounds; ++r) {
      for (std::size_t i = 0; i < cells.size(); ++i) {
        Cell &c = cells[(i + (std::size_t)r * 7) % cells.size()];
        const double ms = ev([&] {
          for (int k = 0; k < kReps; ++k) run_cell(c);
        }) / kReps;
        if (c.ms < 0 || ms < c.ms) c.ms = ms;
      }
    }
    // numerics per cell, against the unsplit reference output
    std::vector<float> ref((std::size_t)kHeads * kHeadDim);
    run_cell(cells[1]);
    hipDeviceSynchronize();
    hipMemcpy(ref.data(), d_out2, ref.size() * sizeof(float), hipMemcpyDeviceToHost);
    for (Cell &c : cells) {
      run_cell(c);
      hipDeviceSynchronize();
      std::vector<float> got((std::size_t)kHeads * kHeadDim);
      hipMemcpy(got.data(), d_out2, got.size() * sizeof(float), hipMemcpyDeviceToHost);
      double num = 0, den = 0, md = 0;
      for (std::size_t i = 0; i < got.size(); ++i) {
        const double d = (double)ref[i] - (double)got[i];
        num += d * d;
        den += (double)ref[i] * ref[i] + (double)got[i] * got[i];
        md = std::max(md, std::fabs(d));
      }
      c.rel = std::sqrt(num / den);
      c.maxd = md;
    }
    const double ms_ship = cells[0].ms;
    const double ms_ship2 = std::min(cells[1].ms, cells[2].ms);
    std::printf("  noise floor (same unsplit kernel as 3 independent cells, min of %d rounds): "
                "%.4f vs %.4f ms = %.3fx\n",
                kRounds, ms_ship, ms_ship2, ms_ship2 / ms_ship);

    std::printf("  %-10s %-10s %10s %8s %12s %10s\n", "splits", "warps/CTA", "ms", "vs unsplit",
                "rel-L2", "grid CTAs");
    double best_ms = ms_ship, policy_ms = -1.0;
    int best_sp = 0, best_wpb = 0, policy_sp = 1, policy_wpb = 0;
    for (const Cell &c : cells) {
      if (c.sp <= 0) continue;
      const bool is_policy = (c.sp == n_splits) && (c.wpb == n_splits_wpb);
      if (is_policy) {
        policy_ms = c.ms;
        policy_sp = c.sp;
        policy_wpb = c.wpb;
      }
      if (c.ms < best_ms) {
        best_ms = c.ms;
        best_sp = c.sp;
        best_wpb = c.wpb;
      }
      std::printf("  %-10d %-10d %10.4f %7.2fx %12.2e %10d%s\n", c.sp, c.wpb, c.ms,
                  ms_ship / c.ms, c.rel, kHeads * c.sp, is_policy ? "  <- ship policy" : "");
    }
    std::printf("  best cell: %d splits x %d warps = %.4f ms (%.3fx unsplit); "
                "ship policy at %d keys = %d splits x %d warps = %.4f ms (%.3fx unsplit)\n",
                best_sp, best_wpb, best_ms, ms_ship / best_ms, keys, policy_sp, policy_wpb,
                policy_ms, policy_ms > 0 ? ms_ship / policy_ms : 0.0);
    std::printf("  best cell vs ship policy: %.3fx (%.4f vs %.4f ms) over %d heads x %d layers "
                "= %.3f ms/token at stake; noise floor %.3fx\n",
                policy_ms / best_ms, best_ms, policy_ms, kHeads, kLayers,
                (policy_ms - best_ms) * kLayers, ms_ship2 / ms_ship);
    const double ms_split = policy_ms > 0 ? policy_ms : -1.0;
    double split_rel = 0.0, split_maxd = 0.0;
    for (const Cell &c : cells)
      if (c.sp == policy_sp && c.wpb == policy_wpb) {
        split_rel = c.rel;
        split_maxd = c.maxd;
      }
    // the three kernels must agree (they only differ in how the slices are grouped)
    std::vector<float> a((std::size_t)kHeads * kHeadDim), b((std::size_t)kHeads * kHeadDim);
    // BUG FIX (frente longctx): this used to read `d_out`, which NO kernel in this
    // bench ever writes -- the "prototype vs ship" rel-L2 below was comparing
    // against uninitialized VRAM (so the M7 prototype's numerics were never
    // actually verified). Take the reference from an explicit run of the unsplit
    // shipping kernel into d_out2.
    (void)rdna4::attn_launch(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, kt, kt,
                             nullptr);
    hipDeviceSynchronize();
    hipMemcpy(a.data(), d_out2, a.size() * sizeof(float), hipMemcpyDeviceToHost);
    // ---- frente longctx: protótipo C, GQA compartilhado com grade larga ----
    // Mesmo grid do kernel com splits (n_head_kv, S) e o mesmo merge; o que muda é
    // o HG (cabeças de consulta por passada sobre a fatia de chaves).
    struct GqaS {
      int hg;
      double ms;
      double rel;
      bool ok;
    };
    std::vector<GqaS> gqas;
    for (int hg : gqa_hg) {
      gqas.push_back({hg, -1.0, 0.0, true});
    }
    {
      // warm-up + timed pass, one event pair per (hg) cell
      EvPair evg;
      for (int r = 0; r < 3; ++r) {
        for (auto &g : gqas) {
          const double ms = evg([&] {
            for (int k = 0; k < 3; ++k) {
              switch (kt) {
                case rdna4::KvType::F16:
                  g.ok = launch_gqa_split<rdna4::KvType::F16, rdna4::KvType::F16>(
                      d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                      gqa_splits_t, g.hg, nullptr);
                  break;
                case rdna4::KvType::Q4_0:
                  g.ok = launch_gqa_split<rdna4::KvType::Q4_0, rdna4::KvType::Q4_0>(
                      d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                      gqa_splits_t, g.hg, nullptr);
                  break;
                case rdna4::KvType::Q8_0:
                  g.ok = launch_gqa_split<rdna4::KvType::Q8_0, rdna4::KvType::Q8_0>(
                      d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                      gqa_splits_t, g.hg, nullptr);
                  break;
                default:
                  g.ok = launch_gqa_split<rdna4::KvType::F32, rdna4::KvType::F32>(
                      d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                      gqa_splits_t, g.hg, nullptr);
                  break;
              }
            }
          }) / 3.0;
          if (g.ms < 0 || ms < g.ms) g.ms = ms;
        }
      }
      std::printf("             gqa-split (grid %dx%d, S=%d), numerics vs unsplit:", kHeadsKv,
                  gqa_splits_t, gqa_splits_t);
      for (auto &g : gqas) {
        hipDeviceSynchronize();
        hipMemcpy(b.data(), d_out2, b.size() * sizeof(float), hipMemcpyDeviceToHost);
        double n2 = 0, d2 = 0, md2 = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          const double d = (double)a[i] - (double)b[i];
          n2 += d * d;
          d2 += (double)a[i] * a[i] + (double)b[i] * b[i];
          md2 = std::max(md2, std::fabs(d));
        }
        g.rel = std::sqrt(n2 / d2);
        std::printf("  HG=%d %.4f ms (%.3fx ship) rel-L2 %.1e%s", g.hg, g.ms,
                    ms_ship / g.ms, g.rel, g.ok ? "" : " LAUNCH_FAILED");
      }
      std::printf("\n");
    }
    // ---- E1: prototype D (row tile), same grid/merges as C, same reference ----
    {
      double ms_d = -1.0, rel_d = 0.0;
      bool ok_d = true;
      EvPair evd;
      for (int r = 0; r < 3; ++r) {
        const double ms = evd([&] {
          for (int k = 0; k < 3; ++k) {
            switch (kt) {
              case rdna4::KvType::F16:
                ok_d = launch_gqa_rowtile<rdna4::KvType::F16, rdna4::KvType::F16>(
                    d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                    gqa_splits_t, nullptr);
                break;
              case rdna4::KvType::Q4_0:
                ok_d = launch_gqa_rowtile<rdna4::KvType::Q4_0, rdna4::KvType::Q4_0>(
                    d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                    gqa_splits_t, nullptr);
                break;
              case rdna4::KvType::Q8_0:
                ok_d = launch_gqa_rowtile<rdna4::KvType::Q8_0, rdna4::KvType::Q8_0>(
                    d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                    gqa_splits_t, nullptr);
                break;
              default:
                ok_d = launch_gqa_rowtile<rdna4::KvType::F32, rdna4::KvType::F32>(
                    d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv, kHeadDim, kScale,
                    gqa_splits_t, nullptr);
                break;
            }
          }
        }) / 3.0;
        if (ms_d < 0 || ms < ms_d) ms_d = ms;
      }
      hipDeviceSynchronize();
      hipMemcpy(b.data(), d_out2, b.size() * sizeof(float), hipMemcpyDeviceToHost);
      {
        double n2 = 0, d2 = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
          const double d = (double)a[i] - (double)b[i];
          n2 += d * d;
          d2 += (double)a[i] * a[i] + (double)b[i] * b[i];
        }
        rel_d = std::sqrt(n2 / d2);
      }
      // E1 pre-registered targets (docs/plano-ninfer-pendente.md E1): >=1.02x vs
      // the ship-policy split cell at the same t.
      const double e1_target = (t == 4096) ? 0.0709 : (t == 16384) ? 0.2381 : -1.0;
      std::printf("             rowtile D (grid %dx%d, S=%d): %.4f ms (%.3fx ship, %.3fx split "
                  "policy %.4f) rel-L2 %.1e%s",
                  kHeadsKv, gqa_splits_t, gqa_splits_t, ms_d, ms_ship / ms_d,
                  ms_split > 0 ? ms_split / ms_d : 0.0, ms_split, rel_d,
                  ok_d ? "" : " LAUNCH_FAILED");
      if (e1_target > 0 && ok_d)
        std::printf("  E1 %s (target %.4f)", ms_d <= e1_target ? "PASS" : "FAIL", e1_target);
      std::printf("\n");
    }
    hipFree(d_partial);

    bool gqa_ok = true;
    const double ms_gqa = time_ms(
        [&] {
          switch (kt) {
            case rdna4::KvType::F16:
              gqa_ok = launch_gqa<rdna4::KvType::F16, rdna4::KvType::F16>(
                  d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, nullptr);
              break;
            case rdna4::KvType::Q4_0:
              gqa_ok = launch_gqa<rdna4::KvType::Q4_0, rdna4::KvType::Q4_0>(
                  d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, nullptr);
              break;
            case rdna4::KvType::Q8_0:
              gqa_ok = launch_gqa<rdna4::KvType::Q8_0, rdna4::KvType::Q8_0>(
                  d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, nullptr);
              break;
            default:
              gqa_ok = launch_gqa<rdna4::KvType::F32, rdna4::KvType::F32>(
                  d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, nullptr);
              break;
          }
        },
        5, 300.0);
    // `a` already holds the unsplit reference (see above); only read the candidate.
    hipMemcpy(b.data(), d_out2, b.size() * sizeof(float), hipMemcpyDeviceToHost);
    double num = 0, den = 0, maxd = 0;
    for (std::size_t i = 0; i < a.size(); ++i) {
      const double d = (double)a[i] - (double)b[i];
      num += d * d;
      den += (double)a[i] * a[i] + (double)b[i] * b[i];
      maxd = std::max(maxd, std::fabs(d));
    }
    const double rel = std::sqrt(num / den);
    const double kv_mib = (double)bytes / (1024.0 * 1024.0);
    // traffic: 2 (K+V) rows of kv_mib, times the GQA redundancy the kernel incurs
    const double grow = (double)kHeads / kHeadsKv;
    std::printf("%8d %10.1f %12.3f %12.3f %12.3f %10.1f %9.2fx", t, kv_mib, ms_ship, ms_split,
                gqa_ok ? ms_gqa : -1.0, ms_split > 0 ? 2 * kv_mib / ms_split : -1.0,
                ms_split > 0 ? ms_ship / ms_split : 0.0);
    std::printf("\n             split vs unsplit: rel-L2 %.2e max|d| %.2e (%d splits); "
                "prototype vs ship: rel-L2 %.1e\n",
                split_rel, split_maxd, n_splits, rel);
    (void)grow;
    // d_partial was already released above (the double free was a real bug here: the
    // second hipFree failed and polluted the error state the next iteration reads)
  }
  return 0;
}
