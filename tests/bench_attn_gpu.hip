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
// `attn_splits_for(keys)` (graph.cuh: keys/kAttnSplitMin, capped at kAttnMaxSplits)
// plus `attn_split_wpb(splits)` (attn.cuh: 16 warps up to kAttnSplitWpbLimit
// splits, 8 above). --smax raises the split cap so the sweep can ask whether the
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

// Kernel resource report: registers and spill bytes decide whether a prototype is
// slow because of its algorithm or because of its occupancy. (The M7 prototype's
// 8-12x was attributed to the narrow grid without this number; measure it.)
template <typename F>
void report_resources(const char *name, F *fn) {
  hipFuncAttributes a{};
  if (hipFuncGetAttributes(&a, (const void *)fn) != hipSuccess) {
    std::printf("  %-22s (attributes unavailable)\n", name);
    return;
  }
  std::printf("  %-22s regs/lane %3d  spill %4zu B  maxThreads %4d  shared %zu B\n", name,
              a.numRegs, a.localSizeBytes, a.maxThreadsPerBlock, a.sharedSizeBytes);
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
    } else if (std::strcmp(argv[i], "--wpb") == 0 && i + 1 < argc) {
      wpb_list.clear();
      list(wpb_list);
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
    } else if (std::strcmp(argv[i], "--ab-rounds") == 0 && i + 1 < argc) {
      ab_rounds = std::atoi(argv[++i]);
    } else {
      ts.push_back(std::atoi(argv[i]));
    }
  }
  if (ts.empty()) ts = {4096, 16384, 65536};
  if (smax < 1) smax = 1;
  for (int sp : split_list) smax = std::max(smax, sp);

  rdna4::KvType kt = rdna4::KvType::F16;
  // The WPB/split entry points are only instantiated for the diagonal pairs (this
  // bench always benches K and V at the same type), so the new formats are added
  // here as diagonal entries -- the shipping attn_launch carries the full 6x6.
  if (std::strcmp(tname, "q4_0") == 0) kt = rdna4::KvType::Q4_0;
  else if (std::strcmp(tname, "q8_0") == 0) kt = rdna4::KvType::Q8_0;
  else if (std::strcmp(tname, "f32") == 0) kt = rdna4::KvType::F32;
  else if (std::strcmp(tname, "q5_0") == 0) kt = rdna4::KvType::Q5_0;
  else if (std::strcmp(tname, "q4_1") == 0) kt = rdna4::KvType::Q4_1;

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
    constexpr int kPolicySplitMin = 512;
    constexpr int kPolicyMaxSplits = 16;
    constexpr int kReps = 5;
    constexpr int kRounds = 3;
    float *d_partial = nullptr;
    if (hipMalloc(&d_partial, rdna4::attn_partial_bytes(kHeads, kHeadDim, smax)) != hipSuccess) {
      std::fprintf(stderr, "hipMalloc partial failed\n");
      return 1;
    }
    const int keys = t + 1;
    int n_splits = keys / kPolicySplitMin;
    if (n_splits < 1) n_splits = 1;
    if (n_splits > kPolicyMaxSplits) n_splits = kPolicyMaxSplits;
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
      int sp;    // 0 = the shipping unsplit kernel, -1 = unsplit with forced wpb
      int wpb;
      double ms;
      double rel;
      double maxd;
    };
    std::vector<Cell> cells;
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship (attn_launch, unsplit)
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship2: identical -> noise floor
    cells.push_back({0, 0, -1.0, 0.0, 0.0});   // ship3: identical -> noise floor
    for (int wpb : wpb_list) cells.push_back({-1, wpb, -1.0, 0.0, 0.0});  // unsplit, forced warps
    for (int sp : split_list)
      for (int wpb : wpb_list) cells.push_back({sp, wpb, -1.0, 0.0, 0.0});

    auto run_cell = [&](const Cell &c) {
      if (c.sp == 0) {
        (void)rdna4::attn_launch(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale, kt,
                                 kt, nullptr);
      } else if (c.sp < 0) {
        (void)rdna4::attn_launch_wpb(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim, kScale,
                                     kt, kt, c.wpb, nullptr);
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
    std::printf("  unsplit kernel with forced warps/CTA:");
    for (const Cell &c : cells)
      if (c.sp < 0) std::printf("  wpb=%d: %.4f ms (%.2fx)", c.wpb, c.ms, ms_ship / c.ms);
    std::printf("\n");

    std::printf("  %-10s %-10s %10s %8s %12s %10s\n", "splits", "warps/CTA", "ms", "vs unsplit",
                "rel-L2", "grid CTAs");
    double best_ms = ms_ship, policy_ms = -1.0;
    int best_sp = 0, best_wpb = 0, policy_sp = 1, policy_wpb = 0;
    for (const Cell &c : cells) {
      if (c.sp <= 0) continue;
      const bool is_policy = (c.sp == n_splits) &&
                             (c.wpb == (c.sp <= rdna4::kAttnSplitWpbLimit
                                            ? 16
                                            : rdna4::kAttnWarpsPerBlock));
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
