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
// usage: bench-attn-gpu [kvtype] [t...]      (kvtype: f16 | q4_0 | q8_0 | f32)
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

void fill_random(std::vector<float> &v, std::size_t n, unsigned seed) {
  v.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    unsigned h = (unsigned)(i * 2654435761u) ^ (seed * 2246822519u);
    h ^= h >> 13;
    h *= 1274126177u;
    v[i] = ((float)(h >> 8) / 8388608.0f) - 1.0f;
  }
}

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

}  // namespace

int main(int argc, char **argv) {
  const char *tname = argc > 1 ? argv[1] : "f16";
  std::vector<int> ts;
  for (int i = 2; i < argc; ++i) ts.push_back(std::atoi(argv[i]));
  if (ts.empty()) ts = {4096, 16384, 65536};

  rdna4::KvType kt = rdna4::KvType::F16;
  if (std::strcmp(tname, "q4_0") == 0) kt = rdna4::KvType::Q4_0;
  else if (std::strcmp(tname, "q8_0") == 0) kt = rdna4::KvType::Q8_0;
  else if (std::strcmp(tname, "f32") == 0) kt = rdna4::KvType::F32;

  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) return 1;
  std::printf("device: %s\nkey/value cache type: %s, head_dim %d, %d heads / %d kv heads "
              "(GQA %dx)\n\n",
              prop.name, tname, kHeadDim, kHeads, kHeadsKv, kHeads / kHeadsKv);

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
    if (!rdna4::kv_fill_launch(kt, d_k, rows, kHeadDim, 11u) ||
        !rdna4::kv_fill_launch(kt, d_v, rows, kHeadDim, 22u)) {
      std::fprintf(stderr, "kv_fill_launch failed\n");
      return 1;
    }

    const double ms_ship =
        time_ms([&] { rdna4::attn_launch(d_q, d_k, d_v, d_out, t, kHeads, kHeadsKv, kHeadDim,
                                         kScale, kt, kt, nullptr); },
                5, 300.0);
    // UNSPLIT kernel with a different warps-per-CTA (the "32 warps" of M7: the
    // commit message claims kAttnWarpsPerBlock 8 -> 32, but the header still
    // says 8, so the shipping unsplit kernel is 8 warps; this is what the
    // measured 8/16/32 curve looks like now).
    {
      std::printf("  unsplit WPB sweep: ");
      std::vector<float> ref0((std::size_t)kHeads * kHeadDim);
      hipMemcpy(ref0.data(), d_out, ref0.size() * sizeof(float), hipMemcpyDeviceToHost);
      for (int wpb : {8, 16, 32}) {
        const double ms =
            time_ms([&] {
              (void)rdna4::attn_launch_wpb(d_q, d_k, d_v, d_out2, t, kHeads, kHeadsKv, kHeadDim,
                                           kScale, kt, kt, wpb, nullptr);
            }, 5, 300.0);
        std::vector<float> got((std::size_t)kHeads * kHeadDim);
        hipMemcpy(got.data(), d_out2, got.size() * sizeof(float), hipMemcpyDeviceToHost);
        double num = 0, den = 0;
        for (std::size_t i = 0; i < got.size(); ++i) {
          const double d = (double)ref0[i] - (double)got[i];
          num += d * d;
          den += (double)ref0[i] * ref0[i] + (double)got[i] * got[i];
        }
        std::printf("wpb=%d: %.4f ms (%.2fx) rel-L2 %.1e | ", wpb, ms, ms_ship / ms,
                    std::sqrt(num / den));
      }
      std::printf("\n");
    }
    // split-KV path (M7 step 2): same inputs, so the only difference is the
    // summation order across key splits
    float *d_partial = nullptr;
    const int n_splits = (t / 2048) < 1 ? 1 : ((t / 2048) > 16 ? 16 : (t / 2048));
    hipMalloc(&d_partial, rdna4::attn_partial_bytes(kHeads, kHeadDim, n_splits));
    std::vector<float> sp((std::size_t)kHeads * kHeadDim);
    double ms_split = -1.0, split_rel = -1.0, split_maxd = -1.0;
    if (n_splits > 1) {
      const bool ok_s = rdna4::attn_launch_split(d_q, d_k, d_v, d_out2, d_partial, t, kHeads,
                                                 kHeadsKv, kHeadDim, kScale, kt, kt, n_splits);
      hipDeviceSynchronize();
      if (ok_s) {
        hipMemcpy(sp.data(), d_out2, sp.size() * sizeof(float), hipMemcpyDeviceToHost);
        std::vector<float> a2((std::size_t)kHeads * kHeadDim);
        hipMemcpy(a2.data(), d_out, a2.size() * sizeof(float), hipMemcpyDeviceToHost);
        double num = 0, den = 0, md = 0;
        for (std::size_t i = 0; i < a2.size(); ++i) {
          const double d = (double)a2[i] - (double)sp[i];
          num += d * d;
          den += (double)a2[i] * a2[i] + (double)sp[i] * sp[i];
          md = std::max(md, std::fabs(d));
        }
        split_rel = std::sqrt(num / den);
        split_maxd = md;
        ms_split = time_ms(
            [&] {
              (void)rdna4::attn_launch_split(d_q, d_k, d_v, d_out2, d_partial, t, kHeads, kHeadsKv,
                                             kHeadDim, kScale, kt, kt, n_splits);
            },
            5, 300.0);
      }
    }
    hipFree(d_partial);

    // ------------------------------------------------------------------
    // Split-count x warps-per-CTA sweep. The engine's auto policy is
    // `keys / 2048` capped at 16, so at 4K keys it runs the UNSplit kernel with
    // 24 CTAs on 64 CUs; the question this table answers is what that costs.
    // WPB is a template knob (attn.cuh) and changing it changes the number of
    // partial slices merged per CTA, i.e. it is a numeric-equivalence change.
    // ------------------------------------------------------------------
    {
      std::vector<float> ref((std::size_t)kHeads * kHeadDim);
      hipMemcpy(ref.data(), d_out, ref.size() * sizeof(float), hipMemcpyDeviceToHost);
      float *dp = nullptr;
      if (hipMalloc(&dp, rdna4::attn_partial_bytes(kHeads, kHeadDim, 16)) != hipSuccess) {
        std::fprintf(stderr, "hipMalloc partial failed\n");
        return 1;
      }
      std::printf("  %-10s %-10s %10s %8s %12s %10s\n", "splits", "warps/CTA", "ms", "vs unsplit",
                  "rel-L2", "grid CTAs");
      for (int sp : {2, 4, 8, 16}) {
        for (int wpb : {8, 16, 32}) {
          const double ms = time_ms(
              [&] {
                (void)rdna4::attn_launch_split_wpb(d_q, d_k, d_v, d_out2, dp, t, kHeads, kHeadsKv,
                                                   kHeadDim, kScale, kt, kt, sp, wpb, nullptr);
              },
              5, 300.0);
          std::vector<float> got((std::size_t)kHeads * kHeadDim);
          hipMemcpy(got.data(), d_out2, got.size() * sizeof(float), hipMemcpyDeviceToHost);
          double num = 0, den = 0;
          for (std::size_t i = 0; i < got.size(); ++i) {
            const double d = (double)ref[i] - (double)got[i];
            num += d * d;
            den += (double)ref[i] * ref[i] + (double)got[i] * got[i];
          }
          std::printf("  %-10d %-10d %10.4f %7.2fx %12.2e %10d\n", sp, wpb, ms, ms_ship / ms,
                      std::sqrt(num / den), kHeads * sp);
        }
      }
      hipFree(dp);
    }

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
    // the two kernels must agree (they only differ in how the slices are grouped)
    std::vector<float> a((std::size_t)kHeads * kHeadDim), b((std::size_t)kHeads * kHeadDim);
    hipMemcpy(a.data(), d_out, a.size() * sizeof(float), hipMemcpyDeviceToHost);
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
  }
  return 0;
}
