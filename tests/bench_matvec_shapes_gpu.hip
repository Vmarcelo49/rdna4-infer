// M8 (ROCM study) — shape-aware matvec bench: what the shipping GEMV really
// costs per tensor, and whether ROWS per CTA / the register budget are free money.
//
// Why this tool exists (the three methodology gaps it closes):
//
//  1. `check-matvec-gpu --bench` measures ONE tensor per type — the largest of
//     that type — and applies the resulting GB/s to every tensor of the type.
//     Two problems with that: (a) the largest tensor of a type is often NOT
//     read per token (`token_embd.weight` is 556 MB of q3_K and decode reads a
//     single 2.2 KB row of it, the q6_K block of the MTP layer is never
//     executed), so a type's "share of the model" is not its share of the
//     per-token traffic; (b) it measures with dims[0]/dims[1] SWAPPED relative
//     to how `Graph::proj` calls the kernel, so the measured shape is not the
//     shape that ships.
//  2. Bandwidth here depends on the shape (rows x blocks-per-row): 48x160 (the
//     ssm_beta/ssm_alpha q8_0 matrices) can never fill 64 CUs with 24 CTAs,
//     while 17408x20 can. Timing every (type, shape) class of the real
//     inventory is the only way to know where the trunk matvec time goes.
//  3. A per-launch `hipEventRecord` + synchronize measures the DPM gap between
//     kernels as well as the kernel (this card drops to ~10 MHz SCLK when idle,
//     PLAN.md M2), which dominates the smallest tensors: a 261 KB q8_0 matrix
//     "costs" 10 us timed that way. The REPLAY mode below launches the whole
//     per-token set back-to-back with ONE event pair, i.e. exactly how the
//     graph calls them, and reports that number next to the sum of the
//     per-tensor ones. The difference is the measurement's own floor.
//
// usage:
//   bench-matvec-shapes-gpu <model.gguf> [--budget-mib N] [--reps N]
//                           [--rows a,b,c] [--minb a,b,c] [--no-ab]
//     --budget-mib  weight data to stage (default 12000 -> the whole inventory)
//     --rows        A/B the rows-per-CTA knob (bit-exact: it only moves which CTA
//                   computes which row)
//     --minb        A/B the __launch_bounds__ register budget (bit-exact: it
//                   only caps the register allocator)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"

namespace {

using rdna4::block_q8_1;

const char *dt_name(int dt) {
  switch (dt) {
    case 1: return "q8_0";    case 2: return "q2_k";   case 3: return "q3_k";
    case 4: return "q4_k";    case 5: return "q5_k";   case 6: return "q6_k";
    case 7: return "iq2_xxs"; case 8: return "iq2_xs"; case 9: return "iq3_xxs";
    case 10: return "iq1_s";  case 11: return "iq4_nl"; case 12: return "iq3_s";
    case 13: return "iq2_s";  case 14: return "iq4_xs"; default: return "?";
  }
}

struct Tensor {
  std::string name;
  int dt = 0;
  std::int64_t nrows = 0;  // output rows
  std::int64_t ncols = 0;  // reduction dim (== dims[0] of the GGUF tensor)
  std::int64_t bpr = 0;    // blocks per row (ncols / qk)
  double bytes = 0;
  void *d_w = nullptr;
};

void fill_x(std::vector<float> &x, std::int64_t n, unsigned seed) {
  x.resize(n);
  unsigned s = seed;
  for (std::int64_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    x[i] = ((float)(s >> 8) / (float)(1u << 24) - 0.5f) * 3.0f;
  }
}

// The launch used by every A/B variant:
//   kind 0 = shipping (per-type rows/WPR/ILP)
//   1 = forced rows, 2 = forced __launch_bounds__ min-blocks
//   3 = single-accumulator unroll (bit-exact MLP), 4 = L2 prefetch
//   5 = forced ILP (independent accumulators; changes the summation order)
//   6 = the shipping launch AGAIN, as its own variant -> the noise floor of this
//       very bench (same kernel, same shapes, measured independently)
//   7 = the LANDED tuning candidate: UNROLL=2 for the types whose shipping ILP is
//       already 1 (bit-exact: same ops, same order, more loads in flight) and the
//       shipping launch for every other type.
inline bool launch_variant(int kind, int value, const Tensor &t, const block_q8_1 *d_a, float *d_y) {
  if (kind == 0 || kind == 6)
    return rdna4::matvec_launch(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  if (kind == 1)
    return rdna4::matvec_launch_rows(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value);
  if (kind == 3)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value,
                                       /*pf=*/false);
  if (kind == 4)
    return rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, 1,
                                       /*pf=*/true);
  if (kind == 5)
    return rdna4::matvec_launch_tuned(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value,
                                      /*pf=*/false);
  if (kind == 8)  // LUT do tipo em LDS (mesma forma; ver matvec.cuh / journal §2)
    return rdna4::matvec_launch_lut_lds(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  if (kind == 9)  // LUT em LDS com UNROLL forcado (mede se a LDS muda o otimo de unroll)
    return rdna4::matvec_launch_lut_lds_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr,
                                               value);
  if (kind == 7)
    return rdna4::matvec_default_ilp(t.dt) == 1
               ? rdna4::matvec_launch_unroll(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, 2,
                                             /*pf=*/false)
               : rdna4::matvec_launch(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr);
  return rdna4::matvec_launch_minb(t.dt, t.d_w, d_a, d_y, t.nrows, t.ncols, nullptr, value);
}

__global__ void null_kernel(float *p) {
  if (p) *p = 1.0f;
}

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

struct Variant {
  const char *name = "ship";
  int kind = 0;   // 0 = ship, 1 = rows, 2 = minb, 3 = unroll, 4 = prefetch, 5 = ilp,
                  // 6 = ship again (noise floor), 7 = landed tuning candidate
  int value = 0;
};

// ---------------------------------------------------------------------
// R9 prototype (BENCH-ONLY, never shipped): stream-K split for the small-N
// trunk GEMV tail.
//
// Target (this bench's own table): classes whose rows are short
// (nrows <= 1024), whose CTA count cannot fill 64 CUs
// (ceil(nrows/shipping ROWS) < 64, e.g. q8_0 48x160 = 24 CTAs), or whose
// measured shipping rate sits below the trunk (~450 GB/s: the short-K
// bpr<=24 classes). Mechanism, exactly like attn_split (attn.cuh:496-660):
// split each row's K range over S CTAs; every split CTA walks its chunk with
// the SAME per-lane mapping (slot/kqs), ILP/UNROLL structure and LUT source
// the shipping kernel uses for that type, publishes one partial per
// (row, split), and a second kernel reduces the S partials. No TH_NT loads
// anywhere (measured dead); default cache policy on all loads; the shipping
// kernel's grid is untouched (separate kernels, separate launches).
//
// NOT bit-exact by construction: the cross-CTA summation order changes, so
// the gate is rel-L2 <= 1e-5 (attn_split-style policy gate), never memcmp.
// ---------------------------------------------------------------------
template <class T, class LUT, int ROWS, int WPR, int ILP, int UNROLL>
__global__ void sk_part_kernel(const void *__restrict__ vx, const block_q8_1 *__restrict__ vy,
                               float *__restrict__ partial, int64_t nrows, int64_t bpr,
                               int n_splits) {
  constexpr int vdr = T::vdr;
  constexpr int qi = T::qi;
  constexpr int qk = T::qk;
  constexpr int slots_per_block = qi / vdr;
  constexpr int blocks_per_iter = vdr * (WPR * 32) / qi;

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int row_group = warp / WPR;
  const int w_in_group = warp % WPR;
  const int tg = w_in_group * 32 + lane;

  const int g = (int)blockIdx.x;
  const int s = (int)blockIdx.y;
  const int row_raw = g * ROWS + row_group;
  const bool active = row_raw < nrows;
  const int row = active ? row_raw : 0;

  const char *rowp =
      (const char *)vx + (int64_t)row * bpr * (int64_t)sizeof(typename T::block_t);
  const int kqs = vdr * (tg % slots_per_block);
  const int slot = tg / slots_per_block;

  __shared__ uint32_t s_lut[LUT::words > 0 ? LUT::words : 1];
  if (LUT::words > 0) {
    const uint32_t *src = LUT::src_words();
    for (int i = tid; i < LUT::words; i += ROWS * WPR * 32) s_lut[i] = src[i];
    __syncthreads();
  }

  // Contiguous K-chunk for this split; the union over s covers [0, bpr).
  // An empty split (S > bpr) publishes 0 and the merge adds nothing.
  const int64_t chunk = (bpr + n_splits - 1) / n_splits;
  const int64_t k_lo = (int64_t)s * chunk;
  int64_t k_hi = k_lo + chunk;
  if (k_hi > bpr) k_hi = bpr;

  float acc[ILP];
#pragma unroll
  for (int i = 0; i < ILP; ++i) acc[i] = 0.0f;

  // First block of this warp's residue class inside the chunk.
  int64_t kb = slot;
  if (kb < k_lo) kb += ((k_lo - kb + blocks_per_iter - 1) / blocks_per_iter) * blocks_per_iter;
  if (UNROLL == 1) {
    for (; kb + (ILP - 1) * blocks_per_iter < k_hi; kb += ILP * blocks_per_iter) {
#pragma unroll
      for (int u = 0; u < ILP; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        acc[u] += T::dot_lut((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs,
                              (const void *)s_lut);
      }
    }
  } else {
    for (; kb + (UNROLL - 1) * blocks_per_iter < k_hi; kb += UNROLL * blocks_per_iter) {
#pragma unroll
      for (int u = 0; u < UNROLL; ++u) {
        const int64_t k = kb + u * blocks_per_iter;
        acc[0] += T::dot_lut((const void *)rowp, vy + k * (qk / QK8_1), (const int)k, kqs,
                              (const void *)s_lut);
      }
    }
  }
  for (; kb < k_hi; kb += blocks_per_iter) {
    acc[0] += T::dot_lut((const void *)rowp, vy + kb * (qk / QK8_1), (const int)kb, kqs,
                          (const void *)s_lut);
  }
  float sum = 0.0f;
#pragma unroll
  for (int i = 0; i < ILP; ++i) sum += acc[i];
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor(sum, off);

  if (WPR == 1) {
    if (active && lane == 0) partial[(int64_t)row * n_splits + s] = sum;
    return;
  }
  __shared__ float part[ROWS][WPR];
  if (lane == 0) part[row_group][w_in_group] = sum;
  __syncthreads();
  if (active && w_in_group == 0 && lane == 0) {
    float t = 0.0f;
#pragma unroll
    for (int i = 0; i < WPR; ++i) t += part[row_group][i];
    partial[(int64_t)row * n_splits + s] = t;
  }
}

// Stage 2: one warp per row reduces the S partials in split order.
__global__ void sk_merge_kernel(const float *__restrict__ partial, float *__restrict__ dst,
                                int64_t nrows, int n_splits) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = (int)blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= nrows) return;
  float t = 0.0f;
  const float *ps = partial + (int64_t)row * n_splits;
  for (int s = 0; s < n_splits; ++s) t += ps[s];
  if (lane == 0) dst[row] = t;
}

template <class T, class LUT, int R, int W, int I, int U>
inline bool sk_launch_t(const void *d_w, const block_q8_1 *d_a, float *d_o, float *d_part,
                        int64_t nrows, int64_t bpr, int n_splits, hipStream_t stream) {
  const int groups = (int)((nrows + R - 1) / R);
  dim3 grid((unsigned)groups, (unsigned)n_splits);
  sk_part_kernel<T, LUT, R, W, I, U>
      <<<grid, R * W * 32, 0, stream>>>(d_w, d_a, d_part, nrows, bpr, n_splits);
  if (hipGetLastError() != hipSuccess) return false;
  const int mgrid = (int)((nrows + 3) / 4);
  sk_merge_kernel<<<mgrid, 128, 0, stream>>>(d_part, d_o, nrows, n_splits);
  return hipGetLastError() == hipSuccess;
}

template <int Dt, class Traits, class LUTt, int QK>
inline bool sk_launch_dt(const void *d_w, const block_q8_1 *d_a, float *d_o, float *d_part,
                         int64_t nrows, int64_t ncols, int n_splits, hipStream_t stream) {
  return sk_launch_t<Traits, LUTt, rdna4::MtShape<Dt>::rows, rdna4::MtShape<Dt>::wpr,
                     rdna4::MtIlp<Dt>::value, rdna4::MtUnroll<Dt>::value>(
      d_w, d_a, d_o, d_part, nrows, ncols / QK, n_splits, stream);
}

// Same traits (incl. the LUT-in-LDS rule) and shape the shipping kernel uses
// for each type, so any measured delta is the split + merge and nothing else.
inline bool streamk_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                           float *d_part, int64_t nrows, int64_t ncols, int n_splits,
                           hipStream_t stream) {
  if (n_splits < 1 || nrows < 1 || ncols < 1) return false;
#define RD_SK(Traits, LUTt, Dt, QK)                                   \
  case Dt:                                                            \
    return sk_launch_dt<Dt, Traits, LUTt, QK>(d_w, d_a, d_o, d_part,   \
                                              nrows, ncols, n_splits, stream)
  switch (dt) {
    RD_SK(rdna4::TQ8_0, rdna4::NoLut, 1, 32);
    RD_SK(rdna4::TQ2K, rdna4::NoLut, 2, 256);
    RD_SK(rdna4::TQ3K, rdna4::NoLut, 3, 256);
    RD_SK(rdna4::TQ4K, rdna4::NoLut, 4, 256);
    RD_SK(rdna4::TQ5K, rdna4::NoLut, 5, 256);
    RD_SK(rdna4::TQ6K, rdna4::NoLut, 6, 256);
    RD_SK(rdna4::TIQ2XXS_S, rdna4::NoLut, 7, 256);
    RD_SK(rdna4::TIQ2XS_S, rdna4::NoLut, 8, 256);
    case 9:
      if (rdna4::matvec_lut_lds_ok(dt, nrows, ncols))
        return sk_launch_dt<9, rdna4::TIQ3XXS_LDS, rdna4::LutIq3XXS, 256>(
            d_w, d_a, d_o, d_part, nrows, ncols, n_splits, stream);
      return sk_launch_dt<9, rdna4::TIQ3XXS_S, rdna4::NoLut, 256>(d_w, d_a, d_o, d_part, nrows,
                                                                 ncols, n_splits, stream);
    RD_SK(rdna4::TIQ1S, rdna4::NoLut, 10, 256);
    RD_SK(rdna4::TIQ4NL, rdna4::NoLut, 11, 32);
    case 12:
      if (rdna4::matvec_lut_lds_ok(dt, nrows, ncols))
        return sk_launch_dt<12, rdna4::TIQ3S_LDS, rdna4::LutIq3S, 256>(
            d_w, d_a, d_o, d_part, nrows, ncols, n_splits, stream);
      return sk_launch_dt<12, rdna4::TIQ3S_S, rdna4::NoLut, 256>(d_w, d_a, d_o, d_part, nrows,
                                                                ncols, n_splits, stream);
    RD_SK(rdna4::TIQ2S_S, rdna4::NoLut, 13, 256);
    RD_SK(rdna4::TIQ4XS, rdna4::NoLut, 14, 256);
    default:
      return false;
  }
#undef RD_SK
}

// ---------------------------------------------------------------------
// R8 scalar-broadcast + q3_K old-vs-new prototypes (BENCH-ONLY, never
// shipped). Gated on env vars so default binary behavior is unchanged:
//   RD_R8=1   explicit __builtin_amdgcn_readfirstlane broadcast of the
//             warp-uniform scalars (q4_K header words, iq3_s block d)
//             vs the shipping dot, same launch geometry, interleaved,
//             with a ship-vs-ship noise cell.
//   RD_Q3K=1  resurrected pre-cb4f3e0 q3_K __vsubss4 body vs the shipping
//             dp4a-linearity body (vecdotq.cuh:279), memcmp + GB/s.
// Codegen context (--save-temps, gfx1201 -O3, CPU-side probes): a
// warp-uniform address+value already compiles to s_load (SMEM/K$) with no
// builtin at all, and an explicit readfirstlane folds to identical ISA, so
// the R8 source change is expected to be a no-op; this section measures
// that (1.00x kill) instead of asserting it. scales[]/d8/u addresses
// depend on kqs (the lane index) and are structurally vector -- not
// uniform, not R8 candidates. Only block d/dm/header bytes are uniform.
// ---------------------------------------------------------------------

// R8-A: q4_K dot, byte-identical to vec_dot_q4_K_q8_1 except the four
// warp-uniform header words are explicitly broadcast from lane 0.
static __device__ __forceinline__ float bench_q4k_r8_dot(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int &kbx, const int &iqs) {
  const rdna4::block_q4_K *bq4_K = (const rdna4::block_q4_K *)vbq + kbx;
  int v[2];
  int u[2 * QR4_K];
  float d8[QR4_K];
  const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
  const int *q4 = (const int *)(bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
  v[0] = q4[0];
  v[1] = q4[4];
  uint4 hdr = ((const uint4 *)bq4_K)[0];
  hdr.x = __builtin_amdgcn_readfirstlane(hdr.x);
  hdr.y = __builtin_amdgcn_readfirstlane(hdr.y);
  hdr.z = __builtin_amdgcn_readfirstlane(hdr.z);
  hdr.w = __builtin_amdgcn_readfirstlane(hdr.w);
  rdna4::half2 dmh;
  dmh.d = (uint16_t)(hdr.x & 0xFFFFu);
  dmh.dmin = (uint16_t)(hdr.x >> 16);
  const int j = bq8_offset / 2;
  const int jm = j & 1;
  const uint32_t s0 = (hdr.y >> (jm << 4)) & 0xFFFFu;
  const uint32_t s2 = (hdr.z >> (jm << 4)) & 0xFFFFu;
  const uint32_t s4 = (hdr.w >> (jm << 4)) & 0xFFFFu;
  const uint32_t hi = (uint32_t) -(int32_t)(j >= 2);
  uint16_t aux[2];
  aux[0] = (uint16_t)(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
  aux[1] = (uint16_t)(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
  const uint8_t *sc = (const uint8_t *)aux;
  const uint8_t *m = sc + 2;
  for (int i = 0; i < QR4_K; ++i) {
    const block_q8_1 *bq8i = bq8_1 + bq8_offset + i;
    d8[i] = rdna4::fp16_to_float((uint16_t)(bq8i->ds & 0xFFFFu));
    const int *q8 = (const int *)bq8i->qs + ((iqs / 2) % 4);
    u[2 * i + 0] = q8[0];
    u[2 * i + 1] = q8[4];
  }
  return rdna4::vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dmh, d8);
}

struct BenchQ4K_R8 {
  static constexpr int qk = 256, qi = QI4_K, vdr = VDR_Q4_K_Q8_1_MMVQ;
  using block_t = rdna4::block_q4_K;
  static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,
                                              const int &kbx, const int &iqs) {
    return bench_q4k_r8_dot(vbq, a, kbx, iqs);
  }
  static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a,
                                                  const int &kbx, const int &iqs,
                                                  const void * /*lut*/) {
    return dot(vbq, a, kbx, iqs);
  }
};

// R8-B: iq3_s perm dot with LUT pointer (mirrors
// vec_dot_iq3_s_q8_1_perm_lut), except the warp-uniform block scale d is
// explicitly broadcast from lane 0.
static __device__ __forceinline__ float bench_iq3s_r8_dot_lut(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1, const int &kbx,
    const int &iqs, const void *__restrict__ lut) {
  const rdna4::block_iq3_s *bq3 = (const rdna4::block_iq3_s *)vbq + kbx;
  const int2 qs_packed =
      make_int2(rdna4::get_int_b2(bq3->qs, iqs + 0), rdna4::get_int_b2(bq3->qs, iqs + 1));
  const uint8_t *qs = (const uint8_t *)&qs_packed;
  const int qh = bq3->qh[iqs / 2];
  const int signs_packed_32 = rdna4::get_int_b2(bq3->signs, iqs / 2);
  const uint8_t *sp = (const uint8_t *)&signs_packed_32;
  int sumi_pos = 0;
  int sumi_all = 0;
#pragma unroll
  for (int l0 = 0; l0 < 8; l0 += 2) {
    const uint32_t gx = ((const uint32_t *)lut)[qs[l0 + 0] | ((qh << (8 - l0)) & 0x100)];
    const uint32_t gy = ((const uint32_t *)lut)[qs[l0 + 1] | ((qh << (7 - l0)) & 0x100)];
    const uint8_t sb = sp[l0 / 2];
    const uint32_t sel_x = (((uint32_t)(sb & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;
    const uint32_t sel_y = (((uint32_t)((sb >> 4) & 0x0F) * 0x00810204u) & 0x04040404u) | 0x03020100u;
    const int gx_pos = (int)__builtin_amdgcn_perm(0u, gx, sel_x);
    const int gy_pos = (int)__builtin_amdgcn_perm(0u, gy, sel_y);
    const int u0 = rdna4::get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
    const int u1 = rdna4::get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
    sumi_pos = rdna4::ggml_cuda_dp4a(gx_pos, u0, sumi_pos);
    sumi_pos = rdna4::ggml_cuda_dp4a(gy_pos, u1, sumi_pos);
    sumi_all = rdna4::ggml_cuda_dp4a((int)gx, u0, sumi_all);
    sumi_all = rdna4::ggml_cuda_dp4a((int)gy, u1, sumi_all);
  }
  int sumi = 2 * sumi_pos - sumi_all;
  sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
  uint32_t draw = bq3->d;  // warp-uniform: same block for every lane
  draw = __builtin_amdgcn_readfirstlane(draw);
  const float d = rdna4::fp16_to_float((uint16_t)draw) *
                  rdna4::fp16_to_float((uint16_t)((bq8_1[iqs / 2].ds) & 0xFFFFu));
  return d * sumi;
}

struct BenchIQ3S_R8 {
  static constexpr int qk = 256, qi = QI3_S, vdr = VDR_IQ3_S_Q8_1_MMVQ;
  using block_t = rdna4::block_iq3_s;
  static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,
                                              const int &kbx, const int &iqs) {
    return bench_iq3s_r8_dot_lut(vbq, a, kbx, iqs, (const void *)rdna4::iq3s_grid);
  }
  static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a,
                                                  const int &kbx, const int &iqs,
                                                  const void *lut) {
    return bench_iq3s_r8_dot_lut(vbq, a, kbx, iqs, lut);
  }
};

// q3_K OLD body (pre-cb4f3e0): per-byte saturating subtract emulation.
// Resurrected verbatim from 23c9e03 for the old-vs-new A/B; the shipping
// kernel uses the dp4a-linearity form (vecdotq.cuh:279).
static __device__ __forceinline__ float bench_q3k_old_impl(
    const int &vl, const int &vh, const int *__restrict__ u,
    const uint8_t *__restrict__ scales, const int &scale_offset, const float &d3,
    const float *__restrict__ d8) {
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    const int isc = scale_offset + 2 * i;
    const int isc_low = isc % (QK_K / 32);
    const int sc_shift_low = 4 * (isc / (QK_K / 32));
    const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xF;
    const int isc_high = isc % (QK_K / 64);
    const int sc_shift_high = 2 * (isc / (QK_K / 64));
    const int sc_high = ((scales[(QK_K / 32) + isc_high] >> sc_shift_high) & 3) << 4;
    const int sc = (sc_low | sc_high) - 32;
    const int vil = (vl >> (2 * i)) & 0x03030303;
    const int vih = ((vh >> i) << 2) & 0x04040404;
    const int vi = rdna4::__vsubss4(vil, vih);
    sumf += d8[i] * (rdna4::ggml_cuda_dp4a(vi, u[i], 0) * sc);
  }
  return d3 * sumf;
}

static __device__ __forceinline__ float bench_q3k_old_dot(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int &kbx, const int &iqs) {
  const rdna4::block_q3_K *bq3_K = (const rdna4::block_q3_K *)vbq + kbx;
  const int bq8_offset = QR3_K * (iqs / (QI3_K / 2));
  const int scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);
  const float d = rdna4::fp16_to_float(bq3_K->d);
  const int vl = rdna4::get_int_b2(bq3_K->qs, iqs);
  const int vh = ~rdna4::get_int_b2(bq3_K->hmask, iqs % (QI3_K / 2)) >> bq8_offset;
  int u[QR3_K];
  float d8[QR3_K];
#pragma unroll
  for (int i = 0; i < QR3_K; ++i) {
    u[i] = rdna4::get_int_b4(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
    d8[i] = rdna4::fp16_to_float((uint16_t)((bq8_1[bq8_offset + i].ds) & 0xFFFFu));
  }
  return bench_q3k_old_impl(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

struct BenchQ3K_OLD {
  static constexpr int qk = 256, qi = QI3_K, vdr = VDR_Q3_K_Q8_1_MMVQ;
  using block_t = rdna4::block_q3_K;
  static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,
                                              const int &kbx, const int &iqs) {
    return bench_q3k_old_dot(vbq, a, kbx, iqs);
  }
  static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a,
                                                  const int &kbx, const int &iqs,
                                                  const void * /*lut*/) {
    return dot(vbq, a, kbx, iqs);
  }
};

inline bool bench_r8_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                            int64_t nrows, int64_t ncols, hipStream_t stream) {
  // Same geometry (ROWS/WPR/ILP/MINB/UNROLL/LUT-source) the shipping kernel
  // uses for the type; only the dot body differs (explicit broadcast).
  if (dt == 4)
    return rdna4::launch_gen<BenchQ4K_R8, rdna4::MtShape<4>::rows, rdna4::MtShape<4>::wpr,
                             rdna4::MtIlp<4>::value, false, rdna4::MtMinb<4>::value,
                             rdna4::MtUnroll<4>::value>(d_w, d_a, d_o, nrows, ncols / 256,
                                                        stream);
  if (dt == 12) {
    if (rdna4::matvec_lut_lds_ok(dt, nrows, ncols))
      return rdna4::launch_gen<BenchIQ3S_R8, rdna4::MtShape<12>::rows,
                               rdna4::MtShape<12>::wpr, rdna4::MtIlp<12>::value, false,
                               rdna4::MtMinb<12>::value, rdna4::MtUnroll<12>::value,
                               rdna4::LutIq3S>(d_w, d_a, d_o, nrows, ncols / 256, stream);
    return rdna4::launch_gen<BenchIQ3S_R8, rdna4::MtShape<12>::rows, rdna4::MtShape<12>::wpr,
                             rdna4::MtIlp<12>::value, false, rdna4::MtMinb<12>::value,
                             rdna4::MtUnroll<12>::value>(d_w, d_a, d_o, nrows, ncols / 256,
                                                         stream);
  }
  return false;
}

inline bool bench_q3k_old_launch(const void *d_w, const block_q8_1 *d_a, float *d_o,
                                 int64_t nrows, int64_t ncols, hipStream_t stream) {
  return rdna4::launch_gen<BenchQ3K_OLD, rdna4::MtShape<3>::rows, rdna4::MtShape<3>::wpr,
                           rdna4::MtIlp<3>::value, false, rdna4::MtMinb<3>::value,
                           rdna4::MtUnroll<3>::value>(d_w, d_a, d_o, nrows, ncols / 256,
                                                      stream);
}

// H1 candidate 2 (BENCH-ONLY, never shipped): q6_K desaturation removal, the
// same dp4a-linearity pattern shipped for q3_K (vecdotq.cuh:271-279).
// Exactness: vil bytes are 4-bit fields (0..15) and vih bytes are
// 0x00/0x10/0x20/0x30, so (vil|vih) bytes are 0..63 and (vil|vih)-32 lands in
// [-32,31]: int8 never saturates, hence
//   dp4a((vil|vih)-32, u) == dp4a(vil|vih, u) - 32*dp4a(0x01010101, u)
// holds exactly in int32 (ggml_cuda_dp4a accumulates without saturation).
// The float tail (dot*sc, d8[i], sumf order, block d) is untouched, hence
// bit-identical. Load sequence mirrors vec_dot_q6_K_q8_1 verbatim.
static __device__ __forceinline__ float bench_q6k_lin_dot(
    const void *__restrict__ vbq, const block_q8_1 *__restrict__ bq8_1,
    const int &kbx, const int &iqs) {
  const rdna4::block_q6_K *bq6_K = (const rdna4::block_q6_K *)vbq + kbx;
  const int bq8_offset =
      2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
  const int scale_offset =
      (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
  const int vh_shift = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));
  const int vl = rdna4::get_int_b2(bq6_K->ql, iqs);
  const int vh = rdna4::get_int_b2(bq6_K->qh,
                                   (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >>
                 vh_shift;
  const int8_t *scales = bq6_K->scales + scale_offset;
  int u[QR6_K];
  float d8[QR6_K];
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    u[i] = rdna4::get_int_b4(bq8_1[bq8_offset + 2 * i].qs, iqs % QI8_1);
    d8[i] = rdna4::fp16_to_float((uint16_t)((bq8_1[bq8_offset + 2 * i].ds) & 0xFFFFu));
  }
  float sumf = 0.0f;
#pragma unroll
  for (int i = 0; i < QR6_K; ++i) {
    const int sc = scales[4 * i];
    const int vil = (vl >> (4 * i)) & 0x0F0F0F0F;
    const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
    const int vor = vil | vih;
    const int dot = rdna4::ggml_cuda_dp4a(vor, u[i], 0) -
                    32 * rdna4::ggml_cuda_dp4a(0x01010101, u[i], 0);
    sumf += d8[i] * (dot * sc);  // same tail as the shipping body
  }
  return rdna4::fp16_to_float(bq6_K->d) * sumf;
}

struct BenchQ6K_LIN {
  static constexpr int qk = 256, qi = QI6_K, vdr = VDR_Q6_K_Q8_1_MMVQ;
  using block_t = rdna4::block_q6_K;
  static __device__ __forceinline__ float dot(const void *vbq, const block_q8_1 *a,
                                              const int &kbx, const int &iqs) {
    return bench_q6k_lin_dot(vbq, a, kbx, iqs);
  }
  static __device__ __forceinline__ float dot_lut(const void *vbq, const block_q8_1 *a,
                                                  const int &kbx, const int &iqs,
                                                  const void * /*lut*/) {
    return dot(vbq, a, kbx, iqs);
  }
};

inline bool bench_q6k_lin_launch(const void *d_w, const block_q8_1 *d_a, float *d_o,
                                 int64_t nrows, int64_t ncols, hipStream_t stream) {
  return rdna4::launch_gen<BenchQ6K_LIN, rdna4::MtShape<6>::rows, rdna4::MtShape<6>::wpr,
                           rdna4::MtIlp<6>::value, false, rdna4::MtMinb<6>::value,
                           rdna4::MtUnroll<6>::value>(d_w, d_a, d_o, nrows, ncols / 256,
                                                      stream);
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr,
                 "usage: bench-matvec-shapes-gpu <model.gguf> [--budget-mib N] [--reps N] "
                 "[--rows a,b,c] [--minb a,b,c] [--unroll 2,4] [--pf] [--ilp 1,2,4] "
                 "[--repeat-ship] [--land] [--lut-lds] [--lut-lds-unroll 1,2,4] [--no-ab]\n"
                 "       [--batch 1,2,4,8,16]  (sweep tokens-per-weight-pass; N=1 = GEMV)\n"
                 "       [--batch-unroll 2,4]  (bit-exact MLP knob inside the batch kernel)\n"
                 "       [--batch-cap]         (same pass on two streams: spare capacity?)\n"
                 "       [--streamk 1,2,4]     (R9 prototype only: K-split x S + merge, numeric gate)\n");
    return 1;
  }
  double budget_mib = 12000.0;
  int reps = 4;
  bool do_ab = true;
  bool pf_only = false;
  bool repeat_ship = false;
  bool land = false;
  bool lut_lds = false;
  std::vector<int> lut_lds_unroll;
  std::vector<int> row_variants, minb_variants, unroll_variants, ilp_variants;
  std::vector<int> batch_list, unroll_list;
  std::vector<int> sk_splits;  // R9 prototype (bench-only): split counts for the stream-K A/B
  bool cap_probe = false;
  for (int i = 2; i < argc; ++i) {
    auto list = [&](std::vector<int> &out) {
      const char *v = argv[++i];
      while (*v) {
        out.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    };
    if (std::strcmp(argv[i], "--budget-mib") == 0 && i + 1 < argc) {
      budget_mib = std::atof(argv[++i]);
    } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
      list(row_variants);
    } else if (std::strcmp(argv[i], "--minb") == 0 && i + 1 < argc) {
      list(minb_variants);
    } else if (std::strcmp(argv[i], "--unroll") == 0 && i + 1 < argc) {
      list(unroll_variants);
    } else if (std::strcmp(argv[i], "--ilp") == 0 && i + 1 < argc) {
      list(ilp_variants);
    } else if (std::strcmp(argv[i], "--pf") == 0) {
      unroll_variants.push_back(1);
      pf_only = true;
    } else if (std::strcmp(argv[i], "--repeat-ship") == 0) {
      repeat_ship = true;
    } else if (std::strcmp(argv[i], "--land") == 0) {
      land = true;
    } else if (std::strcmp(argv[i], "--lut-lds") == 0) {
      lut_lds = true;
    } else if (std::strcmp(argv[i], "--lut-lds-unroll") == 0 && i + 1 < argc) {
      list(lut_lds_unroll);  // aceita "1,2,4" (o atoi direto leria so o 1)
    } else if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
      // --batch a,b,c: sweep the number of tokens that share one weight pass
      // (prefill / MTP). N=1 in the list times the GEMV path as the reference.
      list(batch_list);
    } else if (std::strcmp(argv[i], "--batch-unroll") == 0 && i + 1 < argc) {
      // UNROLL forcado no lote (bit-exato), so' nos tipos com kMtIlp == 1.
      list(unroll_list);
    } else if (std::strcmp(argv[i], "--batch-cap") == 0) {
      // Sondagem de capacidade: a mesma passagem em duas streams (sobreposicao
      // possivel = latencia; 2x = throughput saturado).
      cap_probe = true;
    } else if (std::strcmp(argv[i], "--streamk") == 0 && i + 1 < argc) {
      // R9 prototype (bench-only): split counts for the stream-K A/B.
      list(sk_splits);
    } else if (std::strcmp(argv[i], "--no-ab") == 0) {
      do_ab = false;
    } else {
      std::fprintf(stderr, "unknown argument %s\n", argv[i]);
      return 1;
    }
  }
  if (reps < 1) reps = 1;
  if (!do_ab) {
    row_variants.clear();
    minb_variants.clear();
    unroll_variants.clear();
    ilp_variants.clear();
    repeat_ship = false;
    land = false;
    lut_lds = false;
    lut_lds_unroll.clear();
    sk_splits.clear();
  }
  (void)pf_only;

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(argv[1], err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }

  // ---------------------------------------------------------------------
  // Inventory: exactly the tensors the decode path streams per token.
  // `token_embd` is skipped (one row per token), the MTP block blk.<block_count
  // - 1> is skipped (never executed by this engine), F32 has no matvec.
  // ---------------------------------------------------------------------
  std::vector<Tensor> all;
  double grand_total_bytes = 0, skipped_bytes = 0;
  for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
    const auto &info = ld.meta().tensors[i];
    const std::string &name = info.name;
    rdna4::MatvecShape shape{};
    const int dt = (int)ld.dtype(i);
    if (name == "token_embd.weight" || name == "output_norm.weight" ||
        name.rfind("blk.64.", 0) == 0 || !rdna4::matvec_shape(dt, shape) ||
        info.dims.size() != 2 || (std::int64_t)info.dims[0] % shape.qk != 0) {
      skipped_bytes += (double)ld.tensor_bytes(i);
      continue;
    }
    Tensor t;
    t.name = name;
    t.dt = dt;
    t.ncols = (std::int64_t)info.dims[0];
    t.nrows = (std::int64_t)info.dims[1];
    t.bpr = t.ncols / shape.qk;
    t.bytes = (double)ld.tensor_bytes(i);
    grand_total_bytes += t.bytes;
    all.push_back(t);
  }
  std::printf("inventory: %zu tensors read per token, %.3f GB (%.3f GiB); skipped %.3f GB "
              "(token_embd / MTP / f32)\n",
              all.size(), grand_total_bytes / 1e9, grand_total_bytes / (1024.0 * 1024 * 1024),
              skipped_bytes / 1e9);

  std::sort(all.begin(), all.end(),
            [](const Tensor &a, const Tensor &b) { return a.bytes > b.bytes; });
  const double budget = budget_mib * 1024.0 * 1024.0;
  std::vector<Tensor> sel;
  double used = 0;
  for (const Tensor &t : all) {
    if (used + t.bytes > budget) continue;
    sel.push_back(t);
    used += t.bytes;
  }
  std::printf("staged: %zu tensors, %.3f GiB\n", sel.size(), used / (1024.0 * 1024 * 1024));

  float *d_y = nullptr;
  std::int64_t max_rows = 0;
  for (const Tensor &t : sel) max_rows = std::max(max_rows, t.nrows);
  if (hipMalloc(&d_y, (std::size_t)max_rows * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc d_y failed\n");
    return 1;
  }
  std::vector<std::int64_t> ncols_seen;
  for (const Tensor &t : sel)
    if (std::find(ncols_seen.begin(), ncols_seen.end(), t.ncols) == ncols_seen.end())
      ncols_seen.push_back(t.ncols);
  std::vector<std::pair<std::int64_t, block_q8_1 *>> acts;
  for (std::int64_t nc : ncols_seen) {
    std::vector<float> x;
    fill_x(x, nc, 999u);
    float *d_x = nullptr;
    block_q8_1 *d_a = nullptr;
    (void)hipMalloc(&d_x, (std::size_t)nc * sizeof(float));
    (void)hipMalloc(&d_a, (std::size_t)(nc / QK8_1) * sizeof(block_q8_1));
    (void)hipMemcpy(d_x, x.data(), (std::size_t)nc * sizeof(float), hipMemcpyHostToDevice);
    rdna4::quantize_q8_1_kernel<<<(unsigned)((nc / QK8_1 + 3) / 4), 128>>>(d_x, d_a, nc / QK8_1);
    hipDeviceSynchronize();
    (void)hipFree(d_x);
    acts.push_back({nc, d_a});
  }
  for (Tensor &t : sel) {
    std::vector<std::uint8_t> w;
    const auto *info = ld.find(t.name.c_str());
    if (!ld.load_tensor((std::size_t)(info - ld.meta().tensors.data()), w, err)) {
      std::fprintf(stderr, "%s: %s\n", t.name.c_str(), err.c_str());
      return 1;
    }
    if (hipMalloc(&t.d_w, w.size()) != hipSuccess) {
      std::fprintf(stderr, "hipMalloc %s (%.1f MB) failed\n", t.name.c_str(), w.size() / 1e6);
      return 1;
    }
    (void)hipMemcpy(t.d_w, w.data(), w.size(), hipMemcpyHostToDevice);
  }
  auto act_for = [&](std::int64_t nc) -> block_q8_1 * {
    for (auto &p : acts)
      if (p.first == nc) return p.second;
    return nullptr;
  };

  // Warm up until ~400 ms of GPU time has elapsed (DPM ramp; PLAN.md M2/M5).
  {
    double warm = 0;
    const int per_round = (int)std::min<std::size_t>(sel.size(), (std::size_t)64);
    EvPair ev;
    while (warm < 400.0) {
      warm += ev([&] {
        for (int i = 0; i < per_round; ++i) {
          const Tensor &t = sel[(std::size_t)i * sel.size() / per_round];
          (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols, nullptr);
        }
      });
    }
  }
  EvPair ev;

  // ---------------------------------------------------------------------
  // Launch floor: an empty kernel timed exactly like a tensor (record + sync),
  // and 1000 empty kernels launched back-to-back with one event pair. The first
  // is the floor of every per-tensor number below; the second is what a launch
  // costs when the queue is saturated (the engine's situation).
  // ---------------------------------------------------------------------
  float *d_dummy = nullptr;
  (void)hipMalloc(&d_dummy, sizeof(float));
  double null_once = 1e9, null_stream = 1e9;
  for (int r = 0; r < 5; ++r) {
    null_once = std::min(null_once, ev([&] { null_kernel<<<1, 32, 0, nullptr>>>(d_dummy); }));
    null_stream =
        std::min(null_stream, ev([&] {
                   for (int i = 0; i < 1000; ++i) null_kernel<<<1, 32, 0, nullptr>>>(d_dummy);
                 }) / 1000.0);
  }
  std::printf("launch floor: empty kernel %.3f ms timed with a sync, %.4f ms/launch "
              "back-to-back (1000 in one event pair)\n",
              null_once, null_stream);

  // ---------------------------------------------------------------------
  // REPLAY: one pass over the whole set with ONE event pair — this is how the
  // graph calls the matvecs (queued, no syncs), so it is the honest per-token
  // matvec number; the sum of the per-tensor measurements is inflated by the
  // sync floor above.
  // ---------------------------------------------------------------------
  double replay_ms = 1e9;
  for (int r = 0; r < reps; ++r) {
    replay_ms = std::min(replay_ms, ev([&] {
                           for (const Tensor &t : sel)
                             (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y,
                                                        t.nrows, t.ncols, nullptr);
                         }));
  }
  std::printf("replay (one queued pass over %zu tensors, min of %d): %.3f ms -> %.1f GB/s\n",
              sel.size(), reps, replay_ms, used / (replay_ms * 1e-3) / 1e9);

  // ---------------------------------------------------------------------
  // HIP GRAPH of the same 497-kernel pass. The engine issues ~2200 kernels per
  // token and the measured inter-kernel gap is ~3.6 us (null kernel, queued),
  // i.e. the launch tax is a large share of a 35 ms token -- so the question
  // "how much of that is avoidable?" needs an answer with the real sequence.
  // A captured graph replays the same kernels in the same order with the
  // driver's pre-built launch descriptors, so the difference IS the avoidable
  // part (llama.cpp ships with GGML_HIP_GRAPHS on).
  // ---------------------------------------------------------------------
  {
    hipStream_t cs = nullptr;
    if (hipStreamCreate(&cs) != hipSuccess) {
      std::printf("graph: stream create failed\n");
    } else {
      hipError_t ce = hipStreamBeginCapture(cs, hipStreamCaptureModeThreadLocal);
      if (ce != hipSuccess) {
        std::printf("graph: hipStreamBeginCapture failed (%s)\n", hipGetErrorString(ce));
      } else {
        for (const Tensor &t : sel)
          (void)rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols, cs);
        hipGraph_t graph = nullptr;
        ce = hipStreamEndCapture(cs, &graph);
        if (ce != hipSuccess || graph == nullptr) {
          std::printf("graph: hipStreamEndCapture failed (%s)\n", hipGetErrorString(ce));
        } else {
          hipGraphExec_t gexec = nullptr;
          ce = hipGraphInstantiate(&gexec, graph, nullptr, nullptr, 0);
          if (ce != hipSuccess) {
            std::printf("graph: instantiate failed (%s)\n", hipGetErrorString(ce));
          } else {
            double gms = 1e9;
            for (int r = 0; r < reps; ++r) {
              gms = std::min(gms, ev([&] { (void)hipGraphLaunch(gexec, nullptr); }));
            }
            std::printf("graph replay of the same %zu kernels: %.3f ms (%.1f GB/s) = %.2fx the "
                        "streamed replay (%.3f ms) -> %.2f ms/token of launch tax recoverable\n",
                        sel.size(), gms, used / (gms * 1e-3) / 1e9, replay_ms / gms, replay_ms,
                        replay_ms - gms);
            (void)hipGraphExecDestroy(gexec);
          }
          (void)hipGraphDestroy(graph);
        }
      }
      (void)hipStreamDestroy(cs);
    }
  }

  // ---------------------------------------------------------------------
  // BATCH (prefill / MTP): N tokens share ONE read of the weights. The question
  // this section answers is *which* side runs out first.
  //   - weight side: the CTA pass over a row-block is amortized over N, so a
  //     pass whose cost is flat in N is weight-bound (per-token ms falls ~1/N);
  //   - token side: activation blocks + dp4a scale with N, and so does the
  //     activation working set per CTA (N * ncols bytes; 5 KB per token at
  //     ncols=5120, i.e. 82 KB at N=16) -- far past L1. If ms/token grows with
  //     N, the token side binds, and the growth rate sizes the activation
  //     re-read traffic (the hypothesis the previous session left open,
  //     journal-kernels §10: "a hipotese a medir e' o trafego de re-leitura da
  //     ativacao por linha").
  // The N=1 row is the GEMV pass again -- the same kernel the decode path uses --
  // so the rows of this table are directly comparable to each other.
  // ---------------------------------------------------------------------
  if (!batch_list.empty()) {
    std::int64_t max_rows_b = 0, max_bpr = 0;
    for (const Tensor &t : sel) {
      max_rows_b = std::max(max_rows_b, t.nrows);
      max_bpr = std::max(max_bpr, t.bpr);
    }
    const int n_max = *std::max_element(batch_list.begin(), batch_list.end());
    block_q8_1 *d_ab = nullptr;
    float *d_yb = nullptr;
    if (hipMalloc(&d_ab, (std::size_t)n_max * max_bpr * sizeof(block_q8_1)) != hipSuccess ||
        hipMalloc(&d_yb, (std::size_t)n_max * max_rows_b * sizeof(float)) != hipSuccess) {
      std::printf("batch: staging alloc failed\n");
      return 1;
    }
    // One quantized activation row per token slot, back to back: exactly the
    // layout the graph uses (act_stride = ncols/32).
    {
      std::vector<float> x;
      fill_x(x, max_bpr * QK8_1, 4242u);
      for (int n = 0; n < n_max; ++n) {
        float *d_x = nullptr;
        (void)hipMalloc(&d_x, x.size() * sizeof(float));
        (void)hipMemcpy(d_x, x.data(), x.size() * sizeof(float), hipMemcpyHostToDevice);
        rdna4::quantize_q8_1_kernel<<<(unsigned)((max_bpr + 3) / 4), 128>>>(
            d_x, d_ab + (std::size_t)n * max_bpr, max_bpr);
        hipDeviceSynchronize();
        (void)hipFree(d_x);
      }
    }

    std::printf("\n=== BATCH === staged %.3f GiB, %zu tensors, launch cap %d\n",
                used / (1024.0 * 1024 * 1024), sel.size(), rdna4::matvec_batch_cap());
    std::printf("%5s %12s %12s %12s %14s %10s %12s %10s", "N", "pass_ms", "ms/token",
                "GB/s(pass)", "GB/s/token-eq", "vs GEMV", "reuse_ms", "reuse%");
    for (int U : unroll_list) std::printf(" %13s", U == 1 ? "unr1_ms" : (U == 2 ? "unr2_ms" : "unr4_ms"));
    std::printf("\n");
    double gemv_ms = 1e9, unroll_ref = 1e9;
    // `reuse` = act_stride 0: every token reads the SAME activation row. Same
    // kernel, same instruction count, same dp4a count, same weight traffic --
    // only the activation working set changes (5 KB instead of N * 5 KB), so
    // the difference against the dense pass is the cost of the activation
    // re-read traffic and nothing else. It is the control that separates
    // "issue-bound" from "cache/latency-bound" (journal-kernels §10).
    auto pass = [&](int N, bool reuse) {
      return ev([&] {
        for (const Tensor &t2 : sel) {
          if (N < 2) {
            (void)rdna4::matvec_launch(t2.dt, t2.d_w, act_for(t2.ncols), d_yb, t2.nrows, t2.ncols,
                                       nullptr);
          } else {
            (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                             reuse ? 0 : t2.bpr, N, nullptr);
          }
        }
      });
    };
    // Hybrid pass: the affected types (kMtIlp == 1) use the forced UNROLL, every
    // other type uses the shipping launch. Numbers stay bit-exact either way.
    auto pass_unr = [&](int N, int U) {
      return ev([&] {
        for (const Tensor &t2 : sel) {
          if (!rdna4::matvec_launch_batch_unroll(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                                 t2.bpr, N, U, nullptr)) {
            (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols, t2.bpr,
                                             N, nullptr);
          }
        }
      });
    };
    for (int N : batch_list) {
      // Two rounds: the first timed cell of the section is the one that pays the
      // clock ramp (the DPM gap this bench exists because of), so warm up first.
      (void)pass(N, false);
      double ms = 1e9, ms_reuse = 1e9;
      for (int r = 0; r < reps; ++r) ms = std::min(ms, pass(N, false));
      if (N >= 2) {
        (void)pass(N, true);
        for (int r = 0; r < reps; ++r) ms_reuse = std::min(ms_reuse, pass(N, true));
      }
      const int nn = N < 1 ? 1 : N;
      if (N < 2) gemv_ms = ms;
      const double gbs = used / (ms * 1e-3) / 1e9;
      std::printf("%5d %12.3f %12.3f %12.1f %14.1f %9.2fx %12.3f %10.1f", N, ms, ms / nn, gbs,
                  gbs * nn, gemv_ms / ms * nn, ms_reuse, 100.0 * ms_reuse / ms);
      for (int U : unroll_list) {
        if (N < 8) {
          std::printf(" %14s", "n/a");
          continue;
        }
        (void)pass_unr(N, U);
        double mu = 1e9;
        for (int r = 0; r < reps; ++r) mu = std::min(mu, pass_unr(N, U));
        std::printf(" %13.3f", mu);
        if (U == unroll_list.front()) unroll_ref = mu;
      }
      if (cap_probe) {
        // Capacidade livre: a MESMA passagem em DUAS streams. Se a maquina tem
        // folga (limitada por latencia), as duas se sobrepoem e o par custa bem
        // menos que 2x; se ja' esta' saturada (limitada por throughput), o par
        // custa 2x. Nao ha hazard: as duas escrevem no mesmo d_yb de proposito --
        // o que se mede aqui e' tempo, nao resultado.
        hipStream_t s1 = nullptr, s2 = nullptr;
        (void)hipStreamCreate(&s1);
        (void)hipStreamCreate(&s2);
        auto one = [&](hipStream_t st) {
          for (const Tensor &t2 : sel) {
            if (N < 2) {
              (void)rdna4::matvec_launch(t2.dt, t2.d_w, act_for(t2.ncols), d_yb, t2.nrows,
                                         t2.ncols, st);
            } else {
              (void)rdna4::matvec_launch_batch(t2.dt, t2.d_w, d_ab, d_yb, t2.nrows, t2.ncols,
                                               t2.bpr, N, st);
            }
          }
        };
        (void)ev([&] { one(s1); });
        double two = 1e9;
        for (int r = 0; r < reps; ++r) {
          two = std::min(two, ev([&] {
                           one(s1);
                           one(s2);
                         }));
        }
        std::printf(" %14.3f (%.2fx de 2 passagens)", two, two / ms);
        (void)hipStreamDestroy(s1);
        (void)hipStreamDestroy(s2);
      }
      std::printf("\n");
    }

    // Per-type breakdown of the marginal cost. This is the experiment that
    // separates the two stories the totals cannot separate, and it works because
    // bits-per-weight differ: per byte, a 1.5 bpw type carries 2.3x the ELEMENTS
    // of a 3.4 bpw one, and the activation volume per token is
    // sum(nrows*ncols) -- the element count, not the byte count. So:
    //   - if the marginal cost per token is proportional to ELEMENTS, it is the
    //     activation-load side (L1/L2/IC traffic or its instruction count);
    //   - if it is proportional to BYTES, it is the weight side.
    // Slope from N=2 to N=16 (14 extra tokens), so the per-launch floor and the
    // shared weight pass land in the intercept and cancel out.
    std::printf("\nper-type marginal cost (slope of N=2 -> N=16, 14 extra tokens):\n");
    std::printf("%-10s %7s %9s %10s %12s %14s %14s\n", "type", "tensors", "bytes_MB", "bpw",
                "marg_ms/tok", "ms/tok per TB", "ms/tok per Telem");
    for (int dt = 1; dt <= 14; ++dt) {
      std::vector<const Tensor *> sub;
      double bytes = 0, elems = 0;
      for (const Tensor &t : sel)
        if (t.dt == dt) {
          sub.push_back(&t);
          bytes += t.bytes;
          elems += (double)t.nrows * (double)t.ncols;
        }
      if (sub.size() < 2) continue;
      auto spass = [&](int N) {
        (void)ev([&] {
          for (const Tensor *t2 : sub)
            (void)rdna4::matvec_launch_batch(t2->dt, t2->d_w, d_ab, d_yb, t2->nrows, t2->ncols,
                                             t2->bpr, N, nullptr);
        });
        double best = 1e9;
        for (int r = 0; r < reps; ++r) {
          best = std::min(best, ev([&] {
                             for (const Tensor *t2 : sub)
                               (void)rdna4::matvec_launch_batch(t2->dt, t2->d_w, d_ab, d_yb,
                                                                t2->nrows, t2->ncols, t2->bpr, N,
                                                                nullptr);
                           }));
        }
        return best;
      };
      const double m2 = spass(2), m16 = spass(16);
      const double slope = (m16 - m2) / 14.0;  // ms per extra token
      const double bpw = bytes > 0 && elems > 0 ? bytes * 8.0 / elems : 0.0;
      std::printf("%-10s %7zu %9.1f %10.2f %12.4f %14.2f %14.3f\n", dt_name(dt), sub.size(),
                  bytes / 1e6, bpw, slope, slope / (bytes / 1e12) / 1e3,
                  slope / (elems / 1e12));
    }

    // Register/local/shared usage per (type, N): the cheap way to rule out "the
    // batched kernel spills to scratch" before blaming traffic.
    std::printf("\nbatch kernel resources (hipFuncGetAttributes): type  N  regs  local  smem  threads\n");
    for (int dt = 1; dt <= 14; ++dt) {
      bool any = false;
      for (const Tensor &t : sel)
        if (t.dt == dt) any = true;
      if (!any) continue;
      for (int N : batch_list) {
        if (N < 2) continue;
        hipFuncAttributes a{};
        bool ok = false;
        switch (N) {
          case 2: ok = rdna4::matvec_batch_kernel_attrs<2>(dt, a); break;
          case 3: ok = rdna4::matvec_batch_kernel_attrs<3>(dt, a); break;
          case 4: ok = rdna4::matvec_batch_kernel_attrs<4>(dt, a); break;
          case 8: ok = rdna4::matvec_batch_kernel_attrs<8>(dt, a); break;
          case 16: ok = rdna4::matvec_batch_kernel_attrs<16>(dt, a); break;
          case 32: ok = rdna4::matvec_batch_kernel_attrs<32>(dt, a); break;
          case 64: ok = rdna4::matvec_batch_kernel_attrs<64>(dt, a); break;
          default: break;
        }
        if (!ok) continue;
        std::printf("  %-10s %3d %5d %6zu %5zu %8d\n", dt_name(dt), N, a.numRegs, a.localSizeBytes,
                    a.sharedSizeBytes, a.maxThreadsPerBlock);
      }
    }
    std::printf("\n");
    return 0;
  }

  // ---------------------------------------------------------------------
  // Per-tensor measurement, one launch per tensor per pass (never repeated
  // back-to-back, so no tensor stays in the 64 MB L2). Variant passes are
  // ROTATED per round: a fixed order gives the first variant of a round a
  // systematic penalty (measuring the shipping kernel and the rows=8 launch of
  // the same type — the same instantiation! — 1.5x apart is what exposed it).
  // ---------------------------------------------------------------------
  std::vector<Variant> variants;
  variants.push_back({"ship", 0, 0});
  if (repeat_ship) variants.push_back({"ship2", 6, 0});
  if (land) variants.push_back({"land", 7, 0});
  if (lut_lds) variants.push_back({"lut-lds", 8, 0});
  for (int v : lut_lds_unroll) variants.push_back({"lds-unrl", 9, v});
  for (int v : row_variants) variants.push_back({"rows", 1, v});
  for (int v : minb_variants) variants.push_back({"minb", 2, v});
  for (int v : unroll_variants) variants.push_back({v == 1 ? "prefetch" : "unroll", v == 1 ? 4 : 3, v});
  for (int v : ilp_variants) variants.push_back({"ilp", 5, v});

  struct Meas {
    double ship_ms = -1;
    std::vector<double> var_ms;
  };
  std::vector<Meas> m(sel.size());
  for (auto &mm : m) mm.var_ms.assign(variants.size(), -1.0);
  const int n_cfg = (int)variants.size();
  for (int r = 0; r < reps; ++r) {
    for (int c = 0; c < n_cfg; ++c) {
      const int cfg = (c + r) % n_cfg;
      for (std::size_t i = 0; i < sel.size(); ++i) {
        const double ms = ev([&] {
          (void)launch_variant(variants[cfg].kind, variants[cfg].value, sel[i],
                               act_for(sel[i].ncols), d_y);
        });
        if (ms < 0) continue;
        if (cfg == 0) {
          if (m[i].ship_ms < 0 || ms < m[i].ship_ms) m[i].ship_ms = ms;
        } else if (m[i].var_ms[cfg] < 0 || ms < m[i].var_ms[cfg]) {
          m[i].var_ms[cfg] = ms;
        }
      }
    }
  }

  // ---------------------------------------------------------------------
  // Report per (type, shape) class, then the extrapolated per-token matvec time
  // over ALL tensors of the inventory (not just the staged ones).
  // ---------------------------------------------------------------------
  struct Class {
    int dt;
    std::int64_t nrows, bpr;
    double bytes_staged = 0, bytes_all = 0, ms_staged = 0;
    int n_staged = 0;
  };
  std::vector<Class> cls;
  auto find_class = [&](int dt, std::int64_t nr, std::int64_t bpr) -> Class * {
    for (auto &c : cls)
      if (c.dt == dt && c.nrows == nr && c.bpr == bpr) return &c;
    cls.push_back({dt, nr, bpr, 0, 0, 0, 0});
    return &cls.back();
  };
  for (std::size_t i = 0; i < sel.size(); ++i) {
    Class *c = find_class(sel[i].dt, sel[i].nrows, sel[i].bpr);
    c->bytes_staged += sel[i].bytes;
    c->ms_staged += m[i].ship_ms;
    c->n_staged++;
  }
  for (const Tensor &t : all) find_class(t.dt, t.nrows, t.bpr)->bytes_all += t.bytes;

  struct Row {
    Class *c;
    double rate, ms_tok;
  };
  std::vector<Row> rows;
  double total_ms = 0;
  for (auto &c : cls) {
    const double rate = c.ms_staged > 0 ? c.bytes_staged / (c.ms_staged * 1e-3) / 1e9 : 0.0;
    const double ms_tok = rate > 0 ? c.bytes_all / (rate * 1e9) * 1e3 : 0.0;
    rows.push_back({&c, rate, ms_tok});
    total_ms += ms_tok;
  }
  std::sort(rows.begin(), rows.end(), [](const Row &a, const Row &b) { return a.ms_tok > b.ms_tok; });

  // ms(tok)-fix: the same per-class extrapolation with ONE sync floor removed per
  // tensor (`null_once` = an empty kernel timed with the same record+sync
  // protocol). The per-tensor measurement pays that floor on every launch, which
  // for a 261 KB q8_0 tensor is most of the number -- so the classification of
  // "this class is slow" has to be read off this column, not off ms/tens.
  std::printf("\n%-9s %9s %6s %6s %6s %10s %10s %10s %10s %8s\n", "type", "rows", "bpr", "n",
              "GB/s", "ms(tok)", "ms/tens", "ms-fix", "ms(tok)-fix", "share");
  for (const Row &r : rows) {
    const double per_fix = std::max(0.0, r.c->ms_staged / std::max(1, r.c->n_staged) - null_once);
    const double rate_fix = per_fix > 0 ? r.c->bytes_staged / (per_fix * 1e-3) / 1e9 : 0.0;
    const double ms_fix = rate_fix > 0 ? r.c->bytes_all / (rate_fix * 1e9) * 1e3 : 0.0;
    std::printf("%-9s %9lld %6lld %6d %6.0f %10.3f %10.4f %10.4f %10.3f %7.1f%%\n", dt_name(r.c->dt),
                (long long)r.c->nrows, (long long)r.c->bpr, r.c->n_staged, r.rate, r.ms_tok,
                r.c->ms_staged / std::max(1, r.c->n_staged), per_fix, ms_fix,
                100.0 * r.ms_tok / total_ms);
  }
  std::printf("per-tensor sum: %.3f GB in %.2f ms -> %.1f tok/s (matvec only)\n",
              grand_total_bytes / 1e9, total_ms, 1000.0 / total_ms);
  std::printf("REPLAY (honest, queued): %.2f ms -> %.1f tok/s (matvec only); the per-tensor sum "
              "carries the %.3f ms sync floor x %zu launches\n",
              replay_ms, 1000.0 / replay_ms, null_once, sel.size());
  {
    int unstaged = 0;
    double ub = 0;
    for (const auto &c : cls)
      if (c.bytes_staged == 0) {
        unstaged++;
        ub += c.bytes_all;
      }
    if (unstaged > 0)
      std::printf("WARNING: %d (type,shape) classes have no staged tensor; %.3f GB (%.1f%% of the "
                  "per-token traffic) is missing from the tables\n",
                  unstaged, ub / 1e9, 100.0 * ub / grand_total_bytes);
    else
      std::printf("coverage: every (type,shape) class is represented in the staged set\n");
  }
  std::printf("(real per-token weight traffic excluding token_embd/MTP: %.3f GB; bench reports "
              "%.3f GB per token using loader.total_bytes())\n",
              grand_total_bytes / 1e9, (double)ld.total_bytes() / 1e9);

  // Per-type roll-up (per-tensor sum and replay share the same rates).
  std::printf("\n%-9s %10s %8s %10s %8s\n", "type", "bytes(tok)", "share", "ms(tok)", "GB/s");
  double byt[16] = {0}, mst[16] = {0};
  for (const Row &r : rows) {
    byt[r.c->dt] += r.c->bytes_all;
    mst[r.c->dt] += r.ms_tok;
  }
  for (int dt = 1; dt <= 14; ++dt) {
    if (byt[dt] <= 0) continue;
    std::printf("%-9s %10.3f %7.1f%% %10.3f %8.0f\n", dt_name(dt), byt[dt] / 1e9,
                100.0 * byt[dt] / grand_total_bytes, mst[dt], byt[dt] / (mst[dt] * 1e-3) / 1e9);
  }

  // ---------------------------------------------------------------------
  // Variant A/B (rows / minb). Both axes are bit-exact: ROWS moves which CTA
  // computes which row, minb only caps the register allocator, so neither
  // touches the arithmetic of an output element (see matvec.cuh).
  // ---------------------------------------------------------------------
  if (variants.size() > 1) {
    std::printf("\nVARIANT A/B (min over %d rotated passes)\n", reps);
    std::printf("%-12s %10s %8s   %s\n", "variant", "agg GB/s", "vs ship", "per-class ms(tok)");
    // shipping baseline
    double ship_bytes = 0, ship_ms = 0;
    for (const Row &r : rows) {
      ship_bytes += r.c->bytes_all;
      ship_ms += r.ms_tok;
    }
    std::printf("%-12s %10.0f %7.3fx   %s\n", "ship", ship_bytes / (ship_ms * 1e-3) / 1e9, 1.0,
                "baseline");
    std::vector<double> var_tot(variants.size(), 0);
    for (std::size_t v = 1; v < variants.size(); ++v) {
      double vb = 0, vm = 0;
      for (const Row &r : rows) {
        const Class *c = r.c;
        double use_ms = 0;
        for (std::size_t i = 0; i < sel.size(); ++i) {
          if (sel[i].dt != c->dt || sel[i].nrows != c->nrows || sel[i].bpr != c->bpr) continue;
          if (m[i].var_ms[v] > 0) use_ms += m[i].var_ms[v];
        }
        if (use_ms <= 0) continue;
        const double rate = c->bytes_staged / (use_ms * 1e-3) / 1e9;
        vb += c->bytes_all;
        vm += c->bytes_all / (rate * 1e9) * 1e3;
      }
      var_tot[v] = vm;
      std::printf("%-7s%-4d %10.0f %7.3fx   %.2f ms vs %.2f ms over %.1f%% of the bytes\n",
                  variants[v].name, variants[v].value, vb / (vm * 1e-3) / 1e9,
                  vm > 0 ? ship_ms / vm : 0.0, vm, ship_ms, 100.0 * vb / ship_bytes);
    }

    // ------------------------------------------------------------------
    // Per-type matrix: the tuning table of this campaign (config -> measured,
    // per type). One row per (type, config); `ms(tok)` is the per-token cost
    // extrapolated over ALL tensors of that type in the inventory, so the rows of
    // one type are directly comparable and the ratios are what the tuning table
    // in matvec.cuh is chosen from.
    //
    // `same kernel?` marks the rows that are the SHIPPING kernel measured a
    // second time under a different label: those ratios are the noise floor of
    // this bench, and nothing below that floor is a win.
    // ------------------------------------------------------------------
    std::printf("\nPER-TYPE MATRIX (ms/token per type, min of %d rotated passes)\n", reps);
    std::printf("%-9s %-10s %6s %10s %10s %8s  %s\n", "type", "config", "knob", "ms(tok)",
                "vs ship", "ratio", "same kernel?");
    for (int dt = 1; dt <= 14; ++dt) {
      double ship_t = 0;
      for (std::size_t i = 0; i < sel.size(); ++i)
        if (sel[i].dt == dt && m[i].ship_ms > 0) ship_t += m[i].ship_ms;
      if (ship_t <= 0) continue;
      std::printf("%-9s %-10s %6s %10.3f %10s %8s  yes (baseline)\n", dt_name(dt), "ship", "-",
                  ship_t, "-", "1.00x");
      for (std::size_t v = 1; v < variants.size(); ++v) {
        const bool same = variants[v].kind == 6 ||
                          (variants[v].kind == 1 &&
                           variants[v].value == rdna4::matvec_default_rows(dt)) ||
                          (variants[v].kind == 5 &&
                           variants[v].value == rdna4::matvec_default_ilp(dt));
        double b = 0;
        for (std::size_t i = 0; i < sel.size(); ++i)
          if (sel[i].dt == dt && m[i].var_ms[v] > 0) b += m[i].var_ms[v];
        if (b <= 0) continue;
        std::printf("%-9s %-10s %6d %10.3f %9.3fx %7.2fx  %s\n", dt_name(dt), variants[v].name,
                    variants[v].value, b, ship_t / b, ship_t / b, same ? "yes" : "no");
      }
    }

    // Aggregate noise floor: the shipping kernel measured twice, over the whole
    // inventory (the per-type rows above are the same thing per type).
    for (std::size_t v = 1; v < variants.size(); ++v) {
      if (variants[v].kind != 6) continue;
      double a = 0, b = 0;
      for (std::size_t i = 0; i < sel.size(); ++i)
        if (m[i].ship_ms > 0 && m[i].var_ms[v] > 0) {
          a += m[i].ship_ms;
          b += m[i].var_ms[v];
        }
      std::printf("\nnoise floor (same kernel, whole inventory): ship %.3f ms vs ship2 %.3f ms = "
                  "%.3fx\n",
                  a, b, a / b);
    }
  }

  // ---------------------------------------------------------------------
  // R9 stream-K A/B (--streamk S,...). Tail = classes of THIS bench's table
  // with short rows (nrows <= 1024), too few CTAs to fill 64 CUs
  // (ceil(nrows/shipping ROWS) < 64), or a shipping rate below the trunk
  // (~450 GB/s). Interleaved A/B vs the ship cell on the SAME tensors,
  // min-of-reps event-pair timing, DPM already warm; ship-vs-ship cell =
  // the noise floor of this very comparison. S=1 is the control: same work
  // as ship plus the stage-2 merge, i.e. the tax of the split alone.
  // PASS = tail-aggregate >= 1.02x for some S AND worst rel-L2 <= 1e-5.
  // ---------------------------------------------------------------------
  if (!sk_splits.empty()) {
    std::printf("\n=== STREAM-K prototype (R9, bench-only) ===\n");
    auto sk_rate_of = [&](const Class *c) -> double {
      for (const Row &r : rows)
        if (r.c == c) return r.rate;
      return 0.0;
    };
    auto sk_is_tail = [&](const Class *c) -> bool {
      const int sr = rdna4::matvec_default_config(c->dt).rows;
      const int ctas = (int)((c->nrows + sr - 1) / sr);
      return ctas < 64 || c->nrows <= 1024 || sk_rate_of(c) < 450.0;
    };
    std::vector<std::size_t> tail;
    for (std::size_t i = 0; i < sel.size(); ++i) {
      Class *c = find_class(sel[i].dt, sel[i].nrows, sel[i].bpr);
      if (sk_is_tail(c)) tail.push_back(i);
    }
    {
      double tb = 0, tm = 0;
      for (std::size_t i : tail) {
        tb += sel[i].bytes;
        tm += m[i].ship_ms;
      }
      std::printf("tail selector (ctas<64 || nrows<=1024 || ship<450 GB/s): %zu/%zu staged "
                  "tensors, %.3f GB, ship %.3f ms staged\n",
                  tail.size(), sel.size(), tb / 1e9, tm);
    }

    int sk_smax = 1;
    for (int s : sk_splits) sk_smax = std::max(sk_smax, s);
    std::int64_t sk_prow = 0;
    for (std::size_t i : tail) sk_prow = std::max(sk_prow, sel[i].nrows);
    float *d_y2 = nullptr, *d_part = nullptr;
    if (tail.empty() || sk_prow <= 0 ||
        hipMalloc(&d_y2, (std::size_t)sk_prow * sizeof(float)) != hipSuccess ||
        hipMalloc(&d_part, (std::size_t)sk_prow * sk_smax * sizeof(float)) != hipSuccess) {
      std::printf("streamk: no tail tensors or staging alloc failed\n");
      return 1;
    }
    std::vector<float> h_sh((std::size_t)sk_prow), h_sk((std::size_t)sk_prow);

    // Numeric gate first: rel-L2 / max|d| of prototype vs ship per S.
    // Order across CTAs changes by construction, so this is a tolerance gate
    // (attn_split policy precedent), never a bit-exact oracle.
    std::vector<double> sk_worst_rel(sk_splits.size(), 0.0), sk_worst_abs(sk_splits.size(), 0.0);
    std::vector<std::string> sk_worst_name(sk_splits.size());
    bool sk_launch_ok = true;
    for (std::size_t k = 0; k < sk_splits.size(); ++k) {
      for (std::size_t i : tail) {
        const Tensor &t = sel[i];
        if (!rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols,
                                  nullptr) ||
            hipMemcpy(h_sh.data(), d_y, (std::size_t)t.nrows * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
          sk_launch_ok = false;
          break;
        }
        if (!streamk_launch(t.dt, t.d_w, act_for(t.ncols), d_y2, d_part, t.nrows, t.ncols,
                            sk_splits[k], nullptr) ||
            hipMemcpy(h_sk.data(), d_y2, (std::size_t)t.nrows * sizeof(float),
                      hipMemcpyDeviceToHost) != hipSuccess) {
          sk_launch_ok = false;
          break;
        }
        double num = 0, den = 0, mx = 0;
        for (std::int64_t r = 0; r < t.nrows; ++r) {
          const double d = (double)h_sk[r] - (double)h_sh[r];
          num += d * d;
          den += (double)h_sh[r] * (double)h_sh[r];
          mx = std::max(mx, fabs(d));
        }
        const double rel = den > 0 ? std::sqrt(num / den) : std::sqrt(num);
        if (!(rel < 1.0)) {
          std::printf("streamk S=%d: NON-FINITE rel-L2 on %s\n", sk_splits[k], t.name.c_str());
          sk_launch_ok = false;
          break;
        }
        if (rel > sk_worst_rel[k]) {
          sk_worst_rel[k] = rel;
          sk_worst_abs[k] = mx;
          sk_worst_name[k] = t.name;
        }
      }
      if (!sk_launch_ok) break;
      std::printf("streamk S=%d numeric vs ship over %zu tail tensors: worst rel-L2 %.3g "
                  "(max|d| %.3g) on %s\n",
                  sk_splits[k], tail.size(), sk_worst_rel[k], sk_worst_abs[k],
                  sk_worst_name[k].c_str());
    }
    if (!sk_launch_ok) {
      std::printf("streamk: launch/numeric failure, no timing attempted\n");
      return 1;
    }

    // Timing: configs ship | sk(S...) | ship2(noise), rotated per round.
    std::vector<std::string> sk_names;
    std::vector<int> sk_cfg;  // -1 = ship, 0 = ship2, else S
    sk_names.push_back("ship");
    sk_cfg.push_back(-1);
    for (int s : sk_splits) {
      sk_names.push_back("sk" + std::to_string(s));
      sk_cfg.push_back(s);
    }
    sk_names.push_back("ship2");
    sk_cfg.push_back(0);
    const int sk_ncfg = (int)sk_cfg.size();
    std::vector<std::vector<double>> sk_best(sel.size(), std::vector<double>(sk_ncfg, -1.0));
    auto sk_fire = [&](int c, std::size_t i) -> bool {
      const Tensor &t = sel[i];
      if (sk_cfg[c] <= 0)
        return rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), d_y, t.nrows, t.ncols,
                                    nullptr);
      return streamk_launch(t.dt, t.d_w, act_for(t.ncols), d_y2, d_part, t.nrows, t.ncols,
                            sk_cfg[c], nullptr);
    };
    for (int c = 0; c < sk_ncfg; ++c)
      for (std::size_t i : tail) (void)sk_fire(c, i);  // untimed warm pass per config
    for (int r = 0; r < reps; ++r) {
      for (int c = 0; c < sk_ncfg; ++c) {
        const int cfg = (c + r) % sk_ncfg;
        for (std::size_t i : tail) {
          const double ms = ev([&] { (void)sk_fire(cfg, i); });
          if (ms < 0) continue;
          if (sk_best[i][cfg] < 0 || ms < sk_best[i][cfg]) sk_best[i][cfg] = ms;
        }
      }
    }

    std::printf("\n%-9s %9s %6s %5s %6s %10s %10s", "type", "rows", "bpr", "ctas", "GB/s",
                "ship_ms", "tail?");
    for (int s : sk_splits) std::printf(" %16s", ("sk" + std::to_string(s) + "_ms/ratio").c_str());
    std::printf(" %10s\n", "ship2_ms");
    double sk_ship_agg = 0, sk_s2_agg = 0;
    std::vector<double> sk_agg(sk_splits.size(), 0.0);
    for (const Row &r : rows) {
      const Class *c = r.c;
      if (!sk_is_tail(c)) continue;
      const int sr = rdna4::matvec_default_config(c->dt).rows;
      const int ctas = (int)((c->nrows + sr - 1) / sr);
      double b0 = 0;
      std::vector<double> bk(sk_splits.size(), 0.0);
      double b2 = 0;
      for (std::size_t i : tail) {
        if (sel[i].dt != c->dt || sel[i].nrows != c->nrows || sel[i].bpr != c->bpr) continue;
        if (sk_best[i][0] > 0) b0 += sk_best[i][0];
        for (std::size_t k = 0; k < sk_splits.size(); ++k)
          if (sk_best[i][1 + k] > 0) bk[k] += sk_best[i][1 + k];
        if (sk_best[i][sk_ncfg - 1] > 0) b2 += sk_best[i][sk_ncfg - 1];
      }
      if (b0 <= 0) continue;
      sk_ship_agg += b0;
      sk_s2_agg += b2;
      for (std::size_t k = 0; k < sk_splits.size(); ++k) sk_agg[k] += bk[k];
      std::printf("%-9s %9lld %6lld %5d %6.0f %10.4f %10s", dt_name(c->dt), (long long)c->nrows,
                  (long long)c->bpr, ctas, r.rate, b0, "TAIL");
      for (std::size_t k = 0; k < sk_splits.size(); ++k)
        std::printf(" %16.4f/%5.3fx", bk[k], bk[k] > 0 ? b0 / bk[k] : 0.0);
      std::printf(" %10.4f\n", b2);
    }
    std::printf("tail aggregate staged ms: ship %.3f vs ship2 %.3f (noise %.3fx)", sk_ship_agg,
                sk_s2_agg, sk_s2_agg > 0 ? sk_ship_agg / sk_s2_agg : 0.0);
    for (std::size_t k = 0; k < sk_splits.size(); ++k)
      std::printf(" vs sk%d %.3f (%.3fx)", sk_splits[k], sk_agg[k],
                  sk_agg[k] > 0 ? sk_ship_agg / sk_agg[k] : 0.0);
    std::printf("\n");
    for (std::size_t k = 0; k < sk_splits.size(); ++k) {
      const double ratio = sk_agg[k] > 0 ? sk_ship_agg / sk_agg[k] : 0.0;
      const bool pass = ratio >= 1.02 && sk_worst_rel[k] <= 1e-05;
      std::printf("S=%d verdict: %s (tail %.3fx vs 1.02 gate; worst rel-L2 %.3g vs 1e-05 gate)\n",
                  sk_splits[k], pass ? "PASS" : "FAIL", ratio, sk_worst_rel[k]);
    }
    std::printf("tolerance-gate proposal: accept iff worst tail rel-L2 <= 1e-05 (attn_split-style "
                "policy gate in scripts/check_* + numeric cell in this bench); bit-exact oracle "
                "is impossible by construction (cross-CTA summation order changes).\n");
    (void)hipFree(d_y2);
    (void)hipFree(d_part);
  }

  // ---------------------------------------------------------------------
  // R8 (RD_R8=1) and q3_K old-vs-new (RD_Q3K=1) A/B sections. Interleaved
  // ship/cand/ship2 per round (rotation), warmup >=300 ms GPU-busy with
  // sync in-loop, best-of-N + mean, exact memcmp bit-identity.
  // ---------------------------------------------------------------------
  // cand: 0 = R8 broadcast, 1 = q3_K old body, 2 = q6_K linearity body.
  auto ab_section = [&](const char *tag, int dt, int cand) {
    std::vector<std::size_t> idx;
    double bytes = 0;
    std::int64_t rows_tot = 0, elems_tot = 0;
    std::int64_t maxr = 0;
    for (std::size_t i = 0; i < sel.size(); ++i)
      if (sel[i].dt == dt) {
        idx.push_back(i);
        bytes += sel[i].bytes;
        rows_tot += sel[i].nrows;
        elems_tot += sel[i].nrows * sel[i].ncols;
        maxr = std::max(maxr, sel[i].nrows);
      }
    if (idx.empty()) {
      std::printf("%s: no staged tensors of dt=%d\n", tag, dt);
      return;
    }
    float *d_b = nullptr;
    std::vector<float> h_a((std::size_t)maxr), h_b((std::size_t)maxr);
    if (hipMalloc(&d_b, (std::size_t)maxr * sizeof(float)) != hipSuccess) {
      std::printf("%s: staging alloc failed\n", tag);
      return;
    }
    auto fire = [&](int cfg, std::size_t i) -> bool {
      const Tensor &t = sel[i];
      float *out = (cfg == 1) ? d_b : d_y;
      if (cfg == 1) {
        if (cand == 0)
          return bench_r8_launch(t.dt, t.d_w, act_for(t.ncols), out, t.nrows, t.ncols,
                                 nullptr);
        if (cand == 2)
          return bench_q6k_lin_launch(t.d_w, act_for(t.ncols), out, t.nrows, t.ncols,
                                      nullptr);
        return bench_q3k_old_launch(t.d_w, act_for(t.ncols), out, t.nrows, t.ncols,
                                    nullptr);
      }
      return rdna4::matvec_launch(t.dt, t.d_w, act_for(t.ncols), out, t.nrows, t.ncols,
                                  nullptr);
    };
    // Bit-identity first: ship (d_y) vs candidate (d_b), exact memcmp.
    std::int64_t diffs = 0, checked = 0;
    for (std::size_t i : idx) {
      const Tensor &t = sel[i];
      if (!fire(0, i) || !fire(1, i)) {
        std::printf("%s: launch failed on %s\n", tag, t.name.c_str());
        (void)hipFree(d_b);
        return;
      }
      if (hipMemcpy(h_a.data(), d_y, (std::size_t)t.nrows * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess ||
          hipMemcpy(h_b.data(), d_b, (std::size_t)t.nrows * sizeof(float),
                    hipMemcpyDeviceToHost) != hipSuccess) {
        std::printf("%s: copy failed\n", tag);
        (void)hipFree(d_b);
        return;
      }
      for (std::int64_t r = 0; r < t.nrows; ++r) {
        if (std::memcmp(&h_a[r], &h_b[r], sizeof(float)) != 0) ++diffs;
        ++checked;
      }
    }
    std::printf("%s bit-identity (dt=%d=%s): %lld diffs / %lld outputs "
                "(%.3f GB staged, %lld rows x cols = %lld K-elements)\n",
                tag, dt, dt_name(dt), (long long)diffs, (long long)checked, bytes / 1e9,
                (long long)rows_tot, (long long)elems_tot);
    // Warmup >= 300 ms.
    {
      double warm = 0;
      while (warm < 300.0) {
        for (int c = 0; c < 3; ++c)
          for (std::size_t i : idx) warm += ev([&] { (void)fire(c, i); });
      }
    }
    const int nrep = 6;
    std::vector<double> b_ship(idx.size(), -1.0), b_cand(idx.size(), -1.0),
        b_ship2(idx.size(), -1.0);
    std::vector<double> m_ship(idx.size(), 0.0), m_cand(idx.size(), 0.0),
        m_ship2(idx.size(), 0.0);
    for (int r = 0; r < nrep; ++r) {
      for (int c = 0; c < 3; ++c) {
        const int cfg = (c + r) % 3;
        for (std::size_t k = 0; k < idx.size(); ++k) {
          const double ms = ev([&] { (void)fire(cfg, idx[k]); });
          double &b = cfg == 0 ? b_ship[k] : (cfg == 1 ? b_cand[k] : b_ship2[k]);
          double &m = cfg == 0 ? m_ship[k] : (cfg == 1 ? m_cand[k] : m_ship2[k]);
          if (b < 0 || ms < b) b = ms;
          m += ms;
        }
      }
    }
    double s_ship = 0, s_cand = 0, s_ship2 = 0;
    std::printf("%-9s %9s %6s %10s %10s %10s %9s %9s\n", "tensor(rows)", "bpr", "GB/s-ship",
                "ship_ms", "cand_ms", "ship2_ms", "ratio", "noise");
    for (std::size_t k = 0; k < idx.size(); ++k) {
      const Tensor &t = sel[idx[k]];
      s_ship += b_ship[k];
      s_cand += b_cand[k];
      s_ship2 += b_ship2[k];
      std::printf("%-9s %9lld %6lld %10.1f %10.4f %10.4f %10.4f %9.3fx %9.3fx\n",
                  t.name.size() > 24 ? t.name.c_str() + t.name.size() - 24 : t.name.c_str(),
                  (long long)t.nrows, (long long)t.bpr,
                  t.bytes / (b_ship[k] * 1e-3) / 1e9, b_ship[k], b_cand[k], b_ship2[k],
                  b_ship[k] / b_cand[k], b_ship[k] / b_ship2[k]);
    }
    const double gbs_ship = bytes / (s_ship * 1e-3) / 1e9;
    const double gbs_cand = bytes / (s_cand * 1e-3) / 1e9;
    std::printf("%s aggregate staged (best-of-%d): ship %.3f ms = %.1f GB/s vs cand %.3f ms "
                "= %.1f GB/s -> %.4fx (noise floor ship/ship2 = %.4fx)\n",
                tag, nrep, s_ship, gbs_ship, s_cand, gbs_cand, s_ship / s_cand,
                s_ship / s_ship2);
    // Queued replay over the type set (one event pair): closer to trunk truth.
    auto replay = [&](int cfg) {
      return ev([&] {
        for (std::size_t i : idx) (void)fire(cfg, i);
      });
    };
    (void)replay(0);
    (void)replay(1);
    double r_ship = 1e9, r_cand = 1e9;
    for (int r = 0; r < nrep; ++r) {
      r_ship = std::min(r_ship, replay(0));
      r_cand = std::min(r_cand, replay(1));
    }
    std::printf("%s queued replay: ship %.3f ms = %.1f GB/s vs cand %.3f ms = %.1f GB/s -> "
                "%.4fx\n",
                tag, r_ship, bytes / (r_ship * 1e-3) / 1e9, r_cand,
                bytes / (r_cand * 1e-3) / 1e9, r_ship / r_cand);
    // VGPR/spill attributes (no launch-bounds change in these variants).
    {
      hipFuncAttributes as{}, ac{};
      bool ok = false;
      if (dt == 4)
        ok = hipFuncGetAttributes(
                 &as, reinterpret_cast<const void *>(
                          &rdna4::matvec_kernel_gen<rdna4::TQ4K, rdna4::MtShape<4>::rows,
                                                    rdna4::MtShape<4>::wpr,
                                                    rdna4::MtIlp<4>::value, false,
                                                    rdna4::MtMinb<4>::value,
                                                    rdna4::MtUnroll<4>::value>)) == hipSuccess &&
             hipFuncGetAttributes(
                 &ac, reinterpret_cast<const void *>(
                          &rdna4::matvec_kernel_gen<BenchQ4K_R8, rdna4::MtShape<4>::rows,
                                                    rdna4::MtShape<4>::wpr,
                                                    rdna4::MtIlp<4>::value, false,
                                                    rdna4::MtMinb<4>::value,
                                                    rdna4::MtUnroll<4>::value>)) == hipSuccess;
      if (ok)
        std::printf("%s attrs ship vs cand: regs %d vs %d, local %zu vs %zu, smem %zu vs "
                    "%zu\n",
                    tag, as.numRegs, ac.numRegs, as.localSizeBytes, ac.localSizeBytes,
                    as.sharedSizeBytes, ac.sharedSizeBytes);
    }
    (void)hipFree(d_b);
  };
  {
    const char *e = std::getenv("RD_R8");
    if (e && e[0] == '1') {
      std::printf("\n=== R8 scalar-broadcast A/B (bench-only) ===\n");
      ab_section("R8 q4_K", 4, 0);
      ab_section("R8 iq3_s", 12, 0);
    }
  }
  {
    const char *e = std::getenv("RD_Q3K");
    if (e && e[0] == '1') {
      std::printf("\n=== q3_K old(__vsubss4)-vs-new(linearity) A/B (bench-only) ===\n");
      ab_section("Q3K", 3, 1);
    }
  }
  {
    const char *e = std::getenv("RD_Q6K");
    if (e && e[0] == '1') {
      std::printf("\n=== q6_K ship(__vsubss4)-vs-lin(dp4a-linearity) A/B (bench-only) ===\n");
      ab_section("Q6K", 6, 2);
    }
  }
  // ---------------------------------------------------------------------
  // H1 step 3 audit (RD_QQ=1, bench-only): x32 hoist / activation reuse.
  // Production already serves every decode pair site from ONE quantization
  // (proj_qq: q/k/v in full_attn, qkv/gate/beta/alpha in gdn_layer, gate/up
  // in ffn — graph.cuh:906-908,1061-1064,1136-1137), so this section measures
  // the banked saving and proves nothing remains to ship:
  //   (i)   quantize determinism: same float row twice -> memcmp 0;
  //   (ii)  pair A/B on a real E-wide weight: quant+matvec (proj) vs
  //         matvec-only on the reused blocks (proj_qq), outputs memcmp 0,
  //         saving per pair = t_quant, interleaved with a reuse-vs-reuse
  //         noise cell, warmup >= 300 ms, best+mean of 6;
  //   (iii) queued quantize chain -> per-quantize us; per-token projection
  //         uses the production pair count 16*2 (q/k/v) + 48*3 (gdn) +
  //         64*1 (ffn/up) = 240 eliminated quantizes vs the 26.4 ms token.
  // E is read off the model (blk.0.ffn_gate.weight reduction dim), never
  // hardcoded.
  // ---------------------------------------------------------------------
  {
    const char *e = std::getenv("RD_QQ");
    if (e && e[0] == '1') {
      std::printf("\n=== QQ activation-reuse audit (bench-only) ===\n");
      std::int64_t E = 0;
      for (const Tensor &t : all)
        if (t.name == "blk.0.ffn_gate.weight") E = t.ncols;
      if (E <= 0 || E % QK8_1 != 0) {
        std::printf("QQ: cannot determine E from blk.0.ffn_gate.weight\n");
        return 1;
      }
      const std::int64_t nb = E / QK8_1;
      std::printf("QQ: E=%lld (nb=%lld q8 blocks); eliminated quantizes/token = "
                  "16*2+48*3+64*1 = 240\n",
                  (long long)E, (long long)nb);
      // Representative E-wide weight: the largest staged one.
      const Tensor *rep = nullptr;
      std::size_t n_ewide = 0;
      double ewide_bytes = 0;
      for (const Tensor &t : sel)
        if (t.ncols == E) {
          n_ewide++;
          ewide_bytes += t.bytes;
          if (!rep || t.bytes > rep->bytes) rep = &t;
        }
      if (!rep) {
        std::printf("QQ: no staged tensor with ncols==E\n");
        return 1;
      }
      std::printf("QQ: representative %s (%lld rows, %.3f MB); E-wide staged: %zu "
                  "tensors, %.3f GB\n",
                  rep->name.c_str(), (long long)rep->nrows, rep->bytes / 1e6, n_ewide,
                  ewide_bytes / 1e9);
      float *d_x = nullptr, *d_yq = nullptr, *d_yq2 = nullptr;
      block_q8_1 *d_q = nullptr, *d_q2 = nullptr;
      if (hipMalloc(&d_x, (std::size_t)E * sizeof(float)) != hipSuccess ||
          hipMalloc(&d_yq, (std::size_t)rep->nrows * sizeof(float)) != hipSuccess ||
          hipMalloc(&d_yq2, (std::size_t)rep->nrows * sizeof(float)) != hipSuccess ||
          hipMalloc(&d_q, (std::size_t)nb * sizeof(block_q8_1)) != hipSuccess ||
          hipMalloc(&d_q2, (std::size_t)nb * sizeof(block_q8_1)) != hipSuccess) {
        std::printf("QQ: staging alloc failed\n");
        return 1;
      }
      {
        std::vector<float> x;
        fill_x(x, E, 777u);
        if (hipMemcpy(d_x, x.data(), (std::size_t)E * sizeof(float),
                      hipMemcpyHostToDevice) != hipSuccess) {
          std::printf("QQ: H2D failed\n");
          return 1;
        }
      }
      auto qfire = [&](block_q8_1 *q) {
        rdna4::quantize_q8_1_kernel<<<(unsigned)((nb + 3) / 4), 128>>>(d_x, q, nb);
      };
      // (i) determinism: same row twice -> identical blocks.
      qfire(d_q);
      qfire(d_q2);
      hipDeviceSynchronize();
      {
        std::vector<unsigned> a((std::size_t)nb * sizeof(block_q8_1) / 4),
            b(a.size());
        hipMemcpy(a.data(), d_q, a.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(b.data(), d_q2, b.size() * 4, hipMemcpyDeviceToHost);
        std::int64_t diffs = 0;
        for (std::size_t k = 0; k < a.size(); ++k)
          if (a[k] != b[k]) ++diffs;
        std::printf("QQ determinism (quantize same row twice, %lld dwords): %lld diffs %s\n",
                    (long long)a.size(), (long long)diffs,
                    diffs == 0 ? "BIT-IDENTICO" : "DIFERE");
        if (diffs != 0) return 1;
      }
      // (ii) pair A/B: cfg0 = quant+matvec (proj), cfg1/2 = matvec-only (proj_qq
      // + noise cell). Outputs must be bit-identical (same q8 bytes feed both).
      auto pair_fire = [&](int cfg, float *out) -> bool {
        if (cfg == 0) {
          qfire(d_q);
          if (hipGetLastError() != hipSuccess) return false;
        }
        return rdna4::matvec_launch(rep->dt, rep->d_w, d_q, out, rep->nrows, rep->ncols,
                                    nullptr);
      };
      {
        std::vector<float> h0((std::size_t)rep->nrows), h1((std::size_t)rep->nrows);
        if (!pair_fire(0, d_yq) || !pair_fire(1, d_yq2)) {
          std::printf("QQ: pair launch failed\n");
          return 1;
        }
        hipDeviceSynchronize();
        hipMemcpy(h0.data(), d_yq, h0.size() * 4, hipMemcpyDeviceToHost);
        hipMemcpy(h1.data(), d_yq2, h1.size() * 4, hipMemcpyDeviceToHost);
        const bool same =
            std::memcmp(h0.data(), h1.data(), h0.size() * 4) == 0;
        std::printf("QQ pair outputs (proj vs proj_qq, %lld floats): %s\n",
                    (long long)h0.size(), same ? "BIT-IDENTICO" : "DIFERE");
        if (!same) return 1;
      }
      {
        double warm = 0;
        while (warm < 300.0) {
          for (int c = 0; c < 3; ++c)
            warm += ev([&] { (void)pair_fire(c == 2 ? 1 : c, d_yq); });
        }
      }
      const int nrep = 6;
      double b0 = -1, b1 = -1, b2 = -1, m0 = 0, m1 = 0, m2 = 0;
      for (int r = 0; r < nrep; ++r) {
        for (int c = 0; c < 3; ++c) {
          const int cfg = (c + r) % 3;
          const double ms = ev([&] { (void)pair_fire(cfg == 2 ? 1 : cfg, d_yq); });
          double &b = cfg == 0 ? b0 : (cfg == 1 ? b1 : b2);
          double &m = cfg == 0 ? m0 : (cfg == 1 ? m1 : m2);
          if (b < 0 || ms < b) b = ms;
          m += ms;
        }
      }
      const double saved_pair = b0 - b1;
      std::printf("QQ pair us (best-of-%d, per-event timing): proj %.2f vs reuse %.2f "
                  "(reuse2 %.2f) -> saves %.2f us/pair (means %.2f/%.2f/%.2f, noise "
                  "reuse/reuse2 = %.4fx)\n",
                  nrep, b0 * 1000.0, b1 * 1000.0, b2 * 1000.0, saved_pair * 1000.0,
                  m0 / nrep * 1000.0, m1 / nrep * 1000.0, m2 / nrep * 1000.0, b1 / b2);
      // (iii) queued quantize chain: Q quantizes under one event pair.
      const int Q = 240;  // one token-equivalent of eliminated quantizes
      auto qchain = [&] {
        for (int i = 0; i < Q; ++i) qfire(d_q);
      };
      (void)ev(qchain);
      double qbest = 1e9;
      for (int r = 0; r < nrep; ++r) qbest = std::min(qbest, ev(qchain));
      const double per_q_us = qbest * 1000.0 / Q;
      const double tok_ms = qbest;  // Q == one token-equivalent
      std::printf("QQ queued chain: %d quantizes(E=%lld) in %.3f ms -> %.3f us/quantize; "
                  "one token-equivalent (240x) = %.3f ms = %.2f%% of the 26.4 ms token\n",
                  Q, (long long)E, qbest, per_q_us, tok_ms, 100.0 * tok_ms / 26.4);
      std::printf("QQ verdict: banked saving ~= %.2f%% (already shipped via proj_qq); "
                  "REMAINING pairable quantizes in decode = 0 -> DEAD as new work "
                  "(<1%% bar unreachable at 0 remaining)\n",
                  100.0 * tok_ms / 26.4);
      (void)hipFree(d_x);
      (void)hipFree(d_yq);
      (void)hipFree(d_yq2);
      (void)hipFree(d_q);
      (void)hipFree(d_q2);
    }
  }
  return 0;
}
