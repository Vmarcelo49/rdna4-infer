// Bench isolado das normas do decode (frente kernels de decode, noite 2026-09-14).
//
// Por que existe: `docs/medicoes-banda-e-gargalos.md` §4 mede `rms_norm(1x5120)` em
// **10,90 us por lançamento** contra um piso de despacho de 2,20 us (cadeia de 100),
// e `attn_norm` 13,8 us dentro do grafo; são 128 lançamentos por token
// (`attn_norm` ×64 + `post_norm` ×64) mais o `out_norm`. O kernel tem 1 CTA de 256
// threads para 5120 elementos (20 cargas por thread numa cadeia de fma), ou seja:
// latência de load, não banda (20 KB!) nem ALU.
//
// Protocolo: cadeia de `chain` lançamentos por par de eventos, aquecimento >=320 ms,
// min de `rounds` rodadas × 3 passadas rotacionadas, e o kernel ANTIGO medido duas
// vezes como piso de ruído do próprio harness. Bit-exatidão conferida com memcmp
// contra o kernel antigo (as duas versões têm a mesma ordem de fma e a mesma árvore
// de redução, então qualquer diferença é bug, não arredondamento).
//
// usage: bench-norm-gpu [--rounds N] [--chain N] [--no-noise]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "rdna4/nn.cuh"
#include "rdna4/matvec.cuh"  // RD_FUSE section only: production quantize kernels
                              // as the two-launch baseline (single-TU binary, no ODR issue)

namespace {

constexpr int kNThreads = rdna4::kRmsNormThreads;

// ---------------------------------------------------------------------------
// Cópia LITERAL do kernel antigo (antes da mudança do journal §4), para o A/B e
// para a checagem de bit-exatidão. Não é usada por nenhum caminho de produção.
// ---------------------------------------------------------------------------
__global__ void rms_norm_old_kernel(const float *__restrict__ x, const float *__restrict__ w,
                                    float *__restrict__ y, std::int64_t ncols, float eps) {
  __shared__ float partial[kNThreads];
  const std::int64_t row = blockIdx.x;
  const float *xr = x + row * ncols;
  float *yr = y + row * ncols;
  float acc = 0.0f;
  for (std::int64_t i = threadIdx.x; i < ncols; i += kNThreads) {
    const float v = xr[i];
    acc = fmaf(v, v, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kNThreads / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)ncols + eps);
  for (std::int64_t i = threadIdx.x; i < ncols; i += kNThreads) {
    yr[i] = w ? xr[i] * scale * w[i] : xr[i] * scale;
  }
}

__global__ void l2_norm_old_kernel(const float *__restrict__ x, float *__restrict__ y,
                                   std::int64_t ncols, float eps) {
  __shared__ float partial[kNThreads];
  const std::int64_t row = blockIdx.x;
  const float *xr = x + row * ncols;
  float *yr = y + row * ncols;
  float acc = 0.0f;
  for (std::int64_t i = threadIdx.x; i < ncols; i += kNThreads) {
    const float v = xr[i];
    acc = fmaf(v, v, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kNThreads / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] + eps);
  for (std::int64_t i = threadIdx.x; i < ncols; i += kNThreads) yr[i] = xr[i] * scale;
}

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

struct Case {
  const char *name;
  int nrows;
  int ncols;
  bool with_w;
  bool l2;
  int per_token;  // lançamentos por token no motor (para extrapolar ms/token)
};

// ---------------------------------------------------------------------------
// RD_FUSE sketch (BENCH-ONLY, gated on env RD_FUSE=1): rms_norm(1x5120, w)
// immediately followed by quantize_q8_1 (160 blocks) as today (two launches)
// vs one fused launch. Bit-exactness by construction: phase 1 is the same
// fma chain in the same order with the same reduction tree; phase 2 keeps
// each normalized value in the register of the lane that the production
// quantize warp would load it into (warp w owns blocks b with b%8==w, lane
// j holds element 32b+j -- the same lane assignment as quantize_q8_1_block),
// then runs the same amax/round/qsum/ds arithmetic. memcmp below proves it.
// ---------------------------------------------------------------------------
constexpr int kFuseN = 5120;      // the attn_norm/post_norm row (128x per token)
constexpr int kFuseNB = kFuseN / 32;  // 160 q8 blocks; 160 % 8 == 0 required

__global__ void rmsnorm_quant_fused_kernel(const float *__restrict__ x,
                                           const float *__restrict__ w,
                                           rdna4::block_q8_1 *__restrict__ y,
                                           float eps) {
  __shared__ float partial[kNThreads];
  const std::int64_t row = blockIdx.x;
  const float *xr = x + row * kFuseN;
  rdna4::block_q8_1 *yr = y + row * kFuseNB;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  // Phase 1: same accumulation order as rms_norm_kernel (4 in flight).
  float acc = 0.0f;
  std::int64_t i = threadIdx.x;
  for (; i + 3 * kNThreads < kFuseN; i += 4 * kNThreads) {
    const float v0 = xr[i];
    const float v1 = xr[i + kNThreads];
    const float v2 = xr[i + 2 * kNThreads];
    const float v3 = xr[i + 3 * kNThreads];
    acc = fmaf(v0, v0, acc);
    acc = fmaf(v1, v1, acc);
    acc = fmaf(v2, v2, acc);
    acc = fmaf(v3, v3, acc);
  }
  for (; i < kFuseN; i += kNThreads) {
    const float v = xr[i];
    acc = fmaf(v, v, acc);
  }
  partial[threadIdx.x] = acc;
  __syncthreads();
  for (int stride = kNThreads / 2; stride > 0; stride >>= 1) {
    if ((int)threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  const float scale = rsqrtf(partial[0] / (float)kFuseN + eps);

  // Phase 2: normalized values stay in registers (20 per thread); warp w
  // quantizes blocks w, w+8, ... with the production arithmetic verbatim.
  float yn[kFuseNB / 8];
#pragma unroll
  for (int k = 0; k < kFuseNB / 8; ++k) {
    const std::int64_t idx = (std::int64_t)threadIdx.x + (std::int64_t)k * kNThreads;
    const float xv = xr[idx];
    yn[k] = w ? xv * scale * w[idx] : xv * scale;  // same expression as rms store
  }
#pragma unroll
  for (int k = 0; k < kFuseNB / 8; ++k) {
    const std::int64_t b = warp + (std::int64_t)k * 8;
    const float xi = yn[k];  // lane j == element 32b+j, as in quantize_q8_1_block
    float amax = fabsf(xi);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor(amax, off));
    const float d = amax / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    const int q = (int)roundf(xi * id);
    yr[b].qs[lane] = (int8_t)q;
    int qsum = q;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) qsum += __shfl_xor(qsum, off);
    if (lane == 0) {
      yr[b].ds = (uint32_t)rdna4::float_to_fp16(d) |
                 ((uint32_t)rdna4::float_to_fp16(d * (float)qsum) << 16);
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  int rounds = 12, chain = 100;
  bool noise = true;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--chain") && i + 1 < argc) chain = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--no-noise")) noise = false;
    else { std::fprintf(stderr, "unknown %s\n", argv[i]); return 1; }
  }

  // Formas reais do motor (graph.cuh): attn_norm/post_norm 1x5120 (x64 cada),
  // rms_norm do GDN 48x128 (x48), l2_norm 16x128 (x2 por camada GDN = 96).
  std::vector<Case> cases = {
      {"rms_norm(1 x 5120) w", 1, 5120, true, false, 128},
      {"rms_norm(48 x 128)", 48, 128, false, false, 48},
      {"l2_norm(16 x 128)", 16, 128, false, true, 96},
      {"rms_norm(8 x 5120)", 8, 5120, true, false, 0},
  };

  const int maxn = 48 * 5120;
  float *dx = nullptr, *dw = nullptr, *dy_old = nullptr, *dy_new = nullptr;
  if (hipMalloc(&dx, (std::size_t)maxn * 4) != hipSuccess ||
      hipMalloc(&dw, (std::size_t)5120 * 4) != hipSuccess ||
      hipMalloc(&dy_old, (std::size_t)maxn * 4) != hipSuccess ||
      hipMalloc(&dy_new, (std::size_t)maxn * 4) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  const unsigned bl = 256;
  fill_kernel<<<(unsigned)(((std::size_t)maxn + bl - 1) / bl), bl>>>(dx, maxn, 5u);
  fill_kernel<<<(unsigned)((5120 + bl - 1) / bl), bl>>>(dw, 5120, 6u);
  hipDeviceSynchronize();

  // ---- bit-exatidão -------------------------------------------------------
  std::printf("bit-exatidao (memcmp da saida inteira, antigo vs novo):\n");
  int bad = 0;
  for (const Case &c : cases) {
    const std::size_t bytes = (std::size_t)c.nrows * c.ncols * 4;
    if (c.l2) {
      l2_norm_old_kernel<<<c.nrows, kNThreads>>>(dx, dy_old, c.ncols, 1e-6f);
      rdna4::l2_norm_launch(dx, dy_new, c.nrows, c.ncols, 1e-6f);
    } else {
      rms_norm_old_kernel<<<c.nrows, kNThreads>>>(dx, c.with_w ? dw : nullptr, dy_old, c.ncols, 1e-6f);
      rdna4::rms_norm_launch(dx, c.with_w ? dw : nullptr, dy_new, c.nrows, c.ncols, 1e-6f);
    }
    hipDeviceSynchronize();
    std::vector<float> a(bytes / 4), b(bytes / 4);
    hipMemcpy(a.data(), dy_old, bytes, hipMemcpyDeviceToHost);
    hipMemcpy(b.data(), dy_new, bytes, hipMemcpyDeviceToHost);
    const bool same = std::memcmp(a.data(), b.data(), bytes) == 0;
    std::printf("  %-22s %s\n", c.name, same ? "BIT-IDENTICO" : "DIFERE");
    if (!same) ++bad;
  }
  if (bad) std::printf("  ATENCAO: %d caso(s) diferem — nao embarcar sem gate numerico.\n", bad);

  // ---- tempo --------------------------------------------------------------
  hipEvent_t e0, e1;
  hipEventCreate(&e0);
  hipEventCreate(&e1);
  auto measure = [&](const std::function<void()> &fn) {
    const double w0 = now_ms();
    while (now_ms() - w0 < 320.0) { for (int i = 0; i < chain; ++i) fn(); hipDeviceSynchronize(); }
    double best = 1e30;
    for (int r = 0; r < rounds; ++r) {
      hipEventRecord(e0, nullptr);
      for (int i = 0; i < chain; ++i) fn();
      hipEventRecord(e1, nullptr);
      hipDeviceSynchronize();
      float ms = 0;
      hipEventElapsedTime(&ms, e0, e1);
      best = std::min(best, (double)ms * 1000.0 / chain);
    }
    return best;
  };

  std::printf("\nus por lançamento (cadeia de %d, min de %d rodadas x 3 passadas rotacionadas):\n",
              chain, rounds);
  std::printf("%-22s %10s %10s %10s %8s  %s\n", "caso", "antigo", "novo", "novo 2a vez",
              "razao", "ms/token (antigo -> novo)");
  double tot_old = 0, tot_new = 0;
  for (const Case &c : cases) {
    auto old_fn = [&] {
      if (c.l2) l2_norm_old_kernel<<<c.nrows, kNThreads>>>(dx, dy_old, c.ncols, 1e-6f);
      else rms_norm_old_kernel<<<c.nrows, kNThreads>>>(dx, c.with_w ? dw : nullptr, dy_old, c.ncols, 1e-6f);
    };
    auto new_fn = [&] {
      if (c.l2) (void)rdna4::l2_norm_launch(dx, dy_new, c.nrows, c.ncols, 1e-6f);
      else (void)rdna4::rms_norm_launch(dx, c.with_w ? dw : nullptr, dy_new, c.nrows, c.ncols, 1e-6f);
    };
    double t_old = 1e30, t_new = 1e30, t_new2 = 1e30;
    for (int r = 0; r < 3; ++r) {
      t_old = std::min(t_old, measure(old_fn));
      t_new = std::min(t_new, measure(new_fn));
      if (noise) t_new2 = std::min(t_new2, measure(new_fn));
    }
    const double ms_old = t_old * c.per_token / 1000.0, ms_new = t_new * c.per_token / 1000.0;
    tot_old += ms_old;
    tot_new += ms_new;
    std::printf("%-22s %9.2f %10.2f %10.2f %7.3fx  %.3f -> %.3f\n", c.name, t_old, t_new, t_new2,
                t_old / t_new, ms_old, ms_new);
  }
  std::printf("\n  total nas formas do motor: %.3f -> %.3f ms/token (%.3f ms = %.1f%% do token de 36,0 ms)\n",
              tot_old, tot_new, tot_old - tot_new, 100.0 * (tot_old - tot_new) / 36.0);

  // ---- RD_FUSE: rms_norm(1x5120,w)+quantize pairs --------------------------------
  // Two launches back-to-back in one event pair (as the graph issues them) vs
  // one fused launch. Env-gated (RD_FUSE=1); default binary behavior unchanged.
  {
    const char *e = std::getenv("RD_FUSE");
    if (e && e[0] == '1') {
      const int64_t nb = kFuseNB;  // 160
      float *d_fx = nullptr, *d_fw = nullptr, *d_fy = nullptr;
      rdna4::block_q8_1 *d_q2 = nullptr, *d_qf = nullptr;
      bool ok = hipMalloc(&d_fx, (std::size_t)kFuseN * 4) == hipSuccess &&
                hipMalloc(&d_fw, (std::size_t)kFuseN * 4) == hipSuccess &&
                hipMalloc(&d_fy, (std::size_t)kFuseN * 4) == hipSuccess &&
                hipMalloc(&d_q2, (std::size_t)nb * sizeof(rdna4::block_q8_1)) == hipSuccess &&
                hipMalloc(&d_qf, (std::size_t)nb * sizeof(rdna4::block_q8_1)) == hipSuccess;
      if (!ok) {
        std::printf("\nRD_FUSE: staging alloc failed\n");
        return 1;
      }
      fill_kernel<<<(unsigned)((kFuseN + bl - 1) / bl), bl>>>(d_fx, kFuseN, 11u);
      fill_kernel<<<(unsigned)((kFuseN + bl - 1) / bl), bl>>>(d_fw, kFuseN, 12u);
      hipDeviceSynchronize();
      // Bit-identity: two launches vs fused, exact memcmp over 160 blocks.
      (void)rdna4::rms_norm_launch(d_fx, d_fw, d_fy, 1, kFuseN, 1e-6f);
      rdna4::quantize_q8_1_kernel<<<(unsigned)((nb + 3) / 4), 128>>>(d_fy, d_q2, nb);
      rmsnorm_quant_fused_kernel<<<1, kNThreads>>>(d_fx, d_fw, d_qf, 1e-6f);
      hipDeviceSynchronize();
      std::vector<unsigned> a((std::size_t)nb * sizeof(rdna4::block_q8_1) / 4),
          b(a.size());
      hipMemcpy(a.data(), d_q2, a.size() * 4, hipMemcpyDeviceToHost);
      hipMemcpy(b.data(), d_qf, b.size() * 4, hipMemcpyDeviceToHost);
      std::int64_t diffs = 0;
      for (std::size_t k = 0; k < a.size(); ++k)
        if (a[k] != b[k]) ++diffs;
      std::printf("\nRD_FUSE bit-identity (2 launches vs fused, %lld dwords): %lld diffs %s\n",
                  (long long)a.size(), (long long)diffs, diffs == 0 ? "BIT-IDENTICO" : "DIFERE");
      auto pair_fn = [&] {
        (void)rdna4::rms_norm_launch(d_fx, d_fw, d_fy, 1, kFuseN, 1e-6f);
        rdna4::quantize_q8_1_kernel<<<(unsigned)((nb + 3) / 4), 128>>>(d_fy, d_q2, nb);
      };
      auto fused_fn = [&] { rmsnorm_quant_fused_kernel<<<1, kNThreads>>>(d_fx, d_fw, d_qf, 1e-6f); };
      double t_pair = 1e30, t_fused = 1e30, t_fused2 = 1e30;
      for (int r = 0; r < 3; ++r) {
        t_pair = std::min(t_pair, measure(pair_fn));
        t_fused = std::min(t_fused, measure(fused_fn));
        t_fused2 = std::min(t_fused2, measure(fused_fn));
      }
      const double saved = t_pair - t_fused;
      std::printf("RD_FUSE us/pair (chain %d): two-launch %.2f vs fused %.2f (2nd %.2f) -> "
                  "saves %.2f us/pair (ratio %.3fx, noise %.3fx)\n",
                  chain, t_pair, t_fused, t_fused2, saved, t_pair / t_fused,
                  t_fused / t_fused2);
      std::printf("RD_FUSE extrapolate x128 pairs: %.3f ms/token = %.2f%% of 26.62 ms token "
                  "(x256 quantize-adjacent: %.3f ms = %.2f%%)\n",
                  saved * 128.0 / 1000.0, 100.0 * saved * 128.0 / 1000.0 / 26.62,
                  saved * 256.0 / 1000.0, 100.0 * saved * 256.0 / 1000.0 / 26.62);
      std::printf("RD_FUSE verdict: %s\n",
                  saved * 128.0 / 1000.0 / 26.62 >= 0.01 ? "SHIP-IT candidate (needs prod patch)"
                                                         : "DEAD (<1%% of token)");
      (void)hipFree(d_fx);
      (void)hipFree(d_fw);
      (void)hipFree(d_fy);
      (void)hipFree(d_q2);
      (void)hipFree(d_qf);
    }
  }
  return 0;
}
