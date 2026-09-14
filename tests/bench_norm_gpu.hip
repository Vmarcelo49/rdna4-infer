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
  return 0;
}
