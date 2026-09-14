// P0 do estudo de prefill (docs/estudo-prefill.md): quanto vale TILEJAR a saida e
// ESTAGIAR o peso na LDS, sem usar nenhuma instrucao exotica?
//
// Por que esta bancada existe: medido no dia 14/09, o prefill do llama.cpp Vulkan
// e' 9,66x o nosso (1196,49 tok/s contra 123,68 a 512 tokens, mesma placa e
// modelo) e a decomposicao por variavel de ambiente mostra dois fatores
// multiplicativos: 3,86x vem de tiling de saida + staging na LDS (eles fazem esse
// numero com os matrix cores DESLIGADOS) e 2,50x adicionais vem do coopmat.
//
// O nosso kernel em lote hoje tem 1 elemento de saida por thread por passo de k,
// zero LDS para peso ou ativacao, ~3,3 instrucoes por dp4a e 33 % de instrucoes
// de espera -- 6,4 TOPS efetivos. Esta bancada mede a MESMA conta (int8, dp4a)
// com a estrutura do MMQ: tile de saida BM x BN por CTA, e por thread RM x RN
// acumuladores independentes, com A e W estagiados em LDS e o laco interno
// consumindo os dois direto de la.
//
// Arithmetica que da' o alvo (para o leitor nao precisar refazer):
//   pico dp4a deste cartao = 4096 lanes x 2,97 GHz x 4 MAC = 48,7 T MACs/s
//   peca de peso por MAC: com M tokens, MACs/byte = M / 0,43 B = 2,3 M
//     -> em M=16, o teto de BANDA (633 GB/s) e' 23,4 T MACs/s
//     -> em M=64, 93,7 T MACs/s (acima do pico: vira compute-bound)
//   ou seja: mesmo sem mudar o chunk de 16 tokens, ha' 7x de espaco.
//
// Uso: bench-gemm-gpu [--m 16,64,128] [--n 17408] [--k 5120] [--reps 5]
//                    [--check]   (verifica contra referencia na CPU em 1 caso)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

namespace {

using i8 = signed char;

__device__ __forceinline__ int dp4a(int a, int b, int c) {
  // Mesma forma que include/rdna4/vecdotq.cuh:88 usa no caminho de producao.
  return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
}

__device__ __forceinline__ int load_i32(const i8 *p) { return *(const int *)p; }

// C[m][n] = sum_k A[m][k] * W[n][k], tudo int8, acumulado em int32.
// A: [M][K] row-major (tokens), W: [N][K] row-major (linhas de peso, como no GGUF).
//
// CTA: BM x BN de saida. Threads: (BM/RM) x (BN/RN). Laco de k em passos de BK,
// com A e W estagiados crus (int8) na LDS -- o MMQ de verdade estagia o peso
// DEQUANTIZADO; aqui o objetivo e' medir a estrutura, entao o byte que entra na
// LDS e' o mesmo que sai da global e o custo de dequantizacao fica de fora (a
// frente B/D mede esse custo separado).
template <int BM, int BN, int BK, int RM, int RN>
__global__ void __launch_bounds__((BM / RM) * (BN / RN)) gemm_i8_kernel(
    const i8 *__restrict__ A, const i8 *__restrict__ W, int *__restrict__ C, int N, int K) {
  constexpr int TM = BM / RM;  // threads na dimensao M
  constexpr int TN = BN / RN;  // threads na dimensao N
  __shared__ i8 s_a[BM][BK];
  __shared__ i8 s_w[BN][BK];

  const int tid = threadIdx.x;
  const int tr = tid % TM;       // faixa de linhas de A
  const int tc = tid / TM;       // faixa de linhas de W
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;

  int acc[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) acc[r][c] = 0;

  constexpr int A_ITEMS = BM * BK / 16;  // int4 = 16 int8 por iteracao de staging
  constexpr int W_ITEMS = BN * BK / 16;
  constexpr int NTHREADS = TM * TN;

  for (int k0 = 0; k0 < K; k0 += BK) {
    // staging cooperativo: A (BM x BK) e W (BN x BK), 16 bytes por vez
    for (int i = tid; i < A_ITEMS; i += NTHREADS) {
      const int row = i / (BK / 16);
      const int col = (i % (BK / 16)) * 16;
      const int4 v = *(const int4 *)&A[(int64_t)(m0 + row) * K + k0 + col];
      *(int4 *)&s_a[row][col] = v;
    }
    for (int i = tid; i < W_ITEMS; i += NTHREADS) {
      const int row = i / (BK / 16);
      const int col = (i % (BK / 16)) * 16;
      const int4 v = *(const int4 *)&W[(int64_t)(n0 + row) * K + k0 + col];
      *(int4 *)&s_w[row][col] = v;
    }
    __syncthreads();

    // laco interno: por quad de k, RM cargas de A + RN cargas de W alimentam
    // RM*RN dp4a -- e' esta razao (instrucoes por MAC) que a estrutura muda.
#pragma unroll
    for (int kk = 0; kk < BK; kk += 4) {
      int a[RM], w[RN];
#pragma unroll
      for (int r = 0; r < RM; ++r) a[r] = load_i32(&s_a[tr * RM + r][kk]);
#pragma unroll
      for (int c = 0; c < RN; ++c) w[c] = load_i32(&s_w[tc * RN + c][kk]);
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) acc[r][c] = dp4a(a[r], w[c], acc[r][c]);
    }
    __syncthreads();
  }

#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c)
      C[(int64_t)(m0 + tr * RM + r) * N + n0 + tc * RN + c] = acc[r][c];
}

// Referencia de estrutura: o mesmo trabalho do kernel de hoje -- 1 elemento de
// saida por thread, peso lido da GLOBAL (nada estagiado), k serial. Serve de piso
// para separar "tiling + LDS" de "instrucao diferente".
__global__ void gemm_i8_naive_kernel(const i8 *__restrict__ A, const i8 *__restrict__ W,
                                     int *__restrict__ C, int M, int N, int K) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  const int n = (int)(idx % N);
  const int m = (int)(idx / N);
  if (m >= M) return;
  const i8 *w = W + (int64_t)n * K;
  const i8 *a = A + (int64_t)m * K;
  int acc = 0;
  for (int k = 0; k + 3 < K; k += 4) acc = dp4a(load_i32(a + k), load_i32(w + k), acc);
  C[(int64_t)m * N + n] = acc;
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
    (void)hipEventRecord(e0, nullptr);
    fn();
    (void)hipEventRecord(e1, nullptr);
    (void)hipEventSynchronize(e1);
    float ms = 0;
    (void)hipEventElapsedTime(&ms, e0, e1);
    return ms;
  }
};

void fill_i8(std::vector<i8> &v, std::size_t n, unsigned seed) {
  v.resize(n);
  unsigned s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = (i8)((int)((s >> 16) & 0x7f) - 64);
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<int> ms_list{16, 64, 128, 256, 512};
  int N = 17408, K = 5120, reps = 5;
  bool check = false;
  int BK_sel = 32;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      ms_list.clear();
      const char *v = argv[++i];
      while (*v) {
        ms_list.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      N = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      K = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bk") == 0 && i + 1 < argc) {
      BK_sel = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--check") == 0) {
      check = true;
    }
  }

  const int MMAX = *std::max_element(ms_list.begin(), ms_list.end());
  std::vector<i8> hA, hW;
  fill_i8(hA, (std::size_t)MMAX * K, 7u);
  fill_i8(hW, (std::size_t)N * K, 11u);
  i8 *dA = nullptr, *dW = nullptr;
  int *dC = nullptr;
  if (hipMalloc(&dA, hA.size()) != hipSuccess || hipMalloc(&dW, hW.size()) != hipSuccess ||
      hipMalloc(&dC, (std::size_t)MMAX * N * sizeof(int)) != hipSuccess) {
    std::printf("alloc failed (W = %.1f MB)\n", hW.size() / 1e6);
    return 1;
  }
  (void)hipMemcpy(dA, hA.data(), hA.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(dW, hW.data(), hW.size(), hipMemcpyHostToDevice);
  std::printf("A %d x %d, W %d x %d int8 (W = %.1f MB), reps=%d\n", MMAX, K, N, K,
              hW.size() / 1e6, reps);

  // Verificacao contra a CPU num caso pequeno: um kernel "rapido" que calcula
  // errado nao vale nada.
  if (check) {
    std::vector<int> ref((std::size_t)4 * 256, 0);
    for (int m = 0; m < 4; ++m)
      for (int n = 0; n < 256; ++n) {
        int acc = 0;
        for (int k = 0; k < K; ++k) acc += (int)hA[(std::size_t)m * K + k] * (int)hW[(std::size_t)n * K + k];
        ref[(std::size_t)m * 256 + n] = acc;
      }
    std::vector<int> got((std::size_t)4 * 256, 0);
    // N=256 aqui (e nao o N grande): o kernel usa N como STRIDE de W e de C, entao
    // a verificacao tem de rodar com o mesmo N que a referencia.
    dim3 grid(256 / 64, 1);
    gemm_i8_kernel<16, 64, 64, 4, 4><<<grid, (16 / 4) * (64 / 4)>>>(dA, dW, dC, 256, K);
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(got.data(), dC, got.size() * sizeof(int), hipMemcpyDeviceToHost);
    int bad = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
      if (ref[i] != got[i]) ++bad;
    std::printf("check 4x256 (BM=16,BN=64,RM=4,RN=4): %d de %zu errados\n", bad, ref.size());
    if (bad) return 1;
  }

  // Configuracao por M: o tile tem de caber no M disponivel (BM <= M) e o numero
  // de threads e' DERIVADO do template -- lancar com outro numero de threads faz
  // o kernel nao rodar (e o tempo medido vira ficcao; foi o que aconteceu na
  // primeira versao desta bancada, 1198 T MACs/s contra um pico de 48,7).
  struct Cfg {
    int BM, BN, BK, RM, RN;
    int threads() const { return (BM / RM) * (BN / RN); }
  };
  auto cfg_for = [&](int M) {
    if (M >= 128) return Cfg{128, 128, BK_sel, 8, 8};
    if (M >= 64) return Cfg{64, 128, BK_sel, 4, 8};
    if (M >= 32) return Cfg{32, 128, BK_sel, 2, 8};
    return Cfg{16, 128, BK_sel, 2, 8};
  };
  auto launch_tiled = [&](const Cfg &c, const i8 *A, const i8 *W, int *C, int M, int N, int K) {
    dim3 grid((unsigned)(N / c.BN), (unsigned)((M + c.BM - 1) / c.BM));
    if (c.BM == 128 && c.BK == 64)
      gemm_i8_kernel<128, 128, 64, 8, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    else if (c.BM == 128) gemm_i8_kernel<128, 128, 32, 8, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    else if (c.BM == 64 && c.BK == 64)
      gemm_i8_kernel<64, 128, 64, 4, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    else if (c.BM == 64) gemm_i8_kernel<64, 128, 32, 4, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    else if (c.BM == 32) gemm_i8_kernel<32, 128, 32, 2, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    else gemm_i8_kernel<16, 128, 32, 2, 8><<<grid, c.threads()>>>(A, W, C, N, K);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) {
      std::printf("  LAUNCH FALHOU (BM=%d): %s\n", c.BM, hipGetErrorString(e));
      return false;
    }
    return true;
  };

  EvPair ev;
  std::printf("\n%5s %15s %12s %12s %14s %14s\n", "M", "kernel", "ms", "T MACs/s", "TOPS",
              "GB/s (peso)");
  for (int M : ms_list) {
    const Cfg c = cfg_for(M);
    const double macs = (double)M * (double)N * (double)K;
    const double bytes = (double)N * (double)K;

    // Verificacao POR CONFIGURACAO antes de cronometrar: o naive (1 elemento por
    // thread, sem LDS) e' lento mas correto, e serve de oraculo para o tilejado.
    {
      std::vector<int> ref((std::size_t)M * N, 0), got((std::size_t)M * N, 0);
      const int threads = 256;
      gemm_i8_naive_kernel<<<(int)(((int64_t)M * N + threads - 1) / threads), threads>>>(dA, dW, dC,
                                                                                        M, N, K);
      hipError_t e = hipGetLastError();
      if (e != hipSuccess) {
        std::printf("  naive LAUNCH FALHOU: %s\n", hipGetErrorString(e));
        break;
      }
      (void)hipDeviceSynchronize();
      (void)hipMemcpy(ref.data(), dC, ref.size() * sizeof(int), hipMemcpyDeviceToHost);
      if (!launch_tiled(c, dA, dW, dC, M, N, K)) break;
      (void)hipDeviceSynchronize();
      (void)hipMemcpy(got.data(), dC, got.size() * sizeof(int), hipMemcpyDeviceToHost);
      int bad = 0;
      const int step = std::max(1, (int)(ref.size() / 200000));
      for (std::size_t i = 0; i < ref.size(); i += step)
        if (ref[i] != got[i]) ++bad;
      std::printf("  check M=%d (BM=%d BN=%d RM=%d RN=%d, %d threads): %d de %zu amostras erradas\n",
                  M, c.BM, c.BN, c.RM, c.RN, c.threads(), bad, ref.size() / step);
      if (bad) break;
    }

    auto run_tiled = [&] { (void)launch_tiled(c, dA, dW, dC, M, N, K); };
    auto run_naive = [&] {
      const int threads = 256;
      gemm_i8_naive_kernel<<<(int)(((int64_t)M * N + threads - 1) / threads), threads>>>(dA, dW, dC,
                                                                                        M, N, K);
    };
    for (int which = 0; which < 2; ++which) {
      const char *name = which ? "naive(1/thread)" : "tiled+LDS";
      std::function<void()> fn = which ? std::function<void()>(run_naive)
                                       : std::function<void()>(run_tiled);
      fn();
      (void)hipDeviceSynchronize();
      double best = 1e9;
      for (int r = 0; r < reps; ++r) best = std::min(best, ev(fn));
      const double tmacs = macs / (best * 1e-3) / 1e12;
      std::printf("%5d %15s %12.3f %12.2f %14.2f %14.1f\n", M, name, best, tmacs, 2 * tmacs,
                  bytes / (best * 1e-3) / 1e9);
    }
  }
  std::printf("\n(pico dp4a deste cartao: 48,7 T MACs/s = 97,3 TOPS; 4096 lanes x 2,97 GHz x 4 MAC)\n");
  std::printf("(nosso prefill hoje: 3,19 T MACs/s = 6,4 TOPS)\n");
  (void)hipFree(dA);
  (void)hipFree(dW);
  (void)hipFree(dC);
  return 0;
}
