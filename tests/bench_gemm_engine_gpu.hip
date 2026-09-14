// ===========================================================================
// Bancada do D2 (docs/plano-prefill.md §6): o GEMM tilejado int8 de
// `include/rdna4/gemm.cuh` sobre os TENSORES REAIS do modelo, contra o GEMV em
// lote que embarca hoje (`rdna4::matvec_launch_batch`) nas MESMAS entradas.
//
// O que esta bancada responde, nesta ordem (regra da casa: verificar ANTES de
// cronometrar -- docs/estudo-prefill.md §3b, onde duas bancadas reportaram
// 1198 T-MAC/s de um kernel que nao rodava):
//
//   1. CORRETUDE DO STAGING: os bytes int8 que o kernel escreve na LDS contra a
//      algebra do `vec_dot_iq3_s_q8_1` escrita em CPU sobre os bytes crus do
//      GGUF e a LUT lida do DEVICE (nao uma copia em C).
//   2. CORRETUDE DO GEMM: (a) oraculo de CPU (produto em int32 exato, `1+2*sc`
//      inteiro, `d_w*d_a` em fp32, acumulador fp32 com FMA) num caso pequeno;
//      (b) o `vec_dot_*` DO MOTOR, um thread por (token, linha), somado na
//      mesma ordem de k crescente do GEMM -- e' o gate de BIT-EXATIDAO;
//      (c) o `matvec_launch_batch` que embarca, nas mesmas entradas: quantos
//      elementos diferem, max|d| e rel-L2 (a ordem das somas NAO e' a mesma: o
//      lote do motor anda em k com passo e reduz com butterfly, entao um numero
//      diferente de zero aqui e' esperado -- e e' reportado, nao escondido).
//   3. VELOCIDADE: T-MAC/s em M=16/64/128/512 para os dois caminhos, a razao, o
//      piso de ruido (a MESMA medicao duas vezes) e o aquecimento de DPM antes
//      de cada cronometragem (a frente F mediu a primeira celula de um processo
//      47 % mais lenta que as seguintes).
//   4. `hipGetLastError()` depois de TODO lancamento, contagem de zeros e de
//      nao-finitos na saida (kernel que nao roda aparece como "toda zero").
//
// Uso:
//   bench-gemm-engine-gpu <model.gguf> [--m 16,64,128,512] [--reps 5]
//                        [--tensor NOME] [--list] [--cfg 1,5,6,7]
//                        [--no-verify] [--quick] [--types]
// ===========================================================================
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "rdna4/gemm.cuh"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"
#include "rdna4/quant_tables.h"
#include "rdna4/quants.h"
#include "rdna4/vecdotq.cuh"

// O alvo `bench-gemm-engine-gpu` linka so' a libamdhip64 (CMakeLists:210-217,
// fora do escopo desta frente), entao as fontes do loader entram por include
// direto -- a bancada usa o GgufLoader DE VERDADE.
#include "../src/backend/gguf.cpp"
#include "../src/backend/loader.cpp"

namespace {

using rdna4::block_q8_1;

constexpr int kQK = 256;
constexpr int kQ8 = 32;
constexpr int kIq3S = 12;
constexpr int kIq3XXS = 9;
constexpr int kIq4XS = 14;

// ---------------------------------------------------------------------------
// 0. Utilitarios de host
// ---------------------------------------------------------------------------
bool launch_ok(const char *what) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    std::printf("  LAUNCH FALHOU (%s): %s\n", what, hipGetErrorString(e));
    return false;
  }
  return true;
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

struct Diff {
  std::size_t n_diff = 0, n = 0;
  double max_abs = 0, rel_l2 = 0, max_ulps = 0;
};

uint32_t f2u(const float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, 4);
  return u;
}

Diff diff_bits(const std::vector<float> &a, const std::vector<float> &b, const std::size_t n) {
  Diff d;
  d.n = n;
  double sum_d = 0, sum_r = 0;
  for (std::size_t i = 0; i < n; ++i) {
    const uint32_t ua = f2u(a[i]), ub = f2u(b[i]);
    if (ua != ub) {
      ++d.n_diff;
      const long long ia = (long long)(ua ^ ((ua >> 31) ? 0x80000000u : 0u));
      const long long ib = (long long)(ub ^ ((ub >> 31) ? 0x80000000u : 0u));
      const double u = (double)std::llabs(ia - ib);
      if (u > d.max_ulps) d.max_ulps = u;
    }
    const double dd = (double)a[i] - (double)b[i];
    d.max_abs = std::max(d.max_abs, std::fabs(dd));
    sum_d += dd * dd;
    sum_r += (double)b[i] * (double)b[i];
  }
  d.rel_l2 = sum_r > 0 ? std::sqrt(sum_d / sum_r) : 0;
  return d;
}

// Ativacao float: rms ~1,9, a ordem de grandeza das ativacoes reais.
void fill_a(std::vector<float> &h, const std::size_t n, const unsigned seed) {
  h.resize(n);
  unsigned s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = (float)(s >> 8) / (float)(1u << 24);
    h[i] = (u - 0.5f) * 6.0f;
  }
}

// ---------------------------------------------------------------------------
// 1. Kernels de verificacao
// ---------------------------------------------------------------------------
// (a) O `vec_dot_*` DO MOTOR, um thread por (token, linha), na ordem de k
//     crescente -- a mesma ordem em que o GEMM soma (`k0` crescente, `kb`
//     crescente; dentro do bloco de 32 a soma int32 e' exata, entao a ordem
//     interna nao importa). E' o gate de bit-exatidao: divergencia aqui e'
//     divergencia de ARITMETICA, nao de ordem de soma.
template <class Traits>
__global__ void ref_engine_kernel(const void *__restrict__ W, const block_q8_1 *__restrict__ A,
                                  float *__restrict__ C, const int M, const int N, const int K) {
  constexpr int QK = Traits::qk;
  constexpr int STEP = Traits::vdr;  // kqs = vdr * sub-bloco (matvec.cuh:486)
  const std::int64_t out = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= (std::int64_t)M * N) return;
  const int m = (int)(out / N);
  const int n = (int)(out % N);
  const int bpr = K / QK;
  const char *row =
      (const char *)W + (std::int64_t)n * bpr * (std::int64_t)sizeof(typename Traits::block_t);
  const block_q8_1 *q = A + (std::int64_t)m * (K / kQ8);
  float acc = 0.0f;
  for (int ib = 0; ib < bpr; ++ib) {
#pragma unroll
    for (int s = 0; s < 8; ++s) acc += Traits::dot((const void *)row, q + ib * 8, ib, STEP * s);
  }
  C[out] = acc;
}

// (b) Deposito do staging iq3_s isolado (mesma funcao que o kernel usa).
__global__ void dump_stage_iq3s_kernel(const rdna4::block_iq3_s *__restrict__ W,
                                       int *__restrict__ out, const int nrow, const int bpr,
                                       const int nsub) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= nsub) return;
  const int s = t % 8;
  const char *blk = (const char *)(W + (std::int64_t)nrow * bpr + (t / 8));
  const rdna4::SolverIq3S::Pf pf = rdna4::SolverIq3S::load(blk, s);
  int tmp[8];
  float dw;
  int scf;
  rdna4::SolverIq3S::store(pf, s, tmp, &dw, &scf);
  int *o = out + (std::size_t)t * 9;
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = tmp[i];
  o[8] = (__float_as_int(dw) & 0x00FFFFFF) | (scf << 24);
}

// Copia a LUT do DEVICE: o oraculo de CPU usa a MESMA tabela que o kernel.
__global__ void copy_lut_kernel(uint32_t *__restrict__ out, const int n) {
  for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) out[i] = rdna4::iq3s_grid[i];
}

// ---------------------------------------------------------------------------
// 2. Oraculo de CPU (iq3_s): a algebra do `vec_dot_iq3_s_q8_1` sobre os bytes
//    crus do GGUF, com a LUT lida do device, produto em int32 EXATO,
//    `1+2*sc` em INTEIRO, `d_w*d_a` em fp32 e acumulador fp32 com std::fma
//    (que e' __fmaf_rn).
// ---------------------------------------------------------------------------
void cpu_iq3s_32(const unsigned char *b, const uint32_t *lut, int *iw, float *dw, int *scf) {
  const uint16_t d_h = (uint16_t)(b[0] | (b[1] << 8));
  const unsigned char *qs = b + 2;
  const unsigned char *qh = b + 2 + 64;
  const unsigned char *sg = b + 2 + 72;
  const unsigned char *sc = b + 2 + 104;
  for (int s = 0; s < 8; ++s) {
    for (int j = 0; j < 32; ++j) {
      const int il = j / 8, e = j % 8;
      const int widx = 2 * il + (e < 4 ? 0 : 1);
      const int sh = (e < 4 ? (8 - 2 * il) : (7 - 2 * il));
      const uint32_t g = lut[(int)qs[8 * s + widx] | ((qh[s] << sh) & 0x100)];
      const int mag = (int)((g >> (8 * (e % 4))) & 0xFFu);
      const int sb = (sg[4 * s + il] >> e) & 1;
      iw[s * 32 + j] = sb ? -mag : mag;
    }
    dw[s] = rdna4::fp16_to_float(d_h);
    scf[s] = 1 + 2 * ((sc[s / 2] >> (4 * (s % 2))) & 0x0F);
  }
}

float cpu_oracle_iq3s(const unsigned char *wraw, const std::int64_t bpr, const uint32_t *lut,
                      const block_q8_1 *q8, const int m, const int n, const int K,
                      const std::size_t block_bytes) {
  const unsigned char *row = wraw + (std::size_t)n * (std::size_t)bpr * block_bytes;
  const block_q8_1 *qrow = q8 + (std::size_t)m * (std::size_t)(K / kQ8);
  float acc = 0.0f;
  for (std::int64_t ib = 0; ib < bpr; ++ib) {
    int iw[256];
    float dws[8];
    int scfs[8];
    cpu_iq3s_32(row + (std::size_t)ib * block_bytes, lut, iw, dws, scfs);
    for (int s = 0; s < 8; ++s) {
      const block_q8_1 *q = qrow + ib * 8 + s;
      int isum = 0;
      for (int j = 0; j < 32; ++j) isum += iw[s * 32 + j] * (int)q->qs[j];
      const float da = rdna4::fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
      const float d = dws[s] * da;
      acc = std::fma(d, (float)(isum * scfs[s]), acc);
    }
  }
  return acc;
}

// ---------------------------------------------------------------------------
// 3. Despacho por tipo
// ---------------------------------------------------------------------------
struct TypeDesc {
  const char *name;
  int dt;
  void (*ref)(const void *, const block_q8_1 *, float *, int, int, int);
  std::size_t block_bytes;
};

void ref_iq3s(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TIQ3S_S>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_iq3xxs(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TIQ3XXS_S>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_iq4xs(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TIQ4XS>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}

const TypeDesc *type_of(const int dt) {
  static const TypeDesc kTypes[] = {
      {"iq3_s", kIq3S, ref_iq3s, sizeof(rdna4::block_iq3_s)},
      {"iq3_xxs", kIq3XXS, ref_iq3xxs, sizeof(rdna4::block_iq3_xxs)},
      {"iq4_xs", kIq4XS, ref_iq4xs, sizeof(rdna4::block_iq4_xs)},
  };
  for (const TypeDesc &t : kTypes)
    if (t.dt == dt) return &t;
  return nullptr;
}

const char *type_name(const int dt) {
  const TypeDesc *t = type_of(dt);
  if (t) return t->name;
  switch (dt) {
    case 1: return "q8_0";   case 2: return "q2_K";   case 3: return "q3_K";
    case 4: return "q4_K";   case 5: return "q5_K";   case 6: return "q6_K";
    case 7: return "iq2_xxs"; case 8: return "iq2_xs"; case 10: return "iq1_s";
    case 11: return "iq4_nl"; case 13: return "iq2_s";
    default: return "?";
  }
}

// ---------------------------------------------------------------------------
// 4. Atributos de kernel (registradores / LDS) das configs da bancada
// ---------------------------------------------------------------------------
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB>
bool attrs_of(hipFuncAttributes &attr, int &lds) {
  constexpr int NKB = BK / 32;
  constexpr int WSB = BK + 16;
  lds = (DBUF ? 2 : 1) * (BN * WSB + NKB * BN * 8);
  auto *fn = &rdna4::gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB>;
  return hipFuncGetAttributes(&attr, (const void *)fn) == hipSuccess;
}

void print_attrs(const char *tag, const int dt, const bool use_cfg, const int cfg) {
  hipFuncAttributes a{};
  int lds = 0;
  bool ok = false;
  if (!use_cfg) {
    ok = rdna4::gemm_attrs(dt, 128, a, lds);
  } else {
#define AT(Traits, BM, BN, BK, RM, RN, DB, WH, MB) ok = attrs_of<Traits, BM, BN, BK, RM, RN, DB, WH, MB>(a, lds)
    switch (cfg) {
      case 1: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, true, true, 1); break;
      case 2: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, false, true, 1); break;
      case 3: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, true, true, 2); break;
      case 5: AT(rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1); break;
      case 6: AT(rdna4::SolverIq3S, 128, 64, 64, 8, 4, true, true, 1); break;
      case 7: AT(rdna4::SolverIq3S, 64, 64, 64, 4, 4, true, true, 1); break;
      case 8: AT(rdna4::SolverIq3S, 128, 64, 64, 4, 4, true, true, 1); break;
      default: ok = false; break;
    }
#undef AT
  }
  if (!ok) {
    std::printf("  %-30s attrs indisponiveis\n", tag);
    return;
  }
  std::printf("  %-30s VGPR=%3d  LDS=%5d B (pedido %5d B)  spill/lane=%d B\n", tag, a.numRegs, lds,
              lds, (int)a.localSizeBytes);
}

// ---------------------------------------------------------------------------
// 5. Cronometragem
// ---------------------------------------------------------------------------
struct Timing {
  double ms = 0;  // minimo de `reps`
  double warm_ms = 0;
};

// Aquecimento de DPM ANTES de medir (a frente F mediu a primeira celula de um
// processo ~47 % mais lenta que as seguintes; a frente H mediu o "pico" 2x baixo
// por medir em janelas curtas) e minimo de `reps` repeticoes.
Timing time_it(const std::function<void()> &fn, const int reps, const double warm_target_ms) {
  EvPair ev;
  fn();
  (void)hipDeviceSynchronize();
  Timing t;
  for (int i = 0; i < 20000 && t.warm_ms < warm_target_ms; ++i) t.warm_ms += ev(fn);
  t.ms = 1e9;
  for (int r = 0; r < reps; ++r) t.ms = std::min(t.ms, ev(fn));
  return t;
}

// GEMV em lote para um M qualquer: o motor so' instancia ate' N=64
// (matvec_batch_cap), entao M>64 vira varias chamadas -- e' o que o grafo faz.
void gemv_batch(const int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                const std::int64_t nrows, const std::int64_t ncols, const std::int64_t act_stride,
                const int M, const int sub, const bool check) {
  for (int t0 = 0; t0 < M; t0 += sub) {
    const int n = std::min(sub, M - t0);
    if (!rdna4::matvec_launch_batch(dt, d_w, d_a + (std::int64_t)t0 * act_stride,
                                    d_o + (std::int64_t)t0 * nrows, nrows, ncols, act_stride, n,
                                    nullptr)) {
      std::printf("  matvec_launch_batch falhou (N=%d)\n", n);
      return;
    }
    if (check && !launch_ok("matvec_launch_batch")) return;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  std::vector<int> ms_list{16, 64, 128, 512};
  std::vector<int> cfg_list;
  int reps = 5;
  std::int64_t n_ovr = -1, k_ovr = -1;
  bool verify = true, quick = false, list_only = false, types_only = false;
  std::string tensor_name;
  const char *model = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--m") && i + 1 < argc) {
      ms_list.clear();
      const char *v = argv[++i];
      while (*v) {
        ms_list.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    } else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
      n_ovr = std::atoll(argv[++i]);  // so' nas chamadas: a sonda de onda da frente G
    } else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) {
      k_ovr = std::atoll(argv[++i]);
    } else if (!std::strcmp(argv[i], "--cfg") && i + 1 < argc) {
      const char *v = argv[++i];
      while (*v) {
        cfg_list.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--tensor") && i + 1 < argc) {
      tensor_name = argv[++i];
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      verify = false;
    } else if (!std::strcmp(argv[i], "--quick")) {
      quick = true;
    } else if (!std::strcmp(argv[i], "--list")) {
      list_only = true;
    } else if (!std::strcmp(argv[i], "--types")) {
      types_only = true;
    } else if (argv[i][0] != '-') {
      model = argv[i];
    }
  }
  if (!model) {
    std::printf(
        "uso: bench-gemm-engine-gpu <model.gguf> [--m 16,64,128,512] [--reps 5]\n"
        "                        [--tensor NOME] [--list] [--cfg 1,5,6,7]\n"
        "                        [--no-verify] [--quick] [--types]\n");
    return 1;
  }
  {
    hipDeviceProp_t dp{};
    if (hipGetDeviceProperties(&dp, 0) == hipSuccess) {
      // ATENCAO: `multiProcessorCount` do HIP reporta 32 neste gfx1201 enquanto o
      // rocminfo reporta 64 CUs (`Compute Unit: 64`, conferido na mesma janela):
      // a analise de onda (136 CTAs = 2,125 ondas) usa os 64 do rocminfo.
      std::printf("device: %s (%s), hip multiProcessorCount=%d (rocminfo: 64 CUs), LDS/bloco max %zu B\n",
                  dp.name, dp.gcnArchName, dp.multiProcessorCount, (size_t)dp.sharedMemPerBlock);
    }
  }

  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(model, err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }

  // ---- 0. cobertura do gemm_launch ---------------------------------------
  if (types_only || list_only) {
    // Buffer pequeno e VALIDO (zeros) para os tipos cobertos: nesse caso o
    // kernel roda de verdade e o `hipGetLastError` prova que ele foi lancado
    // (era o bug das bancadas que reportaram 1198 T-MAC/s de um kernel que nao
    // rodava). Os tipos nao cobertos retornam `false` antes de qualquer lancamento.
    void *dWt = nullptr, *dOt = nullptr;
    block_q8_1 *dAt = nullptr;
    const std::int64_t nrt = 128, nct = 256, ast = 8;
    const bool alloc_ok = hipMalloc(&dWt, 1u << 20) == hipSuccess &&
                          hipMalloc(&dAt, (std::size_t)128 * ast * sizeof(block_q8_1)) == hipSuccess &&
                          hipMalloc(&dOt, (std::size_t)128 * nrt * sizeof(float)) == hipSuccess;
    if (alloc_ok) {
      (void)hipMemset(dWt, 0, 1u << 20);
      (void)hipMemset(dAt, 0, (std::size_t)128 * ast * sizeof(block_q8_1));
      (void)hipMemset(dOt, 0, (std::size_t)128 * nrt * sizeof(float));
      (void)hipDeviceSynchronize();
    }
    std::printf("\n== cobertura do gemm_launch (dt -> tipo) ==\n");
    for (int dt : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}) {
      bool ok = false;
      hipError_t e = hipSuccess;
      if (alloc_ok) {
        ok = rdna4::gemm_launch(dt, dWt, dAt, (float *)dOt, nrt, nct, ast, 128, nullptr);
        e = hipGetLastError();
      }
      std::printf("  dt=%-3d %-9s -> %s%s\n", dt, type_name(dt), ok ? "SIM" : "nao",
                  e != hipSuccess ? " (erro de lancamento!)" : "");
    }
    if (alloc_ok) {
      (void)hipFree(dWt);
      (void)hipFree(dAt);
      (void)hipFree(dOt);
    }
    if (types_only) return 0;
  }
  if (list_only) {
    std::printf("\n== tensores do modelo (nome, dims=[K,N], tipo, MB) ==\n");
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      const auto &t = ld.meta().tensors[i];
      if ((std::int64_t)t.dims[0] % 256 != 0) continue;
      std::printf("  %-34s [%6lld x %6lld] %-9s %8.1f MB\n", t.name.c_str(), (long long)t.dims[0],
                  (long long)t.dims[1], type_name((int)ld.dtype(i)),
                  (double)ld.tensor_bytes(i) / 1e6);
    }
    return 0;
  }

  // ---- 1. os tensores que importam ---------------------------------------
  // blk.3.ffn_up (17408x5120) e blk.2.ffn_down (5120x17408) sao os dois shapes
  // citados no plano; a projecao de atencao DESTE modelo e' fundida
  // (attn_qkv, 10240x5120, iq4_xs) -- nao existe tensor 5120x5120 nele.
  std::vector<std::string> names;
  if (!tensor_name.empty()) {
    names.push_back(tensor_name);
  } else {
    // Os tres shapes que importam + os tres tipos cobertos: K grande
    // (ffn_down, 17408), N grande (ffn_up, 17408), a projecao de atencao
    // (attn_qkv 10240x5120, iq4_xs neste modelo -- nao ha' tensor 5120x5120) e
    // um tensor iq3_xxs, que e' o segundo tipo em bytes do inventario.
    names = {"blk.3.ffn_up.weight", "blk.2.ffn_down.weight", "blk.2.attn_qkv.weight",
             "blk.1.ffn_down.weight"};
  }

  const int MMAX = *std::max_element(ms_list.begin(), ms_list.end());
  const int MPAD = ((std::max(MMAX, 128) + 127) / 128) * 128;
  const int kM16 = 16, kNc = 256;  // caso pequeno da comparacao com o motor

  for (const std::string &nm : names) {
    const auto *found = ld.find(nm.c_str());
    if (!found) {
      std::printf("\n[tensor %s nao existe no modelo -- pulado]\n", nm.c_str());
      continue;
    }
    const std::size_t ti = (std::size_t)(found - ld.meta().tensors.data());
    const auto &info = ld.meta().tensors[ti];
    const int dt = (int)ld.dtype(ti);
    const TypeDesc *td = type_of(dt);
    const std::int64_t K = (std::int64_t)info.dims[0];
    const std::int64_t N = (std::int64_t)info.dims[1];
    std::printf(
        "\n=====================================================================\n"
        "tensor %s  [K=%lld x N=%lld]  tipo=%s (dt=%d)  %.1f MB  %.4f B/peso\n",
        info.name.c_str(), (long long)K, (long long)N, type_name(dt), dt,
        (double)ld.tensor_bytes(ti) / 1e6,
        (double)ld.tensor_bytes(ti) / ((double)N * (double)K));
    if (!td) {
      std::printf("  tipo nao coberto pelo gemm_launch -- pulado\n");
      continue;
    }
    if (N % 128 != 0 || K % kQK != 0) {
      std::printf("  N tem de ser multiplo de 128 e K de 256 -- pulado\n");
      continue;
    }
    // Overrides de FORMA (a sonda de quantizacao de onda da frente G: `--n 16384`
    // da 128 CTAs em vez de 136, i.e. 2 ondas exatas de 64 CUs). Os buffers
    // continuam dimensionados pela forma REAL -- so' as chamadas mudam.
    const std::int64_t Nr = (n_ovr > 0 && n_ovr <= N) ? n_ovr : N;
    const std::int64_t Kr = (k_ovr > 0 && k_ovr <= K) ? k_ovr : K;
    const std::int64_t act_full = K / kQ8;  // stride REAL da ativacao (nao muda com --k)
    if (Nr != N || Kr != K)
      std::printf("  FORMA REDUZIDA: N=%lld K=%lld (a ativacao continua com stride %lld)\n",
                  (long long)Nr, (long long)Kr, (long long)act_full);

    const std::uint64_t want = (std::uint64_t)N * (std::uint64_t)(K / kQK) * td->block_bytes;
    std::vector<std::uint8_t> w_raw;
    if (!ld.load_tensor_range(ti, 0, want, w_raw, err)) {
      std::printf("  load_tensor_range: %s\n", err.c_str());
      continue;
    }
    void *dW = nullptr;
    float *dAf = nullptr;
    block_q8_1 *dAQ = nullptr, *hAQ = nullptr;
    float *dC = nullptr, *dC2 = nullptr;
    if (hipMalloc(&dW, w_raw.size()) != hipSuccess) {
      std::printf("  hipMalloc(dW) falhou\n");
      continue;
    }
    (void)hipMemcpy(dW, w_raw.data(), w_raw.size(), hipMemcpyHostToDevice);
    const std::size_t nact = (std::size_t)MPAD * (std::size_t)(K / kQ8);
    std::vector<float> hA;
    fill_a(hA, (std::size_t)MPAD * (std::size_t)K, 7u);
    for (std::size_t i = (std::size_t)MMAX * (std::size_t)K; i < hA.size(); ++i) hA[i] = 0.0f;
    if (hipMalloc(&dAf, hA.size() * sizeof(float)) != hipSuccess ||
        hipMalloc(&dAQ, nact * sizeof(block_q8_1)) != hipSuccess ||
        hipMalloc(&dC, (std::size_t)MPAD * (std::size_t)N * sizeof(float)) != hipSuccess ||
        hipMalloc(&dC2, (std::size_t)MPAD * (std::size_t)N * sizeof(float)) != hipSuccess) {
      std::printf("  hipMalloc falhou (ativacao/saida)\n");
      continue;
    }
    (void)hipMemcpy(dAf, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice);
    if (!rdna4::quantize_q8_1_batch_launch(dAf, dAQ, K / kQ8, MPAD, K, nullptr) ||
        !launch_ok("quantize_q8_1_batch")) {
      continue;
    }
    (void)hipDeviceSynchronize();
    hAQ = (block_q8_1 *)std::malloc(nact * sizeof(block_q8_1));
    (void)hipMemcpy(hAQ, dAQ, nact * sizeof(block_q8_1), hipMemcpyDeviceToHost);

    std::printf("  atributos (config que `gemm_launch` escolhe em M=128):\n");
    print_attrs("producao (BM64 BN128 BK64 RM4 RN8)", dt, false, 0);

    // ---------------------------------------------------------------------
    // 2. VERIFICACAO (antes de qualquer cronometragem)
    // ---------------------------------------------------------------------
    if (verify) {
      std::printf("  -- verificacao --\n");
      // (a) staging iq3_s byte a byte contra o oraculo de CPU
      if (dt == kIq3S) {
        const int nsub = 64;  // 8 super-blocos = 2048 pesos da linha 0
        int *dStage = nullptr;
        uint32_t *dLut = nullptr;
        std::vector<uint32_t> lut(512, 0);
        if (hipMalloc(&dStage, (std::size_t)nsub * 9 * sizeof(int)) != hipSuccess ||
            hipMalloc(&dLut, lut.size() * sizeof(uint32_t)) != hipSuccess) {
          std::printf("  hipMalloc(stage/lut) falhou\n");
          continue;
        }
        copy_lut_kernel<<<1, 128>>>(dLut, 512);
        if (!launch_ok("copy_lut")) continue;
        dump_stage_iq3s_kernel<<<(nsub + 63) / 64, 64>>>((const rdna4::block_iq3_s *)dW, dStage, 0,
                                                         (int)(K / kQK), nsub);
        if (!launch_ok("dump_stage")) continue;
        (void)hipDeviceSynchronize();
        std::vector<int> staged((std::size_t)nsub * 9);
        (void)hipMemcpy(staged.data(), dStage, staged.size() * sizeof(int), hipMemcpyDeviceToHost);
        (void)hipMemcpy(lut.data(), dLut, lut.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
        int bad_w = 0, bad_sc = 0, first_bad = -1;
        for (int t = 0; t < nsub; ++t) {
          int iw[256];
          float dws[8];
          int scfs[8];
          cpu_iq3s_32(w_raw.data() + (std::size_t)(t / 8) * td->block_bytes, lut.data(), iw, dws,
                      scfs);
          const int s = t % 8;
          for (int j = 0; j < 32; ++j) {
            const int got = (staged[(std::size_t)t * 9 + j / 4] >> (8 * (j % 4))) & 0xFF;
            const int wantb = iw[s * 32 + j] & 0xFF;
            if (got != wantb) {
              ++bad_w;
              if (first_bad < 0) first_bad = t * 32 + j;
            }
          }
          const int packed = staged[(std::size_t)t * 9 + 8];
          if ((packed & 0x00FFFFFF) != (int)(f2u(dws[s]) & 0x00FFFFFFu) || (packed >> 24) != scfs[s])
            ++bad_sc;
        }
        std::printf(
            "  staging int8: bytes diferentes %d de 2048%s ; d_w/(1+2*sc) errados %d de 64%s%s\n"
            "    (2048 pesos da linha 0 contra a algebra do `vec_dot_iq3_s_q8_1` em CPU sobre os\n"
            "     bytes crus + a LUT lida do device)\n",
            bad_w, bad_w ? "" : " (ok)", bad_sc, bad_sc ? "" : " (ok)",
            first_bad >= 0 ? "  <-- PRIMEIRO BYTE ERRADO" : "");
        (void)hipFree(dStage);
        (void)hipFree(dLut);
        if (bad_w || bad_sc) {
          std::printf("  REPROVADO: o staging nao reproduz a algebra do motor\n");
          continue;
        }
      }

      // (b) caso pequeno: nao-finitos, zeros, e oraculo de CPU (iq3_s)
      {
        const int Mc = 4, Nc = 64;
        std::vector<float> got((std::size_t)Mc * (std::size_t)Nc, 0.f);
        (void)hipMemset(dC, 0, (std::size_t)Mc * (std::size_t)Nc * sizeof(float));
        if (!rdna4::gemm_launch(dt, dW, dAQ, dC, Nc, K, K / kQ8, Mc, nullptr) ||
            !launch_ok("gemm_launch (caso pequeno)")) {
          std::printf("  gemm_launch(M=%d) falhou\n", Mc);
          continue;
        }
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(got.data(), dC, got.size() * sizeof(float), hipMemcpyDeviceToHost);
        int n_nonfinite = 0, n_zero = 0;
        for (float v : got) {
          if (!std::isfinite(v)) ++n_nonfinite;
          if (v == 0.f) ++n_zero;
        }
        std::printf("  caso pequeno M=%d N=%d K=%lld: nao-finitos=%d zeros=%d de %zu\n", Mc, Nc,
                    (long long)K, n_nonfinite, n_zero, got.size());
        if (n_nonfinite) {
          std::printf("  REPROVADO: saida nao-finita (bug de kernel, nao de precisao)\n");
          continue;
        }
        if (n_zero == (int)got.size()) {
          std::printf("  REPROVADO: C toda zero (o kernel nao rodou?)\n");
          continue;
        }
        if (dt == kIq3S) {
          uint32_t *dLut = nullptr;
          std::vector<uint32_t> lut(512, 0);
          if (hipMalloc(&dLut, lut.size() * sizeof(uint32_t)) != hipSuccess) continue;
          copy_lut_kernel<<<1, 128>>>(dLut, 512);
          if (!launch_ok("copy_lut")) continue;
          (void)hipMemcpy(lut.data(), dLut, lut.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
          (void)hipFree(dLut);
          double mx = 0, mrel = 0, mref = 0;
          std::size_t ndiff = 0;
          for (int m = 0; m < Mc; ++m) {
            for (int n = 0; n < Nc; ++n) {
              const float wantv = cpu_oracle_iq3s(w_raw.data(), K / kQK, lut.data(), hAQ, m, n,
                                                  (int)K, td->block_bytes);
              const float g = got[(std::size_t)m * Nc + n];
              if (f2u(wantv) != f2u(g)) ++ndiff;
              const double d = std::fabs((double)g - (double)wantv);
              mx = std::max(mx, d);
              mref = std::max(mref, std::fabs((double)wantv));
              if (std::fabs((double)wantv) > 0) mrel = std::max(mrel, d / std::fabs((double)wantv));
            }
          }
          std::printf(
              "  oraculo de CPU (M=%d N=%d K=%lld: 256 saidas x %lld blocos de 32; produto int32\n"
              "    exato, `*(1+2*sc)` inteiro, d_w*d_a em fp32, acumulador fp32 com std::fma):\n"
              "    max|ref| %.6g ; max|gpu-cpu| %.3g ; max rel %.3g ; BITS diferentes %zu de %d%s\n",
              Mc, Nc, (long long)K, (long long)(K / 32), mref, mx, mrel, ndiff, Mc * Nc,
              ndiff ? "" : "  (bit-exato)");
        }
      }

      // (c) o `vec_dot_*` DO MOTOR, um thread por (token, linha)
      {
        const int Me = kM16, Ne = kNc;
        std::vector<float> eng((std::size_t)Me * (std::size_t)Ne);
        std::vector<float> mine((std::size_t)Me * (std::size_t)Ne);
        (void)hipMemset(dC2, 0, (std::size_t)Me * (std::size_t)Ne * sizeof(float));
        td->ref((const void *)dW, dAQ, dC2, Me, Ne, (int)K);
        if (!launch_ok("ref_engine_kernel")) continue;
        (void)hipMemset(dC, 0, (std::size_t)Me * (std::size_t)Ne * sizeof(float));
        if (!rdna4::gemm_launch(dt, dW, dAQ, dC, Ne, K, K / kQ8, Me, nullptr) ||
            !launch_ok("gemm_launch (ref)")) {
          continue;
        }
        (void)hipDeviceSynchronize();
        (void)hipMemcpy(eng.data(), dC2, eng.size() * sizeof(float), hipMemcpyDeviceToHost);
        (void)hipMemcpy(mine.data(), dC, mine.size() * sizeof(float), hipMemcpyDeviceToHost);
        const Diff d = diff_bits(mine, eng, mine.size());
        std::printf(
            "  MOTOR (`vec_dot_%s_q8_1` do vecdotq.cuh, 1 thread por saida, mesma ordem de k):\n"
            "    M=%d N=%d -> BITS diferentes %zu de %zu ; max ulp %.0f ; max|d| %.3g ; rel-L2 "
            "%.3g%s\n",
            td->name, Me, Ne, d.n_diff, d.n, d.max_ulps, d.max_abs, d.rel_l2,
            d.n_diff ? "   <-- DIVERGE" : "   (bit-exato)");
      }
    }

    // ---------------------------------------------------------------------
    // 3. VELOCIDADE
    // ---------------------------------------------------------------------
    std::printf("  -- velocidade (min de %d, com aquecimento de DPM antes) --\n", reps);
    std::printf(
        "  %5s | %9s %9s | %9s %9s | %9s %9s | %9s %9s\n"
        "  %5s | %29s | %29s | %9s %9s\n",
        "M", "GEMM ms", "T-MAC/s", "GEMV64 ms", "T-MAC/s", "GEMV16 ms", "T-MAC/s", "x GEMV64",
        "x GEMV16", "", "GEMM tilejado (este, D2)", "matvec_launch_batch sub-lote 64",
        "razao G1", "razao G2");
    const double macs_per = (double)Nr * (double)Kr;
    double noise_a = 0, noise_b = 0, noise_m = 0;
    for (std::size_t mi = 0; mi < ms_list.size(); ++mi) {
      const int M = ms_list[mi];
      const double macs = macs_per * (double)M;
      const std::size_t mid = ms_list.size() > 2 ? 2 : ms_list.size() - 1;
      bool warned = false;
      auto fn_gemm = [&] {
        if (!rdna4::gemm_launch(dt, dW, dAQ, dC, Nr, Kr, act_full, M, nullptr) && !warned) {
          warned = true;
          std::printf("  gemm_launch(M=%d) retornou false\n", M);
        }
      };
      const Timing tg = time_it(fn_gemm, reps, quick ? 60.0 : 250.0);
      if (!launch_ok("gemm_launch (time)")) continue;
      auto fn64 = [&] { gemv_batch(dt, dW, dAQ, dC2, Nr, Kr, act_full, M, 64, false); };
      auto fn16 = [&] { gemv_batch(dt, dW, dAQ, dC2, Nr, Kr, act_full, M, 16, false); };
      const Timing t64 = time_it(fn64, reps, quick ? 60.0 : 250.0);
      if (!launch_ok("matvec_launch_batch (N=64)")) continue;
      const Timing t16 = time_it(fn16, reps, quick ? 60.0 : 250.0);
      if (!launch_ok("matvec_launch_batch (N=16)")) continue;
      if (mi == mid) {
        // piso de ruido: a MESMA medicao do GEMM, uma segunda vez, nesta janela
        const Timing t2 = time_it(fn_gemm, reps, quick ? 60.0 : 250.0);
        noise_a = tg.ms;
        noise_b = t2.ms;
        noise_m = (double)M;
      }
      const double tf = macs / (tg.ms * 1e-3) / 1e12;
      const double r64 = macs / (t64.ms * 1e-3) / 1e12;
      const double r16 = macs / (t16.ms * 1e-3) / 1e12;
      std::printf("  %5d | %9.3f %9.2f | %9.3f %9.2f | %9.3f %9.2f | %8.2fx %8.2fx\n", M, tg.ms, tf,
                  t64.ms, r64, t16.ms, r16, tf / std::max(r64, 1e-9), tf / std::max(r16, 1e-9));

      // ---- corretude contra o GEMV que embarca, nas MESMAS entradas -------
      if (!rdna4::gemm_launch(dt, dW, dAQ, dC, Nr, Kr, act_full, M, nullptr) ||
          !launch_ok("gemm_launch (diff)")) {
        continue;
      }
      gemv_batch(dt, dW, dAQ, dC2, Nr, Kr, act_full, M, std::min(M, 64), true);
      (void)hipDeviceSynchronize();
      const std::size_t n_el = (std::size_t)M * (std::size_t)Nr;
      std::vector<float> a(n_el), b(n_el);
      (void)hipMemcpy(a.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
      (void)hipMemcpy(b.data(), dC2, n_el * sizeof(float), hipMemcpyDeviceToHost);
      const Diff d = diff_bits(a, b, n_el);
      std::size_t nz = 0, nnf = 0;
      for (std::size_t i = 0; i < n_el; ++i) {
        if (a[i] == 0.f) ++nz;
        if (!std::isfinite(a[i])) ++nnf;
      }
      std::printf(
          "        x GEMV em lote (sub-lote N=%d): %zu de %zu elementos com BITS diferentes "
          "(%.2f %%) ;\n"
          "        max ulp %.0f ; max|d| %.3g ; rel-L2 %.3g ; zeros %zu ; nao-finitos %zu\n",
          std::min(M, 64), d.n_diff, n_el, 100.0 * (double)d.n_diff / (double)n_el, d.max_ulps,
          d.max_abs, d.rel_l2, nz, nnf);
    }
    std::printf(
        "  piso de ruido (a MESMA medicao do GEMM duas vezes, M=%.0f): %.3f ms e %.3f ms -> "
        "%.3fx\n",
        noise_m, noise_a, noise_b, noise_b / std::max(noise_a, 1e-9));

    // ---- A/B de configuracao (so' iq3_s, para limitar instanciacoes) ------
    if (dt == kIq3S && !cfg_list.empty()) {
      std::printf("\n  -- A/B de configuracao (iq3_s) --\n");
      for (const int cfg : cfg_list) {
        char tag[96];
        std::snprintf(tag, sizeof(tag), "cfg %d", cfg);
        print_attrs(tag, dt, true, cfg);
        for (const int M : ms_list) {
          bool warned = false;
          auto fn = [&] {
            if (!rdna4::gemm_launch_cfg(dt, dW, dAQ, dC, Nr, Kr, act_full, M, cfg, nullptr) && !warned) {
              warned = true;
              std::printf("  cfg %d indisponivel em M=%d\n", cfg, M);
            }
          };
          const Timing t = time_it(fn, reps, quick ? 60.0 : 200.0);
          if (!launch_ok("gemm_launch_cfg")) continue;
          const double macs = macs_per * (double)M;
          std::printf("  %5d | %9.3f ms  %9.2f T-MAC/s\n", M, t.ms, macs / (t.ms * 1e-3) / 1e12);
        }
      }
    }

    std::free(hAQ);
    (void)hipFree(dW);
    (void)hipFree(dAf);
    (void)hipFree(dAQ);
    (void)hipFree(dC);
    (void)hipFree(dC2);
  }

  std::printf(
      "\n== referencias (mesma maquina; a janela de cada uma esta' no doc) ==\n"
      "  prefill de hoje (matvec em lote, chunk 16) ....... 3,0-3,2 T-MAC/s (medido no grafo)\n"
      "  V6 da frente G (f16 dot2, blocos reais) .......... 12,74 T em M=128, 14,71 em M=512\n"
      "  dp4a tilejado da frente H (mesmo staging) ........ 11,76 T em M=128, 12,47 em M=512\n"
      "  WMMA int8 da frente H ............................ 19,7 T em M=128, 22,8 em M=512\n"
      "  llama.cpp Vulkan coopmat f16, pp512 .............. 32,7 T-MAC/s\n");
  return 0;
}
