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
// `--cfg` aceita as variantes A/B de `gemm_launch_cfg` (gemm.cuh): 1-8 sao
// geometrias de iq3_s, 9/10 sao o STAGING CRU (campos crus na LDS, decodificacao
// dentro do laco de consumo) em BM64/BM16 e 11/12 sao o modo "so' staging"
// (mesma grade, mesmos threads, conta trocada por 4 leituras da LDS) do caminho
// de producao e do caminho cru -- 11/12 existem so' para iq3_s. Todo cfg que
// calcula resultado passa pelo gate de bit-exatidao contra o `vec_dot_*` antes
// de ser cronometrado; a fracao de staging sai de 11 (ou 12) sobre o GEMM cheio.
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
constexpr int kIq2XXS = 7;
constexpr int kQ80 = 1;
constexpr int kIq1S = 10;
constexpr int kIq4NL = 11;

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

void u2f(const uint32_t u, float &f) { std::memcpy(&f, &u, 4); }

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
  constexpr int STEP = Traits::vdr;  // kqs = vdr * (tid % (qi/vdr)) (matvec.cuh:486)
  // NUMERO DE CHAMADAS por super-bloco: `qi/vdr`. Nos tipos IQ cobertos isso da'
  // 8 (qi=16/vdr=2, qi=32/vdr=4), que e' o que o laco original fixava; nos
  // k-quants da' 16 (q2_K, q3_K, q4_K, q5_K) e 32 (q6_K) -- com o 8 fixo o
  // oraculo somava SO' METADE do super-bloco (bug latente, invisivel enquanto
  // so' tipos com qi/vdr == 8 passavam por aqui).
  constexpr int NCALL = Traits::qi / Traits::vdr;
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
    for (int s = 0; s < NCALL; ++s) acc += Traits::dot((const void *)row, q + ib * 8, ib, STEP * s);
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
// 2b. k-quants: oracle de CPU
// ---------------------------------------------------------------------------
// (i) `cpu_dequant_*`: transcricao LITERAL de `include/rdna4/dequant.cuh`, com
//     `tid` virando o indice do laco. E' a decodificacao CANONICA do bloco
//     (llama.cpp) e usa uma parametrizacao DIFERENTE da do vec_dot, entao ela e'
//     um oraculo independente do indexamento que o kernel usa.
// (ii) `cpu_stage_*`: a mesma algebra do `Solver*::store()` do gemm.cuh (quant
//     cru + escalas) -- e' o que o oraculo de GEMM usa para reproduzir o kernel
//     em CPU bit a bit.
struct KqCpu {
  int q[32];        // quant cru por elemento do bloco de 32 (ja' com o offset do tipo)
  float dw, dmn;    // d (ou dm.d) e dmin
  int s0, s1;       // escalas empacotadas como no `KqScal`
};

static inline uint16_t rd16(const unsigned char *p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static inline float rdf16(const unsigned char *p) { return rdna4::fp16_to_float(rd16(p)); }

// ---- dequant canonico -----------------------------------------------------
void cpu_dequant_q2k(const unsigned char *b, float *yy) {
  const unsigned char *sc = b, *qs = b + 16;
  const float d = rdf16(b + 80), dmin = rdf16(b + 82);
  for (int tid = 0; tid < 64; ++tid) {
    const int n = tid / 32, l = tid - 32 * n, is = 8 * n + l / 16;
    const int q = qs[32 * n + l];
    float *y = yy + 128 * n;
    y[l + 0] = d * (sc[is + 0] & 0xF) * ((q >> 0) & 3) - dmin * (sc[is + 0] >> 4);
    y[l + 32] = d * (sc[is + 2] & 0xF) * ((q >> 2) & 3) - dmin * (sc[is + 2] >> 4);
    y[l + 64] = d * (sc[is + 4] & 0xF) * ((q >> 4) & 3) - dmin * (sc[is + 4] >> 4);
    y[l + 96] = d * (sc[is + 6] & 0xF) * ((q >> 6) & 3) - dmin * (sc[is + 6] >> 4);
  }
}
void cpu_dequant_q3k(const unsigned char *b, float *yy) {
  const unsigned char *hm = b, *qs = b + 32, *sc = b + 96;
  const float d = rdf16(b + 108);
  for (int tid = 0; tid < 64; ++tid) {
    const int r = tid / 4, t = r / 2, is0 = r % 2, l0 = 16 * is0 + 4 * (tid % 4);
    const int n = t / 4, j = t - 4 * n;
    const unsigned char m = (unsigned char)(1 << (4 * n + j));
    const int is = 8 * n + 2 * j + is0, shift = 2 * j;
    int us;
    if (is < 4) us = (sc[is] & 0xF) | (((sc[is + 8] >> 0) & 3) << 4);
    else if (is < 8) us = (sc[is] & 0xF) | (((sc[is + 4] >> 2) & 3) << 4);
    else if (is < 12) us = (sc[is - 8] >> 4) | (((sc[is + 0] >> 4) & 3) << 4);
    else us = (sc[is - 8] >> 4) | (((sc[is - 4] >> 6) & 3) << 4);
    const float dl = d * (float)(us - 32);
    float *y = yy + 128 * n + 32 * j;
    // ATENCAO: `q` e' `x->qs + 32*n` no dequant.cuh -- sem o `32*n` o oraculo le'
    // o super-bloco errado quando n = 1 (metade dos blocos) e reprova um staging
    // CORRETO: foi o que aconteceu na primeira corrida desta bancada (751 de 2048
    // "errados" no q3_K, com o kernel certo).
    for (int l = l0; l < l0 + 4; ++l)
      y[l] = dl * (float)((int8_t)((qs[32 * n + l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
  }
}
void cpu_get_scale_min_k4(int j, const unsigned char *q, int &dd, int &mm) {
  if (j < 4) {
    dd = q[j] & 63;
    mm = q[j + 4] & 63;
  } else {
    dd = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    mm = (q[j + 4] >> 4) | ((q[j + 0] >> 6) << 4);
  }
}
void cpu_dequant_q4k(const unsigned char *b, float *yy) {
  const unsigned char *sc = b + 4, *qs = b + 16;
  const float d = rdf16(b + 0), dmin = rdf16(b + 2);
  for (int tid = 0; tid < 32; ++tid) {
    const int il = tid / 8, ir = tid % 8, is = 2 * il;
    float *y = yy + 64 * il + 4 * ir;
    const unsigned char *q = qs + 32 * il + 4 * ir;
    int s1, m1, s2, m2;
    cpu_get_scale_min_k4(is + 0, sc, s1, m1);
    cpu_get_scale_min_k4(is + 1, sc, s2, m2);
    for (int l = 0; l < 4; ++l) {
      y[l + 0] = d * s1 * (q[l] & 0xF) - dmin * m1;
      y[l + 32] = d * s2 * (q[l] >> 4) - dmin * m2;
    }
  }
}
void cpu_dequant_q5k(const unsigned char *b, float *yy) {
  const unsigned char *sc = b + 4, *qh = b + 16, *qs = b + 48;
  const float d = rdf16(b + 0), dmin = rdf16(b + 2);
  for (int tid = 0; tid < 64; ++tid) {
    const int il = tid / 16, ir = tid % 16, is = 2 * il;
    float *y = yy + 64 * il + 2 * ir;
    const unsigned char *ql = qs + 32 * il + 2 * ir;
    const unsigned char *h = qh + 2 * ir;
    int s1, m1, s2, m2;
    cpu_get_scale_min_k4(is + 0, sc, s1, m1);
    cpu_get_scale_min_k4(is + 1, sc, s2, m2);
    unsigned char hm = (unsigned char)(1 << (2 * il));
    y[0] = d * s1 * ((ql[0] & 0xF) + ((h[0] & hm) ? 16 : 0)) - dmin * m1;
    y[1] = d * s1 * ((ql[1] & 0xF) + ((h[1] & hm) ? 16 : 0)) - dmin * m1;
    hm = (unsigned char)(hm << 1);
    y[32] = d * s2 * ((ql[0] >> 4) + ((h[0] & hm) ? 16 : 0)) - dmin * m2;
    y[33] = d * s2 * ((ql[1] >> 4) + ((h[1] & hm) ? 16 : 0)) - dmin * m2;
  }
}
void cpu_dequant_q6k(const unsigned char *b, float *yy) {
  const unsigned char *ql = b, *qh = b + 128;
  const int8_t *sc = (const int8_t *)(const void *)(b + 192);
  const float d = rdf16(b + 208);
  for (int tid = 0; tid < 64; ++tid) {
    const int ip = tid / 32, il = tid - 32 * ip, is = 8 * ip + il / 16;
    float *y = yy + 128 * ip + il;
    const unsigned char *q = ql + 64 * ip + il;
    const unsigned char h = qh[32 * ip + il];
    const int8_t *s = sc + is;
    y[0] = d * s[0] * (float)((int8_t)((q[0] & 0xF) | (((h >> 0) & 3) << 4)) - 32);
    y[32] = d * s[2] * (float)((int8_t)((q[32] & 0xF) | (((h >> 2) & 3) << 4)) - 32);
    y[64] = d * s[4] * (float)((int8_t)((q[0] >> 4) | (((h >> 4) & 3) << 4)) - 32);
    y[96] = d * s[6] * (float)((int8_t)((q[32] >> 4) | (((h >> 6) & 3) << 4)) - 32);
  }
}

// ---- staging (a algebra do gemm.cuh) --------------------------------------
void cpu_stage_q2k(const unsigned char *b, const int sub, KqCpu &o) {
  const unsigned char *sc = b, *qs = b + 16;
  o.dw = rdf16(b + 80);
  o.dmn = rdf16(b + 82);
  const int g0 = sc[2 * sub], g1 = sc[2 * sub + 1];
  o.s0 = (g0 & 0xF) | ((g0 >> 4) << 16);
  o.s1 = (g1 & 0xF) | ((g1 >> 4) << 16);
  for (int e = 0; e < 32; ++e) o.q[e] = (qs[32 * (sub >> 2) + e] >> (2 * (sub & 3))) & 3;
}
void cpu_stage_q3k(const unsigned char *b, const int sub, KqCpu &o) {
  const unsigned char *hm = b, *qs = b + 32, *sc = b + 96;
  o.dw = rdf16(b + 108);
  o.dmn = 0.0f;
  const int is = 2 * sub, quad = is >> 2, sh2 = 2 * quad;
  const unsigned lo = rd16(sc + (is < 8 ? is : is - 8));
  const unsigned hi = rd16(sc + 8 + (is & 3));
  const int l0 = (is < 8) ? (int)(lo & 0x0Fu) : (int)((lo >> 4) & 0x0Fu);
  const int l1 = (is < 8) ? (int)((lo >> 8) & 0x0Fu) : (int)((lo >> 12) & 0x0Fu);
  o.s0 = (l0 | (((int)(hi >> sh2) & 3) << 4)) - 32;
  o.s1 = (l1 | (((int)(hi >> (sh2 + 8)) & 3) << 4)) - 32;
  for (int e = 0; e < 32; ++e) {
    const int q2 = (qs[32 * (sub >> 2) + e] >> (2 * (sub & 3))) & 3;
    o.q[e] = q2 - ((hm[e] >> sub) & 1 ? 0 : 4);
  }
}
void cpu_stage_q45k(const int is5, const unsigned char *b, const int sub, KqCpu &o) {
  const unsigned char *sc = b + 4;
  const unsigned char *qs = b + (is5 ? 48 : 16);
  const unsigned char *qh = is5 ? (b + 16) : nullptr;
  o.dw = rdf16(b + 0);
  o.dmn = rdf16(b + 2);
  int s6, m6;
  cpu_get_scale_min_k4(sub, sc, s6, m6);
  o.s0 = s6 | (m6 << 16);
  o.s1 = 0;
  for (int e = 0; e < 32; ++e) {
    int v = (qs[32 * (sub >> 1) + e] >> (4 * (sub & 1))) & 0xF;
    if (is5) v |= ((qh[e] >> sub) & 1) << 4;
    o.q[e] = v;
  }
}
void cpu_stage_q6k(const unsigned char *b, const int sub, KqCpu &o) {
  const unsigned char *ql = b, *qh = b + 128;
  const int8_t *sc = (const int8_t *)(const void *)(b + 192);
  const int n = sub >> 2, q = sub & 1, i2 = (sub >> 1) & 1;
  o.dw = rdf16(b + 208);
  o.dmn = 0.0f;
  const int g = 8 * n + 2 * q + 4 * i2;
  o.s0 = sc[g];
  o.s1 = sc[g + 1];
  for (int e = 0; e < 32; ++e) {
    const int v = ((ql[64 * n + 32 * q + e] >> (4 * i2)) & 0xF) |
                  (((qh[32 * n + e] >> (2 * (sub & 3))) & 3) << 4);
    o.q[e] = v - 32;
  }
}
// desempacota o record (o mesmo `scm()` do gemm.cuh) -> (escala, minimo) do grupo
void cpu_scm(const int dt, const KqCpu &s, int grp, int &sc, int &mm) {
  switch (dt) {
    case 2:
      sc = (grp ? s.s1 : s.s0) & 0xFFFF;
      mm = ((grp ? s.s1 : s.s0) >> 16) & 0xFFFF;
      break;
    case 3: sc = grp ? s.s1 : s.s0; mm = 0; break;
    case 4:
    case 5: sc = s.s0 & 0xFFFF; mm = (s.s0 >> 16) & 0xFFFF; break;
    case 6: sc = grp ? s.s1 : s.s0; mm = 0; break;
    default: sc = 0; mm = 0; break;
  }
}
// oraculo do GEMM na MESMA formulacao do kernel (inteiro exato + fma na ordem
// do motor): o gate aqui e' BIT-EXATIDAO, porque e' a mesma sequencia.
float cpu_oracle_kq(const int dt, const unsigned char *wraw, const std::int64_t bpr,
                    const block_q8_1 *q8, const int m, const int n, const std::int64_t K,
                    const std::size_t bb) {
  const unsigned char *row = wraw + (std::size_t)n * (std::size_t)bpr * bb;
  const block_q8_1 *qrow = q8 + (std::size_t)m * (std::size_t)(K / kQ8);
  const bool two = (dt == 2 || dt == 3 || dt == 6);
  const bool has_min = (dt == 2 || dt == 4 || dt == 5);
  float acc = 0.0f;
  for (std::int64_t ib = 0; ib < bpr; ++ib) {
    const unsigned char *blk = row + (std::size_t)ib * bb;
    for (int sub = 0; sub < 8; ++sub) {
      KqCpu s{};
      switch (dt) {
        case 2: cpu_stage_q2k(blk, sub, s); break;
        case 3: cpu_stage_q3k(blk, sub, s); break;
        case 4: cpu_stage_q45k(0, blk, sub, s); break;
        case 5: cpu_stage_q45k(1, blk, sub, s); break;
        default: cpu_stage_q6k(blk, sub, s); break;
      }
      const block_q8_1 *qa = qrow + ib * 8 + sub;
      const float da = rdna4::fp16_to_float((uint16_t)(qa->ds & 0xFFFFu));
      int I0 = 0, I1 = 0, S0 = 0, S1 = 0;
      for (int e = 0; e < 32; ++e) {
        const int a = (int)qa->qs[e];
        if (two && e >= 16) I1 += s.q[e] * a;
        else I0 += s.q[e] * a;
        if (has_min) {
          if (two && e >= 16) S1 += a;
          else S0 += a;
        }
      }
      int sc0, m0, sc1, m1;
      cpu_scm(dt, s, 0, sc0, m0);
      cpu_scm(dt, s, 1, sc1, m1);
      const int X = two ? (I0 * sc0 + I1 * sc1) : (I0 * sc0);
      acc = std::fma(s.dw, da * (float)X, acc);  // std::fma(float,float,float) = fmaf
      if (has_min) {
        const int Y = two ? (S0 * m0 + S1 * m1) : (S0 * m0);
        acc = std::fma(-s.dmn, da * (float)Y, acc);
      }
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
void ref_q2k(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TQ2K>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_q3k(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TQ3K>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_q4k(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TQ4K>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_q5k(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TQ5K>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_q6k(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  ref_engine_kernel<rdna4::TQ6K>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}
void ref_iq2xxs(const void *w, const block_q8_1 *a, float *c, int M, int N, int K) {
  // Gate de producao: TIQ2XXS_S = `_perm2` (o que o matvec em lote usa).
  ref_engine_kernel<rdna4::TIQ2XXS_S>
      <<<(unsigned)(((std::int64_t)M * N + 127) / 128), 128>>>(w, a, c, M, N, K);
}

// ---- k-quants (D2/kq): ref do motor + oraculos de CPU ----------------------
constexpr int kQ2K = 2, kQ3K = 3, kQ4K = 4, kQ5K = 5, kQ6K = 6;

void deq_q2k(const unsigned char *b, float *y) { cpu_dequant_q2k(b, y); }
void deq_q3k(const unsigned char *b, float *y) { cpu_dequant_q3k(b, y); }
void deq_q4k(const unsigned char *b, float *y) { cpu_dequant_q4k(b, y); }
void deq_q5k(const unsigned char *b, float *y) { cpu_dequant_q5k(b, y); }
void deq_q6k(const unsigned char *b, float *y) { cpu_dequant_q6k(b, y); }

const TypeDesc *type_of(const int dt) {
  static const TypeDesc kTypes[] = {
      {"iq3_s", kIq3S, ref_iq3s, sizeof(rdna4::block_iq3_s)},
      {"iq3_xxs", kIq3XXS, ref_iq3xxs, sizeof(rdna4::block_iq3_xxs)},
      {"iq4_xs", kIq4XS, ref_iq4xs, sizeof(rdna4::block_iq4_xs)},
      {"q2_K", kQ2K, ref_q2k, sizeof(rdna4::block_q2_K)},
      {"q3_K", kQ3K, ref_q3k, sizeof(rdna4::block_q3_K)},
      {"q4_K", kQ4K, ref_q4k, sizeof(rdna4::block_q4_K)},
      {"q5_K", kQ5K, ref_q5k, sizeof(rdna4::block_q5_K)},
      {"q6_K", kQ6K, ref_q6k, sizeof(rdna4::block_q6_K)},
      {"iq2_xxs", kIq2XXS, ref_iq2xxs, sizeof(rdna4::block_iq2_xxs)},
  };
  for (const TypeDesc &t : kTypes)
    if (t.dt == dt) return &t;
  return nullptr;
}

// Dequant canonico do tipo (nullptr se nao ha' oraculo de CPU).
void (*dequant_of(const int dt))(const unsigned char *, float *) {
  switch (dt) {
    case kQ2K: return deq_q2k;
    case kQ3K: return deq_q3k;
    case kQ4K: return deq_q4k;
    case kQ5K: return deq_q5k;
    case kQ6K: return deq_q6k;
    default: return nullptr;
  }
}

// Dump do staging de um k-quant: 8 palavras int8 (a ordem de k) + o record.
template <class Traits>
__global__ void dump_stage_kq_kernel(const char *__restrict__ W, int *__restrict__ out,
                                     const int nrow, const int bpr, const int nsub) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= nsub) return;
  const int s = t % 8;
  const char *blk = W + ((std::int64_t)nrow * bpr + (t / 8)) * Traits::block_bytes;
  const typename Traits::Pf pf = Traits::load(blk, s);
  int tmp[8];
  rdna4::KqScal k;
  Traits::store(pf, s, tmp, k);
  int *o = out + (std::size_t)t * 12;
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = tmp[i];
  o[8] = __float_as_int(k.dw);
  o[9] = __float_as_int(k.dmn);
  o[10] = k.s0;
  o[11] = k.s1;
}

template <class Traits>
void dump_stage_kq(const char *dW, int *dOut, const int nrow, const int bpr, const int nsub) {
  dump_stage_kq_kernel<Traits><<<(nsub + 63) / 64, 64>>>(dW, dOut, nrow, bpr, nsub);
}

void dump_stage_kq_dt(const int dt, const char *dW, int *dOut, const int nrow, const int bpr,
                      const int nsub) {
  switch (dt) {
    case kQ2K: dump_stage_kq<rdna4::SolverQ2K>(dW, dOut, nrow, bpr, nsub); break;
    case kQ3K: dump_stage_kq<rdna4::SolverQ3K>(dW, dOut, nrow, bpr, nsub); break;
    case kQ4K: dump_stage_kq<rdna4::SolverQ4K>(dW, dOut, nrow, bpr, nsub); break;
    case kQ5K: dump_stage_kq<rdna4::SolverQ5K>(dW, dOut, nrow, bpr, nsub); break;
    default: dump_stage_kq<rdna4::SolverQ6K>(dW, dOut, nrow, bpr, nsub); break;
  }
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
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB,
          bool RAW = false, bool STAGE_ONLY = false>
bool attrs_of(hipFuncAttributes &attr, int &lds) {
  constexpr int NKB = BK / 32;
  constexpr int WSB = BK + 16;
  constexpr int RSTR = NKB * ((TR::raw_words + 3) & ~3);
  if (RAW) {
    lds = (DBUF ? 2 : 1) * (BN * RSTR * (int)sizeof(int));
  } else {
    lds = (DBUF ? 2 : 1) * (BN * WSB + NKB * BN * 8);
  }
  auto *fn = &rdna4::gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB, RAW, STAGE_ONLY>;
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
#define ATR(Traits, BM, BN, BK, RM, RN, DB, WH, MB, RW, SO) \
  ok = attrs_of<Traits, BM, BN, BK, RM, RN, DB, WH, MB, RW, SO>(a, lds)
    switch (cfg) {
      case 1: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, true, true, 1); break;
      case 2: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, false, true, 1); break;
      case 3: AT(rdna4::SolverIq3S, 128, 128, 64, 8, 8, true, true, 2); break;
      case 5: AT(rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1); break;
      case 6: AT(rdna4::SolverIq3S, 128, 64, 64, 8, 4, true, true, 1); break;
      case 7: AT(rdna4::SolverIq3S, 64, 64, 64, 4, 4, true, true, 1); break;
      case 8: AT(rdna4::SolverIq3S, 128, 64, 64, 4, 4, true, true, 1); break;
      // 9/10 = staging cru + decodificacao no consumo ; 11/12 = so' staging
      case 9:
#define ATC(Traits) ATR(Traits, 64, 128, 64, 4, 8, true, true, 1, true, false)
        switch (dt) {
          case kIq3S: ATC(rdna4::SolverIq3S); break;
          case kIq3XXS: ATC(rdna4::SolverIq3XXS); break;
          case kIq4XS: ATC(rdna4::SolverIq4XS); break;
          default: ok = false; break;
        }
#undef ATC
        break;
      case 10:
#define ATC10(Traits) ATR(Traits, 16, 128, 64, 1, 8, true, true, 1, true, false)
        switch (dt) {
          case kIq3S: ATC10(rdna4::SolverIq3S); break;
          case kIq3XXS: ATC10(rdna4::SolverIq3XXS); break;
          case kIq4XS: ATC10(rdna4::SolverIq4XS); break;
          default: ok = false; break;
        }
#undef ATC10
        break;
      case 11:
        if (dt == kIq3S) {
          ATR(rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1, false, true);
        } else {
          ok = false;
        }
        break;
      case 12:
        if (dt == kIq3S) {
          ATR(rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1, true, true);
        } else {
          ok = false;
        }
        break;
      default: ok = false; break;
    }
#undef ATR
#undef AT
  }
  if (!ok) {
    std::printf("  %-30s attrs indisponiveis\n", tag);
    return;
  }
  std::printf("  %-30s VGPR=%3d  LDS=%5d B (pedido %5d B, attr %5zu B)  spill/lane=%d B\n", tag,
              a.numRegs, lds, lds, (size_t)a.sharedSizeBytes, (int)a.localSizeBytes);
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

// ---------------------------------------------------------------------------
// R1 PROTOTYPE (BENCH-ONLY): staging coalescido do iq3_s via W com campos
// intercalados / bloqueado por janela (brief §8 R1). NAO TOCA gemm.cuh/loader/
// engine: o repack e' host-side nesta bancada e o load path prototipo vive SÓ
// aqui. Se FAIL (<1,3x ou sniff de bits), fica como negativo documentado.
// Layout (repack offline, uma vez): 16 B por (linha, sub-bloco de 32 pesos):
//   bytes 0-7: qs[8] | 8-11: signs[4] | 12: qh | 13-14: d (bits fp16) | 15: scales.
//   = 8+4+1+2+1 = 16 B = 1x b128 alinhado. Por janela BK=64: 2x16 B = 32 B contiguos.
// Ordem window-major (bloqueada por janela): idx = (w*N + n)*2 + kb,
//   w = k/64 (0..K/64-1), kb = (k/32)%2. Para w fixo, linhas consecutivas sao
//   contiguas em 32 B: a warp (16 linhas) le 512 B contiguos = 4 linhas de 128 B
//   a ~100% de utilizacao (contra ~23% do shipping, brief §8.2.2).
// Load path: 1x b128 sequencial por tarefa (default cache policy, sem TH_NT) +
//   o MESMO SolverIq3S::store/consume do shipping -> bit-identico por construcao.
// Grade/memoria/computo IDENTICOS ao shipping BM64 BN128 BK64 RM4 RN8 DBUF
// (producao em M>=64): a razao mede SÓ o staging coalescido.
// Caminhos proibidos NAO tentados aqui (medidos mortos): raw-staging transplant,
// familia f16-dot2, sem-Dbuf em BM64, BK128, BN136, TH_NT.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256, 1) gemm_iq3s_coal_kernel(
    const int4 *__restrict__ Wp, const block_q8_1 *__restrict__ A, float *__restrict__ C,
    const int M, const int N, const int K, const int act_stride) {
  constexpr int WSB = 80;
  __shared__ __align__(16) unsigned char s_w[2 * 128 * 80];
  __shared__ __align__(16) int2 s_dwsc[2 * 2 * 128];
  const int tid = threadIdx.x;
  const int tr = tid / 16;
  const int tc = tid % 16;
  const int m0 = blockIdx.y * 64;
  const int n0 = blockIdx.x * 128;
  const int prow = tid / 2;
  const int pkb = tid % 2;
  const int nloc = tc;
  int mrow[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int m = m0 + tr + r * 16;
    mrow[r] = (m < M) ? m : (M - 1);
  }
  float F[4][8];
#pragma unroll
  for (int r = 0; r < 4; ++r)
#pragma unroll
    for (int c = 0; c < 8; ++c) F[r][c] = 0.0f;
  rdna4::SolverIq3S::Pf pf;
  // prologo: load(k0=0) + store(buf0)
  {
    int n = n0 + prow;
    if (n >= N) n = N - 1;
    const int kk = 0 + 32 * pkb;
    const int w = 0;
    const int idx = (w * N + n) * 2 + pkb;
    const int4 rec = Wp[idx];
    const unsigned w3 = (unsigned)rec.w;
    pf.qs0 = rec.x;
    pf.qs1 = rec.y;
    pf.sg = (unsigned)rec.z;
    pf.qh = w3 & 0xFFu;
    pf.dsc = w3 >> 8;
    float dw;
    int scf;
    rdna4::SolverIq3S::store(pf, (kk >> 5) & 7,
                             (int *)(void *)(s_w + prow * WSB + pkb * 32), &dw, &scf);
    s_dwsc[pkb * 128 + prow] = make_int2(__float_as_int(dw), scf);
  }
  __syncthreads();
  for (int k0 = 0, cur = 0; k0 < K; k0 += 64, cur ^= 1) {
    const int cb = cur;
    if (k0 + 64 < K) {
      int n = n0 + prow;
      if (n >= N) n = N - 1;
      const int kk = (k0 + 64) + 32 * pkb;
      const int w = (k0 + 64) >> 6;
      const int idx = (w * N + n) * 2 + pkb;
      const int4 rec = Wp[idx];
      const unsigned w3 = (unsigned)rec.w;
      pf.qs0 = rec.x;
      pf.qs1 = rec.y;
      pf.sg = (unsigned)rec.z;
      pf.qh = w3 & 0xFFu;
      pf.dsc = w3 >> 8;
    }
    // conta: copia verbatim do caminho WHOLD/cadeia-unica do shipping
    // (BM64 BN128 BK64 RM4 RN8), mesmos dp4a e mesma correcao.
#pragma unroll 1
    for (int kb = 0; kb < 2; ++kb) {
      const int ka = (k0 >> 5) + kb;
      int I[4][8];
#pragma unroll
      for (int r = 0; r < 4; ++r)
#pragma unroll
        for (int c = 0; c < 8; ++c) I[r][c] = 0;
      int ww[8][8];
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int *wp =
            (const int *)(const void *)(s_w + cb * (128 * WSB) + (nloc + c * 16) * WSB + kb * 32);
#pragma unroll
        for (int g = 0; g < 8; ++g) ww[c][g] = wp[g];
      }
#pragma unroll
      for (int g = 0; g < 8; ++g) {
        int ar[4];
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
          ar[r] = ((const int *)(const void *)q->qs)[g];
        }
#pragma unroll
        for (int r = 0; r < 4; ++r)
#pragma unroll
          for (int c = 0; c < 8; ++c) I[r][c] = rdna4::ggml_cuda_dp4a(ww[c][g], ar[r], I[r][c]);
      }
      float da[4];
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
        da[r] = rdna4::fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
      }
      const int2 *wsc = s_dwsc + cb * (2 * 128) + kb * 128;
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int2 ws = wsc[nloc + c * 16];
        const float dw = __int_as_float(ws.x);
        const int scf = ws.y;
#pragma unroll
        for (int r = 0; r < 4; ++r) {
          const float d = dw * da[r];
          F[r][c] = __fmaf_rn(d, rdna4::SolverIq3S::corr(I[r][c], scf), F[r][c]);
        }
      }
    }
    if (k0 + 64 < K) {
      float dw;
      int scf;
      const int kk2 = (k0 + 64) + 32 * pkb;
      rdna4::SolverIq3S::store(pf, (kk2 >> 5) & 7,
                               (int *)(void *)(s_w + (cb ^ 1) * (128 * WSB) + prow * WSB + pkb * 32),
                               &dw, &scf);
      s_dwsc[(cb ^ 1) * (2 * 128) + pkb * 128 + prow] = make_int2(__float_as_int(dw), scf);
    }
    __syncthreads();
  }
  // epilogo: C[m][n] token-major, igual ao shipping.
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const int m = m0 + tr + r * 16;
    if (m < M) {
#pragma unroll
      for (int c = 0; c < 8; ++c) {
        const int n = n0 + nloc + c * 16;
        if (n < N) C[(std::int64_t)m * N + n] = F[r][c];
      }
    }
  }
}

// Repack host-side (offline, uma vez): iq3_s cru -> window-major 16 B/sub.
// w_raw: N linhas x bpr super-blocos x 110 B. packed: N*NSUB int4 (16 B cada).
void repack_iq3s_coal(const std::vector<std::uint8_t> &w_raw, std::vector<int> &packed, const int N,
                      const int K) {
  const int bpr = K / 256;
  const int nsub = K / 32;
  packed.resize((std::size_t)N * (std::size_t)nsub * 4);
  for (int n = 0; n < N; ++n) {
    for (int ib = 0; ib < bpr; ++ib) {
      const unsigned char *blk = w_raw.data() + ((std::size_t)n * bpr + ib) * 110;
      const unsigned d = (unsigned)blk[0] | ((unsigned)blk[1] << 8);
      const unsigned char *qs = blk + 2;
      const unsigned char *qh = blk + 2 + 64;
      const unsigned char *sg = blk + 2 + 72;
      const unsigned char *sc = blk + 2 + 104;
      for (int sub = 0; sub < 8; ++sub) {
        int qs0 = 0, qs1 = 0, sgw = 0;
        std::memcpy(&qs0, qs + 8 * sub, 4);
        std::memcpy(&qs1, qs + 8 * sub + 4, 4);
        std::memcpy(&sgw, sg + 4 * sub, 4);
        const unsigned w3 = (unsigned)qh[sub] | (d << 8) | ((unsigned)sc[sub >> 1] << 24);
        const int w = ib * 4 + (sub >> 1);
        const int kb = sub & 1;
        const std::size_t idx = ((std::size_t)w * (std::size_t)N + (std::size_t)n) * 2 + kb;
        packed[idx * 4 + 0] = qs0;
        packed[idx * 4 + 1] = qs1;
        packed[idx * 4 + 2] = sgw;
        packed[idx * 4 + 3] = (int)w3;
      }
    }
  }
}

bool launch_iq3s_coal(const int4 *dWp, const block_q8_1 *dA, float *dC, const int N, const int K,
                      const int act_stride, const int M) {
  const dim3 grid((unsigned)((N + 127) / 128), (unsigned)((M + 63) / 64), 1u);
  gemm_iq3s_coal_kernel<<<grid, 256, 0, nullptr>>>(dWp, dA, dC, M, N, (int)K, (int)act_stride);
  return hipGetLastError() == hipSuccess;
}

// ---------------------------------------------------------------------------
// H9.1 PROTOTYPE (BENCH-ONLY): TH cache-policy variants on the GEMM staging
// path (docs/plano-hipfire-gfx1201.md H9 step 1). Mirror of hipfire's
// `kernels/src/gfx12_weight_cache_policy.inc` restricted to staging streams.
//
// Triage (o que continua vivo): R2 TH_NT esta' MORTO nos loads
// latency-critical do GEMV (tombstone em vecdotq.cuh:126-132, -11..-14%);
// R10 MUBUF-generalizado morto por custo/beneficio; s_prefetch_data morto
// (M2). Vivo: TH/RT/HT nos streams BULK nao-latency-critical. O staging do
// GEMM e' o candidato: pf_load() emite os loads uma janela-K inteira de conta
// acima do uso (load->use de centenas de instrucoes) e o stream e'
// DRAM-bound (76,6 MB em 0,134 ms em M=128 = 572 GB/s, comentario em
// gemm.cuh). Ativacao (q8_1, usada no dp4a logo apos o load), LUT e o GEMV
// (load -> perm/tabela -> dp4a em dezenas de instrucoes) ficam cached.
//
// Desenho: SolverIq3S_TH<AUX> reusa gemm_i8_kernel<BM64/BN128/BK64/RM4/RN8/
// DBUF/WHOLD> VERBATIM -- so' TR::load muda (FLAT -> MUBUF raw_buffer_load com
// aux literal). store/corr sao os do SolverIq3S, entao a saida tem de ser
// BIT-IDENTICA a' producao (o gate). aux espelha o hipfire exato: 16=RT+DEV
// (controle MUBUF vs staging FLAT de producao), 18=HT+DEV, 20=NT_RT+DEV,
// 22=NT_HT+DEV. Codegen: `th:`/`scope:` no buffer_load_* (sonda /tmp/thprobe2:
// 16->`scope:SCOPE_DEV`, 18->`th:TH_LOAD_HT scope:SCOPE_DEV`,
// 20->`th:TH_LOAD_NT_RT ...`, 22->`th:TH_LOAD_NT_HT ...`).
// ---------------------------------------------------------------------------
__constant__ const char *g_th_base;  // base do W (offset = blk - base)

template <int AUX>
struct SolverIq3S_TH {
  static constexpr int qk = rdna4::SolverIq3S::qk;
  static constexpr int block_bytes = (int)sizeof(rdna4::block_iq3_s);
  static constexpr bool is_kq = false;
  static constexpr bool dual_scale = false;
  using Pf = rdna4::SolverIq3S::Pf;
  static __device__ __forceinline__ int th_ld32(const unsigned voff) {
    auto rsrc = __builtin_amdgcn_make_buffer_rsrc(const_cast<char *>(g_th_base), 0,
                                                 0xffffffffu, 0x31004000u);
    return __builtin_amdgcn_raw_buffer_load_b32(rsrc, voff, 0u, AUX);
  }
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    Pf p;
    // block_iq3_s (quants.h): d@0, qs@2, qh@66, signs@74, scales@106.
    const unsigned sb = (unsigned)(blk - g_th_base);
    const unsigned o_qs = sb + 2u + (unsigned)sub * 8u;
    p.qs0 = th_ld32(o_qs);
    p.qs1 = th_ld32(o_qs + 4u);
    p.sg = (unsigned)th_ld32(sb + 74u + (unsigned)sub * 4u);
    const unsigned wqh = (unsigned)th_ld32(sb + 66u + ((unsigned)sub & ~3u));
    p.qh = (wqh >> (8 * (sub & 3))) & 0xFFu;
    const unsigned w0 = (unsigned)th_ld32(sb);  // d nos 16 bits baixos
    const unsigned sub2 = (unsigned)sub >> 1;
    const unsigned wsc = (unsigned)th_ld32(sb + 106u + (sub2 & ~3u));
    p.dsc = (w0 & 0xFFFFu) | (((wsc >> (8 * (sub2 & 3))) & 0xFFu) << 16);
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, float *dw,
                                              int *scf) {
    rdna4::SolverIq3S::store(p, sub, dst, dw, scf);
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return rdna4::SolverIq3S::corr(sumi, scf);
  }
  static constexpr int raw_words = 4;
  static __device__ __forceinline__ void raw_pack(const Pf &p, int *rec) {
    rdna4::SolverIq3S::raw_pack(p, rec);
  }
  static __device__ __forceinline__ void decode_raw(const int *rec, const int sub, int *dst,
                                                   float *dw, int *scf) {
    rdna4::SolverIq3S::decode_raw(rec, sub, dst, dw, scf);
  }
};

template <int AUX>
bool launch_th(const void *dW, const block_q8_1 *dA, float *dC, const std::int64_t N,
               const std::int64_t K, const std::int64_t ast, const int M) {
  const char *base = (const char *)dW;
  if (hipMemcpyToSymbol(HIP_SYMBOL(g_th_base), &base, sizeof(base)) != hipSuccess) return false;
  return rdna4::gemm_launch_t<SolverIq3S_TH<AUX>, 64, 128, 64, 4, 8, true, true, 1>(
      dW, dA, dC, N, K, ast, M, nullptr);
}

// ---------------------------------------------------------------------------
// IQ2_XXS PROTOTYPE (BENCH-ONLY -- nao entra em gemm.cuh nem no despacho).
// Segue o precedente SolverIq2XS (gemm.cuh:514): mesma estrutura de cadeia
// unica (is_kq=false, um fator inteiro por bloco de 32, correcao inteira +
// fma na ordem do motor), outra grade/tabela: iq2xxs_grid (uint64 x 256) +
// sinais via unpack_ksigns por grupo de 8 pesos + escala unica no topo.
//
// Transcricao termo a termo do `vec_dot_iq2_xxs_q8_1` (vecdotq.cuh:565-595):
//   sub-bloco s (32 pesos) = qs[4*s .. 4*s+4) (4 uint16 = 8 B):
//     q0 = bytes 0..3 -> 4 indices de grid (8 pesos cada)
//     aux32 = bytes 4..7 -> 4 campos de sinal de 7 bits + escala (bits 28..31)
//   grupo il (8 pesos) = grid aux8[il] (8 magnitudes): x = bytes 0..3,
//     y = bytes 4..7; sv = aux32>>(7*il) com paridade no bit 7, nibble baixo
//     assina x, nibble alto assina y (mascaras 0x08040201 / 0x80402010).
//   correcao: sumi = sumi*ls/8 com ls = (aux32>>27)|1 = scale*2+1 (o bit 27,
//     topo do 4o campo de sinal, e' absorvido pelo |1: 2*scale e' par).
// O gate de producao (TIQ2XXS_S) e' o `_perm2`, que recomputa os mesmos
// inteiros via 2*pos-all -- exato por linearidade inteira (mesmo argumento do
// cabecalho do iq2_s em gemm.cuh:437 para _perm/_perm2). dp4a inteiro e'
// exato e a ordem de k e' a mesma, entao o prototipo tem de ser BIT-EXATO
// contra o motor (0/4096, max ulp 0), como os tipos IQ de cadeia unica.
// ---------------------------------------------------------------------------
struct BenchSolverIq2XXS {
  static constexpr int qk = 256;
  static constexpr int block_bytes = (int)sizeof(rdna4::block_iq2_xxs);
  static constexpr bool is_kq = false;
  static constexpr bool dual_scale = false;
  struct Pf {
    int q0, q1;  // qs[4*sub .. 4*sub+4) como 2 ints (4 uint16)
    unsigned d;  // d (fp16) do super-bloco
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const rdna4::block_iq2_xxs *b = (const rdna4::block_iq2_xxs *)blk;
    Pf p;
    const int *q = (const int *)(const void *)(b->qs + 4 * sub);
    p.q0 = q[0];
    p.q1 = q[1];
    p.d = b->d;
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int /*sub*/, int *dst, float *dw,
                                               int *scf) {
    const unsigned char *aux8 = (const unsigned char *)&p.q0;
    const unsigned aux32 = (unsigned)p.q1;
    *dw = rdna4::fp16_to_float((uint16_t)p.d);
    *scf = (int)((aux32 >> 27) | 1u);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
      const uint2 grid = ((const uint2 *)rdna4::iq2xxs_grid)[aux8[il]];
      const unsigned signs = rdna4::unpack_ksigns((std::uint8_t)(aux32 >> (7 * il)));
      const int signs0 = rdna4::__vcmpne4(signs & 0x08040201u, 0u);
      const int signs1 = rdna4::__vcmpne4(signs & 0x80402010u, 0u);
      dst[2 * il + 0] = rdna4::__vsub4((int)(grid.x ^ (unsigned)signs0), signs0);
      dst[2 * il + 1] = rdna4::__vsub4((int)(grid.y ^ (unsigned)signs1), signs1);
    }
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)(sumi * scf / 8);
  }
  // Sem caminho RAW (mesmo motivo do iq2_s/iq2_xs): so' para o RSTR do kernel.
  static constexpr int raw_words = 4;
};

// Lancamento do prototipo na MESMA grade da producao (RD_GEMM em gemm.cuh):
// BM64/RM4/RN8 em M>=64, BM16/RM1/RN8 abaixo.
bool launch_xxs_proto(const void *dW, const block_q8_1 *dA, float *dC, const std::int64_t N,
                      const std::int64_t K, const std::int64_t ast, const int M) {
  // Espelha o RD_GEMM (gemm.cuh): a restricao e' no K (ncols), nao no N
  // (nrows) -- o kernel prende linhas fora do tile (n = N-1) e guarda as
  // escritas com n < N, entao Nc=64 do caso pequeno funciona.
  if (K % 256 != 0 || K % 64 != 0) return false;
  if (M >= 64)
    return rdna4::gemm_launch_t<BenchSolverIq2XXS, 64, 128, 64, 4, 8, true, true, 1>(
        dW, dA, dC, N, K, ast, M, nullptr);
  return rdna4::gemm_launch_t<BenchSolverIq2XXS, 16, 128, 64, 1, 8, true, true, 1>(
      dW, dA, dC, N, K, ast, M, nullptr);
}

// Roteador bench-only: dt=7 vai ao prototipo, o resto ao despacho de producao.
// (O `--types` continua sondando o `gemm_launch` PURO -- este roteador nao e'
// usado la.)
bool bench_gemm_launch(const int dt, const void *dW, const block_q8_1 *dA, float *dC,
                       const std::int64_t N, const std::int64_t K, const std::int64_t ast,
                       const int M) {
  if (dt == kIq2XXS) return launch_xxs_proto(dW, dA, dC, N, K, ast, M);
  return rdna4::gemm_launch(dt, dW, dA, dC, N, K, ast, M, nullptr);
}

void print_xxs_proto_attrs() {
  for (int bm = 64; bm >= 16; bm /= 4) {
    hipFuncAttributes a{};
    int lds = 0;
    const bool ok = (bm == 64)
                        ? rdna4::gemm_attrs_t<BenchSolverIq2XXS, 64, 128, 64, 4, 8, true, true, 1>(
                              a, lds)
                        : rdna4::gemm_attrs_t<BenchSolverIq2XXS, 16, 128, 64, 1, 8, true, true, 1>(
                              a, lds);
    char tag[64];
    std::snprintf(tag, sizeof(tag), "prototipo iq2_xxs (BM%d BN128 BK64)", bm);
    if (!ok) {
      std::printf("  %-30s attrs indisponiveis\n", tag);
      continue;
    }
    std::printf("  %-30s VGPR=%3d  LDS=%5d B (pedido %5d B, attr %5zu B)  spill/lane=%d B\n",
                tag, a.numRegs, lds, lds, (size_t)a.sharedSizeBytes, (int)a.localSizeBytes);
  }
}

// Copia a grade iq2xxs do DEVICE para o oraculo de CPU (mesma tabela do kernel).
__global__ void copy_xxs_grid_kernel(unsigned long long *__restrict__ out) {
  for (int i = (int)threadIdx.x; i < 256; i += (int)blockDim.x)
    out[i] = rdna4::iq2xxs_grid[i];
}

// Deposito do staging do prototipo isolado (mesma funcao que o kernel usa).
__global__ void dump_stage_xxs_kernel(const char *__restrict__ W, int *__restrict__ out,
                                      const int nrow, const int bpr, const int nsub) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= nsub) return;
  const int s = t % 8;
  const char *blk =
      W + ((std::int64_t)nrow * bpr + (t / 8)) * (std::int64_t)BenchSolverIq2XXS::block_bytes;
  const BenchSolverIq2XXS::Pf pf = BenchSolverIq2XXS::load(blk, s);
  int tmp[8];
  float dw;
  int scf;
  BenchSolverIq2XXS::store(pf, s, tmp, &dw, &scf);
  int *o = out + (std::size_t)t * 10;
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = tmp[i];
  o[8] = __float_as_int(dw);
  o[9] = scf;
}

// Verificacao (a3): o staging int8 do prototipo contra a algebra do
// `vec_dot_iq2_xxs_q8_1` em CPU sobre os bytes crus + a grade lida do device.
// Retorna true se 0/2048 pesos e 0/64 (dw, scf).
bool check_stage_xxs(const std::vector<std::uint8_t> &w_raw, const void *dW, const std::int64_t K) {
  const int nsub = 64;  // 8 super-blocos = 2048 pesos da linha 0
  int *dStage = nullptr;
  unsigned long long *dGrid = nullptr;
  if (hipMalloc(&dStage, (std::size_t)nsub * 10 * sizeof(int)) != hipSuccess ||
      hipMalloc(&dGrid, 256 * sizeof(unsigned long long)) != hipSuccess) {
    std::printf("  hipMalloc(stage/grid xxs) falhou\n");
    if (dStage) (void)hipFree(dStage);
    if (dGrid) (void)hipFree(dGrid);
    return false;
  }
  copy_xxs_grid_kernel<<<1, 128>>>(dGrid);
  if (!launch_ok("copy_xxs_grid")) {
    (void)hipFree(dStage);
    (void)hipFree(dGrid);
    return false;
  }
  dump_stage_xxs_kernel<<<(nsub + 63) / 64, 64>>>((const char *)dW, dStage, 0, (int)(K / 256),
                                                 nsub);
  if (!launch_ok("dump_stage_xxs")) {
    (void)hipFree(dStage);
    (void)hipFree(dGrid);
    return false;
  }
  (void)hipDeviceSynchronize();
  std::vector<int> staged((std::size_t)nsub * 10);
  std::vector<unsigned long long> hgrid(256, 0);
  (void)hipMemcpy(staged.data(), dStage, staged.size() * sizeof(int), hipMemcpyDeviceToHost);
  (void)hipMemcpy(hgrid.data(), dGrid, hgrid.size() * sizeof(unsigned long long),
                  hipMemcpyDeviceToHost);
  (void)hipFree(dStage);
  (void)hipFree(dGrid);
  int bad_w = 0, bad_sc = 0, first_bad = -1;
  for (int t = 0; t < nsub; ++t) {
    const int s = t % 8;
    const unsigned char *blk = w_raw.data() + (std::size_t)(t / 8) * 66;
    const unsigned char *qb = blk + 2 + 8 * s;
    unsigned aux32 = 0;
    std::memcpy(&aux32, qb + 4, 4);
    for (int il = 0; il < 4; ++il) {
      const unsigned long long g = hgrid[qb[il]];
      unsigned sv = (aux32 >> (7 * il)) & 0x7Fu;
      sv ^= (unsigned)(__builtin_popcount(sv) & 1u) << 7;
      for (int j = 0; j < 4; ++j) {
        const int gb_x = (int)((g >> (8 * j)) & 0xFFu);
        const int gb_y = (int)((g >> (8 * (j + 4))) & 0xFFu);
        const int want_x = ((sv >> j) & 1u) ? -gb_x : gb_x;
        const int want_y = ((sv >> (j + 4)) & 1u) ? -gb_y : gb_y;
        const int got_x = (staged[(std::size_t)t * 10 + 2 * il] >> (8 * j)) & 0xFF;
        const int got_y = (staged[(std::size_t)t * 10 + 2 * il + 1] >> (8 * j)) & 0xFF;
        if (got_x != (want_x & 0xFF) || got_y != (want_y & 0xFF)) {
          ++bad_w;
          if (first_bad < 0) first_bad = t * 32 + il * 8 + j;
        }
      }
    }
    const float want_dw = rdna4::fp16_to_float((uint16_t)(blk[0] | (blk[1] << 8)));
    const int want_scf = (int)((aux32 >> 27) | 1u);
    if (staged[(std::size_t)t * 10 + 8] != (int)f2u(want_dw) ||
        staged[(std::size_t)t * 10 + 9] != want_scf)
      ++bad_sc;
  }
  std::printf(
      "  staging int8 xxs: bytes diferentes %d de 2048%s ; d_w/ls errados %d de 64%s%s\n"
      "    (2048 pesos da linha 0 contra a algebra do `vec_dot_iq2_xxs_q8_1` em CPU sobre os\n"
      "     bytes crus + a grade lida do device)\n",
      bad_w, bad_w ? "" : " (ok)", bad_sc, bad_sc ? "" : " (ok)",
      first_bad >= 0 ? "  <-- PRIMEIRO BYTE ERRADO" : "");
  if (bad_w || bad_sc) {
    std::printf("  REPROVADO: o staging do prototipo nao reproduz a algebra do motor\n");
    return false;
  }
  return true;
}

// Sonda GEMV-only (tipos sem solver: q8_0, iq1_s, iq4_nl): batched-GEMV em
// M=128, sub-lote 64, warmup >= 300 ms, melhor+media de R>=10, piso de ruido.
// Reporta T-MAC/s e GB/s de PESO (M x bytes-da-matriz / tempo).
void run_gemv_only_probe(const char *name, const int dt, const void *dW, const block_q8_1 *dAQ,
                         float *dC2, const std::int64_t N, const std::int64_t K,
                         const std::int64_t ast, const std::uint64_t wbytes, const int reps) {
  const int M = 128;
  std::printf("  -- sonda GEMV-only %s (sem solver: so' matvec em lote) --\n", name);
  bool warned = false;
  auto fn = [&] { gemv_batch(dt, dW, dAQ, dC2, N, K, ast, M, 64, false); };
  fn();
  (void)hipDeviceSynchronize();
  if (!launch_ok("gemv_only (warmup)")) return;
  EvPair ev;
  double warm = 0;
  for (int i = 0; i < 20000 && warm < 300.0; ++i) warm += ev(fn);
  if (!launch_ok("gemv_only (burner)")) {
    (void)warned;
    return;
  }
  const int R = std::max(reps, 10);
  double best = 1e9, sum = 0;
  for (int r = 0; r < R; ++r) {
    const double t = ev(fn);
    best = std::min(best, t);
    sum += t;
  }
  if (!launch_ok("gemv_only (time)")) return;
  const double mean = sum / R;
  const double macs = (double)M * (double)N * (double)K;
  const double tmac = macs / (best * 1e-3) / 1e12;
  const double gbs = ((double)M * (double)wbytes) / (best * 1e-3) / 1e9;
  std::printf(
      "  M=%d N=%lld K=%lld: GEMV64 %.3f ms (melhor de %d) / %.3f ms (media) -> %.2f T-MAC/s ;\n"
      "    trafego de peso %.2f GB/s (M x %.3f MB / tempo; pico do cartao ~633)\n",
      M, (long long)N, (long long)K, best, R, mean, tmac, gbs, (double)wbytes / 1e6);
}

// Veredito iq2_xxs: A/B intercalado prototipo vs GEMV64 em cada M, com burner
// de DPM >= 300 ms, melhor+media de R>=10 e piso proto-vs-proto. A porta de
// entrada (bit-exatidao 0/4096 vs o vec_dot de producao) ja' saiu no gate (c)
// do fluxo generico, que para dt=7 usa o prototipo via bench_gemm_launch.
void run_xxs_verdict(const std::vector<int> &ms_list, const int reps, const int dt, const void *dW,
                     const block_q8_1 *dAQ, float *dC, float *dC2, const std::int64_t N,
                     const std::int64_t K, const std::int64_t ast) {
  std::printf("\n  -- VEREDITO iq2_xxs: prototipo vs GEMV em lote (A/B intercalado) --\n");
  print_xxs_proto_attrs();
  const double macs_per = (double)N * (double)K;
  for (std::size_t mi = 0; mi < ms_list.size(); ++mi) {
    const int M = ms_list[mi];
    const double macs = macs_per * (double)M;
    bool warned = false;
    auto fn_p = [&] {
      if (!launch_xxs_proto(dW, dAQ, dC, N, K, ast, M) && !warned) {
        warned = true;
        std::printf("  prototipo(M=%d) retornou false\n", M);
      }
    };
    auto fn_g = [&] { gemv_batch(dt, dW, dAQ, dC2, N, K, ast, M, 64, false); };
    EvPair ev;
    double burner = 0;
    for (int i = 0; i < 20000 && burner < 300.0; ++i) burner += ev(fn_p);
    if (!launch_ok("xxs_verdict burner")) continue;
    const int R = std::max(reps, 10);
    double best_p = 1e9, best_g = 1e9, sum_p = 0, sum_g = 0;
    for (int r = 0; r < R; ++r) {
      const double tp = ev(fn_p);
      const double tg = ev(fn_g);
      best_p = std::min(best_p, tp);
      best_g = std::min(best_g, tg);
      sum_p += tp;
      sum_g += tg;
    }
    if (!launch_ok("xxs_verdict A/B")) continue;
    double best_1 = 1e9, best_2 = 1e9;
    for (int r = 0; r < R; ++r) {
      const double t1 = ev(fn_p);
      const double t2 = ev(fn_p);
      best_1 = std::min(best_1, t1);
      best_2 = std::min(best_2, t2);
    }
    const double rp = macs / (best_p * 1e-3) / 1e12;
    const double rg = macs / (best_g * 1e-3) / 1e12;
    std::printf(
        "  M=%-4d proto %.3f ms (%.2f T-MAC/s, media %.3f) vs GEMV64 %.3f ms (%.2f T-MAC/s, "
        "media %.3f)\n"
        "         -> %.3fx (melhor) / %.3fx (media) ; piso proto-vs-proto %.3fx (R=%d, burner "
        "%.0f ms)\n",
        M, best_p, rp, sum_p / R, best_g, rg, sum_g / R, best_g / std::max(best_p, 1e-9),
        (sum_g / R) / std::max(sum_p / R, 1e-9), best_2 / std::max(best_1, 1e-9), R, burner);
  }
}

// H9.1 veredito: staging FLAT de producao vs 4 celulas MUBUF TH-policy, mesmo
// binario, A/B intercalado com ordem rotacionada, burner >= 300 ms, R >= 10,
// piso 1%. M=128, iq3_s, tile BM64 (a config de producao em M>=64).
// Celulas: 0=prod(FLAT) 1=RT16 2=HT18 3=NT_RT20 4=NT_HT22.
void run_th_verdict(const char *tname, const void *dW, const block_q8_1 *dAQ, float *dC,
                    float *dC2, const std::int64_t N, const std::int64_t K,
                    const std::int64_t ast, const int reps) {
  const int M = 128;
  std::printf("\n  -- H9.1 TH-policy no staging iq3_s (BENCH-ONLY, M=%d, BM64/BN128/BK64) --\n",
              M);
  std::printf("  tensor %s [K=%lld x N=%lld]\n", tname, (long long)K, (long long)N);
  std::printf("  NOTA: a referencia e' o dp4a BM64 (`gemm_launch_t<SolverIq3S,64,128,...>`, o\n"
              "    caminho bit-exato), NAO o despacho `gemm_launch(12)` (que em M>=64 vai ao\n"
              "    WMMA, 1.75x por outra conta). So' assim o A/B isola a policy do staging:\n"
              "    mesma grade, mesma conta, mesmos bytes -- so' TR::load muda (FLAT->MUBUF).\n"
              "    O WMMA consome o MESMO staging (verbatim, gemm.cuh:1510), entao um ganho\n"
              "    aqui transfere; o numero WMMA sai como contexto, fora do A/B.\n");
  auto launch_prod = [&]() -> bool {
    return rdna4::gemm_launch_t<rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1>(
        dW, dAQ, dC, N, K, ast, M, nullptr);
  };
  {
    hipFuncAttributes a{};
    int lds = 0;
    if (rdna4::gemm_attrs_t<rdna4::SolverIq3S, 64, 128, 64, 4, 8, true, true, 1>(a, lds))
      std::printf("  attrs producao dp4a: VGPR=%d LDS=%d spill=%d\n", a.numRegs, lds,
                  (int)a.localSizeBytes);
    const int auxs[4] = {16, 18, 20, 22};
    for (int i = 0; i < 4; ++i) {
      hipFuncAttributes av{};
      int lv = 0;
      bool ok = false;
      switch (auxs[i]) {
        case 16:
          ok = rdna4::gemm_attrs_t<SolverIq3S_TH<16>, 64, 128, 64, 4, 8, true, true, 1>(av,
                                                                                        lv);
          break;
        case 18:
          ok = rdna4::gemm_attrs_t<SolverIq3S_TH<18>, 64, 128, 64, 4, 8, true, true, 1>(av,
                                                                                        lv);
          break;
        case 20:
          ok = rdna4::gemm_attrs_t<SolverIq3S_TH<20>, 64, 128, 64, 4, 8, true, true, 1>(av,
                                                                                        lv);
          break;
        default:
          ok = rdna4::gemm_attrs_t<SolverIq3S_TH<22>, 64, 128, 64, 4, 8, true, true, 1>(av,
                                                                                        lv);
          break;
      }
      if (ok)
        std::printf("  attrs aux=%d: VGPR=%d LDS=%d spill=%d\n", auxs[i], av.numRegs, lv,
                    (int)av.localSizeBytes);
    }
  }
  auto run_cell = [&](const int c) -> bool {
    switch (c) {
      case 0: return launch_prod();
      case 1: return launch_th<16>(dW, dAQ, dC2, N, K, ast, M);
      case 2: return launch_th<18>(dW, dAQ, dC2, N, K, ast, M);
      case 3: return launch_th<20>(dW, dAQ, dC2, N, K, ast, M);
      default: return launch_th<22>(dW, dAQ, dC2, N, K, ast, M);
    }
  };
  const char *cnames[5] = {"dp4a(FLAT)", "RT16", "HT18", "NT_RT20", "NT_HT22"};
  // ---- gate: BIT-EXATIDAO tensor cheio vs producao (mesmo Pf -> mesmos bits;
  // diverge aqui = MUBUF lendo byte errado, nao tolerancia) ----
  {
    if (!launch_prod() || !launch_ok("th gate producao")) {
      std::printf("  TH: producao falhou -- veredito pulado\n");
      return;
    }
    (void)hipDeviceSynchronize();
    const std::size_t n_el = (std::size_t)M * (std::size_t)N;
    std::vector<float> pref(n_el);
    (void)hipMemcpy(pref.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
    bool gate_ok = true;
    for (int c = 1; c <= 4; ++c) {
      if (!run_cell(c) || !launch_ok("th gate variante")) {
        gate_ok = false;
        continue;
      }
      (void)hipDeviceSynchronize();
      std::vector<float> got(n_el);
      (void)hipMemcpy(got.data(), dC2, n_el * sizeof(float), hipMemcpyDeviceToHost);
      const Diff d = diff_bits(got, pref, n_el);
      std::size_t nnf = 0;
      for (float v : got)
        if (!std::isfinite(v)) ++nnf;
      std::printf("  gate %-10s vs producao M=%d N=%lld cheio: BITS diferentes %zu de %zu ; "
                  "max ulp %.0f ; nao-finitos %zu%s\n",
                  cnames[c], M, (long long)N, d.n_diff, n_el, d.max_ulps, nnf,
                  (d.n_diff == 0 && nnf == 0) ? "   (bit-identico)" : "   <-- DIVERGE");
      if (d.n_diff != 0 || nnf != 0) gate_ok = false;
    }
    if (!gate_ok) {
      std::printf("  TH: gate bit-exato REPROVOU -- sem cronometragem\n");
      return;
    }
  }
  // ---- A/B intercalado: burner >= 300 ms na producao, ordem rotacionada ----
  EvPair ev;
  double burner = 0;
  {
    auto fn_p = [&] {
      if (!launch_prod()) std::printf("  TH burner: producao retornou false\n");
    };
    for (int i = 0; i < 20000 && burner < 300.0; ++i) burner += ev(fn_p);
  }
  if (!launch_ok("th burner")) return;
  std::printf("  burner DPM: %.1f ms de producao dp4a antes do A/B\n", burner);
  const int R = std::max(reps, 10);
  double best[5], sum[5];
  for (int c = 0; c < 5; ++c) {
    best[c] = 1e9;
    sum[c] = 0;
  }
  for (int r = 0; r < R; ++r) {
    for (int k = 0; k < 5; ++k) {
      const int c = (r + k) % 5;  // ordem rotacionada por round
      double t = 0;
      if (c == 0) {
        t = ev([&] {
          if (!launch_prod()) std::printf("  TH A/B: producao retornou false\n");
        });
      } else {
        t = ev([&, c] {
          if (!run_cell(c)) std::printf("  TH A/B: celula %d retornou false\n", c);
        });
      }
      best[c] = std::min(best[c], t);
      sum[c] += t;
    }
  }
  if (!launch_ok("th A/B intercalado")) return;
  // piso: producao vs producao intercalada (mesmo padrao, sem variante no meio)
  double best_1 = 1e9, best_2 = 1e9;
  for (int r = 0; r < R; ++r) {
    const double t1 = ev([&] { launch_prod(); });
    const double t2 = ev([&] { launch_prod(); });
    best_1 = std::min(best_1, t1);
    best_2 = std::min(best_2, t2);
  }
  const double noise = best_2 / std::max(best_1, 1e-9);
  const double macs = (double)N * (double)K * (double)M;
  std::printf("  A/B intercalado (min/media de %d, M=%d, burner %.0f ms):\n", R, M, burner);
  for (int c = 0; c < 5; ++c) {
    const double rate = macs / (best[c] * 1e-3) / 1e12;
    const double ratio_min = best[0] / std::max(best[c], 1e-9);
    const double ratio_mean = (sum[0] / R) / std::max(sum[c] / R, 1e-9);
    std::printf("    %-10s %7.3f ms (media %7.3f, %5.2f T-MAC/s) -> prod/cel = %.4fx (min) / "
                "%.4fx (media)%s\n",
                cnames[c], best[c], sum[c] / R, rate, ratio_min, ratio_mean,
                c == 0 ? "   (referencia)" : "");
  }
  std::printf("  piso prod-vs-prod intercalado (min de %d): %.3f ms e %.3f ms -> %.4fx\n", R,
              best_1, best_2, noise);
  std::printf("  VEREDITO H9.1 (piso 1%% = 1.0100x, ruido medido %.4fx):\n", noise);
  for (int c = 1; c <= 4; ++c) {
    const double ratio_min = best[0] / std::max(best[c], 1e-9);
    const char *v;
    if (ratio_min >= 1.01) v = "SHIP (ganho alem do piso; propor knob)";
    else if (ratio_min >= 1.0 / noise && ratio_min >= 1.0 / 1.01) v = "NEUTRO (dentro do ruido)";
    else v = "DEAD (regressao ou neutro sem ganho)";
    std::printf("    %-10s prod/cel %.4fx -> %s\n", cnames[c], ratio_min, v);
  }
  // Contexto fora do A/B: o despacho padrao em M=128 (WMMA salvo RD_GEMM_WMMA=0).
  {
    const bool wmma = rdna4::gemm_use_wmma_iq3s(M);
    auto fn_w = [&] { rdna4::gemm_launch(12, dW, dAQ, dC, N, K, ast, M, nullptr); };
    double bw = 1e9;
    for (int r = 0; r < R; ++r) bw = std::min(bw, ev(fn_w));
    if (!launch_ok("th contexto despacho")) return;
    std::printf("  contexto (fora do A/B): despacho padrao M=128 [%s] %.3f ms = %.2f T-MAC/s\n",
                wmma ? "WMMA" : "dp4a(RD_GEMM_WMMA=0)", bw, macs / (bw * 1e-3) / 1e12);
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
  bool kq_auto = false;
  bool thpol = false;  // H9.1: so' o veredito TH-policy (2 shapes iq3_s, M=128)
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
    } else if (!std::strcmp(argv[i], "--kq")) {
      // SELECAO POR TIPO, nao por nome: o mesmo nome de projecao e' um dtype
      // diferente em camadas diferentes neste modelo, entao a bancada escolhe o
      // MAIOR tensor de cada tipo k-quant (menos token_embd, que e' gather).
      kq_auto = true;
    } else if (!std::strcmp(argv[i], "--thpol")) {
      thpol = true;  // H9.1: veredito TH-policy no staging (2 shapes iq3_s, M=128)
    } else if (argv[i][0] != '-') {
      model = argv[i];
    }
  }
  if (!model) {
    std::printf(
        "uso: bench-gemm-engine-gpu <model.gguf> [--m 16,64,128,512] [--reps 5]\n"
        "                        [--tensor NOME] [--list] [--cfg 1,5,6,7]\n"
        "                        [--no-verify] [--quick] [--types] [--thpol]\n");
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
  } else if (kq_auto) {
    // Um tensor REAL por tipo k-quant, escolhido pelo DTYPE (nao pelo nome).
    for (int dt : {kQ2K, kQ3K, kQ4K, kQ5K, kQ6K}) {
      std::uint64_t best_bytes = 0;
      std::string best;
      for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
        const auto &t = ld.meta().tensors[i];
        if ((int)ld.dtype(i) != dt) continue;
        if (t.name.find("token_embd") != std::string::npos) continue;  // gather, nao GEMM
        if ((std::int64_t)t.dims[0] % 256 != 0 || (std::int64_t)t.dims[1] % 128 != 0) continue;
        if (ld.tensor_bytes(i) > best_bytes) {
          best_bytes = ld.tensor_bytes(i);
          best = t.name;
        }
      }
      if (!best.empty()) {
        std::printf("[--kq] %-9s -> %s (%.1f MB)\n", type_name(dt), best.c_str(),
                    (double)best_bytes / 1e6);
        names.push_back(best);
      }
    }
  } else {
    // Os tres shapes que importam + os tres tipos cobertos: K grande
    // (ffn_down, 17408), N grande (ffn_up, 17408), a projecao de atencao
    // (attn_qkv 10240x5120, iq4_xs neste modelo -- nao ha' tensor 5120x5120) e
    // um tensor iq3_xxs, que e' o segundo tipo em bytes do inventario.
    names = {"blk.3.ffn_up.weight", "blk.2.ffn_down.weight", "blk.2.attn_qkv.weight",
             "blk.1.ffn_down.weight"};
    if (thpol) names = {"blk.3.ffn_up.weight", "blk.2.ffn_down.weight"};  // H9.1: 2 shapes iq3_s
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
    // Sonda GEMV-only (bench-only): q8_0/iq1_s/iq4_nl nao tem solver; o fluxo
    // mede o GEMV em lote em M=128 no mesmo binario. O qk deles difere (32 em
    // q8_0 e iq4_nl), entao `want` usa o qk/bloco do proprio tipo.
    const int bb_nosolv = (dt == kQ80)     ? (int)sizeof(rdna4::block_q8_0)
                          : (dt == kIq1S)  ? (int)sizeof(rdna4::block_iq1_s)
                          : (dt == kIq4NL) ? (int)sizeof(rdna4::block_iq4_nl)
                                           : 0;
    const int qk_dt = (dt == kQ80 || dt == kIq4NL) ? 32 : kQK;
    if (!td && bb_nosolv == 0) {
      std::printf("  tipo nao coberto pelo gemm_launch -- pulado\n");
      continue;
    }
    if (td && (N % 128 != 0 || K % kQK != 0)) {
      std::printf("  N tem de ser multiplo de 128 e K de 256 -- pulado\n");
      continue;
    }
    if (!td && (K % qk_dt != 0)) {
      std::printf("  K tem de ser multiplo de %d (sonda GEMV-only) -- pulado\n", qk_dt);
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

    const std::size_t bb_dt = td ? td->block_bytes : (std::size_t)bb_nosolv;
    const std::uint64_t want = (std::uint64_t)N * (std::uint64_t)(K / qk_dt) * bb_dt;
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

    // Tipos sem solver: so' a sonda GEMV-only (M=128), sem verify/speed de GEMM.
    if (!td) {
      run_gemv_only_probe(info.name.c_str(), dt, dW, dAQ, dC2, N, K, act_full,
                          ld.tensor_bytes(ti), reps);
      std::free(hAQ);
      (void)hipFree(dW);
      (void)hipFree(dAf);
      (void)hipFree(dAQ);
      (void)hipFree(dC);
      (void)hipFree(dC2);
      continue;
    }

    std::printf("  atributos (config que `gemm_launch` escolhe em M=128):\n");
    print_attrs("producao (BM64 BN128 BK64 RM4 RN8)", dt, false, 0);
    if (dt == kIq2XXS) print_xxs_proto_attrs();

    // H9.1 (--thpol): so' o veredito TH-policy (o gate bit-exato vive dentro
    // do run_th_verdict); pula a tabela MxGEMV para segurar o hold curto.
    if (thpol) {
      if (dt == kIq3S && Nr == N && Kr == K) {
        run_th_verdict(info.name.c_str(), dW, dAQ, dC, dC2, N, K, act_full, reps);
      } else {
        std::printf("  --thpol: tensor nao-iq3_s ou forma reduzida -- pulado\n");
      }
      std::free(hAQ);
      (void)hipFree(dW);
      (void)hipFree(dAf);
      (void)hipFree(dAQ);
      (void)hipFree(dC);
      (void)hipFree(dC2);
      continue;
    }

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

      // (a2) staging dos k-quants: o quant cru + o record contra o DEQUANT
      //      CANONICO (`dequant.cuh`, parametrizacao de threads do llama.cpp).
      //      Reconstroi w_e = dw*sc_g*q_e - dmn*m_g a partir do que o kernel
      //      depositou e compara com o dequant do MESMO bloco: uma troca de
      //      nibble/bit/sinal (o defeito do `.x/.y` do iq4_xs, 4096 de 4096
      //      errados) aparece aqui como erro de ordem 1.
      {
        void (*deq)(const unsigned char *, float *) = dequant_of(dt);
        if (deq) {
          const int nsub = 64;  // 8 super-blocos = 2048 pesos da linha 0
          int *dStage = nullptr;
          if (hipMalloc(&dStage, (std::size_t)nsub * 12 * sizeof(int)) != hipSuccess) continue;
          dump_stage_kq_dt(dt, (const char *)dW, dStage, 0, (int)(K / kQK), nsub);
          if (!launch_ok("dump_stage_kq")) continue;
          (void)hipDeviceSynchronize();
          std::vector<int> staged((std::size_t)nsub * 12);
          (void)hipMemcpy(staged.data(), dStage, staged.size() * sizeof(int),
                          hipMemcpyDeviceToHost);
          (void)hipFree(dStage);
          const bool two = (dt == kQ2K || dt == kQ3K || dt == kQ6K);
          int bad = 0, first_bad = -1;
          double maxrel = 0;
          for (int t = 0; t < nsub; ++t) {
            const int sb = t / 8, sub = t % 8;
            float yy[256];
            deq(w_raw.data() + (std::size_t)sb * td->block_bytes, yy);
            KqCpu s{};
            u2f(staged[(std::size_t)t * 12 + 8], s.dw);
            u2f(staged[(std::size_t)t * 12 + 9], s.dmn);
            s.s0 = staged[(std::size_t)t * 12 + 10];
            s.s1 = staged[(std::size_t)t * 12 + 11];
            for (int e = 0; e < 32; ++e)
              s.q[e] = (int8_t)((unsigned)staged[(std::size_t)t * 12 + e / 4] >>
                                (8 * (e % 4)));  // byte e/4, posicao e%4
            for (int e = 0; e < 32; ++e) {
              int sc, mm;
              cpu_scm(dt, s, (two && e >= 16) ? 1 : 0, sc, mm);
              const float got = (s.dw * (float)sc) * (float)s.q[e] - s.dmn * (float)mm;
              const float want = yy[32 * sub + e];
              const double err = std::fabs((double)got - (double)want);
              const double ref = std::fabs((double)want) + 1e-30;
              if (err > 1e-5 * ref + 1e-30) {
                ++bad;
                if (first_bad < 0) first_bad = 32 * sub + e;
                maxrel = std::max(maxrel, err / ref);
              }
            }
          }
          std::printf(
              "  staging k-quant: pesos reconstruidos do staging diferentes do dequant\n"
              "    canonico: %d de 2048%s%s (recon: dw*sc*q - dmin*m; tolerancia rel 1e-5,\n"
              "    que so' cobre ordem de arredondamento -- um nibble/bit trocado da' erro ~1)\n",
              bad, bad ? "" : " (ok)",
              first_bad >= 0 ? "  <-- PRIMEIRO PESO ERRADO" : "");
          (void)maxrel;
          if (bad) {
            std::printf("  REPROVADO: o staging nao reproduz o dequant canonico\n");
            continue;
          }
        }
      }

      // (a3) staging do prototipo iq2_xxs (bench-only): int8 + (dw, ls) contra
      // a algebra do vec_dot em CPU. O gate real (0/4096 vs o vec_dot de
      // producao `_perm2`) sai em (c), que para dt=7 usa o prototipo via
      // bench_gemm_launch.
      if (dt == kIq2XXS) {
        if (!check_stage_xxs(w_raw, dW, K)) continue;
      }

      // (b) caso pequeno: nao-finitos, zeros, e oraculo de CPU (iq3_s)
      {
        const int Mc = 4, Nc = 64;
        std::vector<float> got((std::size_t)Mc * (std::size_t)Nc, 0.f);
        (void)hipMemset(dC, 0, (std::size_t)Mc * (std::size_t)Nc * sizeof(float));
        if (!bench_gemm_launch(dt, dW, dAQ, dC, Nc, K, K / kQ8, Mc) ||
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
        // oraculo de CPU dos k-quants: a MESMA formulacao do kernel (produto
        // int32 exato por metade, escala/minimo inteiros, dw*da e dmn*da em
        // fp32, std::fma na ordem do motor) -> tem de ser BIT-EXATO
        if (dequant_of(dt)) {
          std::size_t ndiff = 0;
          double mx = 0, mrel = 0, mref = 0;
          for (int m = 0; m < Mc; ++m) {
            for (int n = 0; n < Nc; ++n) {
              const float wantv = cpu_oracle_kq(dt, w_raw.data(), K / kQK, hAQ, m, n, (int)K,
                                                td->block_bytes);
              const float g = got[(std::size_t)m * Nc + n];
              if (f2u(wantv) != f2u(g)) ++ndiff;
              const double d = std::fabs((double)g - (double)wantv);
              mx = std::max(mx, d);
              mref = std::max(mref, std::fabs((double)wantv));
              if (std::fabs((double)wantv) > 0) mrel = std::max(mrel, d / std::fabs((double)wantv));
            }
          }
          std::printf(
              "  oraculo de CPU (mesma formulacao do kernel: int32 exato por metade, escala e\n"
              "    minimo INTEIROS, dw*da e dmn*da em fp32, std::fma na ordem do motor):\n"
              "    M=%d N=%d K=%lld -> max|ref| %.6g ; max|gpu-cpu| %.3g ; max rel %.3g ;\n"
              "    BITS diferentes %zu de %d%s\n",
              Mc, Nc, (long long)K, mref, mx, mrel, ndiff, Mc * Nc,
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
        if (!bench_gemm_launch(dt, dW, dAQ, dC, Ne, K, K / kQ8, Me) ||
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
    // tempo do GEMM de producao por M (o denominador da fracao de staging)
    std::vector<double> gemm_ms(ms_list.size(), -1.0);
    for (std::size_t mi = 0; mi < ms_list.size(); ++mi) {
      const int M = ms_list[mi];
      const double macs = macs_per * (double)M;
      const std::size_t mid = ms_list.size() > 2 ? 2 : ms_list.size() - 1;
      bool warned = false;
      auto fn_gemm = [&] {
        if (!bench_gemm_launch(dt, dW, dAQ, dC, Nr, Kr, act_full, M) && !warned) {
          warned = true;
          std::printf("  gemm_launch(M=%d) retornou false\n", M);
        }
      };
      const Timing tg = time_it(fn_gemm, reps, quick ? 60.0 : 250.0);
      gemm_ms[mi] = tg.ms;
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
      if (!bench_gemm_launch(dt, dW, dAQ, dC, Nr, Kr, act_full, M) ||
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

    // ---- VEREDITO iq2_xxs (prototipo bench-only, A/B intercalado) -----------
    if (dt == kIq2XXS && Nr == N && Kr == K) {
      run_xxs_verdict(ms_list, reps, dt, dW, dAQ, dC, dC2, N, K, act_full);
    }

    // ---- R1 prototipo coalescido (BENCH-ONLY, iq3_s) ------------------------
    // Regra pre-registrada: PASS = staging+consume prototipo >= 1,3x o tempo do
    // shipping em tensor iq3_s real (classe blk.3.ffn_up, M=128) com saidas
    // BIT-IDENTICAS linha-a-linha (tolerancia NAO se aplica: le os mesmos valores).
    // FAIL abaixo disso -> negativo documentado, sem proposta de producao.
    // Protocolo: burner DPM >= 300 ms primeiro, A/B intercalado vs celula ship,
    // min-of-reps event-pair, mais piso ship-vs-ship. Baseline = tabela acima.
    if (dt == kIq3S && Nr == N && Kr == K) {
      const int Mp = 128;
      std::printf("\n  -- R1 prototipo coalescido iq3_s (BENCH-ONLY, M=%d) --\n", Mp);
      std::printf(
          "  layout: window-major 16 B/sub (qs8+sg4+qh1+d2+sc1), 32 B/(linha,janela BK64),\n"
          "    idx=(w*N+n)*2+kb, 1x b128 sequencial, default cache policy (sem TH_NT).\n"
          "    grade/memoria/computo = shipping BM64 BN128 BK64 RM4 RN8 DBUF.\n");
      hipFuncAttributes attr_s{}, attr_p{};
      int lds_s = 0, lds_p = 0;
      const bool ok_s = rdna4::gemm_attrs(dt, Mp, attr_s, lds_s);
      auto *fn_p = &gemm_iq3s_coal_kernel;
      const bool ok_p = hipFuncGetAttributes(&attr_p, (const void *)fn_p) == hipSuccess;
      lds_p = 2 * (128 * 80 + 2 * 128 * 8);
      if (ok_s && ok_p) {
        std::printf("  attrs ship: VGPR=%d LDS=%d spill=%d | proto: VGPR=%d LDS=%d spill=%d\n",
                    attr_s.numRegs, lds_s, (int)attr_s.localSizeBytes, attr_p.numRegs, lds_p,
                    (int)attr_p.localSizeBytes);
      }
      std::vector<int> hP;
      repack_iq3s_coal(w_raw, hP, (int)N, (int)K);
      const std::size_t nsub = (std::size_t)(K / 32);
      std::printf("  repack host (offline, 1x): %.1f MB -> %.1f MB (+%.1f %%)\n",
                  (double)w_raw.size() / 1e6, (double)hP.size() * sizeof(int) / 1e6,
                  100.0 * ((double)hP.size() * sizeof(int) / (double)w_raw.size() - 1.0));
      int4 *dWp = nullptr;
      const bool ok_alloc =
          hipMalloc(&dWp, hP.size() * sizeof(int)) == hipSuccess && (std::size_t)N * nsub * 4 == hP.size();
      if (!ok_alloc) {
        std::printf("  R1: hipMalloc(dWp) falhou -- prototipo pulado\n");
      } else {
        (void)hipMemcpy(dWp, hP.data(), hP.size() * sizeof(int), hipMemcpyHostToDevice);
        (void)hipDeviceSynchronize();
        bool warned_s = false, warned_p = false;
        auto fn_ship = [&] {
          if (!rdna4::gemm_launch(dt, dW, dAQ, dC, N, K, act_full, Mp, nullptr) && !warned_s) {
            warned_s = true;
            std::printf("  R1: gemm_launch(ship) retornou false\n");
          }
        };
        auto fn_proto = [&] {
          if (!launch_iq3s_coal(dWp, dAQ, dC2, (int)N, (int)K, (int)act_full, Mp) && !warned_p) {
            warned_p = true;
            std::printf("  R1: launch_iq3s_coal retornou false\n");
          }
        };
        if (!launch_ok("R1 alloc")) {
        }
        // prova de match exato: ship vs proto nas MESMAS entradas, M=128 cheio.
        fn_ship();
        (void)hipDeviceSynchronize();
        if (!launch_ok("R1 ship (match)")) {
        }
        fn_proto();
        (void)hipDeviceSynchronize();
        if (!launch_ok("R1 proto (match)")) {
        } else {
          const std::size_t n_el = (std::size_t)Mp * (std::size_t)N;
          std::vector<float> a(n_el), b(n_el);
          // re-roda ordenado para comparar: ship->a, proto->b.
          fn_ship();
          (void)hipDeviceSynchronize();
          (void)hipMemcpy(a.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
          fn_proto();
          (void)hipDeviceSynchronize();
          (void)hipMemcpy(b.data(), dC2, n_el * sizeof(float), hipMemcpyDeviceToHost);
          const Diff dd = diff_bits(b, a, n_el);
          std::size_t nnf = 0;
          for (float v : b)
            if (!std::isfinite(v)) ++nnf;
          std::printf(
              "  MATCH EXATO vs shipping (M=%d N=%lld cheio, linha-a-linha): BITS diferentes %zu de "
              "%zu ; max ulp %.0f ; max|d| %.3g ; nao-finitos %zu%s\n",
              Mp, (long long)N, dd.n_diff, n_el, dd.max_ulps, dd.max_abs, nnf,
              (dd.n_diff == 0 && nnf == 0) ? "   (bit-identico)" : "   <-- DIVERGE");
          // burner DPM >= 300 ms primeiro (protocolo), depois A/B intercalado.
          EvPair ev;
          double burner = 0;
          for (int i = 0; i < 20000 && burner < 300.0; ++i) burner += ev(fn_ship);
          std::printf("  burner DPM: %.1f ms de ship antes do A/B\n", burner);
          const int Rp = std::max(reps, 10);
          double best_s = 1e9, best_p = 1e9;
          double sum_s = 0, sum_p = 0;
          for (int r = 0; r < Rp; ++r) {
            const double ts = ev(fn_ship);
            const double tp = ev(fn_proto);
            best_s = std::min(best_s, ts);
            best_p = std::min(best_p, tp);
            sum_s += ts;
            sum_p += tp;
          }
          if (!launch_ok("R1 A/B intercalado")) {
          }
          // piso ship-vs-ship intercalado (mesmo padrao, sem proto no meio).
          double best_s1 = 1e9, best_s2 = 1e9;
          for (int r = 0; r < Rp; ++r) {
            const double t1 = ev(fn_ship);
            const double t2 = ev(fn_ship);
            best_s1 = std::min(best_s1, t1);
            best_s2 = std::min(best_s2, t2);
          }
          const double macs = (double)N * (double)K * (double)Mp;
          const double rate_s = macs / (best_s * 1e-3) / 1e12;
          const double rate_p = macs / (best_p * 1e-3) / 1e12;
          const double ratio = best_s / std::max(best_p, 1e-9);
          const double noise = best_s2 / std::max(best_s1, 1e-9);
          std::printf(
              "  A/B intercalado ship-vs-proto (min de %d, M=%d): ship %.3f ms (%.2f T-MAC/s) vs "
              "proto %.3f ms (%.2f T-MAC/s) -> proto/ship = %.3fx (tempo ship/proto)\n"
              "    medias: ship %.3f ms proto %.3f ms\n"
              "  piso ship-vs-ship intercalado (min de %d): %.3f ms e %.3f ms -> %.3fx\n",
              Rp, Mp, best_s, rate_s, best_p, rate_p, ratio, sum_s / Rp, sum_p / Rp, Rp, best_s1,
              best_s2, noise);
          const bool pass = (dd.n_diff == 0) && (ratio >= 1.3);
          std::printf("  VEREDITO R1 vs regra 1,3x: %s (ratio %.3fx, bits %zu)%s\n", pass ? "PASS" : "FAIL",
                      ratio, dd.n_diff,
                      pass ? "" : " -- negativo documentado, sem proposta de producao");
        }
        (void)hipFree(dWp);
      }
    }

    // ---- A/B de configuracao ----------------------------------------------
    // cfg 1-8 existem so' para iq3_s (limite de instanciacoes); cfg 9-12
    // (staging cru + decodificacao no consumo, e o modo "so' staging") valem
    // para os tipos cobertos. Todo cfg que CALCULA resultado passa pelo mesmo
    // gate de BIT-EXATIDAO do caminho de producao (o `vec_dot_*` do motor em
    // M=16 N=256) antes de ser cronometrado.
    const bool cfg_ge9 =
        std::any_of(cfg_list.begin(), cfg_list.end(), [](const int c) { return c >= 9; });
    if (!cfg_list.empty() && (dt == kIq3S || cfg_ge9)) {
      std::printf("\n  -- A/B de configuracao (mesma janela da tabela acima) --\n");
      std::vector<std::vector<double>> cfg_ms(cfg_list.size(),
                                              std::vector<double>(ms_list.size(), -1.0));
      for (std::size_t ci = 0; ci < cfg_list.size(); ++ci) {
        const int cfg = cfg_list[ci];
        const bool stage_only = (cfg == 11 || cfg == 12);
        char tag[96];
        std::snprintf(tag, sizeof(tag), "cfg %d%s", cfg, stage_only ? " (so' staging)" : "");
        print_attrs(tag, dt, true, cfg);
        if (stage_only) {
          // o modo de staging nao calcula resultado, entao nao passa pelo gate
          // de bit-exatidao -- mas TEM de passar pela prova de que o kernel
          // existe: cronometrar um `gemm_launch_cfg` que devolve `false` mede um
          // lambda vazio (a armadilha das bancadas que reportaram 1198 T-MAC/s).
          if (!rdna4::gemm_launch_cfg(dt, dW, dAQ, dC, kNc, K, K / kQ8, kM16, cfg, nullptr) ||
              !launch_ok("gemm_launch_cfg (probe do modo de staging)")) {
            std::printf("    cfg %d indisponivel para este tipo -- pulado\n", cfg);
            continue;
          }
        } else {
          // gate de bit-exatidao contra o `vec_dot_*` DO MOTOR, o mesmo M/N do
          // caminho de producao (M=16, N=256)
          const int Me = kM16, Ne = kNc;
          std::vector<float> engv((std::size_t)Me * Ne), gotv((std::size_t)Me * Ne);
          (void)hipMemset(dC2, 0, engv.size() * sizeof(float));
          td->ref((const void *)dW, dAQ, dC2, Me, Ne, (int)K);
          if (!launch_ok("ref_engine_kernel (cfg)")) continue;
          (void)hipMemset(dC, 0, gotv.size() * sizeof(float));
          if (!rdna4::gemm_launch_cfg(dt, dW, dAQ, dC, Ne, K, K / kQ8, Me, cfg, nullptr) ||
              !launch_ok("gemm_launch_cfg (verify)")) {
            std::printf("    cfg %d indisponivel para este tipo -- pulado\n", cfg);
            continue;
          }
          (void)hipDeviceSynchronize();
          (void)hipMemcpy(engv.data(), dC2, engv.size() * sizeof(float), hipMemcpyDeviceToHost);
          (void)hipMemcpy(gotv.data(), dC, gotv.size() * sizeof(float), hipMemcpyDeviceToHost);
          const Diff dd = diff_bits(gotv, engv, engv.size());
          std::size_t nz = 0, nnf = 0;
          for (float v : gotv) {
            if (v == 0.f) ++nz;
            if (!std::isfinite(v)) ++nnf;
          }
          std::printf(
              "    BIT-EXATIDAO vs `vec_dot_%s_q8_1` (M=%d N=%d): BITS diferentes %zu de %zu ; "
              "max ulp %.0f ; zeros %zu ; nao-finitos %zu%s\n",
              td->name, Me, Ne, dd.n_diff, dd.n, dd.max_ulps, nz, nnf,
              dd.n_diff ? "   <-- DIVERGE" : "   (bit-exato)");
        }
        for (std::size_t mi = 0; mi < ms_list.size(); ++mi) {
          const int M = ms_list[mi];
          bool warned = false;
          auto fn = [&] {
            if (!rdna4::gemm_launch_cfg(dt, dW, dAQ, dC, Nr, Kr, act_full, M, cfg, nullptr) &&
                !warned) {
              warned = true;
              std::printf("  cfg %d indisponivel em M=%d\n", cfg, M);
            }
          };
          const Timing t = time_it(fn, reps, quick ? 60.0 : 200.0);
          if (!launch_ok("gemm_launch_cfg")) continue;
          cfg_ms[ci][mi] = t.ms;
          const double macs = macs_per * (double)M;
          std::printf("  %5d | %9.3f ms  %9.2f T-MAC/s%s\n", M, t.ms, macs / (t.ms * 1e-3) / 1e12,
                      stage_only ? "   (so' staging: nao conta como GEMM)" : "");
        }
      }
      // ---- fracao de staging: t(so' staging) / t(GEMM completo) -----------
      // Mesma grade, mesmos threads, mesmo prologo e mesmo staging do caminho
      // cheio; a conta vira 4 leituras da LDS. O modo NAO esconde a latencia
      // das cargas globais (nao tem conta para sobrepor), entao a fracao e' um
      // LIMITE SUPERIOR do custo de staging.
      auto find_cfg = [&](const int cfg) -> int {
        for (std::size_t i = 0; i < cfg_list.size(); ++i)
          if (cfg_list[i] == cfg) return (int)i;
        return -1;
      };
      const int i11 = find_cfg(11), i12 = find_cfg(12), i9 = find_cfg(9);
      if ((i11 >= 0 || i12 >= 0) && dt == kIq3S) {
        std::printf("\n  fracao de staging (t_so_staging / t_GEMM completo, mesma janela):\n");
        for (std::size_t mi = 0; mi < ms_list.size(); ++mi) {
          const int M = ms_list[mi];
          if (i11 >= 0 && cfg_ms[i11][mi] > 0 && gemm_ms[mi] > 0)
            std::printf("    M=%-4d caminho de producao (LDS int8): %.3f / %.3f ms = %.1f %%\n", M,
                        cfg_ms[i11][mi], gemm_ms[mi], 100.0 * cfg_ms[i11][mi] / gemm_ms[mi]);
          if (i12 >= 0 && cfg_ms[i12][mi] > 0 && i9 >= 0 && cfg_ms[i9][mi] > 0)
            std::printf("    M=%-4d caminho CRU (LDS de campos):   %.3f / %.3f ms = %.1f %%\n", M,
                        cfg_ms[i12][mi], cfg_ms[i9][mi], 100.0 * cfg_ms[i12][mi] / cfg_ms[i9][mi]);
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
