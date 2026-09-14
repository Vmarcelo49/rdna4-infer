// Frente D do estudo de prefill (docs/estudo-prefill-d-wmma.md): quanto valem os
// matrix cores (WMMA) neste gfx1201 contra o dp4a que o motor usa hoje, e onde
// o caminho trava.
//
// O que este bench decide, com numero:
//   1. LAYOUT  -- o builtin `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
//      compila e RODA? Qual o empacotamento exato dos operandos por lane (A, B)
//      e do acumulador (D)? A resposta e' varrida: 32 hipoteses sao testadas
//      contra uma referencia de CPU e a que zera as divergencias e' a resposta.
//   2. PROBE   -- verificacao independente: acende UM byte do fragmento por vez
//      e o resultado devolve o (i, j) real daquele byte, sem depender da
//      hipotese. O mapa inteiro e' conferido como bijecao sobre o tile 16x16.
//   3. PEAK    -- laco apertado de WMMAs independentes: MACs/s e TOPS de WMMA
//      int8, WMMA f16 (acumulador f32 e f16) e dp4a, lado a lado; mais ciclos
//      por instrucao por SIMD e um laco misto WMMA+dp4a para dizer se o matrix
//      core e' um pipe separado ou divide o slot de issue com a VALU.
//   4. LDS     -- banda de LDS medida, contra a que o staging do minigemm pede.
//   5. EXATO   -- o int32 por bloco de 32 do WMMA e' identico ao do dp4a?
//      (decide se um caminho WMMA pode ser bit-exato ou exige gate numerico)
//      Inclui +-127/-128 para testar a largura do acumulador do matrix core.
//   6. GEMM    -- minigemm C[M x N] = A[M x K] . W[N x K] int8 com correcao de
//      escala fp32 por bloco de 32 (esquema k-quant/Q8_1), pesos e ativacao em
//      LDS; e a variante f16 (peso dequantizado para f16 no staging), que e' o
//      caminho que o llama.cpp/Vulkan usa no prefill (frente COM, confirmado no
//      codigo: mul_mm.comp:340-342 coopmat<FLOAT_TYPE=f16> com ACC_TYPE=f32).
//
// Toda corrida de GPU passa por ./scripts/gpu-lock.sh com o timeout dentro do
// lock. Buffers pequenos, poucas repeticoes, aquecimento (ramp de DPM) e piso
// de ruido medido com a MESMA celula duas vezes.
//
// usage: bench-wmma-gpu [--only all|layout|probe|peak|lds|exact|gemm]
//                       [--reps N] [--rounds N] [--iters N] [--gm a,b] [--gn a,b]
//                       [--gk a,b]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

// ---------------------------------------------------------------------------
// Fragmentos. gfx12 wave32: o WMMA int8 16x16x16 toma int32x2 (8 int8) por lane
// em cada operando e devolve int32x8; o f16 16x16x16 toma 8 halves (16 B) por
// lane e devolve f32x8 (ou f16x8, variante de acumulador f16).
// ---------------------------------------------------------------------------
typedef int8_t i8x8_t __attribute__((ext_vector_type(8)));
typedef int32_t i32x2_t __attribute__((ext_vector_type(2)));
typedef int32_t i32x8_t __attribute__((ext_vector_type(8)));
typedef _Float16 f16x8_t __attribute__((ext_vector_type(8)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));

#define WMMA_I8(a, b, c) \
  __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, (a), true, (b), (c), true)
#define WMMA_F16(a, b, c) __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12((a), (b), (c))
#define WMMA_F16ACC(a, b, c) __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12((a), (b), (c))
#define DP4A(a, b, c) __builtin_amdgcn_sudot4(true, (a), true, (b), (c), false)

__device__ __forceinline__ i32x2_t pack_i32x2(const int lo, const int hi) {
  i32x2_t v;
  v[0] = lo;
  v[1] = hi;
  return v;
}
__device__ __forceinline__ i32x8_t zero_i32x8() {
  i32x8_t z;
#pragma unroll
  for (int i = 0; i < 8; ++i) z[i] = 0;
  return z;
}
__device__ __forceinline__ f32x8_t zero_f32x8() {
  f32x8_t z;
#pragma unroll
  for (int i = 0; i < 8; ++i) z[i] = 0.0f;
  return z;
}
__device__ __forceinline__ _Float16 u16_as_f16(const uint16_t u) {
  union {
    uint16_t u;
    _Float16 h;
  } c;
  c.u = u;
  return c.h;
}
__device__ __forceinline__ uint16_t f16_as_u16(const _Float16 h) {
  union {
    uint16_t u;
    _Float16 h;
  } c;
  c.h = h;
  return c.u;
}
// 8 halves de um array local -> um vetor de fragmento (sem depender de
// alinhamento de 16 B do array local).
__device__ __forceinline__ f16x8_t f16_frag(const _Float16 *a) {
  f16x8_t v;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint16_t u = f16_as_u16(a[i]);
    v[i] = u16_as_f16(u);
  }
  return v;
}
__device__ __forceinline__ int i32_at(const int8_t *p) {
  return *(const int *)p;
}

// ===========================================================================
// 1/2. LAYOUT: hipoteses de empacotamento
// ===========================================================================
// A varredura cobre 4 variantes de A x 4 de B x 2 de D. Cada variante e' uma
// funcao (lane, byte p) -> (i, j). A matriz A tem shape [16][16] = [m][k]; o
// operando B e' lido de uma matriz Bn[16][16] = [n][k] (a transposta do
// "B[k][n]" do produto -- e' como o llama.cpp e o hipfire guardam o operando B:
// linha = n, coluna = k).
//
//   0: i = lane%16,         j = 8*(lane/16) + p      ("linha por lane")
//   1: i = 8*(lane/16) + p, j = lane%16              (transposta)
//   2: i = lane%16,         j = 2*p + (lane/16)      (k intercalado)
//   3: i = lane%16,         j = 8*(1 - lane/16) + p  (metades trocadas)
//
// D: 0 -> slot j do lane L = C[i = 8*(L/16)+j][j2 = L%16]
//    1 -> slot j do lane L = C[i = L%16][j2 = 8*(L/16)+j]
__device__ __forceinline__ void frag_index(const int variant, const int lane, const int p, int *i,
                                           int *j) {
  const int lo = lane & 15, hi = lane >> 4;
  switch (variant) {
    case 0: *i = lo; *j = 8 * hi + p; break;
    case 1: *i = 8 * hi + p; *j = lo; break;
    case 2: *i = lo; *j = 2 * p + hi; break;
    default: *i = lo; *j = 8 * (1 - hi) + p; break;
  }
}

// Um warp, um tile 16x16. T = int8_t (ACC = int) ou _Float16 (ACC = float).
template <class T, class ACC>
__global__ void decode_kernel(const T *__restrict__ A, const T *__restrict__ Bn,
                              ACC *__restrict__ C, int av, int bv, int dv) {
  const int L = threadIdx.x;
  T a[8], b[8];
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    int am, ak, bn, bk;
    frag_index(av, L, p, &am, &ak);
    frag_index(bv, L, p, &bn, &bk);
    a[p] = A[am * 16 + ak];
    b[p] = Bn[bn * 16 + bk];
  }
  const int lo = L & 15, hi = L >> 4;
  if constexpr (sizeof(T) == 1) {
    int a0, a1, b0, b1;
    __builtin_memcpy(&a0, &a[0], 4);
    __builtin_memcpy(&a1, &a[4], 4);
    __builtin_memcpy(&b0, &b[0], 4);
    __builtin_memcpy(&b1, &b[4], 4);
    const i32x8_t c = WMMA_I8(pack_i32x2(a0, a1), pack_i32x2(b0, b1), zero_i32x8());
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int cm = (dv == 0) ? 8 * hi + j : lo;
      const int cn = (dv == 0) ? lo : 8 * hi + j;
      C[cm * 16 + cn] = (ACC)c[j];
    }
  } else {
    const f32x8_t c = WMMA_F16(f16_frag(a), f16_frag(b), zero_f32x8());
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int cm = (dv == 0) ? 8 * hi + j : lo;
      const int cn = (dv == 0) ? lo : 8 * hi + j;
      C[cm * 16 + cn] = (ACC)c[j];
    }
  }
}

// Probe: um unico byte aceso. `which` = 0 acende A, 1 acende Bn; o outro
// operando recebe (k+1) em cada byte, com k vindo da hipotese. Resultado:
// C[m][n] = 0 em todo lugar menos na linha (which=0) ou coluna (which=1) do byte
// aceso, onde vale k_verdadeiro+1. Ou seja, o probe devolve o (i, j) REAL do
// byte (lane, p), e a hipotese so' decide onde o kernel escreve o resultado.
__global__ void probe_kernel(int *__restrict__ C, int which, int probe_lane, int probe_p, int av,
                             int bv, int dv) {
  const int L = threadIdx.x;
  int a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    int i, j;
    frag_index(which == 0 ? av : bv, L, p, &i, &j);
    const int val = j + 1;  // k+1: o valor observado revela o k verdadeiro
    if (which == 0) {
      a[p] = val;
      b[p] = (L == probe_lane && p == probe_p) ? 1 : 0;
    } else {
      b[p] = val;
      a[p] = (L == probe_lane && p == probe_p) ? 1 : 0;
    }
  }
  const i32x8_t c = WMMA_I8(pack_i32x2(a[0] | (a[1] << 8) | (a[2] << 16) | (a[3] << 24),
                                       a[4] | (a[5] << 8) | (a[6] << 16) | (a[7] << 24)),
                            pack_i32x2(b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24),
                                       b[4] | (b[5] << 8) | (b[6] << 16) | (b[7] << 24)),
                            zero_i32x8());
  const int lo = L & 15, hi = L >> 4;
#pragma unroll
  for (int j = 0; j < 8; ++j) {
    const int cm = (dv == 0) ? 8 * hi + j : lo;
    const int cn = (dv == 0) ? lo : 8 * hi + j;
    C[cm * 16 + cn] = c[j];
  }
}

// ===========================================================================
// 3. PEAK: lacos apertados independentes
// ===========================================================================
// Cada warp emite U*NACC WMMAs por iteracao, com NACC cadeias independentes (sem
// dependencia entre acumuladores) e U o desenrolar. O numero de iteracoes e' um
// argumento de kernel (nada e' dobrado em tempo de compilacao) e o resultado vai
// para a memoria global para nada ser eliminado por DCE.
template <int U, int NACC>
__global__ void peak_wmma_i8_kernel(int iters, int *out) {
  const int L = threadIdx.x & 31;
  i32x8_t acc[NACC];
  i32x2_t a[NACC], b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    acc[i] = zero_i32x8();
    a[i] = pack_i32x2(0x01010101 * (L + 1 + i), 0x01020304 * (i + 1));
    b[i] = pack_i32x2(0x7f7f7f7f - i, 0x01010101 * (L + 3));
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) acc[i] = WMMA_I8(a[i], b[i], acc[i]);
    }
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i)
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[i][j];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

template <int U, int NACC, bool F16ACC>
__global__ void peak_wmma_f16_kernel(int iters, float *out) {
  const int L = threadIdx.x & 31;
  _Float16 av[8], bv[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    av[i] = (_Float16)(0.5f + 0.001f * (float)(i + L));
    bv[i] = (_Float16)(0.25f + 0.001f * (float)(i + L));
  }
  const f16x8_t a = f16_frag(av), b = f16_frag(bv);
  if constexpr (F16ACC) {
    f16x8_t acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) {
      _Float16 z[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) z[k] = (_Float16)0;
      acc[i] = f16_frag(z);
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = WMMA_F16ACC(a, b, acc[i]);
      }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) s += (float)acc[i][j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  } else {
    f32x8_t acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) acc[i] = zero_f32x8();
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = WMMA_F16(a, b, acc[i]);
      }
    }
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) s += acc[i][j];
    out[blockIdx.x * blockDim.x + threadIdx.x] = s;
  }
}

template <int U, int NACC>
__global__ void peak_dp4a_kernel(int iters, int *out) {
  const int L = threadIdx.x & 31;
  int acc[NACC], a[NACC], b[NACC];
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    acc[i] = 0;
    a[i] = 0x01010101 * (L + 1 + i);
    b[i] = 0x7f7f7f01 - i;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) acc[i] = DP4A(a[i], b[i], acc[i]);
    }
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) s += acc[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Laco misto: 1 WMMA + 8 dp4a por passo, acumuladores disjuntos. Se o tempo for
// ~max(WMMA, dp4a) o matrix core e' um pipe separado; se for ~soma, os dois
// disputam o mesmo slot de issue.
template <int U, int NACC>
__global__ void peak_mix_kernel(int iters, int *out) {
  const int L = threadIdx.x & 31;
  i32x8_t wacc[NACC];
  int dacc[NACC], da[NACC], db[NACC];
  const i32x2_t a = pack_i32x2(0x01010101 * (L + 1), 0x01020304);
  const i32x2_t b = pack_i32x2(0x7f7f7f7f, 0x01010101 * (L + 3));
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    wacc[i] = zero_i32x8();
    dacc[i] = 0;
    da[i] = 0x01010101 * (L + 1 + i);
    db[i] = 0x7f7f7f01 - i;
  }
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int i = 0; i < NACC; ++i) wacc[i] = WMMA_I8(a, b, wacc[i]);
#pragma unroll
      for (int i = 0; i < NACC; ++i) dacc[i] = DP4A(da[i], db[i], dacc[i]);
    }
  }
  int s = 0;
#pragma unroll
  for (int i = 0; i < NACC; ++i) {
    s += dacc[i];
#pragma unroll
    for (int j = 0; j < 8; ++j) s += wacc[i][j];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// Relogio: FMA fp32 e' taxa cheia (1 instrucao/ciclo/SIMD32) neste hardware, o
// que faz desse laco um contador de ciclos. E' o unico jeito de dizer "ciclos
// por WMMA" sem confiar no clock nominal -- e a suposicao esta' marcada.
template <int U>
__global__ void peak_fma_kernel(int iters, float *out) {
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = (float)(threadIdx.x + i) * 1e-8f;
  const float m = 1.0000001f, c = 1e-9f;
  for (int it = 0; it < iters; ++it) {
#pragma unroll
    for (int u = 0; u < U; ++u) {
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = fmaf(acc[i], m, c);
    }
  }
  float s = 0.0f;
#pragma unroll
  for (int i = 0; i < 8; ++i) s += acc[i];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ===========================================================================
// 4. LDS
// ===========================================================================
// Copia LDS->LDS com int4 (16 B por lane, endereco consecutivo por lane: sem
// conflito de banco). Le 16 B e escreve 16 B por iteracao por thread.
__global__ void lds_copy_kernel(int iters, int *out) {
  extern __shared__ int4 sm[];
  const int t = threadIdx.x;
  int4 v = sm[t & 1023];
  for (int i = 0; i < iters; ++i) {
    v = sm[(t + i) & 1023];
    sm[(t + 512 + i) & 1023] = v;
  }
  out[t] = v.x;
}

// ===========================================================================
// 5. EXATO: int32 por bloco de 32, WMMA vs dp4a
// ===========================================================================
// A[16][K] e W[16][K] int8. Dois modos:
//   MODE 0 (WMMA): um warp faz o tile 16x16 inteiro, bloco a bloco, e grava o
//           acumulador int32 de CADA bloco de 32 (2 WMMAs por bloco).
//   MODE 1 (dp4a): 256 threads, cada uma um par (m,n), 8 dp4a por bloco de 32.
// Os dois arrays de saida tem que ser IDENTICOS se o matrix core acumular em
// int32 exato (a ordem nao importa em inteiro).
// `extreme` = 1 usa -128/+127x127: 32*127*127 = 516128 e 16*127*127 = 258064,
// numeros que nao sobrevivem a nenhum truncamento em 16 bits.
template <int MODE>
__global__ void exact_block_kernel(const int8_t *__restrict__ A, const int8_t *__restrict__ W,
                                   int *__restrict__ out, int K) {
  const int nblk = K / 32;
  const int L = threadIdx.x & 31;
  if (MODE == 0) {
    const int lo = L & 15, hi = L >> 4;
    for (int b = 0; b < nblk; ++b) {
      i32x8_t acc = zero_i32x8();
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        // fragmento A: lane L -> A[m=lo][k=8*hi+p]; fragmento B: lane L -> Bn[n=lo][k=8*hi+p]
        const int8_t *ap = A + (size_t)lo * K + b * 32 + h * 16 + 8 * hi;
        const int8_t *bp = W + (size_t)lo * K + b * 32 + h * 16 + 8 * hi;
        acc = WMMA_I8(pack_i32x2(i32_at(ap), i32_at(ap + 4)),
                      pack_i32x2(i32_at(bp), i32_at(bp + 4)), acc);
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) out[(size_t)b * 256 + (8 * hi + j) * 16 + lo] = acc[j];
    }
  } else {
    const int m = threadIdx.x >> 4, n = threadIdx.x & 15;
    const int8_t *ar = A + (size_t)m * K;
    const int8_t *wr = W + (size_t)n * K;
    for (int b = 0; b < nblk; ++b) {
      int sumi = 0;
#pragma unroll
      for (int q = 0; q < 8; ++q) {
        const int k = b * 32 + q * 4;
        sumi = DP4A(i32_at(ar + k), i32_at(wr + k), sumi);
      }
      out[(size_t)b * 256 + m * 16 + n] = sumi;
    }
  }
}

// ===========================================================================
// 6. GEMM: C[M][N] = A[M][K] . W[N][K], int8 + escala fp32 por bloco de 32
// ===========================================================================
// CTA = WARPS warps de 32 threads; tile do CTA = BM (tokens) x 64 (linhas de
// peso); BK = 32 (um bloco de quant por iteracao). Cada warp cobre tiles 16x16:
// mt = warp % nMT (tokens) e um pedaco das 4 colunas de 16.
//
// LDS: sA[BM][40] e sW[64][40]. O stride de 40 B (32 + 8 de padding) faz os 16
// lanes de cada metade da wave caírem em 16 bancos pares distintos nos loads de
// 8 B -> sem conflito de banco. Na variante f16 o stride e' 40 halves = 80 B
// (mesma propriedade, alinhado em 16 B).
//
// Correcao de escala: o acumulador do WMMA e' zerado a cada bloco de 32 (2
// WMMAs), o int32 resultante e' multiplicado por d_w*d_a em fp32 e somado no
// acumulador fp32 -- exatamente a sequencia do motor (a secao EXATO mostra que
// o int32 por bloco e' identico ao do dp4a). No layout D descoberto o slot j do
// lane L vale C[m = 8*(L/16)+j][n = L%16] DENTRO do tile: d_w e' 1 valor por
// lane (n = L%16) e d_a sao 8 valores consecutivos (m = 8*(L/16)+j).
//
// Variante f16: o staging converte int8*d para f16 (a escala fica embutida no
// valor, como o llama.cpp faz com k-quant -> f16 no shared) e o acumulador
// fp32 segue por todo o K sem correcao por bloco.
template <int BM, int WARPS, bool F16>
__global__ void gemm_kernel(const int8_t *__restrict__ A, const float *__restrict__ da,
                            const int8_t *__restrict__ W, const float *__restrict__ dw,
                            float *__restrict__ C, int M, int N, int K) {
  constexpr int BN = 64, BK = 32, STRIDE = 40;
  constexpr int nMT = BM / 16, nNT = BN / 16;
  constexpr int warps_per_mt = WARPS / nMT;
  constexpr int nnt = nNT / warps_per_mt;
  static_assert(nMT * 16 == BM && nNT * 16 == BN && (WARPS % nMT) == 0 &&
                    (nNT % warps_per_mt) == 0 && nnt >= 1,
                "geometria de warp invalida");
  using ST = typename std::conditional<F16, _Float16, int8_t>::type;

  __shared__ __align__(16) ST sA[BM * STRIDE];
  __shared__ __align__(16) ST sW[BN * STRIDE];
  __shared__ float sda[BM];
  __shared__ float sdw[BN];

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int lo = lane & 15, hi = lane >> 4;

  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int mt = warp % nMT;
  const int nt0 = (warp / nMT) * nnt;
  const int nKD = K / 32;

  float facc[nnt][8];
#pragma unroll
  for (int t = 0; t < nnt; ++t)
#pragma unroll
    for (int j = 0; j < 8; ++j) facc[t][j] = 0.0f;

  for (int kb = 0; kb < nKD; ++kb) {
    __syncthreads();
    // --- staging cooperativo: BM*32 B de A e 64*32 B de W ---
    for (int idx = tid; idx < BM * 4; idx += WARPS * 32) {
      const int m = idx >> 2, g = idx & 3;
      const bool ok = (m0 + m) < M;
      const uint2 u = ok ? *(const uint2 *)(A + (size_t)(m0 + m) * K + kb * BK + g * 8)
                         : make_uint2(0u, 0u);
      const float s = ok ? da[(size_t)(m0 + m) * nKD + kb] : 0.0f;
      if constexpr (F16) {
        _Float16 *dst = sA + m * STRIDE + g * 8;
        const int8_t *q = (const int8_t *)&u;
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[e] = (_Float16)((float)q[e] * s);
      } else {
        *(uint2 *)(sA + m * STRIDE + g * 8) = u;
      }
    }
    for (int idx = tid; idx < BN * 4; idx += WARPS * 32) {
      const int n = idx >> 2, g = idx & 3;
      const bool ok = (n0 + n) < N;
      const uint2 u = ok ? *(const uint2 *)(W + (size_t)(n0 + n) * K + kb * BK + g * 8)
                         : make_uint2(0u, 0u);
      const float s = ok ? dw[(size_t)(n0 + n) * nKD + kb] : 0.0f;
      if constexpr (F16) {
        _Float16 *dst = sW + n * STRIDE + g * 8;
        const int8_t *q = (const int8_t *)&u;
#pragma unroll
        for (int e = 0; e < 8; ++e) dst[e] = (_Float16)((float)q[e] * s);
      } else {
        *(uint2 *)(sW + n * STRIDE + g * 8) = u;
      }
    }
    if (tid < BM) sda[tid] = (m0 + tid < M) ? da[(size_t)(m0 + tid) * nKD + kb] : 0.0f;
    if (tid >= 128 - BN && tid < 128) {
      const int n = tid - (128 - BN);
      sdw[n] = (n0 + n < N) ? dw[(size_t)(n0 + n) * nKD + kb] : 0.0f;
    }
    __syncthreads();

    float da8[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) da8[j] = F16 ? 0.0f : sda[mt * 16 + 8 * hi + j];

    int32_t iacc[nnt][8];
#pragma unroll
    for (int t = 0; t < nnt; ++t)
#pragma unroll
      for (int j = 0; j < 8; ++j) iacc[t][j] = 0;

#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int k0 = h * 16;
      i32x2_t af;
      f16x8_t aff;
      if constexpr (F16) {
        aff = *reinterpret_cast<const f16x8_t *>(sA + (mt * 16 + lo) * STRIDE + k0 + 8 * hi);
      } else {
        const int8_t *ap = sA + (mt * 16 + lo) * STRIDE + k0 + 8 * hi;
        af = pack_i32x2(i32_at(ap), i32_at(ap + 4));
      }
#pragma unroll
      for (int t = 0; t < nnt; ++t) {
        const int nt = nt0 + t;
        if constexpr (F16) {
          const f16x8_t bf =
              *reinterpret_cast<const f16x8_t *>(sW + (nt * 16 + lo) * STRIDE + k0 + 8 * hi);
          f32x8_t acc;
#pragma unroll
          for (int j = 0; j < 8; ++j) acc[j] = facc[t][j];
          acc = WMMA_F16(aff, bf, acc);
#pragma unroll
          for (int j = 0; j < 8; ++j) facc[t][j] = acc[j];
        } else {
          const int8_t *bp = sW + (nt * 16 + lo) * STRIDE + k0 + 8 * hi;
          const i32x2_t bf = pack_i32x2(i32_at(bp), i32_at(bp + 4));
          i32x8_t c;
#pragma unroll
          for (int j = 0; j < 8; ++j) c[j] = iacc[t][j];
          c = WMMA_I8(af, bf, c);
#pragma unroll
          for (int j = 0; j < 8; ++j) iacc[t][j] = c[j];
        }
      }
    }
    if constexpr (!F16) {
#pragma unroll
      for (int t = 0; t < nnt; ++t) {
        const float wsc = sdw[(nt0 + t) * 16 + lo];
#pragma unroll
        for (int j = 0; j < 8; ++j)
          facc[t][j] = fmaf((float)iacc[t][j], wsc * da8[j], facc[t][j]);
      }
    }
  }

  // escrita: slot j do lane L -> C[m = m0 + mt*16 + 8*hi + j][n = n0 + nt*16 + lo]
  const int cm0 = m0 + mt * 16 + 8 * hi;
#pragma unroll
  for (int t = 0; t < nnt; ++t) {
    const int cn = n0 + (nt0 + t) * 16 + lo;
    if (cn >= N) continue;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int cm = cm0 + j;
      if (cm < M) C[(size_t)cm * N + cn] = facc[t][j];
    }
  }
}

__global__ void empty_kernel() {}

// ===========================================================================
// host
// ===========================================================================
namespace {

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

// Aquecimento (ramp de DPM) + min de `rounds` medicoes de `reps` lancamentos.
template <class F>
double best_ms(F &&launch, int reps, int rounds, double warm_ms = 250.0) {
  const auto w0 = std::chrono::steady_clock::now();
  do {
    launch();
    hipDeviceSynchronize();
  } while (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count() <
           warm_ms);
  EvPair ev;
  double best = 1e30;
  for (int r = 0; r < rounds; ++r) {
    const double ms = ev([&] {
      for (int i = 0; i < reps; ++i) launch();
    }) / reps;
    best = std::min(best, ms);
  }
  return best;
}

uint32_t rng_state = 12345u;
uint32_t rng_next() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 17;
  rng_state ^= rng_state << 5;
  return rng_state;
}

#define HIPCHECK(expr)                                                       \
  do {                                                                       \
    const hipError_t e_ = (expr);                                            \
    if (e_ != hipSuccess) {                                                  \
      std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #expr, __FILE__,   \
                   __LINE__, hipGetErrorString(e_));                         \
      std::exit(1);                                                          \
    }                                                                        \
  } while (0)

void decode_ref_i8(const std::vector<int> &A, const std::vector<int> &Bn, std::vector<int> &C) {
  C.assign(256, 0);
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      int64_t s = 0;
      for (int k = 0; k < 16; ++k) s += (int64_t)A[m * 16 + k] * Bn[n * 16 + k];
      C[m * 16 + n] = (int)s;
    }
}

// fp32 -> fp16 -> fp32 (round-to-nearest-even), so' para a referencia f16.
float f16_round(float x) {
  uint32_t u;
  std::memcpy(&u, &x, 4);
  const uint32_t sign = u & 0x80000000u;
  const int32_t exp = (int32_t)((u >> 23) & 0xff) - 127;
  uint32_t mant = u & 0x7fffffu;
  if (exp > 15) return x;  // fora do range f16 (nao acontece neste bench)
  if (exp < -25) return sign ? -0.0f : 0.0f;
  uint32_t h;  // bits de f16 (sem sinal)
  if (exp < -14) {
    const int shift = -14 - exp;
    const uint32_t m = mant | 0x800000u;
    uint32_t q = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1u);
    const uint32_t half = 1u << (shift - 1);
    if (rem > half || (rem == half && (q & 1u))) q++;
    h = q;
  } else {
    const uint32_t hm = mant >> 13;
    const uint32_t rem = mant & 0x1fffu;
    h = ((uint32_t)(exp + 15) << 10) | hm;
    if (rem > 0x1000u || (rem == 0x1000u && (hm & 1u))) h++;
  }
  const uint32_t out = sign | (h << 13);
  float f;
  std::memcpy(&f, &out, 4);
  return f;
}

// Referencia de CPU do GEMM int8: reproduz a MESMA sequencia aritmetica do
// kernel (int32 por bloco de 32, multiplicacao fp32 por d_w*d_a, soma no
// acumulador fp32, bloco a bloco). Se o kernel bate bit a bit com isto, ele
// reproduz o esquema de correcao do motor.
void gemm_ref_i8(const std::vector<int8_t> &A, const std::vector<float> &da,
                 const std::vector<int8_t> &W, const std::vector<float> &dw, int M, int N, int K,
                 std::vector<float> &C) {
  const int nKD = K / 32;
  C.assign((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = 0.0f;
      for (int b = 0; b < nKD; ++b) {
        int32_t sumi = 0;
        for (int k = 0; k < 32; ++k)
          sumi += (int32_t)A[(size_t)m * K + b * 32 + k] * (int32_t)W[(size_t)n * K + b * 32 + k];
        acc = std::fma((float)sumi, da[(size_t)m * nKD + b] * dw[(size_t)n * nKD + b], acc);
      }
      C[(size_t)m * N + n] = acc;
    }
}

// Referencia f16: os pesos e ativacoes passam por f16 (como no staging) e o
// somatorio e' feito em fp64 sobre os valores f16 -- o que sobra para o kernel
// e' so' o erro de acumulacao fp32 do matrix core.
void gemm_ref_f16(const std::vector<int8_t> &A, const std::vector<float> &da,
                  const std::vector<int8_t> &W, const std::vector<float> &dw, int M, int N, int K,
                  std::vector<float> &C) {
  const int nKD = K / 32;
  C.assign((size_t)M * N, 0.0f);
  std::vector<float> Ah((size_t)M * K), Wh((size_t)N * K);
  for (int m = 0; m < M; ++m)
    for (int b = 0; b < nKD; ++b) {
      const float as = da[(size_t)m * nKD + b];
      for (int k = 0; k < 32; ++k)
        Ah[(size_t)m * K + b * 32 + k] = f16_round((float)A[(size_t)m * K + b * 32 + k] * as);
    }
  for (int n = 0; n < N; ++n)
    for (int b = 0; b < nKD; ++b) {
      const float ws = dw[(size_t)n * nKD + b];
      for (int k = 0; k < 32; ++k)
        Wh[(size_t)n * K + b * 32 + k] = f16_round((float)W[(size_t)n * K + b * 32 + k] * ws);
    }
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double s = 0.0;
      for (int k = 0; k < K; ++k) s += (double)Ah[(size_t)m * K + k] * (double)Wh[(size_t)n * K + k];
      C[(size_t)m * N + n] = (float)s;
    }
}

void fill_i8(std::vector<int8_t> &v, size_t n, int lo, int hi) {
  v.resize(n);
  const uint32_t span = (uint32_t)(hi - lo + 1);
  for (size_t i = 0; i < n; ++i) v[i] = (int8_t)(lo + (int)(rng_next() % span));
}

double rel_l2(const std::vector<float> &a, const std::vector<float> &b) {
  double num = 0, den = 0;
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    const double d = (double)a[i] - b[i];
    num += d * d;
    den += (double)a[i] * a[i] + (double)b[i] * b[i];
  }
  return std::sqrt(num / std::max(1e-300, den));
}

}  // namespace

int main(int argc, char **argv) {
  int reps = 5, rounds = 3, iters_override = 0;
  bool do_layout = true, do_probe = true, do_peak = true, do_lds = true, do_exact = true,
       do_gemm = true;
  std::vector<int> gm{16, 64}, gn{1024, 4096}, gk{5120};

  auto parse_list = [](const char *s, std::vector<int> &out) {
    out.clear();
    while (*s) {
      out.push_back(std::atoi(s));
      while (*s && *s != ',') ++s;
      if (*s == ',') ++s;
    }
  };
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
      const char *s = argv[++i];
      do_layout = do_probe = do_peak = do_lds = do_exact = do_gemm = false;
      if (!std::strcmp(s, "all")) do_layout = do_probe = do_peak = do_lds = do_exact = do_gemm = true;
      else if (!std::strcmp(s, "layout")) do_layout = true;
      else if (!std::strcmp(s, "probe")) do_probe = true;
      else if (!std::strcmp(s, "peak")) do_peak = true;
      else if (!std::strcmp(s, "lds")) do_lds = true;
      else if (!std::strcmp(s, "exact")) do_exact = true;
      else if (!std::strcmp(s, "gemm")) do_gemm = true;
      else {
        std::fprintf(stderr, "secao desconhecida: %s\n", s);
        return 2;
      }
    } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--iters") && i + 1 < argc) iters_override = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--gm") && i + 1 < argc) parse_list(argv[++i], gm);
    else if (!std::strcmp(argv[i], "--gn") && i + 1 < argc) parse_list(argv[++i], gn);
    else if (!std::strcmp(argv[i], "--gk") && i + 1 < argc) parse_list(argv[++i], gk);
    else {
      std::fprintf(stderr, "argumento desconhecido: %s\n", argv[i]);
      return 2;
    }
  }
  reps = std::max(1, reps);
  rounds = std::max(1, rounds);

  hipDeviceProp_t prop{};
  HIPCHECK(hipGetDeviceProperties(&prop, 0));
  int cu = 0, clk_khz = 0;
  HIPCHECK(hipDeviceGetAttribute(&cu, hipDeviceAttributeMultiprocessorCount, 0));
  HIPCHECK(hipDeviceGetAttribute(&clk_khz, hipDeviceAttributeClockRate, 0));
  std::printf("device: %s | %d CU | clock reportado pelo driver %d MHz\n", prop.name, cu,
              clk_khz / 1000);
  std::printf("secoes: layout=%d probe=%d peak=%d lds=%d exact=%d gemm=%d (reps=%d rounds=%d)\n\n",
              (int)do_layout, (int)do_probe, (int)do_peak, (int)do_lds, (int)do_exact, (int)do_gemm,
              reps, rounds);

  // -----------------------------------------------------------------------
  // 1. LAYOUT
  // -----------------------------------------------------------------------
  int win_a = 0, win_b = 0, win_d = 0;
  double clk_hz = 2.5e9;  // SUPOSICAO ate' a secao PEAK medir o relogio
  double fma_per_s = 0.0;
  if (do_layout) {
    std::printf("== 1. LAYOUT: varredura de hipoteses contra a referencia de CPU ==\n");
    std::vector<int> Ah(256), Bh(256), Cref(256);
    for (int i = 0; i < 256; ++i) {
      Ah[i] = (int)(rng_next() % 201) - 100;
      Bh[i] = (int)(rng_next() % 201) - 100;
    }
    decode_ref_i8(Ah, Bh, Cref);
    std::vector<int8_t> A8(256), B8(256);
    for (int i = 0; i < 256; ++i) {
      A8[i] = (int8_t)Ah[i];
      B8[i] = (int8_t)Bh[i];
    }
    int *d_C = nullptr;
    int8_t *d_A = nullptr, *d_B = nullptr;
    float *d_Cf = nullptr;
    _Float16 *d_Ah = nullptr, *d_Bh = nullptr;
    HIPCHECK(hipMalloc(&d_C, 256 * sizeof(int)));
    HIPCHECK(hipMalloc(&d_A, 256));
    HIPCHECK(hipMalloc(&d_B, 256));
    HIPCHECK(hipMalloc(&d_Cf, 256 * sizeof(float)));
    HIPCHECK(hipMalloc(&d_Ah, 256 * sizeof(_Float16)));
    HIPCHECK(hipMalloc(&d_Bh, 256 * sizeof(_Float16)));
    HIPCHECK(hipMemcpy(d_A, A8.data(), 256, hipMemcpyHostToDevice));
    HIPCHECK(hipMemcpy(d_B, B8.data(), 256, hipMemcpyHostToDevice));

    std::vector<int> got(256, 0);
    size_t best_mis = (size_t)-1;
    int best = -1;
    std::printf("  hipotese (A,B,D) -> divergencias / 256:\n");
    for (int av = 0; av < 4; ++av)
      for (int bv = 0; bv < 4; ++bv)
        for (int dv = 0; dv < 2; ++dv) {
          HIPCHECK(hipMemset(d_C, 0, 256 * sizeof(int)));
          decode_kernel<int8_t, int><<<1, 32>>>(d_A, d_B, d_C, av, bv, dv);
          HIPCHECK(hipDeviceSynchronize());
          HIPCHECK(hipMemcpy(got.data(), d_C, 256 * sizeof(int), hipMemcpyDeviceToHost));
          size_t mis = 0;
          int maxd = 0;
          for (int i = 0; i < 256; ++i)
            if (got[i] != Cref[i]) {
              ++mis;
              maxd = std::max(maxd, std::abs(got[i] - Cref[i]));
            }
          std::printf("    A%d B%d D%d  ->  %3zu   max|d| %d%s\n", av, bv, dv, mis, maxd,
                      mis == 0 ? "   <== EXATA" : "");
          if (mis < best_mis) {
            best_mis = mis;
            best = (av * 4 + bv) * 2 + dv;
          }
        }
    if (best < 0 || best_mis != 0) {
      std::printf("  NENHUMA hipotese reproduziu o tile int8 16x16x16: o builtin nao bate a\n"
                  "  referencia de CPU com nenhum empacotamento testado.\n");
    } else {
      win_a = best / 8;
      win_b = (best / 2) % 4;
      win_d = best % 2;
      const char *an = win_a == 0 ? "L%16" : (win_a == 1 ? "8*(L/16)+p" : "L%16");
      const char *ak = win_a == 0 ? "8*(L/16)+p"
                                  : (win_a == 1 ? "L%16" : (win_a == 2 ? "2*p+L/16" : "8*(1-L/16)+p"));
      const char *bn = win_b == 0 ? "L%16" : (win_b == 1 ? "8*(L/16)+p" : "L%16");
      const char *bk = win_b == 0 ? "8*(L/16)+p"
                                  : (win_b == 1 ? "L%16" : (win_b == 2 ? "2*p+L/16" : "8*(1-L/16)+p"));
      std::printf("  vencedora: A%d B%d D%d (0 divergencias em 256)\n", win_a, win_b, win_d);
      std::printf("    A: lane L, byte p = A[m=%s][k=%s]\n", an, ak);
      std::printf("    B (guardado [n][k]): lane L, byte p = Bn[n=%s][k=%s]\n", bn, bk);
      std::printf("    D: slot j do lane L = C[m=%s][n=%s]\n", win_d == 0 ? "8*(L/16)+j" : "L%16",
                  win_d == 0 ? "L%16" : "8*(L/16)+j");

      // estresse: largura do acumulador interno (16 x 127 x 127 = 258064 > 2^16)
      std::vector<int8_t> Amax(256, 127), Bmax(256, 127), Amin(256, -128);
      HIPCHECK(hipMemcpy(d_A, Amax.data(), 256, hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_B, Bmax.data(), 256, hipMemcpyHostToDevice));
      decode_kernel<int8_t, int><<<1, 32>>>(d_A, d_B, d_C, win_a, win_b, win_d);
      HIPCHECK(hipDeviceSynchronize());
      HIPCHECK(hipMemcpy(got.data(), d_C, 256 * sizeof(int), hipMemcpyDeviceToHost));
      int nb = 0;
      for (int i = 0; i < 256; ++i) nb += (got[i] != 16 * 127 * 127);
      std::printf("  estresse +127/+127: esperado %d, obtido %d, fora do esperado %d/256\n",
                  16 * 127 * 127, got[0], nb);
      HIPCHECK(hipMemcpy(d_A, Amin.data(), 256, hipMemcpyHostToDevice));
      decode_kernel<int8_t, int><<<1, 32>>>(d_A, d_B, d_C, win_a, win_b, win_d);
      HIPCHECK(hipDeviceSynchronize());
      HIPCHECK(hipMemcpy(got.data(), d_C, 256 * sizeof(int), hipMemcpyDeviceToHost));
      nb = 0;
      for (int i = 0; i < 256; ++i) nb += (got[i] != 16 * (-128) * 127);
      std::printf("  estresse -128/+127: esperado %d, obtido %d, fora do esperado %d/256\n",
                  16 * (-128) * 127, got[0], nb);

      // semantica do 1o/3o booleano (o 6o e' o "clamp" do acumulador)
      {
        std::vector<int8_t> a1(256, (int8_t)0xFF), b1(256, 1);
        HIPCHECK(hipMemcpy(d_A, a1.data(), 256, hipMemcpyHostToDevice));
        HIPCHECK(hipMemcpy(d_B, b1.data(), 256, hipMemcpyHostToDevice));
        decode_kernel<int8_t, int><<<1, 32>>>(d_A, d_B, d_C, win_a, win_b, win_d);
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(got.data(), d_C, 256 * sizeof(int), hipMemcpyDeviceToHost));
        std::printf("  semantica: A=0xFF, B=+1 -> %d  (=%d se o operando A for lido COM sinal, "
                    "=%d se SEM sinal)\n", got[0], 16 * -1, 16 * 255);
      }

      // ---- f16: mesmo tile 16x16x16 com half ----
      std::printf("  --- f16 (8 halves/lane, acumulador f32) ---\n");
      std::vector<int> Ah16(256), Bh16(256), Cref16(256, 0);
      for (int i = 0; i < 256; ++i) {
        Ah16[i] = (int)(rng_next() % 17) - 8;
        Bh16[i] = (int)(rng_next() % 17) - 8;
      }
      for (int m = 0; m < 16; ++m)
        for (int n = 0; n < 16; ++n) {
          int64_t s = 0;
          for (int k = 0; k < 16; ++k) s += (int64_t)Ah16[m * 16 + k] * Bh16[n * 16 + k];
          Cref16[m * 16 + n] = (int)s;
        }
      std::vector<_Float16> A16(256), B16(256);
      for (int i = 0; i < 256; ++i) {
        A16[i] = (_Float16)Ah16[i];
        B16[i] = (_Float16)Bh16[i];
      }
      HIPCHECK(hipMemcpy(d_Ah, A16.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_Bh, B16.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
      size_t bestf = (size_t)-1;
      int bestf_id = -1;
      std::vector<float> gotf(256, 0.0f);
      for (int av = 0; av < 4; ++av)
        for (int bv = 0; bv < 4; ++bv)
          for (int dv = 0; dv < 2; ++dv) {
            HIPCHECK(hipMemset(d_Cf, 0, 256 * sizeof(float)));
            decode_kernel<_Float16, float><<<1, 32>>>(d_Ah, d_Bh, d_Cf, av, bv, dv);
            HIPCHECK(hipDeviceSynchronize());
            HIPCHECK(hipMemcpy(gotf.data(), d_Cf, 256 * sizeof(float), hipMemcpyDeviceToHost));
            size_t mis = 0;
            for (int i = 0; i < 256; ++i)
              if (std::fabs(gotf[i] - (float)Cref16[i]) > 1e-3f) ++mis;
            if (mis < bestf) {
              bestf = mis;
              bestf_id = (av * 4 + bv) * 2 + dv;
            }
          }
      if (bestf == 0) {
        std::printf("  f16 vencedora: A%d B%d D%d (0 divergencias); mesma receita do int8: %s\n",
                    bestf_id / 8, (bestf_id / 2) % 4, bestf_id % 2,
                    bestf_id == best ? "SIM" : "NAO");
      } else {
        std::printf("  f16: melhor hipotese tem %zu/256 divergencias (nenhuma exata)\n", bestf);
      }

      // precisao do acumulador f32 dentro do WMMA (16 produtos por k-tile)
      {
        std::vector<_Float16> a16(256), b16(256);
        double rel = 0.0, mxd = 0.0;
        for (int rep = 0; rep < 8; ++rep) {
          std::vector<double> ref(256, 0.0);
          for (int i = 0; i < 256; ++i) {
            a16[i] = (_Float16)((float)((int)(rng_next() % 4097) - 2048) * 0.5f);
            b16[i] = (_Float16)((float)((int)(rng_next() % 4097) - 2048) * 0.5f);
          }
          for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n) {
              double s = 0;
              for (int k = 0; k < 16; ++k)
                s += (double)(float)a16[m * 16 + k] * (double)(float)b16[n * 16 + k];
              ref[m * 16 + n] = s;
            }
          HIPCHECK(hipMemcpy(d_Ah, a16.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpy(d_Bh, b16.data(), 256 * sizeof(_Float16), hipMemcpyHostToDevice));
          decode_kernel<_Float16, float><<<1, 32>>>(d_Ah, d_Bh, d_Cf, bestf_id / 8,
                                                    (bestf_id / 2) % 4, bestf_id % 2);
          HIPCHECK(hipDeviceSynchronize());
          HIPCHECK(hipMemcpy(gotf.data(), d_Cf, 256 * sizeof(float), hipMemcpyDeviceToHost));
          for (int i = 0; i < 256; ++i) {
            const double d = std::fabs(gotf[i] - ref[i]);
            rel = std::max(rel, d / std::max(1e-9, std::fabs(ref[i])));
            mxd = std::max(mxd, d);
          }
        }
        std::printf("  precisao f16 (k-tile de 16, |termos| ate ~1e6): rel maxima %.2e, |d| max %.4g\n",
                    rel, mxd);
      }
    }
    HIPCHECK(hipFree(d_C));
    HIPCHECK(hipFree(d_A));
    HIPCHECK(hipFree(d_B));
    HIPCHECK(hipFree(d_Cf));
    HIPCHECK(hipFree(d_Ah));
    HIPCHECK(hipFree(d_Bh));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // 2. PROBE
  // -----------------------------------------------------------------------
  if (do_probe) {
    std::printf("== 2. PROBE: mapa (lane, byte) -> indice, um byte aceso por vez ==\n");
    std::printf("  (usa a hipotese vencedora A%d B%d D%d so' para ESCREVER o resultado;\n"
                "   o indice lido e' o real, entao o mapa vale por si)\n", win_a, win_b, win_d);
    int *d_C = nullptr;
    HIPCHECK(hipMalloc(&d_C, 256 * sizeof(int)));
    std::vector<int> got(256, 0);
    int imap[2][32][8], jmap[2][32][8];
    std::memset(imap, 0xff, sizeof(imap));
    std::memset(jmap, 0xff, sizeof(jmap));
    for (int which = 0; which < 2; ++which)
      for (int L = 0; L < 32; ++L)
        for (int p = 0; p < 8; ++p) {
          HIPCHECK(hipMemset(d_C, 0, 256 * sizeof(int)));
          probe_kernel<<<1, 32>>>(d_C, which, L, p, win_a, win_b, win_d);
          HIPCHECK(hipDeviceSynchronize());
          HIPCHECK(hipMemcpy(got.data(), d_C, 256 * sizeof(int), hipMemcpyDeviceToHost));
          int ii = -1, jj = -1, nz = 0;
          for (int m = 0; m < 16; ++m)
            for (int n = 0; n < 16; ++n)
              if (got[m * 16 + n] != 0) {
                ++nz;
                ii = (which == 0) ? m : n;
                jj = got[m * 16 + n] - 1;
              }
          if (nz != 16)
            std::printf("  AVISO: probe which=%d lane=%d p=%d deu %d celulas nao nulas "
                        "(esperado 16)\n", which, L, p, nz);
          imap[which][L][p] = ii;
          jmap[which][L][p] = jj;
        }
    for (int which = 0; which < 2; ++which) {
      bool seen[16][16];
      std::memset(seen, 0, sizeof(seen));
      int dup = 0, oob = 0;
      for (int L = 0; L < 32; ++L)
        for (int p = 0; p < 8; ++p) {
          const int i = imap[which][L][p], j = jmap[which][L][p];
          if (i < 0 || i > 15 || j < 0 || j > 15) {
            ++oob;
            continue;
          }
          if (seen[i][j]) ++dup;
          seen[i][j] = true;
        }
      std::printf("  mapa %s: (i,j) duplicados %d, fora do tile %d (esperado 0/0)\n",
                  which == 0 ? "A(16x16)" : "B(16x16)", dup, oob);
    }
    std::printf("   lane byte ->  A: (m,k)      B: (n,k)\n");
    const int show[8] = {0, 1, 7, 8, 15, 16, 17, 31};
    for (int s = 0; s < 8; ++s) {
      const int L = show[s];
      for (int p = 0; p < 8; ++p)
        std::printf("   %4d %4d ->  m=%2d k=%2d       n=%2d k=%2d\n", L, p, imap[0][L][p],
                    jmap[0][L][p], imap[1][L][p], jmap[1][L][p]);
    }
    HIPCHECK(hipFree(d_C));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // 3. PEAK
  // -----------------------------------------------------------------------
  if (do_peak) {
    std::printf("== 3. PEAK: WMMA int8, WMMA f16 (acc f32 e f16), dp4a, misto ==\n");
    const int threads = 256;
    const int blocks = cu;  // 1 CTA por "multiprocessor" do driver (= 1 WGP = 2 CU)
    const int nthreads = blocks * threads, nwarps = nthreads / 32;
    int *d_i = nullptr;
    float *d_f = nullptr;
    // a varredura de CTAs vai ate' 8x `blocks`: o buffer de saida tem que
    // acompanhar (o kernel escreve uma celula por thread)
    HIPCHECK(hipMalloc(&d_i, (size_t)nthreads * 8 * sizeof(int)));
    HIPCHECK(hipMalloc(&d_f, (size_t)nthreads * 8 * sizeof(float)));
    // Registradores: o teto de ocupacao de um laco com 8 acumuladores e' o que
    // decide quantos warps cabem por SIMD, i.e. se o numero medido e' limite de
    // vazao do pipe ou de latencia de UMA cadeia.
    {
      hipFuncAttributes a{}, b{}, c{};
      const int ra = hipFuncGetAttributes(&a, (const void *)peak_wmma_i8_kernel<8, 8>) == hipSuccess
                         ? a.numRegs : 0;
      const int rb =
          hipFuncGetAttributes(&b, (const void *)peak_wmma_f16_kernel<8, 8, false>) == hipSuccess
              ? b.numRegs : 0;
      const int rc = hipFuncGetAttributes(&c, (const void *)peak_dp4a_kernel<8, 8>) == hipSuccess
                         ? c.numRegs : 0;
      std::printf("  regs/lane (U=8, 8 acumuladores): wmma_i8 %d -> %d warps/SIMD; wmma_f16 %d -> "
                  "%d; dp4a %d -> %d (teto de 256 VGPRS por lane)\n", ra, 256 / std::max(1, ra), rb,
                  256 / std::max(1, rb), rc, 256 / std::max(1, rc));
    }

    auto calibrate = [&](const std::function<void(int)> &launch) {
      if (iters_override) return iters_override;
      launch(20);
      HIPCHECK(hipDeviceSynchronize());
      EvPair ev;
      const double t1 = std::max(1e-6, ev([&] { launch(20); }) / 20.0);
      return (int)std::max(20.0, 3.0 / t1);
    };

    // --- relogio, pelo laco de FMA fp32 ---
    // O driver reporta 32 "multiprocessors" para esta placa, que tem 64 CU: o
    // que ele conta sao WGPs (par de CU). Quem decide isso e' a medicao: o laco
    // de FMA da' os lanes de fp32, e com 128 lanes/CU a contagem sai certa.
    const double clk_nominal = clk_khz * 1e3;
    const int cu_phys = 2 * cu;
    auto fma_cell = [&](const char *tag) {
      const int U = 8;
      std::function<void(int)> launch = [&](int iters) {
        peak_fma_kernel<U><<<2 * blocks, threads>>>(iters, d_f);
      };
      launch(200);
      HIPCHECK(hipDeviceSynchronize());
      EvPair ev;
      const double t1 = std::max(1e-6, ev([&] { launch(200); }) / 200.0);
      const int iters = iters_override ? iters_override : (int)std::max(200.0, 40.0 / t1);
      const double ms = best_ms([&] { launch(iters); }, 2, rounds, 1500.0);
      const double ops = (double)(2 * blocks * threads) * iters * U * 8;
      const double per_s = ops / (ms * 1e-3);
      std::printf("  laco de FMA fp32 %s (U=%d, 8 cadeias, %d CTAs): %.3f ms -> %.2f T-FMA/s = "
                  "%.2f T-FLOP/s\n", tag, U, 2 * blocks, ms, per_s / 1e12, 2.0 * per_s / 1e12);
      std::printf("    %s: %.1f%% do pico de fp32 desta placa a 2,97 GHz (24,35 T-FMA/s); "
                  "o laco de FMA e'\n    a referencia de issue (SUPOSICAO: 1 FMA/ciclo/SIMD32), "
                  "logo os ciclos abaixo saem\n    normalizados por ELE, nao por um clock "
                  "suposto\n", tag, 100.0 * per_s / 24.35e12);
      return per_s;
    };
    fma_per_s = fma_cell("inicio");
    clk_hz = fma_per_s / ((double)cu_phys * 4.0 * 32.0);
    std::printf("  clock implicito %.3f GHz com %d CU x 128 lanes (driver: %d multiprocessors, "
                "%.3f GHz)\n", clk_hz / 1e9, cu_phys, cu, clk_nominal / 1e9);
    // slots de issue de SIMD por segundo: nao dependem da contagem de CU, so' da
    // suposicao de que o FMA fp32 e' 1 instrucao por ciclo por SIMD32.
    const double simd_slots = fma_per_s / 32.0;

    struct Res {
      double ms = 0, per_iter = 0, instr_per_iter = 0, macs_per_instr = 0, warps = 0;
      double ms_per_unit_U = 0;  // ajuste linear: ms por +1 de U (8 instrucoes/warp)
      double instr_per_s() const { return instr_per_iter * warps / (per_iter * 1e-3); }
      double macs_per_s() const { return instr_per_s() * macs_per_instr; }
      double tops() const { return 2.0 * macs_per_s() / 1e12; }
    };
    int iters_sweep = iters_override ? iters_override : 0;
    auto measure = [&](const char *name, int nblocks, int unroll,
                       const std::function<void(int)> &launch, double instr_per_iter,
                       double macs_per_instr) {
      const int iters = iters_sweep ? iters_sweep : calibrate(launch);
      Res r;
      r.ms = best_ms([&] { launch(iters); }, reps, rounds);
      r.per_iter = r.ms / iters;
      r.instr_per_iter = instr_per_iter;
      r.macs_per_instr = macs_per_instr;
      r.warps = (double)nblocks * threads / 32.0;
      std::printf("  %-22s CTAs=%-4d U=%-2d %7.3f ms  iters=%-5d  %7.2f G-instr-warp/s  %8.2f "
                  "T-MACs/s %6.1f TOPS\n", name, nblocks, unroll, r.ms, iters,
                  r.instr_per_s() / 1e9, r.macs_per_s() / 1e12, r.tops());
      return r;
    };
    auto l_i8 = [&](int nb, int U) {
      return std::function<void(int)>([&, nb, U](int iters) {
        switch (U) {
          case 1: peak_wmma_i8_kernel<1, 8><<<nb, threads>>>(iters, d_i); break;
          case 4: peak_wmma_i8_kernel<4, 8><<<nb, threads>>>(iters, d_i); break;
          default: peak_wmma_i8_kernel<8, 8><<<nb, threads>>>(iters, d_i); break;
        }
      });
    };
    auto l_dp = [&](int nb, int U) {
      return std::function<void(int)>([&, nb, U](int iters) {
        if (U == 1) peak_dp4a_kernel<1, 8><<<nb, threads>>>(iters, d_i);
        else peak_dp4a_kernel<8, 8><<<nb, threads>>>(iters, d_i);
      });
    };
    auto l_f16 = [&](int nb, bool f16acc, int U) {
      return std::function<void(int)>([&, nb, f16acc, U](int iters) {
        if (f16acc) {
          if (U == 1) peak_wmma_f16_kernel<1, 8, true><<<nb, threads>>>(iters, d_f);
          else peak_wmma_f16_kernel<8, 8, true><<<nb, threads>>>(iters, d_f);
        } else {
          if (U == 1) peak_wmma_f16_kernel<1, 8, false><<<nb, threads>>>(iters, d_f);
          else peak_wmma_f16_kernel<8, 8, false><<<nb, threads>>>(iters, d_f);
        }
      });
    };

    std::printf("  --- varredura de CTAs: 1 CTA = 8 warps; %d CU fisicas -> warps/SIMD = "
                "CTAs*8/%d\n", cu_phys, cu_phys * 4);
    struct Cell {
      int nb;
      Res wi8, dp;
    };
    std::vector<Cell> sweep;
    for (int nb : {blocks, 2 * blocks, 4 * blocks, 8 * blocks}) {
      char tag[64];
      std::snprintf(tag, sizeof(tag), "wmma_i8 U=8");
      Res r = measure(tag, nb, 8, l_i8(nb, 8), 64.0, 4096.0);
      Res d = measure("dp4a U=8", nb, 8, l_dp(nb, 8), 64.0, 128.0);
      sweep.push_back({nb, r, d});
    }
    // ajuste linear em U (normalizado por iteracao, porque `iters` difere):
    // t(U)/iters = a*U + b, a = ms por unidade de U = 8 instrucoes por warp
    {
      const int nb = sweep[0].nb;
      const Res w1 = measure("wmma_i8 U=1", nb, 1, l_i8(nb, 1), 8.0, 4096.0);
      const Res w8 = sweep[0].wi8;
      const Res d1 = measure("dp4a U=1", nb, 1, l_dp(nb, 1), 8.0, 128.0);
      const Res d8 = sweep[0].dp;
      const double aw = (w8.per_iter - w1.per_iter) / 7.0;
      const double ad = (d8.per_iter - d1.per_iter) / 7.0;
      const double nw = (double)nb * threads / 32.0;
      std::printf("  ajuste t(U)/iters = a*U + b (a = ms por 8 instrucoes por warp):\n");
      std::printf("    WMMA int8: a=%.5f ms -> %.2f T-MACs/s no limite U->inf\n", aw,
                  4096.0 * 8.0 * nw / (aw * 1e-3) / 1e12);
      std::printf("    dp4a     : a=%.5f ms -> %.2f G-instr-warp/s = %.2f T-MACs/s\n", ad,
                  8.0 * nw / (ad * 1e-3) / 1e9, 128.0 * 8.0 * nw / (ad * 1e-3) / 1e12);
      std::printf("    (o desenrolar tira o overhead de laco; o valor por U das linhas acima ja'\n"
                  "     e' o do laco cheio, a diferenca mede o custo de controle)\n");
    }
    // escolhe a melhor celula da varredura para o resto das medidas
    int best_cell = 0;
    for (size_t i = 1; i < sweep.size(); ++i)
      if (sweep[i].wi8.macs_per_s() > sweep[best_cell].wi8.macs_per_s()) best_cell = (int)i;
    const int nb_best = sweep[best_cell].nb;
    const Res wi8_8 = sweep[best_cell].wi8;
    const Res dp_8 = sweep[best_cell].dp;
    const Res wi8_8b = measure("wmma_i8 (repetido)", nb_best, 8, l_i8(nb_best, 8), 64.0, 4096.0);
    const Res dp_8b = measure("dp4a (repetido)", nb_best, 8, l_dp(nb_best, 8), 64.0, 128.0);
    const Res f16_8 = measure("wmma_f16 acc f32", nb_best, 8, l_f16(nb_best, false, 8), 64.0, 4096.0);
    const Res f16_1 = measure("wmma_f16 acc f32", nb_best, 1, l_f16(nb_best, false, 1), 8.0, 4096.0);
    const Res f16a_8 = measure("wmma_f16 acc f16", nb_best, 8, l_f16(nb_best, true, 8), 64.0, 4096.0);

    std::printf("  --- misto: 1 WMMA + 8 dp4a por passo, acumuladores disjuntos ---\n");
    auto l_mix = [&](int nb, int U) {
      return std::function<void(int)>([&, nb, U](int iters) {
        if (U == 1) peak_mix_kernel<1, 8><<<nb, threads>>>(iters, d_i);
        else peak_mix_kernel<4, 8><<<nb, threads>>>(iters, d_i);
      });
    };
    const Res mx = measure("mix 1 wmma + 8 dp4a", nb_best, 4, l_mix(nb_best, 4), 64.0, 0.0);
    {
      const double per_step = mx.per_iter / 4.0;   // ms por (1 WMMA + 8 dp4a) por warp
      const double wmma = wi8_8.per_iter / 8.0;    // ms por 8 WMMAs (uma unidade de U)
      const double dp4a = dp_8.per_iter / 8.0;     // ms por 8 dp4a
      std::printf("  por (1 WMMA + 8 dp4a) por warp: misto %.5f ms | 1/8 de wmma %.5f + 1/8 de "
                  "dp4a %.5f = %.5f ms\n", per_step, wmma, dp4a, wmma + dp4a);
      std::printf("  leitura: misto/soma = %.2f (1.00 = dividem o mesmo slot de issue; <1 = "
                  "pipes separados); misto/max = %.2f\n", per_step / (wmma + dp4a),
                  per_step / std::max(wmma, dp4a));
    }
    {
      const double wmma_per_s = wi8_8.instr_per_s();
      const double dp4a_per_s = dp_8.instr_per_s();
      std::printf("  ciclos por WMMA int8 (U=8, melhor celula): por SIMD %.2f, por CU %.2f\n",
                  simd_slots / wmma_per_s, simd_slots / 4.0 / wmma_per_s);
      std::printf("  ciclos por dp4a (U=8): por SIMD %.3f, por CU %.3f\n", simd_slots / dp4a_per_s,
                  simd_slots / 4.0 / dp4a_per_s);
      std::printf("  MACs/ciclo/SIMD: WMMA int8 %.1f, WMMA f16 %.1f, dp4a %.2f; MACs/ciclo/CU: "
                  "%.0f / %.0f / %.1f\n", 4096.0 * wmma_per_s / simd_slots,
                  4096.0 * f16_8.instr_per_s() / simd_slots, 128.0 * dp4a_per_s / simd_slots,
                  4.0 * 4096.0 * wmma_per_s / simd_slots, 4.0 * 4096.0 * f16_8.instr_per_s() / simd_slots,
                  4.0 * 128.0 * dp4a_per_s / simd_slots);
      std::printf("  razoes de MACs/s: WMMA int8 / dp4a = %.2fx; WMMA f16 / dp4a = %.2fx; f16 acc "
                  "f32 / f16 acc f16 = %.2fx\n", wi8_8.macs_per_s() / dp_8.macs_per_s(),
                  f16_8.macs_per_s() / dp_8.macs_per_s(), f16_8.macs_per_s() / f16a_8.macs_per_s());
      std::printf("  piso de ruido (mesma celula, min de %d rodadas, em TAXA que e' o que nao\n"
                  "  depende do numero de iteracoes calibrado): wmma_i8 %.2f vs %.2f T-MACs/s "
                  "(%.3fx); dp4a %.2f vs %.2f (%.3fx)\n", rounds, wi8_8.macs_per_s() / 1e12,
                  wi8_8b.macs_per_s() / 1e12, wi8_8b.macs_per_s() / wi8_8.macs_per_s(),
                  dp_8.macs_per_s() / 1e12, dp_8b.macs_per_s() / 1e12,
                  dp_8b.macs_per_s() / dp_8.macs_per_s());
      std::printf("  taxa de instrucao: WMMA %.1f G-instr-warp/s, dp4a %.1f G-instr-warp/s "
                  "(razao %.2fx)\n", wmma_per_s / 1e9, dp4a_per_s / 1e9, dp4a_per_s / wmma_per_s);
      // O que o motor faz HOJE, em unidades comparaveis: 30,2e9 pesos por token
      // = 236e6 instrucoes de warp de dp4a por token (128 MACs cada). No lote de
      // N=16 o custo marginal e' 6,04 ms/token, i.e. 16 tokens em 96,6 ms
      // (docs/journal-lote.md, Medida 1).
      {
        const double eng_dp4a_per_s = 16.0 * 236e6 / (16.0 * 6.04e-3);
        const double eng_macs = eng_dp4a_per_s * 128.0;
        std::printf("  --- referencia do motor (docs/journal-lote.md, Medida 1) ---\n");
        std::printf("  16 tokens = 3,78e9 instr-warp de dp4a em 96,6 ms (custo marginal 6,04 "
                    "ms/token em N=16)\n");
        std::printf("  = %.1f G-instr-warp/s = %.2f T-MACs/s = %.1f TOPS-equivalente -> %.1f%% do "
                    "pico de dp4a e %.1f%% do pico de WMMA medidos aqui\n",
                    eng_dp4a_per_s / 1e9, eng_macs / 1e12, 2.0 * eng_macs / 1e12,
                    100.0 * eng_dp4a_per_s / dp4a_per_s, 100.0 * eng_macs / wi8_8.macs_per_s());
      }
    }
    HIPCHECK(hipFree(d_i));
    HIPCHECK(hipFree(d_f));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // 4. LDS
  // -----------------------------------------------------------------------
  if (do_lds) {
    std::printf("== 4. LDS: banda medida contra a que o staging do minigemm pede ==\n");
    const int threads = 256, blocks = 2 * cu, nthreads = blocks * threads;
    const size_t smem = 1024 * sizeof(int4);  // 16 KB (o kernel indexa & 1023)
    int *d_i = nullptr;
    HIPCHECK(hipMalloc(&d_i, (size_t)nthreads * sizeof(int)));
    std::function<void(int)> launch = [&](int iters) {
      lds_copy_kernel<<<blocks, threads, smem>>>(iters, d_i);
    };
    int iters = iters_override;
    if (!iters) {
      launch(100);
      HIPCHECK(hipDeviceSynchronize());
      EvPair ev;
      const double t1 = std::max(1e-6, ev([&] { launch(100); }) / 100.0);
      iters = (int)std::max(20.0, 3.0 / t1);
    }
    const double ms = best_ms([&] { launch(iters); }, reps, rounds);
    const double bytes = 2.0 * 16.0 * (double)nthreads * iters;  // 16 B lidos + 16 B escritos
    std::printf("  copia LDS->LDS (int4, iters=%d): %.3f ms, %.1f GB/s (lido+escrito), %.1f "
                "B/ciclo/CU\n", iters, bytes / (ms * 1e-3) / 1e9,
                bytes / (ms * 1e-3) / ((double)cu * clk_hz));
    std::printf("  modelo de 32 bancos x 4 B = 128 B/ciclo/CU = %.1f GB/s no clock derivado\n",
                128.0 * (double)cu * clk_hz / 1e9);
    HIPCHECK(hipFree(d_i));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // 5. EXATO
  // -----------------------------------------------------------------------
  if (do_exact) {
    std::printf("== 5. EXATO: int32 por bloco de 32, WMMA (2 x k16) vs dp4a (8 x k4) ==\n");
    const int K = 5120, nblk = K / 32;
    std::vector<int8_t> A, W;
    int *d_wa = nullptr, *d_dp = nullptr;
    int8_t *d_A = nullptr, *d_W = nullptr;
    HIPCHECK(hipMalloc(&d_wa, (size_t)nblk * 256 * sizeof(int)));
    HIPCHECK(hipMalloc(&d_dp, (size_t)nblk * 256 * sizeof(int)));
    HIPCHECK(hipMalloc(&d_A, (size_t)16 * K));
    HIPCHECK(hipMalloc(&d_W, (size_t)16 * K));
    std::vector<int> a((size_t)nblk * 256), b((size_t)nblk * 256);
    for (int extreme = 0; extreme < 2; ++extreme) {
      if (extreme) {
        fill_i8(A, (size_t)16 * K, -128, 127);
        for (size_t i = 0; i < A.size(); ++i) A[i] = (i & 1) ? (int8_t)127 : (int8_t)-128;
        W = A;
      } else {
        fill_i8(A, (size_t)16 * K, -100, 100);
        fill_i8(W, (size_t)16 * K, -100, 100);
      }
      HIPCHECK(hipMemcpy(d_A, A.data(), A.size(), hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_W, W.data(), W.size(), hipMemcpyHostToDevice));
      exact_block_kernel<0><<<1, 32>>>(d_A, d_W, d_wa, K);
      HIPCHECK(hipDeviceSynchronize());
      exact_block_kernel<1><<<1, 256>>>(d_A, d_W, d_dp, K);
      HIPCHECK(hipDeviceSynchronize());
      HIPCHECK(hipMemcpy(a.data(), d_wa, a.size() * sizeof(int), hipMemcpyDeviceToHost));
      HIPCHECK(hipMemcpy(b.data(), d_dp, b.size() * sizeof(int), hipMemcpyDeviceToHost));
      size_t mis = 0;
      int64_t mx = 0;
      for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != b[i]) {
          ++mis;
          mx = std::max<int64_t>(mx, std::llabs((int64_t)a[i] - (int64_t)b[i]));
        }
      std::printf("  %s (K=%d, %d blocos de 32 x 256 pares): divergencias WMMA-vs-dp4a %zu/%zu "
                  "(maior |d| %lld) [WMMA[0]=%d dp4a[0]=%d]\n",
                  extreme ? "extremo -128/+127" : "aleatorio -100..100", K, nblk, mis, a.size(),
                  (long long)mx, a[0], b[0]);
    }
    HIPCHECK(hipFree(d_wa));
    HIPCHECK(hipFree(d_dp));
    HIPCHECK(hipFree(d_A));
    HIPCHECK(hipFree(d_W));
    std::printf("\n");
  }

  // -----------------------------------------------------------------------
  // 6. GEMM
  // -----------------------------------------------------------------------
  if (do_gemm) {
    std::printf("== 6. GEMM: C[M][N] = A[M][K] . W[N][K] int8 + escala fp32 por bloco de 32 ==\n");
    const double llama_macs = 15.5e12 / 0.4596;  // 512 x 30,2e9 MACs em 459,6 ms
    std::printf("  referencia (mesmo cartao/modelo, docs/journal-lote.md): llama.cpp Vulkan pp512 "
                "= %.1f T-MACs/s (67 TOPS-equiv);\n  este motor = %.2f T-MACs/s (7,5 TOPS int8). "
                "TOPS = 2 x MACs.\n", llama_macs / 1e12, 15.5e12 / 4.13 / 1e12);
    std::printf("  %5s %6s %7s | %-13s %9s %7s %7s %7s | %-13s %9s %7s %7s %7s\n", "M", "N", "K",
                "int8 ms", "T-MACs/s", "TOPS", "%llama", "LDS/W", "f16 ms", "T-MACs/s", "TOPS",
                "%llama", "LDS/W");

    for (int K : gk) {
      const int nKD = K / 32;
      for (int Mi : gm) {
        for (int Ni : gn) {
          std::vector<int8_t> A, W;
          std::vector<float> da((size_t)Mi * nKD), dw((size_t)Ni * nKD);
          fill_i8(A, (size_t)Mi * K, -100, 100);
          fill_i8(W, (size_t)Ni * K, -100, 100);
          for (size_t i = 0; i < da.size(); ++i) da[i] = 0.0009f + 0.0001f * (float)(i % 7);
          for (size_t i = 0; i < dw.size(); ++i) dw[i] = 0.0007f + 0.0001f * (float)(i % 5);
          int8_t *d_A = nullptr, *d_W = nullptr;
          float *d_da = nullptr, *d_dw = nullptr, *d_C = nullptr;
          HIPCHECK(hipMalloc(&d_A, A.size()));
          HIPCHECK(hipMalloc(&d_W, W.size()));
          HIPCHECK(hipMalloc(&d_da, da.size() * sizeof(float)));
          HIPCHECK(hipMalloc(&d_dw, dw.size() * sizeof(float)));
          HIPCHECK(hipMalloc(&d_C, (size_t)Mi * Ni * sizeof(float)));
          HIPCHECK(hipMemcpy(d_A, A.data(), A.size(), hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpy(d_W, W.data(), W.size(), hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpy(d_da, da.data(), da.size() * sizeof(float), hipMemcpyHostToDevice));
          HIPCHECK(hipMemcpy(d_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice));

          const double macs = (double)Mi * Ni * K;
          std::function<void()> li8, lf16;
          switch (Mi) {
            case 16:
              li8 = [&] { gemm_kernel<16, 4, false><<<dim3((Ni + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              lf16 = [&] { gemm_kernel<16, 4, true><<<dim3((Ni + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              break;
            case 32:
              li8 = [&] { gemm_kernel<32, 4, false><<<dim3((Ni + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              lf16 = [&] { gemm_kernel<32, 4, true><<<dim3((Ni + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              break;
            case 64:
              li8 = [&] { gemm_kernel<64, 4, false><<<dim3((Ni + 63) / 64, (Mi + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              lf16 = [&] { gemm_kernel<64, 4, true><<<dim3((Ni + 63) / 64, (Mi + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              break;
            default:
              li8 = [&] { gemm_kernel<128, 8, false><<<dim3((Ni + 63) / 64, (Mi + 127) / 128), dim3(256)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              lf16 = [&] { gemm_kernel<128, 8, true><<<dim3((Ni + 63) / 64, (Mi + 127) / 128), dim3(256)>>>(d_A, d_da, d_W, d_dw, d_C, Mi, Ni, K); };
              break;
          }
          const double ms_i8 = best_ms(li8, reps, rounds);
          const double ms_f16 = best_ms(lf16, reps, rounds);
          // LDS por WMMA (ver docs): A 8 B por lane + os fragmentos de escala
          const double lds_per_wmma = 3712.0 / 8.0;
          std::printf("  %5d %6d %7d | %-13.4f %9.2f %7.1f %6.1f%% %7.0f | %-13.4f %9.2f %7.1f "
                      "%6.1f%% %7.0f\n", Mi, Ni, K, ms_i8, macs / (ms_i8 * 1e-3) / 1e12,
                      2 * macs / (ms_i8 * 1e-3) / 1e12, 100.0 * (macs / (ms_i8 * 1e-3)) / llama_macs,
                      lds_per_wmma, ms_f16, macs / (ms_f16 * 1e-3) / 1e12,
                      2 * macs / (ms_f16 * 1e-3) / 1e12, 100.0 * (macs / (ms_f16 * 1e-3)) / llama_macs,
                      lds_per_wmma);
          hipFree(d_A);
          hipFree(d_W);
          hipFree(d_da);
          hipFree(d_dw);
          hipFree(d_C);
        }
      }
    }

    // --- corretude: M=16, N=128, K=5120 ---
    {
      const int M = 16, N = 128, K = 5120, nKD = K / 32;
      std::vector<int8_t> A, W;
      std::vector<float> da((size_t)M * nKD), dw((size_t)N * nKD);
      fill_i8(A, (size_t)M * K, -100, 100);
      fill_i8(W, (size_t)N * K, -100, 100);
      for (size_t i = 0; i < da.size(); ++i) da[i] = 0.0009f + 0.0001f * (float)(i % 7);
      for (size_t i = 0; i < dw.size(); ++i) dw[i] = 0.0007f + 0.0001f * (float)(i % 5);
      int8_t *d_A = nullptr, *d_W = nullptr;
      float *d_da = nullptr, *d_dw = nullptr, *d_C = nullptr;
      HIPCHECK(hipMalloc(&d_A, A.size()));
      HIPCHECK(hipMalloc(&d_W, W.size()));
      HIPCHECK(hipMalloc(&d_da, da.size() * sizeof(float)));
      HIPCHECK(hipMalloc(&d_dw, dw.size() * sizeof(float)));
      HIPCHECK(hipMalloc(&d_C, (size_t)M * N * sizeof(float)));
      HIPCHECK(hipMemcpy(d_A, A.data(), A.size(), hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_W, W.data(), W.size(), hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_da, da.data(), da.size() * sizeof(float), hipMemcpyHostToDevice));
      HIPCHECK(hipMemcpy(d_dw, dw.data(), dw.size() * sizeof(float), hipMemcpyHostToDevice));
      std::vector<float> ref, got((size_t)M * N, 0.0f);
      gemm_ref_i8(A, da, W, dw, M, N, K, ref);
      gemm_kernel<16, 4, false><<<dim3((N + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, M, N, K);
      HIPCHECK(hipDeviceSynchronize());
      HIPCHECK(hipMemcpy(got.data(), d_C, got.size() * sizeof(float), hipMemcpyDeviceToHost));
      size_t bits = 0, far = 0;
      for (size_t i = 0; i < ref.size(); ++i) {
        if (std::memcmp(&ref[i], &got[i], 4) != 0) ++bits;
        if (std::fabs(ref[i] - got[i]) > 1e-4f * std::fabs(ref[i])) ++far;
      }
      std::printf("  corretude int8 (M=%d N=%d K=%d): elementos com bits diferentes da referencia "
                  "%zu/%zu, fora de 1e-4 rel %zu, rel-L2 %.2e\n", M, N, K, bits, ref.size(), far,
                  rel_l2(ref, got));
      std::vector<float> ref16;
      gemm_ref_f16(A, da, W, dw, M, N, K, ref16);
      gemm_kernel<16, 4, true><<<dim3((N + 63) / 64), dim3(128)>>>(d_A, d_da, d_W, d_dw, d_C, M, N, K);
      HIPCHECK(hipDeviceSynchronize());
      HIPCHECK(hipMemcpy(got.data(), d_C, got.size() * sizeof(float), hipMemcpyDeviceToHost));
      double mxd = 0;
      for (size_t i = 0; i < ref16.size(); ++i) mxd = std::max(mxd, std::fabs((double)ref16[i] - got[i]));
      std::printf("  corretude f16  (M=%d N=%d K=%d): rel-L2 vs fp64 dos pesos f16 %.2e, |d| max "
                  "%.3g\n", M, N, K, rel_l2(ref16, got), mxd);
      std::printf("  f16 vs int8 nos mesmos dados: rel-L2 %.2e (o desvio que a via f16 introduz)\n",
                  rel_l2(ref, got));
      hipFree(d_A);
      hipFree(d_W);
      hipFree(d_da);
      hipFree(d_dw);
      hipFree(d_C);
    }
    std::printf("\n");
  }

  // piso de lancamento: separa custo de kernel de custo de despacho
  {
    auto launch = [] {
      for (int i = 0; i < 100; ++i) empty_kernel<<<16, 32>>>();
    };
    const double ms = best_ms(launch, 5, 3, 100.0);
    std::printf("piso de lancamento: kernel vazio %.4f ms por lancamento (100 por medicao)\n",
                ms / 100.0);
  }
  return 0;
}
