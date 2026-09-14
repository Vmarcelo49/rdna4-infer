// ===========================================================================
// Frente H do estudo de prefill (docs/plano-prefill.md, degrau D4): MMQ int8
// com o WMMA 16x16x16 do gfx1201 sobre os BLOCOS REAIS do modelo (iq3_s), com
// a ativacao no layout `block_q8_1` do motor e a correcao de escala por bloco
// de 32 identica a do `vec_dot_iq3_s_q8_1` (vecdotq.cuh:692-729).
//
// Por que esta bancada existe (a pergunta que a frente D deixou aberta):
// a `bench-wmma-gpu` mediu o PICO (182 T-MAC/s o WMMA int8 contra 44 do dp4a,
// 4,2x) e a `bench-gemm-quant-gpu` mediu o caminho VETORIAL com os pesos reais
// (9,2-11,7 T-MAC/s com staging f16, 42 % do tempo em staging). Ninguem tinha
// rodado o WMMA int8 sobre os blocos reais, e e' isso que decide se o teto de
// 4,2x se converte em prefill.
//
// Tres resultados, nesta ordem de prioridade:
//   1. CORRETUDE: oraculo de CPU (algebra do dequant do motor, somas em int
//      exato, FMA correto) e o `vec_dot_iq3_s_q8_1` DO MOTOR como referencia de
//      GPU, num caso pequeno, com contagem de bytes/elementos divergentes;
//   2. BIT-EXATIDAO: o MESMO staging int8 e a MESMA sequencia de correcao, com
//      o laco interno trocado (WMMA x dp4a) -- quantos elementos de C diferem;
//   3. VELOCIDADE: T-MAC/s em M=16/64/128/512 com a correcao fp32 por bloco de
//      32 (a matematica correta), a fracao do pico medido NA MESMA JANELA e
//      onde o tempo vai (staging / correcao / matrix core), por modos do MESMO
//      kernel com a mesma grade.
//
// Regra da casa (docs/estudo-prefill.md secao 3b): a bancada anterior reportou
// 1198 T-MAC/s porque o kernel NAO RODAVA. Aqui: hipGetLastError() depois de
// todo lancamento, verificacao contra oraculo ANTES de cronometrar, checagem
// de que a saida nao e' toda zero, aquecimento de DPM antes do laco e minimo
// de `reps` repeticoes. A placa e' compartilhada: as razoes dentro da janela
// valem mais que os absolutos entre janelas, e o pico e' remedido aqui dentro.
//
// Uso:
//   bench-mmq-wmma-gpu <model.gguf> [--m 16,64,128,512] [--cfg ID] [--reps 5]
//                      [--tensor NOME] [--n N] [--k K] [--bk N] [--list]
//                      [--no-verify] [--dbuf] [--peak-only]
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

#include "rdna4/dequant.cuh"  // dequantize_iq3_s (oraculo de GPU da frente F)
#include "rdna4/fp16.h"
#include "rdna4/loader.h"
#include "rdna4/matvec.cuh"  // quantize_q8_1_block / _batch_kernel: a ativacao DO MOTOR
#include "rdna4/quants.h"
#include "rdna4/quant_tables.h"
#include "rdna4/vecdotq.cuh"  // vec_dot_iq3_s_q8_1: a referencia bit a bit

// O alvo `bench-mmq-wmma-gpu` linka so' a libamdhip64 (CMakeLists:210-217) e o
// CMakeLists nao esta' no escopo desta frente, entao as duas fontes do loader
// entram por include direto -- a bancada usa o GgufLoader DE VERDADE em vez de
// duplicar um parser de GGUF.
#include "../src/backend/gguf.cpp"
#include "../src/backend/loader.cpp"

namespace {

using rdna4::block_q8_1;
using rdna4::block_iq3_s;

constexpr int kQK = 256;   // pesos por super-bloco iq3_s
constexpr int kQ8 = 32;    // QK8_1: valores por bloco q8_1
constexpr int kQBytes = 36;  // sizeof(block_q8_1)
constexpr int kBlockBytes = 110;
constexpr int kIq3S = 12;  // ggml_type id (ver matvec.cuh dt_name)

// ---------------------------------------------------------------------------
// 1. Fragmentos do WMMA gfx12 wave32
// ---------------------------------------------------------------------------
// Receita do operando (docs/estudo-prefill-d-wmma.md secao 1, varredura de 32
// hipoteses + probe byte a byte): para um tile 16x16x16,
//   A: byte p do lane L = A[m = L%16][k = 8*(L/16) + p]
//   B: byte p do lane L = Bn[n = L%16][k = 8*(L/16) + p]      (linha = n)
//   D: slot j do lane L = C[m = 8*(L/16) + j][n = L%16]       (TRANSPOSTO de A/B)
// e o bool de sinalidade e' `true` para os dois operandos (medido com
// A=0xFF,B=+1 -> -16). O ultimo argumento (clamp) fica `true` como no bench que
// validou a receita; os acumuladores sao zerados a cada bloco de 32, entao
// |sumi| <= 32*15*127 = 60960 e nao ha' saturacao possivel.
typedef int i32x2_t __attribute__((ext_vector_type(2)));
typedef int i32x8_t __attribute__((ext_vector_type(8)));

__device__ __forceinline__ i32x8_t wmma_i8(const i32x2_t a, const i32x2_t b, const i32x8_t c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, true);
}
__device__ __forceinline__ i32x8_t zero_i32x8() {
  i32x8_t z;
#pragma unroll
  for (int i = 0; i < 8; ++i) z[i] = 0;
  return z;
}
// Leitura de fragmento (8 B) da LDS: alinhada a 8 B por construcao (stride de
// linha BK+8 e offset 16*kk + 8*(L/16)).
__device__ __forceinline__ i32x2_t lds_frag(const void *p) {
  return *reinterpret_cast<const i32x2_t *>(p);
}

// ---------------------------------------------------------------------------
// 2. Staging: bloco iq3_s real -> int8 ASSINADO na LDS, na ordem do WMMA
// ---------------------------------------------------------------------------
// Um sub-bloco de 32 pesos (s = 0..7 do super-bloco) vira 8 palavras int32 (32
// bytes) em ordem de k. A algebra e' EXATAMENTE a do `vec_dot_iq3_s_q8_1`
// (vecdotq.cuh:697-725), que por sua vez e' a do `stage_iq3s_32` da frente F:
//   qs[8s .. 8s+8)     8 palavras de grade (4 magnitudes cada)
//   qh[s]              bit alto por palavra de grade
//   signs[4s .. 4s+4)  8 bits de sinal (2 por palavra de grade)
//   scales[s/2]        nibble 4*(s%2)  ->  o fator (1 + 2*sc) = 1..31
// O sinal entra pelo truque de 2 instrucoes por 4 bytes (__vcmpne4 + __vsub4).
// O fator (1+2*sc) NAO cabe no int8 (15*31 = 465), entao ele sai daqui como
// INTEIRO separado e e' aplicado no int32 do bloco, antes do fp32 -- e' o que
// docs/estudo-prefill-d-wmma.md secao 5 exige para a bit-exatidao.
__device__ __forceinline__ void stage_iq3s_32_i8(const block_iq3_s *__restrict__ blk, const int s,
                                                 int *__restrict__ dst, float *dw, int *scf) {
  const unsigned char *qs = blk->qs + 8 * s;
  const unsigned char *sg = blk->signs + 4 * s;
  const int qh = blk->qh[s];
  *dw = rdna4::fp16_to_float(blk->d);
  *scf = 1 + 2 * ((blk->scales[s / 2] >> (4 * (s % 2))) & 0x0F);
#pragma unroll
  for (int il = 0; il < 4; ++il) {
    const int g1 = rdna4::iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)];
    const int g2 = rdna4::iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)];
    const int m0 = rdna4::__vcmpne4(((sg[il] & 0x03) << 7) | ((sg[il] & 0x0C) << 21), 0);
    const int m1 = rdna4::__vcmpne4(((sg[il] & 0x30) << 3) | ((sg[il] & 0xC0) << 17), 0);
    dst[2 * il + 0] = rdna4::__vsub4(g1 ^ m0, m0);  // k = 8*il + [0,4)
    dst[2 * il + 1] = rdna4::__vsub4(g2 ^ m1, m1);  // k = 8*il + [4,8)
  }
}

// ---------------------------------------------------------------------------
// 3. O GEMM tilejado com o laco interno em WMMA int8
// ---------------------------------------------------------------------------
// Grade: x -> N (BN por CTA), y -> M (BM por CTA). Warps em WMW x WNK; cada
// warp cobre MT = BM/(16*WMW) tiles de 16 em M e NT = BN/(16*WNK) em N, com 8
// acumuladores int32 + 8 fp32 por tile.
//
// LDS por janela de BK:
//   s_w[BN][BK+8]  int8   peso (linha = n, coluna = k), o operando B do WMMA
//   s_a[BM][BK+8]  int8   ativacao q8_1, o operando A
//   s_dwsc[NKB*BN] int2   x = bits de d_w, y = 1+2*sc   (por n, por bloco de 32)
//   s_da[NKB*BM]   float  d_a = fp16(block_q8_1.ds & 0xffff)  (por m, por bloco)
// O stride BK+8 (8 B alinhado; 18 palavras com BK=64, e 18 e' par -> os 16 lanes
// de linha caem em 16 bancos distintos) evita conflito nas leituras de
// fragmento. d_w/d_a ficam TRANSPOSTOS (bloco de k maior, linha menor) de
// proposito: e' o que faz a correcao custar 8 floats consecutivos (2 LDS.128) +
// 1 int2 por lane, e nao um gather -- consequencia direta de D ser a transposta
// de A/B (doc D secao 1).
//
// MODE (mesmo kernel, mesma grade, so' o que vem depois do staging muda -- e' o
// que permite atribuir a diferenca ao staging e nao a outro knob):
//   0 = GEMM completo; 1 = staging (A + peso); 2 = so' a ativacao; 3 = so' o peso
// `nocorr` desliga a correcao fp32 mantendo WMMA + leituras de fragmento: e' a
// medicao que separa o matrix core da correcao (t_gemm - t_nocorr).
template <int BM, int BN, int BK, int WMW, int WNK, bool DBUF>
__global__ void __launch_bounds__(WMW * WNK * 32) gemm_wmma_i8_kernel(
    const block_iq3_s *__restrict__ W, const block_q8_1 *__restrict__ AQ, float *__restrict__ C,
    const int M, const int N, const int K, const int mode, const int nocorr) {
  constexpr int MT = BM / (16 * WMW);  // tiles 16x16 por warp em M
  constexpr int NT = BN / (16 * WNK);  // tiles 16x16 por warp em N
  constexpr int NTH = WMW * WNK * 32;
  constexpr int NKB = BK / 32;  // blocos de 32 por janela de k
  constexpr int WSB = BK + 8;   // stride de linha das duas LDS int8
  static_assert(BK % 32 == 0, "BK tem de ser multiplo de 32 (o bloco da escala)");
  static_assert(BN % (16 * WNK) == 0 && BM % (16 * WMW) == 0, "");
  static_assert(MT * NT <= 4, "16 acumuladores por tile: no maximo 4 tiles por lane");
  (void)M;

  __shared__ __align__(16) unsigned char s_w[DBUF ? 2 : 1][BN * WSB];
  __shared__ __align__(16) unsigned char s_a[DBUF ? 2 : 1][BM * WSB];
  __shared__ __align__(16) int2 s_dwsc[DBUF ? 2 : 1][NKB * BN];
  __shared__ __align__(16) float s_da[DBUF ? 2 : 1][NKB * BM];

  const int tid = threadIdx.x;
  const int L = tid & 31;
  const int warp = tid >> 5;
  const int wm = (warp % WMW) * 16 * MT;  // base de m do warp dentro do tile
  const int wn = (warp / WMW) * 16 * NT;  // base de n do warp dentro do tile
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int bpr = K / kQK;   // super-blocos iq3_s por linha
  const int nkbt = K / kQ8;  // blocos q8_1 por linha
  const int lo = L & 15, hi = L >> 4;
  const int row_a = wm + lo;       // linha de A que este lane carrega
  const int row_b = wn + lo;       // linha de B (n) que este lane carrega
  const int da_row = wm + 8 * hi;  // primeira das 8 linhas de d_a do lane
  const int n_lane = wn + lo;      // n do lane dentro do primeiro tile

  // ---- staging de uma janela de BK ----------------------------------------
  auto stage_tile = [&](const int buf, const int k0) {
    // (a) peso real: iq3_s -> int8 na LDS (1 sub-bloco de 32 por tarefa)
    for (int task = tid; task < BN * NKB; task += NTH) {
      const int row = task / NKB;
      const int kb = task - row * NKB;
      const int k0g = k0 + kb * 32;
      float dw;
      int scf;
      stage_iq3s_32_i8(W + (std::int64_t)(n0 + row) * bpr + (k0g >> 8), (k0g >> 5) & 7,
                       (int *)(s_w[buf] + row * WSB + kb * 32), &dw, &scf);
      s_dwsc[buf][kb * BN + row] = make_int2(__float_as_int(dw), scf);
    }
    // (b) ativacao q8_1 -> LDS: 32 bytes contiguos por (linha, bloco de 32), uma
    // palavra de 4 B por tarefa (escritas consecutivas dentro do warp).
    for (int task = tid; task < BM * NKB * 8; task += NTH) {
      const int row = task / (NKB * 8);
      const int r2 = task - row * (NKB * 8);
      const int kb = r2 >> 3;
      const int i = r2 & 7;
      const block_q8_1 *q = AQ + (std::int64_t)(m0 + row) * nkbt + (k0 >> 5) + kb;
      ((int *)(s_a[buf] + row * WSB + kb * 32))[i] = ((const int *)q->qs)[i];
    }
    for (int task = tid; task < BM * NKB; task += NTH) {
      const int row = task / NKB;
      const int kb = task - row * NKB;
      const block_q8_1 *q = AQ + (std::int64_t)(m0 + row) * nkbt + (k0 >> 5) + kb;
      s_da[buf][kb * BM + row] = rdna4::fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
    }
  };
  // variante minima (modo 2/3), para o split A x W da contabilidade
  auto stage_a_only = [&](const int buf, const int k0) {
    for (int task = tid; task < BM * NKB * 8; task += NTH) {
      const int row = task / (NKB * 8);
      const int r2 = task - row * (NKB * 8);
      const int kb = r2 >> 3;
      const int i = r2 & 7;
      const block_q8_1 *q = AQ + (std::int64_t)(m0 + row) * nkbt + (k0 >> 5) + kb;
      ((int *)(s_a[buf] + row * WSB + kb * 32))[i] = ((const int *)q->qs)[i];
    }
  };
  auto stage_w_only = [&](const int buf, const int k0) {
    for (int task = tid; task < BN * NKB; task += NTH) {
      const int row = task / NKB;
      const int kb = task - row * NKB;
      const int k0g = k0 + kb * 32;
      float dw;
      int scf;
      stage_iq3s_32_i8(W + (std::int64_t)(n0 + row) * bpr + (k0g >> 8), (k0g >> 5) & 7,
                       (int *)(s_w[buf] + row * WSB + kb * 32), &dw, &scf);
      s_dwsc[buf][kb * BN + row] = make_int2(__float_as_int(dw), scf);
    }
  };

  i32x8_t D[MT][NT];
  float F[MT][NT][8];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      D[mt][t] = zero_i32x8();
#pragma unroll
      for (int j = 0; j < 8; ++j) F[mt][t][j] = 0.0f;
    }
  float sink = 0.0f;

  for (int k0 = 0, buf = 0; k0 < K; k0 += BK, buf ^= 1) {
    // ATENCAO: com DBUF=false existe UM buffer, entao o indice tem de ser 0 --
    // deixar `buf` alternar aqui escreveria fora do array de LDS (bug real
    // desta bancada, pego pela verificacao antes de qualquer numero).
    const int cb = DBUF ? buf : 0;
    if constexpr (DBUF) {
      // Pipeline de software: o staging de k+BK e' emitido ANTES da conta de k,
      // e as duas barreiras por iteracao viram uma.
      if (k0 + BK < K) stage_tile(cb ^ 1, k0 + BK);
    }
    if (mode == 0 || mode == 1) {
      stage_tile(cb, k0);
    } else if (mode == 2) {
      stage_a_only(cb, k0);
    } else {
      stage_w_only(cb, k0);
    }
    __syncthreads();

    if (mode == 0) {
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        i32x2_t af[MT], bf[NT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
          af[mt] = lds_frag(s_a[cb] + (row_a + 16 * mt) * WSB + 16 * kk + 8 * hi);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
          bf[t] = lds_frag(s_w[cb] + (row_b + 16 * t) * WSB + 16 * kk + 8 * hi);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
          for (int t = 0; t < NT; ++t) D[mt][t] = wmma_i8(af[mt], bf[t], D[mt][t]);

        if ((kk & 1) == 1) {  // fecha o bloco de 32: correcao de escala
          const int kb = kk >> 1;
          if (nocorr) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
              for (int t = 0; t < NT; ++t) {
#pragma unroll
                for (int j = 0; j < 8; ++j) F[mt][t][j] += (float)D[mt][t][j];
              }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
              for (int t = 0; t < NT; ++t) D[mt][t] = zero_i32x8();
          } else {
            float daq[MT][8];
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
              const float *p = s_da[cb] + kb * BM + da_row + 16 * mt;
              const float4 v0 = *(const float4 *)p;        // 8 floats consecutivos
              const float4 v1 = *(const float4 *)(p + 4);  // = 2 LDS.128 por lane
              daq[mt][0] = v0.x; daq[mt][1] = v0.y; daq[mt][2] = v0.z; daq[mt][3] = v0.w;
              daq[mt][4] = v1.x; daq[mt][5] = v1.y; daq[mt][6] = v1.z; daq[mt][7] = v1.w;
            }
#pragma unroll
            for (int t = 0; t < NT; ++t) {
              const int2 ws = s_dwsc[cb][kb * BN + n_lane + 16 * t];
              const float dw = __int_as_float(ws.x);
              const int scf = ws.y;
#pragma unroll
              for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                  // a MESMA sequencia do motor (vecdotq.cuh:725-728):
                  // sumi *= 1+2*sc em INTEIRO, depois d = d_w*d_a em fp32 e
                  // acc += d*sumi -- com o FMA fixado por __fmaf_rn.
                  const float d = dw * daq[mt][j];
                  F[mt][t][j] = __fmaf_rn(d, (float)(D[mt][t][j] * scf), F[mt][t][j]);
                }
            }
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
              for (int t = 0; t < NT; ++t) D[mt][t] = zero_i32x8();
          }
        }
      }
    } else {
      // Consumo minimo dependente de `tid`: o compilador nao consegue provar
      // que a LDS nao e' necessaria, entao o staging tem de ser materializado.
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        if (mode != 3) {
          const int ra = (tid * 5 + i * 17) % BM, ka = (tid * 13 + i * 3) % BK;
          s += (float)(signed char)s_a[cb][ra * WSB + ka];
        }
        if (mode != 2) {
          const int rw = (tid * 7 + i * 29) % BN, kw = (tid * 11 + i * 5) % BK;
          s += (float)(signed char)s_w[cb][rw * WSB + kw];
        }
      }
      sink += s;
    }
    if (!DBUF) __syncthreads();
  }

  if (mode == 0) {
    // C no layout do acumulador (D transposto): cada lane escreve 8 linhas de m
    // na mesma coluna n; os 16 lanes de n consecutivo escrevem 64 B contiguos.
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const int n = n0 + n_lane + 16 * t;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int m = m0 + wm + 8 * hi + j + 16 * mt;
          C[(std::int64_t)m * N + n] = F[mt][t][j];
        }
      }
  } else if (__float_as_uint(sink) == 0x7F800001u) {
    C[0] = sink;  // guarda impossivel: mantem a soma viva sem escrever C
  }
}

// ---------------------------------------------------------------------------
// 4. O MESMO staging, laco interno em dp4a (a comparacao bit a bit e de taxa)
// ---------------------------------------------------------------------------
// Definicao de "semantica do motor" usada aqui: os mesmos bytes int8 na LDS
// (mesma algebra de dequant), a mesma ordem em k, o mesmo `sumi *= 1+2*sc` em
// inteiro e o mesmo `d = d_w*d_a` em fp32 com FMA fixado. A UNICA diferenca
// para o kernel do item 3 e' a instrucao de MAC: `sudot4` (4 MAC/lane) contra o
// WMMA (128 MAC/lane). Tudo o mais -- staging, correcao, ordem das somas -- e'
// identico, entao a contagem de elementos diferentes isola a aritmetica.
template <int BM, int BN, int BK, int RM, int RN, bool DBUF>
__global__ void __launch_bounds__((BM / RM) * (BN / RN)) gemm_dp4a_i8_kernel(
    const block_iq3_s *__restrict__ W, const block_q8_1 *__restrict__ AQ, float *__restrict__ C,
    const int M, const int N, const int K, const int mode, const int nocorr) {
  constexpr int TM = BM / RM;  // threads na dimensao M
  constexpr int TN = BN / RN;  // threads na dimensao N
  constexpr int NTH = TM * TN;
  constexpr int NKB = BK / 32;
  constexpr int WSB = BK + 8;
  static_assert(BK % 32 == 0, "");
  (void)M;

  __shared__ __align__(16) unsigned char s_w[DBUF ? 2 : 1][BN * WSB];
  __shared__ __align__(16) unsigned char s_a[DBUF ? 2 : 1][BM * WSB];
  __shared__ __align__(16) int2 s_dwsc[DBUF ? 2 : 1][NKB * BN];
  __shared__ __align__(16) float s_da[DBUF ? 2 : 1][NKB * BM];

  const int tid = threadIdx.x;
  const int tr = tid / TN;  // linha de A do thread
  const int tc = tid % TN;  // linha de W (n) do thread
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int bpr = K / kQK;
  const int nkbt = K / kQ8;

  auto stage_tile = [&](const int buf, const int k0) {
    for (int task = tid; task < BN * NKB; task += NTH) {
      const int row = task / NKB;
      const int kb = task - row * NKB;
      const int k0g = k0 + kb * 32;
      float dw;
      int scf;
      stage_iq3s_32_i8(W + (std::int64_t)(n0 + row) * bpr + (k0g >> 8), (k0g >> 5) & 7,
                       (int *)(s_w[buf] + row * WSB + kb * 32), &dw, &scf);
      s_dwsc[buf][kb * BN + row] = make_int2(__float_as_int(dw), scf);
    }
    for (int task = tid; task < BM * NKB * 8; task += NTH) {
      const int row = task / (NKB * 8);
      const int r2 = task - row * (NKB * 8);
      const int kb = r2 >> 3;
      const int i = r2 & 7;
      const block_q8_1 *q = AQ + (std::int64_t)(m0 + row) * nkbt + (k0 >> 5) + kb;
      ((int *)(s_a[buf] + row * WSB + kb * 32))[i] = ((const int *)q->qs)[i];
    }
    for (int task = tid; task < BM * NKB; task += NTH) {
      const int row = task / NKB;
      const int kb = task - row * NKB;
      const block_q8_1 *q = AQ + (std::int64_t)(m0 + row) * nkbt + (k0 >> 5) + kb;
      s_da[buf][kb * BM + row] = rdna4::fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
    }
  };

  float F[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) F[r][c] = 0.0f;
  float sink = 0.0f;

  for (int k0 = 0, buf = 0; k0 < K; k0 += BK, buf ^= 1) {
    const int cb = DBUF ? buf : 0;  // idem: um buffer quando DBUF=false
    if constexpr (DBUF) {
      if (k0 + BK < K) stage_tile(cb ^ 1, k0 + BK);
    }
    stage_tile(cb, k0);
    __syncthreads();

    if (mode == 0) {
#pragma unroll
      for (int kb = 0; kb < NKB; ++kb) {
        int I[RM][RN];
#pragma unroll
        for (int r = 0; r < RM; ++r)
#pragma unroll
          for (int c = 0; c < RN; ++c) I[r][c] = 0;
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          int a[RM], w[RN];
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            a[r] = ((const int *)(s_a[cb] + (tr + r * TM) * WSB + kb * 32))[g];
          }
#pragma unroll
          for (int c = 0; c < RN; ++c) {
            w[c] = ((const int *)(s_w[cb] + (tc + c * TN) * WSB + kb * 32))[g];
          }
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) I[r][c] = rdna4::ggml_cuda_dp4a(w[c], a[r], I[r][c]);
        }
        if (nocorr) {
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) F[r][c] += (float)I[r][c];
        } else {
          float da[RM];
#pragma unroll
          for (int r = 0; r < RM; ++r) da[r] = s_da[cb][kb * BM + tr + r * TM];
          int2 ws[RN];
#pragma unroll
          for (int c = 0; c < RN; ++c) ws[c] = s_dwsc[cb][kb * BN + tc + c * TN];
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) {
              const float d = __int_as_float(ws[c].x) * da[r];
              F[r][c] = __fmaf_rn(d, (float)(I[r][c] * ws[c].y), F[r][c]);
            }
        }
      }
    } else {
      float s = 0.0f;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int ra = (tid * 5 + i * 17) % BM, ka = (tid * 13 + i * 3) % BK;
        s += (float)(signed char)s_a[cb][ra * WSB + ka];
        const int rw = (tid * 7 + i * 29) % BN, kw = (tid * 11 + i * 5) % BK;
        s += (float)(signed char)s_w[cb][rw * WSB + kw];
      }
      sink += s;
    }
    if (!DBUF) __syncthreads();
  }

  if (mode == 0) {
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        C[(std::int64_t)(m0 + tr + r * TM) * N + n0 + tc + c * TN] = F[r][c];
      }
  } else if (__float_as_uint(sink) == 0x7F800001u) {
    C[0] = sink;
  }
}

// ---------------------------------------------------------------------------
// 5. A referencia do MOTOR: o `vec_dot_iq3_s_q8_1` de verdade, um thread por
//    elemento de saida, na mesma ordem em k do kernel tilejado
// ---------------------------------------------------------------------------
// Nao ha' reescrita aqui: e' a funcao do motor (vecdotq.cuh:692) chamada com
// `kbx = ib` (super-bloco) e `iqs = 2*s` -- a convencao de quem chama dentro do
// motor (matvec.cuh:384 passa kqs = vdr*(tg%(qi/vdr)) = 2*(tg%8)). O `acc +=`
// fica na forma do motor: se o compilador contrai para FMA, a comparacao com o
// kernel do item 3 (que fixa __fmaf_rn) vira comparacao de INSTRUCAO (conferida
// na ISA), nao de sorte.
__global__ void ref_engine_kernel(const block_iq3_s *__restrict__ W,
                                  const block_q8_1 *__restrict__ AQ, float *__restrict__ C,
                                  const int M, const int N, const int K) {
  const std::int64_t out = (std::int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (out >= (std::int64_t)M * N) return;
  const int m = (int)(out / N), n = (int)(out % N);
  const int bpr = K / kQK;
  const block_iq3_s *row = W + (std::int64_t)n * bpr;
  const block_q8_1 *q = AQ + (std::int64_t)m * (K / kQ8);
  float acc = 0.0f;
  for (int ib = 0; ib < bpr; ++ib) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
      acc += rdna4::vec_dot_iq3_s_q8_1((const void *)row, q + ib * 8, ib, 2 * s);
    }
  }
  C[out] = acc;
}

// Deposito do staging ISOLADO: mesmo `stage_iq3s_32_i8` do kernel, escrevendo na
// global em vez da LDS -- separa "o staging esta' certo" de "o GEMM esta' certo".
__global__ void dump_stage_i8_kernel(const block_iq3_s *__restrict__ W, int *__restrict__ out,
                                     const int nrow, const int bpr, const int nsub) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= nsub) return;
  int tmp[8];
  float dw;
  int scf;
  stage_iq3s_32_i8(W + (std::int64_t)nrow * bpr + (t / 8), t % 8, tmp, &dw, &scf);
  int *o = out + (std::size_t)t * 9;
#pragma unroll
  for (int i = 0; i < 8; ++i) o[i] = tmp[i];
  o[8] = (__float_as_int(dw) & 0x00FFFFFF) | (scf << 24);  // pacote p/ o check
}

// Copia a LUT do DEVICE: o oraculo de CPU usa a MESMA tabela que o kernel.
__global__ void copy_lut_kernel(uint32_t *__restrict__ out, const int n) {
  for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) out[i] = rdna4::iq3s_grid[i];
}

// ---------------------------------------------------------------------------
// 6. Pico remedido NA MESMA JANELA (o normalizador do "% do pico")
// ---------------------------------------------------------------------------
// 8 acumuladores independentes, U=8: o mesmo desenho da `bench-wmma-gpu` que
// mediu 182 T-MAC/s. Sem isto o "% do pico" compararia janelas diferentes, e a
// placa e' compartilhada (a frente D viu 1,5-2x de variacao entre corridas).
template <int U, int NACC, int WHICH>
__global__ void peak_loop_kernel(int iters, int *out) {
  const int L = threadIdx.x & 31;
  int s = 0;
  if constexpr (WHICH == 0) {
    i32x8_t acc[NACC];
    i32x2_t a[NACC], b[NACC];
#pragma unroll
    for (int i = 0; i < NACC; ++i) {
      acc[i] = zero_i32x8();
      a[i][0] = 0x01010101 * (L + 1 + i);
      a[i][1] = 0x01020304 * (i + 1);
      b[i][0] = 0x7f7f7f7f - i;
      b[i][1] = 0x01010101 * (L + 3);
    }
    for (int it = 0; it < iters; ++it) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
#pragma unroll
        for (int i = 0; i < NACC; ++i) acc[i] = wmma_i8(a[i], b[i], acc[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < NACC; ++i)
#pragma unroll
      for (int j = 0; j < 8; ++j) s += acc[i][j];
  } else {
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
        for (int i = 0; i < NACC; ++i) acc[i] = rdna4::ggml_cuda_dp4a(a[i], b[i], acc[i]);
      }
    }
#pragma unroll
    for (int i = 0; i < NACC; ++i) s += acc[i];
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

// ---------------------------------------------------------------------------
// 7. Utilitarios de host
// ---------------------------------------------------------------------------
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

bool launch_ok(const char *what) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) {
    std::printf("  LAUNCH FALHOU (%s): %s\n", what, hipGetErrorString(e));
    return false;
  }
  return true;
}

// Ativacao float: rms ~1,9, ordem de grandeza das ativacoes reais.
void fill_a(std::vector<float> &h, const std::size_t n, const unsigned seed) {
  h.resize(n);
  unsigned s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = (float)(s >> 8) / (float)(1u << 24);
    h[i] = (u - 0.5f) * 6.0f;
  }
}

// Oracolo de CPU: a algebra do `stage_iq3s_32_i8` (que e' a do
// `vec_dot_iq3_s_q8_1`) sobre os bytes crus do GGUF e a LUT lida do device,
// com a soma de produtos em int32 EXATO, `* (1+2*sc)` em inteiro, `d = d_w*d_a`
// em fp32 e o acumulador em fp32 com FMA correto (std::fma == __fmaf_rn).
void cpu_iq3s_32_i8(const unsigned char *b, const uint32_t *lut, int *iw, float *dw, int *scf) {
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
    if (dw) {
      dw[s] = rdna4::fp16_to_float(d_h);
      scf[s] = 1 + 2 * ((sc[s / 2] >> (4 * (s % 2))) & 0x0F);
    }
  }
}

float cpu_oracle(const unsigned char *wraw, const std::int64_t bpr, const uint32_t *lut,
                 const block_q8_1 *q8, const int m, const int n, const int K) {
  const unsigned char *row = wraw + (std::size_t)n * (std::size_t)bpr * kBlockBytes;
  const block_q8_1 *qrow = q8 + (std::size_t)m * (std::size_t)(K / kQ8);
  float acc = 0.0f;
  for (std::int64_t ib = 0; ib < bpr; ++ib) {
    int iw[256];
    float dws[8];
    int scfs[8];
    cpu_iq3s_32_i8(row + (std::size_t)ib * kBlockBytes, lut, iw, dws, scfs);
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

struct CfgW {
  int BM, BN, BK, WMW, WNK;
  const char *tag;
};
struct CfgD {
  int BM, BN, BK, RM, RN;
  const char *tag;
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

// ---- despacho em tempo de compilacao (a config e' runtime, o template nao) --
const block_iq3_s *g_W = nullptr;
const block_q8_1 *g_AQ = nullptr;
float *g_C = nullptr;

template <int BM, int BN, int BK, int WMW, int WNK>
bool launch_wmma_t(const int M, const std::int64_t N, const std::int64_t K, const int mode,
                   const int nocorr, const bool dbuf) {
  const dim3 grid((unsigned)(N / BN), (unsigned)((M + BM - 1) / BM), 1u);
  const int th = WMW * WNK * 32;
  if (dbuf) {
    gemm_wmma_i8_kernel<BM, BN, BK, WMW, WNK, true>
        <<<grid, th>>>(g_W, g_AQ, g_C, M, (int)N, (int)K, mode, nocorr);
  } else {
    gemm_wmma_i8_kernel<BM, BN, BK, WMW, WNK, false>
        <<<grid, th>>>(g_W, g_AQ, g_C, M, (int)N, (int)K, mode, nocorr);
  }
  return launch_ok("gemm_wmma_i8");
}

bool launch_wmma(const CfgW &c, const int M, const std::int64_t N, const std::int64_t K,
                 const int mode, const int nocorr, const bool dbuf) {
  const int key = c.BM * 100000 + c.BN * 1000 + c.BK * 10 + c.WMW;
  switch (key) {
    case 16 * 100000 + 64 * 1000 + 64 * 10 + 1: return launch_wmma_t<16, 64, 64, 1, 4>(M, N, K, mode, nocorr, dbuf);
    case 16 * 100000 + 128 * 1000 + 64 * 10 + 1: return launch_wmma_t<16, 128, 64, 1, 4>(M, N, K, mode, nocorr, dbuf);
    case 32 * 100000 + 64 * 1000 + 64 * 10 + 2: return launch_wmma_t<32, 64, 64, 2, 2>(M, N, K, mode, nocorr, dbuf);
    case 64 * 100000 + 64 * 1000 + 64 * 10 + 4: return launch_wmma_t<64, 64, 64, 4, 1>(M, N, K, mode, nocorr, dbuf);
    case 64 * 100000 + 64 * 1000 + 64 * 10 + 2: return launch_wmma_t<64, 64, 64, 2, 2>(M, N, K, mode, nocorr, dbuf);
    case 64 * 100000 + 64 * 1000 + 32 * 10 + 4: return launch_wmma_t<64, 64, 32, 4, 1>(M, N, K, mode, nocorr, dbuf);
    case 64 * 100000 + 128 * 1000 + 64 * 10 + 4: return launch_wmma_t<64, 128, 64, 4, 2>(M, N, K, mode, nocorr, dbuf);
    case 128 * 100000 + 64 * 1000 + 64 * 10 + 8: return launch_wmma_t<128, 64, 64, 8, 1>(M, N, K, mode, nocorr, dbuf);
    case 128 * 100000 + 64 * 1000 + 128 * 10 + 8: return launch_wmma_t<128, 64, 128, 8, 1>(M, N, K, mode, nocorr, dbuf);
    case 128 * 100000 + 128 * 1000 + 64 * 10 + 8: return launch_wmma_t<128, 128, 64, 8, 2>(M, N, K, mode, nocorr, dbuf);
    default:
      std::printf("  config WMMA nao instanciada: BM=%d BN=%d BK=%d WMW=%d\n", c.BM, c.BN, c.BK,
                  c.WMW);
      return false;
  }
}

template <int BM, int BN, int BK, int RM, int RN>
bool launch_dp4a_t(const int M, const std::int64_t N, const std::int64_t K, const int mode,
                   const int nocorr, const bool dbuf) {
  const dim3 grid((unsigned)(N / BN), (unsigned)((M + BM - 1) / BM), 1u);
  const int th = (BM / RM) * (BN / RN);
  if (dbuf) {
    gemm_dp4a_i8_kernel<BM, BN, BK, RM, RN, true>
        <<<grid, th>>>(g_W, g_AQ, g_C, M, (int)N, (int)K, mode, nocorr);
  } else {
    gemm_dp4a_i8_kernel<BM, BN, BK, RM, RN, false>
        <<<grid, th>>>(g_W, g_AQ, g_C, M, (int)N, (int)K, mode, nocorr);
  }
  return launch_ok("gemm_dp4a_i8");
}

bool launch_dp4a(const CfgD &c, const int M, const std::int64_t N, const std::int64_t K,
                 const int mode, const int nocorr, const bool dbuf) {
  switch (c.BM * 100000 + c.BN * 1000 + c.BK * 10 + c.RM) {
    case 64 * 100000 + 64 * 1000 + 64 * 10 + 8: return launch_dp4a_t<64, 64, 64, 8, 4>(M, N, K, mode, nocorr, dbuf);
    case 128 * 100000 + 64 * 1000 + 64 * 10 + 8: return launch_dp4a_t<128, 64, 64, 8, 4>(M, N, K, mode, nocorr, dbuf);
    default:
      std::printf("  config dp4a nao instanciada: BM=%d BN=%d BK=%d RM=%d\n", c.BM, c.BN, c.BK,
                  c.RM);
      return false;
  }
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<int> ms_list{16, 64, 128, 512};
  int reps = 5, BK_ovr = 0;
  std::vector<int> cfg_sel;
  bool DBUF = false, verify = true, peak_only = false, list_only = false;
  std::int64_t n_ovr = -1, k_ovr = -1;
  std::string tensor_name = "blk.3.ffn_up.weight";
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
    } else if (!std::strcmp(argv[i], "--cfg") && i + 1 < argc) {
      const char *v = argv[++i];
      while (*v) {
        cfg_sel.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    } else if (!std::strcmp(argv[i], "--bk") && i + 1 < argc) {
      BK_ovr = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (!std::strcmp(argv[i], "--n") && i + 1 < argc) {
      n_ovr = std::atoll(argv[++i]);
    } else if (!std::strcmp(argv[i], "--k") && i + 1 < argc) {
      k_ovr = std::atoll(argv[++i]);
    } else if (!std::strcmp(argv[i], "--tensor") && i + 1 < argc) {
      tensor_name = argv[++i];
    } else if (!std::strcmp(argv[i], "--dbuf")) {
      DBUF = true;
    } else if (!std::strcmp(argv[i], "--no-verify")) {
      verify = false;
    } else if (!std::strcmp(argv[i], "--peak-only")) {
      peak_only = true;
    } else if (!std::strcmp(argv[i], "--list")) {
      list_only = true;
    } else if (argv[i][0] != '-') {
      model = argv[i];
    }
  }
  if (!model) {
    std::printf(
        "uso: bench-mmq-wmma-gpu <model.gguf> [--m 16,64,128,512] [--cfg ID] [--reps 5]\n"
        "                       [--tensor NOME] [--n N] [--k K] [--bk N] [--dbuf] [--list]\n"
        "                       [--no-verify] [--peak-only]\n");
    return 1;
  }
  {
    hipDeviceProp_t dp{};
    if (hipGetDeviceProperties(&dp, 0) == hipSuccess) {
      std::printf("device: %s (%s), %d CUs, LDS/bloco max %zu B\n", dp.name, dp.gcnArchName,
                  dp.multiProcessorCount, (size_t)dp.sharedMemPerBlock);
    }
  }

  // ---- 0. o tensor real ---------------------------------------------------
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(model, err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  if (list_only) {
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      const auto &t = ld.meta().tensors[i];
      if ((int)ld.dtype(i) != kIq3S) continue;
      std::printf("  %-42s dims=[%lld, %lld]  %8.1f MB\n", t.name.c_str(), (long long)t.dims[0],
                  (long long)t.dims[1], (double)ld.tensor_bytes(i) / 1e6);
    }
    return 0;
  }
  const auto *found = ld.find(tensor_name.c_str());
  if (!found) {
    std::fprintf(stderr, "tensor %s nao existe (use --list)\n", tensor_name.c_str());
    return 1;
  }
  const std::size_t ti = (std::size_t)(found - ld.meta().tensors.data());
  const auto &info = ld.meta().tensors[ti];
  const std::int64_t K_full = (std::int64_t)info.dims[0];
  const std::int64_t N_full = (std::int64_t)info.dims[1];
  const std::int64_t K = (k_ovr > 0) ? k_ovr : K_full;
  const std::int64_t N = (n_ovr > 0) ? n_ovr : N_full;
  std::printf("tensor: %s  [%lld x %lld] dtype=%d, %.1f MB; K=%lld N=%lld\n", info.name.c_str(),
              (long long)K_full, (long long)N_full, (int)ld.dtype(ti),
              (double)ld.tensor_bytes(ti) / 1e6, (long long)K, (long long)N);
  if ((int)ld.dtype(ti) != kIq3S) {
    std::fprintf(stderr, "o tensor nao e' iq3_s\n");
    return 1;
  }
  if (K % kQK != 0 || N % 128 != 0) {
    std::fprintf(stderr, "K tem de ser multiplo de 256 e N de 128\n");
    return 1;
  }
  const std::int64_t bpr = K / kQK;
  const std::uint64_t want = (std::uint64_t)N * (std::uint64_t)bpr * kBlockBytes;
  std::vector<std::uint8_t> w_raw;
  if (!ld.load_tensor_range(ti, 0, want, w_raw, err)) {
    std::fprintf(stderr, "load_tensor_range: %s\n", err.c_str());
    return 1;
  }
  std::printf("pesos crus: %.1f MB, %.4f B/peso (110 B por 256, iq3_s)\n",
              (double)w_raw.size() / 1e6, (double)w_raw.size() / ((double)N * (double)K));

  // ---- 1. ativacao (float -> q8_1 pelo kernel DO MOTOR) -------------------
  const int MMAX = *std::max_element(ms_list.begin(), ms_list.end());
  // As linhas alem de MMAX existem para os tiles BM > M nao lerem fora do
  // buffer; ficam zeradas, entao a ativacao quantizada delas e' toda zero e as
  // linhas extras de C saem 0,0f (nao entram em nenhuma comparacao).
  const int MPAD = ((std::max(MMAX, 16) + 127) / 128) * 128;
  std::vector<float> hA;
  fill_a(hA, (std::size_t)MPAD * (std::size_t)K, 7u);
  for (std::size_t i = (std::size_t)MMAX * (std::size_t)K; i < hA.size(); ++i) hA[i] = 0.0f;

  void *dW = nullptr;
  float *dAf = nullptr;
  block_q8_1 *dAQ = nullptr;
  float *dC = nullptr, *dC2 = nullptr;
  if (hipMalloc(&dW, w_raw.size()) != hipSuccess ||
      hipMalloc(&dAf, hA.size() * sizeof(float)) != hipSuccess ||
      hipMalloc(&dAQ, (std::size_t)MPAD * (std::size_t)(K / kQ8) * sizeof(block_q8_1)) !=
          hipSuccess ||
      hipMalloc(&dC, (std::size_t)MPAD * (std::size_t)N * sizeof(float)) != hipSuccess ||
      hipMalloc(&dC2, (std::size_t)MPAD * (std::size_t)N * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc falhou\n");
    return 1;
  }
  (void)hipMemcpy(dW, w_raw.data(), w_raw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(dAf, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice);
  g_W = (const block_iq3_s *)dW;
  g_AQ = dAQ;
  g_C = dC;
  if (!rdna4::quantize_q8_1_batch_launch(dAf, dAQ, K / kQ8, MPAD, K, nullptr)) {
    std::fprintf(stderr, "quantize_q8_1 falhou\n");
    return 1;
  }
  if (!launch_ok("quantize_q8_1_batch")) return 1;
  (void)hipDeviceSynchronize();

  // A ativacao volta para o host: o oraculo de CPU tem de ver EXATAMENTE os
  // mesmos inteiros q8_1 que o kernel (nada de requantizar na CPU).
  std::vector<block_q8_1> hAQ((std::size_t)MPAD * (std::size_t)(K / kQ8));
  (void)hipMemcpy(hAQ.data(), dAQ, hAQ.size() * sizeof(block_q8_1), hipMemcpyDeviceToHost);

  // ---- 2. a LUT: device -> host (uma unica fonte de verdade) --------------
  std::vector<uint32_t> lut(512, 0);
  uint32_t *dLut = nullptr;
  if (hipMalloc(&dLut, lut.size() * sizeof(uint32_t)) != hipSuccess) return 1;
  copy_lut_kernel<<<1, 128>>>(dLut, 512);
  if (!launch_ok("copy_lut")) return 1;
  (void)hipMemcpy(lut.data(), dLut, lut.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
  {
    uint32_t sum = 0;
    for (uint32_t v : lut) sum = sum * 31u + v;
    std::printf("LUT iq3s_grid lida do device: 512 entradas, hash=%08x\n", sum);
  }

  // ---- 3. CORRETUDE -------------------------------------------------------
  if (verify && !peak_only) {
    std::printf("\n== 1/4. verificacao (antes de qualquer cronometragem) ==\n");
    // (a) staging de peso: int8 na LDS contra o oraculo de CPU, peso a peso
    const int nsub = 64;  // 8 super-blocos = K 2048 da linha 0
    int *dStage = nullptr;
    if (hipMalloc(&dStage, (std::size_t)nsub * 9 * sizeof(int)) != hipSuccess) return 1;
    dump_stage_i8_kernel<<<(nsub + 63) / 64, 64>>>(g_W, dStage, 0, (int)bpr, nsub);
    if (!launch_ok("dump_stage_i8")) return 1;
    (void)hipDeviceSynchronize();
    std::vector<int> staged((std::size_t)nsub * 9);
    (void)hipMemcpy(staged.data(), dStage, staged.size() * sizeof(int), hipMemcpyDeviceToHost);
    int bad_w = 0, bad_sc = 0, first_bad = -1;
    for (int t = 0; t < nsub; ++t) {
      int iw[256];
      float dws[8];
      int scfs[8];
      cpu_iq3s_32_i8(w_raw.data() + (std::size_t)(t / 8) * kBlockBytes, lut.data(), iw, dws, scfs);
      const int s = t % 8;
      for (int j = 0; j < 32; ++j) {
        const int got = (staged[(std::size_t)t * 9 + j / 4] >> (8 * (j % 4))) & 0xFF;
        const int want = iw[s * 32 + j] & 0xFF;
        if (got != want) {
          ++bad_w;
          if (first_bad < 0) first_bad = t * 32 + j;
        }
      }
      const int packed = staged[(std::size_t)t * 9 + 8];
      if ((packed & 0x00FFFFFF) != (f2u(dws[s]) & 0x00FFFFFF) || (packed >> 24) != scfs[s]) {
        ++bad_sc;
      }
    }
    std::printf(
        "check do staging int8 (64 sub-blocos = 2048 pesos da linha 0, contra a algebra do\n"
        "  `vec_dot_iq3_s_q8_1` escrita em CPU sobre os bytes crus + a LUT do device):\n"
        "  bytes int8 diferentes: %d de 2048%s ; d_w e (1+2*sc) errados: %d de 64\n",
        bad_w, bad_w ? "" : " (ok)", bad_sc);
    if (first_bad >= 0) std::printf("  primeiro byte errado: indice %d\n", first_bad);
    (void)hipFree(dStage);
    if (bad_w || bad_sc) {
      std::printf("  REPROVADO: o staging nao reproduz a algebra do motor\n");
      return 1;
    }

    // (b) caso pequeno contra o oraculo de CPU: M=4, N=64
    const int Mc = 4, Nc = 64;
    const CfgW cw{64, 64, 64, 4, 1, "BM64 BN64"};
    std::vector<float> got((std::size_t)Mc * (std::size_t)Nc, 0.f);
    (void)hipMemset(dC, 0, (std::size_t)MPAD * (std::size_t)N * sizeof(float));
    if (!launch_wmma(cw, Mc, Nc, K, 0, 0, false)) return 1;
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(got.data(), dC, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    double mx = 0, mrel = 0, mref = 0;
    std::size_t ndiff = 0;
    int n_nonfinite = 0, n_zero = 0;
    for (int m = 0; m < Mc; ++m) {
      for (int n = 0; n < Nc; ++n) {
        const float wantv = cpu_oracle(w_raw.data(), bpr, lut.data(), hAQ.data(), m, n, (int)K);
        const float g = got[(std::size_t)m * Nc + n];
        if (!std::isfinite(g)) ++n_nonfinite;
        if (g == 0.f) ++n_zero;
        if (f2u(wantv) != f2u(g)) ++ndiff;
        const double d = std::fabs((double)g - (double)wantv);
        mx = std::max(mx, d);
        mref = std::max(mref, std::fabs((double)wantv));
        if (std::fabs((double)wantv) > 0) mrel = std::max(mrel, d / std::fabs((double)wantv));
      }
    }
    std::printf(
        "oraculo de CPU (M=%d, N=%d, K=%lld: 256 saidas x 160 blocos de 32; produto em int32\n"
        "  exato, `* (1+2*sc)` inteiro, d_w*d_a em fp32, acumulador fp32 com std::fma):\n"
        "  max |ref| = %.6g ; max |gpu-cpu| = %.3g ; max rel = %.3g ; nao-finitos = %d ; zeros = "
        "%d\n"
        "  elementos com BITS diferentes: %zu de %d\n",
        Mc, Nc, (long long)K, mref, mx, mrel, n_nonfinite, n_zero, ndiff, Mc * Nc);
    if (n_nonfinite) {
      std::printf("  REPROVADO: saida nao-finita (bug de kernel, nao de precisao)\n");
      return 1;
    }
    if (n_zero == Mc * Nc) {
      std::printf("  REPROVADO: C toda zero (o kernel nao rodou?)\n");
      return 1;
    }

    // (b2) o `vec_dot_iq3_s_q8_1` do MOTOR, um thread por saida
    const int Me = 16, Ne = 256;
    std::vector<float> eng((std::size_t)Me * (std::size_t)Ne);
    ref_engine_kernel<<<(unsigned)((Me * Ne + 127) / 128), 128>>>(g_W, dAQ, dC2, Me, Ne, (int)K);
    if (!launch_ok("ref_engine_kernel")) return 1;
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(eng.data(), dC2, eng.size() * sizeof(float), hipMemcpyDeviceToHost);
    std::vector<float> got2((std::size_t)Me * (std::size_t)Ne), got3((std::size_t)Me * (std::size_t)Ne);
    (void)hipMemset(dC, 0, (std::size_t)MPAD * (std::size_t)N * sizeof(float));
    g_C = dC;
    if (!launch_wmma(cw, Me, Ne, K, 0, 0, false)) return 1;
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(got2.data(), dC, got2.size() * sizeof(float), hipMemcpyDeviceToHost);
    const CfgD cd{64, 64, 64, 8, 4, "BM64 BN64 RM8 RN4"};
    (void)hipMemset(dC2, 0, (std::size_t)MPAD * (std::size_t)N * sizeof(float));
    g_C = dC2;
    if (!launch_dp4a(cd, Me, Ne, K, 0, 0, false)) return 1;
    (void)hipDeviceSynchronize();
    (void)hipMemcpy(got3.data(), dC2, got3.size() * sizeof(float), hipMemcpyDeviceToHost);
    const Diff dw_eng = diff_bits(got2, eng, got2.size());
    const Diff dd_eng = diff_bits(got3, eng, got3.size());
    const Diff dw_dd = diff_bits(got2, got3, got2.size());
    std::printf(
        "referencia do MOTOR (`vec_dot_iq3_s_q8_1` do vecdotq.cuh, M=%d, N=%d, K=%lld), %zu "
        "saidas:\n"
        "  WMMA x motor : %zu bits diferentes, max ulp %.0f, max|d| %.3g, rel-L2 %.3g\n"
        "  dp4a x motor : %zu bits diferentes, max ulp %.0f, max|d| %.3g, rel-L2 %.3g\n"
        "  WMMA x dp4a  : %zu bits diferentes, max ulp %.0f, max|d| %.3g, rel-L2 %.3g\n",
        Me, Ne, (long long)K, got2.size(), dw_eng.n_diff, dw_eng.max_ulps, dw_eng.max_abs,
        dw_eng.rel_l2, dd_eng.n_diff, dd_eng.max_ulps, dd_eng.max_abs, dd_eng.rel_l2, dw_dd.n_diff,
        dw_dd.max_ulps, dw_dd.max_abs, dw_dd.rel_l2);
  }

  // ---- 4. PICO na mesma janela -------------------------------------------
  EvPair ev;
  double peak_wmma = 0, peak_dp4a = 0;
  {
    int *d_i = nullptr;
    const int threads = 256;  // 8 warps por CTA
    const int ctas[] = {32, 64, 128, 256};
    const std::size_t maxth = (std::size_t)256 * threads;
    if (hipMalloc(&d_i, maxth * sizeof(int)) != hipSuccess) return 1;
    auto cell = [&](const int which, const char *name, const double macs_per_warp_iter) {
      double best_of_all = 0;
      int best_ctas = 0, best_iters = 0;
      double best_ms = 0;
      for (int ct : ctas) {
        auto launch = [&](const int iters) {
          if (which == 0) peak_loop_kernel<8, 8, 0><<<ct, threads>>>(iters, d_i);
          else peak_loop_kernel<8, 8, 1><<<ct, threads>>>(iters, d_i);
        };
        launch(20);
        (void)hipDeviceSynchronize();
        const double t1 = std::max(1e-6, ev([&] { launch(20); }) / 20.0);
        const int iters = (int)std::max(50.0, 4.0 / t1);
        double best = 1e9;
        for (int r = 0; r < 5; ++r) best = std::min(best, ev([&] { launch(iters); }));
        const double per_s = macs_per_warp_iter * (double)ct * iters / (best * 1e-3);
        std::printf("  pico %s CTAs=%-4d (U=8, 8 cadeias, iters=%d): %.3f ms -> %.2f T-MAC/s\n",
                    name, ct, iters, best, per_s / 1e12);
        if (per_s > best_of_all) {
          best_of_all = per_s;
          best_ctas = ct;
          best_iters = iters;
          best_ms = best;
        }
      }
      std::printf("  pico %s MELHOR: CTAs=%d, iters=%d, %.3f ms -> %.2f T-MAC/s\n", name, best_ctas,
                  best_iters, best_ms, best_of_all / 1e12);
      return best_of_all;
    };
    std::printf("\n== 2/4. pico remedido nesta janela ==\n");
    peak_wmma = cell(0, "WMMA int8 16x16x16", 8.0 * 8.0 * 4096.0);
    peak_dp4a = cell(1, "dp4a              ", 8.0 * 8.0 * 128.0);
    if (!launch_ok("peak loops")) return 1;
    std::printf("  razao pico WMMA/dp4a nesta janela: %.2fx (frente D, outra janela: 4,2x)\n",
                peak_wmma / peak_dp4a);
    (void)hipFree(d_i);
  }
  if (peak_only) {
    (void)hipFree(dW); (void)hipFree(dAf); (void)hipFree(dAQ); (void)hipFree(dC); (void)hipFree(dC2);
    (void)hipFree(dLut);
    return 0;
  }
  g_C = dC;

  // ---- 5. VELOCIDADE ------------------------------------------------------
  static const CfgW kW[] = {
      {16, 64, 64, 1, 4, "BM16  BN64  BK64  W1x4 MT1 NT1"},
      {16, 128, 64, 1, 4, "BM16  BN128 BK64  W1x4 MT1 NT2"},
      {32, 64, 64, 2, 2, "BM32  BN64  BK64  W2x2 MT1 NT2"},
      {64, 64, 64, 4, 1, "BM64  BN64  BK64  W4x1 MT1 NT4"},
      {64, 64, 64, 2, 2, "BM64  BN64  BK64  W2x2 MT2 NT2"},
      {64, 64, 32, 4, 1, "BM64  BN64  BK32  W4x1 MT1 NT4"},
      {64, 128, 64, 4, 2, "BM64  BN128 BK64  W4x2 MT1 NT4"},
      {128, 64, 64, 8, 1, "BM128 BN64  BK64  W8x1 MT1 NT4"},
      {128, 64, 128, 8, 1, "BM128 BN64  BK128 W8x1 MT1 NT4"},
      {128, 128, 64, 8, 2, "BM128 BN128 BK64  W8x2 MT1 NT4"},
  };
  const int ncfg = (int)(sizeof(kW) / sizeof(kW[0]));
  static const int kDefault[] = {3, 7};  // BM64 (4 warps) e BM128 (8 warps)
  const CfgD kD{64, 64, 64, 8, 4, "BM64 BN64 BK64 RM8 RN4"};

  // Warm-up global: a frente F mediu que a PRIMEIRA config de um processo sai
  // ~47 % mais lenta que as seguintes (DPM), entao o aquecimento tem de vir
  // antes do laco de medicao.
  {
    auto fn = [&] { (void)launch_wmma(kW[kDefault[0]], MMAX, N, K, 0, 0, false); };
    fn();
    (void)hipDeviceSynchronize();
    double warm = 0;
    for (int it = 0; it < 4000 && warm < 400.0; ++it) warm += ev(fn);
    std::printf("\n(aquecimento global: %.0f ms de GPU)\n", warm);
  }

  std::printf("\n== 3/4. GEMM com os blocos REAIS: WMMA int8 x dp4a, MESMO staging ==\n");
  std::printf("%5s %-33s %10s %10s %8s %10s %8s %9s %9s\n", "M", "cfg", "t_GEMM ms", "T-MAC/s",
              "% picoW", "T-MAC/s d4", "d4/WMMA", "stage ms", "nocorr ms");
  for (const int M : ms_list) {
    for (int ci = 0; ci < ncfg; ++ci) {
      {
        bool want = false;
        if (!cfg_sel.empty()) {
          for (int d : cfg_sel)
            if (d == ci) want = true;
        } else {
          for (int d : kDefault)
            if (d == ci) want = true;
        }
        if (!want) continue;
      }
      CfgW c = kW[ci];
      if (BK_ovr > 0) c.BK = BK_ovr;
      if (c.BM > MPAD) continue;  // o kernel escreve tiles inteiros ate' MPAD
      if (DBUF) {
        const std::size_t lds = 2 * ((std::size_t)c.BN * (c.BK + 8) + (std::size_t)c.BM * (c.BK + 8) +
                                     (std::size_t)(c.BK / 32) * (c.BN * 8 + c.BM * 4));
        if (lds > 65536) {
          std::printf("  (config %s: duplo buffer pede %zu B de LDS > 65536, pulada)\n", c.tag, lds);
          continue;
        }
      }
      // ---- verificacao da config ANTES de cronometrar -------------------
      if (verify) {
        const std::size_t n_el = (std::size_t)M * (std::size_t)N;
        (void)hipMemset(dC, 0, n_el * sizeof(float));
        (void)hipMemset(dC2, 0, n_el * sizeof(float));
        g_C = dC;
        if (!launch_wmma(c, M, N, K, 0, 0, DBUF)) break;
        g_C = dC2;
        if (!launch_dp4a(kD, M, N, K, 0, 0, false)) break;
        (void)hipDeviceSynchronize();
        std::vector<float> a(n_el), b(n_el);
        (void)hipMemcpy(a.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
        (void)hipMemcpy(b.data(), dC2, n_el * sizeof(float), hipMemcpyDeviceToHost);
        const Diff d = diff_bits(a, b, n_el);
        std::size_t nz = 0;
        for (std::size_t i = 0; i < n_el; ++i)
          if (a[i] == 0.f) ++nz;
        std::printf("  check M=%-4d %-26s WMMA x dp4a: %zu de %zu bits diferentes (max ulp %.0f, "
                    "rel-L2 %.3g, zeros %zu)%s\n",
                    M, c.tag, d.n_diff, n_el, d.max_ulps, d.rel_l2, nz,
                    nz == n_el ? "   TODA ZERO!" : "");
        if (nz == n_el) break;
      }
      // ---- cronometragem: minimo de `reps` depois do aquecimento --------
      g_C = dC;
      auto t_of = [&](const int mode, const int nocorr) {
        auto fn = [&] { (void)launch_wmma(c, M, N, K, mode, nocorr, DBUF); };
        fn();
        (void)hipDeviceSynchronize();
        double warm = 0;
        for (int it = 0; it < 4000 && warm < 150.0; ++it) warm += ev(fn);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) best = std::min(best, ev(fn));
        return best;
      };
      const double t_gemm = t_of(0, 0);
      const double t_stage = t_of(1, 0);
      const double t_nocorr = t_of(0, 1);
      g_C = dC2;
      auto fn_d = [&] { (void)launch_dp4a(kD, M, N, K, 0, 0, false); };
      fn_d();
      (void)hipDeviceSynchronize();
      {
        double warm = 0;
        for (int it = 0; it < 4000 && warm < 150.0; ++it) warm += ev(fn_d);
      }
      double t_d4 = 1e9;
      for (int r = 0; r < reps; ++r) t_d4 = std::min(t_d4, ev(fn_d));
      g_C = dC;
      const double macs = (double)M * (double)N * (double)K;
      const double tw = macs / (t_gemm * 1e-3) / 1e12;
      const double td = macs / (t_d4 * 1e-3) / 1e12;
      std::printf("%5d %-33s %10.3f %10.2f %8.1f %10.2f %8.2f %9.3f %9.3f\n", M, c.tag, t_gemm, tw,
                  100.0 * tw / (peak_wmma / 1e12), td, td / tw, t_stage, t_nocorr);
      const int mt = (M + c.BM - 1) / c.BM;
      const double wts = (double)N * (double)K * mt;
      const double src_b = wts * (double)kBlockBytes / (double)kQK;
      std::printf("        staging(W+A) %.3f ms = %.0f%% do GEMM -> %.1f G pesos/s, %.1f GB/s de "
                  "fonte; sem correcao %.3f ms -> %.2f T-MAC/s; conta pura %.3f ms -> %.2f "
                  "T-MAC/s\n",
                  t_stage, 100.0 * t_stage / t_gemm, wts / (t_stage * 1e-3) / 1e9,
                  src_b / (t_stage * 1e-3) / 1e9, t_nocorr, macs / (t_nocorr * 1e-3) / 1e12,
                  t_gemm - t_stage, macs / ((t_gemm - t_stage) * 1e-3) / 1e12);
      std::printf("        correcao (t_gemm - t_nocorr) %.3f ms = %.0f%% do GEMM\n",
                  t_gemm - t_nocorr, 100.0 * (t_gemm - t_nocorr) / t_gemm);
    }
  }

  std::printf(
      "\n== 4/4. referencias (mesma maquina; a janela de cada uma esta' no doc) ==\n"
      "  nosso prefill hoje (matvec em lote) .............. 3,0-3,2 T-MAC/s\n"
      "  llama.cpp Vulkan coopmat f16, pp512 .............. 32,7 T-MAC/s\n"
      "  prototipo int8+dp4a tilejado (sintetico) M=512 ... 12,74 T-MAC/s\n"
      "  prototipo f16 dot2 com blocos reais M=128/512 ..... 9,2-11,7 T-MAC/s\n"
      "  pico WMMA int8 (frente D, outra janela) .......... 182 T-MAC/s\n");
  (void)hipFree(dW);
  (void)hipFree(dAf);
  (void)hipFree(dAQ);
  (void)hipFree(dC);
  (void)hipFree(dC2);
  (void)hipFree(dLut);
  return 0;
}
