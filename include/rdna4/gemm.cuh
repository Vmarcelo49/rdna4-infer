// ===========================================================================
// D2 do plano do prefill (docs/plano-prefill.md §6): GEMM tilejado int8 com
// staging na LDS, consumindo os BLOCOS REAIS do modelo (iq3_s e os outros tipos
// IQ de cadeia unica) e a ativacao `block_q8_1` do motor.
//
// A receita e' a MEDIDA, nao uma escolha: `docs/estudo-prefill-g-staging2.md`
// §Veredito (V6 = BM128 BN128 BK64 RM=RN=8, 256 threads, prefetch dos campos do
// peso em registrador, duplo buffer SO' do W, ativacao lida direto da global) e
// `docs/estudo-prefill-h-wmma-mmq.md` §2 para a condicao de BIT-EXATIDAO contra
// o dp4a do motor:
//
//   1. o peso vira int8 ASSINADO na LDS (magnitude 1..15 do grid + sinal), na
//      ordem de k, com o fator inteiro `1+2*sc` FORA do byte (15*31 = 465 nao
//      cabe em int8);
//   2. o laco interno acumula int32 por bloco de 32 (dp4a), como o
//      `vec_dot_iq3_s_q8_1` (vecdotq.cuh:705-723);
//   3. a correcao e' `sumi *= 1+2*sc` em INTEIRO, depois `d = d_w*d_a` em fp32,
//      depois `acc = fma(d, (float)sumi, acc)` -- a MESMA sequencia de
//      instrucoes que a ISA do motor mostra (`v_mul_lo_u32`, `v_cvt_f32_i32`,
//      `v_mul_f32`, `v_fmac_f32`). Trocar a ordem muda o arredondamento.
//
// Consequencia: a saida e' BIT A BIT igual ao `vec_dot_*` do motor somado em
// ordem de k crescente (e' o gate do `tests/bench_gemm_engine_gpu.hip`), o que
// dispensa PPL/gate numerico para esta rota (plano §6, item 1).
//
// Orcamento de LDS (BM=BN=128, BK=64, duplo buffer do W + os campos d_w/sc):
//   s_w   2 * 128 * (64+16) B  = 20 480 B
//   s_dwsc 2 * 2 * 128 * 8 B   =  4 096 B
//   total                      = 24 576 B <= 32 KB (2 CTAs/CU cabem em LDS; o
//   que decide a ocupacao e' o VGPR -- ver `gemm_kernel_attrs`).
//
// O stride de linha do W na LDS e' BK+16 (80 B) e as colunas de cada thread sao
// ESTRIDADAS (n = tc + c*TN): assim as 16 (ou 32) linhas lidas por uma warp caem
// em 16 bancos distintos e nao ha' conflito nem de 4 B nem de 16 B.
//
// API publica: `gemm_launch`, espelhando `matvec_launch_batch` para o grafo
// poder trocar um pelo outro. `false` = nao suportado (o chamador decide); nunca
// ha' queda silenciosa para outro caminho.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

#include "rdna4/fp16.h"
#include "rdna4/quants.h"
#include "rdna4/quant_tables.h"
#include "rdna4/vecdotq.cuh"  // ggml_cuda_dp4a / __vcmpne4 / __vsub4 / as LUTs

namespace rdna4 {

// ===========================================================================
// 0. Mascara de sinal por byte a partir de 4 bits (o truque do `xoradd`)
// ===========================================================================
// `(g^m) + (m & 0x01010101)` e' a negacao por byte em complemento de dois SEM
// carry entre bytes, e vale porque todo byte dos grids iq3s/iq3xxs e' IMPAR
// (~g + 1 = -g e ~g <= 0xFE, entao o +1 nunca transborda o byte). E' a mesma
// identidade que o `vec_dot_iq3_s_q8_1_xoradd` do motor usa.
//
// O spread dos 4 bits nos 4 LSBs de byte e' `(b * 0x00204081) & 0x01010101`
// (b*255 = 0xFF por byte marcado, sem carry -- 0x01010101*0xFF = 0xFFFFFFFF).
__device__ __forceinline__ int gemm_sign_mask4(const unsigned bits) {
  return (int)((((bits & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu);
}

// ===========================================================================
// 1. Traits de staging por tipo: bloco real -> int8 assinado na LDS
// ===========================================================================
// Interface (uma por tipo):
//   struct Pf                     campos crus do peso, em REGISTRADOR (prefetch)
//   QK                            pesos por super-bloco
//   block_bytes                   bytes por super-bloco
//   static Pf load(blk, sub)      carrega os campos do sub-bloco `sub` (32 pesos)
//   static void store(pf, sub, dst8, &dw, &scf)
//                                 dequantiza -> 8 palavras int32 em ordem de k,
//                                 d_w (fp16 -> float) e o fator inteiro do bloco
//   static float corr(sumi, scf)  a correcao do motor em INTEIRO -> float
//
// Um "sub-bloco" e' sempre 32 pesos (= QK8_1 = um bloco da ativacao), que e' a
// granularidade em que o motor acumula int32 e aplica a escala.

// ---- iq3_s (docs/estudo-prefill-f-staging.md §1; validado bit a bit pela
// frente H: 0 de 2048 bytes). E' o tipo dos tensores ffn_up/ffn_down deste
// modelo (110 B por 256 pesos, 0,4297 B/peso).
struct SolverIq3S {
  static constexpr int qk = QK_K;  // 256
  static constexpr int block_bytes = (int)sizeof(block_iq3_s);
  struct Pf {
    int qs0, qs1;      // qs[8*sub .. 8*sub+8)  (8 indices de grid = 32 pesos)
    unsigned qh;       // qh[sub]                (bit alto do indice)
    unsigned sg;       // signs[4*sub .. 4*sub+4) (8 bits de sinal)
    unsigned dsc;      // d (fp16) | scales[sub/2] << 16
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_iq3_s *b = (const block_iq3_s *)blk;
    Pf p;
    const int *qs = (const int *)(const void *)(b->qs + 8 * sub);
    p.qs0 = qs[0];
    p.qs1 = qs[1];
    p.qh = b->qh[sub];
    p.sg = ((const unsigned *)(const void *)b->signs)[sub];
    p.dsc = (unsigned)b->d | ((unsigned)b->scales[sub >> 1] << 16);
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, float *dw,
                                               int *scf) {
    const unsigned char *qs = (const unsigned char *)&p.qs0;
    const int qh = (int)p.qh;
    const unsigned char *sg = (const unsigned char *)&p.sg;
    *dw = fp16_to_float((uint16_t)(p.dsc & 0xFFFFu));
    *scf = 1 + 2 * (int)(((p.dsc >> 16) >> (4 * (sub & 1))) & 0x0Fu);
#pragma unroll
    for (int il = 0; il < 4; ++il) {
      const int g1 = iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)];
      const int g2 = iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)];
      const int m0 = gemm_sign_mask4(((sg[il] & 0x03u) << 7) | ((sg[il] & 0x0Cu) << 21));
      const int m1 = gemm_sign_mask4(((sg[il] & 0x30u) << 3) | ((sg[il] & 0xC0u) << 17));
      dst[2 * il + 0] = (g1 ^ m0) + (m0 & 0x01010101);
      dst[2 * il + 1] = (g2 ^ m1) + (m1 & 0x01010101);
    }
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)(sumi * scf);
  }
};

// ---- iq3_xxs (2,75 bpw, 17,8 % dos bytes do inventario). Mesma forma de cadeia
// unica do iq3_s, com duas diferencas que a ISA do motor impoe:
//   * o campo de sinais tem 7 bits por byte e o 8o e' a PARIDADE (`sv ^= popc&1
//     << 7`, vecdotq.cuh:1134) -- tem de ser reproduzido, nao "simplificado";
//   * a correcao NAO e' um produto: `sumi = (ls*sumi + sumi/2)/2`
//     (vecdotq.cuh:1152) -- inteiro, com as duas divisoes truncando para zero.
struct SolverIq3XXS {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_iq3_xxs);
  struct Pf {
    int qs0, qs1;  // qs[8*sub .. 8*sub+8)
    unsigned aux;  // qs[64 + 4*sub .. +4): 8 bits de sinal por 8 pesos + ls no topo
    unsigned d;    // d (fp16) do super-bloco
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_iq3_xxs *b = (const block_iq3_xxs *)blk;
    Pf p;
    const int *qs = (const int *)(const void *)(b->qs + 8 * sub);
    p.qs0 = qs[0];
    p.qs1 = qs[1];
    p.aux = ((const unsigned *)(const void *)(b->qs + QK_K / 4))[sub];
    p.d = b->d;
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int /*sub*/, int *dst, float *dw,
                                               int *scf) {
    const unsigned char *q3 = (const unsigned char *)&p.qs0;
    const unsigned aux = p.aux;
    *dw = fp16_to_float((uint16_t)p.d);
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
      const unsigned gx = iq3xxs_grid[q3[l0 + 0]];
      const unsigned gy = iq3xxs_grid[q3[l0 + 1]];
      unsigned sv = (unsigned)(uint8_t)(aux >> (7 * (l0 / 2)));
      sv ^= (unsigned)(__popc(sv) & 1u) << 7;
      const unsigned mx = (unsigned)gemm_sign_mask4(sv & 0x0Fu);
      const unsigned my = (unsigned)gemm_sign_mask4((sv >> 4) & 0x0Fu);
      dst[l0 + 0] = (int)((gx ^ mx) + (mx & 0x01010101u));
      dst[l0 + 1] = (int)((gy ^ my) + (my & 0x01010101u));
    }
    *scf = (int)(aux >> 28);  // ls
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)((scf * sumi + sumi / 2) / 2);
  }
};

// ---- iq4_xs (4 bits "extra", 9,5 % dos bytes; e' o tipo das projecoes de
// atencao deste modelo: blk.N.attn_qkv = 10240x5120, 38 de 40 tensores por
// camada). Os pesos sao a PROPRIA tabela `kvalues_iq4nl` (int8, -127..113), sem
// truque de sinal; a correcao e' `sumi *= ls - 32` com ls de 6 bits.
//
// A ORDEM de k nao e' a ordem dos bytes: o motor pareia o nibble BAIXO de
// qs[16*m + 4*j + p] com o byte de ativacao 4*j + p e o nibble ALTO com
// 16 + 4*j + p (vecdotq.cuh:791-798, onde `v.x` = nibbles baixos e `v.y` =
// altos, casados com `get_int_b4(qs, j)` e `get_int_b4(qs, j+4)`). Ou seja: os
// 16 primeiros k`s do bloco sao os nibbles baixos e os 16 ultimos os altos.
struct SolverIq4XS {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_iq4_xs);
  struct Pf {
    int q0, q1, q2, q3;  // qs[16*sub .. 16*sub+16)  (32 nibbles = 32 pesos)
    unsigned dsl;        // d (fp16) | scales_l[sub/2] << 16
    unsigned sh;         // scales_h
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_iq4_xs *b = (const block_iq4_xs *)blk;
    Pf p;
    const int *q = (const int *)(const void *)(b->qs + 16 * sub);
    p.q0 = q[0];
    p.q1 = q[1];
    p.q2 = q[2];
    p.q3 = q[3];
    p.dsl = (unsigned)b->d | ((unsigned)b->scales_l[sub >> 1] << 16);
    p.sh = b->scales_h;
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, float *dw,
                                               int *scf) {
    // `get_int_from_table_16` do motor: byte p do resultado = kvalues[nibble p],
    // o primeiro int com os nibbles PARES e o segundo com os impares.
    const int2 lo = get_int_from_table_16(p.q0, kvalues_iq4nl);
    const int2 hi = get_int_from_table_16(p.q1, kvalues_iq4nl);
    const int2 lo2 = get_int_from_table_16(p.q2, kvalues_iq4nl);
    const int2 hi2 = get_int_from_table_16(p.q3, kvalues_iq4nl);
    // k = 0..15 = nibbles baixos (bytes 0..15 = q0,q1,q2,q3 em ordem)
    dst[0] = lo.x;
    dst[1] = lo.y;
    dst[2] = lo2.x;
    dst[3] = lo2.y;
    // k = 16..31 = nibbles altos
    dst[4] = hi.x;
    dst[5] = hi.y;
    dst[6] = hi2.x;
    dst[7] = hi2.y;
    *dw = fp16_to_float((uint16_t)(p.dsl & 0xFFFFu));
    const int sl = (int)((p.dsl >> 16) & 0xFFu);
    const int sh = (int)p.sh;
    const int ls = ((sl >> (4 * (sub & 1))) & 0x0F) | (((sh >> (2 * sub)) & 0x03) << 4);
    *scf = ls - 32;
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)(sumi * scf);
  }
};

// ===========================================================================
// 2. O GEMM tilejado
// ===========================================================================
// Grade: x -> N (BN colunas de W = linhas de saida), y -> M (BM tokens por CTA).
// Threads na forma (tr, tc) = (tid/TN, tid%TN); cada thread cobre RM linhas de
// ativacao m = m0 + tr + r*TM e RN colunas de peso n = n0 + tc + c*TN, com o
// mesmo par de acumuladores (int32 por bloco de 32, fp32 final) do motor.
//
// O laco externo e' o pipeline medido no V6:
//   prologo: carrega campos(k0=0) -> dequant+STS em buf0            ; barreira
//   iteracao: carrega campos(k0+BK) [cargas em voo] -> CONTA(buf) ->
//             dequant+STS(buf^1) -> barreira
// ou seja: uma barreira por janela de K e a dequantizacao da janela seguinte
// rodando junto com a conta da atual (o `db` do V6), com as cargas globais
// emitidas ANTES da conta para esconder a latencia (o `pf` do V6).
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB>
__global__ void __launch_bounds__((BM / RM) * (BN / RN), MINB) gemm_i8_kernel(
    const char *__restrict__ W, const block_q8_1 *__restrict__ A, float *__restrict__ C,
    const int M, const int N, const int K, const int act_stride) {
  using Pf = typename TR::Pf;
  constexpr int QK = TR::qk;
  constexpr int BB = TR::block_bytes;
  constexpr int TM = BM / RM;  // threads na dimensao M
  constexpr int TN = BN / RN;  // threads na dimensao N
  constexpr int NTH = TM * TN;
  constexpr int NKB = BK / 32;  // blocos de 32 (da escala) por janela de k
  constexpr int WSB = BK + 16;  // stride de linha da LDS do W (multiplo de 16)
  constexpr int NTASK = (BN * NKB + NTH - 1) / NTH;
  static_assert(BK % 32 == 0, "BK tem de ser multiplo de 32 (o bloco da escala)");
  static_assert(WSB % 16 == 0, "a linha do W na LDS tem de ser 16 B alinhada");
  static_assert(QK % 32 == 0, "");
  static_assert(BM % RM == 0 && BN % RN == 0, "");
  static_assert(NKB <= 8, "");

  __shared__ __align__(16) unsigned char s_w[DBUF ? 2 : 1][BN * WSB];
  __shared__ __align__(16) int2 s_dwsc[DBUF ? 2 : 1][NKB * BN];

  const int tid = threadIdx.x;
  const int tr = tid / TN;
  const int tc = tid % TN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int bpr = K / QK;  // super-blocos por linha de peso

  // ---- staging de uma janela: 1 sub-bloco de 32 pesos por thread ----------
  // Os campos do peso ficam em registrador entre a carga (emitida antes da
  // conta) e o consumo (depois da conta): e' o `pf` do V6.
  Pf pf[NTASK];
  auto pf_load = [&](const int k0) {
#pragma unroll
    for (int i = 0; i < NTASK; ++i) {
      const int task = tid + i * NTH;
      const int row = task / NKB;
      const int kb = task - row * NKB;
      int n = n0 + row;
      if (n >= N) n = N - 1;  // linha repetida: as colunas extra nunca sao escritas
      const int kk = k0 + 32 * kb;
      pf[i] = TR::load(W + ((std::int64_t)n * bpr + (kk >> 8)) * BB, (kk >> 5) & 7);
    }
  };
  auto pf_store = [&](const int buf, const int k0) {
#pragma unroll
    for (int i = 0; i < NTASK; ++i) {
      const int task = tid + i * NTH;
      const int row = task / NKB;
      const int kb = task - row * NKB;
      float dw;
      int scf;
      const int kk = k0 + 32 * kb;
      TR::store(pf[i], (kk >> 5) & 7, (int *)(void *)(s_w[buf] + row * WSB + kb * 32), &dw, &scf);
      s_dwsc[buf][kb * BN + row] = make_int2(__float_as_int(dw), scf);
    }
  };

  // ---- linhas de ativacao deste thread -----------------------------------
  int mrow[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    int m = m0 + tr + r * TM;
    mrow[r] = (m < M) ? m : (M - 1);
  }
  const int nbase = n0 + tc;

  float F[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) F[r][c] = 0.0f;

  // ---- prologo -----------------------------------------------------------
  if (DBUF) {
    pf_load(0);
    pf_store(0, 0);
  }
  __syncthreads();

  for (int k0 = 0, cur = 0; k0 < K; k0 += BK, cur ^= 1) {
    const int cb = DBUF ? cur : 0;
    if (DBUF) {
      if (k0 + BK < K) pf_load(k0 + BK);
    } else {
      // buffer unico: a dequantizacao da janela corrente serializa com a conta
      if (k0 > 0) {
        pf_load(k0);
        pf_store(0, k0);
        __syncthreads();
      }
    }

    // ---- conta ----
#pragma unroll
    for (int kb = 0; kb < NKB; ++kb) {
      const int ka = (k0 >> 5) + kb;
      int I[RM][RN];
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) I[r][c] = 0;

      if (WHOLD) {
        int ww[RN][8];
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const int *wp = (const int *)(const void *)(s_w[cb] + (nbase + c * TN) * WSB + kb * 32);
#pragma unroll
          for (int g = 0; g < 8; ++g) ww[c][g] = wp[g];
        }
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          int ar[RM];
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
            ar[r] = ((const int *)(const void *)q->qs)[g];
          }
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) I[r][c] = ggml_cuda_dp4a(ww[c][g], ar[r], I[r][c]);
        }
      } else {
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          int ar[RM];
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
            ar[r] = ((const int *)(const void *)q->qs)[g];
          }
          int wc[RN];
#pragma unroll
          for (int c = 0; c < RN; ++c) {
            const int *wp = (const int *)(const void *)(s_w[cb] + (nbase + c * TN) * WSB + kb * 32);
            wc[c] = wp[g];
          }
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) I[r][c] = ggml_cuda_dp4a(wc[c], ar[r], I[r][c]);
        }
      }

      // ---- correcao (a sequencia do motor: inteiro -> (float) -> fma) ----
      float da[RM];
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
        da[r] = fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
      }
      const int2 *wsc = s_dwsc[cb] + kb * BN;
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        const int2 ws = wsc[nbase + c * TN];
        const float dw = __int_as_float(ws.x);
        const int scf = ws.y;
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          const float d = dw * da[r];
          F[r][c] = __fmaf_rn(d, TR::corr(I[r][c], scf), F[r][c]);
        }
      }
    }

    if (DBUF) {
      if (k0 + BK < K) pf_store(cb ^ 1, k0 + BK);
      __syncthreads();
    } else {
      __syncthreads();
    }
  }

  // ---- epilogo: C[m][n] (token-major, igual ao matvec em lote) ------------
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int m = m0 + tr + r * TM;
    if (m < M) {
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        const int n = nbase + c * TN;
        if (n < N) C[(std::int64_t)m * N + n] = F[r][c];
      }
    }
  }
}

// ===========================================================================
// 3. Lancadores
// ===========================================================================
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB>
inline bool gemm_launch_t(const void *d_w, const block_q8_1 *d_a, float *d_o, std::int64_t nrows,
                          std::int64_t ncols, std::int64_t act_stride, int n_tokens,
                          hipStream_t stream) {
  const dim3 grid((unsigned)((nrows + BN - 1) / BN), (unsigned)((n_tokens + BM - 1) / BM), 1u);
  const int th = (BM / RM) * (BN / RN);
  gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB><<<grid, th, 0, stream>>>(
      (const char *)d_w, d_a, d_o, n_tokens, (int)nrows, (int)ncols, (int)act_stride);
  return hipGetLastError() == hipSuccess;
}

// Diagnostico: registradores/LDS da config que `gemm_launch` escolheria.
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB>
inline bool gemm_attrs_t(hipFuncAttributes &attr, int &lds_bytes) {
  constexpr int NKB = BK / 32;
  constexpr int WSB = BK + 16;
  lds_bytes = (DBUF ? 2 : 1) * (BN * WSB + NKB * BN * 8);
  auto *fn = &gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB>;
  return hipFuncGetAttributes(&attr, (const void *)fn) == hipSuccess;
}

// Escolha de tile por M (medida: em M=16 o GEMM perde para o GEMV de qualquer
// forma -- 1,08x no plano §0.1 --, entao o tile so' precisa acompanhar M para
// nao desperdicar linhas; de M>=64 para cima o tile grande e' o que ganha).
inline int gemm_pick_bm(int n_tokens) { return n_tokens >= 128 ? 128 : (n_tokens >= 64 ? 64 : 16); }

inline bool gemm_launch(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                        std::int64_t nrows, std::int64_t ncols, std::int64_t act_stride,
                        int n_tokens, hipStream_t stream) {
  if (!d_w || !d_a || !d_o) return false;
  if (nrows <= 0 || ncols <= 0 || n_tokens <= 0) return false;
  if (act_stride * 32 < ncols) return false;  // a ativacao tem de cobrir K
  if (nrows > 0x7FFFFFFF || ncols > 0x7FFFFFFF || n_tokens > 0x7FFFFFFF) return false;
  const int bm = gemm_pick_bm(n_tokens);
#define RD_GEMM(Traits, Dt, QK)                                                              \
  case Dt: {                                                                                 \
    if (ncols % QK != 0 || ncols % 64 != 0) return false;                                    \
    if (bm == 128)                                                                           \
      return gemm_launch_t<Traits, 128, 128, 64, 8, 8, true, true, 1>(d_w, d_a, d_o, nrows,   \
                                                                     ncols, act_stride,       \
                                                                     n_tokens, stream);       \
    if (bm == 64)                                                                             \
      return gemm_launch_t<Traits, 64, 128, 64, 4, 8, true, true, 1>(d_w, d_a, d_o, nrows,     \
                                                                    ncols, act_stride,        \
                                                                    n_tokens, stream);        \
    return gemm_launch_t<Traits, 16, 128, 64, 1, 8, true, true, 1>(d_w, d_a, d_o, nrows, ncols, \
                                                                  act_stride, n_tokens, stream); \
  }
  switch (dt) {
    RD_GEMM(SolverIq3S, 12, 256)
    RD_GEMM(SolverIq3XXS, 9, 256)
    RD_GEMM(SolverIq4XS, 14, 256)
    default:
      return false;  // sem kernel: o chamador decide (SPEC 1.3)
  }
#undef RD_GEMM
}

inline bool gemm_attrs(int dt, int n_tokens, hipFuncAttributes &attr, int &lds_bytes) {
  const int bm = gemm_pick_bm(n_tokens);
#define RD_ATTR(Traits, Dt)                                                                  \
  case Dt:                                                                                   \
    if (bm == 128) return gemm_attrs_t<Traits, 128, 128, 64, 8, 8, true, true, 1>(attr, lds_bytes); \
    if (bm == 64) return gemm_attrs_t<Traits, 64, 128, 64, 4, 8, true, true, 1>(attr, lds_bytes);   \
    return gemm_attrs_t<Traits, 16, 128, 64, 1, 8, true, true, 1>(attr, lds_bytes);
  switch (dt) {
    RD_ATTR(SolverIq3S, 12)
    RD_ATTR(SolverIq3XXS, 9)
    RD_ATTR(SolverIq4XS, 14)
    default:
      return false;
  }
#undef RD_ATTR
}

// ---------------------------------------------------------------------------
// Bancada: variantes da MESMA geometria para A/B atribuivel (so' iq3_s, para
// manter o numero de instanciacoes limitado). cfg:
//   0 = producao (a escolha de `gemm_launch` para este M)
//   1 = BM128 BN128 BK64 RM8 RN8, duplo buffer do W, W em registrador (WHOLD)
//   2 = idem 1 sem o duplo buffer (buffer unico, o caminho da frente F)
//   3 = idem 1 com __launch_bounds__(256, 2) (teto de 128 VGPR)
//   4 = idem 1 com o W lido da LDS a cada uso (WHOLD=false)
//   5 = BM64 BN128 BK64 RM4 RN8, duplo buffer
//   6 = BM128 BN64 BK64 RM8 RN4, duplo buffer
// ---------------------------------------------------------------------------
inline bool gemm_launch_cfg(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                            std::int64_t nrows, std::int64_t ncols, std::int64_t act_stride,
                            int n_tokens, int cfg, hipStream_t stream) {
  if (cfg == 0) return gemm_launch(dt, d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
  if (dt != 12) return false;  // as variantes de A/B existem so' para iq3_s
  if (ncols % 256 != 0 || ncols % 64 != 0) return false;
  switch (cfg) {
    case 1:
      return gemm_launch_t<SolverIq3S, 128, 128, 64, 8, 8, true, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 2:
      return gemm_launch_t<SolverIq3S, 128, 128, 64, 8, 8, false, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 3:
      return gemm_launch_t<SolverIq3S, 128, 128, 64, 8, 8, true, true, 2>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 4:
      return gemm_launch_t<SolverIq3S, 128, 128, 64, 8, 8, true, false, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 5:
      return gemm_launch_t<SolverIq3S, 64, 128, 64, 4, 8, true, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 6:
      return gemm_launch_t<SolverIq3S, 128, 64, 64, 8, 4, true, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    default:
      return false;
  }
}

}  // namespace rdna4
