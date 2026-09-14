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
// ---------------------------------------------------------------------------
// O QUE A MEDICAO MUDOU NA RECEITA (o numero esta' em
// `bench-gemm-engine-gpu`, janela unica, piso de ruido <= 1 %):
//
//   * **BM=64, nao BM=128.** Com RM=RN=8 e RM=4/RN=8 a MESMA janela da'
//     11,20 T (M=128) e 13,80 T (M=512) contra 15,50 e 17,35 do BM=64. A causa
//     esta' na ISA: RM=RN=8 com BM=128 usa 256 VGPRs e transborda 56 B/lane
//     para `scratch_*`; RM=4/RN=8 com BM=64 cabe em 157 VGPRs com ZERO spill.
//     (No caminho f16 da frente G o BM=128 ganhava -- o custo de registrador do
//     dp4a com RM=RN=8 e' que decide aqui.)
//   * **`#pragma unroll 1` no laco dos dois blocos de 32.** Desenrolado, o
//     alocador mantem A e W das DUAS janelas em registrador: 238 acessos a
//     `scratch_*` por corpo de laco (11 % das instrucoes) contra 10 (0,9 %).
//   * **2 CTAs/CU nao paga**: a config de 105 VGPRs com 12 KB de LDS (cfg 7)
//     faz 13,86 T contra 15,50 do 1 CTA/CU -- o mesmo que a frente G mediu com
//     o V4 (mais CTAs por CU, sozinho, nao e' alavanca).
//   * **O STAGING CRU COM A DECODIFICACAO NO CONSUMO PERDE 1,7-4,8x** (medido
//     nesta frente com `--cfg 9,10`, mesma janela da producao, min de 5 reps,
//     piso de ruido 1,000x). E' a receita do NInfer
//     (`q4_rowsplit_gemm_simt.cuh:77-110` deposita bytes crus com copia
//     vetorial e decodifica dentro do consumo) transplantada para ca: o
//     staging grava 1 int4 de campos crus por sub-bloco e o consumo refaz LUT,
//     sinal, `d_w` e `1+2*sc`. T-MAC/s (M=64/128/512), producao -> cru:
//       iq3_s  blk.3.ffn_up  (5120x17408): 13,41/15,06/16,48 -> 6,23/7,28/7,99
//       iq3_xxs blk.5.ffn_up (5120x17408): 11,11/12,46/13,49 -> 5,59/6,59/7,28
//       iq4_xs blk.2.attn_qkv(5120x10240): 13,33/16,15/17,18 -> 7,60/9,39/10,31
//     A bit-exatidao contra o `vec_dot_*` FICA (0 de 4096, max ulp 0, nos tres
//     tipos): o caminho cru desembrulha o record e chama o MESMO `store()`.
//     A causa esta' medida, nao suposta:
//       (a) o deposito JA' E' VETORIAL. Na ISA da producao, a janela inteira
//           (2 blocos de 32) usa 16 `ds_load_b128` + 8 `ds_load_2addr_b64` no
//           consumo e 2 `ds_store_b128` + 1 `ds_store_b64` no staging. Nao ha'
//           round-trip escalar para remover;
//       (b) a decodificacao no staging NAO custa tempo: o modo "so' staging"
//           (cfg 11 -- mesma grade, mesmos threads, mesmo prologo, conta
//           trocada por 4 leituras da LDS) da' 17,5-18,3 % do kernel em M>=64,
//           e esse tempo e' o trafego COMPULSORIO do peso (281,5 KB por CTA;
//           76,6 MB em 0,134 ms em M=128 = 572 GB/s, ~90 % dos ~633 GB/s do
//           cartao). O staging cru (cfg 12) tem 2,25x MENOS instrucoes (198
//           contra 446 na ISA) e NAO e' mais rapido: 0,058 contra 0,076 ms em
//           M=64, 0,144 contra 0,134 em M=128, 1,179 contra 0,512 em M=512;
//       (c) o preco esta' no consumo: a LUT passa a ser consultada por TM =
//           BM/RM = 16 threads em vez de uma, entao as consultas saem de 8 por
//           janela (staging) para 136 (consumo, 17x) e as instrucoes vivas por
//           janela vao de 1126 para 3114 (+177 %, contadas na ISA), com os
//           MESMOS 512 dp4a e o MESMO bloco `ww[RN][8]` de registradores: LDS
//           de 40 para 16 acessos, ALU de ~500 para ~2200.
//     A fracao de staging cai de 17,5 % para 9,2 % em M=128 -- NAO porque o
//     staging ficou mais barato (0,134 -> 0,144 ms), e sim porque o consumo
//     ficou 2,3x mais lento. Fracao de staging sozinha nao e' criterio.
//
// Orcamento de LDS (BM=64, BN=128, BK=64, duplo buffer do W + os campos d_w/sc):
//   s_w   2 * 128 * (64+16) B  = 20 480 B
//   s_dwsc 2 * 2 * 128 * 8 B   =  4 096 B
//   total                      = 24 576 B <= 32 KB (caberiam 2 CTAs/CU em LDS; o
//   que decide a ocupacao e' o VGPR -- 157 em 256 threads = 1 CTA/CU, que e' o
//   medido como melhor. `gemm_attrs` reporta os dois numeros.)
//
// O stride de linha do W na LDS e' BK+16 (80 B) e as colunas de cada thread sao
// ESTRIDADAS (n = tc + c*TN): assim as 16 (ou 32) linhas lidas por uma warp caem
// em 16 bancos distintos e nao ha' conflito nem de 4 B nem de 16 B.
//
// API publica: `gemm_launch`, espelhando `matvec_launch_batch` para o grafo
// poder trocar um pelo outro. `false` = nao suportado (o chamador decide); nunca
// ha' queda silenciosa para outro caminho.
//
// ===========================================================================
// COBERTURA k-QUANT (q2_K, q3_K, q4_K, q5_K, q6_K = 21,8 % dos bytes do modelo)
// ===========================================================================
// MEDIDO em 14/09 (a janela esta' em `bench-gemm-engine-gpu --kq`, min de 5 reps,
// piso de ruido 1,000x, DPM aquecido, `hipGetLastError` depois de todo lancamento,
// UM tensor REAL por tipo escolhido pelo DTYPE -- o mesmo nome de projecao e' um
// tipo diferente em camadas diferentes). T-MAC/s contra o GEMV em lote que embarca
// (sub-lote 64, o caminho do grafo hoje em n>16 era sub-lote 16):
//
//   tipo  tensor (K x N)             M=16   M=64   M=128  M=512  |  x GEMV64 (16/64/128/512)
//   q2_K  blk.3.attn_q  5120x12288    6,33   9,09   9,41  10,05  |  2,11 3,51 3,63 3,85
//   q3_K  blk.7.ffn_down 17408x5120   3,35   6,96   9,59  11,97  |  1,11 2,54 3,45 4,38
//   q4_K  blk.63.ffn_down 17408x5120  4,26   7,49  10,39  12,81  |  2,00 2,97 4,12 6,39
//   q5_K  output.weight 5120x248320   6,87  11,41  11,44  11,51  |  2,70 3,66 3,69 3,73
//   q6_K  blk.64.ffn_down 17408x5120  3,60   7,46  10,25  12,49  |  1,49 3,33 4,62 5,62
//
// O GEMV em lote nesses tipos e' MAIS LENTO do que nos tipos IQ (2,0-3,1 T-MAC/s
// contra 5-6), entao a razao e' maior do que a que o plano estimou. Em M=16 o
// GEMM so' ganha com folga nos tipos de 32 bits por peso (q4_K/q5_K); o grafo nao
// usa essa geometria para n <= 16 (vai de GEMV em lote), mas usa para n = 17..63.
//
// TOLERANCIA DECLARADA (o que substitui a bit-exatidao dos tipos IQ):
//   contra o `vec_dot_*_q8_1` DO MOTOR, M=16 N=256 (4096 elementos), mesma janela:
//     q2_K rel-L2 4,51e-07 max|d| 5,72e-06 | q3_K 7,39e-07 / 1,34e-05
//     q4_K rel-L2 8,44e-07 max|d| 1,53e-05 | q5_K 4,69e-07 / 5,48e-06
//     q6_K rel-L2 9,23e-07 max|d| 1,53e-05
//     (todos os 4096 elementos divergem em bits, como esperado: a divergencia e'
//      de ARREDONDAMENTO, nao de valor)
//   contra o `matvec_launch_batch` que embarca, nos tensores INTEIROS das tabelas
//   acima, M=16..512: rel-L2 3,38e-07..6,15e-07, max|d| 7,6e-06..3,05e-05.
//   (os tipos IQ sao 0 bits e max ulp 0 contra o vec_dot; contra o GEMV em lote a
//    diferenca documentada e' 2,4e-07 / 3,3e-06 por projecao -- os k-quants ficam
//    na MESMA ordem, ~1,5x em rel-L2 e ate ~3x em max|d|.)
//
// Por que nao da' para ser bit-exato aqui: ver a secao 1b (o motor arredonda por
// termo de 4 elementos nas DUAS cadeias; o acumulador inteiro colapsa o bloco).
//
// Custo: VGPR 189/236/203/229/219 (q2/q3/q4/q5/q6, BM=64) contra 157 do iq3_s,
// LDS 28 672 B (16 B de record por (linha, bloco de k) contra 8 B) e ZERO spill em
// todos os cinco (`hipFuncGetAttributes` e a ISA, mesma janela).
//
// O QUE FALHOU NO CAMINHO (registrado porque cada um custou uma corrida):
//   1. `ref_engine_kernel` da bancada somava 8 chamadas por super-bloco, o que so'
//      esta' certo para qi/vdr == 8 (os tres tipos IQ); nos k-quants qi/vdr = 16
//      (q2_K..q5_K) e 32 (q6_K), entao o oraculo somava METADE do super-bloco.
//   2. A transcricao do `dequantize_q3_K` na bancada perdeu o `32*n` do ponteiro
//      `qs`: reprovou 751 de 2048 pesos de um staging CORRETO.
//   3. O bit alto do q6_K e' o par de bits (2*(sub%4)) do byte de `qh`, nao
//      (2*(sub%2)): 735 de 2048 pesos errados de verdade. Achei reproduzindo o
//      layout em CPU com bytes ALEATORIOS (sem GPU), que e' o jeito barato de
//      iterar nisso.
// ===========================================================================
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
  static constexpr bool is_kq = false;  // cadeia unica: caminho de producao antigo
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
      // `signs[4*sub + il]` tem 8 bits de sinal: os 4 BAIXOS assinam os 4 bytes
      // de `g1` e os 4 ALTOS os de `g2` (e' a mesma separacao que o motor faz
      // com `(sg & 0x03) << 7 | (sg & 0x0C) << 21` e `(sg & 0x30) << 3 |
      // (sg & 0xC0) << 17`, so' que ja' espalhada nos bits 7/8/23/24).
      const int m0 = gemm_sign_mask4((unsigned)sg[il] & 0x0Fu);
      const int m1 = gemm_sign_mask4(((unsigned)sg[il] >> 4) & 0x0Fu);
      dst[2 * il + 0] = (g1 ^ m0) + (m0 & 0x01010101);
      dst[2 * il + 1] = (g2 ^ m1) + (m1 & 0x01010101);
    }
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)(sumi * scf);
  }

  // ---- caminho "raw" (cfg 9/10 da bancada): o staging guarda os CAMPOS CRUS e
  // a decodificacao acima acontece no laco de consumo ------------------------
  // O record de 16 B cabe em int4 (1 ds_load_b128 por coluna). O `dsc` (d nos
  // bits 0-15, byte de scales nos 16-23) desliza 8 bits para a esquerda: o byte
  // de `qh` entra nos bits 0-7, `d` fica em 8-23 e o byte de scales em 24-31.
  // O custo e' 1 shift + 1 or no staging e 1 shift no consumo.
  static constexpr int raw_words = 4;
  static __device__ __forceinline__ void raw_pack(const Pf &p, int *rec) {
    rec[0] = p.qs0;
    rec[1] = p.qs1;
    rec[2] = (int)p.sg;
    rec[3] = (int)((p.qh & 0xFFu) | (p.dsc << 8));
  }
  // Desembrulha o record e chama o MESMO `store()` -- a aritmetica do caminho
  // raw e' a do caminho de producao por CONSTRUCAO, nao por coincidencia.
  static __device__ __forceinline__ void decode_raw(const int *rec, const int sub, int *dst,
                                                    float *dw, int *scf) {
    Pf p;
    const unsigned w3 = (unsigned)rec[3];
    p.qs0 = rec[0];
    p.qs1 = rec[1];
    p.sg = (unsigned)rec[2];
    p.qh = w3 & 0xFFu;
    p.dsc = w3 >> 8;
    store(p, sub, dst, dw, scf);
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
  static constexpr bool is_kq = false;
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

  // ---- caminho "raw" (cfg 9/10): `Pf` JA' tem 4 palavras (16 B) -----------
  static constexpr int raw_words = 4;
  static __device__ __forceinline__ void raw_pack(const Pf &p, int *rec) {
    rec[0] = p.qs0;
    rec[1] = p.qs1;
    rec[2] = (int)p.aux;
    rec[3] = (int)p.d;
  }
  static __device__ __forceinline__ void decode_raw(const int *rec, const int /*sub*/, int *dst,
                                                    float *dw, int *scf) {
    Pf p;
    p.qs0 = rec[0];
    p.qs1 = rec[1];
    p.aux = (unsigned)rec[2];
    p.d = (unsigned)rec[3];
    store(p, 0, dst, dw, scf);
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
  static constexpr bool is_kq = false;
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
    // `get_int_from_table_16` do motor, sobre CADA grupo de 4 bytes:
    //   .x = os nibbles BAIXOS dos 4 bytes do grupo, .y = os ALTOS.
    // A ordem de k vem do `dequantize_iq4_xs` do proprio motor
    // (dequant.cuh:301-315):
    //   y[4*il + j]      = kvalues[qs[16*ib+4*il+j] & 0xf]   <- nibble BAIXO
    //   y[16 + 4*il + j] = kvalues[qs[16*ib+4*il+j] >> 4]    <- nibble ALTO
    // isto e': k = 0..15 sao os nibbles BAIXOS de qs[16*sub + k] e k = 16..31 os
    // ALTOS de qs[16*sub + (k-16)]. Associar `.x`/`.y` do MESMO grupo (a primeira
    // versao desta frente fazia isso) da' 4096 de 4096 elementos errados -- foi o
    // que a verificacao contra o `vec_dot_iq4_xs_q8_1` pegou.
    const int2 g0 = get_int_from_table_16(p.q0, kvalues_iq4nl);  // bytes 0..3
    const int2 g1 = get_int_from_table_16(p.q1, kvalues_iq4nl);  // bytes 4..7
    const int2 g2 = get_int_from_table_16(p.q2, kvalues_iq4nl);  // bytes 8..11
    const int2 g3 = get_int_from_table_16(p.q3, kvalues_iq4nl);  // bytes 12..15
    dst[0] = g0.x;
    dst[1] = g1.x;
    dst[2] = g2.x;
    dst[3] = g3.x;
    dst[4] = g0.y;
    dst[5] = g1.y;
    dst[6] = g2.y;
    dst[7] = g3.y;
    *dw = fp16_to_float((uint16_t)(p.dsl & 0xFFFFu));
    const int sl = (int)((p.dsl >> 16) & 0xFFu);
    const int sh = (int)p.sh;
    const int ls = ((sl >> (4 * (sub & 1))) & 0x0F) | (((sh >> (2 * sub)) & 0x03) << 4);
    *scf = ls - 32;
  }
  static __device__ __forceinline__ float corr(const int sumi, const int scf) {
    return (float)(sumi * scf);
  }

  // ---- caminho "raw" (cfg 9/10): 4 palavras de nibbles + dsl/sh = 6 --------
  // (nao cabe em 16 B: o record e' 1 int4 + 1 int2, com o slot alinhado a 16 B)
  static constexpr int raw_words = 6;
  static __device__ __forceinline__ void raw_pack(const Pf &p, int *rec) {
    rec[0] = p.q0;
    rec[1] = p.q1;
    rec[2] = p.q2;
    rec[3] = p.q3;
    rec[4] = (int)p.dsl;
    rec[5] = (int)p.sh;
  }
  static __device__ __forceinline__ void decode_raw(const int *rec, const int sub, int *dst,
                                                    float *dw, int *scf) {
    Pf p;
    p.q0 = rec[0];
    p.q1 = rec[1];
    p.q2 = rec[2];
    p.q3 = rec[3];
    p.dsl = (unsigned)rec[4];
    p.sh = (unsigned)rec[5];
    store(p, sub, dst, dw, scf);
  }
};

// ===========================================================================
// 1b. k-quants (q2_K..q6_K): QUANT CRU na LDS + escala/minimo no record
// ===========================================================================
// Diferenca estrutural em relacao aos tipos IQ ja' cobertos: o peso do motor
// NAO e' `d_w * q` com um unico fator inteiro por bloco de 32. Sao tres formas
//
//   q2_K, q4_K, q5_K:  w_e = d*q_e*sc_g - dmin*m_g          (DUAS cadeias fp32)
//   q3_K:              w_e = d*(us_g - 32)*(q_e + (hmask ? 0 : -4))
//   q6_K:              w_e = d*sc_g*(q_e - 32)
//
// com escala por 16 elementos (q2_K, q3_K, q6_K: duas por bloco de 32 -> dois
// acumuladores int, `ngrp = 2`) ou por 32 (q4_K, q5_K: uma, `ngrp = 1`).
//
// O que a LDS guarda, portanto, e' o QUANT CRU na ordem de k (1 int8 por peso,
// 8 palavras por bloco de 32, exatamente o layout do caminho IQ -- o `ws` da
// LDS e' o MESMO) e o record `KqScal` guarda (d, dmin, escala(s)). A correcao
// continua INTEIRA sobre o acumulador do bloco de 32:
//
//   X = I0*s0 + I1*s1   (produto de inteiros EXATO, |X| < 2^24 nos 5 tipos)
//   Y = S0*m0 + S1*m1   (idem; S = soma da ATIVACAO do bloco, por TOKEN)
//   F = fma(dw*da, X, F) ; F = fma(-(dmin*da), Y, F)
//
// O termo do minimo precisa de `S = sum a_e`, que e' por token e nao por coluna:
// custa RM*8 dp4a por bloco contra RM*RN*8 do produto (+12,5 % com RM=4/RN=8).
//
// ---------------------------------- BIT-EXATIDAO ---------------------------
// NAO e' bit-exato e nao da' para ser, com o acumulador inteiro: o motor soma,
// por chamada de `vec_dot`, o termo `d8*(dp4a(4 pesos)*sc)` -- 8 arredondamentos
// fp32 por bloco de 32 em `sumf_d` (mais 8 em `sumf_m`) -- enquanto o acumulador
// inteiro colapsa o bloco inteiro num produto exato com UM arredondamento por
// escala. Reproduzir a sequencia exigiria os 8 `dot_m` de 4 elementos SEPARADOS
// (`vec_dot_q4_K_q8_1_impl_vmmq` deixa o dot por par de palavras: 8 valores int
// por elemento de saida por bloco) + ~2x16 fma/4 pesos, i.e. 2x os acumuladores
// fp32 e ~4x a correcao. Medido (bancada, M=16 N=256): a divergencia fica em
// ulps de fp32 -- ver `bench-gemm-engine-gpu`, secao de tolerancia.
// ---------------------------------------------------------------------------
//
// Alinhamento: os blocos q3_K (110 B) e q6_K (210 B) NAO sao multiplos de 4, o
// que faz o stride de linha sair 4-alinhado apenas quando `bpr` e' par. As cargas
// do staging sao de 32 bits (o mesmo que o caminho iq3_s, cujo bloco tambem tem
// 110 B e cujo staging e' validado byte a byte); cargas vetoriais de 16 B ficam
// de fora porque o bloco nao garante 16 B de alinhamento.
struct KqScal {
  float dw;    // d (ou dm.d)
  float dmn;   // dmin (0.0f nos tipos de cadeia unica)
  int s0, s1;  // escala(s) do bloco de 32: s0 -> elementos 0..15, s1 -> 16..31
};

// Carga de N palavras de 32 bits sem prometer alinhamento: os blocos q3_K (110 B)
// e q6_K (210 B) NAO sao multiplos de 4, entao um `(const int *)` mentiria sobre
// o alinhamento (o caminho iq3_s, de bloco 110 B, ja' faz essa leitura de 32 bits
// e o staging dele e' validado byte a byte). Larga de 16 B fica de fora: o bloco
// nao garante 16 B de alinhamento.
template <int N>
static __device__ __forceinline__ void kq_ld8(const void *p, int *dst) {
  const unsigned char *b = (const unsigned char *)p;
#pragma unroll
  for (int i = 0; i < N; ++i) {
    unsigned w;
    __builtin_memcpy(&w, b + 4 * i, 4);
    dst[i] = (int)w;
  }
}

// ---- q2_K (84 B / 256 pesos, 0,97 % dos bytes deste modelo) ---------------
// 16 sub-blocos de 16 pesos, escala E minimo de 4 bits: w = d*sc*q - dmin*m.
// Bloco de 32 t: elementos e = 0..31 em qs[32*(t/4) + e], campo (t%4) de 2 bits;
// grupo g = 2t + e/16 (o low/high nibble de scales[g]).
struct SolverQ2K {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_q2_K);
  static constexpr bool is_kq = true;
  static constexpr int ngrp = 2;
  static constexpr bool has_min = true;
  static constexpr int raw_words = 4;  // so' para o constexpr RSTR do kernel
  struct Pf {
    int q[8];     // qs[32*(sub/4) .. +32)
    unsigned dms; // half2 {d, dmin}
    unsigned sc2; // scales[2*sub] | scales[2*sub+1] << 8
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_q2_K *b = (const block_q2_K *)blk;
    Pf p;
    kq_ld8<8>((const void *)(b->qs + 32 * (sub >> 2)), p.q);
    p.dms = *(const unsigned *)(const void *)&b->dm;
    p.sc2 = (unsigned)((const uint16_t *)(const void *)b->scales)[sub];
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, KqScal &k) {
    const int sh = 2 * (sub & 3);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = (p.q[i] >> sh) & 0x03030303;
    k.dw = fp16_to_float((uint16_t)(p.dms & 0xFFFFu));
    k.dmn = fp16_to_float((uint16_t)(p.dms >> 16));
    const int g0 = (int)(p.sc2 & 0xFFu), g1 = (int)((p.sc2 >> 8) & 0xFFu);
    k.s0 = (g0 & 0x0F) | ((g0 >> 4) << 16);
    k.s1 = (g1 & 0x0F) | ((g1 >> 4) << 16);
  }
  static __device__ __forceinline__ void scm(const KqScal &k, int &s0, int &m0, int &s1,
                                             int &m1) {
    s0 = k.s0 & 0xFFFF;
    m0 = (k.s0 >> 16) & 0xFFFF;
    s1 = k.s1 & 0xFFFF;
    m1 = (k.s1 >> 16) & 0xFFFF;
  }
};

// ---- q3_K (110 B / 256, 7,87 %): 6 bits de escala por 16 pesos, SEM minimo ---
// w = d*(us-32)*(q + (hmask ? 0 : -4)), q de 2 bits.
// Bloco t: qs[32*(t/4) + e] campo (t%4) de 2 bits; hmask[e] bit t; grupo 2t+e/16.
// As 16 escalas de 6 bits moram em scales[12]: nibble baixo de scales[0..7] =
// 4 bits baixos das escalas 0..7, nibble alto das MESMAS = escalas 8..15, e
// bytes 8..11 = 2 bits altos das escalas {k, k+4, k+8, k+12} (k = indice%4).
struct SolverQ3K {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_q3_K);
  static constexpr bool is_kq = true;
  static constexpr int ngrp = 2;
  static constexpr bool has_min = false;
  static constexpr int raw_words = 4;
  struct Pf {
    int q[8];    // qs[32*(sub/4) .. +32)
    int hm[8];   // hmask[0..32)
    unsigned d;  // d (fp16) do super-bloco
    unsigned sc2;  // uint16 das escalas baixas | uint16 dos 2 bits altos << 16
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_q3_K *b = (const block_q3_K *)blk;
    Pf p;
    kq_ld8<8>((const void *)(b->qs + 32 * (sub >> 2)), p.q);
    kq_ld8<8>((const void *)b->hmask, p.hm);
    p.d = b->d;
    const int is = 2 * sub;
    const uint8_t *sc = b->scales;
    const unsigned lo =
        (unsigned)((const uint16_t *)(const void *)(sc + (is < 8 ? is : is - 8)))[0];
    const unsigned hi = (unsigned)((const uint16_t *)(const void *)(sc + 8 + (is & 3)))[0];
    p.sc2 = lo | (hi << 16);
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, KqScal &k) {
    const int sh = 2 * (sub & 3);
    const int bit = sub;  // o bit do hmask deste bloco de 32
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int q2 = (p.q[i] >> sh) & 0x03030303;
      const int m1 = (p.hm[i] >> bit) & 0x01010101;  // 1 onde o bit do hmask e' 1
      const int s4 = (~m1 & 0x01010101) << 2;        // 4 onde ele e' 0
      dst[i] = __vsubss4(q2, s4);
    }
    k.dw = fp16_to_float((uint16_t)p.d);
    k.dmn = 0.0f;
    const int is = 2 * sub, quad = is >> 2, sh2 = 2 * quad;
    const unsigned lo = p.sc2 & 0xFFFFu, hi = p.sc2 >> 16;
    const int l0 = (is < 8) ? (int)(lo & 0x0Fu) : (int)((lo >> 4) & 0x0Fu);
    const int l1 = (is < 8) ? (int)((lo >> 8) & 0x0Fu) : (int)((lo >> 12) & 0x0Fu);
    k.s0 = (l0 | (((int)(hi >> sh2) & 3) << 4)) - 32;
    k.s1 = (l1 | (((int)(hi >> (sh2 + 8)) & 3) << 4)) - 32;
  }
  static __device__ __forceinline__ void scm(const KqScal &k, int &s0, int &m0, int &s1,
                                             int &m1) {
    s0 = k.s0;
    m0 = 0;
    s1 = k.s1;
    m1 = 0;
  }
};

// ---- q4_K (144 B / 256, 1,94 %): 8 sub-blocos de 32, escala e minimo de 6 bits
// (get_scale_min_k4, o empacotamento de K_SCALE_SIZE=12 bytes). Duas cadeias.
// Bloco t: elementos e em qs[32*(t/2) + e], nibble (t%2); grupo = t.
struct SolverQ4K {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_q4_K);
  static constexpr bool is_kq = true;
  static constexpr int ngrp = 1;
  static constexpr bool has_min = true;
  static constexpr int raw_words = 4;
  struct Pf {
    int q[8];    // qs[32*(sub/2) .. +32)
    int sc[3];   // scales[0..12)
    unsigned dms;
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_q4_K *b = (const block_q4_K *)blk;
    Pf p;
    kq_ld8<8>((const void *)(b->qs + 32 * (sub >> 1)), p.q);
    kq_ld8<3>((const void *)b->scales, p.sc);
    p.dms = *(const unsigned *)(const void *)&b->dm;
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, KqScal &k) {
    const int sh = 4 * (sub & 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[i] = (p.q[i] >> sh) & 0x0F0F0F0F;
    k.dw = fp16_to_float((uint16_t)(p.dms & 0xFFFFu));
    k.dmn = fp16_to_float((uint16_t)(p.dms >> 16));
    // `get_scale_min_k4` (dequant.cuh:91): j = sub
    const int j = sub;
    const uint32_t w0 = (uint32_t)p.sc[0], w1 = (uint32_t)p.sc[1], w2 = (uint32_t)p.sc[2];
    const int shl = (j & 3) * 8;
    uint32_t wj = (j < 4) ? w0 : w1;  // byte j
    const uint32_t wj4 = (j < 4) ? w1 : w2;  // byte j+4
    const uint32_t wjm4 = w0;                // byte j-4 (so' para j >= 4)
    const int bj = (int)((wj >> shl) & 0xFFu);
    const int bj4 = (int)((wj4 >> shl) & 0xFFu);
    const int bjm4 = (int)((wjm4 >> shl) & 0xFFu);
    const int sc6 = (j < 4) ? (bj & 63) : ((bj4 & 0x0F) | ((bjm4 >> 6) << 4));
    const int m6 = (j < 4) ? (bj4 & 63) : ((bj4 >> 4) | ((bj >> 6) << 4));
    k.s0 = sc6 | (m6 << 16);
    k.s1 = 0;
  }
  static __device__ __forceinline__ void scm(const KqScal &k, int &s0, int &m0, int &s1,
                                             int &m1) {
    s0 = k.s0 & 0xFFFF;
    m0 = (k.s0 >> 16) & 0xFFFF;
    s1 = 0;
    m1 = 0;
  }
};

// ---- q5_K (176 B / 256, 8,19 %): como q4_K + o 5o bit em qh[e], bit t -------
struct SolverQ5K {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_q5_K);
  static constexpr bool is_kq = true;
  static constexpr int ngrp = 1;
  static constexpr bool has_min = true;
  static constexpr int raw_words = 4;
  struct Pf {
    int qs[8];  // qs[32*(sub/2) .. +32)
    int qh[8];  // qh[0..32)
    int sc[3];
    unsigned dms;
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_q5_K *b = (const block_q5_K *)blk;
    Pf p;
    kq_ld8<8>((const void *)(b->qs + 32 * (sub >> 1)), p.qs);
    kq_ld8<8>((const void *)b->qh, p.qh);
    kq_ld8<3>((const void *)b->scales, p.sc);
    p.dms = *(const unsigned *)(const void *)&b->dm;
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, KqScal &k) {
    const int sh = 4 * (sub & 1);
    const int bit = sub;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int v = (p.qs[i] >> sh) & 0x0F0F0F0F;
      const int hb = (p.qh[i] >> bit) & 0x01010101;
      dst[i] = v | (hb << 4);
    }
    k.dw = fp16_to_float((uint16_t)(p.dms & 0xFFFFu));
    k.dmn = fp16_to_float((uint16_t)(p.dms >> 16));
    const int j = sub;
    const uint32_t w0 = (uint32_t)p.sc[0], w1 = (uint32_t)p.sc[1], w2 = (uint32_t)p.sc[2];
    const int shl = (j & 3) * 8;
    const uint32_t wj = (j < 4) ? w0 : w1;
    const uint32_t wj4 = (j < 4) ? w1 : w2;
    const int bj = (int)((wj >> shl) & 0xFFu);
    const int bj4 = (int)((wj4 >> shl) & 0xFFu);
    const int bjm4 = (int)((w0 >> shl) & 0xFFu);
    const int sc6 = (j < 4) ? (bj & 63) : ((bj4 & 0x0F) | ((bjm4 >> 6) << 4));
    const int m6 = (j < 4) ? (bj4 & 63) : ((bj4 >> 4) | ((bj >> 6) << 4));
    k.s0 = sc6 | (m6 << 16);
    k.s1 = 0;
  }
  static __device__ __forceinline__ void scm(const KqScal &k, int &s0, int &m0, int &s1,
                                             int &m1) {
    s0 = k.s0 & 0xFFFF;
    m0 = (k.s0 >> 16) & 0xFFFF;
    s1 = 0;
    m1 = 0;
  }
};

// ---- q6_K (210 B / 256, 2,86 %): escala int8 por 16 pesos, quant de 6 bits
// COM sinal deslocado: w = d*sc*(q-32). Bloco t: n = t/4, q = t%2, i2 = (t/2)%2
//   ql[64*n + 32*q + e], nibble i2; qh[32*n + e] bits (2q, 2q+1);
//   grupo 8n + 2q + 4*i2 + e/16 (dois por bloco de 32).
struct SolverQ6K {
  static constexpr int qk = QK_K;
  static constexpr int block_bytes = (int)sizeof(block_q6_K);
  static constexpr bool is_kq = true;
  static constexpr int ngrp = 2;
  static constexpr bool has_min = false;
  static constexpr int raw_words = 4;
  struct Pf {
    int ql[8];
    int qh[8];
    unsigned d;
    unsigned sc2;  // scales[g], scales[g+1] (int8 cada)
  };
  static __device__ __forceinline__ Pf load(const char *blk, const int sub) {
    const block_q6_K *b = (const block_q6_K *)blk;
    Pf p;
    const int n = sub >> 2, q = sub & 1, i2 = (sub >> 1) & 1;
    kq_ld8<8>((const void *)(b->ql + 64 * n + 32 * q), p.ql);
    kq_ld8<8>((const void *)(b->qh + 32 * n), p.qh);
    p.d = b->d;
    const int g = 8 * n + 2 * q + 4 * i2;
    p.sc2 = (unsigned)((const uint16_t *)(const void *)(b->scales + g))[0];
    return p;
  }
  static __device__ __forceinline__ void store(const Pf &p, const int sub, int *dst, KqScal &k) {
    const int i2 = (sub >> 1) & 1, q = sub & 1;
    const int sh = 4 * i2;
    // Os 2 bits altos do quant sao os bits (2*(sub%4)) do byte qh[e] do bloco --
    // NAO (2*(sub%2)): o par de bits anda de 2 em 2 com o indice do bloco DENTRO
    // do grupo de 4 (o `vh_shift`/`4*i` do `vec_dot_q6_K_q8_1` e o `{0,2,4,6}[j]`
    // do `dequantize_q6_K` sao a mesma coisa). Trocar isso deixa 36 % dos pesos
    // errados -- foi o que a bancada pegou (735 de 2048).
    const int hsh = 2 * (sub & 3);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int v = ((p.ql[i] >> sh) & 0x0F0F0F0F) | (((p.qh[i] >> hsh) & 0x03030303) << 4);
      dst[i] = __vsubss4(v, 0x20202020);  // q - 32, sem saturacao (0..63 -> -32..31)
    }
    k.dw = fp16_to_float((uint16_t)p.d);
    k.dmn = 0.0f;
    k.s0 = (int)(int8_t)(p.sc2 & 0xFFu);
    k.s1 = (int)(int8_t)((p.sc2 >> 8) & 0xFFu);
  }
  static __device__ __forceinline__ void scm(const KqScal &k, int &s0, int &m0, int &s1,
                                             int &m1) {
    s0 = k.s0;
    m0 = 0;
    s1 = k.s1;
    m1 = 0;
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
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB,
          bool RAW = false, bool STAGE_ONLY = false>
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
  // Caminho RAW: 1 record de campos crus por (linha, bloco de 32). O slot e' o
  // record arredondado para cima em palavras de 16 B, entao todo offset de
  // record e' 16 B alinhado e a leitura cabe em ds_load_b128.
  constexpr int RWW = TR::raw_words;
  constexpr int RSLOT = (RWW + 3) & ~3;
  constexpr int RSTR = NKB * RSLOT;  // palavras por linha do tile do W
  constexpr int NBUF = DBUF ? 2 : 1;
  static_assert(BK % 32 == 0, "BK tem de ser multiplo de 32 (o bloco da escala)");
  static_assert(WSB % 16 == 0, "a linha do W na LDS tem de ser 16 B alinhada");
  static_assert(QK % 32 == 0, "");
  static_assert(BM % RM == 0 && BN % RN == 0, "");
  static_assert(NKB <= 8, "");
  static_assert(!RAW || (RWW >= 4 && RWW <= 6), "o record raw tem de ser 1 int4 (+1 int2)");
  static_assert(!RAW || WHOLD, "o caminho raw decodifica para o bloco de registradores (WHOLD)");

  __shared__ __align__(16) unsigned char s_w[RAW ? 16 : NBUF * BN * WSB];
  __shared__ __align__(16) int2 s_dwsc[TR::is_kq ? 1 : (RAW ? 1 : NBUF * NKB * BN)];
  __shared__ __align__(16) KqScal s_kq[TR::is_kq ? NBUF * NKB * BN : 1];
  __shared__ __align__(16) int s_wr[RAW ? NBUF * BN * RSTR : 4];

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
      const int kk = k0 + 32 * kb;
      if constexpr (RAW) {
        // Staging CRU: 1 int4 (16 B) por sub-bloco, sem LUT, sem mascara de
        // sinal e sem `s_dwsc` -- e' o minimo que o consumidor consegue
        // decodificar, e sai da frente do `__syncthreads` inteiro.
        int rec[RWW];
        TR::raw_pack(pf[i], rec);
        int *dst = s_wr + buf * (BN * RSTR) + row * RSTR + kb * RSLOT;
        *(int4 *)(void *)dst = make_int4(rec[0], rec[1], rec[2], rec[3]);
        if constexpr (RWW > 4) *(int2 *)(void *)(dst + 4) = make_int2(rec[4], rec[5]);
      } else if constexpr (TR::is_kq) {
        // k-quant: o quant CRU vai para a LDS (mesmo layout de 8 palavras do
        // caminho IQ) e (d, dmin, escalas) para o record `KqScal`.
        KqScal k;
        TR::store(pf[i], (kk >> 5) & 7,
                  (int *)(void *)(s_w + buf * (BN * WSB) + row * WSB + kb * 32), k);
        s_kq[buf * (NKB * BN) + kb * BN + row] = k;
      } else {
        float dw;
        int scf;
        TR::store(pf[i], (kk >> 5) & 7,
                  (int *)(void *)(s_w + buf * (BN * WSB) + row * WSB + kb * 32), &dw, &scf);
        s_dwsc[buf * (NKB * BN) + kb * BN + row] = make_int2(__float_as_int(dw), scf);
      }
    }
  };

  // ---- linhas de ativacao deste thread -----------------------------------
  int mrow[RM];
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    int m = m0 + tr + r * TM;
    mrow[r] = (m < M) ? m : (M - 1);
  }
  // ATENCAO: `nloc` e' o indice DENTRO do tile (0..BN-1) e e' ele que indexa a
  // LDS; `n0 + nloc` e' a coluna global, e so' ela indexa C e o peso. Trocar os
  // dois da' um kernel que acerta SO' o CTA de blockIdx.x == 0 (a primeira
  // versao desta frente tinha exatamente esse defeito: o caso pequeno com N=64
  // passava e N=256 divergia em metade dos elementos).
  const int nloc = tc;

  float F[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) F[r][c] = 0.0f;

  // Sonda do modo STAGE_ONLY (a fracao de staging medida pela frente G/H): le'
  // 4 palavras espalhadas do tile depositado na LDS para o compilador nao
  // eliminar o deposito e no fim so' escreve em C se `sink` for um NaN que
  // nunca aparece -- assim o modo mede staging e nao aritmetica.
  float sink = 0.0f;
  (void)sink;

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
    // `#pragma unroll 1`: o desenrolar das DUAS janelas de 32 dobra o conjunto
    // de operandos vivos (o alocador mantem A e W em registrador para as duas
    // ao mesmo tempo) e o kernel passa de 256 VGPRs COM spill: medido na ISA,
    // 238 acessos a `scratch_*` por corpo de laco (11 % das instrucoes) no
    // desenrolado contra 10 (0,9 %) aqui -- e' o mesmo modo de falha que a
    // frente G mediu com o `pf` sem liberar LDS (413 scratch = 0,18x). O custo
    // e' a aritmetica de endereco refeita por bloco (+/- 25 % das instrucoes
    // do corpo), que e' o que a medicao de GPU decide.
#pragma unroll 1
    for (int kb = 0; kb < NKB; ++kb) {
      const int ka = (k0 >> 5) + kb;
      int I[RM][RN];
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) I[r][c] = 0;

      if constexpr (STAGE_ONLY) {
        // "modo 1" da frente G: mesmo prologo, mesmo staging, mesma grade e
        // mesmos threads, com a conta trocada por 4 leituras da LDS.
        constexpr int NW = RAW ? BN * RSTR : (BN * WSB) / 4;
        const int *base = RAW ? (s_wr + cb * (BN * RSTR))
                              : (const int *)(const void *)(s_w + cb * (BN * WSB));
#pragma unroll
        for (int i = 0; i < 4; ++i) sink += (float)base[(tid * 7 + i * 29) % NW];
      } else if constexpr (TR::is_kq) {
        // ---- k-quants: quant CRU na LDS + correcao INTEIRA com escala/minimo --
        // Dois acumuladores int quando a escala muda no meio do bloco de 32
        // (q2_K, q3_K, q6_K: `ngrp = 2`; palavras 0..3 = elementos 0..15) e um so'
        // quando ela e' constante no bloco (q4_K, q5_K). O termo do minimo usa a
        // SOMA DA ATIVACAO do bloco -- `S` e' por TOKEN, nao por coluna, entao
        // custa RM*8 dp4a contra RM*RN*8 do produto. `S` e' exatamente o
        // `dp4a(0x01010101, u)` que o `*_impl_vmmq` do motor usa.
        int I0[RM][RN], I1[RM][RN];
        int Sq0[RM], Sq1[RM];
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          Sq0[r] = 0;
          Sq1[r] = 0;
#pragma unroll
          for (int c = 0; c < RN; ++c) {
            I0[r][c] = 0;
            I1[r][c] = 0;
          }
        }
        int ww[RN][8];
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const int *wp =
              (const int *)(const void *)(s_w + cb * (BN * WSB) + (nloc + c * TN) * WSB + kb * 32);
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
          if constexpr (TR::ngrp == 2) {
            // o `g` esta' desenrolado, entao `g < 4` e' constante de compilacao
            if (g < 4) {
#pragma unroll
              for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c)
                  I0[r][c] = ggml_cuda_dp4a(ww[c][g], ar[r], I0[r][c]);
            } else {
#pragma unroll
              for (int r = 0; r < RM; ++r)
#pragma unroll
                for (int c = 0; c < RN; ++c)
                  I1[r][c] = ggml_cuda_dp4a(ww[c][g], ar[r], I1[r][c]);
            }
          } else {
#pragma unroll
            for (int r = 0; r < RM; ++r)
#pragma unroll
              for (int c = 0; c < RN; ++c)
                I0[r][c] = ggml_cuda_dp4a(ww[c][g], ar[r], I0[r][c]);
          }
          if constexpr (TR::has_min) {
#pragma unroll
            for (int r = 0; r < RM; ++r) {
              if (TR::ngrp == 2 && g >= 4) Sq1[r] = ggml_cuda_dp4a(0x01010101, ar[r], Sq1[r]);
              else                         Sq0[r] = ggml_cuda_dp4a(0x01010101, ar[r], Sq0[r]);
            }
          }
        }
        float da[RM];
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
          da[r] = fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
        }
        const KqScal *ks = s_kq + cb * (NKB * BN) + kb * BN;
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const KqScal k = ks[nloc + c * TN];
          int s0, m0, s1, m1;
          TR::scm(k, s0, m0, s1, m1);
          const float nm = -k.dmn;
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            // X/Y sao produtos de inteiros EXATOS (|X| < 2^24 nos 5 tipos), e a
            // ORDEM e' a do motor: a escala entra antes do `d` do super-bloco
            // (`dm.x*sumf_d` no fim do vec_dot) e o `da` esta' dentro da soma.
            int X, Y = 0;
            if constexpr (TR::ngrp == 2) X = I0[r][c] * s0 + I1[r][c] * s1;
            else                         X = I0[r][c] * s0;
            F[r][c] = __fmaf_rn(k.dw, da[r] * (float)X, F[r][c]);
            if constexpr (TR::has_min) {
              if constexpr (TR::ngrp == 2) Y = Sq0[r] * m0 + Sq1[r] * m1;
              else                         Y = Sq0[r] * m0;
              F[r][c] = __fmaf_rn(nm, da[r] * (float)Y, F[r][c]);
            }
          }
        }
      } else if constexpr (RAW) {
        // ---- decodificacao DENTRO do laco de consumo -----------------------
        // 1 ds_load_b128 por coluna traz os campos crus; a LUT, a mascara de
        // sinal, `d_w` e `1+2*sc` sao refeitos AQUI, e `TR::decode_raw` chama o
        // MESMO `store()` do caminho de producao (a aritmetica e' a mesma por
        // construcao). O quadrado `ww[RN][8]` de registradores e' identico ao
        // do caminho WHOLD: o que muda e' de onde ele vem (LDS decodificada vs
        // LDS crua + decodificacao).
        int ww[RN][8];
        float wdw[RN];
        int wsc[RN];
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const int *src = s_wr + cb * (BN * RSTR) + (nloc + c * TN) * RSTR + kb * RSLOT;
          const int4 r0 = *(const int4 *)(const void *)src;
          int rec[RWW];
          rec[0] = r0.x;
          rec[1] = r0.y;
          rec[2] = r0.z;
          rec[3] = r0.w;
          if constexpr (RWW > 4) {
            const int2 r1 = *(const int2 *)(const void *)(src + 4);
            rec[4] = r1.x;
            rec[5] = r1.y;
          }
          TR::decode_raw(rec, ka & 7, ww[c], &wdw[c], &wsc[c]);
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
        // mesma sequencia de correcao do caminho de producao, com `dw`/`scf`
        // vindos do record cru em vez do `s_dwsc`
        float da[RM];
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
          da[r] = fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
        }
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const float dw = wdw[c];
          const int scf = wsc[c];
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            const float d = dw * da[r];
            F[r][c] = __fmaf_rn(d, TR::corr(I[r][c], scf), F[r][c]);
          }
        }
      } else if (WHOLD) {
        int ww[RN][8];
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const int *wp =
              (const int *)(const void *)(s_w + cb * (BN * WSB) + (nloc + c * TN) * WSB + kb * 32);
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
            const int *wp =
                (const int *)(const void *)(s_w + cb * (BN * WSB) + (nloc + c * TN) * WSB + kb * 32);
            wc[c] = wp[g];
          }
#pragma unroll
          for (int r = 0; r < RM; ++r)
#pragma unroll
            for (int c = 0; c < RN; ++c) I[r][c] = ggml_cuda_dp4a(wc[c], ar[r], I[r][c]);
        }
      }

      // ---- correcao (a sequencia do motor: inteiro -> (float) -> fma) ----
      // (o caminho RAW faz a sua dentro do ramo acima: la' o `dw`/`scf` esta'
      //  no record cru e nao no `s_dwsc`)
      if constexpr (!RAW && !STAGE_ONLY && !TR::is_kq) {
        float da[RM];
#pragma unroll
        for (int r = 0; r < RM; ++r) {
          const block_q8_1 *q = A + (std::int64_t)mrow[r] * act_stride + ka;
          da[r] = fp16_to_float((uint16_t)(q->ds & 0xFFFFu));
        }
        const int2 *wsc = s_dwsc + cb * (NKB * BN) + kb * BN;
#pragma unroll
        for (int c = 0; c < RN; ++c) {
          const int2 ws = wsc[nloc + c * TN];
          const float dw = __int_as_float(ws.x);
          const int scf = ws.y;
#pragma unroll
          for (int r = 0; r < RM; ++r) {
            const float d = dw * da[r];
            F[r][c] = __fmaf_rn(d, TR::corr(I[r][c], scf), F[r][c]);
          }
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
  if constexpr (STAGE_ONLY) {
    // o modo de staging nao escreve saida (so' o NaN impossivel, que nunca
    // acontece -- e' o que impede o compilador de eliminar o deposito)
    if (__float_as_uint(sink) == 0x7F800001u)
      C[(std::int64_t)blockIdx.y * gridDim.x + blockIdx.x] = sink;
    return;
  }
#pragma unroll
  for (int r = 0; r < RM; ++r) {
    const int m = m0 + tr + r * TM;
    if (m < M) {
#pragma unroll
      for (int c = 0; c < RN; ++c) {
        const int n = n0 + nloc + c * TN;
        if (n < N) C[(std::int64_t)m * N + n] = F[r][c];
      }
    }
  }
}

// ===========================================================================
// 3. Lancadores
// ===========================================================================
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB,
          bool RAW = false, bool STAGE_ONLY = false>
inline bool gemm_launch_t(const void *d_w, const block_q8_1 *d_a, float *d_o, std::int64_t nrows,
                          std::int64_t ncols, std::int64_t act_stride, int n_tokens,
                          hipStream_t stream) {
  const dim3 grid((unsigned)((nrows + BN - 1) / BN), (unsigned)((n_tokens + BM - 1) / BM), 1u);
  const int th = (BM / RM) * (BN / RN);
  gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB, RAW, STAGE_ONLY>
      <<<grid, th, 0, stream>>>((const char *)d_w, d_a, d_o, n_tokens, (int)nrows, (int)ncols,
                                (int)act_stride);
  return hipGetLastError() == hipSuccess;
}

// Diagnostico: registradores/LDS da config que `gemm_launch` escolheria.
template <class TR, int BM, int BN, int BK, int RM, int RN, bool DBUF, bool WHOLD, int MINB,
          bool RAW = false, bool STAGE_ONLY = false>
inline bool gemm_attrs_t(hipFuncAttributes &attr, int &lds_bytes) {
  constexpr int NKB = BK / 32;
  constexpr int WSB = BK + 16;
  constexpr int RSTR = NKB * ((TR::raw_words + 3) & ~3);
  if constexpr (RAW) {
    lds_bytes = (DBUF ? 2 : 1) * (BN * RSTR * (int)sizeof(int));
  } else if constexpr (TR::is_kq) {
    lds_bytes = (DBUF ? 2 : 1) * (BN * WSB + NKB * BN * (int)sizeof(KqScal));
  } else {
    lds_bytes = (DBUF ? 2 : 1) * (BN * WSB + NKB * BN * 8);
  }
  auto *fn = &gemm_i8_kernel<TR, BM, BN, BK, RM, RN, DBUF, WHOLD, MINB, RAW, STAGE_ONLY>;
  return hipFuncGetAttributes(&attr, (const void *)fn) == hipSuccess;
}

// Escolha de tile por M. MEDIDA nesta frente (`bench-gemm-engine-gpu --cfg`):
// em M=16 o GEMM perde para o GEMV de qualquer jeito (0,84-0,93x), e em M>=64 o
// tile BM=64 e' o melhor -- 15,47 T em M=128 e 17,04 em M=512 contra 11,20 e
// 13,80 do BM=128 na MESMA janela. A causa esta' na ISA: RM=RN=8 com BM=128 pede
// 256 VGPRs e transborda 56 B/lane para `scratch_*`, enquanto BM=64 (RM=4, RN=8)
// cabe em 166 VGPRs com ZERO spill. Nao e' o que o caminho f16 da frente G mediu
// (la' o BM=128 ganhava) -- e' o custo de registrador do dp4a com RM=RN=8.
inline int gemm_pick_bm(int n_tokens) { return n_tokens >= 64 ? 64 : 16; }

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
    RD_GEMM(SolverQ2K, 2, 256)
    RD_GEMM(SolverQ3K, 3, 256)
    RD_GEMM(SolverQ4K, 4, 256)
    RD_GEMM(SolverQ5K, 5, 256)
    RD_GEMM(SolverQ6K, 6, 256)
    default:
      return false;  // sem kernel: o chamador decide (SPEC 1.3)
  }
#undef RD_GEMM
}

inline bool gemm_attrs(int dt, int n_tokens, hipFuncAttributes &attr, int &lds_bytes) {
  const int bm = gemm_pick_bm(n_tokens);
#define RD_ATTR(Traits, Dt)                                                                  \
  case Dt:                                                                                   \
    if (bm == 64) return gemm_attrs_t<Traits, 64, 128, 64, 4, 8, true, true, 1>(attr, lds_bytes);   \
    return gemm_attrs_t<Traits, 16, 128, 64, 1, 8, true, true, 1>(attr, lds_bytes);
  switch (dt) {
    RD_ATTR(SolverIq3S, 12)
    RD_ATTR(SolverIq3XXS, 9)
    RD_ATTR(SolverIq4XS, 14)
    RD_ATTR(SolverQ2K, 2)
    RD_ATTR(SolverQ3K, 3)
    RD_ATTR(SolverQ4K, 4)
    RD_ATTR(SolverQ5K, 5)
    RD_ATTR(SolverQ6K, 6)
    default:
      return false;
  }
#undef RD_ATTR
}

// ---------------------------------------------------------------------------
// Bancada: variantes da MESMA geometria para A/B atribuivel (cfg 1-8 so' para
// iq3_s, para manter o numero de instanciacoes limitado; cfg 9-12 valem para os
// tres tipos cobertos, exceto 11-12 que so' existem para iq3_s). cfg:
//   0 = producao (a escolha de `gemm_launch` para este M)
//   1 = BM128 BN128 BK64 RM8 RN8, duplo buffer do W, W em registrador (WHOLD)
//   2 = idem 1 sem o duplo buffer (buffer unico, o caminho da frente F)
//   3 = idem 1 com __launch_bounds__(256, 2) (teto de 128 VGPR)
//   4 = idem 1 com o W lido da LDS a cada uso (WHOLD=false)
//   5 = BM64 BN128 BK64 RM4 RN8, duplo buffer
//   6 = BM128 BN64 BK64 RM8 RN4, duplo buffer
//   7 = BM64  BN64 BK64 RM4 RN4, duplo buffer (32 acumuladores/thread)
//   8 = BM128 BN64 BK64 RM4 RN4, duplo buffer (512 threads)
//   9 = STAGING CRU + decodificacao no laco de consumo, BM64 (geometria de
//       producao em M>=64): o staging grava 1 int4 por sub-bloco e o consumo
//       refaz LUT + sinal + d_w + (1+2*sc). MEDIDO: PERDE 1,7-2,2x (cabecalho)
//  10 = idem 9 com BM16 (geometria de producao em M<64). MEDIDO: PERDE 3,2-4,8x
//       -- com RM=1 cada palavra decodificada alimenta UM dp4a so'
//  11 = STAGE_ONLY (o "modo 1" da frente G) na geometria de producao BM64 e no
//       caminho de producao: mesma grade/threads/prologo/staging, conta trocada
//       por 4 leituras da LDS -- mede a fracao de staging
//  12 = idem 11 no caminho de staging cru (a fracao de staging DEPOIS)
// ---------------------------------------------------------------------------
inline bool gemm_launch_cfg(int dt, const void *d_w, const block_q8_1 *d_a, float *d_o,
                            std::int64_t nrows, std::int64_t ncols, std::int64_t act_stride,
                            int n_tokens, int cfg, hipStream_t stream) {
  if (cfg == 0) return gemm_launch(dt, d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
  if (cfg >= 9) {
    // Variantes do caminho CRU (e o modo de staging): valem para os tres tipos
    // cobertos, exceto o modo de staging puro, que existe so' para iq3_s.
    if (cfg >= 11 && dt != 12) return false;
    if (ncols % 256 != 0 || ncols % 64 != 0) return false;
#define RD_CFG_RAW(Traits, Dt)                                                                \
  case Dt: {                                                                                  \
    if (cfg == 9)                                                                             \
      return gemm_launch_t<Traits, 64, 128, 64, 4, 8, true, true, 1, true, false>(            \
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);                         \
    if (cfg == 10)                                                                            \
      return gemm_launch_t<Traits, 16, 128, 64, 1, 8, true, true, 1, true, false>(            \
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);                         \
    if (cfg == 11)                                                                            \
      return gemm_launch_t<Traits, 64, 128, 64, 4, 8, true, true, 1, false, true>(            \
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);                         \
    if (cfg == 12)                                                                            \
      return gemm_launch_t<Traits, 64, 128, 64, 4, 8, true, true, 1, true, true>(             \
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);                         \
    return false;                                                                             \
  }
    switch (dt) {
      RD_CFG_RAW(SolverIq3S, 12)
      RD_CFG_RAW(SolverIq3XXS, 9)
      RD_CFG_RAW(SolverIq4XS, 14)
      default:
        return false;
    }
#undef RD_CFG_RAW
  }
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
    case 7:  // 32 acumuladores por thread: cabe em 128 VGPR (2 CTAs/CU?)
      return gemm_launch_t<SolverIq3S, 64, 64, 64, 4, 4, true, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    case 8:  // 512 threads, 32 acumuladores por thread
      return gemm_launch_t<SolverIq3S, 128, 64, 64, 4, 4, true, true, 1>(
          d_w, d_a, d_o, nrows, ncols, act_stride, n_tokens, stream);
    default:
      return false;
  }
}

}  // namespace rdna4
