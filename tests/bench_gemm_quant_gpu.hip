// ===========================================================================
// Frente F do estudo de prefill (docs/plano-prefill.md, degrau D2): o GEMM
// tilejado com os BLOCOS REAIS do modelo (iq3_s) e o peso dequantizado para
// f16 na LDS.
//
// Por que esta bancada existe: a `bench-gemm-gpu` (frente P0) mediu 10,34 T
// MAC/s em M=128 e 12,74 T em M=512 com dado SINTETICO int8 -- ou seja, mediu
// a ESTRUTURA (tile + LDS + dp4a) e deixou de fora as duas coisas que decidem
// se o kernel existe de verdade:
//   1. o peso real (iq3_s, 110 B por 256 pesos: LUT + sinal + escala por 32)
//      tem de ser dequantizado NO STAGING, e esse custo nao estava medido;
//   2. a frente B §7.6 fixou f16 (nao int8) para os tipos IQ, com acumulador
//      f32 -- e o caminho f16 muda a instrucao de MAC: `v_dot2_f32_f16` faz
//      2 MAC/lane contra os 4 MAC/lane do `v_dot4_i32_iu8` (dp4a) que o
//      prototipo int8 usa, com o teto de issue correspondente (19,5 contra
//      39,0 T MAC/s com a mesma geometria de registrador).
//
// Esta bancada responde, com numero, as quatro perguntas do degrau D2:
//   1. STAGING: quanto custa ler iq3_s real e escrever f16 na LDS (G pesos/s e
//      GB/s de fonte), com o trafego de LDS contabilizado e por CTA;
//   2. GEMM f16 (dot2, acumulador f32) em M=16/64/128, N=17408, K=5120 (a
//      forma real do `ffn_gate`), em T MAC/s;
//   3. CORRECAO: contra o dequant do PROPRIO motor (`include/rdna4/dequant.cuh`,
//      `dequantize_iq3_s`) numa referencia de GPU -- warp por saida, em fp32 --
//      e contra um oraculo de CPU (M=4, N=64) que usa a LUT lida do device em
//      runtime (nada de tabela redigitada aqui);
//   4. ONDE VAI O TEMPO: staging, copia da ativacao e conta, medidos no MESMO
//      run (tres modos do MESMO kernel, mesma grade e mesmos threads) -- e o
//      veredito f16 x int8-dp4a.
//
// Uso:
//   bench-gemm-quant-gpu <model.gguf> [--m 16,64,128] [--bk 64] [--bm 0|16|32|64|128]
//                        [--reps 5] [--dbuf] [--tensor NOME] [--n N] [--k K] [--list]
//                        [--no-verify]
//     --bm    forca o tile de M (varre ocupacao de LDS x paralelismo)
//     --dbuf  duplo buffer no laco de K (off por padrao; medido: nao paga)
//
// Regra de ouro desta casa (docs/estudo-prefill.md §3b): a primeira versao da
// bancada anterior reportou 1198 T MACs/s porque o lancamento tinha o numero de
// threads errado e o kernel NAO RODAVA. Aqui: `hipGetLastError()` depois de
// todo lancamento, verificacao de cada configuracao contra oraculo ANTES de
// cronometrar, checagem de que a saida nao e' toda zero, e warm-up de ~150 ms
// de GPU por modo (a placa cai para ~10 MHz ociosa e o primeiro kernel medido
// sairia com o clock errado).
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

#include "rdna4/dequant.cuh"  // dequantize_iq3_s -- o dequant do motor (oraculo de GPU)
#include "rdna4/fp16.h"
#include "rdna4/loader.h"
#include "rdna4/quants.h"
#include "rdna4/quant_tables.h"
#include "rdna4/vecdotq.cuh"  // __vcmpne4/__vsub4/iq3s_grid -- o mesmo codigo do vec_dot

// O alvo `bench-gemm-quant-gpu` linka so' a libamdhip64: as duas fontes do
// loader do motor nao estao na lista dele e o CMakeLists nao esta' no escopo
// desta frente (regra explicita do pedido). Entram por include direto, entao a
// bancada usa o GgufLoader DE VERDADE -- com a validacao de dtype e de
// geometria do M1 -- sem duplicar um parser de GGUF dentro de um arquivo de
// teste (que seria a alternativa, e a pior delas).
#include "../src/backend/gguf.cpp"
#include "../src/backend/loader.cpp"

namespace {

using rdna4::block_iq3_s;

constexpr int kQK = 256;      // pesos por super-bloco iq3_s
constexpr int kBlockBytes = 110;
constexpr int kIq3S = 12;     // ggml_type id (ver matvec.cuh dt_name)

// ---------------------------------------------------------------------------
// 1. dot2 f16 com acumulador f32
// ---------------------------------------------------------------------------
// `__builtin_amdgcn_fdot2` compila para UMA instrucao `v_dot2_f32_f16` no
// gfx1201 (conferido no .s: "v_dot2_f32_f16 v1, v1, v2, 0"), com o acumulador
// f32 no terceiro operando -- que e' a ressalva obrigatoria da frente B §7.4
// ("nunca v_pk_fma_f16, que acumula em f16": com K=5120 o acumulador f16
// estoura). 2 MAC por lane por instrucao, contra 4 do dp4a.
__device__ __forceinline__ float fdot2(const __half2 a, const __half2 b, const float c) {
  return __builtin_amdgcn_fdot2(a, b, c, false);
}

// ---------------------------------------------------------------------------
// 2. staging: bloco iq3_s real -> f16 na LDS
// ---------------------------------------------------------------------------
// 4 bytes assinados (de __vsub4) -> 4 f16, escala `d` aplicada em fp32 e
// arredondamento feito UMA vez, na conversao (frente B §7.1: "LUT + escala em
// fp32 + arredondamento para f16 na escrita").
__device__ __forceinline__ void iq3s_store4(__half2 *dst, const int g, const float d) {
  dst[0] = __halves2half2(__float2half_rn((float)(signed char)(g & 0xff) * d),
                          __float2half_rn((float)(signed char)((g >> 8) & 0xff) * d));
  dst[1] = __halves2half2(__float2half_rn((float)(signed char)((g >> 16) & 0xff) * d),
                          __float2half_rn((float)(signed char)((g >> 24) & 0xff) * d));
}

// Um sub-bloco de 32 pesos (s = 0..7 dentro do super-bloco) -> 32 f16 na LDS,
// em ordem de k. Layout do bloco (ggml-common.h / include/rdna4/quants.h):
//   qs[8s .. 8s+8)      8 palavras de grade (4 magnitudes cada)
//   qh[s]               1 bit alto por palavra de grade
//   signs[4s .. 4s+4)   8 bits de sinal por par de palavras
//   scales[s/2]         nibble 4*(s%2)
// E' exatamente a convencao do `dequantize_row_iq3_s` do ggml e do
// `dequantize_iq3_s` do motor (include/rdna4/dequant.cuh). O sinal usa o mesmo
// truque do `vec_dot_iq3_s_q8_1` (vecdotq.cuh): __vcmpne4 monta a mascara
// 0x00/0xFF por byte e `(g^m)-m` nega em complemento de dois.
__device__ __forceinline__ void stage_iq3s_32(const block_iq3_s *__restrict__ blk, const int s,
                                              __half2 *__restrict__ dst) {
  const unsigned char *qs = blk->qs + 8 * s;
  const unsigned char *sg = blk->signs + 4 * s;
  const int qh = blk->qh[s];
  const float d = rdna4::fp16_to_float(blk->d) *
                  (float)(1 + 2 * ((blk->scales[s / 2] >> (4 * (s % 2))) & 0x0F));
#pragma unroll
  for (int il = 0; il < 4; ++il) {
    const int g1 = rdna4::iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)];
    const int g2 = rdna4::iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)];
    const int m0 = rdna4::__vcmpne4(((sg[il] & 0x03) << 7) | ((sg[il] & 0x0C) << 21), 0);
    const int m1 = rdna4::__vcmpne4(((sg[il] & 0x30) << 3) | ((sg[il] & 0xC0) << 17), 0);
    iq3s_store4(dst + 4 * il + 0, rdna4::__vsub4(g1 ^ m0, m0), d);
    iq3s_store4(dst + 4 * il + 2, rdna4::__vsub4(g2 ^ m1, m1), d);
  }
}

// ---------------------------------------------------------------------------
// 3. O kernel: GEMM tilejado BM x BN, BK de K por iteracao, RM x RN por thread
// ---------------------------------------------------------------------------
// MODE 0 = GEMM completo (staging + conta)
// MODE 1 = so' staging (ativacao f16 + peso real -> LDS): e' o item 1
// MODE 2 = so' a copia da ativacao f16 (isola a outra metade do staging)
// DBUF   = duplo buffer no laco de K (software pipelining: o staging da
//          iteracao k+1 e' emitido ANTES da conta da iteracao k, entao as
//          latencias de L1/global do staging ficam em voo durante a conta)
//
// Mapeamento de threads: tr = tid / TN (linhas de A) e tc = tid % TN (linhas de
// W). Dentro de uma wave32 os 16 valores de `tc` sao distintos, entao as 16
// linhas do tile ficam a `stride` palavras de distancia; com o pad de 1 half2
// por linha (stride = BK/2+1 half2 = impar) os 16 acessos caem em 16 bancos
// distintos -- sem conflito de LDS nem no A nem no W.
template <int BM, int BN, int BK, int RM, int RN, int MODE, bool DBUF>
__global__ void __launch_bounds__((BM / RM) * (BN / RN)) gemm_iq3s_f16_kernel(
    const __half *__restrict__ A, const block_iq3_s *__restrict__ W, float *__restrict__ C,
    const int N, const int K) {
  constexpr int TM = BM / RM;              // threads na dimensao M
  constexpr int TN = BN / RN;              // threads na dimensao N
  constexpr int NT = TM * TN;
  constexpr int NIQS = BK / 32;            // sub-blocos de 32 pesos por linha por iteracao
  constexpr int AS = BK + 2;               // halves por linha do tile A (+1 half2 de pad)
  constexpr int WS = BK + 2;
  constexpr int A2 = AS / 2, W2 = WS / 2;  // os mesmos strides em half2
  constexpr int NBA = BM * AS, NBW = BN * WS;
  constexpr bool DO_W = (MODE != 2);
  constexpr bool DO_GEMM = (MODE == 0);
  constexpr bool PIPE = (DBUF && DO_GEMM);  // duplo buffer so' faz sentido no GEMM
  static_assert(BK % 32 == 0, "BK tem de ser multiplo de 32 (o sub-bloco da escala)");
  static_assert(BN % RN == 0 && BM % RM == 0, "");

  __shared__ __align__(16) __half s_a[PIPE ? 2 * NBA : NBA];
  __shared__ __align__(16) __half s_w[PIPE ? 2 * NBW : NBW];

  const int tid = threadIdx.x;
  const int tr = tid / TN;
  const int tc = tid % TN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int bpr = K / kQK;  // blocos iq3_s por linha

  float acc[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) acc[r][c] = 0.f;

  float sink = 0.f;  // modos de staging: impede o compilador de apagar a LDS

  // ---- staging de uma iteracao de K ---------------------------------------
  auto stage_tile = [&](__half *ta, __half *tw, const int k0) {
    // (a) ativacao: no caminho novo ela JA' e' f16 (o `act_quant` de 0,227
    // ms/token deixa de existir), entao aqui e' copia direta, sem dequant.
    for (int i = tid; i < BM * (BK / 2); i += NT) {
      const int row = i / (BK / 2);
      const int c2 = i % (BK / 2);
      *(__half2 *)&ta[row * AS + 2 * c2] =
          *(const __half2 *)&A[(std::int64_t)(m0 + row) * K + k0 + 2 * c2];
    }
    // (b) peso real: iq3_s da global -> f16 na LDS (o staging que decide)
    if constexpr (DO_W) {
      const int ib = k0 >> 8;          // super-bloco de 256
      const int s0 = (k0 & 255) >> 5;  // primeiro sub-bloco dentro dele
      for (int task = tid; task < BN * NIQS; task += NT) {
        const int row = task / NIQS;
        const int ti = task % NIQS;
        stage_iq3s_32(W + (std::int64_t)(n0 + row) * bpr + ib, s0 + ti,
                      (__half2 *)&tw[row * WS + ti * 32]);
      }
    }
  };
  // ---- conta de uma iteracao de K -----------------------------------------
  auto compute_tile = [&](const __half *ta, const __half *tw) {
    const __half2 *a2 = (const __half2 *)ta;
    const __half2 *w2 = (const __half2 *)tw;
#pragma unroll
    for (int k2 = 0; k2 < BK / 2; ++k2) {
      __half2 a[RM], w[RN];
#pragma unroll
      for (int r = 0; r < RM; ++r) a[r] = a2[(tr + r * TM) * A2 + k2];
#pragma unroll
      for (int c = 0; c < RN; ++c) w[c] = w2[(tc + c * TN) * W2 + k2];
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) acc[r][c] = fdot2(a[r], w[c], acc[r][c]);
    }
  };

  if constexpr (PIPE) {
    // Software pipelining: o staging de k+BK e' emitido ANTES da conta de k e
    // as duas barreiras por iteracao viram uma. E' a resposta medida ao achado
    // de que, com buffer unico, staging e conta se SERIALIZAM (a soma dos dois
    // tempos medidos e' o tempo total).
    stage_tile(s_a, s_w, 0);
    __syncthreads();
    for (int k0 = 0, buf = 0; k0 < K; k0 += BK, buf ^= 1) {
      __half *na = s_a + (buf ^ 1) * NBA;
      __half *nw = s_w + (buf ^ 1) * NBW;
      if (k0 + BK < K) stage_tile(na, nw, k0 + BK);
      compute_tile(s_a + buf * NBA, s_w + buf * NBW);
      __syncthreads();
    }
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      stage_tile(s_a, s_w, k0);
      __syncthreads();
      if constexpr (DO_GEMM) {
        compute_tile(s_a, s_w);
      } else {
        // Consumo minimo. Os indices dependem de `tid`, entao o compilador nao
        // prova que a LDS inteira nao e' necessaria e o staging tem de ser
        // materializado de verdade (o risco desta medicao virar ficcao).
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const int rw = (tid * 7 + i * 29) % BN;
          const int kw = (tid * 11 + i * 5) % BK;
          sink += __half2float(s_w[rw * WS + kw]);
          const int ra = (tid * 5 + i * 17) % BM;
          const int ka = (tid * 13 + i * 3) % BK;
          sink += __half2float(s_a[ra * AS + ka]);
        }
      }
      __syncthreads();
    }
  }

  if constexpr (DO_GEMM) {
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
      for (int c = 0; c < RN; ++c)
        C[(std::int64_t)(m0 + tr + r * TM) * N + n0 + tc + c * TN] = acc[r][c];
  } else {
    // Guarda impossivel: mantem a soma viva sem escrever C (uma passada de
    // escrita de C seria 8,9 MB em M=128 -- isso NAO e' staging).
    if (__float_as_uint(sink) == 0x7F800001u) C[0] = sink;
  }
}

// ---------------------------------------------------------------------------
// 3b. FRENTE G -- variantes de staging, medidas na MESMA janela
// ---------------------------------------------------------------------------
// O diagnostico da frente F (docs/estudo-prefill-f-staging.md §1): o staging
// custa 42 % do kernel, ocupa ~2 % dos slots de issue e roda a 12-28 % da banda
// -- e' LATENCIA. Por sub-bloco de 32 pesos e por thread a cadeia e'
//
//   LDG qs/qh/signs -> indice -> LDG da LUT (L1) -> cvt/mul -> pack -> STS
//
// com UMA cadeia por thread (BM=128/BK=64: 256 threads para 256 sub-blocos por
// iteracao de K) e um `__syncthreads` entre staging e conta -- o mesmo warp faz
// as duas coisas e elas se serializam (medido: t_stage + t_conta = t_GEMM). A
// ISA confirma: 5 cargas de campo por thread (`global_load_u16` do `d`,
// `global_load_b64` do `qs`, `global_load_u8` do `qh`, `global_load_b32` dos
// `signs`, `global_load_u8` da escala), depois 8 `global_load_b32` da LUT num
// `s_clause 0x7` + `s_wait_loadcnt 0x8` (espera pelas OITO), e so' entao
// 16 `v_fma_mixlo_f16` + 8 `ds_store_2addr_b32`.
//
// As variantes atacam a cadeia uma de cada vez, como BITS do template (STG),
// para que o A/B seja entre kernels que so' diferem no que o bit muda:
//
//   bit 0 (1)  AVEC : a copia da ativacao usa `int4` (16 B) em vez de `half2`
//                     (4 B): 4x menos instrucoes e 4x menos pedidos (o loop cai
//                     de 16 para 4 iteracoes). Exige stride de linha 16-B
//                     alinhado, entao AS vai de BK+2 para BK+8 halves. O pad par
//                     NAO cria conflito no A: dentro de uma warp so' existem 2
//                     enderecos de A (tr e tr+1), cada um broadcast para 16
//                     lanes; quem precisa de stride impar e' o W (W2 = BK/2+1),
//                     e esse nao mudou.
//   bit 1 (2)  WPF  : prefetch dos CAMPOS do peso de k+BK para REGISTRADORES,
//                     emitido depois do `__syncthreads` e ANTES da conta de k. O
//                     compilador nao pode afundar a carga atraves da barreira,
//                     entao a janela de voo e' a conta inteira; a latencia do
//                     LDG sai da cadeia do staging. Nao custa LDS.
//   bit 2 (4)  WDB  : duplo buffer SO' do W (o A fica com um buffer), o que
//                     deixa a dequantizacao+STS de k+BK rodar DURANTE a conta de
//                     k. O duplo buffer completo da frente F nao caberia em
//                     BM=128 (67,6 KB); este cabe porque o A nao e' duplicado.
//   bit 3 (8)  AG   : a ativacao NAO vai para a LDS -- o laco interno le direto
//                     da global (broadcast de 2 enderecos por warp). Mata a
//                     copia da ativacao (frente F: 0,178-0,190 ms = 35 % do
//                     staging, 178 MB por passada em BM=128) e libera 18 KB de
//                     LDS por CTA.
//
// MODE: 0 = GEMM, 1 = staging (A+W) com consumo minimo, 3 = staging do W para
// int8 na LDS -- a alternativa da regra 5 do pedido, na convencao do
// `vec_dot_iq3_s_q8_1`: magnitude 1..15 com o sinal aplicado por
// `__vcmpne4`/`__vsub4` e o `1+2*sc` FORA do byte (ele e' multiplicacao inteira
// no int32 do bloco, `vecdotq.cuh:725`, e trocar a ordem muda o arredondamento).
//
// Toda variante tem de dar C BIT A BIT igual ao kernel da frente F: elas so'
// mudam POR ONDE o operando anda, nunca a algebra. A bancada verifica isso e
// imprime `bitid`.
struct WFields {
  unsigned long long qs;  // 8 B de blk->qs + 8s
  unsigned int signs;     // 4 B de blk->signs + 4s
  unsigned short d;       // blk->d (fp16)
  unsigned char qh;       // blk->qh[s]
  unsigned char sc;       // blk->scales[s/2]
};

// `__builtin_memcpy` em vez de cast com alinhamento assumido: o `qs` mora no
// offset 2 de um bloco de 110 B, ou seja 2 mod 8 -- a carga de 8 B deste campo e'
// DESALINHADA por construcao. O kernel da frente F ja' faz isso (a ISA dele
// mostra `global_load_b64 ..., off offset:2`, vindo do `get_int_b2` de dois u16)
// e passa bit a bit contra o oraculo: o gfx1201 aceita a carga desalinhada.
__device__ __forceinline__ WFields wf_load(const block_iq3_s *__restrict__ blk, const int s) {
  WFields f;
  __builtin_memcpy(&f.qs, blk->qs + 8 * s, 8);
  __builtin_memcpy(&f.signs, blk->signs + 4 * s, 4);
  f.d = blk->d;
  f.qh = blk->qh[s];
  f.sc = blk->scales[s / 2];
  return f;
}

// Mesma algebra de `stage_iq3s_32`, em cima dos campos ja' em registrador.
__device__ __forceinline__ void wf_dequant(const WFields &f, const int s, __half2 *__restrict__ dst) {
  const unsigned char *qs = (const unsigned char *)&f.qs;
  const unsigned char *sg = (const unsigned char *)&f.signs;
  const int qh = f.qh;
  const float d = rdna4::fp16_to_float(f.d) * (float)(1 + 2 * ((f.sc >> (4 * (s % 2))) & 0x0F));
#pragma unroll
  for (int il = 0; il < 4; ++il) {
    const int g1 = rdna4::iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)];
    const int g2 = rdna4::iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)];
    const int m0 = rdna4::__vcmpne4(((sg[il] & 0x03) << 7) | ((sg[il] & 0x0C) << 21), 0);
    const int m1 = rdna4::__vcmpne4(((sg[il] & 0x30) << 3) | ((sg[il] & 0xC0) << 17), 0);
    iq3s_store4(dst + 4 * il + 0, rdna4::__vsub4(g1 ^ m0, m0), d);
    iq3s_store4(dst + 4 * il + 2, rdna4::__vsub4(g2 ^ m1, m1), d);
  }
}

// O MESMO sub-bloco de 32 pesos para int8 na LDS: 32 B por sub-bloco em vez de
// 64 B de f16, sem `cvt`/`mul`/`pack` (a magnitude 1..15 ja' sai do grid com o
// sinal por `__vsub4`) e com dois `ds_store_b128` no lugar de oito stores.
__device__ __forceinline__ void stage_iq3s_i8_32(const block_iq3_s *__restrict__ blk, const int s,
                                                 signed char *__restrict__ dst) {
  const unsigned char *qs = blk->qs + 8 * s;
  const unsigned char *sg = blk->signs + 4 * s;
  const int qh = blk->qh[s];
  int out[8];
#pragma unroll
  for (int il = 0; il < 4; ++il) {
    const int g1 = rdna4::iq3s_grid[qs[2 * il + 0] | ((qh << (8 - 2 * il)) & 0x100)];
    const int g2 = rdna4::iq3s_grid[qs[2 * il + 1] | ((qh << (7 - 2 * il)) & 0x100)];
    const int m0 = rdna4::__vcmpne4(((sg[il] & 0x03) << 7) | ((sg[il] & 0x0C) << 21), 0);
    const int m1 = rdna4::__vcmpne4(((sg[il] & 0x30) << 3) | ((sg[il] & 0xC0) << 17), 0);
    out[2 * il + 0] = rdna4::__vsub4(g1 ^ m0, m0);
    out[2 * il + 1] = rdna4::__vsub4(g2 ^ m1, m1);
  }
  *(int4 *)&dst[0] = *(const int4 *)&out[0];
  *(int4 *)&dst[16] = *(const int4 *)&out[4];
}

// O kernel das variantes: mesma grade, mesmos threads, mesmo tile e mesmo laco
// interno do kernel da frente F -- so' o caminho do dado muda.
template <int BM, int BN, int BK, int RM, int RN, int STG, int MODE>
__global__ void __launch_bounds__((BM / RM) * (BN / RN)) gemm_v2_kernel(
    const __half *__restrict__ A, const block_iq3_s *__restrict__ W, float *__restrict__ C,
    const int N, const int K) {
  constexpr int TM = BM / RM;
  constexpr int TN = BN / RN;
  constexpr int NT = TM * TN;
  constexpr int NIQS = BK / 32;
  constexpr bool AVEC = (STG & 1) != 0;
  constexpr bool WPF = (STG & 2) != 0;
  constexpr bool WDB = (STG & 4) != 0;
  constexpr bool AG = (STG & 8) != 0;
  constexpr bool I8 = (MODE == 3);
  constexpr bool DO_GEMM = (MODE == 0);
  constexpr int AS = AVEC ? (BK + 8) : (BK + 2);  // halves por linha do tile A
  constexpr int WS = BK + 2;
  constexpr int A2 = AS / 2, W2 = WS / 2;
  constexpr int NBA = BM * AS, NBW = BN * WS;
  constexpr int WI8S = BK + 16;  // stride em BYTES do tile W em int8 (16-B alinhado)
  static_assert(BK % 32 == 0 && BM % RM == 0 && BN % RN == 0, "");
  // O prefetch exige que cada thread tenha EXATAMENTE uma task de W, fixa em
  // todas as iteracoes de K: e' ela que define o que vai para o registrador.
  static_assert(!(WPF || WDB || I8) || BN * NIQS == NT, "prefetch exige 1 task de W por thread");

  __shared__ __align__(16) __half s_a[(AG || I8) ? 1 : NBA];
  __shared__ __align__(16) __half s_w[WDB ? 2 * NBW : NBW];
  __shared__ __align__(16) signed char s_wi[I8 ? BN * WI8S : 1];

  const int tid = threadIdx.x;
  const int tr = tid / TN;
  const int tc = tid % TN;
  const int m0 = blockIdx.y * BM;
  const int n0 = blockIdx.x * BN;
  const int bpr = K / kQK;

  float acc[RM][RN];
#pragma unroll
  for (int r = 0; r < RM; ++r)
#pragma unroll
    for (int c = 0; c < RN; ++c) acc[r][c] = 0.f;

  float sink = 0.f;

  // task de W fixa (1 por thread) + endereco da linha calculado UMA vez
  const int wrow = tid / NIQS;
  const int wti = tid % NIQS;
  const block_iq3_s *const wrowp = W + (std::int64_t)(n0 + wrow) * bpr;
  const int wdst = wrow * WS + wti * 32;  // em halves
  const int i8dst = wrow * WI8S + wti * 32;

  // ---- (a) ativacao -> LDS ------------------------------------------------
  auto a_copy = [&](__half *ta, const int k0) {
    if constexpr (AVEC) {
      constexpr int NI4 = BK / 8;  // `int4` = 8 halves
      for (int i = tid; i < BM * NI4; i += NT) {
        const int row = i / NI4;
        const int c8 = i % NI4;
        *(int4 *)&ta[row * AS + 8 * c8] =
            *(const int4 *)&A[(std::int64_t)(m0 + row) * K + k0 + 8 * c8];
      }
    } else {
      for (int i = tid; i < BM * (BK / 2); i += NT) {
        const int row = i / (BK / 2);
        const int c2 = i % (BK / 2);
        *(__half2 *)&ta[row * AS + 2 * c2] =
            *(const __half2 *)&A[(std::int64_t)(m0 + row) * K + k0 + 2 * c2];
      }
    }
  };
  // ---- (b) peso iq3_s -> f16 na LDS ---------------------------------------
  auto w_stage = [&](__half *tw, const int k0) {
    stage_iq3s_32(wrowp + (k0 >> 8), ((k0 & 255) >> 5) + wti, (__half2 *)&tw[wdst]);
  };
  auto wf_at = [&](const int k0) { return wf_load(wrowp + (k0 >> 8), ((k0 & 255) >> 5) + wti); };
  auto wf_to = [&](__half *tw, const WFields &f, const int k0) {
    wf_dequant(f, ((k0 & 255) >> 5) + wti, (__half2 *)&tw[wdst]);
  };
  auto w_stage_i8 = [&](const int k0) {
    stage_iq3s_i8_32(wrowp + (k0 >> 8), ((k0 & 255) >> 5) + wti, &s_wi[i8dst]);
  };
  // ---- (c) o laco interno -------------------------------------------------
  auto compute = [&](const __half *ta, const __half *tw, const int k0) {
    const __half2 *a2 = (const __half2 *)ta;
    const __half2 *w2 = (const __half2 *)tw;
    const __half2 *ag = (const __half2 *)(A + (std::int64_t)(m0 + tr) * K + k0);
#pragma unroll
    for (int k2 = 0; k2 < BK / 2; ++k2) {
      __half2 a[RM], w[RN];
#pragma unroll
      for (int r = 0; r < RM; ++r) {
        if constexpr (AG) {
          a[r] = ag[(r * TM) * (K / 2) + k2];
        } else {
          a[r] = a2[(tr + r * TM) * A2 + k2];
        }
      }
#pragma unroll
      for (int c = 0; c < RN; ++c) w[c] = w2[(tc + c * TN) * W2 + k2];
#pragma unroll
      for (int r = 0; r < RM; ++r)
#pragma unroll
        for (int c = 0; c < RN; ++c) acc[r][c] = fdot2(a[r], w[c], acc[r][c]);
    }
  };
  // ---- consumo minimo dos modos de staging --------------------------------
  auto sink_read = [&]() {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int rw = (tid * 7 + i * 29) % BN;
      const int kw = (tid * 11 + i * 5) % BK;
      sink += __half2float(s_w[rw * WS + kw]);
      if constexpr (!AG) {
        const int ra = (tid * 5 + i * 17) % BM;
        const int ka = (tid * 13 + i * 3) % BK;
        sink += __half2float(s_a[ra * AS + ka]);
      }
    }
  };

  if constexpr (MODE == 3) {
    for (int k0 = 0; k0 < K; k0 += BK) {
      w_stage_i8(k0);
      __syncthreads();
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int rw = (tid * 7 + i * 29) % BN;
        const int kw = (tid * 11 + i * 5) % BK;
        sink += (float)s_wi[rw * WI8S + kw];
      }
      __syncthreads();
    }
  } else if constexpr (WPF && DO_GEMM) {
    if constexpr (!AG) a_copy(s_a, 0);
    WFields wf = wf_at(0);
    if constexpr (WDB) {
      wf_to(s_w, wf, 0);
      if (K > BK) wf = wf_at(BK);
      __syncthreads();
      for (int k0 = 0, buf = 0; k0 < K; k0 += BK, buf ^= 1) {
        if (k0 + BK < K) {
          wf_to(s_w + (buf ^ 1) * NBW, wf, k0 + BK);
          if (k0 + 2 * BK < K) wf = wf_at(k0 + 2 * BK);
        }
        compute(s_a, s_w + buf * NBW, k0);
        __syncthreads();
        if constexpr (!AG) {
          if (k0 + BK < K) a_copy(s_a, k0 + BK);
        }
        __syncthreads();
      }
    } else {
      wf_to(s_w, wf, 0);
      __syncthreads();
      for (int k0 = 0; k0 < K; k0 += BK) {
        if (k0 + BK < K) wf = wf_at(k0 + BK);  // em voo durante a conta
        compute(s_a, s_w, k0);
        __syncthreads();
        if (k0 + BK < K) {
          wf_to(s_w, wf, k0 + BK);
          if constexpr (!AG) a_copy(s_a, k0 + BK);
        }
        __syncthreads();
      }
    }
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      if constexpr (!AG) a_copy(s_a, k0);
      w_stage(s_w, k0);
      __syncthreads();
      if constexpr (DO_GEMM) {
        compute(s_a, s_w, k0);
      } else {
        sink_read();
      }
      __syncthreads();
    }
  }

  if constexpr (DO_GEMM) {
#pragma unroll
    for (int r = 0; r < RM; ++r)
#pragma unroll
      for (int c = 0; c < RN; ++c)
        C[(std::int64_t)(m0 + tr + r * TM) * N + n0 + tc + c * TN] = acc[r][c];
  } else {
    if (__float_as_uint(sink) == 0x7F800001u) C[0] = sink;
  }
}

// ---------------------------------------------------------------------------
// 4. O oraculo de GPU: um warp por elemento de saida
// ---------------------------------------------------------------------------
// Reusa `dequantize_iq3_s` do MOTOR, sem reescrita: as 32 lanes cobrem os 256
// pesos do super-bloco (8 valores cada). O produto com a ativacao f16 e' em
// fp32, entao a diferenca contra o kernel tilejado e' (i) o arredondamento do
// peso para f16 e (ii) a ordem das somas -- exatamente as duas coisas que o
// degrau D2 introduz.
//
// ATENCAO (bug real que esta bancada pegou): `dequantize_iq3_s` escreve nas
// posicoes 32*(lane%8) + 8*(lane/8) + [0,8) do SUPER-BLOCO inteiro, e nao num
// buffer de 8 floats por lane -- com um `float w[8]` local isso estoura a pilha
// e o oraculo devolve NaN (medido: 59961 de 69632 saidas nao-finitas). Por isso
// o destino e' um buffer de 256 floats por warp, na LDS.
__global__ void ref_gemm_kernel(const __half *__restrict__ A, const block_iq3_s *__restrict__ W,
                                float *__restrict__ C, const int M, const int N, const int K) {
  __shared__ float s_w[8][kQK];
  const int warp = (int)(threadIdx.x >> 5);
  const int lane = (int)(threadIdx.x & 31u);
  const std::int64_t out = (std::int64_t)blockIdx.x * (blockDim.x >> 5) + warp;
  const bool active = out < (std::int64_t)M * N;
  // Linha valida mesmo quando inativo: os __syncthreads() tem de ser uniformes.
  const int m = active ? (int)(out / N) : 0;
  const int n = active ? (int)(out % N) : 0;
  const int bpr = K / kQK;
  const block_iq3_s *row = W + (std::int64_t)n * bpr;
  const __half *a = A + (std::int64_t)m * K;
  const int kk = 32 * (lane % 8) + 8 * (lane / 8);  // offset que o dequant cobre
  float acc = 0.f;
  for (int ib = 0; ib < bpr; ++ib) {
    rdna4::dequantize_iq3_s(row, ib, s_w[warp], lane);
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 8; ++j) acc += s_w[warp][kk + j] * __half2float(a[ib * kQK + kk + j]);
    __syncthreads();
  }
#pragma unroll
  for (int off = 16; off; off >>= 1) acc += __shfl_xor_sync(0xffffffffull, acc, off);
  if (active && lane == 0) C[out] = acc;
}

// Deposito do staging ISOLADO: um thread por sub-bloco de 32 pesos, do mesmo
// jeito que o kernel faz (mesma funcao `stage_iq3s_32`), mas escrevendo o
// resultado na global em vez da LDS. E' o que separa "o staging esta' certo" de
// "o GEMM esta' certo" -- sem isto, um erro de staging apareceria disfarcado de
// erro de f16 na comparacao contra o oraculo.
__global__ void dump_stage_kernel(const block_iq3_s *__restrict__ W, __half *__restrict__ out,
                                  const int nrow, const int bpr, const int nsub) {
  const int t = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (t >= nsub) return;
  const int ib = t / 8;
  const int s = t % 8;
  __half2 tmp[16];
  stage_iq3s_32(W + (std::int64_t)nrow * bpr + ib, s, tmp);
  __half2 *o = (__half2 *)(out + (std::size_t)t * 32);
#pragma unroll
  for (int i = 0; i < 16; ++i) o[i] = tmp[i];
}

// Copia a LUT do DEVICE para a memoria global: o oraculo de CPU usa a MESMA
// tabela que o kernel, entao ela nao pode divergir por redigitacao.
__global__ void copy_lut_kernel(uint32_t *__restrict__ out, const int n) {
  for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) out[i] = rdna4::iq3s_grid[i];
}

// ---------------------------------------------------------------------------
// 5. Utilitarios de host
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

void fill_a(std::vector<uint16_t> &h, const std::size_t n, const unsigned seed) {
  // Ativacao f16 (o formato do caminho novo), rms ~1,9 -- ordem de grandeza das
  // ativacoes reais (o valor exato nao muda nenhuma conclusao estrutural).
  h.resize(n);
  unsigned s = seed;
  for (std::size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const float u = (float)(s >> 8) / (float)(1u << 24);  // [0,1)
    h[i] = rdna4::float_to_fp16((u - 0.5f) * 6.0f);
  }
}

// Oracolo de CPU: a MESMA algebra do `dequantize_iq3_s` do motor (dequant.cuh),
// escrita a mao em cima dos bytes crus do GGUF e da LUT lida do device.
float cpu_iq3s_weight(const unsigned char *b, const int j, const uint32_t *lut) {
  const uint16_t d_h = (uint16_t)(b[0] | (b[1] << 8));
  const unsigned char *qs = b + 2;         // 64 B
  const unsigned char *qh = b + 2 + 64;    // 8 B
  const unsigned char *sg = b + 2 + 72;    // 32 B
  const unsigned char *sc = b + 2 + 104;   // 4 B
  const int s = j / 32;                    // sub-bloco
  const int p = j % 32;                    // posicao dentro dele
  const int il = p / 8;                    // par de palavras de grade (0..3)
  const int e = p % 8;                     // posicao no par
  const int widx = 2 * il + (e < 4 ? 0 : 1);
  const int sh = (e < 4 ? (8 - 2 * il) : (7 - 2 * il));
  const uint32_t g = lut[(int)qs[8 * s + widx] | ((qh[s] << sh) & 0x100)];
  const int mag = (int)((g >> (8 * (e % 4))) & 0xFFu);
  const int sbit = (sg[4 * s + il] >> e) & 1;
  const float d = rdna4::fp16_to_float(d_h) *
                  (float)(1 + 2 * ((sc[s / 2] >> (4 * (s % 2))) & 0x0F));
  return d * (float)(sbit ? -mag : mag);
}

struct Cfg {
  int BM, BN, BK, RM, RN;
  const char *tag;
  int threads() const { return (BM / RM) * (BN / RN); }
};

// Lancamento do oraculo: 1 WARP por elemento de saida, entao a grade e'
// ceil(saidas / warps_por_bloco) -- contar threads aqui (e nao warps) deixa a
// maior parte de dRef sem escrever, e o "nao bate com o oraculo" vira ficcao.
void launch_ref(const __half *dA, const block_iq3_s *dW, float *dC, const int M, const std::int64_t N,
                const int K) {
  constexpr int kThreads = 256;  // 8 warps por bloco
  const std::int64_t n_out = (std::int64_t)M * N;
  const unsigned blocks = (unsigned)((n_out + (kThreads / 32) - 1) / (kThreads / 32));
  ref_gemm_kernel<<<blocks, kThreads>>>(dA, dW, dC, M, (int)N, K);
}

}  // namespace

namespace {

// Despacho em tempo de compilacao (o Cfg e' runtime, o template nao).
#define RD_MODE_CASE(MODE)                                                                   \
  case MODE:                                                                                 \
    gemm_iq3s_f16_kernel<BM, BN, BK, RM, RN, MODE, false><<<grid, th>>>(dA, dW, dC, N, K);   \
    break;

template <int BM, int BN, int BK, int RM, int RN>
void launch_cfg(const int mode, const dim3 grid, const int th, const __half *dA,
                const block_iq3_s *dW, float *dC, const int N, const int K) {
  switch (mode) {
    RD_MODE_CASE(0)
    RD_MODE_CASE(1)
    RD_MODE_CASE(2)
  }
}
#undef RD_MODE_CASE

// Variante com duplo buffer (so' MODE 0). Nao existe para (BM=128, BK=64):
// 2*(128*66 + 128*66)*2 B = 67 584 B passa dos 65 536 B de LDS do cartao.
template <int BM, int BN, int BK, int RM, int RN>
void launch_cfg_dbuf(const dim3 grid, const int th, const __half *dA, const block_iq3_s *dW,
                     float *dC, const int N, const int K) {
  gemm_iq3s_f16_kernel<BM, BN, BK, RM, RN, 0, true><<<grid, th>>>(dA, dW, dC, N, K);
}

// ---------------------------------------------------------------------------
// Despacho das variantes da frente G (template <..., STG, MODE>)
// ---------------------------------------------------------------------------
// Devolve os atributos da funcao (regs/LDS) do ultimo lancamento, que e' o que
// diz se a ocupacao estimada bate com a medida: `ctas/CU` sai de
// min(LDS do cartao / LDS por CTA, 16 - regs*threads/(256*4)) -- a segunda
// parcela e' o limite de VGPRs (256 KB por SIMD, 4 SIMD por CU), que e' o que a
// frente D viu limitar o WMMA a 2 warps por SIMD.
template <int BM, int BN, int BK, int RM, int RN, int STG, int MODE>
void launch_v2_one(const dim3 grid, const int th, const __half *dA, const block_iq3_s *dW, float *dC,
                   const int N, const int K, int *regs_out, std::size_t *lds_out, int *occ_out) {
  gemm_v2_kernel<BM, BN, BK, RM, RN, STG, MODE><<<grid, th>>>(dA, dW, dC, N, K);
  if (regs_out) {
    hipFuncAttributes a{};
    if (hipFuncGetAttributes(&a, (const void *)gemm_v2_kernel<BM, BN, BK, RM, RN, STG, MODE>) ==
        hipSuccess) {
      *regs_out = a.numRegs;
      *lds_out = a.sharedSizeBytes;
    }
  }
  if (occ_out) {
    *occ_out = 0;
    int nb = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &nb, gemm_v2_kernel<BM, BN, BK, RM, RN, STG, MODE>, th, 0) == hipSuccess) {
      *occ_out = nb;
    }
  }
}

#define RD_V2_STG_CASE(STG_)                                                                  \
  case STG_:                                                                                  \
    if (mode == 0)                                                                            \
      launch_v2_one<BM, 128, 64, RM, 8, STG_, 0>(grid0, th, dA, dW, dC, N, K, regs, lds, occ); \
    else if (mode == 1)                                                                       \
      launch_v2_one<BM, 128, 64, RM, 8, STG_, 1>(grid0, th, dA, dW, dC, N, K, nullptr, nullptr, \
                                                 nullptr);                                    \
    else if (mode == 3)                                                                       \
      launch_v2_one<BM, 128, 64, RM, 8, STG_, 3>(grid0, th, dA, dW, dC, N, K, nullptr, nullptr, \
                                                 nullptr);                                    \
    break;

template <int BM, int RM>
void launch_v2(const int stg, const int mode, const dim3 grid, const int th, const __half *dA,
               const block_iq3_s *dW, float *dC, const int N, const int K, int *regs,
               std::size_t *lds, int *occ) {
  const dim3 grid0 = grid;
  switch (stg) {
    RD_V2_STG_CASE(0)
    RD_V2_STG_CASE(1)
    RD_V2_STG_CASE(3)
    RD_V2_STG_CASE(7)
    RD_V2_STG_CASE(9)
    RD_V2_STG_CASE(11)
    RD_V2_STG_CASE(15)
  }
}
#undef RD_V2_STG_CASE

bool launch(const Cfg &c, const int mode, const int M, const std::int64_t N, const std::int64_t K,
            const __half *dA, const block_iq3_s *dW, float *dC, const bool dbuf = false) {
  const dim3 grid((unsigned)(N / c.BN), (unsigned)((M + c.BM - 1) / c.BM), 1u);
  const int th = c.threads();
  const bool pipe = dbuf && mode == 0;
#define RD_DISPATCH(BM_, BK_, RM_)                                                            \
  do {                                                                                        \
    if (pipe) launch_cfg_dbuf<BM_, 128, BK_, RM_, 8>(grid, th, dA, dW, dC, (int)N, (int)K);    \
    else launch_cfg<BM_, 128, BK_, RM_, 8>(mode, grid, th, dA, dW, dC, (int)N, (int)K);        \
  } while (0)
  if (c.BM == 128) {
    if (c.BK == 32) {
      RD_DISPATCH(128, 32, 8);
    } else {
      if (pipe) std::printf("  (duplo buffer indisponivel em BM=128 BK=64: 67,6 KB de LDS)\n");
      launch_cfg<128, 128, 64, 8, 8>(mode, grid, th, dA, dW, dC, (int)N, (int)K);
    }
  } else if (c.BM == 64) {
    if (c.BK == 32) RD_DISPATCH(64, 32, 4);
    else RD_DISPATCH(64, 64, 4);
  } else if (c.BM == 32) {
    if (c.BK == 32) RD_DISPATCH(32, 32, 2);
    else RD_DISPATCH(32, 64, 2);
  } else {
    if (c.BK == 32) RD_DISPATCH(16, 32, 2);
    else RD_DISPATCH(16, 64, 2);
  }
#undef RD_DISPATCH
  return launch_ok("gemm_iq3s_f16");
}

}  // namespace

int main(int argc, char **argv) {
  std::vector<int> ms_list{16, 64, 128};
  int BK_sel = 64, reps = 5, BM_ovr = 0;
  bool DBUF = false, AB = false;
  std::int64_t n_ovr = -1, k_ovr = -1;
  std::string tensor_name;
  bool list_only = false, verify = true;
  const char *model = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--m") == 0 && i + 1 < argc) {
      ms_list.clear();
      const char *v = argv[++i];
      while (*v) {
        ms_list.push_back(std::atoi(v));
        while (*v && *v != ',') ++v;
        if (*v == ',') ++v;
      }
    } else if (std::strcmp(argv[i], "--bk") == 0 && i + 1 < argc) {
      BK_sel = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--bm") == 0 && i + 1 < argc) {
      BM_ovr = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--dbuf") == 0) {
      DBUF = true;
    } else if (std::strcmp(argv[i], "--ab") == 0) {
      AB = true;  // frente G: tabela A/B das variantes de staging
    } else if (std::strcmp(argv[i], "--reps") == 0 && i + 1 < argc) {
      reps = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--n") == 0 && i + 1 < argc) {
      n_ovr = std::atoll(argv[++i]);
    } else if (std::strcmp(argv[i], "--k") == 0 && i + 1 < argc) {
      k_ovr = std::atoll(argv[++i]);
    } else if (std::strcmp(argv[i], "--tensor") == 0 && i + 1 < argc) {
      tensor_name = argv[++i];
    } else if (std::strcmp(argv[i], "--list") == 0) {
      list_only = true;
    } else if (std::strcmp(argv[i], "--no-verify") == 0) {
      verify = false;
    } else if (argv[i][0] != '-') {
      model = argv[i];
    }
  }
  if (!model) {
    std::printf("uso: bench-gemm-quant-gpu <model.gguf> [--m 16,64,128] [--bk 64] [--bm 0|16|32|64|128]\n"
                "                          [--reps 5] [--dbuf] [--tensor NOME] [--n N] [--k K]\n"
                "                          [--list] [--no-verify]\n");
    return 1;
  }

  // ---- 0. o tensor real ---------------------------------------------------
  rdna4::GgufLoader ld;
  std::string err;
  if (!ld.open(model, err)) {
    std::fprintf(stderr, "open: %s\n", err.c_str());
    return 1;
  }
  if (list_only) {
    std::printf("tensores iq3_s no arquivo:\n");
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      const auto &t = ld.meta().tensors[i];
      if ((int)ld.dtype(i) != kIq3S) continue;
      std::printf("  %-42s dims=[%lld, %lld]  %8.1f MB\n", t.name.c_str(), (long long)t.dims[0],
                  (long long)t.dims[1], (double)ld.tensor_bytes(i) / 1e6);
    }
    return 0;
  }

  std::size_t ti = ld.n_tensors();
  if (!tensor_name.empty()) {
    const auto *found = ld.find(tensor_name.c_str());
    if (!found) {
      std::fprintf(stderr, "tensor %s nao existe\n", tensor_name.c_str());
      return 1;
    }
    ti = (std::size_t)(found - ld.meta().tensors.data());
  } else {
    // Preferencia: a forma do `ffn_gate/up` do tronco (K=5120 x N=17408), que e'
    // a que o plano cita; senao o maior iq3_s do arquivo.
    std::size_t best = ld.n_tensors();
    std::uint64_t best_b = 0;
    for (std::size_t i = 0; i < ld.n_tensors(); ++i) {
      if ((int)ld.dtype(i) != kIq3S) continue;
      const auto &t = ld.meta().tensors[i];
      if (t.dims.size() != 2) continue;
      const std::uint64_t b = ld.tensor_bytes(i);
      if (t.dims[0] == 5120 && t.dims[1] == 17408) {
        best = i;
        best_b = b;
        break;
      }
      if (best == ld.n_tensors() || b > best_b) {
        best = i;
        best_b = b;
      }
    }
    if (best == ld.n_tensors()) {
      std::fprintf(stderr, "nenhum tensor iq3_s de 2 dimensoes no arquivo\n");
      return 1;
    }
    ti = best;
  }
  const auto &info = ld.meta().tensors[ti];
  const std::int64_t K_full = (std::int64_t)info.dims[0];
  const std::int64_t N_full = (std::int64_t)info.dims[1];
  const std::int64_t K = (k_ovr > 0) ? k_ovr : K_full;
  const std::int64_t N = (n_ovr > 0) ? n_ovr : N_full;
  std::printf("tensor: %s  [%lld x %lld] iq3_s, %.1f MB; K=%lld N=%lld\n", info.name.c_str(),
              (long long)K_full, (long long)N_full, (double)ld.tensor_bytes(ti) / 1e6,
              (long long)K, (long long)N);
  if (K % kQK != 0 || K % BK_sel != 0 || N % 128 != 0) {
    std::fprintf(stderr, "K tem de ser multiplo de 256 e de BK, N multiplo de 128\n");
    return 1;
  }

  const std::int64_t bpr = K / kQK;
  const std::uint64_t want = (std::uint64_t)N * (std::uint64_t)bpr * kBlockBytes;
  std::vector<std::uint8_t> w_raw;
  if (!ld.load_tensor_range(ti, 0, want, w_raw, err)) {
    std::fprintf(stderr, "load_tensor_range: %s\n", err.c_str());
    return 1;
  }
  std::printf("pesos crus: %.1f MB, %.4f B/peso (media do tensor iq3_s)\n",
              (double)w_raw.size() / 1e6, (double)w_raw.size() / ((double)N * (double)K));

  // ---- 1. ativacoes (f16) e buffers ---------------------------------------
  const int MMAX = *std::max_element(ms_list.begin(), ms_list.end());
  std::vector<uint16_t> hA;
  fill_a(hA, (std::size_t)MMAX * (std::size_t)K, 7u);

  void *dW = nullptr;
  __half *dA = nullptr;
  float *dC = nullptr, *dRef = nullptr;
  if (hipMalloc(&dW, w_raw.size()) != hipSuccess ||
      hipMalloc(&dA, hA.size() * sizeof(uint16_t)) != hipSuccess ||
      hipMalloc(&dC, (std::size_t)MMAX * (std::size_t)N * sizeof(float)) != hipSuccess ||
      hipMalloc(&dRef, (std::size_t)MMAX * (std::size_t)N * sizeof(float)) != hipSuccess) {
    std::fprintf(stderr, "hipMalloc falhou\n");
    return 1;
  }
  (void)hipMemcpy(dW, w_raw.data(), w_raw.size(), hipMemcpyHostToDevice);
  (void)hipMemcpy(dA, hA.data(), hA.size() * sizeof(uint16_t), hipMemcpyHostToDevice);
  const block_iq3_s *const Wblk = (const block_iq3_s *)dW;

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
    std::printf("LUT iq3s_grid lida do device (rdna4::iq3s_grid): 512 entradas, hash=%08x\n", sum);
  }

  // ---- 3. oraculo de CPU num caso pequeno (M=4, N=64) ---------------------
  // O caso pequeno roda com o MESMO N do kernel (N e' o stride de C). A cadeia
  // de verificacao fica: bytes crus -> oraculo de CPU (fp32) == warp-oraculo de
  // GPU (que usa o dequant do motor) == kernel tilejado (f16).
  if (verify) {
    const int Mc = 4, Nc = 64;
    (void)hipMemset(dRef, 0, (std::size_t)MMAX * (std::size_t)N * sizeof(float));
    launch_ref(dA, Wblk, dRef, Mc, N, (int)K);
    if (!launch_ok("ref_kernel (oraculo de CPU)")) return 1;
    (void)hipDeviceSynchronize();
    std::vector<float> got((std::size_t)Mc * (std::size_t)N, 0.f);
    (void)hipMemcpy(got.data(), dRef, got.size() * sizeof(float), hipMemcpyDeviceToHost);
    double max_abs = 0, max_rel = 0, max_mag = 0;
    int n_nonfinite = 0;
    std::size_t first_bad = (std::size_t)-1;
    double bad_val = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
      if (!std::isfinite(got[i])) {
        ++n_nonfinite;
        if (first_bad == (std::size_t)-1) {
          first_bad = i;
          bad_val = (double)got[i];
        }
      }
    }
    if (n_nonfinite) {
      std::printf("  dRef: %d valores nao-finitos; primeiro i=%zu (m=%zu n=%zu) = %g\n", n_nonfinite,
                  first_bad, first_bad / (std::size_t)N, first_bad % (std::size_t)N, bad_val);
    }
    for (int m = 0; m < Mc; ++m) {
      for (int n = 0; n < Nc; ++n) {
        const unsigned char *b = w_raw.data() + ((std::size_t)n * (std::size_t)bpr) * kBlockBytes;
        double acc = 0;
        for (int ib = 0; ib < bpr; ++ib) {
          for (int j = 0; j < kQK; ++j) {
            const double w = cpu_iq3s_weight(b + (std::size_t)ib * kBlockBytes, j, lut.data());
            acc += w * (double)rdna4::fp16_to_float(hA[(std::size_t)m * (std::size_t)K + ib * kQK + j]);
          }
        }
        const double g = got[(std::size_t)m * (std::size_t)N + n];
        max_mag = std::max(max_mag, std::fabs(acc));
        max_abs = std::max(max_abs, std::fabs(g - acc));
        if (std::fabs(acc) > 0) max_rel = std::max(max_rel, std::fabs(g - acc) / std::fabs(acc));
      }
    }
    std::printf(
        "oraculo de CPU (M=%d, N=%d, K=%lld; dequant do motor em fp64 no acumulador):\n"
        "  max |ref| = %.6g ; max |gpu-cpu| = %.3g ; max rel = %.3g\n",
        Mc, Nc, (long long)K, max_mag, max_abs, max_rel);

    // ---- staging ISOLADO contra o mesmo oraculo ---------------------------
    // Primeiros `nchk` sub-blocos de 32 pesos da linha 0: o kernel de staging
    // escreve o que ele poe na LDS e a comparacao e' peso a peso (nao soma).
    const int nsub = 64;  // 8 super-blocos = K 2048
    __half *dStage = nullptr;
    if (hipMalloc(&dStage, (std::size_t)nsub * 32 * sizeof(__half)) != hipSuccess) return 1;
    dump_stage_kernel<<<(nsub + 63) / 64, 64>>>(Wblk, dStage, 0, (int)bpr, nsub);
    if (!launch_ok("dump_stage")) return 1;
    (void)hipDeviceSynchronize();
    std::vector<uint16_t> staged((std::size_t)nsub * 32);
    (void)hipMemcpy(staged.data(), dStage, staged.size() * sizeof(uint16_t), hipMemcpyDeviceToHost);
    double w_rel = 0, w_abs = 0, w_ref = 0;
    int w_bitdiff = 0;
    for (int t = 0; t < nsub; ++t) {
      const unsigned char *blk = w_raw.data() + (std::size_t)(t / 8) * kBlockBytes;
      for (int j = 0; j < 32; ++j) {
        const double want = (double)cpu_iq3s_weight(blk, (t % 8) * 32 + j, lut.data());
        const double gotw = (double)rdna4::fp16_to_float(staged[(std::size_t)t * 32 + j]);
        w_abs = std::max(w_abs, std::fabs(gotw - want));
        w_ref = std::max(w_ref, std::fabs(want));
        if (std::fabs(want) > 0) w_rel = std::max(w_rel, std::fabs(gotw - want) / std::fabs(want));
        if (staged[(std::size_t)t * 32 + j] != rdna4::float_to_fp16((float)want)) ++w_bitdiff;
      }
    }
    std::printf(
        "check do staging (%d sub-blocos = %d pesos da linha 0, contra o oraculo de CPU):\n"
        "  max |w| = %.5g ; max |dw| = %.3g ; max rel = %.3g (f16: 2^-11 = 4,9e-4)\n"
        "  bits diferentes do arredondamento f16 esperado (RN, feito no host): %d de %d\n",
        nsub, nsub * 32, w_ref, w_abs, w_rel, w_bitdiff, nsub * 32);
    (void)hipFree(dStage);
    if (w_bitdiff) {
      std::printf("  REPROVADO: o staging nao reproduz o dequant do motor\n");
      return 1;
    }
    if (max_abs > 1e-3) {
      std::printf("  REPROVADO: o oraculo de GPU nao concorda com o de CPU\n");
      return 1;
    }
  }

  // ---- 4. configuracao por M (mesma geometria da bench-gemm-gpu int8) -----
  // `--bm` forca o tile de M (para varrer ocupacao de LDS x paralelismo); o
  // RM sai de BM/16 no mesmo desenho da bench int8 (TM=16, TN=16, 256 threads).
  static const char *kTags[4] = {"BM16  BN128 RM2 RN8", "BM32  BN128 RM2 RN8",
                                 "BM64  BN128 RM4 RN8", "BM128 BN128 RM8 RN8"};
  auto cfg_for = [&](const int M) {
    int bm = 0;
    if (BM_ovr == 16) bm = 0;
    else if (BM_ovr == 32) bm = 1;
    else if (BM_ovr == 64) bm = 2;
    else if (BM_ovr == 128) bm = 3;
    else {
      bm = (M >= 128) ? 3 : (M >= 64) ? 2 : (M >= 32) ? 1 : 0;
    }
    if (M < 32 && bm > 0) bm = 0;  // tile maior que o M disponivel
    static const int kBM[4] = {16, 32, 64, 128};
    static const int kRM[4] = {2, 2, 4, 8};
    return Cfg{kBM[bm], 128, BK_sel, kRM[bm], 8, kTags[bm]};
  };

  EvPair ev;
  // AQUECIMENTO GLOBAL -- e ele que faz a medicao valer. Medido nesta bancada:
  // a PRIMEIRA configuracao de um processo sai ~47 % mais lenta que as
  // seguintes (1,757 contra 1,195 ms na MESMA config, `--m 128,128,128`), e a
  // partir da segunda a repeticao concorda em 0,1 %. Ou seja: o aquecimento tem
  // de vir ANTES do laco de medicao; o warm-up por modo (abaixo) nao basta.
  {
    const Cfg cw = cfg_for(MMAX);
    auto fn = [&] { (void)launch(cw, 0, MMAX, N, K, dA, Wblk, dC, DBUF); };
    fn();
    (void)hipDeviceSynchronize();
    double warm = 0;
    for (int it = 0; it < 4000 && warm < 600.0; ++it) warm += ev(fn);
    std::printf("(aquecimento global: %.0f ms de GPU em %s)\n", warm, cw.tag);
  }
  // ---- 5. FRENTE G: A/B das variantes de staging --------------------------
  // Tudo na MESMA janela e no MESMO processo: cada variante e' um kernel que so'
  // difere no caminho do dado, entao a razao entre duas linhas e' o efeito da
  // mudanca (a placa e' compartilhada e o absoluto varia entre janelas em 1,5 %).
  if (AB) {
    struct V {
      const char *tag;
      int BM, RM, STG;
      bool i8;
    };
    static const V vs[] = {
        {"V0  base", 128, 8, 0, false},
        {"V1  avec", 128, 8, 1, false},
        {"V2  avec+pf", 128, 8, 3, false},
        {"V3  avec+pf+dbuf", 128, 8, 7, false},
        {"V4  avec+ag", 128, 8, 9, false},
        {"V5  avec+pf+ag", 128, 8, 11, false},
        {"V6  avec+pf+db+ag", 128, 8, 15, false},
        {"I8  int8 x f16 (so' W)", 128, 8, 9, true},
        {"V0  base     BM64", 64, 4, 0, false},
        {"V2  avec+pf  BM64", 64, 4, 3, false},
        {"V5  avec+pf+ag BM64", 64, 4, 11, false},
        {"V6  pf+db+ag BM64", 64, 4, 15, false},
        {"V0  base (repete)", 128, 8, 0, false},
    };
    std::printf("\n== 5. FRENTE G: A/B do staging (mesma janela, mesmo processo) ==\n");
    std::printf(
        "  t_stg = modo 1 (staging com consumo minimo, sem a conta). Nas linhas com '+ag' a\n"
        "  ativacao nao passa pela LDS, entao o modo 1 so' mede o staging do PESO: a fracao\n"
        "  de staging dessas linhas e' um PISO. `bitid` = max|C - C(V0)| (0 = bit a bit igual).\n"
        "  Na linha I8 as duas colunas de tempo sao o A/B do staging DO PESO na mesma grade:\n"
        "  `t_GEMM` = iq3_s -> f16 na LDS (com ag, ou seja sem a ativacao) e `t_stg` = o MESMO\n"
        "  sub-bloco -> int8 na LDS (32 B em vez de 64 B, sem cvt/mul/pack).\n");
    std::printf("%5s %-19s %4s %5s %9s %4s %10s %9s %9s %6s %9s %7s %8s\n", "M", "variante", "th",
                "regs", "LDS KB", "CTA/CU", "t_GEMM ms", "T MAC/s", "t_stg ms", "stg %",
                "G peso/s", "bitid", "rel-L2");

    for (const int M : ms_list) {
      const std::size_t n_el = (std::size_t)M * (std::size_t)N;
      (void)hipMemset(dRef, 0, n_el * sizeof(float));
      launch_ref(dA, Wblk, dRef, M, N, (int)K);
      if (!launch_ok("ref_kernel (A/B)")) break;
      (void)hipDeviceSynchronize();
      std::vector<float> ref(n_el), got(n_el), base(n_el, 0.f);
      (void)hipMemcpy(ref.data(), dRef, n_el * sizeof(float), hipMemcpyDeviceToHost);
      int g_regs = 0, g_occ = 0;
      std::size_t g_lds = 0;

      auto run_v2 = [&](const V &v, const int mode) -> bool {
        const int th = (v.BM / v.RM) * 16;  // TN = 128/8 = 16
        const dim3 g((unsigned)(N / 128), (unsigned)((M + v.BM - 1) / v.BM), 1u);
        int regs = 0, occ = 0;
        std::size_t lds = 0;
        if (v.BM == 128) {
          launch_v2<128, 8>(v.STG, mode, g, th, dA, Wblk, dC, (int)N, (int)K, &regs, &lds, &occ);
        } else if (v.BM == 64) {
          launch_v2<64, 4>(v.STG, mode, g, th, dA, Wblk, dC, (int)N, (int)K, &regs, &lds, &occ);
        } else {
          return false;
        }
        if (mode == 0) {
          g_regs = regs;
          g_lds = lds;
          g_occ = occ;
        }
        return launch_ok("gemm_v2");
      };
      auto timeit = [&](const V &v, const int mode) {
        auto fn = [&] { (void)run_v2(v, mode); };
        fn();
        (void)hipDeviceSynchronize();
        double warm = 0;
        for (int it = 0; it < 4000 && warm < 250.0; ++it) warm += ev(fn);
        double best = 1e9;
        for (int r = 0; r < reps; ++r) best = std::min(best, ev(fn));
        return best;
      };

      bool first = true;
      for (const V &v : vs) {
        if (v.BM > M) continue;
        const int mode_g = v.i8 ? 1 : 0;  // o int8 e' um modo de STAGING
        if (!run_v2(v, mode_g)) break;
        (void)hipDeviceSynchronize();
        int regs = g_regs, occ = g_occ;
        const std::size_t lds = g_lds;
        (void)hipMemcpy(got.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
        double sum_d = 0, sum_r = 0, mx = 0;
        for (std::size_t i = 0; i < n_el; ++i) {
          const double d = (double)got[i] - (double)ref[i];
          sum_d += d * d;
          sum_r += (double)ref[i] * (double)ref[i];
          mx = std::max(mx, std::fabs((double)got[i] - (double)base[i]));
        }
        if (first) {
          base = got;
          mx = 0;  // a primeira linha e' a referencia (V0) do `bitid`
          first = false;
        }
        const double rel_l2 = std::sqrt(sum_d / sum_r);
        const double t0 = timeit(v, mode_g);
        const double t1 = timeit(v, v.i8 ? 3 : 1);
        const double macs = (double)M * (double)N * (double)K;
        const double tmacs = (v.i8 ? 0.0 : macs / (t0 * 1e-3) / 1e12);
        const int m_tiles = (M + v.BM - 1) / v.BM;
        const double wts = (double)N * (double)K * (double)m_tiles;
        std::printf("%5d %-19s %4d %5d %9.1f %4d %10.3f %9s %9.3f %6.1f %9.1f %7.0f %8.2e\n", M,
                    v.tag, (v.BM / v.RM) * 16, regs, (double)lds / 1024.0, occ, t0,
                    v.i8 ? "   (n/a)" : std::to_string(tmacs).c_str(), t1, 100.0 * t1 / t0,
                    wts / (t1 * 1e-3) / 1e9, mx, rel_l2);
        std::fflush(stdout);
      }
    }
  }


  std::printf("%5s %-22s %11s %11s %11s %11s %11s %11s\n", "M", "cfg", "t_GEMM ms", "T MAC/s",
              "TOPS", "t_stage ms", "t_A ms", "t_W ms");
  for (const int M : ms_list) {
    const Cfg c = cfg_for(M);
    if (verify) {
      (void)hipMemset(dRef, 0, (std::size_t)M * (std::size_t)N * sizeof(float));
      launch_ref(dA, Wblk, dRef, M, N, (int)K);
      if (!launch_ok("ref_kernel")) break;
      (void)hipDeviceSynchronize();
      if (!launch(c, 0, M, N, K, dA, Wblk, dC)) break;
      (void)hipDeviceSynchronize();
      const std::size_t n_el = (std::size_t)M * (std::size_t)N;
      std::vector<float> got(n_el), ref(n_el);
      (void)hipMemcpy(got.data(), dC, n_el * sizeof(float), hipMemcpyDeviceToHost);
      (void)hipMemcpy(ref.data(), dRef, n_el * sizeof(float), hipMemcpyDeviceToHost);
      double mx = 0, sum_d = 0, sum_r = 0, mx_rel = 0, mx_ref = 0;
      bool allzero = true;
      for (std::size_t i = 0; i < n_el; ++i) {
        const double d = (double)got[i] - (double)ref[i];
        mx = std::max(mx, std::fabs(d));
        sum_d += d * d;
        sum_r += (double)ref[i] * (double)ref[i];
        mx_ref = std::max(mx_ref, std::fabs((double)ref[i]));
        if (got[i] != 0.f) allzero = false;
      }
      // Erro relativo so' onde ele significa algo: |ref| > 5 % do maior |ref|
      // (com |ref| ~ 1e-3 o relativo mede cancelamento, nao precisao).
      for (std::size_t i = 0; i < n_el; ++i) {
        const double r = std::fabs((double)ref[i]);
        if (r > 0.05 * mx_ref) {
          mx_rel = std::max(mx_rel, std::fabs((double)got[i] - (double)ref[i]) / r);
        }
      }
      std::printf(
          "  check M=%-4d %s (%d threads, grade %u x %u): rel-L2=%.3g  max|d|=%.3g  "
          "max|ref|=%.3g  max rel (|ref|>5%% do max)=%.3g%s\n",
          M, c.tag, c.threads(), (unsigned)(N / c.BN), (unsigned)((M + c.BM - 1) / c.BM),
          std::sqrt(sum_d / sum_r), mx, mx_ref, mx_rel, allzero ? "   TODA ZERO!" : "");
      if (allzero) break;
    }
    double t[3] = {1e9, 1e9, 1e9};
    for (int mode = 0; mode < 3; ++mode) {
      auto fn = [&] { (void)launch(c, mode, M, N, K, dA, Wblk, dC, DBUF); };
      fn();
      (void)hipDeviceSynchronize();
      double warm = 0;  // DPM: a placa cai para ~10 MHz ociosa
      for (int it = 0; it < 4000 && warm < 250.0; ++it) warm += ev(fn);
      for (int r = 0; r < reps; ++r) t[mode] = std::min(t[mode], ev(fn));
    }
    const double macs = (double)M * (double)N * (double)K;
    const double tmacs = macs / (t[0] * 1e-3) / 1e12;
    std::printf("%5d %-22s%11.3f %11.2f %11.1f %11.3f %11.3f %11.3f\n", M,
                DBUF ? (std::string(c.tag) + " +dbuf").c_str() : c.tag, t[0], tmacs,
                2 * tmacs, t[1], t[2], t[1] - t[2]);
    // item 1: staging isolado. Com BM < M o peso e' estagiado DE NOVO por
    // fatia de M (a grade tem M/BM blocos em y), entao o total de pesos
    // estagiados por passada e' N*K*(M/BM) -- e' o custo que decide.
    const int m_tiles = (M + c.BM - 1) / c.BM;
    const double wts = (double)N * (double)K * (double)m_tiles;
    const double src_b = wts * (double)kBlockBytes / (double)kQK;
    const double lds_b = wts * 2.0;
    const int n_cta = (int)(N / c.BN) * m_tiles;
    const double t_w = (t[1] - t[2]) > 0 ? (t[1] - t[2]) : t[1];
    std::printf(
        "      staging (ativacao+peso, modo 1): %.3f ms -> %.1f G pesos/s | fonte %.1f GB/s |\n"
        "        LDS escrita %.1f GB/s | %d CTAs -> %.2f G pesos/s por CTA | %.2f GB/s de fonte "
        "por CTA\n",
        t[1], wts / (t[1] * 1e-3) / 1e9, src_b / (t[1] * 1e-3) / 1e9, lds_b / (t[1] * 1e-3) / 1e9,
        n_cta, wts / (t[1] * 1e-3) / 1e9 / n_cta, src_b / (t[1] * 1e-3) / 1e9 / n_cta);
    std::printf(
        "      staging do peso sozinho (1-2):   %.3f ms -> %.1f G pesos/s | fonte %.1f GB/s | "
        "LDS escrita %.1f GB/s\n",
        t_w, wts / (t_w * 1e-3) / 1e9, src_b / (t_w * 1e-3) / 1e9, lds_b / (t_w * 1e-3) / 1e9);
    const double t_comp = t[0] - t[1];
    std::printf(
        "      conta (GEMM - staging): %.3f ms (=%.0f%% do GEMM) -> %.2f T MAC/s de conta pura\n",
        t_comp, 100.0 * t_comp / t[0], macs / (t_comp * 1e-3) / 1e12);
  }

  std::printf(
      "\n== 4. referencias (mesma maquina, docs/estudo-prefill.md) ==\n"
      "  nosso prefill hoje (matvec em lote) ...........  3,18 T MAC/s\n"
      "  bench-gemm-gpu int8+dp4a (sintetico): M=128 ... 10,34 T MAC/s ; M=512 ... 12,74 T\n"
      "  llama.cpp Vulkan sem coopmat (f16+dot2) ....... 12,26 T MAC/s\n"
      "  llama.cpp Vulkan com coopmat (f16+WMMA) ....... 30,66 T MAC/s\n"
      "  teto de issue do dot2 f16 (RM=RN=8: 0,625 instr/MAC, 3,80e11 slots/s) ... 19,5 T MAC/s\n"
      "  teto do dp4a int8 (mesmo tile: 0,3125 instr/MAC) ......................... 39,0 T MAC/s\n");
  (void)hipFree(dW);
  (void)hipFree(dA);
  (void)hipFree(dC);
  (void)hipFree(dRef);
  (void)hipFree(dLut);
  return 0;
}
