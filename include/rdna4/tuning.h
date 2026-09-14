#pragma once
// ---------------------------------------------------------------------------
// TABELA DE TUNING — gfx1201 (AMD RX 9070 XT, RDNA4), ROCm 7.2
//
// Este e o UNICO lugar onde os parametros medidos deste motor vivem. As tabelas
// de compilacao que os consomem (`MtShape`/`MtIlp`/`MtUnroll` em matvec.cuh,
// `kAttnWarpsPerBlock`/`kAttnSplitWpbLimit` em attn.cuh, `kAttnSplitMin`/
// `kAttnMaxSplits` em graph.cuh) LEEM estes valores -- nao existe uma segunda
// copia dos numeros em lugar nenhum. O gate `check-tuning` (CPU puro, sem GPU e
// sem modelo) imprime esta tabela e a compara com a referencia commitada
// `tests/golden/ml_tuning.txt`: um refactor generico que mexa em qualquer um
// destes numeros quebra o gate em vez de mudar o desempenho em silencio.
//
// Fonte de cada numero: docs/autotuning-gfx1201.md (medido nesta placa, com
// `bench-matvec-shapes-gpu` e `bench-attn-gpu`, aquecimento de DPM >=300 ms e
// A/B intercalado por rodadas). O piso de ruido medido esta naquele documento e
// repetido aqui: nada abaixo dele foi aceito como ganho.
//
// SEM DEPENDENCIA DE HIP de proposito: este header e incluido tanto pelo device
// code quanto pelo teste CPU que faz o gate.
// ---------------------------------------------------------------------------

namespace rdna4 {
namespace tuned {

// ---------------------------------------------------------------------------
// 1. Matvec (GEMV, decode) por tipo de quantizacao
//
// Ordinal do tipo (o mesmo `dt` de matvec_shape()/matvec_launch):
//   1 q8_0    2 q2_K    3 q3_K    4 q4_K    5 q5_K    6 q6_K   7 iq2_xxs
//   8 iq2_xs  9 iq3_xxs 10 iq1_s  11 iq4_nl 12 iq3_s 13 iq2_s  14 iq4_xs
//
// rows  = linhas por CTA (livre/bit-exato: so muda qual CTA calcula qual linha)
// wpr   = warps cooperando por linha (muda a ordem de soma -> medido por tipo)
// ilp   = acumuladores independentes (muda a ordem de soma -> medido por tipo)
// unroll= blocos por iteracao no MESMO acumulador (bit-exato: mesmas operacoes,
//         mesma ordem; so mais loads em voo). ILP e UNROLL sao mutuamente
//         exclusivos no kernel, entao unroll>1 exige ilp==1.
//
// Ganhos medidos em 2026-09-13 (bench-matvec-shapes-gpu, inventario real de 497
// tensores / 10,36 GiB por token, 6 rodadas rotacionadas, piso de ruido 1,001x):
//   - `unroll=2` nos tipos em que o ILP ja era 1: iq3_s 1,075x, iq3_xxs 1,038x,
//     iq2_s 1,011x, iq4_nl 1,055x -> agregado do matvec 1,031x (25,16 -> 24,41 ms
//     de 10,36 GiB por token). Bit-exato por construcao.
//   - `rows=1` nos tipos de linha longa (bpr alto) que NAO usam unroll: iq2_xs
//     1,021x, iq2_xxs 1,009x. Bit-exato (ROWS nao toca a aritmetica).
//
// ATENCAO — as duas alavancas NAO compoem (medido): em iq3_s, `rows=1` vale
// 1,032x sozinho e `unroll=2` vale 1,075x com `rows=8`, mas `rows=1 + unroll=2`
// mediu 8,081 ms contra 7,723 ms do `rows=8 + unroll=2` (4,6% PIOR). Por isso
// iq3_s e iq4_xs ficam com rows=8 (o valor historico) e ganham só o unroll.
// ---------------------------------------------------------------------------
inline constexpr int kMtTypes = 14;
inline constexpr const char *kMtNames[kMtTypes] = {
    "q8_0", "q2_K", "q3_K", "q4_K", "q5_K", "q6_K", "iq2_xxs",
    "iq2_xs", "iq3_xxs", "iq1_s", "iq4_nl", "iq3_s", "iq2_s", "iq4_xs"};
inline constexpr int kMtRows[kMtTypes] = {2, 2, 1, 8, 4, 1, 1, 1, 2, 1, 8, 8, 2, 8};
inline constexpr int kMtWpr[kMtTypes] = {1, 1, 8, 1, 1, 4, 1, 2, 1, 4, 1, 1, 1, 1};
inline constexpr int kMtIlp[kMtTypes] = {4, 2, 2, 2, 2, 2, 2, 4, 1, 1, 1, 1, 1, 2};
inline constexpr int kMtUnroll[kMtTypes] = {1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 2, 2, 2, 1};

constexpr bool mt_tables_valid() {
  for (int i = 0; i < kMtTypes; ++i) {
    if (kMtRows[i] < 1 || kMtRows[i] > 16) return false;
    if (kMtWpr[i] < 1 || kMtWpr[i] > 8) return false;
    if (kMtRows[i] * kMtWpr[i] * 32 > 1024) return false;  // workgroup limit
    if (kMtIlp[i] < 1 || kMtIlp[i] > 4) return false;
    if (kMtUnroll[i] < 1 || kMtUnroll[i] > 4) return false;
    if (kMtIlp[i] > 1 && kMtUnroll[i] > 1) return false;  // static_assert do kernel
  }
  return true;
}
static_assert(mt_tables_valid(),
              "tabela de matvec invalida em include/rdna4/tuning.h (rows*wpr*32 <= 1024, "
              "ILP e UNROLL mutuamente exclusivos)");

// ---------------------------------------------------------------------------
// 2. Atencao
//
// kAttnWarpsPerBlock   warps por CTA no kernel sem split (8/16/32 instanciados)
// kAttnSplitWpbLimit   ate este numero de splits por cabeca o kernel COM split
//                      usa 16 warps; acima disso usa kAttnWarpsPerBlock
// kAttnSplitMin        chave por split: splits = keys/kAttnSplitMin (so para o KV
//                      COM dequant -- ver kAttnSplitCtasDense)
// kAttnMaxSplits       teto de splits por cabeca (o buffer de parciais e
//                      dimensionado por este valor)
// kAttnSplitCtasDense  alvo de CTAs (n_head x splits) quando o KV NAO precisa de
//                      desquantizacao no laco interno (f16/f32): o otimo medido e
//                      um numero FIXO de CTAs, nao uma fracao das chaves
//
// Medido em f16 (head_dim 256, 24 cabecas / 4 kv, 1 token de query) a 4K/16K/64K
// e q4_0 a 131K: a politica que esta aqui e a melhor celula medida ou esta dentro
// do piso de ruido dela (ver docs/autotuning-gfx1201.md §Atencao). Nao foi
// trocada por um "ganho" de 4% que o proprio bench mostrou ser ordem de medicao.
//
// N-POL (2026-09-14): a varredura de splits x contexto derrubou a parte "dentro do
// piso de ruido" para o KV sem dequant. f16, celula 4 splits x 16 warps = 96 CTAs:
// e a MELHOR celula medida em 512, 1K, 2K, 4K, 8K, 16K, 32K, 64K e 131K chaves, e
// o A/B intercalado de 15 rodadas contra a politica antiga (chaves/512, teto 16 =
// ate 384 CTAs) da 1,047x a 4K (15/15), 1,089x a 8K (15/15), 1,030x a 16K (15/15),
// 1,023x a 32K (15/15), 1,020x a 64K (15/15) e 1,004x a 131K (13/15, no piso) --
// piso de ruido medido 1,001-1,005x com 4-11/15 de sinal. f32 (mesma familia, sem
// dequant) confirma: 1,076x a 16K e 1,021x a 64K (15/15).
// O KV quantizado mediu o OPOSTO, e por isso NAO entra nesta regra: com q4_0/q8_0
// a politica antiga (16 splits) e a melhor celula a 16K/64K/131K e 4 splits perde
// 6-13% (0/15 rodadas). O que separa os dois casos e o dequant estar ou nao no
// laco interno -- e nao o numero de chaves.
// ---------------------------------------------------------------------------
// kAttnSplitWpbWide    a partir deste numero de splits por cabeca a CTA volta a
//                      ser larga (16 warps). A regra medida NAO e monotonica:
//                      poucos splits -> CTA larga (caminhada de chaves longa por
//                      warp), splits medios (5..15) -> CTA estreita (o merge por
//                      LDS domina), muitos splits -> CTA larga de novo (com o KV
//                      q4_0 a caminhada fica curta e o que falta e paralelismo).
// kAttnSplitCtasDense: 4 splits x 24 cabecas = 96 CTAs medidos como melhor celula
// de 512 a 131072 chaves com KV f16 (e 4K/16K/64K com f32). Com 24 cabecas da 4
// splits por cabeca, que e <= kAttnSplitWpbLimit, entao a CTA e a larga (16 warps)
// -- a celula 4x16 medida. Para um modelo com outro n_head a regra escala pelo
// numero de CTAs, nao pelo numero de splits (e cai em kAttnMaxSplits se preciso).
inline constexpr int kAttnSplitCtasDense = 96;

inline constexpr int kAttnWarpsPerBlock = 8;
inline constexpr int kAttnSplitWpbLimit = 4;
inline constexpr int kAttnSplitWpbWide = 16;
inline constexpr int kAttnSplitMin = 512;
inline constexpr int kAttnMaxSplits = 16;

// ---------------------------------------------------------------------------
// 3. Prefill em batch (matvec com N tokens)
//
// O kernel batched tem instanciacoes de compilacao para N = {2,3,4,8,16}; o
// prefill do CLI decompoe o prompt com o maior N que couber (16,8,4,3,2) e so
// cai no caminho por token quando sobra 1 token -- que e o otimo medido, porque
// o custo POR TOKEN cai monotonicamente com N (check-batch-gpu: 16,3 / 7,7 /
// 4,9 / 1,9 / 0,87 ms por token para N = 2/3/4/8/16).
// ---------------------------------------------------------------------------
inline constexpr int kBatchNs[] = {2, 3, 4, 8, 16};
inline constexpr int kBatchCount = 5;
inline constexpr int kBatchCap = 16;

}  // namespace tuned
}  // namespace rdna4
