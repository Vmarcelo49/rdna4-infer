# Plano do prefill: do chunk de 16 ao tilejado com unidade de matriz

Este é o plano de implementação que sai do estudo de 14/09 (`docs/estudo-prefill.md` e as quatro
frentes `docs/estudo-prefill-{a-vulkan,b-mmq,c-nosso,d-wmma}.md`). Todo número aqui tem a medição
que o sustenta; nenhuma escolha é por preferência. O que **não** está medido está marcado
`INFERIDO`.

## 0. Alvo e degraus

| degrau | o que muda | ganho medido/derivado | prefill esperado |
|---|---|---|---|
| **hoje** | chunk 16 + GEMV em lote (dp4a, 1 saída/thread, sem LDS) | — | **123,4 tok/s** (`--prefill 512`, melhor de 3) |
| **D1** | chunk maior com o MESMO kernel | **MEDIDO: −18 % em N=32, −71 % em N=64** (ver §0.1) | ~100 / 58 tok/s — **não fazer** |
| **D2** | D1 + GEMM tilejado com staging na LDS, caminho vetorial (`dot2`/dp4a) | protótipo: 10,34 T MAC/s em M=128 vs 3,18 T do motor = **3,24×** | **~400-480 tok/s** |
| **D3** | D2 + laço interno na **unidade de matriz** (coopmat/WMMA f16→f32) | 1196/478 = **2,50×** (medido no llama.cpp, mesma máquina) | **~1000-1200 tok/s** |
| referência | llama.cpp Vulkan, `-ub 512` | — | 1170-1196 tok/s |

Os degraus não são independentes: **D2 não rende nada sem D1** (medido: o mesmo GEMM tilejado em
M=16 dá 3,46 T MAC/s = 1,08× o kernel de hoje; em M=128 dá 10,34 T = 3,24×). E **D3 não existe
sem o staging de D2** — a unidade de matriz precisa ler os operandos da LDS.


### 0.1 O degrau D1 foi medido e REPROVOU (14/09)

A previsão do modelo de custo era +10 % em N=64 (só a parcela fixa de 13,9 ms amortiza). Medido
com `bench-matvec-shapes-gpu --batch 8,16,32,64` (inventário real, min de 3):

| N | ms/token | vs N=16 | registradores (`iq3_s`) | scratch |
|---|---|---|---|---|
| 8 | 8,074 | +16 % | — | 0 |
| **16** | **6,964** | — | **62** | 0 |
| 32 | 8,242 | **+18 %** | 78 | 0 |
| 64 | 11,916 | **+71 %** | **133** | **0 (sem spill)** |

**Mecanismo**: `acc[N][ILP]` custa N registradores por thread; em N=64 o `iq3_s` vai de 62 para
133 registradores, o que **derruba a ocupação pela metade** (133 × 256 threads = 34 k registradores
por CTA) — e como este kernel é limitado por latência, perder ocupação custa mais do que o peso
amortizado ganha. **Não há spill** (`localSizeBytes = 0` em todos), então não é transbordo: é
ocupação.

Consequências para o plano:
1. **O degrau D1 não existe como degrau.** Subir o chunk sem trocar o kernel é **regressão**.
2. Isso *reforça* a dependência do D2: quem dá reuse em M grande é o **staging na LDS** (o peso
   entra uma vez por tile e serve os 128 tokens), não mais acumuladores por thread. O tilejado
   usa 8×8 = 64 acumuladores *fixos* (independentes de M), com o M grande multiplicando o
   trabalho por CTA e não os registradores.
3. O `kMaxBatch = 16` do motor não é um número mal escolhido para o kernel que existe — é o ótimo
   dele (medido). Só faz sentido mudar junto com o kernel novo.

## 1. Por que a unidade de matriz é obrigatória (e não uma otimização)

Medido, nesta máquina:

| caminho | instruções por 4 MAC | MAC por instrução de warp | teto prático |
|---|---|---|---|
| dp4a no kernel de hoje (ISA, medido) | 3,31 (33 % esperas) | 44,7 | ~27 TOPS, e realizamos 6,4 |
| dp4a num tile bom (4×4 + LDS) | 2,5-3,0 | 128 | **~12-13 T MAC/s** — o protótipo chegou em 12,74 |
| f16 `v_dot2_f32_f16` empacotado | ~2,56 | 64 | 39-49 TOPS, **saturado de issue** (o fallback do Vulkan roda a ~94 % dos slots) |
| **matriz (WMMA/coopmat 16×16×16)** | **~0,0103** | **4096** | **97,4 T MAC/s fp16** (o mesmo teto do dp4a *por ciclo*, mas a 1/250 das instruções) |

O protótipo `bench-gemm-gpu` (12,74 T MAC/s em M=512) já está a **65-78 % dos slots de issue** do
cartão. Ou seja: **o caminho vetorial está no fim da estrada; de ~480 tok/s para cima só existe a
unidade de matriz.** Não há terceira via, e a única pergunta em aberto é *qual* família de matriz
— que a frente D está medindo agora (f16 WMMA vs int8 WMMA vs dp4a, em TOPS, neste cartão).

## 2. Decisões fixadas pelo estudo

1. **f16, não int8, para os tipos IQ** (frente B §7, com argumento de precisão e de código):
   arredondar um peso de 3-4 bits para f16 erra 4,9e-4 — 20-100× menos que o erro da própria
   quantização —, enquanto o int8 introduz uma quantização nova de 8 bits na ativação; e f16
   dispensa o quantizador no layout MMQ, os termos de correção e o staging int8 com sinal.
   **Acumulador f32 sempre**: nunca `v_pk_fma_f16`.
   → Consequência de projeto: **a ativação do prefill passa a ser f16, não `q8_1`**, o que de
   quebra elimina o kernel `act_quant` que a frente C mediu em **0,227 ms/token**.
2. **O dp4a fica no decode** (M=1): 4 MAC/lane contra 2 do `dot2`. Não mexer no caminho por
   token, que está a 1,16-1,30× do llama.cpp e é bit-exato com gate.
3. **Alvo de chunk = 128, não 512.** A curva medida do llama.cpp tem o joelho em 128
   (16 → 199, 64 → 664, **128 → 1008**, 256 → 1018, 512 → 1170). 128 é também o `BM` do tile
   deles (`ggml-vulkan.cpp:4524-4532`), então o mesmo número serve para o tile e para o chunk.
4. **Limiar GEMV/GEMM explícito**, como o `mul_mat_vec_max_cols = 8` do Vulkan
   (`ggml-vulkan.cpp:404`, decisão em `:10433`): com N ≤ 8 usa o kernel por token; acima disso,
   o GEMM. Hoje `matvec_launch_batch` é a única porta e não há segunda opção.
5. **Tile**: `BM=128 × BN=128 × BK=32`, 256 threads, `RM=RN=8` por thread (o protótipo já usa
   exatamente isto e passa na verificação contra oráculo). LDS: A 128×32×2 B = 8 KB, W
   128×32×2 B = 8 KB, duplo buffer = 32 KB de 64 KB disponíveis (`maxComputeSharedMemorySize =
   65536` no cartão, medido pela frente A) — sobra para a escada de `coopmat` (24 KB na versão do
   Vulkan).

## 3. O que muda no motor, arquivo por arquivo (`INFERIDO` quanto ao esforço)

| # | mudança | arquivos | por que |
|---|---|---|---|
| 1 | chunk do prefill 16 → 128 | `include/rdna4/graph.cuh` (`kMaxBatch`, `batch_supported`, buffers `d_aqb_`/`d_xb_`), `src/main.hip` (`prefill_ids`) | sem isto o GEMM não tem M para amortizar (medido: 1,08× em M=16) |
| 2 | GEMM f16 tilejado + staging na LDS | novo `include/rdna4/gemm.cuh`; variante bench primeiro | o kernel do D2/D3 |
| 3 | ativação f16 no lote (substitui `quantize_q8_1_batch`) | `include/rdna4/matvec.cuh` (`quantize_batch`), `graph.cuh` | remove `act_quant` (0,227 ms/token) e é o formato que o `coopmat` consome |
| 4 | atenção causal em lote com M=128 | `include/rdna4/attn.cuh` (`attn_batch_kernel`, `attn_split_batch_launch`) | hoje o lote é por token; a frente C mediu `attention` + `qk_norm_rope_kv` + `attn_gate_out` em 0,088 ms/token, e ela precisa acompanhar o M |
| 5 | laço sequencial do GDN dentro de um chunk de 128 | `include/rdna4/gdn.cuh` (`delta_rule`, conv) | a recorrência continua sequencial; só a *projeção* vira GEMM (é o que o llama.cpp faz: matmul com M grande e o scan por dentro) |
| 6 | laço interno `coopmat` f16→f32 | `gemm.cuh` (só o corpo interno muda) | D3 |

## 4. Gates (o caminho **não** é bit-exato, e isso é uma decisão consciente)

O GEMM muda a ordem das somas e o tipo da ativação, então **não vale bit-exatidão** — o
`PLAN.md` M2/M3 já fixou o protocolo para mudança numérica:

- `scripts/compare_ppl.sh` (PPL contra llama.cpp por posição) com limite declarado **0,5 %**;
- `check-regression-gpu` e `check-graph-gpu` (oráculo por nó) para a estrutura;
- `scripts/check_golden_run.sh` para a geração gananciosa;
- o caminho antigo fica atrás de um `RD_*` explícito (padrão da casa) para A/B intercalado;
- **precisão do f16 precisa de um número, não de uma promessa**: medir a KL/PPL do prefill em
  f16 contra o prefill em `q8_1`+dp4a na mesma prompt, e reportar as duas.

## 5. Riscos, e o que já os reduz

| risco | estado |
|---|---|
| staging da dequantização custa mais que o ganho | medido do lado deles: o mesmo staging é usado a 478 tok/s **sem** unidade de matriz; e o nosso protótipo com staging já dá 3,24× em M=128 |
| LDS estoura | 32 KB de 64 KB no duplo buffer; o tile do Vulkan usa 24 KB (frente A) |
| ganho some no andaime | a frente C mediu: o andaime é 9,7 % do prefill e o despacho 0,06 % — o ganho não some |
| chunk 128 quebra a atenção/GDN | é o item 4/5 da tabela; a frente C mediu os dois como 8,7 % do prefill somados, então há espaço para pagar |
| `coopmat` não ser alcançável do HIP/gfx1201 | é exatamente o que a frente D mede agora; se falhar, o plano para em D2 (~480 tok/s = 3,9×) |

## 6. Ordem de execução proposta

1. **Medir a frente D** (WMMA f16/int8 vs dp4a em TOPS neste cartão) — decide se D3 é viável e
   com que família.
2. **Chunk 16 → 128** com o kernel atual estendido (N=64/128 no `matvec_kernel_batch`): entrega
   a infraestrutura e mede o ganho real do D1 (previsto +10 %, e a previsão é falsificável).
3. **GEMM f16 tilejado** (`gemm.cuh`) atrás de `RD_GEMM=1`, gate numérico, A/B contra o kernel
   atual na mesma árvore: alvo 3,2×.
4. **Laço `coopmat`** com o mesmo staging: alvo 2,5× sobre o passo 3.
5. Só depois: as fusões (2,2 ms/token no decode, 0,06 % no prefill) e a atenção GQA por workgroup
   (contexto longo).
