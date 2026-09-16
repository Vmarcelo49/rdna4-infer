# Autotuning no gfx1201 (RX 9070 XT) — task 6 + suite de regressão (task 7)

Este documento é o registro do **tuning medido** deste motor na placa real e da
**suíte de regressão numérica** que passa a ser gate de pré-merge. Ele responde
quatro perguntas, sempre com número medido nesta máquina:

1. qual é o espaço de busca de verdade (enumerado a partir do código);
2. o que foi medido, com que metodologia e com que **piso de ruído**;
3. o que foi **embarcado** e onde (e o que foi **rejeitado com dado**);
4. como rodar os gates e como **re-medir** quando a placa/compilador mudar.

Base: `feat/autotuning` sobre `aa15eea`. Baseline que não pode regredir (medido
antes deste trabalho): decode 29,3 tok/s a 4096, 26,8 no fim de 4K, prefill
~70 tok/s em batch, 64K f16 18,9 tok/s, 131K q4_0 13,0 tok/s, PPL dentro de
0,25 % do llama.cpp.

---

## 0. O que foi embarcado (resumo de uma tela)

| # | mudança | onde | ganho medido | bit-exato? |
|---|---|---|---|---|
| 1 | `unroll=2` (MLP com acumulador único) em `iq3_s`, `iq3_xxs`, `iq2_s`, `iq4_nl` | `include/rdna4/tuning.h` (`kMtUnroll`), consumido por `MtUnroll` em `matvec.cuh` | `iq3_s` **1,073x**, `iq3_xxs` 1,031x, `iq2_s` 1,011x, `iq4_nl` 1,055x | **sim** (mesmas ops, mesma ordem) |
| 2 | `rows=1` (linhas por CTA) em `iq2_xs` e `iq2_xxs` | `tuning.h` (`kMtRows`) | 1,020x e 1,002–1,009x | **sim** (ROWS só move qual CTA calcula qual linha) |
| 3 | CTA larga (16 warps) volta a partir de 16 splits na atenção | `tuning.h` (`kAttnSplitWpbWide`) + 3 linhas em `attn.cuh` (`attn_split_wpb`) | 131K com KV q4_0: **1,050x** (12/15 rodadas de A/B intercalado) | não (equiv. numérica: ordem do merge) |
| 4 | **gate**: tabela de tuning vs referência commitada | `tests/check_tuning.cpp` + `tests/golden/ml_tuning.txt` (alvo `check-tuning`) | — | — |
| 5 | **gate**: suíte de regressão numérica | `tests/check_regression_gpu.hip` + `scripts/check_regression.sh` + `tests/golden/regression_greedy_f16.txt` | — | — |

Agregado do matvec no inventário real (497 tensores, 10,36 GiB por token):
**25,2 → 24,5 ms por token = 1,028x** (mais detalhe na §2). O resto do que foi
varrido **não passou do piso de ruído** e está registrado na §4 — inclusive o que
já tinha sido rejeitado com dado em `docs/rocm-estudo.md` e foi re-medido aqui.

---

## 1. O espaço de busca, enumerado a partir do código

Nada aqui é "achismo": cada linha aponta o sítio do código onde o parâmetro vive.

### 1.1 Matvec (GEMV do decode) — 14 tipos de quantização

`matvec_kernel_gen<T, ROWS, WPR, ILP, PF, MINB, UNROLL>` em
`include/rdna4/matvec.cuh`:

| eixo | o que é | onde | natureza |
|---|---|---|---|
| `ROWS` | linhas por CTA (grid = ⌈nrows/ROWS⌉) | `MtShape<Dt>::rows` | **livre/bit-exato** (só muda o mapeamento CTA→linha) |
| `WPR` | warps cooperando por linha | `MtShape<Dt>::wpr` | muda a ordem de soma (redução por LDS) |
| `ILP` | acumuladores independentes | `MtIlp<Dt>::value` ∈ {1,2,4} | muda a ordem de soma |
| `UNROLL` | blocos por iteração no **mesmo** acumulador | `MtUnroll<Dt>::value` (novo) | **bit-exato** (mesmas ops, mesma ordem) |
| `PF` | prefetch L2 real (`__builtin_amdgcn_s_prefetch_data`) | `PF` | livre (não muda aritmética) |
| `MINB` | `__launch_bounds__` mínimo de blocos/CU (orçamento de registrador) | `MINB` | livre |
| dispatch | kernel por tipo, sem `switch` em runtime | `matvec_launch()` | — |

Restrições estruturais que reduzem o espaço: `ILP>1` e `UNROLL>1` são
**mutuamente exclusivos** (`static_assert` no kernel) e `ROWS*WPR*32 <= 1024`
(limite de workgroup). Os valores por tipo estão em `MtShape`/`MtIlp`/`MtUnroll`
(13+14+14 combinações legais por tipo, 14 tipos).

### 1.2 Prefill em batch — N do kernel batched

`matvec_kernel_batch<T, ROWS, WPR, ILP, N>` + `matvec_launch_batch()`
(`matvec.cuh`) tem instanciações de compilação `N ∈ {2,3,4,8,16}`
(`matvec_batch_cap() == 16`), e a decomposição do prompt no CLI
(`prefill_ids`, `src/main.hip:167`) usa o maior N que couber (16,8,4,3,2), caindo
no caminho por token só quando sobra 1. O espaço de busca aqui é: (a) *quais* N
existem, (b) *como* o prompt é decomposto.

### 1.3 Atenção

| eixo | onde | observação |
|---|---|---|
| `kAttnWarpsPerBlock` (WPB) | `attn.cuh` (lido de `tuning.h`) | template `WPB` do kernel, 8/16/32 instanciados |
| política de WPB por nº de splits | `attn_split_wpb()` | `splits <= kAttnSplitWpbLimit → 16 warps`, senão `kAttnWarpsPerBlock`; **agora** também `splits >= kAttnSplitWpbWide → 16` |
| nº de splits por cabeça | `graph.cuh:attn_splits_for()` | `keys/kAttnSplitMin` limitado a `kAttnMaxSplits`; `RD_ATTN_SPLITS` força (diagnóstico) |
| caminho de carga do KV | `kv_load8<KvType>()` quando `head_dim/32 == 8` | head_dim 256 ⇒ `dpw=8` ⇒ sempre `kv_load8` (carga vetorizada); `kv_load` só no caminho genérico |
| tipo de KV | `--cache-type-k/-v` | f16 (default), q8_0, q4_0, f32 |

### 1.4 Fora do escopo (já rejeitado com dado, e não re-medido)

`docs/rocm-estudo.md` §D: `minb` (0,82x — 18 % **pior**), prefetch L2 (0,98x),
HIP graph do matvec (1,06x), `WPB=32`, compartilhar K/V entre cabeças GQA
(8–12x mais lento), fusão de kernels pequenos (estimativa, não medida). Nada
disso foi reaberto aqui; o que foi re-medido está na §4 com o número novo.

---

## 2. Metodologia e piso de ruído (o que torna os números utilizáveis)

Esta placa cai para um estado de DPM profundo entre kernels (SCLK 9–16 MHz
observado): **toda** medição usada aqui aquece até ≥300–400 ms de tempo de GPU
antes de cronometrar (`PLAN.md` M2 §"Passo 5b"), e a medição é feita por par de
eventos em torno de N lançamentos (nunca um `sync` por lançamento).

Três níveis de ruído foram medidos **nesta campanha**, e nenhum ganho abaixo do
nível correspondente foi aceito:

| harness | protocolo | piso de ruído medido |
|---|---|---|
| `bench-matvec-shapes-gpu` | 6 rodadas **rotacionadas**, mínimo por configuração, inventário real (497 tensores / 10,36 GiB por token) | **1,001x** (`ship` vs `ship2`, mesma configuração, células independentes) — por tipo, ≤1,01x em todos os tipos materiais |
| `bench-attn-gpu` (varredura de células) | 3 rodadas rotacionadas, mínimo por célula | **até 1,2x** entre células — a varredura sequencial NÃO é confiável no nível de poucos por cento (ver §4.3) |
| `bench-attn-gpu --no-sweep --ab/--cand` | **A/B intercalado**, 15 rodadas, ordem invertida a cada rodada, referência e candidato na mesma rodada | **1,000x a 1,012x** (a referência contra ela mesma) + teste de sinal (quantas rodadas o candidato ganhou) |

Regra que este trabalho seguiu: **um ganho só foi embarcado se (a) ficou acima
do piso de ruído do harness que o mediu e (b) ganhou a maioria das rodadas** no
protocolo intercalado. Onde o número ficou na fronteira, ele está marcado como
marginal na §3.

---

## 3. Tabela de tuning medida (config → medido, por tipo)

`ms(tok)` = custo daquele tipo **por token**, extrapolado para todos os tensores
do inventário real (não o "maior tensor do tipo" — essa foi a lição do M2). Piso
de ruído da corrida: **1,001x**. Fonte: `build/bench-matvec-shapes-gpu <model>
--rows 2,4,8 --unroll 2 --repeat-ship --reps 6`.

| tipo | bytes(tok) | share | **ship (novo)** | rows<8> | rows<4> | rows<2> | unroll 2 | ILP 1 | ILP 2 | ILP 4 |
|---|---|---|---|---|---|---|---|---|---|---|
| iq3_s | 3,717 GB | 33,4 % | **7,740** | 8,367 | 8,339 | 8,162 | 7,740 | 10,27 | 12,24 | 12,01 |
| iq4_xs | 2,596 | 23,3 % | **4,958** | 4,962 | 4,962 | 4,965 | 4,957 | — | 4,961 | 4,956 |
| iq3_xxs | 1,863 | 16,7 % | **4,383** | 4,630 | 4,612 | 4,538 | 4,328 | 5,865 | 6,322 | 6,268 |
| q5_k | 0,986 | 8,9 % | **1,691** | 1,684 | 1,692 | 1,693 | 1,685 | 1,698 | 1,685 | 1,696 |
| iq2_s | 0,581 | 5,2 % | **1,134** | 1,142 | 1,142 | 1,145 | 1,133 | 1,958 | 2,221 | — |
| q3_k | 0,401 | 3,6 % | **1,305** | 1,393 | 1,399 | 1,340 | 1,294 | 1,218 | 1,276 | 1,243 |
| iq2_xs | 0,292 | 2,6 % | **0,666** | 0,722 | 0,710 | 0,681 | 0,647 | 1,291 | 1,321 | 1,296 |
| iq2_xxs | 0,269 | 2,4 % | **0,645** | 0,644 | 0,651 | 0,646 | 0,628 | 1,145 | 1,226 | 1,233 |
| q4_k | 0,233 | 2,1 % | **0,570** | 0,568 | 0,568 | 0,568 | 0,559 | 0,567 | 0,569 | 0,564 |
| q2_k | 0,117 | 1,1 % | **0,266** | 0,264 | 0,265 | 0,265 | 0,271 | 0,264 | 0,265 | 0,269 |
| iq1_s | 0,035 | 0,3 % | **0,115** | 0,122 | 0,114 | 0,111 | 0,111 | 0,114 | 0,113 | 0,116 |
| q8_0 | 0,025 | 0,2 % | **0,992** | 0,996 | 0,992 | 0,990 | 1,007 | 1,233 | 1,003 | 0,992 |
| q6_k | 0,004 | 0,0 % | **0,016** | 0,031 | 0,016 | 0,016 | 0,016 | 0,029 | 0,016 | 0,016 |
| iq4_nl | 0,003 | 0,0 % | **0,013** | 0,014 | 0,014 | 0,014 | 0,013 | 0,014 | 0,014 | 0,014 |
| **total** | **11,12 GB** | 100 % | **24,46–24,49** | 25,56 | 25,49 | 25,15 | 24,49 | 30,6 | 33,2 | 32,9 |

(colunas `rows`/`unroll`/`ILP` = a variante forçada contra o `ship` da mesma
corrida; `rows<8>` é a configuração **antiga** para `iq2_xs`/`iq3_s`/`iq4_xs`.)

Leituras diretas da tabela:

* **`UNROLL=2` é o único ganho grande e é bit-exato** — e só onde o `ILP` já era
  1: `iq3_s` 8,367 → 7,740 (−7,5 %), `iq3_xxs` 4,538 → 4,383 (−3,4 %),
  `iq2_s` −1,0 %, `iq4_nl` −5,5 %. Onde o `ILP` era >1, trocar por `UNROLL=2`
  **piora** (`q8_0` 0,992 → 1,007; `q2_k` 0,266 → 0,271) ou empata.
* **`ILP` sozinho não é a alavanca**: forçar `ILP` globalmente em 1/2/4 dá 0,82x/
  0,76x/0,77x — o valor por tipo da tabela é o que importa (`iq3_s` com `ILP=2`
  custa 12,24 ms contra 7,74 ms do `UNROLL=2`).
* **`ROWS` é livre mas não é indiferente**: em `iq3_s` `rows=1` vale 1,032x
  sozinho; em `q4_k`/`q5_k`/`iq4_xs` `rows` não move nada (≤0,5 %); em
  `iq6_k`/`iq4_nl` (classes minúsculas) o ruído domina.
* **As alavancas NÃO compõem** (medido, e é a armadilha desta campanha):
  `rows=1 + unroll=2` em `iq3_s` mediu **8,081 ms** contra **7,723 ms** do
  `rows=8 + unroll=2` — 4,6 % **pior**. Por isso `iq3_s`/`iq4_xs` mantêm `rows=8`
  e ganham só o unroll (documentado dentro de `tuning.h`).

### 3.1 Prefill em batch (N)

`check-batch-gpu` (mesma árvore): custo **por token** cai monotonicamente com N —
16,3 ms/tok (N=2), 7,7 (3), 4,9 (4), 1,9 (8), **0,87 (16)** — e as 5 instanciações
existem justamente para isso. A decomposição "maior N primeiro" do CLI é o ótimo
dessa curva: **nada a mudar**; qualquer reordenação que use N menor por token é
pior por construção. Isto está no `tuning.h` (`kBatchNs`) e no gate.

### 3.2 Atenção (varredura + A/B intercalado)

f16, head_dim 256, 24 cabeças / 4 kv, 1 token de query (por camada; 16 camadas):

| keys | ship policy | melhor célula | A/B intercalado (15 rodadas) | veredito |
|---|---|---|---|---|
| 4097 | 8 splits × 8 warps = 0,0860 ms | 4 × 16 = 0,0823 | **1,046x, 15/15 rodadas** | real, mas vale 0,06 ms/token (0,18 % do passo) |
| 16385 | 16 × 8 = 0,2789 | 4 × 16 = 0,2666 (13/15) / 8 × 16 = 0,2664 (11/15) | 1,046x / 1,047x | real; vale 0,20 ms/token (~0,5 %) |
| 65537 | 16 × 8 = 0,9955 | 16 × 16 = 0,9914 (6/15) / 24 × 8 (0/15) | 1,004x / 0,983x | **nada acima do ruído** (piso 1,0–1,2 %) |
| 131073 f16 | 16 × 8 = 1,9581 | 16 × 16 (9/15) / 24 × 16 (5/15) | 0,994x / 0,991x | nada acima do ruído |
| 131073 q4_0 | 16 × 8 = 2,4460 | 24 × 16 = 2,2509 (12/15) / 16 × 16 = 2,3300 (12/15) | **1,087x / 1,050x** | **real e material**: 1,86 ms/token = +2,4 % no decode a 131K |

O que foi **embarcado** de tudo isso é só a parte que não regride em nenhum
contexto: `kAttnSplitWpbWide = 16` faz a CTA voltar a ser larga (16 warps) a
partir de 16 splits. Efeito por contexto: 4K **inalterado** (8 splits → 8 warps),
16K 0,989x (ruído), 64K 1,000x, 131K f16 0,994x, **131K q4_0 1,050x**. Isto é,
troca-se ≤1 % (dentro do ruído) num contexto por +5 % no contexto mais longo.

O que **não** foi embarcado, com o motivo medido:

* **`kAttnMaxSplits` 16 → 24**: em q4_0/131K dá +2,8 % (24×8) e +8,7 % (24×16),
  mas em f16 16K/64K/131K as 24 splits medem 0,921x/0,974x/0,991x — **regressão**
  clara. O ótimo do número de splits depende do tipo de KV (a linha q4_0 é 3,5x
  menor que a f16), e uma constante única não expressa os dois. Candidato
  registrado na §5.
* **"poucos splits + CTA larga" a 4K/16K** (4×16: 1,046x no kernel): a política
  atual casa `splits = keys/kAttnSplitMin` com o WPB por *número de splits*, e
  para chegar em 4 splits a 4K seria preciso mexer em `kAttnSplitMin` (o que
  piora 16K/64K) ou tornar a regra dependente de `keys` (fora do escopo desta
  tarefa: é política, não tabela). Ganho end-to-end de 0,18 % a 4K e ~0,5 % a
  16K — abaixo ou na fronteira do ruído do decode (±1–2 %).
* **`WPB=32`**, **`kAttnWarpsPerBlock=16`**, **GQA sharing**: continuam
  piores/neutros, como em `docs/rocm-estudo.md` §D.

### 3.3 Fim a fim: A/B controlado, dois binários, janela limpa

Para não comparar sessões diferentes, a tabela abaixo é de um A/B **intercalado
na mesma janela** (GPU livre, verificada antes de cada rodada — ver §7.1): o
binário "old" é o `tuning.h` com os valores do HEAD (unroll=1 em tudo, `rows` do
HEAD, regra de WPB sem a ponta alta), o "new" é a tabela medida. Cada linha é
`bench -n 32` (ou 24) com aquecimento dentro da ferramenta.

| configuração | rodadas old | rodadas new | ratio (média) | rodadas em que new ganhou |
|---|---|---|---|---|
| decode curto, posições ~5–37 (`-p "The capital of France is"`) | 27,91 / 27,54 / 27,68 | **28,38 / 28,22 / 28,44** | **+2,3 %** | **3/3** |
| decode no fim de 4K (cache sintético) | 28,25 / 27,46 / 27,11 | 29,16 / 27,49 / 27,17 | +1,2 % | 3/3 (2 empates) |
| decode 131K com KV q4_0 | 12,73 / 13,10 | 13,12 / 12,92 | +0,8 % | 1/2 |

Leitura honesta: o ganho do matvec (1,028x, medido com piso de 1,001x) aparece
**fim a fim no contexto curto, +2,3 %, 3/3 rodadas** — que é o caso do baseline
documentado. No fim de 4K e a 131K q4_0 a diferença fim a fim fica **dentro do
ruído** daquelas configurações (±3 % entre rodadas, porque metade do tempo é
atenção com cache sintético), mesmo que o ganho no kernel esteja medido: é
exatamente a distinção que este documento faz entre "medido no kernel" e
"medido no motor".

O prefill em batch não mudou (69–73 tok/s em 512 tokens, dentro do ruído).

---

## 4. Rejeitados **com dado** (o que não vale repetir)

1. **`rows` global**: `rows ∈ {1,2,4,8}` muda ≤1 % no agregado (1,010x/1,005x/
   0,991x/0,988x em 3 corridas independentes) → por tipo só rende nos dois casos
   da §3, e `rows=1` em `iq3_s` só é ganho **quando não há unroll**.
2. **`ILP` global**: 0,82x (1), 0,76x (2), 0,77x (4). O `ILP` por tipo é carga
   histórica medida; a alavanca para os tipos de `ILP=1` é o `UNROLL`.
3. **`unroll=4`**: 0,917x. **`unroll=2` nos tipos com `ILP>1`**: 0,98–1,02x.
4. **Prefetch L2 real** (`__builtin_amdgcn_s_prefetch_data`): 0,984x. Confirma
   `docs/rocm-estudo.md` §D.4 com o intrínseco que de fato executa.
5. **`minb`** (orçamento de registrador): não re-medido (0,82x já medido);
   `--minb` continua no bench para quem quiser repetir.
6. **Atenção com mais splits**: 24/32/48/64 splits medidos em 4K/16K/64K/131K
   f16 — sempre ≥ do que 8–16 splits, exceto no KV q4_0 a 131K (§3.2).
7. **Prefill `N`**: a curva por token é monotônica em N; não há reordenação a
   testar (§3.1).
8. **Varredura de células da atenção sem A/B intercalado**: o mesmo kernel
   medido em 3 células independentes deu 15–20 % de espalhamento (a primeira
   célula depois do preenchimento do cache é sistematicamente mais lenta). As
   "vitórias" de 4–5 % dessa varredura **não** foram aceitas; só o que passou no
   A/B intercalado (§2) foi embarcado. Quem for re-medir atenção deve usar
   `--no-sweep --ab ... --cand ...`.

### 4.1 O que a varredura por classe revelou e ainda não foi explorado

A tabela por (tipo, forma) do `bench-matvec-shapes-gpu` mostra uma classe
patológica: **96 tensores `q8_0` de forma 48×160** (bpr 160) rodando a **25 GB/s**
e custando **1,00 ms/token** (4 % de todo o matvec) — porque 48 linhas dão um
grid de 24 CTAs com `rows=2` numa GPU de 64 CU. A tabela atual é indexada só por
tipo e **não consegue** expressar "esta forma precisa de outra configuração" (a
lição que o M2 já tinha registrado: *"a tabela precisa ser indexada por (tipo,
blocos-por-linha)"*). Fica na §5 como o candidato #1.

---

## 5. Os 3 próximos (estimativas, não medido)

1. **Tabela de shape indexada por (tipo, blocos-por-linha)** — atacar a classe
   `q8_0` 48×160 (25 GB/s, 1,00 ms/token) e as demais formas de grid pequeno.
   *Estimativa: +2 % a +3,5 % no decode* (recuperando metade a tudo daquele
   1,0 ms), bit-exato por construção (`ROWS` é livre; `WPR` exigiria validação).
2. **`iq3_s` no teto de leitura**: é 33 % do tráfego e roda a ~443 GB/s contra
   um teto de leitura medido em 619 GB/s (mesma travessia). Caminho com evidência
   no repo: LDS para a LUT `iq3s_grid` (o Vulkan/RADV faz isso) e/ou a
   formulação fp32 estilo Vulkan. *Estimativa: 8,08 → ~6,5 ms = +5 % no decode*
   (se metade da distância ao teto for recuperada).
3. **Política de atenção dependente de `keys`** (poucos splits + 16 warps a
   4K/16K; 24 splits com 16 warps a 131K q4_0): o ganho está medido no kernel
   (+4,6 % a 4K/16K, +8,7 % a 131K q4_0), mas exige uma regra nova em
   `attn_split_wpb()`/`attn_splits_for()`. *Estimativa: +0,2 % a 4K, +0,5 % a 16K,
   +2,4 % a 131K q4_0* — vale a pena se o alvo for contexto longo.

---

## 6. Task 7 — suíte de regressão numérica (gate de pré-merge)

### 6.1 O que ela é

`tests/check_regression_gpu.hip` roda 7 prompts fixos pelo caminho real do motor
(mesmo `prefill_ids` do CLI, `forward_batch`/`forward_tokens`) e decodifica em
**greedy puro** (argmax, sem penalidades, sem RNG). Compara contra
`tests/golden/regression_greedy_f16.txt`:

* **ids gerados: bit-exatos** (greedy é determinístico: qualquer mudança de
  aritmética que troque um argmax aparece aqui, no token e no índice);
* **logits do último passo**: rel-L2 ≤ **1e-5** numa subamostra de passo 97
  (≥2000 valores por caso) e `|Δ||logits||| / ||logits|| ≤ 1e-5`. A tolerância
  existe para reordenamento legítimo da atenção (M7 mediu rel-L2 ~2,5e-7 nesse
  tipo de mudança) — bit-exato aqui quebraria a cada otimização legítima.

| caso | o que cobre | tokens de prompt | gerados |
|---|---|---|---|
| `short` | prompt curto, argmax trivial | 3 | 32 |
| `long` | prosa em português, ~120 tokens | ~120 | 24 |
| `code` | continuação de código C | ~90 | 32 |
| `cjk` | multilíngue: japonês, chinês, acentos, cedilha | ~90 | 24 |
| `chat` | template de chat do modelo (`chat_render`, thinking off) | ~40 | 24 |
| `ctx4k` | **prefill real de 4096 tokens** (o baseline de decode) | 4096 | 8 |
| `ctx32k` | contexto mais longo que **cabe** com KV f16 (cache sintético semeado, igual ao `bench --start-pos`), onde a atenção com 16 splits roda | ~120 + salto p/ 32768 | 8 |

Comando:

```bash
cmake --build build --target check-regression-gpu
./scripts/check_regression.sh                    # gate: ele MESMO pega o gpu-lock
UPDATE=1 ./scripts/check_regression.sh           # regrava o golden (só depois de validar)
GPU_LOCK_HELD=1 ./scripts/check_regression.sh    # quem já tem o lock (ex. check_all.sh)
```

O script se trava sozinho (`exec scripts/gpu-lock.sh "$0" "$@"` quando
`GPU_LOCK_HELD != 1`) e **não** trava de novo com `GPU_LOCK_HELD=1` — sem isso
quem chama o gate por dentro de outro lock (ou direto, como aconteceu neste lote)
ou fura a fila ou trava para sempre.

**Custo medido: ver §6.4** (o alvo do gate é ≤5 min e a suíte imprime o prefill e o
decode de cada caso, para o número não virar folclore). Roda **depois de cada
mudança**.

#### Por que o caso longo é 32K e não 64K/131K (medido, e vale para o README)

O KV f16 deste modelo custa **64 KB por token** (16 camadas × 4 cabeças kv × 256
dims × 2 B × 2 para K e V). Medido com `bench --prefill 512`:

| ctx | VRAM em uso | prefill |
|---|---|---|
| 4096 | 11,87 GiB | **69,2 tok/s** |
| 32768 | 13,62 GiB | **69,7 tok/s** |
| 65600 | ~15,9 GiB (no limite) | **6,9 tok/s** |

Ou seja: acima de ~32K com KV f16 a alocação fica no limite da placa e o driver
passa a paginar — o prefill cai **10x** (69 → 6,9 tok/s) sem nenhum erro. Não é
o kernel: é VRAM. Por isso o caso longo da suíte fica em 32768 (13,6 GiB, margem
de 2,3 GiB) e o 131K só é possível com KV quantizado (q4_0 = 18 KB/token).

### 6.2 Anti-teste-vazio (a regra que este repo já violou duas vezes)

O gate **falha** se: a referência não existir; a referência tiver menos de 7
casos; um caso gerar menos tokens do que declara; a subamostra de logits tiver
menos de 2000 valores; a referência tiver um número de casos diferente da
execução. O script também recusa um PASS sem a linha de comparação de logits.
Toda comparação imprime o número que comparou (rel-L2, max|d|, Δnorma).

### 6.3 Custo medido e resultado na árvore atual

Saída real (ver §6.4 para o log completo):

```
check-regression-gpu: 7 casos, ctx 32832, kv f16/f16, greedy puro
caso        prompt    gen         ms       rel-L2  d||logits||      ids
short            5     32     1346.7            -            -       32
long           137     24     2689.0            -            -       24
code           145     32     3089.2            -            -       32
cjk            103     24     2233.6            -            -       24
chat            51     24     1495.8            -            -       24
ctx4k         4217      8    60399.6            -            -        8
ctx32k         137      8     2201.8            -            -        8

comparacao com tests/golden/regression_greedy_f16.txt
caso        prompt    gen       rel-L2       max|d|  d||logits||  ids
short            5     32     0.00e+00     0.00e+00     1.41e-09  OK
long           137     24     0.00e+00     0.00e+00     1.34e-11  OK
code           145     32     0.00e+00     0.00e+00     2.65e-09  OK
cjk            103     24     0.00e+00     0.00e+00     1.96e-09  OK
chat            51     24     0.00e+00     0.00e+00     1.90e-09  OK
ctx4k         4217      8     0.00e+00     0.00e+00     5.85e-10  OK
ctx32k         137      8     0.00e+00     0.00e+00     9.33e-11  OK
check-regression-gpu: OK (7 casos, 7 ids bit-exatos, rel-L2 <= 1e-05)
check-regression: check-regression-gpu rc=0, parede 99 s (.../regression_greedy_f16.txt)
```

**Parede: 99 s** (a suíte imprime prefill e decode de cada caso; o caso de 4K
consome 60 s dos 99). O `rel-L2` de 0,00e+00 é o esperado (mesmo binário, mesmo
caminho); o `d||logits||` de ~1e-9 é a precisão com que o golden guarda os
logits, ou seja a comparação está 4 ordens de grandeza dentro da tolerância.

### 6.4 Controle negativo (prova de que o gate falha quando deve falhar)

Método: perturbar **uma** constante de kernel que participa da tolerância —
`include/rdna4/matvec.cuh:37`, `const float d = amax / 127.0f` → `amax / 126.0f`
(o `d` da quantização q8_1 da ativação, ou seja o erro entra em **toda** projeção)
— rodar o script, e reverter. Log completo em `reference/` não: está colado aqui
e reproduzível com os comandos da §8.

Com a perturbação, a suíte **FALHA** (15 falhas, rc=1):

```
check-regression-gpu: FAIL: short: rel-L2 dos logits 5.428e-03 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: short: norma L2 dos logits difere 1.743e-03 > 1e-05
check-regression-gpu: FAIL: long: rel-L2 dos logits 8.920e-03 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: code: rel-L2 dos logits 1.421e-02 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: cjk: rel-L2 dos logits 9.161e-03 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: chat: rel-L2 dos logits 1.030e-02 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: ctx4k: rel-L2 dos logits 1.640e-02 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAIL: ctx32k: id gerado difere no passo 0: 198 contra 1296 na referencia (8 ids comparados)
check-regression-gpu: FAIL: ctx32k: rel-L2 dos logits 8.269e-01 > 1e-05 (2560 valores comparados)
check-regression-gpu: FAILED (15 falha(s))
check-regression: check-regression-gpu rc=1, parede 77 s (.../regression_greedy_f16.txt)
FAIL: suite de regressao FALHOU (rc=1) — saida acima
```

e, depois de reverter a constante (mesmo binário reconstruído, mesmo golden, sem
tocar na referência), volta a **PASSAR** com `rel-L2 0.00e+00` (rc=0, parede 76 s).

Duas leituras que valem registrar:

* a tolerância de logits pegou o erro em **todos** os 7 casos (rel-L2 5e-3 a
  1,6e-2 = 500x a 1600x o limite de 1e-5): é ela que segura a mudança pequena;
* os **ids** só viraram em 1 dos 7 casos (o de contexto longo com cache sintético,
  onde a atenção cancela muito e o erro relativo é 8e-1 — o mesmo fenômeno que
  `docs/medicoes-m7.md` descreve). Ou seja: o gate de ids é o que pega mudança
  semântica, o de logits é o que pega deriva numérica — os dois são necessários, e
  é por isso que a tolerância não é bit-exata.

---

## 7. Gate de pré-merge (rodar nesta ordem antes de qualquer merge)

### 7.1 Regra de janela (aprendida na marra neste lote)

Neste lote **dois gates ficaram vermelhos por contenção, não por código**: um
`rdna4-infer ppl` rodou fora do `flock` (minha culpa, 00:14) e o
`check_golden_run.sh`/`check_attn_split.sh` do agente vizinho falharam com
`hipMalloc failed` disfarçado de "engine exited non-zero" / PPL vazio. No meu
worktree esses dois scripts ainda são a versão **antiga** (não se travam
sozinhos — o conserto está na `main`), então a regra aqui é:

```bash
# SEMPRE com o lock, e conferindo a janela antes (VRAM do amdgpu em sysfs):
cat /sys/class/drm/card1/device/mem_info_vram_used     # < ~2 GiB = janela livre
scripts/gpu-lock.sh ./scripts/check_golden_run.sh
scripts/gpu-lock.sh ./scripts/check_attn_split.sh
scripts/gpu-lock.sh ./scripts/check_regression.sh      # (este já se trava sozinho)
```

Um gate vermelho por contenção é pior do que um gate não rodado: ele vira
conclusão errada no relatório. Todos os números deste documento foram medidos
com o lock; os dois gates citados foram **re-rodados em janela verificada**
(§7.2) e o resultado de cada um está no relatório do agente com a marca de
"com lock / janela limpa".

```bash
# 0. tabela de tuning: a configuração embarcada é a medida (CPU, instantâneo)
./build/check-tuning tests/golden/ml_tuning.txt

# 1. regressão numérica (7 prompts, ~2 min, GPU)
./scripts/check_regression.sh

# 2. gates de kernel (GPU)
GRAPH_LAST_TOKEN=1 ./scripts/gpu-lock.sh ./build/check-graph-gpu <model> <dump?> -
./scripts/gpu-lock.sh ./build/check-matvec-gpu  <model>
./scripts/gpu-lock.sh ./build/check-dequant-gpu <model>
./scripts/gpu-lock.sh ./build/check-batch-gpu   <model>
./scripts/gpu-lock.sh ./build/check-kvctx-gpu   <model>
./scripts/check_golden_run.sh
./scripts/check_attn_split.sh
```

(Os comandos exatos de cada um estão em `scripts/`; o que importa é que a lista
inteira passa na árvore commitada.)

### 7.2 Resultado na árvore commitada — cada gate com lock e janela verificada

Todos rodados com `scripts/gpu-lock.sh` e com a janela conferida antes
(`mem_info_vram_used` < 200 MiB e nenhum processo de motor vivo — ver §7.1):

| gate | resultado | parede | janela |
|---|---|---|---|
| `build/check-tuning` (CPU) | OK — 22 linhas comparadas, 22 chaves obrigatórias | <1 s | não usa GPU |
| `scripts/check_regression.sh` (auto-lock) | OK — 7 casos, ids bit-exatos, rel-L2 0 | 79 s | limpa (166 MiB / 0 proc) |
| `check_regression.sh` com `GPU_LOCK_HELD=1` + lock externo | OK | 83 s | limpa |
| `GRAPH_LAST_TOKEN=1 check-graph-gpu <model> - reference/argmax_fib_cpu.txt 727 73111 1393 1590` | **PASS** — argmax 198=198, top-5 **5/5** | 3,7 s | limpa |
| `check-matvec-gpu` | OK — 14 tipos | 0,19 s | limpa |
| `check-dequant-gpu` | OK — 14 tipos | 0,15 s | limpa |
| `check-batch-gpu` | OK — prefill em batch bit-idêntico | 7,3 s | limpa |
| `check-kvctx-gpu` | OK — splits(16) vs sem split a 64K, argmax igual | 11,1 s | limpa |
| `scripts/check_golden_run.sh` | **OK** | 40,3 s | limpa (164 MiB / 0 proc) |
| `scripts/check_attn_split.sh` | **OK** — PPL 5,1989 vs 5,2054 = **0,125 %** (limite 0,5 %) | 222 s | limpa (164 MiB / 0 proc) |

Nota honesta: `check_golden_run.sh` e `check_attn_split.sh` apareceram
**vermelhos duas vezes por contenção** antes disso (uma vez sem lock — erro meu,
um `ppl` às 00:14 — e uma vez com lock, contaminados pelo resíduo do processo
anterior). Os dois foram re-rodados em janela verificada e estão verdes; nenhum
número deste documento foi medido fora do lock.

Sobre `check-graph-gpu` com **dump** (`... <dump.txt> <argmax>`): essa forma
continua vermelha de forma **pré-existente** (o dump `reference/oracle_*_cpu.txt`
não corresponde ao estado atual do grafo — já registrado em
`docs/rocm-estudo.md` §F.1, que mostra as mesmas falhas no HEAD limpo). O critério
de aceite do PLAN (`argmax igual ao do llama.cpp`, com `-` no lugar do dump) passa
5/5 no top-5.

---

## 8. Como re-medir (quando a placa, a ROCm ou um kernel mudar)

```bash
source scripts/rocm-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j12

# matvec: inventário real, 6 rodadas rotacionadas, com piso de ruído embutido
MODEL=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MODEL \
    --rows 2,4,8 --ilp 1,2,4 --unroll 2 --repeat-ship --reps 6

# atenção: varredura (orientação) e A/B intercalado (decisão)
scripts/gpu-lock.sh ./build/bench-attn-gpu f16 4096 16384 65536 \
    --smax 32 --splits 1,2,3,4,6,8,12,16,24,32 --wpb 8,16,32
scripts/gpu-lock.sh ./build/bench-attn-gpu q4_0 131072 --no-sweep --ab-rounds 15 \
    --ab 16:8 --cand 16:16 --cand 24:16 --cand 16:8

# regressão (e, se a mudança for numérica e validada, regravar as referências)
./scripts/check_regression.sh
./build/check-tuning --record tests/golden/ml_tuning.txt
```

Depois de mexer em `tuning.h`: `./build/check-tuning --record tests/golden/ml_tuning.txt`
**no mesmo commit**, com o número medido escrito neste documento.

---

## 9. H0 — Portões, skews e âncoras re-based (auditoria 2026-09-16, só leitura)

H0 do `docs/plano-hipfire-gfx1201.md` §4 (higiene de gates). Método: grep por
builtins WMMA/f16 (`wmma_*f16*`, `__builtin_amdgcn_wmma*`) e por predicados de
dispatch (`is_gfx1201`, `__gfx1201__`, `RD_*`) em `include/` + `src/` + `tests/`,
cada hit classificado em usado-com-teste / usado-sem-teste / morto / bench-only;
três skews resolvidos por leitura (linhas do plano podem estar defasadas —
todas foram re-verificadas abaixo). Nenhum teste de GPU foi rodado nesta
auditoria (sem lock); `check-tuning` (CPU puro) foi re-rodado: **OK, 23 linhas**.

### 9.1 Âncoras re-based (os números do plano §2 são pré-WMMA e estão stale)

| workload | âncora vigente | fonte |
|---|---|---|
| decode short-ctx (pos 5..21) | **37,9 tok/s** best (36,3 mean) | `docs/baseline-2026-09-16.md:43` (37,88), confirmado 37,94 em `docs/chunk-scale-2026-09-16.md:40` |
| decode @4K end (pos 4090..4096, 6 tokens) | **36,3 tok/s** best (33,4 mean) | `docs/baseline-2026-09-16.md:43` (36,33) |
| prefill-512 | **451,1 tok/s** best | HEAD `4013c0c` (WMMA-int8 iq3_s n≥64: 408,5 → 451,1, +10,4 %); pré-WMMA era 411,22 @cap 512 (`chunk-scale:38`), 355,81 @cap 128 (`baseline:40`) |
| default vigente | `RD_PREFILL_CHUNK` = **512** | `include/rdna4/device.h:212-257` (fonte única; `a176fd1` flipou 128 → 512) |

### 9.2 Tabela de portões (nome, arquivo:linha, estado, teste que cobre)

Estados: **T** = usado-com-teste · **B** = bench-only / rejeitado-com-dado (vivo,
fora de produção) · **M** = MORTO (achado) · **A** = ausente (nada a auditar).

| # | portão | arquivo:linha | estado | teste que cobre |
|---|---|---|---|---|
| G1 | WMMA-int8 iq3_s (`wmma_i8`, `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`) | `include/rdna4/gemm.cuh:1530-1531` (kernel `:1544`, launch `:1690`) | **T** — SHIPPED, não é achado | `check-batch-gpu` linha `iq3_s-wmma` M=64 (`tests/check_batch_gpu.hip:120`) + faixas n>16 fim-a-fim + `RD_GEMM_WMMA=0` A/B |
| G2 | porta WMMA `n >= 64` (`gemm_use_wmma_iq3s`) + espelhos | `gemm.cuh:1702-1708`, despacho `:1748-1758`, attrs `:1810-1813` | **T** (fonte única, sem cópia) | `check-batch-gpu` + `bench-gemm-engine-gpu` |
| G3 | dp4a `__builtin_amdgcn_sudot4` + select `__gfx1201__` | `include/rdna4/vecdotq.cuh:86-88` | **T** | `check-matvec-gpu` + `check-dequant-gpu` (14 tipos) |
| G4 | gather LUT `__builtin_amdgcn_perm` | `vecdotq.cuh:148-157`, fileiras iq3 `:926ff` | **T** | mesmos do G3 |
| G5 | fail-closed `is_gfx1201` | `include/rdna4/device.h:34`; `src/main.hip:292,683,905,1169`; `src/server/serve.hip:364` | **T** (definição única, sem cópia) | build `--offload-arch=gfx1201` + `scripts/check_target.sh` |
| G6 | regra WPB duas-pontas (`attn_split_wpb`) | `include/rdna4/attn.cuh:943-949`, consts `tuning.h:147-148` | **T** | `bench-attn-gpu` A/B 15 rounds + `scripts/check_attn_split.sh` + `check-kvctx-gpu` |
| G7 | N-POL (`attn_splits_for`) + escape `RD_ATTN_SPLITS` | `include/rdna4/graph.cuh:488-519` | **T** | `check-kvctx-gpu` (split vs sem split @64K) + `check_attn_split.sh` (PPL 0,125 % vs gate 0,5 %) |
| G8 | faixa exata GEMV-batch N∈{2,3,4,8,16} | `matvec.cuh:1025-1030`; `Graph::batch_supported` `graph.cuh:203-205`; `chunk_ok` `:210-212` | **T**, mas ver M2 (predicado copiado) | `check-batch-gpu` (bit-exato n≤16, `:469`) + `check-tuning` (`batch.ns/cap`) |
| G9 | despacho GEMV×GEMM n>16 (`proj_batch`) | `graph.cuh:1367-1396` | **T** | `check-batch-gpu` faixas toleradas (`kTolChunkRelL2/kTolChunkMaxAbs`) + PPL/golden |
| G10 | teto runtime do chunk (`prefill_chunk_cap`, default 512) | `device.h:212-257`; `kMaxChunkHost` `graph.cuh:200` | **T** | `check-batch-gpu` (cap-aware, `:449-473`) + `docs/chunk-scale-2026-09-16.md` |
| G11 | `RD_ATTN_SPLIT_BATCH` (fallback per-token) | `graph.cuh:172-178` | **T** (hatch de A/B) | `check-batch-gpu` com `=0` (`docs/journal-prefill.md` §9) |
| G12 | `RD_PREFILL_BATCH` (andaime por token) | `graph.cuh:260-267` | **T** (hatch de A/B) | A/B idem |
| G13 | `RD_GDN_FUSED` / `RD_GDN_RESIDENT` | `gdn.cuh:714-719`; `graph.cuh:1627-1631` | **T** (default coberto; hatch manual) | `check-batch-gpu` + `check-graph-gpu` (caminho default) |
| G14 | `RD_MTP_*` (concat + governor) | `mtp.cuh:362`; `mtp_gen.h:213-224` | **T** | `check-mtp-gpu` |
| G15 | N=32/64 + `matvec_batch_cap()==64` | `matvec.cuh:913`, `:1034-1035` | **B** (faixa D1, fora de produção: `proj_batch` nunca passa n>16 ao GEMV-batch) | `bench-mmq-wmma-gpu` (`:1036-1043`), `bench-gemm-engine-gpu` (`:750-756`, `:1800-1802`), bound em `check-matmul-gpu` (`:208`) |
| G16 | prefetch `PF` (`rdna4_prefetch_l2`) | `matvec.cuh:288-294` (produção sempre `PF=false`) | **B** (rejeitado 0,984x, §4.4) | `bench-matvec-shapes-gpu` |
| G17 | WMMA-f16 (`_w32_gfx12` f16/f16-acc) | `tests/bench_wmma_gpu.hip:59-60` | **B** (pesquisa frente D; **zero uso em produção**) | `bench-wmma-gpu` (não é gate) |
| G18 | protótipos MMQ-WMMA | `tests/bench_mmq_wmma_gpu.hip` §§1-8 | **B** (sucedidos pelo port de produção `gemm.cuh:1497+`) | bench (não é gate) |
| G19 | knob WPB do caminho *unsplit* | `attn.cuh:94` (parâmetro), corpo ignora em `:121,165,171,178` + gêmeo batched `:348,388,394,401`; launchers `:201-249` | **M** — ver skew (b) | nenhum (só o bench consome, `:871` — e mede warps redundantes) |
| G20 | `kBatchCap`/`kBatchNs` | `tuning.h:161-163` (único consumidor: `check_tuning.cpp:81` imprime) | **M-parcial** — predicado real é a cópia hardcoded G8; a constante é só ecoada | `check-tuning` (eco, não comportamento) — ver skew (a) |
| G21 | DPP/`permlanex16` no GDN (classe do bug hipfire #757) | `gdn.cuh` (grep: zero ocorrências) | **A** — alvo da auditoria H2 não existe nesta árvore | n/a |

**Contagem de portões mortos: 2** (G19, G20). Nenhum builtin f16/WMMA não-confirmado
em produção: o único `__builtin_amdgcn_wmma*` produtivo é o G1 (int8, gateado).

### 9.3 Vereditos dos três skews

**(a) `kBatchCap=16` vs `matvec_batch_cap()==64` + D1 + `n_tokens <= cap` — sem
contradição, três tetos distintos.** (i) `kBatchCap=16`/`kBatchNs` = contrato de
bit-exatidão do GEMV-batch (n≤16, `graph.cuh:1367-1372`); (ii)
`matvec_batch_cap()==64` (`matvec.cuh:913`) = teto das *instanciações* do kernel,
incluindo N=32/64 bench-only do degrau D1 (`:1031-1035`); (iii)
`batch_max_`/`prefill_chunk_cap()` (default 512) = teto *runtime* do chunk
(`chunk_ok`, `graph.cuh:210-212`). Contrato real, verificado no despacho:
`chunk_ok(n) = 2≤n≤batch_max_ ∧ (n≤16 → batch_supported(n))`, e `proj_batch`
nunca passa n>16 ao `matvec_launch_batch` (n>16 → `gemm_launch` ou sub-lotes
≤16 + cauda de 1, `:1374-1396`). Fica o G20: `batch_supported()` recopia a
lista em vez de ler `tuning.h` (já apontado em `docs/journal-review.md:75`) —
proposta de assert P1 na mensagem ao coordenador.

**(b) `attn_kernel` ignora WPB (limitação 6 do README) — confirmado, 8 sítios;
recomendação: REMOVER, não plumbar.** Corpo usa `kAttnWarpsPerBlock` em
`attn.cuh:121,165,171,178` (unsplit) e `:348,388,394,401` (batched unsplit);
produção está correta *por coincidência* (sempre instancia o default 8 — nenhum
chamador produtivo de `attn_launch_wpb`; único consumidor é o bench,
`bench_attn_gpu.hip:871`, cujas células unsplit `--wpb` medem warps redundantes
que o merge descarta). Caminho split (o que embarca) usa WPB corretamente
(`:520-521`, `:565ff`, `:814ff`) e não é afetado. Motivo de remover em vez de
plumbar: unsplit = chaves<512 (contexto curto), nunca foi alavanca de tuning
(toda a evidência §3.2 é do split); plumbar criaria matriz 3×36 de
instanciações sem headroom medido. Converge com `docs/adversarial-noite.md:129`
e `docs/auditoria-qualidade.md:615`. Proposta P3 (assert + remoção exata) na
mensagem ao coordenador; `README.md:531-534` já documenta até o landing.

**(c) `kv-memoria-desenho.md` §7 — lado do doc correto nos 5 itens.**
1. MTP 351 MB vs 42,7 MB: doc correto (`§4.1` + `device.h:58-60` concordam em
351 MB / 0,327 GiB); `rocm-estudo.md` §A.2.6 stale. Sem ação de código.
2. 17-vs-16 camadas: doc correto (16); **comentário obsoleto ainda vivo** em
`device.h:36` ("17 full-attention layers") ao lado da correção (`:40-42`) e da
constante certa (`:43`). Proposta P4 (troca literal do comentário).
3. `medicoes-m5.md` 262 ms@64K: stale pré-M7; correto hoje ≈53 ms/token (atenção
19 ms, `medicoes-m7.md:107`). Requer nota de revisão no M5 — fora do escopo de
edição H0 (só este arquivo), fica como proposta ao coordenador.
4. "32 warps/CTA": **não está mais no README** (reescrito; limitação 6 cobre o
skew) — o texto vive em `SPEC.md:74` ("cargas vetorizadas, 32 warps por CTA"),
fora do escopo de edição H0. Proposta ao coordenador.
5. splits `keys/2048` vs `keys/512`: código correto hoje (`keys/512`,
`graph.cuh:487`, M8 +14,7%@4K); texto do M7 historicamente correto na sua
janela. Sem ação; contrato atual em `tuning.h:107-109` + §3.2.

Nota de higiene (não-achado): `tests/check_tuning.cpp:33` diz "22 linhas" mas o
gate compara 23 (entrou `attn.split_ctas_dense` depois); comentário cosmético,
sem efeito no gate (que passou verde aqui).

---

## 10. Arquivos tocados (para o coordenador)

| arquivo | dono | o que mudou |
|---|---|---|
| `include/rdna4/tuning.h` | **novo** (este trabalho) | a tabela única dos parâmetros medidos (matvec por tipo, atenção, batch) |
| `include/rdna4/matvec.cuh` | branch do estudo ROCm (já merged) | **só as tabelas**: `MtShape`/`MtIlp` passam a ler `tuning.h`; nova `MtUnroll`; `matvec_launch` passa `UNROLL`; helper novo `matvec_default_unroll()` (diagnóstico) |
| `include/rdna4/attn.cuh` | idem | `kAttnWarpsPerBlock`/`kAttnSplitWpbLimit` leem `tuning.h`; nova constante `kAttnSplitWpbWide` + 2 linhas em `attn_split_wpb()` (regra das duas pontas, medida) |
| `include/rdna4/graph.cuh` | idem | **só** `kAttnSplitMin`/`kAttnMaxSplits` passam a ler `tuning.h` (nomes, política e estrutura intactos) |
| `tests/bench_matvec_shapes_gpu.hip` | idem | variantes novas (`--ilp`, `--repeat-ship`, `--land`), matriz por tipo e piso de ruído do próprio bench |
| `tests/bench_attn_gpu.hip` | idem | varredura rotacionada, células de ruído, e o modo `--ab/--cand/--no-sweep` (A/B intercalado) |
| `tests/check_tuning.cpp`, `tests/golden/ml_tuning.txt` | **novos** | gate da tabela (CPU) |
| `tests/check_regression_gpu.hip`, `scripts/check_regression.sh`, `tests/golden/regression_greedy_f16.txt` | **novos** | suíte de regressão + gate |
| `CMakeLists.txt` | compartilhado | **append** no fim: `check-tuning` e `check-regression-gpu` |
