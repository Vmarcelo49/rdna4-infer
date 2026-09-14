# Diário da frente prefill — `feat/noite-prefill`

Worktree `../rdna4-wt-noite-prefill`, base `main` = tag `noite-baseline-2026-09-14` (`e48f3c4`).
Regras: `docs/noite-regras.md`. Toda medição passa por `scripts/gpu-lock.sh` e diz se a janela
estava limpa (`mem_info_vram_used`, `fuser /dev/kfd`).

Alvo da frente: **prefill (tok/s do prompt)**. Ponto de partida: 72,9 tok/s em 512 tokens
(7,02 s até o primeiro token) contra 1143 tok/s do llama.cpp `pp512` — 15×.

Ferramenta oficial de profiling (`rocprof`/`rocprofv3`/`omniperf`) **não existe nesta
instalação** (ver `docs/medicoes-banda-e-gargalos.md` §0): tudo aqui é evento HIP, contagem de
lançamentos e subtração A/B.

---

## 0. Instrumento novo: `bench-phases-gpu --prefill N`

- **Referência**: `tests/bench_phases_gpu.hip` (frente de medições) media só DECODE
  (`--ctx/--pos/--tokens`); o prefill (`Graph::forward_batch`) não tinha orçamento por fase.
- **Hipótese**: o prefill é dominado pelo andaime que continua **por token** dentro do lote
  (atenção, recorrência GDN, normas), não pelo matvec — o matvec em lote já lê cada peso uma vez
  para os 16 tokens (`docs/medicoes-m8.md`), então o que sobra por token é o resto do token de
  decode (~11 ms medidos a 4K, `docs/medicoes-banda-e-gargalos.md` §1.2).
- **Comando**: `--prefill N` roda uma passada de prefill em lote com a MESMA política de chunks
  de `src/main.hip::prefill_ids` (16/8/4/3/2 e cauda de 1), cronometra **cada chunk** e arma o
  mesmo profiler de fases opt-in do decode; `--count-only` conta lançamentos sem eventos.
- **Veredito**: MANTIDO (instrumento). É o que permitiu escolher o que atacar; números em §2.

---

## 1. Curva de escala do prefill (baseline, ANTES de qualquer mudança de kernel)

- **Referência**: pedido do briefing ("meça a curva primeiro").
- **Hipótese**: se o prefill é dominado por custo **por token**, o tok/s é ~constante em N (só a
  atenção cresce com o contexto, e ela é 3,8 % de um token a 4K).
- **Comando** (janela limpa: `mem_info_vram_used` = 198 MB antes, nenhum processo em
  `/dev/kfd`; um único `gpu-lock.sh`, `timeout 1800`, série inteira):
  `./build/rdna4-infer bench -m IQ3_S -p "The capital of France is" --prefill N --prefill-reps 2
   -n 1 --reps 1 --warmup 0` para N = 64/256/512/2048/4096. `--prefill-reps` é opção nova desta
  frente (cronometra o prefill K vezes e reporta a melhor passada, como o decode já fazia).
- **Resultado** (as duas passadas de cada configuração, na ordem; VRAM durante a execução ≲ 1,6 GiB):

  | N | passada 1 | passada 2 | ms/token (melhor) | vs N=64 |
  |---|---|---|---|---|
  | 64 | 73,01 tok/s (0,877 s) | **75,42** (0,849 s) | 13,26 | — |
  | 256 | 73,92 (3,463 s) | **74,17** (3,452 s) | 13,48 | +1,7 % |
  | 512 | 72,98 (7,016 s) | **73,05** (7,009 s) | 13,69 | +3,2 % |
  | 2048 | 71,56 (28,618 s) | **72,41** (28,284 s) | 13,81 | +4,1 % |
  | 4096 | 71,82 (57,030 s) | **71,82** (57,028 s) | 13,92 | +5,0 % |

  Piso de ruído do harness: as duas passadas da mesma configuração ficam dentro de **1,2 %**
  (0,3 % em 512; em 4096 as duas passadas deram 57,030 e 57,028 s — 0,004 %); a reprodutibilidade
  do número de 512 contra o briefing (72,9 tok/s) é 0,2 %.
- **Leitura**: o prefill é **plano em N** (75,4 → 71,8 tok/s de 64 a 4096 tokens). O custo por
  token praticamente não depende do tamanho do lote, o que só acontece se o custo **por token**
  (e não por byte de peso) domina. Confirma a hipótese: ~11 ms dos ~13,7 ms de cada token de
  prefill são o andaime que ainda roda uma vez por token.
- **Veredito**: medido, nada mudado. É a referência de tudo abaixo.

### 1.1 Orçamento por fase do prefill — o instrumento não estava armado

- **Referência**: hook `RD_PHASE` de `include/rdna4/phase_prof.cuh`.
- **Hipótese**: `bench-phases-gpu --prefill N` daria a tabela por fase do prefill.
- **Comando**: `./build/bench-phases-gpu IQ3_S --prefill 64 --level 2` (janela limpa).
- **Resultado**: **0,1 marca/token** — as fases apareceram zeradas (`matvec` 9,916 ms/token era o
  único bucket, com 0,0 marcas). Causa: o hook `RD_PHASE` só existe no caminho **por token**
  (`forward_run`) e em `proj`/`proj_qq`; `forward_batch_layer` não tinha nenhuma marca, e o
  caminho em lote usa `proj_batch`/`quantize_batch`, que também não tinham.
- **Veredito**: MANTIDO (falha do instrumento, corrigida em §3): marcas de fase adicionadas ao
  caminho em lote, com os MESMOS nomes de bucket do decode, para as duas tabelas serem
  comparáveis. `RD_PHASE` é inerte sem profiler armado (nenhum número do motor muda).

---

## 2. Andaime por token dentro do lote → kernels em lote (a mudança principal)

### 2.1 O que foi feito

Cinco kernels/launchers novos, todos com a mesma invariante: **cada token mantém a aritmética que
tem no caminho por token**, então o resultado continua bit-idêntico e `check-batch-gpu` continua
sendo um gate de verdade.

| peça | antes (dentro de `forward_batch_layer`) | agora |
|---|---|---|
| atenção: split q/portão, QK-norm, RoPE, escrita de KV, portão | 16 lançamentos **por token** (13 por token de atenção + 3) | 1 lançamento por estágio para o chunk (`attn.cuh` novo kernel em lote; `rope_kernel` já era multi-token; `kv_write_batch` = 1 chamada do kernel por linha com a largura do chunk) |
| atenção em si | `attn_launch`/`attn_launch_split` por token | 1 lançamento para o chunk quando `splits == 1` (`attn_batch_kernel`, corpo idêntico ao não-dividido); contexto longo (> 512 chaves) mantém o laço por token, porque a ordem de soma dos splits faz parte dos números gravados |
| GDN: escalares (sigmoid/add/softplus/mul) | 4 lançamentos por token | 4 lançamentos para o chunk (`add_bcast`/`mul_bcast`: `y[i] = a[i] op b[i % nb]`) |
| GDN: `conv1d` | 1 por token | 1 para o chunk, andando os tokens em ordem dentro do kernel (`conv1d_state_batch_kernel`) |
| GDN: `l2_norm` q/k | 2 por token | 1 para o chunk (linhas contíguas) |
| GDN: `delta_rule` | 1 por token (48 CTAs, 83 µs/camada) | 1 para o chunk (`delta_rule_batch_rows_kernel`: 192 CTAs de 32 threads, laço de tokens dentro; 32 linhas = 16 KiB de estado por CTA, que é o que mantém a linha quente entre tokens) |
| GDN: `rms_norm`+`silu`+`mul`+cópia D2D | 3 lançamentos + 1 `hipMemcpy` por token | 3 lançamentos para o chunk, sem cópia (o `delta_rule` escreve direto no buffer que a projeção lê) |

Interruptor de A/B **no binário**: `RD_PREFILL_BATCH=0` volta ao andaime por token sem rebuild
(mesma janela, mesmo estado de DPM) — é como o A/B abaixo foi medido.

### 2.2 Gates

- **Comando**: `./scripts/gpu-lock.sh bash` (série com um único lock, janela limpa: 198 MB antes,
  nenhum processo em `/dev/kfd`).
- `check-batch-gpu`: **BIT-EXACT em N = 2/3/4/8/16 e no prompt completo** (rel-L2 0, max|d| 0) —
  e o ganho do lote subiu de **2,18×/3,47×** (N=2/N=16) para **3,47×** em N=16 contra o
  per-token: 32,860 → **9,481 ms/token** (o caminho por token medido no mesmo binário).
- `check-graph-gpu` com `GRAPH_LAST_TOKEN=1`: **PASS** (por token; a mudança não toca esse caminho).

### 2.3 A/B e curva depois

(ver §2.4 — preenchido com a corrida `/tmp/ab-prefill.log`)

---
