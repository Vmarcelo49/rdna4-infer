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

## 3. Orçamento por fase do prefill em lote (DEPOIS do §2) — o matvec é 70 %

- **Comando**: `./build/bench-phases-gpu IQ3_S --prefill 16|64 --level 2` e `--count-only`
  (janela: VRAM em repouso 198 MB antes; durante a série houve atividade de outra frente em
  parte das corridas — anotado; as duas passadas de cada configuração concordam dentro de
  0,5 %, então o número é utilizável).
- **Resultado** (N=16, passada limpa 9,505 ms/token = 105,2 tok/s; instrumentada 9,957 ms/token,
  inflação +4,8 % com 93,1 marcas/token; um `hipEventRecord` custa 4,04 µs nesta medição):

  | bucket | ms/token | % | marcas/token | o que é |
  |---|---|---|---|---|
  | **matvec** | **6,926** | **70,4 %** | 31,0 | 496 matvecs em lote por chunk de 16 + o head |
  | **gdn_delta** | **2,186** | **22,2 %** | 3,0 | `delta_rule_batch` nas 48 camadas GDN |
  | act_quant | 0,200 | 2,0 % | 20,1 | 322 quantizações em lote por chunk |
  | ffn_residual | 0,089 | 0,9 % | 4,0 | |
  | post_norm | 0,084 | 0,9 % | 4,0 | |
  | gdn_l2norm | 0,070 | 0,7 % | 3,0 | |
  | gdn_scalars / norm_silu / conv | 0,052 / 0,052 / 0,051 | 1,5 % | 3,0 cada | |
  | qk_norm_rope_kv / attention / attn_gate_out | 0,032 / 0,030 / 0,012 | 0,8 % | 1,0 cada | |
  | ffn_gate_up / ffn_down / gdn_proj / gdn_out_proj / qkv_proj / attn_out_proj | ≤ 0,017 cada | 0,7 % | | |

  Contagem de lançamentos: **93,1 marcas/token = 1490 por chunk de 16** (era 13 168 por chunk
  antes, ou 823/token). O andaime deixou de ser "muitos lançamentos curtos" e virou
  "dois kernels grandes".
- **Leitura** (a parte que muda o plano da noite): o gargalo do prefill **não é mais o andaime**,
  é o **matvec em lote**. 496 matvecs de lote por chunk custam 110 ms para ler 12,0 GB de pesos
  ⇒ **109 GB/s efetivos**, contra **446 GB/s** do mesmo matvec no caminho por token (24,94 ms
  para 11,13 GB). Ou seja: o kernel em lote amortiza a LEITURA do peso entre 16 tokens (ganho
  medido 3,6× por token), mas o **trabalho de ALU por byte multiplica por 16** e ele fica
  limitado por issue, não por banda — 4× abaixo da eficiência que o mesmo código tem no decode.
  Consequência para a noite: o teto do prefill com este matvec é ~105 tok/s; para passar disso é
  preciso o caminho tiled/MMQ (int8/WMMA), que **não** é desta frente (`matvec.cuh` pertence à
  frente de kernels) — este parágrafo é o pedido de encaminhamento.
- **Veredito**: medido. O bucket `gdn_delta` (22 %) é o que sobra na minha mão e é o §4.

---

## 4. `delta_rule` em lote: linha do estado com float4

- **Referência**: `include/rdna4/gdn.cuh` (estado transposto: `M[j*S+i]`, uma thread por linha `j`)
  e a medida de §3 (`gdn_delta` = 2,19 ms/token = 22 % do prefill).
- **Hipótese**: thread `j` lê `row[i]` e as 32 threads da warp leem endereços a **512 B** de
  distância: cada acesso de 4 B puxa um setor de 32 B ⇒ **8× de amplificação**, e a linha é
  varrida 2× por token (leitura+escrita em cada passada). Quatro elementos por acesso (float4)
  derrubam a amplificação para 2× e o número de instruções do laço serial por 4, **sem tocar na
  aritmética**: as quatro FMAs de um vetor continuam saindo em ordem crescente de `i`, então o
  arredondamento da soma é o mesmo. Ganho esperado: o bucket de 2,19 ms/token cai para ~0,6-1,1.
- **Comando**: `./build/check-batch-gpu` + `./build/rdna4-infer bench --prefill 512
  --prefill-reps 3` + `bench-phases-gpu --prefill 16 --level 2` (janela: ver §4.1).
- **Resultado**: ver §4.1 (preenchido com `/tmp/vec-check.log`).
- **Veredito**: ver §4.1.

### 4.1 Números

- **Resultado** (janela limpa: VRAM em repouso 124 MB antes, 198 MB depois; uma única tomada de
  lock; `bench --prefill 512 --prefill-reps 3`, duas rodadas):

  | | 512 tokens | tok/s |
  |---|---|---|
  | antes (só §2) | 4,898 / 4,915 s | 104,5 / 104,2 |
  | depois (float4) | **4,131 / 4,155 s** | **123,9 / 123,2** |

  ⇒ **+18,6 %** (ganho muito acima do piso de ruído de 1,2 % do harness; neste caso as três
  passadas de uma rodada concordam dentro de 0,2 %).
- `check-batch-gpu`: **BIT-EXACT** em N = 2/3/4/8/16 e no prompt completo; o ganho do lote em
  N=16 sobe de 3,47× para **4,11×** (32,834 → 7,980 ms/token), que é o `cf` que a frente de MTP
  usa para escolher D:

  | N | per-token ms | em lote ms/token | ganho |
  |---|---|---|---|
  | 2 | 39,474 | 16,373 | 2,41× |
  | 3 | 33,062 | 12,203 | 2,71× |
  | 4 | 33,036 | 10,752 | 3,07× |
  | 8 | 32,958 | 9,396 | 3,51× |
  | 16 | 32,834 | 7,980 | **4,11×** |

- **Veredito**: **MANTIDO** (+18,6 % no prefill de 512 tokens, bit-exato).
- Observação de método: a tabela de fases de §3 foi medida com o binário ANTERIOR (eu reconstruí
  `rdna4-infer` e `check-batch-gpu`, mas não `bench-phases-gpu`, antes daquela corrida) — o bucket
  `gdn_delta` daquela tabela (2,19 ms/token) é o valor ANTES do float4; a tabela reconstruída
  está em §5.

---

## 5. Orçamento por fase DEPOIS do float4 (binário reconstruído)

- **Comando**: `./build/bench-phases-gpu IQ3_S --prefill 16 --level 2` (com `bench-phases-gpu`
  reconstruído — a corrida de §3 usou o binário anterior e por isso mostra `gdn_delta` = 2,19).
- **Resultado**: ver `/tmp/gates3.log` + `/tmp/speed-prefill.log` (preenchido abaixo).
- **Veredito**: —

## 6. Chunking e staging (prioridade 3 do briefing)

- **`prefill_ids`**: a política 16/8/4/3/2 + cauda de 1 **continua correta e não muda** depois
  dos kernels em lote. A razão agora é medida, não suposta: `check-batch-gpu` dá o custo de um
  chunk de N tokens — N=16 → 7,98 ms/token, N=8 → 9,40, N=4 → 10,75, N=3 → 12,20, N=2 → 16,37
  (o custo por token é monótono em N ⇒ maior-primeiro continua ótimo), e um chunk de 2 tokens
  (32,7 ms no total) custa o mesmo que **um** token do caminho por token (32,8 ms): a cauda de 1
  não é um caso patológico, é o preço de não ter instanciação para N=1.
- **Tail de verdade** (N=500 = 31 chunks de 16 + 1 de 4): medido em §5; o esperado é um custo por
  token ~2 % maior que em N=512, não mais que isso.
- **GPU ociosa entre chunks?** Não, e a prova é interna ao instrumento: na passada instrumentada
  a **soma dos buckets** (157,5 ms) coincide com o tempo de parede da passada (159,3 ms) dentro de
  1,1 %, e o único trecho fora dos buckets é o dreno depois da última marca. Com ~1490 lançamentos
  por chunk a 2,2 µs = 3,3 ms de fila de host contra 129 ms de GPU por chunk, o host está 40×
  adiantado: não há starvation (o que havia antes era trabalho de kernel pequeno demais, não
  fila vazia).
- **Prefill vs decode trocando cache**: as passadas de prefill medidas antes e depois de um token
  de decode (o `bench` re-prefixa na volta de cada rep) concordam dentro de 1 % (7,009/7,016 s no
  baseline; 4,131/4,140 s depois) — não há evidência de thrashing. Medição direta de L2/DRAM
  **não foi feita**: os contadores de hardware não existem nesta instalação
  (`rocprof`/`omniperf` ausentes, `docs/medicoes-banda-e-gargalos.md` §0) e a frente não tinha
  orçamento de GPU para montar um instrumento equivalente com eventos.

## 7. O teto: o matvec em lote é 85 % do prefill — ENCAMINHADO, não implementado

- **Referência**: `include/rdna4/matvec.cuh:302-345` (`matvec_kernel_batch`) e o orçamento por
  fase de §3/§5.
- **Hipótese (do briefing)**: "o matvec não é o gargalo do prefill".
- **Medida que a mata**: o matvec em lote lê 12,0 GB por chunk de 16 tokens (11,13 GB de tronco +
  0,87 GB de `output.weight`) em 110 ms ⇒ **109 GB/s**. O MESMO matvec no caminho por token faz
  11,13 GB em 24,94 ms ⇒ **446 GB/s**. Não é banda: é issue de ALU — `matvec_kernel_batch` chama
  `T::dot(rowp, abase + n*act_stride, ...)` **uma vez por token**, então a desquantização
  (`iq3s_grid`, montagem de sinais, `__vsub4`) é reexecutada N vezes e só o *load* do peso é
  amortizado. Com o andaime fora do caminho, esse bucket virou **70 % do prefill antes do float4
  e ~85 % depois** (6,93 de 8,07 ms/token).
- **O que daria**: dequantizar o bloco uma vez em registrador e fazer N `dp4a` (bit-exato, mesma
  ordem) vale ~2-2,5× no matvec ⇒ prefill ~200 tok/s; o caminho MMQ/int8 WMMA vale os 446 GB/s
  ⇒ ~380 tok/s (a distância para o llama.cpp está aqui, não no andaime).
- **Por que não foi feito**: `matvec.cuh`/`vecdotq.cuh` são da frente de kernels nesta rodada
  (regra 6.5 do `docs/noite-regras.md`); o protótipo MMQ autorizado no briefing exigiria
  reimplementar a desquantização de 15 dtypes num arquivo novo e não caberia no resto da noite
  sem arriscar os gates. **Encaminhado ao coordenador com a linha de código e os números.**
- **Veredito**: **ABANDONADO nesta frente, com medida** (o ganho não é meu; o achado está
  reportado). É o item de maior valor que sobrou.

