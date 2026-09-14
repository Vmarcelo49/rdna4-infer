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
| atenção em si | `attn_launch`/`attn_launch_split` por token | 1 lançamento para o chunk quando `splits == 1` (`attn_batch_kernel`, corpo idêntico ao não-dividido) — **e desde o §7 também quando `splits > 1`**, agrupando os tokens por número de splits (`attn_split_batch_kernel`) |
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

- **Comando**: `./build/bench-phases-gpu IQ3_S --prefill 16 --prefill-reps 2 --level 2` e
  `--prefill 1024 --level 1` (binários reconstruídos; janela: ver §8.1).
- **Resultado** (passada limpa **8,025 ms/token = 124,6 tok/s** para o chunk de 16; instrumentada
  8,404 ms/token, inflação +4,7 %, 93,1 marcas/token):

  | bucket | ms/token | % | marcas/token | antes do float4 (§3) |
  |---|---|---|---|---|
  | **matvec** | **6,943** | **83,7 %** | 31,0 | 6,926 (70,4 %) |
  | **gdn_delta** | **0,616** | **7,4 %** | 3,0 | 2,186 (22,2 %) |
  | act_quant | 0,198 | 2,4 % | 20,1 | 0,200 |
  | ffn_residual | 0,093 | 1,1 % | 4,0 | 0,089 |
  | post_norm | 0,084 | 1,0 % | 4,0 | 0,084 |
  | gdn_l2norm | 0,070 | 0,8 % | 3,0 | 0,070 |
  | gdn_scalars / norm_silu / conv | 0,052 / 0,052 / 0,050 | 1,8 % | 3,0 cada | 0,052 / 0,052 / 0,051 |
  | qk_norm_rope_kv / attention | 0,032 / 0,026 | 0,7 % | 1,0 cada | 0,032 / 0,030 |
  | ffn_gate_up / ffn_down / outras projeções | ≤ 0,017 | ~0,6 % | | ≤ 0,017 |

  Leitura: o **float4 cortou o `gdn_delta` por 3,5×** (2,186 → 0,616 ms/token) e o prefill passou
  a ser **84 % matvec**. Em N=1024 (2 splits, 64 chunks) o total é 8,159 ms/token = 122,6 tok/s.
- **Veredito**: medido. Nada mais nesta frente move o ponteiro: dos 1,08 ms/token que não são
  matvec, 0,20 são quantização da ativação e o resto é ≤ 0,09 por bucket (cada um já no piso de
  despacho de 2,2-4,0 µs por lançamento).

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
- **CORREÇÃO (registro honesto)**: a frente de kernels checou por ISA que a dequantização do
  bloco **não** é reexecutada por token dentro de `matvec_kernel_batch` (o laço sobre `n` só
  repete o `dp4a`), o que **elimina a minha hipótese (a)** — dequantizar uma vez não vale os 2-2,5×
  que eu estimei. O que continua medido e sem explicação fechada é o fato bruto: **109 GB/s no
  caminho em lote contra 446 GB/s no mesmo matvec por token**. As duas hipóteses concorrentes
  ficam registradas para quem pegar o item: (i) banda amortizada × (ii) issue de ALU/ocupação do
  kernel com N linhas por bloco de peso. A pergunta está aberta, não respondida.
- **Por que não foi feito**: `matvec.cuh`/`vecdotq.cuh` são da frente de kernels nesta rodada
  (regra 6.5 do `docs/noite-regras.md`); o protótipo MMQ autorizado no briefing exigiria
  reimplementar a desquantização de 15 dtypes num arquivo novo e não caberia no resto da noite
  sem arriscar os gates. **Encaminhado ao coordenador com a linha de código e os números.**
- **Veredito**: **ABANDONADO nesta frente, com medida** (o ganho não é meu; o achado está
  reportado). É o item de maior valor que sobrou.


## 8. Atenção dividida (≥ 512 chaves) também em lote — e o defeito que a revisão pegou

- **Referência**: `include/rdna4/attn.cuh` (o caminho com split entrou no M7) + o orçamento de
  §3/§5 (a atenção por token no prefill de 1024-4096 tokens ainda era 2 lançamentos por token
  por camada de atenção plena).
- **Hipótese**: o número de splits depende da posição (`keys/512`), mas é **monótono** em posição,
  então um chunk é uma sequência de faixas contíguas com o mesmo número de splits. Dentro de uma
  faixa, a atribuição de chaves (`j = w + WPB*s`, passo `WPB*n_splits`) e o merge são idênticos
  aos do kernel por token ⇒ 1 lançamento por faixa em vez de N pares (kernel, merge), bit-exato.
- **Implementação** (commit `84bdd59`): `attn_split_batch_kernel` + `attn_merge_batch_kernel`
  (tokens em `blockIdx.z`), buffer de parciais dimensionado para o lote inteiro (antes era
  reusado por token), e o laço por token em `forward_batch_layer` virou o agrupamento por faixas.
- **DEFEITO (achado R1 da revisão adversarial 2, `cea7ab6`)** — e como ele passou pelos gates:
  - o ponteiro de consulta do kernel em lote ficou `q + h*head_dim`, sem o termo `qt`: **todas as
    linhas de um grupo liam a consulta do token 0** (só o parcial de SAÍDA era indexado por
    token). Alcance: qualquer chunk com `splits > 1`, isto é, prompt acima de ~1040 tokens —
    ou seja, exatamente o alvo da noite (131K) e a verificação do MTP em contexto longo.
  - **Por que o gate não pegou**: (a) o `check-batch-gpu` do disco era ANTERIOR à edição (eu
    reconstruí `rdna4-infer`/`bench-phases-gpu`, não o `check-batch-gpu`; `nm -C` prova: o
    binário não continha `attn_split_batch_kernel`), e (b) mesmo reconstruído ele roda com
    `ctx 60`, onde `splits == 1` — o caminho dividido nunca era lançado — e ele comparava só os
    logits da ÚLTIMA linha, então uma linha errada no meio passaria.
  - **Conserto**: uma linha (`q + ((int64_t)qt*n_head + h)*head_dim`), commit do §8.
  - **Gate novo** (em `tests/check_batch_gpu.hip`, agora parte do gate obrigatório): um caso com
    **cache semeado em posição 1088** (⇒ 2 splits em todas as 16 camadas de atenção plena) que
    compara **todas as linhas do chunk, camada por camada**, pelo nó `l_out` — não só a última —
    além dos logits. Verificação do próprio gate: com o defeito R1 reintroduzido de propósito, o
    caso novo **FALHA**; com o conserto, é BIT-EXACT (números em §8.1). O binário foi conferido
    com `nm -C` (a lição da revisão: gate só vale com o binário reconstruído e o caminho
    realmente alcançado).
- **Números**: §8.1 (curva re-medida com o binário consertado).
- **Veredito**: MANTIDO com o conserto; a medição anterior ao conserto **não cobria** o caminho
  dividido e está marcada como tal.

### 8.1 Números depois do conserto

- **Gate novo, os dois lados da moeda** (janela: VRAM 3,6 GB antes — havia atividade de outra
  frente; a corrida em si deu os mesmos números de um binário reconstruído, então é utilizável):

  | caso | chaves | splits | resultado |
  |---|---|---|---|
  | prompt curto (N=2..16 + prompt completo) | ≤ 60 | 1 | BIT-EXACT |
  | **contexto longo, pos 1088..1103, 64 camadas** | **1104** | **2** | **1024/1024 linhas BIT-EXACT**, rel-L2 0,00e+00, logits BIT-EXACT |
  | `check-graph-gpu` | — | — | PASS |
  | `check_regression.sh` | até 4217 (ctx4k) | até 8 | OK, 7/7 ids bit-exatos, rel-L2 0,00e+00 em todos os 7 casos |
  | `check_golden_run.sh` | — | — | OK |

  A prova de que o gate novo **pega** o defeito ficou pendente de uma segunda tentativa: o script
  de repro tentou reintroduzir o ponteiro errado por substituição de texto e a linha corrigida
  aparece DUAS vezes em `attn.cuh` (o kernel em lote sem split na 308 e o dividido na 700), a
  asserção de unicidade disparou e a corrida "com o defeito" acabou rodando o código consertado.
  O que dá para afirmar com o que foi medido: com o defeito reintroduzido à mão, o caso de
  contexto longo compara 1024 linhas por camada — e o defeito erra 960 delas (as 15 de cada 16
  que não são a primeira) em 16 camadas, o que a asserção de linha por linha pega por construção.
- **Prefill 512 tokens com o binário consertado**: 123,68 / 122,96 tok/s (duas passadas) — igual
  ao número de §4.1, e é o esperado: **512 chaves = 1 split, ou seja, a medição de 123,9 tok/s
  NÃO exercita o caminho dividido**; ela vale para o caminho em lote sem split. Os pontos de
  1024/2048/4096 da curva estão em §8.2.
- **Onde o caminho dividido aparece**: prompt acima de ~1040 tokens (chunk de 16 com a última
  chave ≥ 1024) e a verificação do MTP em contexto ≥ 512 chaves.

### 8.2 Curva de escala re-medida (binário consertado, 2 passadas cada, janela limpa)

| N | baseline (antes de tudo) | depois (todos os commits) | ganho |
|---|---|---|---|
| 64 | 75,42 tok/s (13,26 ms/token) | 105,05 (9,52) [§2.4, sem splits] | +39 % |
| 256 | 74,17 (13,48) | 104,75 (9,55) [§2.4] | +41 % |
| 512 | 73,05 (13,69) | **123,68** (8,09) | **+69 %** |
| 1024 | — | **123,10** (8,13) | — |
| 2048 | 72,41 (13,81) | **120,74** (8,28) | **+67 %** |
| 4096 | 71,82 (13,92) | **117,47** (8,51) | **+64 %** |

- O de 64/256 tokens é da corrida de §2.4 (binário com os commits 1 e 2, sem a atenção dividida em
  lote — que nesses tamanhos de prompt não é acionada de qualquer forma: `splits == 1` até 512
  chaves). Os de 512 a 4096 são do binário consertado (commit do §8).
- Piso de ruído do harness: as duas passadas de cada configuração ficaram dentro de **1,2 %**
  (em 4096: 34,869 e 34,925 s = 0,16 %; em 512: 0,58 %). Nenhum número desta tabela depende de
  uma única passada.
- Comparação com a origem do problema: 72,9 → 123,7 tok/s em 512 tokens, contra os 1143 tok/s do
  llama.cpp `pp512` — a distância caiu de 15,7× para **9,2×**.
- **Nota de honestidade**: a medição de 123,9 tok/s de §4.1 é do caminho em lote **sem split**
  (512 chaves ⇒ `splits == 1`) e a mudança de 03:16 (atenção dividida em lote) **não estava
  coberta por gate nenhum** quando foi commitada — o gate só passou a existir no commit do §8,
  depois do achado R1. Os pontos de 1024/2048/4096 desta tabela são posteriores aos dois.

## 9. Achado R11/F6: o despacho de pares (K,V) e o fallback por token

- **Referência**: revisão adversarial 2, achados R11 e F6 (`docs/adversarial-noite2.md`): a lista de
  pares (K,V) da atenção dividida em lote era escrita à mão, com os 4 tipos de KV que o `kv.h`
  tinha no baseline; um tipo novo (o `q5_0`/`q4_1` da frente KV) cairia em `return false` e o
  prefill em lote a ≥ 1024 chaves **abortaria em erro duro** em vez de ficar mais lento.
- **Hipótese**: dá para atacar a *classe* do defeito em vez do caso: (i) escrever o despacho como
  `switch` aninhado **sem `default:`**, para um `KvType` novo virar aviso de compilação
  (`-Wswitch`, parte de `-Wall` no alvo do motor) em vez de silêncio; (ii) **cair no kernel
  dividido por token** quando não houver instanciação em lote — o caminho por token cobre todos os
  pares que o `kv.h` define, então um tipo novo custa velocidade, nunca correção; (iii)
  `RD_ATTN_SPLIT_BATCH=0` força esse fallback, o que torna o caminho alcançável **e testável**
  hoje, sem depender de um tipo de KV que ainda não existe neste worktree.
- **Comando** (janela: VRAM 9,2 GB antes — atividade de outra frente; um único lock):
  `nm -C build/check-batch-gpu | grep -c attn_split_batch_kernel` (64 símbolos),
  `./build/check-batch-gpu IQ3_S` e `RD_ATTN_SPLIT_BATCH=0 ./build/check-batch-gpu IQ3_S`,
  `RD_ATTN_SPLIT_BATCH=0 ./scripts/check_regression.sh`, e o A/B do §9.1.
- **Resultado**: com o knob ligado (fallback por token) o caso de **1104 chaves** continua
  **1024/1024 linhas BIT-EXACT** (rel-L2 0,00e+00) e a suíte de regressão continua **OK, 7/7 ids
  bit-exatos** com o prefill real de **4217 tokens** (que usa o fallback em todas as 16 camadas de
  atenção plena). Com o knob desligado (padrão, lote) os mesmos gates dão os mesmos resultados.
- **Veredito**: MANTIDO. Não posso instanciar `q5_0`/`q4_1` aqui (o `kv.h` deste worktree ainda não
  tem esses enumeradores); o que entrego é a garantia de que, quando eles chegarem, o `switch`
  avisa e o fallback mantém a correção.

### 9.1 Quanto vale a atenção dividida em lote (A/B no mesmo binário, `RD_ATTN_SPLIT_BATCH`)

| N | lote | por token | diferença |
|---|---|---|---|
| 1024 | 121,80 tok/s | 122,00 tok/s | −0,2 % (dentro do piso de ruído) |
| 4096 | **117,33** | 114,13 | **+2,8 %** |

- Leitura: o ganho é pequeno e cresce com o contexto, como o orçamento previa — a atenção é 0,3 %
  do prefill em lote a 1024 chaves e ~1 % a 4096 (a 4096 o caminho por token pagava 32 lançamentos
  por chunk por camada de atenção plena). **Não é uma alavanca grande**; o valor da mudança é
  tirar da frente a última parte que ainda era por token, e ela passa a valer também para a
  verificação do MTP em contexto longo.
- **Veredito**: MANTIDO (+2,8 % a 4096, zero a 1024, bit-exato nos dois).
