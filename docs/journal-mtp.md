# Diário — frente MTP (verify batelado), branch `feat/noite-mtp`

Frente: fazer o MTP (NextN / decodificação especulativa) **entregar ganho** em vez
dos −9 % medidos. Contrato: `docs/noite-regras.md`. Baseline: `main` no tag
`noite-baseline-2026-09-14` (`e48f3c4`), worktree
`/home/marcelo/Projetos/rdna4-wt-noite-mtp`.

Estado de partida (medido neste worktree, `build/` próprio, IQ3_S, f16 KV):

- `--mtp --draft 3` é **9-11 % mais lento** que ganancioso puro, porque verifica
  um rascunho por passo de trunk (`docs/mtp.md` §4, e o gate `check-mtp-gpu`
  reconfirmou: ganancioso 30,10 tok/s, `spec(D=3)` 27,12 tok/s = +11,0 %).
- Aceitação da cabeça: **86,7 %** em texto de wiki (95,3 % no prompt curto do
  gate).
- `Graph::forward_batch` existe e é bit-exato contra o caminho por token
  (`tests/check_batch_gpu.hip`), mas só devolve o hidden/logits do **último**
  token, e não há rollback do estado recorrente.

## 1. Verify batelado: desenho (antes de medir)

- **Referência**: `docs/medicoes-m8.md` §"Decodificação especulativa com a cabeça
  MTP" (projeção 1,3-1,5×), `docs/mtp.md` §7 ("verify batelado" como o que falta),
  `include/rdna4/graph.cuh` (`forward_batch`, `forward_batch_layer`),
  `include/rdna4/mtp_gen.h` (o laço especulativo atual).
- **Hipótese**: uma rodada que hoje custa `D_eff+1` passos de trunk (~36 ms cada a
  4K) passa a custar **um** `forward_batch` sobre `D_eff+1` tokens: um passe de
  pesos (24,9 ms medidos) + `D_eff+1` × o custo por token dentro do batch.
- **Desenho implementado**:
  1. `Graph::forward_batch_all(tokens, pos, h_all, logits_all)`: o mesmo passe de
     camadas do `forward_batch`, mas com a norma final em **todas** as linhas e a
     LM head batelada (`proj_batch`), devolvendo `h` e logits de **cada** linha —
     é o que a verificação precisa (a linha `j` dá a distribuição da posição
     `pos+2+j`, que é onde o rascunho `j+1` apostou). `forward_batch` **não** foi
     tocado (os gates de prefill continuam com os mesmos números).
  2. `Graph::state_snapshot()` / `state_restore()`: cópia D2D de `d_state_` +
     `d_convst_` (156,4 MiB neste modelo). A KV **não** precisa de snapshot:
     atenção é causal e toda linha é reescrita por posição antes de ser lida.
  3. Rodada batelada: rascunha `k`, um `forward_batch_all([id, d1..dk])`, decide a
     aceitação pelas linhas de logits, e:
     - **aceitou tudo**: o estado já está exatamente no último token commitado —
       sem rollback;
     - **rejeitou**: `state_restore()` + **um segundo** `forward_batch_all` sobre
       `[id, d1..d_a, t]` (o prefixo commitado), que devolve estado, `h` e logits
       da próxima rodada de uma vez.
  4. O *rebuild* da KV do bloco de rascunho (linhas das posições commitadas, a
     partir do `h` do **trunk**) agora roda com `want_logits=false`: a LM head
     compartilhada (874 MiB de pesos, ~2/3 do passo) não é lida, porque nenhum
     consumidor lê logits ali. Isso é ortogonal ao batch e vale para os dois
     caminhos.
  5. Pré-preenchimento do bloco MTP batelado (`mtp_prefill_batched`): o prompt
     passa pelo trunk em blocos de 16 (`forward_batch_all`) e o bloco é passo a
     passo sem a head, bit-idêntico ao laço por token — é o que torna uma medição
     a 16K viável (10 min → ~30 s de prefill).
- **Limite**: o batch tem instanciações 2/3/4/8/16; o replay pós-rejeição usa
  `a+2` linhas (2..4), então o caminho batelado limita o rascunho a `k ≤ 3`
  (`--draft 4..16` continua disponível no caminho serial, `--mtp-serial`).

<!-- resultados a seguir -->

## 2. Verify batelado: implementação e primeiro A/B (dry-run, sem GPU)

- **Referência**: `docs/medicoes-m8.md` §MTP; briefing da frente; `graph.cuh`
  `forward_batch_layer`/`forward_batch`; `mtp_gen.h` laço especulativo.
- **Mudanças (todas aditivas)**:
  - `graph.cuh`: `forward_batch_all()` (norma final em todas as linhas + LM head
    batelada + leitura de `h` e logits de cada linha), `state_snapshot()` /
    `state_restore()` (cópia D2D assíncrona de `d_state_`+`d_convst_`, buffers
    alocados no primeiro uso), `state_bytes()`; `forward_batch()` e `forward_run()`
    **intocados**.
  - `mtp.cuh`: `step_host`/`step_chained`/`run` ganham `want_logits` (default
    `true`, compatível com todas as chamadas existentes); `mtp_required_bytes()`
    (orçamento do bloco: pesos + **duas** caches KV).
  - `mtp_gen.h`: ramo `batch_verify` (default ligado) com o laço da rodada
    batelada; `mtp_prefill_batched()` para o pré-preenchimento; campos novos em
    `MtpGenStats` (`verify_forwards`, `replay_forwards`, `rollbacks`,
    `rollback_tokens`, `snapshot_ms`); `--mtp-serial` no CLI para o A/B.
  - `main.hip`: flag `--mtp-serial`, pré-preenchimento batelado
    (`RD_MTP_PREFILL=serial` volta ao por-token), estatísticas de verify/rollback
    e a checagem de orçamento do bloco MTP.
- **Decisão de desenho que vale registrar**: numa rejeição, o token **do trunk**
  naquela posição *não* é commitado na mesma rodada (ao contrário do caminho
  serial). A rodada termina no último rascunho aceito, o estado é restaurado e o
  prefixo commitado é re-rodado (`[id, d1..d_a]`, `a+1` linhas); o token do trunk
  abre a rodada seguinte, onde o sampler o reproduz a partir dos mesmos logits.
  Isso encurta o replay em uma linha **e** libera o limite de tamanho de batch
  (o replay usa 1..k linhas, e não 2..k+1).
- **Limite explícito**: `matvec_launch_batch` só tem instanciações 2/3/4/8/16.
  Com o desenho acima o verify de `k` rascunhos usa `k+1` linhas, então o caminho
  batelado trabalha com **k ≤ 3** (verify de 4 linhas; replay de 1..3). `D = 4`
  (verify de 5 linhas) exigiria uma instanciação `N = 5` em `matvec.cuh` — arquivo
  de outra frente, então não foi feito; a medição de custo marginal por linha
  (2/3/4) diz se valeria.

## 3. Orçamento de VRAM do bloco MTP entra na conta (achado da revisão adversarial)

- **Referência**: aviso do coordenador (achado da revisão adversarial) +
  `mtp.cuh:365-372` (`alloc_kv` aloca **duas** caches) + `device.h:84-87`
  (`required_bytes` só conhece o trunk) + `docs/mtp.md:269` (descrevia **uma**).
- **Hipótese**: `--mtp` numa configuração que `info` aprova pode morrer em
  `hipMalloc failed`, porque o bloco de rascunho acrescenta ~0,33 GiB de pesos
  (15 tensores) e **duas** caches de `max_ctx × 4 × 256` elementos — +0,25 GiB a
  64K, +0,5 GiB a 128K (f16), fora do orçamento.
- **Comando**: `./build/rdna4-infer run -m <IQ3_S> -p hi -n 1 --mtp --ctx-size 65536`
  (e `info --ctx-size 65536`), mais a aritmética de `mtp_required_bytes()`.
- **Resultado**: (a) `mtp_required_bytes()` no IQ3_S = **335 MiB de pesos + 2 ×
  ctx × 4 × kv_row_bytes** = 16 MiB a 4K, 256 MiB a 64K, 512 MiB a 128K; (b) a
  64K f16 o orçamento antigo dava `12,04 + 2,15 + 1,0 = 15,19 GiB` "aprovado"
  contra 15,9 GiB da placa, e a alocação real precisa de **+0,58 GiB** (bloco +
  caches + snapshot do estado do verify, 156 MiB) — exatamente a faixa em que
  `hipMalloc` falha com fragmentação; (c) o CLI agora imprime
  `MTP budget: trunk+file X GiB + draft block and its 2 KV caches Y GiB` e recusa
  com `insufficient VRAM` **antes** de mapear o modelo.
- **Veredito**: MANTIDO (`mtp_required_bytes()` em `mtp.cuh` + checagem em
  `cmd_run`; `docs/mtp.md` §7 corrigido para duas caches e para o orçamento).

## 4. Verify batelado: exatidão **provada** e custos de fase (gate completo)

- **Referência**: `tests/check_mtp_gpu.hip` (seções novas), `docs/noite-regras.md` §2.
- **Hipótese**: com o snapshot do estado GDN, o caminho batelado tem de ser
  *bit-idêntico* ao serial — não "próximo": mesmos rascunhos, mesma aritmética de
  trunk (`forward_batch` é bit-exato por gate), mesmo rollback exato.
- **Comando** (janela limpa, VRAM 2,67 GB antes → 4,04 GB depois; lock via
  `scripts/gpu-lock.sh timeout 3000 bash /tmp/gates-mtp.sh`):
  `./build/check-batch-gpu`, `GRAPH_LAST_TOKEN=1 ./build/check-graph-gpu ...`,
  `./build/check-mtp-gpu <IQ3_S> 64 3`, `./scripts/check_golden_run.sh`,
  `./scripts/check_regression.sh`.
- **Resultado**:
  - `check-batch-gpu` **OK** (6 s), `check-graph-gpu` **OK** (3 s),
    `check_golden_run.sh` **OK** (32 s), `check_regression.sh` **OK** (74 s) —
    nenhum gate numérico mexeu.
  - `check-mtp-gpu`, seção `batched verification vs serial`:

    | D | batelado | serial | ids | logits finais |
    |---|---|---|---|---|
    | 1 | 30/33 (90,9 %), 33 rodadas, 33 verify, 3 rollbacks | 29/31 (93,5 %), 64 passos de trunk | idênticos ao ganancioso e entre si | **rel-L2 0.000e+00, max\|d\| 0.000e+00** |
    | 2 | 40/45 (88,9 %), 23 rodadas, 4 rollbacks/7 linhas | 36/42 (85,7 %) | idem | **max\|d\| 0.000e+00** |
    | 3 | 44/57 (77,2 %), 19 rodadas, 8 rollbacks/19 linhas | 41/50 (82,0 %) | idem | **max\|d\| 0.000e+00** |

  - rollback isolado (snapshot → lote de 4 linhas → restore → re-execução de 3):
    `max|d| 0.000e+00` entre as linhas — o estado volta exatamente para onde
    estava.
  - **Custos de fase medidos** (posição 20, 4 K de contexto alocado):
    estado recorrente **149,6 MiB**; snapshot+restore **0,54 ms por par**
    (~580 GB/s, perto do pico); passo do bloco MTP **2,22 ms com a LM head** e
    **0,83 ms sem ela** (a head compartilhada é 874 MiB = 63 % do passo, e o
    *rebuild* da KV não a precisa — daí o `want_logits=false`); forward por token
    **32,6 ms**; forward batelado de 2/3/4 linhas **41,2 / 51,0 / 63,2 ms** ⇒
    custo marginal de uma linha dentro do lote **≈ 11 ms**, e um verify de 4
    linhas custa **2,06× menos** que 4 forwards por token.
- **Observação honesta sobre a taxa de aceitação**: as duas variantes commitam a
  *mesma* sequência (ids e logits bit-idênticos), mas contam a aceitação em
  fronteiras de rodada diferentes (no caminho batelado o token do trunk que
  substitui um rascunho rejeitado abre a rodada seguinte, em vez de ser commitado
  na mesma) — daí 77,2 % vs 82,0 % por rascunho em D=3. É estatística de
  contagem, não regressão de qualidade: a sequência commitada é a mesma.
- **Veredito**: MANTIDO. O gate de exatidão passa bit a bit; os gates numéricos
  pré-existentes continuam verdes.

## 5. Merge da frente de prefill (`main` → `feat/noite-mtp`): o verify ficou barato

- **Referência**: aviso do coordenador (commit `264044e`, `feat/noite-prefill`
  mergeado na `main`): `forward_batch` agora batela *todo* o andaime por token
  (atenção com máscara causal, recorrência GDN, conv1d, normas, `kv_write`),
  bit-exato, com `RD_PREFILL_BATCH=0` para A/B.
- **Hipótese**: meu `forward_batch_all` chama `forward_batch_layer`, então herda
  os kernels novos sem mudar uma linha — e é o andaime por token que definia o
  custo do verify (a medição antiga dava ~11 ms de custo marginal por linha, ou
  seja `~1/3` de um token). Se o lote novo entregar os ~8-16 ms/token medidos
  pelo `check-batch-gpu`, o verify de `D+1` linhas cai de ~63 ms para ~43 ms a 4K
  e o teto do ganho sobe.
- **Comando**: `git merge main` (sem conflito — as mudanças do MTP são funções
  novas em `graph.cuh` e blocos delimitados em `main.hip`), recompilação completa,
  e a bateria da fase A: `check-batch-gpu`, `check-graph-gpu`,
  `check-mtp-gpu 64 3`, `check_golden_run.sh`, `check_regression.sh` e a matriz de
  4K — tudo sob uma tomada de lock (`scripts/gpu-lock.sh timeout 2400 bash
  /tmp/mtp-phaseA.sh`).
- **Resultado**: ver §6.
- **Veredito**: (ver §6)

## 6. Matriz medida a 4K e 16K (e um bug de exatidão que ela pegou)

- **Referência**: briefing do coordenador (alvo ≥ 1,5×, `--mtp` byte-idêntico ao
  ganancioso) + `docs/noite-regras.md` §3 (piso de ruído, janela limpa).
- **Comando** (uma tomada de lock por fase; janela limpa: VRAM 119,9 MB antes,
  120,6 MB depois, nenhum outro processo do modelo):
  `scripts/gpu-lock.sh timeout 2400 bash /tmp/mtp-phaseA.sh` (= gates +
  `/tmp/mtp-matrix.sh 4096 64 2 plain: b2:--mtp\ --draft\ 2 b3:--mtp\ --draft\ 3
  s3:--mtp\ --draft\ 3\ --mtp-serial`) e `... /tmp/mtp-phaseB.sh` (o mesmo a 16384).
  Prompt: parágrafo do `wiki.test.raw` repetido até encher o contexto (3 996
  tokens a 4K, 16 282 a 16K), terminando numa frase completa + "In the following
  years, the" (um prompt de wiki cru faz o modelo emitir EOS no primeiro token —
  medido na primeira tentativa, com 0 tokens gerados).
- **Resultado (IQ3_S, f16 KV, greedy, 64 tokens, 2 repetições intercaladas)**:

  | ctx | modo | tok/s (rep1 / rep2) | ganho | tokens/rodada | aceitação/rascunho | rollbacks |
  |---|---|---|---|---|---|---|
  | 4K | ganancioso | 29,33 / 29,33 | 1,00× | — | — | — |
  | 4K | `--mtp --draft 2` (batelado) | 50,15 / 50,14 | **1,71×** | 2,78 | 88,9 % | 3/23 |
  | 4K | `--mtp --draft 3` (batelado) | 51,47 / 51,46 | **1,75×** | 3,56 | 84,9 % | 3/18 |
  | 4K | `--mtp --draft 3 --mtp-serial` (pré-existente) | 25,99 / 25,99 | 0,89× | 3,76 | 89,8 % | 0/17 |
  | 16K | ganancioso | 26,81 / 26,88 | 1,00× | — | — | — |
  | 16K | `--mtp --draft 3` (batelado) | 23,14 / 23,14 | 0,86× | 2,67 | **54,9 %** | **19/24** |
  | 16K | `--mtp --draft 2` (batelado) | 19,74 / 19,75 | 0,74× | 1,73 | **35,6 %** | 25/37 |

  O piso de ruído deste harness é **zero na prática**: as duas repetições de cada
  configuração dão o mesmo tok/s (2 casas) e o mesmo md5 de stdout.
- **BUG DE EXATIDÃO (achado pela própria matriz)**: a 4K a saída do `--mtp` **não**
  era a do ganancioso — o primeiro token já divergia (" actor actor actor …" contra
  "! Boulter starred in two films…"), e **b2, b3 e s3 davam exatamente o mesmo
  md5 entre si**, ou seja, a divergência era comum aos três e vinha de antes do
  laço especulativo. O único componente comum aos três e **não coberto** pelo
  gate em processo (`check-mtp-gpu`, que pré-preenche token a token) era o
  pré-preenchimento batelado do bloco MTP (`mtp_prefill_batched`, que usa
  `forward_batch_all` em blocos de 16). A 20 posições o gate é bit-exato; a 4K
  não era.
- **Correção (MANTIDA)**: o default do pré-preenchimento MTP volta a ser o laço
  por token (o caminho que o gate valida desde a feat/mtp); o batelado fica atrás
  de `RD_MTP_PREFILL=batched`, com o motivo escrito no código. Custo: ~5 s a mais
  de prefill a 4K. O gate novo `RD_MTP_XALL_POS=4096` do `check-mtp_gpu` compara
  `forward_batch_all` com o caminho por token **em contexto longo**, linha a linha
  (h e logits) — é o teste que faltava e que teria pegado isso.
- **16K, aceitação 54,9 %**: com o pré-preenchimento por token o número foi
  re-medido (§7); a hipótese era que o mesmo bug também estivesse degradando o
  estado do bloco de rascunho em contexto longo.

## 7. Atribuição do bug e re-medição com a `main` consertada

- **Referência**: mensagem do coordenador (revisão adversarial 2): **R9** — o
  `forward_batch` não honrava `want_argmax_`, então o braço *ganancioso* gerava o
  token 0 (`!`) como primeiro token quando o último chunk do prefill era um lote
  (justamente o caso a 4K). Ou seja: na §6 quem estava errado era o **baseline**,
  não o `--mtp` (que amostra no host a partir de `logits`). Conserto na `main`
  (`7ecb400`, helper `Graph::finish_argmax()` + gate diferencial no
  `check_golden_run.sh`).
  **R1** — `attn_split_batch_kernel` lia o `q` da primeira linha do grupo para
  todas as linhas, o que corrompia a **atenção dividida em lote** — em contexto
  longo, splits > 1, exatamente o que a verificação do MTP usa. Conserto na `main`
  (`30e9668`), coberto pelo gate novo do `check-batch-gpu` (posição 1088, splits 2).
  **R11** — o despacho em lote da atenção dividida não conhecia `q5_0`/`q4_1`.
- **Hipótese**: com R1 e R9 consertados, (a) a saída do `--mtp` volta a ser
  byte-idêntica ao ganancioso a 4K, (b) o ganho a 4K se confirma, (c) a aceitação
  a 16K melhora (o estado do bloco de rascunho estava sendo construído a partir de
  uma atenção errada).
- **Comando**: `git merge main` (duas vezes: R9 às 05:17, R1/R11 às 05:40),
  rebuild, e `/tmp/mtp-now.sh` sob uma tomada de lock:
  `RD_MTP_XALL_POS=4096 ./build/check-mtp-gpu <IQ3_S> 32 3` (gate do
  `forward_batch_all` contra o caminho por token numa posição de contexto longo,
  linha a linha, N=2/4/16), depois
  `/tmp/mtp-matrix.sh 4096 64 1 plain: b2:… b3:… s3:…` e
  `/tmp/mtp-matrix.sh 16384 64 1 plain: b3:… b2:…`.
- **Resultado**: §8.

## 8. Números finais (R1 + R9 + R11 mergeados, baseline correto)

- **Comando**: `/tmp/mtp-now.sh` sob uma tomada de lock (janela limpa: VRAM
  275 MB antes / 1,31 GB depois, nenhum outro processo do modelo):
  `RD_MTP_XALL_POS=4096 ./build/check-mtp-gpu <IQ3_S> 32 3`,
  `/tmp/mtp-matrix.sh 4096 64 1 plain: b2:--mtp\ --draft\ 2 b3:--mtp\ --draft\ 3
  s3:--mtp\ --draft\ 3\ --mtp-serial`,
  `/tmp/mtp-matrix.sh 16384 64 1 plain: b3:--mtp\ --draft\ 3 b2:--mtp\ --draft\ 2`.
- **Resultado**:

  | ctx | modo | tok/s | ganho | aceitação/rascunho | linhas/rodada | rollbacks | md5 stdout |
  |---|---|---|---|---|---|---|---|
  | 4K (3 996 tok) | ganancioso puro | 29,37 | 1,00× | — | — | — | `84099989b3dca42c` |
  | 4K | `--mtp --draft 2` (batelado) | **34,56** | **1,18×** | 67,9 % | 2,96 | 13/27 | `84099989b3dca42c` |
  | 4K | `--mtp --draft 3` (batelado) | 30,11 | 1,03× | 52,1 % | 3,92 | 17/25 | `84099989b3dca42c` |
  | 4K | `--mtp --draft 3 --mtp-serial` | 25,85 | 0,88× | 58,2 % | — | 0/19 | `84099989b3dca42c` |
  | 16K (16 282 tok) | ganancioso puro | 26,81 | 1,00× | — | — | — | `e5ac04e772dd553f` |
  | 16K | `--mtp --draft 3` (batelado) | **30,53** | **1,14×** | 67,7 % | 3,95 | 10/21 | `e5ac04e772dd553f` |

  **A saída é byte-idêntica ao ganancioso puro nas duas pontas, em todas as
  variantes de `--mtp`** — é a propriedade que torna a especulação segura, e é o
  que o `check_regression.sh` (7 prompts, ids bit-exatos, rel-L2 ≤ 1e-5) também
  cobre do lado do motor.
- **Correção honesta de um número anterior**: os 1,71-1,75× da §6 foram medidos
  contra o baseline **quebrado pelo R9**, que gerava o token 0 como primeiro token
  e caía em texto degenerado ("actor actor actor…"). Texto degenerado infla a
  aceitação do rascunho (88,9 % → **67,9 %** com texto coerente), e é daí que
  vinha quase todo o ganho aparente. O contraste que sobrevive — e que é a tese da
  frente — é **0,88× (verify serial) → 1,18× (verify batelado)**, mesma janela,
  mesma saída byte a byte.
- **R1 explica o 16K**: com o conserto da atenção dividida em lote a aceitação
  subiu de 54,9 % → **67,7 %** e o ganho virou **+14 %** em vez de −14 %. Sem esse
  diagnóstico eu teria reportado um negativo falso.
- **O ótimo medido é D=2, não D=3** (e isso contraria a projeção do
  `docs/medicoes-m8.md`, que apontava D=4): a 4K, D=2 dá 1,18× e D=3 dá 1,03×.
  O motivo está na tabela: a aceitação por rascunho cai de 67,9 % (D=2) para
  52,1 % (D=3) — os rascunhos encadeados acertam bem menos que o primeiro — e a
  verificação cresce de 2,96 para 3,92 linhas por rodada, enquanto a fração de
  rodadas com *rollback* sobe de 48 % para 68 %. Ou seja: cada rascunho a mais
  custa 2,2 ms de rascunho + 0,83 ms de rebuild + uma linha de verify, e devolve
  cada vez menos.
- **Onde o tempo vai (4K, D=2, 27 rodadas para 64 tokens = 68,6 ms/rodada)**:
  o verify (1 `forward_batch_all` de ~3 linhas por rodada) mais o *replay* das 13
  rodadas com rejeição (48 %) domina; o rascunho custa ~4,4 ms/rodada (2 passos de
  2,2 ms, cada um 63 % LM head compartilhada) e o *rebuild* da KV ~1,1 ms/rodada
  (1,37 linhas aceitas a 0,83 ms, sem a head). Custos unitários medidos:
  passo de trunk por token 34,0 ms; `forward_batch_all` bit-exato contra o caminho
  por token em posição 4084 (h e logits, max|d| 0,0) para N=2 e N=4; snapshot +
  restore do estado recorrente 0,54 ms por par (149,6 MiB).
- **D=4 não foi entregue, e o motivo é estrutural**: `matvec_launch_batch` só tem
  instanciações N = 2/3/4/8/16 e o verify de `k` rascunhos precisa de `k+1` linhas
  (D=4 → 5 linhas). Fazer isso exige uma instanciação N=5 em `matvec.cuh` (arquivo
  de outra frente); com o ganho de D=3 já abaixo do de D=2 (a aceitação marginal
  cai e o replay sobe), o custo não se justifica — o coordenador confirmou a
  decisão de não gastar a janela nisso.
