# rdna4-infer — MTP (NextN): o bloco 64 como cabeça de rascunho

O GGUF tem 65 blocos: 64 camadas de trunk mais o bloco 64, que é o cabeça
**MTP / NextN** (`qwen35.nextn_predict_layers = 1`). O trunk nunca o executava
(`Graph::n_layer()` = `block_count - nextn_predict_layers` = 64). Este trabalho
carrega o bloco, roda o forward de rascunho e mede se ele vale a pena.

Resposta curta, medida no IQ3_S: **a cabeça acerta 86,7 % dos rascunhos contra a
continuação gananciosa do trunk em texto de wiki** (95,3 % num prompt que o
modelo repete em ciclo), mas **a decodificação especulativa ainda é ~9 % mais
lenta** neste motor, porque o trunk não é batelado: verificar N rascunhos custa
N passos de trunk, exatamente o mesmo que a decodificação gananciosa, e o
rascunho é só custo extra. O ganho depende de um *verify* batelado, que não
existe (ver §5).

## 1. O que o bloco faz (e o que é certo x inferido)

Transcrito de `llama_model_qwen35::graph_mtp` (`src/models/qwen35.cpp` do
llama.cpp **b10902**, o mesmo arquivo de onde o trunk já vinha) e confirmado
nó a nó contra esse grafo (§3):

```
h_t   = hidden do trunk DEPOIS do output_norm        <- é o `hidden` que
x     = eh_proj[ enorm(emb(tok)) ; hnorm(h_t) ]         forward_tokens devolve
x    += attn(rms_norm(x, attn_norm)) * sigmoid(gate)    (cache KV própria)
x    += ffn(rms_norm(x, post_attention_norm))
h'    = rms_norm(x, nextn.shared_head_norm)
logits= output.weight @ h'                            <- LM head compartilhado
```

O bloco é um bloco de **atenção cheia** (tem `attn_q/k/v/output` + q/k norms e
nenhum `ssm_*`), com a mesma montagem do trunk: `attn_q` emite
`[q (head_dim) | gate (head_dim)]` por cabeça, as normas q/k são por cabeça,
RoPE idêntico (`rope_dim_count`, split-half), escala `1/sqrt(head_dim)`.
`h'` também é o que o llama.cpp realimenta para rascunhar mais de um token
(`pending_h` em `common/speculative.cpp`) — é o `step_chained()`.

**Certo por medição** (não por leitura de código):

| ponto | evidência |
|---|---|
| `h_t` é o hidden **pós-`output_norm`** | `MTPORACLE: h 7` do oráculo = `2.724328 -2.937548 3.077741 ... sum -76.578262` e o `result_norm` do dump `oracle_prompt6_ub1_tok7_cpu.txt` do próprio projeto = `2.7243 -2.9375 3.0777 ... sum -76.578186` |
| concat é **enorm antes de hnorm** | `mtp_concat-64 = CONCAT(mtp_enorm-64, mtp_hnorm-64)` no dump; e pela taxa de aceitação: `[e;h]` = 96,9 % vs `[h;e]` = **0,0 %** (`check-mtp-gpu`, 32 posições) |
| pareamento `(token em p, h do trunk em p−1) → prevê p+1` | o oráculo do llama.cpp dá 21/24 com esse pareamento e 1/24 com o deslocado |
| embed do rascunho = `token_embd.weight` do trunk | `mtp_tok_embd-64 = GET_ROWS(token_embd.weight)`, soma idêntica ao trunk; o GGUF não tem `nextn.embed_tokens` |
| LM head = `output.weight` do trunk | `result_output = MUL_MAT(output.weight)`; o GGUF não tem `nextn.shared_head_head` |
| norm final própria do bloco | `h_nextn = MUL(norm, blk.64.nextn.shared_head_norm.weight)`, não o `output_norm` do trunk |
| cache KV **própria** do bloco | `build_attn_inp_kv()` no grafo MTP, e cada linha é `(token i, h_(i-1))`; `h_(-1) = 0` (o `pending_h` zerado do speculative.cpp) |

**Inferido** (não verificado): o `h_(-1) = 0` da primeira linha é o que o
`pending_h` zerado do llama.cpp implica, e o oráculo reproduz os nós do passo 7
com ele; nada no modelo "define" esse valor. Também não é verificável aqui se a
cabeça foi treinada com essa convenção ou se o Qwen treina com um `h` anterior
diferente — os números batem, então a convenção está certa para este arquivo.

## 2. Arquivos

| arquivo | o quê |
|---|---|
| `include/rdna4/mtp.cuh` | `MtpHead`: carrega os 15 tensores do bloco 64 (nome, dtype e dims checados — nunca fallback), cache KV própria, `step_host()` (h do trunk) e `step_chained()` (h próprio) |
| `include/rdna4/mtp_gen.h` | os três modos do CLI (ganancioso puro, `--mtp-score`, `--mtp --draft N`); header-only de propósito, para o `CMakeLists.txt` continuar append-only no merge das frentes |
| `include/rdna4/graph.cuh` | aditivo: `mtp_shared_weights()` (empresta `token_embd` + `output.weight`) e o include do header novo. Nada do forward do trunk mudou e, sem MTP, nenhum tensor do bloco 64 é carregado (mesma VRAM de antes) |
| `src/main.hip` | só as flags (`--mtp`, `--mtp-score`, `--draft N`) e o dispatch, dentro de um bloco delimitado; o laço de decode existente não foi tocado |
| `tests/check_mtp_gpu.hip` | estrutura, oráculo nó a nó, taxa de aceitação, equivalência com a gananciosa, custo do passo de rascunho |
| `tests/oracle_mtp.cpp` | ferramenta de oráculo: roda o grafo MTP do llama.cpp na CPU e despeja os nós |

## 3. O oráculo: existe, e não é o llama.cpp "ignorando o bloco 64"

Em versões antigas do llama.cpp o bloco 64 era ignorado ("unused tensor
blk.64.attn_output.weight ... ignoring"). **Neste checkout não é**: o b10902
implementa `llama_model_qwen35::graph_mtp`, acionado por um segundo contexto com
`ctx_type = LLAMA_CONTEXT_TYPE_MTP` + `llama_set_embeddings_nextn(ctx, true,
false)` (é o caminho de `--spec-type draft-mtp` do `llama-cli`). Ou seja, este
bloco **tem** oráculo, no mesmo estilo do resto do repo:

```bash
# captura (CPU, ~23 s; gera reference/oracle_mtp_prompt6_cpu.txt, gitignored)
./build/oracle-mtp <model.gguf> 9419 1814 11 411 369 264 1228 13 --steps 24 \
  > reference/oracle_mtp_prompt6_cpu.txt
# comparação (GPU, dentro do lock como tudo que toca a GPU)
scripts/gpu-lock.sh ./build/check-mtp-gpu <model.gguf> 64 3
```

`oracle-mtp` preenche as linhas 0..6 da cache KV do bloco com `(token i, h_(i-1))`
e despeja os **23 nós** de um passo no formato do `llama-eval-callback`, o mesmo
que o `check-graph-gpu` já lê. O `check-mtp-gpu` repete o passo e compara 20 nós
(2 são artefatos de `view` do ggml):

```
mtp_tok_embd-64      sum   -1.734 vs   -1.734  rel 0.000
mtp_enorm-64         sum  -93.588 vs  -93.589  rel 0.000
mtp_hnorm-64         sum  -85.841 vs  -87.370  rel 0.018
mtp_concat-64        sum -179.429 vs -180.959  rel 0.008
mtp_eh_proj-64       sum  -52.657 vs  -55.979  rel 0.059
mtp_attn_norm-64     sum  -58.505 vs  -63.209  rel 0.074
mtp_Qcur_full-64     sum -20466.5 vs -20609.3  rel 0.007
mtp_Qcur_normed-64   sum -224.639 vs -229.337  rel 0.020
mtp_Kcur_normed-64   sum  -62.594 vs  -60.795  rel 0.030
mtp_Vcur-64          sum  -34.743 vs  -35.447  rel 0.020
mtp_gate-64          sum -20135.8 vs -20268.9  rel 0.007
mtp_attn_pregate-64  sum  304.516 vs  302.002  rel 0.008
mtp_attn_out-64      sum  -20.336 vs  -21.188  rel 0.040
mtp_attn_residual-64 sum  -72.993 vs  -77.167  rel 0.054
mtp_attn_post_norm-64 sum -48.324 vs  -52.309  rel 0.076   <- pior nó
mtp_ffn_out-64       sum -117.003 vs -120.426  rel 0.028
mtp_post_ffn-64      sum -189.996 vs -197.593  rel 0.038
h_nextn              sum -140.879 vs -145.728  rel 0.033
result_output        sum -709397  vs -710525   rel 0.002
```

O gate é a **soma com erro relativo ≤ 10 %** (piso 1,0 para somas que cancelam);
a coluna de amostras individuais é informativa, porque valores de `Qcur`/`Vcur`
passam perto de zero e a razão explode lá. Os 0,8-7,6 % que sobram são o ruído do
próprio trunk entrando por dois caminhos (o `h` de entrada e as linhas 0..6 da
cache do bloco, que o llama.cpp construiu com os `h` dele): o `check-graph-gpu`
mede 6,2 % de desvio em `result_norm`, então essa é a ordem de grandeza esperada.
Os dois nós que saem **bit-iguais** (`mtp_tok_embd`, `mtp_enorm`) e o
`result_output` a 0,2 % delimitam esse ruído.

**O token rascunhado bate com o do oráculo**: os dois melhores logits daquele
passo estão a **5,2e-4** (271 em 15,427459 vs 198 em 15,426939), e o nosso argmax
é 271 — o mesmo empate resolvido do mesmo lado. Vale registrar que a primeira
versão do teste dava 198, e a causa não era o empate: era o *teste* alimentando
`h_7` no passo da posição 7 em vez de `h_6` (o motor sempre esteve certo). Esse
erro deslocava `mtp_hnorm` em 69 % e `mtp_attn_pregate` em 319 % — ou seja, a
comparação nó a nó pegou um erro de pareamento de uma linha, que é exatamente
para isso que ela serve.

O dump é de um arquivo de pesos específico e o `oracle-mtp` grava o tamanho do
modelo nele: rodar o `check-mtp-gpu` no IQ4_XS **pula** a comparação (dizendo o
motivo) em vez de acusar 34 % de desvio que seria só a quantização diferente.
Para comparar o IQ4_XS, capture um dump dele com o mesmo comando.

## 4. Medições

Tudo com `scripts/gpu-lock.sh`, IQ3_S, `--greedy` (temp 0), KV f16, ctx 4096.
Prompt de wiki = um parágrafo de `reference/data/wikitext-2-raw/wiki.test.raw`
(72 tokens); prompt "França" = o que o modelo continua em ciclo
(`The capital of France is Paris, ...`, 20 tokens).

### Taxa de aceitação (`--mtp-score`, geração inalterada)

| prompt | rascunhos | aceitos | taxa |
|---|---|---|---|
| wiki.test.raw, 128 tokens | 128 | 111 | **86,7 %** |
| "França" (continuação em ciclo), 128 tokens | 128 | 122 | 95,3 % |
| prompt do `check-mtp-gpu`, 64 tokens | 64 | 61 | 95,3 % |
| wiki.test.raw, 128 tokens, **IQ4_XS** | 87 | 74 | 85,1 % |
| o próprio drive MTP do llama.cpp, prompt de 8 tokens | 24 | 21 | 87,5 % |

A última linha é do `tests/oracle_mtp.cpp` rodando o grafo MTP do llama.cpp: é o
mesmo cabeça, medida do outro lado. As duas primeiras dizem o que interessa: em
texto natural a cabeça acerta ~87 %, e em texto que o modelo já está repetindo
sobe para ~95 %.

### Decodificação especulativa (`--mtp --draft N`), prompt de wiki

| modo | aceitos/rascunhos | passo de trunk | tempo | tok/s |
|---|---|---|---|---|
| ganancioso puro | — | 128 | 4,61 s | **27,79** |
| `--mtp-score` | 111/128 (86,7 %) | 128 | 4,95 s | 25,87 (−6,9 %) |
| `--mtp --draft 1` | 54/63 (85,7 %) | 128 | 4,91 s | 26,04 (−6,3 %) |
| `--mtp --draft 2` | 69/90 (76,7 %) | 128 | 5,04 s | 25,38 (−8,7 %) |
| `--mtp --draft 3` | 76/109 (69,7 %) | 128 | 5,07 s | 25,27 (−9,1 %) |
| `--mtp --draft 4` | 79/125 (63,2 %) | 128 | 5,14 s | 24,88 (−10,5 %) |

`D=3` commita 3,46 tokens por rodada e `D=4` 4,00 (32 rodadas). No IQ4_XS o
`check-mtp-gpu` mede, no mesmo A/B em processo: ganancioso 28,06 tok/s,
`--mtp-score` −8,8 %, `--mtp --draft 3` −11,0 % (mesma conclusão, mesmo efeito). A queda da taxa
por rascunho com D é esperada: o primeiro rascunho usa o `h` do trunk, os
seguintes são encadeados pelo `h` do próprio bloco.

**Por que é mais lento, e não mais rápido:** o número de passos de trunk é
**idêntico** ao da decodificação gananciosa (128 nos dois casos, uma coluna da
tabela que o teste verifica explicitamente). Verificar um rascunho aqui custa um
forward de trunk de um token, porque o trunk é token a token (a limitação de
prefill não batelado do README). Aceitar um rascunho economiza um *sample*, não
um forward. O custo extra é o bloco de rascunho:

| | medido |
|---|---|
| passo do bloco MTP (334,7 MiB de pesos) | **2,28 ms** em sequência, 2,32 ms com sync por passo |
| passo do trunk a 4K | 36,1 ms |
| rascunho / trunk | **6,3 %** |

Daí os −6 % a −10 %: o trunk custa o mesmo e o rascunho entra por cima
(`--mtp-score` paga 1 passo de rascunho por token, `--mtp --draft 3` paga ~5,4).

### O que faria valer a pena (extrapolação, não medição)

Com um verify **batelado** (um passe de trunk para os D+1 tokens, via o
`matvec_launch_batch` do M6 + um kernel de atenção multi-query com máscara
causal, que não existe hoje — `attn.cuh` é do outro agente), a conta com os
números medidos fica: uma rodada de D=4 custaria ~36 ms (um passe) + 10 passos de
rascunho (~23 ms) para ~4,0 tokens commitados ≈ **14,7 ms/token**, contra 36,1 ms
agora — **~2,5×**. É a taxa de aceitação medida (63-87 % por rascunho, 3,5-4,0
tokens por rodada) que sustenta essa conclusão; sem ela, nada disso importaria.

## 5. Correção: o que é garantido

O trunk decide todo token commitado; o rascunho só propõe. Com `temp ≤ 0` a
sequência é **idêntica** à decodificação gananciosa, e isso é testado de três
formas:

1. `check-mtp-gpu`: A/B no mesmo processo e no mesmo grafo (resetando o estado),
   comparando os ids de `--mtp-score`, `--mtp --draft 1/2/3/4` e do caminho
   ganancioso (`ok speculative decoding is token-for-token plain greedy decode`);
2. CLI, md5 do stdout: `plain == score == draft 1/2/3/4` no prompt de wiki e no
   da França;
3. nenhum forward é feito em token não commitado, então o estado recorrente
   (GDN) **nunca** precisa de rollback: um rascunho rejeitado simplesmente não é
   executado, e as linhas de KV já escritas são reescritas por posição antes de
   qualquer leitura (atenção é causal).

Um bug real que essa comparação pegou, e vale registrar: uma rodada especulativa
podia commitar mais que `n_predict`, e como o callback de streaming escreve os
bytes no momento do commit, o texto do `--mtp` divergia da gananciosa (md5
diferente) mesmo com os ids iguais. A rodada agora é limitada para não passar de
`n_predict` e o excesso é erro explícito, com checagem no teste
(`no mode streamed more than n_predict tokens`).

Limites, explícitos:

- **só `temp ≤ 0`**: com temperatura > 0 o rascunho mudaria o número de chamadas
  ao sampler e portanto o fluxo do RNG, e a comparação bit a bit deixaria de
  valer. O CLI recusa `--mtp` com `temp > 0`. Fazer amostragem especulativa
  correta exigiria o rejection sampling de verdade (aceitar com probabilidade
  `p_trunk(x)/p_draft(x)` e reamostrar da residual) — não implementado.
- `--mtp-score` compara com o **argmax cru** dos logits do trunk; com
  `repeat_penalty != 1.0` o trunk escolhe um token penalizado e a taxa medida é
  um piso (o CLI avisa).
- um único bloco NextN (`nextn_predict_layers == 1`); `MtpHead::init` falha
  alto em qualquer outro valor.

## 6. Como rodar

```bash
source scripts/rocm-env.sh && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j6

MOD=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf

# taxa de aceitação (geração inalterada)
scripts/gpu-lock.sh ./build/rdna4-infer run -m $MOD -p "..." -n 128 --greedy --mtp-score
# decodificação especulativa
scripts/gpu-lock.sh ./build/rdna4-infer run -m $MOD -p "..." -n 128 --greedy --mtp --draft 3
# gate completo (estrutura, oráculo, aceitação, equivalência, custo)
scripts/gpu-lock.sh ./build/check-mtp-gpu $MOD 64 3
```

No IQ4_XS o gate roda igual (estrutura, aceitação, equivalência, custo) e só a
comparação com o oráculo é pulada, porque o dump é do IQ3_S:

```
scripts/gpu-lock.sh ./build/check-mtp-gpu $IQ4_XS 64 3
   == oracle: ... was captured from a 12040883104-byte model, this one is
      14252845984 bytes (different quantization): comparison skipped ==
   ... PASS (0 failures)      # spec D=3: 41/50 (82,0%), plain 26,30 tok/s, +10,9%
```

`RD_MTP_CONCAT=he` inverte a ordem do concat (só para medir; ver §1),
`RD_MTP_IDS=1` imprime os ids gerados no stderr.

## 7. O que ficou de fora

- **Verify batelado** (o que daria o ganho): precisa de atenção multi-query com
  máscara causal e do forward batelado do trunk, que toca `attn.cuh`/`gdn.cuh`.
- **Amostragem especulativa correta** (rejection sampling) para `temp > 0`.
- **MTP em contexto longo**: medido só a 4K; a cache KV do bloco é f16 e custa
  8 MiB a 4K (2 MB por 1K de contexto), então a 64K são 128 MiB.
- **Ajuste do custo do passo de rascunho**: 2,28 ms para 334,7 MiB de pesos é
  ~147 GB/s efetivos, bem abaixo dos ~600 GB/s da placa — são 8 matvecs pequenos
  mais dois readbacks por passo. Se o verify batelado acontecer, esse custo passa
  a dominar e vale otimizar.
