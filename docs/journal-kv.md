
---

## 8. Gates pós-merge (obrigatório: o merge preventivo com a `main`)

O merge com a `main` (frente de prefill `264044e`, argmax no device, teto de
`--ctx-size`, `check_all.sh`, `gpu-lock` aninhável) foi **limpo em texto — nenhum
conflito** — mas deixou **um buraco real que só aparece lendo o resultado**:

**`attn_batch_launch` (`attn.cuh`, da frente de prefill) tem a sua PRÓPRIA lista de
pares mantida à mão** (16 pares, o 4×4 menos espelhos, sem Q5_0/Q4_1). O chamador em
`graph.cuh:1040` trata `false` como **erro duro** ("batch attn (batched) launch
failed") e não há fallback: com `--cache-type-k q5_0` o prefill em lote **abortaria**,
não degradaria. As duas mudanças eram corretas isoladas e incompletas juntas.
Corrigido restaurando o produto 6×6 completo (commit `a1fc8bd`). Verifiquei também
`Graph::kv_write_batch` (`graph.cuh:796`), que quantiza linhas em lote com
`kv_store_row_launch(kv_k_, ..., n_tok*NKV*HD)`: é correto para os formatos novos
porque o quantizador é puramente local ao bloco de 32 e toda linha tem 8 blocos
inteiros, então o layout plano coincide com o por linha. Não precisou de mudança.

**Comando**: `./scripts/gpu-lock.sh bash /tmp/kv-gates.sh` (timeout **dentro** do
lock, regra 1.2 corrigida). Todos os gates rodaram sobre o binário **pós-merge e
pós-conserto** (rebuild completo, 04:34).

| gate | resultado |
|---|---|
| `check-kvtype` (novo, CPU) | **OK** |
| `check_attn_split.sh` | **OK** — unsplit 5,1989 / split 5,2054 (0,125%) e a ponta larga 5,2114 (0,240%), limites de 0,5% |
| `check_golden_run.sh` | **OK** — inclui `kv q4_0: 130 bytes`, `kv f16/q8_0: ok`, `kv q8_0/f16: ok`, `kv f32/q4_0: ok` |
| `check_regression.sh` | **OK** — 7 casos, **7 ids bit-exatos**, `rel-L2 0.00e+00` em todos (o gate numérico mais forte) |
| `check-graph-gpu` (f16/f16, oráculo por nó) | **PASS** |
| `check-batch-gpu` | **não rodou**: invoquei sem o argumento do modelo (`usage: check-batch-gpu <file.gguf>`), erro meu de script, não um gate vermelho |

Ou seja: **f16, q8_0 e q4_0 não mudaram** — `check_regression` compara ids gerados e
logits contra o golden commitado e dá bit-exatidão nos 7 casos, e o
`check_golden_run` cobre explicitamente as combinações mistas antigas.

**Janela**: o lock me foi entregue às 04:42:33 com `vram_used` = 10 842 058 752 B
(10,1 GiB) — resíduo do dono anterior ainda drenando, não um processo concorrente.
Registro porque a regra 1.6 pede: **estes gates são comparações determinísticas
contra arquivos golden commitados, não medidas de tempo**, então o resíduo de VRAM
não os invalida. (Uma medida de tok/s nessas condições seria lixo; duas delas
naquela janela não existem neste diário.)
