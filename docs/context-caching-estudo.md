# Context caching (prefix KV-cache reuse) — estudo e proposta

> **Escopo.** Research + proposta, SEM implementação. **Nenhum comando de GPU foi
> executado** (sem lock, sem `bench`): tudo abaixo vem de leitura de código
> (pins `arquivo:linha`) ou de aritmética sobre medições já documentadas.
> Data: 2026-09-16. Único arquivo criado por esta tarefa (proibido tocar
> código, outros docs, `CMakeLists.txt`; sem commit).
>
> **Correção importante aos números do enunciado:** `478 tok/s` NÃO é o nosso
> prefill — é a **referência llama.cpp/Vulkan com coopmat desligado**
> (`docs/estudo-prefill.md:21`, `docs/estudo-prefill-b-mmq.md:446`). O nosso
> prefill-512 vigente é **451,1 tok/s best** (`docs/autotuning-gfx1201.md:540`).
> O §1.5 ancora todos os números usados no cálculo de ganho.

---

## 0. Veredito antecipado

- **Viável: SIM** — e mais barato aqui do que nas referências, pelo caso
  degenerado: servidor single-request + cache contíguo + `reset_state()` que já
  NÃO limpa o KV. O slice 1 ("last-prefix live cache") custa **zero VRAM extra,
  zero cópia**, e reaproveita a infra de snapshot do MTP.
- **Ganho esperado:** todo o prefill do prefixo compartilhado some:
  ~**1,1 s a cada 512 tokens** (451,1 tok/s), ~**11,6 s num prefixo de 4096**.
  Um chat multi-turno típico (prompt 512 + 32 decodes ≈ 2,0 s) cai para ~0,9 s
  (**~55%**). Decode inalterado (igual ao vLLM).
- **Maior risco:** exatidão bit-a-bit na continuação a partir de `L` arbitrário
  no regime GEMM (chunk 512 tem tolerância declarada, não bit-exatidão) +
  divergência silenciosa de tokenização nos turns re-renderizados.
- **Primeiro slice:** cache vivo de UMA entrada (última sequência) em
  `serve.hip` (`cached_ids` + estado GDN/KV vivo, sem cópias) com gate de
  bit-exatidão em `RD_PREFILL_CHUNK=16` antes de liberar no default 512.

---

## 1. Este motor hoje (verificado por leitura)

### 1.1 KV: contíguo, por camada, sem paging, uma sequência

`docs/kv-memoria-desenho.md:14-81` (lido na íntegra) estabelece o desenho:

- Duas alocações (`d_k_`, `d_v_`, `graph.cuh:285`), cada uma
  `n_full_attn × max_ctx × n_head_kv` linhas; endereço = base + slot de
  atenção + `t · n_head_kv · row_bytes + h · row_bytes` (`graph.cuh:694-711`
  write, `attn.cuh:119,142` read). Com `n_full_attn = 16`, `n_head_kv = 4`,
  `head_dim = 256`: linha de token = 2048 B (f16) / 576 B (q4_0).
- **Sem paging, block table, arena, seq-id ou realocação** — array
  `[camada][token][head][dim]` dimensionado uma vez por `max_ctx`.
- `Graph::reset_state()` (`include/rdna4/graph.cuh:2367-2382`, lido) só zera
  `d_state_`/`d_convst_` (GDN); **o KV nem é limpo** — a atenção é causal e
  cada linha é reescrita à medida que a sequência avança de 0.
- Tamanhos (`kv-memoria-desenho.md:90-106`): K+V total f16 @4K = 0,250 GiB;
  q4_0 @131K = **2,250 GiB**. VRAM medida: IQ3_S 4K/f16 = 11,87 GiB;
  131K/q4_0 = 13,86–13,99 GiB, **1,93 GiB livres, sem spill**
  (`docs/baseline-2026-09-16.md:46`). Placa: 15,92 GiB úteis.

### 1.2 Caminho do servidor: reset total por request, sem reúso

`src/server/serve.hip` (lido §§430-647) + `docs/servidor-openai.md:121-131`:

1. `Engine::build_prompt` (`serve.hip:449-469`): `chat_render` (template qwen35
   + thinking/reasoning_effort) → `tk_.encode(text, parse_special=true)` →
   `prompt_ids`. **O ponto de detecção de hit é aqui: a saída em ids.**
2. `Engine::generate` (`serve.hip:488-536`): `reset_state()` **incondicional**
   (`serve.hip:503-506`), depois `graph_->prefill(prompt_ids, 0, …)` com a
   política única de chunks (`serve.hip:512-526`).
3. Loop de decode token a token (`serve.hip:559-622`), uma request por vez,
   single-threaded (`serve.hip:25-29`).
4. Limite declarado: "No prefix/KV reuse between requests … Every request
   calls `Graph::reset_state()`" (`docs/servidor-openai.md:128-131`).

Divergência achada: `docs/servidor-openai.md:112-115` ainda diz "Prefill is
not batched … n_tokens × ~35 ms" — **stale**: o servidor usa
`Graph::prefill()` em lote desde a frente prefill-chunk (`serve.hip:526`).

### 1.3 Prefill em chunks (cap 512), com contrato de exatidão

- `prefill_chunk_cap()` (`include/rdna4/device.h:212-257`, lido): default
  **512** (flip 128→512 em `a176fd1`; `autotuning-gfx1201.md:541`), teto
  `kMaxChunkHost = 512` (`graph.cuh:200`, via `chunk-scale:81`).
- `Graph::prefill` (`graph.cuh:1941-1967`) corta o prompt pelo plano DP
  (`prefill_plan`, `graph.cuh:1890-1939`); chunks ≥ 2 vão a `forward_batch`,
  cauda de 1 token ao caminho por token.
- Contrato de exatidão (`device.h:217-250`, `check_batch_gpu.hip`): **n ≤ 16 é
  GEMV em lote, bit-exato** contra o caminho por token (M8:
  `medicoes-m8.md:23-34`); **n > 16 usa GEMM tilejado, NÃO bit-exato**
  (tolerância declarada `kTolChunkRelL2/kTolChunkMaxAbs`, PPL-neutro). Isto
  decide o gate do §4(f): a continuação a partir de um `L` arbitrário muda os
  pontos de corte dos chunks e, no regime GEMM, o arredondamento.

### 1.4 GDN: estado recorrente por camada + infra de snapshot JÁ existente

- Híbrido 64 blocos: `is_recr(il) = (il+1) % full_attention_interval != 0`
  (`graph.cuh:368-370`) → **48 camadas GDN + 16 de atenção plena**.
- Por camada recorrente: `d_state_` = `nvh·S·S = 48·128·128` floats =
  **3,00 MiB**; `d_convst_` = `(K-1)·chan = 3·10240` floats; total
  `d_state_` = **144 MiB** + `d_convst_` = **5,6 MiB** ≈ **150 MiB**
  (`kv-memoria-desenho.md:216-224`; `graph.cuh:2058-2068` `state_bytes()`).
- **Snapshot/restore device-to-device já implementado** para o MTP
  (`graph.cuh:2070-2120` `state_snapshot`/`state_restore`, buffers
  `d_state_snap_`/`d_convst_snap_`, ~0,5 ms por par apud `medicoes-m8.md:87`).
  É a peça que o servidor do llama.cpp chama de "context checkpoint" (§2.4) —
  aqui ela já existe e é testada (`check-mtp-gpu`).
- Recorrência processada em ordem dentro de cada chunk
  (`conv1d_state_batch_launch`, `gdn.cuh:405-437`) e o estado atravessa chunks
  via `d_state_`/`d_convst_` vivos — ou seja, **continuar um prefill a partir
  de `L` é mecanicamente o mesmo que uma fronteira de chunk**.

### 1.5 Números vigentes (âncoras — usar estes, não os do enunciado)

| workload (IQ3_S, RX 9070 XT) | número vigente | fonte |
|---|---|---|
| decode short-ctx | **37,9 tok/s best** (26,4 ms/tok) | `baseline:43`, `chunk-scale:40`, `autotuning:538` |
| decode @4K fim | 36,3 tok/s | `baseline:43` |
| decode @64K q4_0 | 22,3 tok/s (44,8 ms/tok) | `baseline:45` |
| **prefill-512** | **451,1 tok/s best** (2,217 ms/tok → **1,135 s/512**) | `autotuning:540` (pós-WMMA-int8) |
| prefill-512 pré-WMMA @cap512 | 411,22 tok/s (1,245 s/512) | `chunk-scale:38` |
| prefill-4096 @cap512 | 353,39 tok/s (**11,6 s/4096**) | `chunk-scale:39` |
| default chunk | 512 | `device.h:251`, `autotuning:541` |

O `478 tok/s ≈ 1,07 s` do enunciado é a referência **llama.cpp/Vulkan sem
coopmat** (`estudo-prefill.md:21`; `estudo-prefill-b-mmq.md:446,
`estudo-prefill-a-vulkan.md:49`), não medição nossa. Decode `39 tok/s` ≈
nosso 37,9 (ruído + cap) — aceitável.

---

## 2. llama.cpp hoje (checkout local — mecanismo real, não docs)

Leitura direta de `tools/server/server-context.cpp`, `server-common.cpp`,
`server-task.{h,cpp}`, `src/llama-kv-cache.cpp`, `src/llama-memory-*.cpp`,
`common/arg.cpp`. Todas as linhas abaixo foram lidas (não inferidas).

### 2.1 Reúso de prefixo por slot: match de sequência inteira, granularidade = 1 token

No `SLOT_STATE_STARTED` (`server-context.cpp:3142-3426`):

- Hit detection = **`get_common_prefix(input_tokens)`**
  (`server-context.cpp:3219`), comparação id-a-id, exata, **a partir da
  posição 0**, implementada em `server-common.cpp:697-745` (ramo sem imagem:
  `:700-710`, loop linear que retorna o primeiro índice divergente).
- `n_past` = tamanho do prefixo comum; `keep_first(n_past)`
  (`server-common.cpp:653-684`) trunca o registro; `seq_rm(slot.id, p0, -1)`
  (`server-context.cpp:3444`) apaga as células além de `p0`; só o **sufixo**
  entra no batch. `cache_prompt=false` força `n_past = 0` (`:3286-3289`).
- **Granularidade: token individual, mas SOMENTE prefixo desde 0** — não há
  hashing por bloco nem reúso de interior (isso é o `--cache-reuse`, §2.2).
  Miss total = re-prefill integral; não há "meio-termo" fora do prefixo.
- Garantia de progresso: se `n_past == n_tokens`, decrementa 1 ("need to
  evaluate at least 1 token", `:3402-3407`); estatística `n_prompt_cached`
  (`:3409-3412`, visível como `cache_n` na API).
- Flag: `--cache-prompt/--no-cache-prompt`, **default ligado**
  (`common/arg.cpp:3561-3567`); por request `cache_prompt` (`server-task.h:53`,
  `server-schema.cpp:31`). Caveat declarado: com cache, logits **não** são
  garantidos bit-a-bit (batch sizes diferentes) (`tools/server/README.md:587`).

### 2.2 `--cache-reuse N`: reúso de chunks internos via KV shifting

- `n_cache_reuse` (`common/arg.cpp:3568-3577`, default 0 = desligado;
  `server-task.h:65`): varre cache × prompt procurando runs idênticos com
  `n_match >= n_cache_reuse` e **desloca o KV** para a nova posição
  (`server-context.cpp:3237-3285`): `seq_rm(id, head_p, head_c)` +
  `seq_add(id, head_c, head_c+n_match, kv_shift)` (`:3267-3270`).
- Só se `llama_memory_can_shift()` (`:3229-3235`): o K-shift exige RoPE
  re-aplicável — `llama_kv_cache::get_can_shift()` (`src/llama-kv-cache.cpp:
  1188-1197`) recusa Step35 e `n_pos_per_embd > 1`; o shift roda um grafo
  dedicado (`:857-893`). Multimodal recusa (`:3244-3246`, `:1191-1193`).
- `seq_rm`/`seq_add` do KV denso operam por **célula (seq, pos)**
  (`llama-kv-cache.cpp:382-449`, `:570-618`): remover = soltar células no
  intervalo; deslocar = `pos_add` nos metadados (sem mover bytes de KV).

### 2.3 `n_keep`: política de overflow (context shift), não de cache

- Quando o slot estoura: mantém os primeiros `n_keep` tokens (`-1` = todos),
  descarta `n_discard` (default = metade do resto) e desloca o resto para
  baixo (`server-context.cpp:2934-2968`, com reescrita do vetor em
  `:2956-2968`). É **evicção com perda**, ortogonal ao reúso: este motor
  responde 400 em overflow (`servidor-openai.md:141-156`), então `n_keep`
  não transfere — mas o par `(seq_rm, seq_add)` é o vocabulário que
  transferiríamos se um dia fizéssemos shift.

### 2.4 Recorrentes/SWA: checkpoints de estado (a parte que nos diz respeito)

Este é o ponto que o híbrido GDN torna obrigatório — e o llama.cpp já
resolveu para Mamba/RWKV/SWA:

- `server_prompt` carrega `tokens + list<common_prompt_checkpoint>`
  (`server-task.h:566-586`); checkpoints criados durante o decode
  (`create_checkpoint`, `server-context.cpp:2309-2372`) com teto
  `--ctx-checkpoints N` (default **32**) e espaçamento `--checkpoint-min-step`
  (default **8192**) (`tools/server/README.md:165-166`).
- No hit parcial, o servidor procura o checkpoint com
  `pos_min < pos_min_thold` e o **restaura** (`server-context.cpp:3349-3376`
  `load_tgt/load_dft`); **sem checkpoint válido, força reprocessamento total**
  (`:3379-3384`, com link para a discussão do PR #13194).
- Por quê: `llama_memory_recurrent::seq_rm` (`src/llama-memory-recurrent.cpp:
  161-247`) documenta que Mamba/RWKV **não apagam estado parcial no fim** —
  rollback só via índice de snapshot por token, limitado a `n_rs_seq`
  (`:194-203`); fora disso, **recusa** (`return false`). E
  `llama_memory_hybrid::seq_rm` (`src/llama-memory-hybrid.cpp:143-149`, via
  grep) tenta o recorrente primeiro — se ele recusa, nada anda. O `seq_add`
  recorrente só desloca `pos` (`llama-memory-recurrent.cpp:318-346`;
  `get_can_shift() == true`, `:716-719`).
- `README-dev.md:58`: "`server_prompt_checkpoint`: para modelos recorrentes
  (ex. RWKV) e SWA, guarda snapshots do estado do KV cache. Permite reúso
  quando requests seguintes partilham o mesmo prefixo."

Tradução para nós: **KV é endereçável por posição, GDN só por snapshot de
prefixo inteiro** — exatamente a semântica do §4(b).

### 2.5 Arquivo `--prompt-cache`, seleção de slot, evicção

- `--prompt-cache FNAME` (+ `--prompt-cache-all`, `--prompt-cache-ro`) é
  **mecanismo de CLI/sessão via `llama_state_load_file`**
  (`common/arg.cpp:1864-1883`; `tools/completion/completion.cpp:206-226`) —
  persistência de startup, não cache entre requests. Não transfere para o
  servidor (o nosso problema é inter-request em VRAM, não disco).
- Seleção de slot por similaridade LCP (`server-context.cpp:1560-1610`):
  `f_sim = lcp/novos`, limiar `slot_prompt_similarity`; se `f_keep < 0,5`
  (vai perder > metade do contexto), salva o estado no prompt-cache antes.
  Fallback = LRU por `t_last_used` (`:1612-1628`). Irrelevante aqui
  (temos 1 slot), mas o **critério `f_keep`/`f_sim`** é bom vocabulário para
  logs e métricas.
- Prompt-cache entre requests (`server-task.cpp:1711-1832`): dedup
  (prompt contido → skip, `:1713-1720`), remoção de obsoletos (`:1737-1748`),
  **FIFO por tamanho** até `limit_size` (`:1750-1758`), `bad_alloc` encolhe o
  limite (`:1763-1777`); no load, exige `f_keep_cur >= 0,25` e maximiza
  `f_keep × f_sim` (`:1804-1823`).

### 2.6 Defrag: não há mais (cells + head pointer)

Só 3 ocorrências de "defrag" no `src/` (grep): `llama-kv-cells.h:105`
(move de célula, legado), `llama-context.cpp:3641`
(`defrag_thold = -1.0f`, desligado). O cache atual é cell-based com
reaproveitamento via `head` (`llama-kv-cache.cpp:403-420`) + K-shift — **não
há compactação copiando KV**. Bom precedente: contiguidade sem defrag é
sustentável; nós sequer precisamos do head pointer (uma sequência).

### 2.7 Resumo llama.cpp — o que é e o que não é

| aspecto | comportamento real | pin |
|---|---|---|
| match | prefixo exato de ids desde pos 0, granularidade 1 token | `server-common.cpp:697-710`; `server-context.cpp:3219` |
| hit | pula prefill dos `n_past`, avalia só o sufixo | `server-context.cpp:3414-3444` |
| miss | `n_past=0`, prefill integral | `server-context.cpp:3286-3289` |
| interior | só com `--cache-reuse N` via K-shift (desligado por default) | `server-context.cpp:3237-3285`; `arg.cpp:3568-3577` |
| recorrente | só via checkpoint restaurado; senão, reprocessa tudo | `server-context.cpp:3349-3384`; `llama-memory-recurrent.cpp:182-203` |
| overflow | `n_keep` + meio-resto descartado + shift | `server-context.cpp:2934-2968` |
| evicção | LRU de slots; prompt-cache FIFO por bytes | `server-context.cpp:1612-1628`; `server-task.cpp:1750-1758` |
| exatidão | reúso NÃO promete bit-exatidão | `tools/server/README.md:587` |

---

## 3. vLLM APC + SGLang RadixAttention (web)

Fontes: [design doc APC do vLLM](https://docs.vllm.ai/en/stable/design/prefix_caching/),
[explicação operacional de APC](https://packet.ai/blog/vllm-prefix-caching),
[post RadixAttention/SGLang (LMSYS)](https://www.lmsys.org/blog/2024-01-17-sglang/),
[paper SGLang](https://arxiv.org/pdf/2312.07104),
[RadixAttention vs PagedAttention](https://llm-academy.dev/inference/sglang-radixattention/).

### 3.1 vLLM Automatic Prefix Caching

- KV gerido em **blocos fixos (16 tokens, default)** pelo block manager
  (PagedAttention); cada bloco é identificado por um **hash encadeado**:
  hash(tokens do bloco + hash do bloco pai) — hit exige **match exato
  token-a-token de todo o prefixo até aquele bloco**.
- **Só blocos cheios entram no cache** — prefixo que termina no meio do bloco
  não ganha crédito parcial. Implicação de template: conteúdo estático
  primeiro, variável por último (um token divergente cedo quebra a cadeia).
- Os blocos paginados fazem dupla função (endereçamento + chave de cache),
  com refcount/copy-on-write; **evicção LRU preferindo caudas de prefixos
  longos** (preserva prefixos curtos compartilhados); hash configurável
  (`--prefix-caching-hash-algo`: sha256 default, xxhash rápido);
  `cache_salt` por tenant; métricas `prefix_cache_queries/hits`; **ligado por
  default no V1**; só acelera prefill, nunca decode.

### 3.2 SGLang RadixAttention

- **Radix tree** (CPU) mapeando sequência de tokens → tensores de KV (GPU,
  layout paginado; página = 1 token no desenho original, `--page-size 16`
  hoje): busca/inserção/evicção de prefixo em uma estrutura só, com
  **splits de nó** quando duas sessões divergem após prefixo comum.
- **Evicção LRU de folhas** + refcount anti-evicção de nós em uso; **cache-aware
  scheduling** (ordena o batch para maximizar reúso; roteia request ao worker
  que já tem o prefixo); frontend manda o prompt cheio, runtime faz
  match/reúso/cache automaticamente; até **5× throughput** em programas
  multi-chamada (few-shot, self-consistency, chat multi-turno, ToT); **sempre
  ligado** (overhead ~nulo no miss).

### 3.3 O que transfere × o que assume paging que não temos

| ideia | transfere? | porquê |
|---|---|---|
| hash encadeado por bloco de 16 | **parcial** — como *índice*, não como unidade de reúso | sem block table, "reusar bloco k" = manter linhas `[k·16,(k+1)·16)` onde já estão (custo zero); o hash serviria p/ cache multi-entrada (§4c) |
| full-blocks-only / estático-primeiro | **sim** — guia de template | chat_render já põe system primeiro; documentar "não pôr IDs/timestamps antes do system" |
| LRU + refcount + métricas queries/hits | **sim** | baratos em CPU; `n_cached` por request no log `-v` |
| radix tree com granularidade fina | **não** — colapsa numa cadeia | §4(b): sem estados GDN por bloco, nós intermediários da árvore seriam inutilizáveis; 1 cadeia (última sequência) cobre o caso |
| cache-aware scheduling / routing | **não** | 1 request por vez, 1 worker, 1 GPU |
| copy-on-write de blocos | **não precisa** | 1 sequência: nunca há fork; manter linhas no lugar é o CoW degenerado |
| sempre-ligado | **sim, como meta** | miss custa só um `memcmp` de ids em CPU |

A intuição comum aos três sistemas que VALE aqui: **hit = prefixo exato de
ids desde 0; miss = recomputa do ponto de divergência; recorrência exige
snapshot posicional, nunca reconstrução por bloco**.

---

## 4. Proposta para o rdna4-infer

Princípio: somos o caso degenerado mais favorável — 1 sequência, cache
contíguo causal, `reset_state()` que já preserva o KV, snapshot GDN pronto.
O desenho imita o llama.cpp §2.1/§2.4, não o vLLM/SGLang (sem paging, sem
árvore).

### (a) ONDE o hit é detectado

- **Entrada:** `prompt_ids` (saída de `Engine::build_prompt`,
  `serve.hip:449-469` — pós-template, pós-`encode(parse_special=true)`).
  Comparar **ids, nunca texto** (imune a re-tokenização divergente: vira
  apenas `L` menor, §4f).
- **Estrutura:** `PrefixCache` em CPU no `Engine` (serve.hip): guarda
  `cached_ids: vector<int32_t>` da última sequência aceita (+ hash das
  configurações que afetam ids/aritmética, §4f). Hit =
  `L = longest_common_prefix(cached_ids, prompt_ids)`, O(n) `memcmp`-like —
  microssegundos contra segundos de prefill.
- **Limiar:** reusar se `L >= kMinReuse` (ex. 32 tokens; abaixo disso o
  bookkeeping não paga nem o ruído). `L == prompt.size()` (prompt idêntico,
  ex. retry): cai na regra "avaliar ≥1 token" do llama.cpp (`:3402-3407`) —
  re-prefilla só o último token para obter logits frescos.
- Vale para **ambos** os endpoints (`/v1/chat/completions` e
  `/v1/completions`): os dois afunilam em `build_prompt → generate`.

### (b) O QUE é cacheado e reusado — o crux GDN

Estado vivo ao fim do request N (sem `reset`, sem cópia): KV com as linhas
`[0, T)` válidas em todas as 16 camadas + `d_state_`/`d_convst_` =
**estado-GDN-exatamente-após-T**. O request N+1 com prefixo comum `L ≤ T`:

**PODE reusar (correto por construção):**

1. **Linhas KV `[0, L)` por camada** — ficam onde estão; o prefill do sufixo
   começa em `start_pos = L` e a causalidade reescreve `[L, …)` por cima.
   Custo: zero bytes movidos (o CoW degenerado do §3.3).
2. **Estado GDN+conv exatamente em `L`, SE `L == T`** (caso multi-turno
   canônico: novo prompt = prompt anterior + resposta anterior + novo turno,
   e a resposta foi gerada por NÓS — o estado vivo ao fim de N é
   exatamente o estado após `T = |P|+|R|`). Custo: zero — é só NÃO chamar
   `reset_state()` e chamar `prefill(sufixo, start_pos=L)`.
3. **`L < T` (divergência no meio, ex. usuário editou turno antigo):**
   KV `[0, L)` ainda vale (endereçável por posição, como no llama.cpp §2.1),
   mas o GDN vivo está em `T`, não em `L`. Opções, em ordem de custo:
   - (i) **snapshot sob demanda**: tirar `state_snapshot()` (~150 MiB D2D,
     ~0,5 ms apud M8) em fronteiras baratas durante prefill/decode do
     request N (ex. ao fim do prefill + a cada K tokens de decode) e, no hit
     `L < T`, `state_restore()` do snapshot com maior `t ≤ L` + re-prefill
     de `[t, L)` (poucas dezenas de tokens no pior caso). É o
     `n_ctx_checkpoints`/`checkpoint-min-step` do llama.cpp (§2.4) com
     números nossos: 150 MiB por snapshot em VRAM é caro (×32 impensável) —
     usar **anel de 2–4 snapshots** (300–600 MiB; cabe folgado em ctx ≤ 16K,
     proibir em 131K) ou **offload para host** (150 MB / ~25 GB/s ≈ 6 ms,
     ainda ≪ segundos de prefill).
   - (ii) fallback: `L < T` sem snapshot cobrindo → trata como `L' = 0`
     (reset + prefill integral). Simples, sempre correto; o caso canônico
     (`L == T`) — que é ~todo o tráfego de chat — nunca precisa disso.

**NÃO PODE reusar (e porquê):**

4. **Blocos interiores estilo vLLM** (match no meio sem match do início):
   o KV até permitiria, mas o GDN é `S_t = f(S_{t-1}, x_t)` — função de TODO
   o prefixo, sem inverso nem endereçamento por token. Reusar o bloco
   `[a, b)` exigiria `S_a`, que só existe se foi snapshoteado quando `a`
   era o presente. É exatamente por isso que o llama.cpp **recusa**
   `seq_rm` parcial recorrente (`llama-memory-recurrent.cpp:182-203`) e
   força reprocessamento total sem checkpoint (`server-context.cpp:3379-3384`).
   **Regra de ouro: granularidade de reúso do GDN = prefixo-desde-0 com
   snapshot; qualquer outra granularidade é incorreta, não apenas lenta.**
5. **Across mudança de regime aritmético/tipos:** `RD_PREFILL_CHUNK`,
   `--cache-type-k/v`, `--ctx-size`, pesos, template/thinking settings —
   qualquer um invalida `cached_ids` (limpar cache; é um `clear()` de vetor,
   custo zero).
6. **Across processos:** sem persistência em disco no slice 1 (o
   `--prompt-cache` do llama.cpp é sessão de CLI, §2.5 — não é o nosso
   problema).

### (c) Evicção + orçamento VRAM (o penhasco de 16 GB)

- **Slice 1 (proposto): UMA entrada viva = evicção trivial "último request
  vence".** Custo VRAM extra: **0** (linhas KV já alocadas para `max_ctx`;
  GDN vivo já existe; `cached_ids` em host). Prompts que não cabem continuam
  400 (`servidor-openai.md:141-156`) — o cache nunca cresce com o contexto.
- **Snapshots (só se o slice 2 ligar `(b)(i)`):** anel device de 2–4 ×
  150 MiB = 300–600 MiB. Cabe em ctx 4K (11,87 + 0,6 ≪ 15,92) e 16K; em
  64K/131K q4_0 (13,86–13,99 GiB, 1,93 livres) o anel device NÃO cabe →
  regra: snapshots só em host (6 ms/restore) ou só `L == T` (custo zero).
  O KV q4_0 já é a alavanca de VRAM (`kv-memoria:372-376`: q4_0 economiza
  bytes, não tempo) — o cache não muda essa conta.
- **Slice 3 (futuro): N entradas com offload.** Cada entrada fria =
  `cached_ids` (host, KB) + snapshot GDN (host, 150 MB) + linhas KV do
  prefixo (host, até 2,25 GiB @131K q4_0; ~90 ms de H2D — ainda ≪ prefill
  de 131K). Evicção = LRU por bytes com `limit_size`, copiando a política
  `server-task.cpp:1750-1758` (FIFO + `f_keep ≥ 0,25` no load). NÃO propor
  agora: o slice 1 captura o tráfego real (1 usuário, 1 conversa).

### (d) Superfície de API

- **Automático por default** (tese SGLang: miss ≈ grátis), com escape:
  `--prefix-cache / --no-prefix-cache` no `serve` (default: off até o gate
  §4f passar; on depois). Sem `cache_salt`/multi-tenant (servidor local,
  1 usuário) — documentar a ausência como decisão, não lacuna.
- **Observabilidade:** log `-v` por request com `prompt_n`, `n_cached=L`,
  `suffix`, `prefill_s` (espelho do `cache_n`/`timings` do llama.cpp,
  `server-context.cpp:3409-3412`); erro 400 e formato de erro inalterados.
- **Sem mudança no wire protocol:** nenhum campo novo no JSON (o
  `cache_prompt` por-request do llama.cpp pode vir depois, se pedirem).

### (e) Ganho esperado, quantificado dos NOSSOS números

Prefill poupado = `L / throughput_prefill`; decode inalterado.

| prefixo compartilhado L | segundos poupados (451,1 tok/s) | contexto |
|---|---|---|
| 512 | **~1,1 s** | turno típico c/ system+histórico curto |
| 2048 | **~4,5 s** | conversa média |
| 4096 | **~11,6 s** (medido: 353,39 tok/s) | conversa longa / RAG |
| 8192 (default `--ctx-size` do serve) | **~18–23 s** (estimativa: usa 353–451/tok/s como banda) | teto do ctx default |

Exemplo ponta-a-ponta (números vigentes §1.5): request com prompt 512 +
30 decodes hoje = 1,14 s + 0,79 s ≈ **1,9 s**; com hit `L = 512` =
~0 + 0,79 s ≈ **0,8 s** (**~58%**). Em 4096 (11,6 s + decodes), o hit tira
**> 85%** da latência. Decode nunca melhora (prefill-only, como no vLLM).
Custo do miss: um scan O(n) de ids em CPU (< 0,1 ms) — por isso o default-on
é defensável após o gate.

### (f) Riscos + gates (a porta de entrada é a bit-exatidão)

**Gate G0 (bloqueante): bit-exatidão do prefixo reusado = logits idênticos
ao prefill integral.** Procedimento: `RD_PREFILL_CHUNK=16` (regime GEMV,
bit-exato por `medicoes-m8.md:23-34`) primeiro — hit deve dar
`rel-L2 = max|d| = 0` via `check-graph-gpu` (oráculo por nó) +
`scripts/check_golden_run.sh` + `compare_llama_greedy.sh`; depois repetir no
default 512 dentro das tolerâncias declaradas
(`kTolChunkRelL2/kTolChunkMaxAbs`, `check_batch_gpu.hip`) + PPL
(`check_attn_split.sh`, banda 0,5%) — porque continuar de `L` arbitrário
muda os cortes do `prefill_plan` e o GEMM arredonda por corte. O llama.cpp
**declina** essa promessa (`README.md:587`); nós devemos **exigi-la** no
slice 1 (regime 16) e **delimitá-la** no 512.

Riscos residuais e mitigação:

| # | risco | severidade | mitigação |
|---|---|---|---|
| R1 | RoPE/posição do sufixo errada (`start_pos=L`) | alta | posições já são upload por chunk (`medicoes-m8.md:13`); G0 cobre |
| R2 | carry recorrente escondido fora de `d_state_`/`d_convst_` (scratch por chunk que vaza entre chunks) | alta | auditoria de `forward_batch_layer` (`graph.cuh:1411+`): `d_qkb_`/`d_vcb_` são scratch por chunk (re-escritos, não carry); G0 com `L` desaliado de fronteira cobre |
| R3 | re-tokenização dos turns do assistente ≠ ids gerados (fronteiras BPE entre roles) | média | seguro por construção (compara ids; só encolhe `L`); quantificar `L` real em tráfego de chat no gate |
| R4 | settings que mudam ids/logits sem mudar o texto (thinking on/off, template kwargs, seed não — seed não afeta prefill) | média | hash de settings na chave; `clear()` em qualquer mudança de flag |
| R5 | `L==T` mas resposta anterior parou por `stop`/truncamento e o template re-renderiza diferente | baixa | idem R3 (ids decidem) |
| R6 | regressão de VRAM em 131K se ligarem snapshots device | média | regra §4c: anel device proibido em 64K+; só host ou só `L==T` |
| R7 | futura concorrência/batching quebra a invariante "1 sequência viva" | futura | documentar: batching exige block-table ou cache por slot (voltaria o §3.3) |

Ordem de implementação (fatiado, cada fatia com gate):

1. **Slice 1 — `L == T` vivo, sem cópias:** `cached_ids` + `cached_len` no
   `Engine`; `generate()` só chama `reset_state()` se `L < kMinReuse`;
   `prefill(sufixo, L)`; log `n_cached`; G0 em chunk 16, depois 512.
2. **Slice 2 — `L < T`:** anel de 2–4 snapshots (ou host-offload) nos fins de
   prefill; `state_restore` + re-prefill `[t, L)`; G0 idem.
3. **Slice 3 (se o tráfego pedir) — N entradas frias em host** com LRU por
   bytes; métricas queries/hits.

---

## 5. O que NÃO foi verificado (sem GPU por proibição expressa)

1. **Nenhum número novo foi medido.** Todos os tok/s, ms/tok, GiB e
   contagens de lançamentos são citações de `baseline-2026-09-16.md`,
   `chunk-scale-2026-09-16.md`, `autotuning-gfx1201.md:538-541`,
   `kv-memoria-desenho.md` e `medicoes-m8.md`. Em particular, "0,5 ms por
   snapshot" vem de `medicoes-m8.md:87` (contexto MTP), não de medição de
   cache.
2. **Continuação a partir de `L` arbitrário nunca foi executada** — a
   equivalência "fronteira de hit = fronteira de chunk" (§1.4) é argumento
   por leitura de `graph.cuh:1411+`/`gdn.cuh:405-437`, não teste. O G0 existe
   precisamente porque isto é hipótese.
3. **Hit-rate em tráfego real é desconhecida** — `L` típico em chats
   multi-turno com o template qwen35 + thinking não foi amostrado (exige
   servidor instrumentado + GPU).
4. **Largura de banda host↔device** (~25 GB/s usada na estimativa de 6 ms e
   ~90 ms do §4c) é ordem de grandeza PCIe, não medição desta placa.
5. **Pins llama.cpp** são do checkout em `/home/marcelo/Projetos/llama.cpp`
   no estado de hoje — sem pin de commit ( iceberg: `git log` não consultado
   para não tocar nada fora do escopo de leitura; todos os arquivos foram
   só lidos).
6. **vLLM/SGLang**: resumidos das fontes linkadas (§3), sem leitura de código
   nem reprodução de benchmarks (o "5×" é do post LMSYS em A10G/Llama-7B —
   workload multi-chamada, não comparável 1:1 ao nosso decode-bound).
7. **Divergência achada mas não perseguida:** `servidor-openai.md:112-115`
   (prefill não-batelado) está stale face a `serve.hip:526` — registrado no
   §1.2 para o dono do doc, sem tocar no arquivo.
