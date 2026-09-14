# ninfer: o que um motor CUDA *sem* cuBLAS, *sem* tensor core em lote pequeno e *com* GDN em blocos faz que nós não fazemos

Estudo de código, 14/09 (leitura pura, sem GPU). Fonte: clone raso de
`https://github.com/Neroued/ninfer` em `/tmp/ninfer`, HEAD `d492968`
(`perf(runtime): improve materialization search and adapt planning budgets`). Todas as
referências de arquivo foram lidas nesta sessão; nada aqui vem do README de qualquer dos dois
projetos sem estar confirmado no código. O que é dedução está marcado `INFERIDO` com o que
confirmaria.

**Convenção de citação, para não haver ambiguidade entre os dois projetos** (os dois têm
`README.md` e os dois têm `docs/`):

| forma | dono |
|---|---|
| `arquivo.ext:linha` sem prefixo (ex. `q4_dispatch.cpp:7`) | **ninfer**, relativo a `/tmp/ninfer` |
| prefixado com `ninfer ` (ex. `ninfer README.md:12`, `ninfer docs/maintainer/...`) | **ninfer** |
| prefixado com `nosso ` (ex. `nosso README.md:498`) | **este repo** |
| `include/rdna4/...`, `src/...`, `docs/estudo-*`, `docs/plano-*`, `docs/journal-*`, `scripts/...` | **este repo** |

Contexto que este documento assume (medido, não re-derivado aqui): o nosso prefill está
**8,54×** atrás do llama.cpp Vulkan, decomposto em **2,50× micro-lote × 1,38× estrutura ×
2,47× unidades de matriz** (`docs/estudo-prefill.md` §53-67, `docs/plano-prefill.md` §10-17);
o staging do GEMM tilejado é **50-61 %** do tempo do kernel e a correção de escala é
**53-56 %** do miolo (`docs/plano-prefill.md` §244-249); o chunk é `kMaxBatch = 16`
(`include/rdna4/graph.cuh:191`); o GDN é **0,852 ms/token** (`docs/plano-prefill.md` §100); a atenção
relê o KV **6×** por token e roda a **93-97 %** da banda do Infinity Cache
(`nosso README.md:498-507`).

---

## TL;DR (5 linhas)

1. **ninfer é C++20/CUDA de 331 275 linhas, `sm_120a` obrigatório, um RTX 5090, cinco
   artefatos Qwen3.6/3.8** (27B dense e 35B-A3B MoE), com 476 arquivos e 64 303 linhas só
   na camada de kernels (`src/ops`) — e **zero cuBLAS, zero CUTLASS, zero Triton**: todos os
   matmuls são escritos à mão (`grep -rn "cublas\|cutlass" src include apps` → **0**
   ocorrências).
2. Ele faz **3 218 tok/s de prefill** num 27B a 7 680 tokens e **1 614,8** a 260 096 tokens
   (`ninfer docs/performance/qwen3.6-27b.md:47-52`) **sem `wgmma`, sem `tcgen05`, com `mma.sync`
   em 6 instruções num único arquivo** (`src/ops/common/mma.cuh:36,45,53,62,71,89`) — ou
   seja, o alvo que perseguimos é alcançável na geração de instruções que o gfx1201 *tem*.
3. **Top 5 a roubar**, cada um com o gargalo nosso que ele ataca:
   (a) **dequantizar LDS→registrador dentro do laço de consumo, nunca LDS→LDS**, com a
   escala dobrada sobre um tile de colunas — `q4_rowsplit_gemm_simt.cuh:112-159` — ataca
   **staging + correção de escala**, que é **50-61 %** do nosso kernel de GEMM e **53-56 %**
   do miolo (`docs/plano-prefill.md:244-249`);
   (b) **`ColsPerTile ≤ 8` com `static_assert`** e a paralelização em `RowsPerCta` (warps),
   não em acumuladores por thread — `q4_rowsplit_gemm_simt.cuh:51-52`, 8 704 B de LDS —
   é a regra de projeto que faltou ao nosso D1, que subiu N e foi de 62 para **133 VGPRs**
   (`docs/plano-prefill.md:29-40`), e o limiar dele confirma o nosso: **`T ≤ 16 → SIMT,
   T > 16 → unidade de matriz`** (`q4_dispatch.cpp:14-17`);
   (c) **um lançador compilado por valor exato de T** (array `constexpr` para T = 2..8 e
   2..20, `q4_small_t_mma.cu:14-16,34-44`) com **K dividido entre 8 warps e redução na LDS**
   (`q4_small_t_mma.cuh:32-38,61,82-90,201-260`) — ataca **micro-lote e MTP**, isto é o
   coeficiente **6,04 ms/token** do nosso `pass_ms ≈ 13,9 + 6,04·N`;
   (d) **GDN em blocos de 64 com três estágios paralelos** (`kChunkSize = 64`,
   `gated_delta_net/common.h:8`; `prepare_wy_wu` → `state_passing` → `output`,
   `chunked/launch.cu:31-73`) — ataca a **recorrência do GDN (0,852 ms/token)**, que no
   degrau D4 passa a ser ~52 % do que sobra;
   (e) **um CTA por cabeça de KV, dono de todas as cabeças de consulta do grupo** — grade
   `(KVHeads, splits, batch)` com `kv_head = blockIdx.x` (`small_t.cu:102`,
   `small_t_bf16.cuh:53`) e o teto `TokenTile · GroupSize ≤ 48` (`small_t_bf16.cuh:27`) —
   ataca a **releitura de 6× do KV no decode** (**25,77 GB** lógicos contra **4,295 GB**
   únicos a 64K, `nosso README.md:498-507`).
   *Menções que não entram no top 5 mas têm mecanismo próprio:* **PDL** em 28 pontos de 9
   arquivos (`src/core/pdl.cuh:40,43`, ataca as nossas 1 940 partidas/token) e a **política
   de split em degraus** com teto **85** contra o nosso teto **16** (`small_t.cu:38-41`,
   `geometry.cuh:12`; nosso: `include/rdna4/graph.cuh:432-445`,
   `include/rdna4/tuning.h:102`) — §3.6 e §3.7.
4. O que **não** sobrevive: `cp.async`, `ldmatrix`, `mma.sync`, TMA, PDL e `cub::BlockMergeSort`
   não existem no gfx1201 — mas em todos os casos o **mecanismo** (pipeline de estágios
   separando emissão de consumo, LDS com swizzle XOR, tile de registro em N, K dividido
   entre warps) é o que carrega o ganho, e esses sobrevivem (§3).
5. A descoberta mais desconfortável é que o ninfer tem **duas** estruturas de atenção, e
   elas são opostas: no **decode** um CTA é dono de **uma cabeça de KV e de todas as cabeças
   de consulta do grupo** — `kv_head = blockIdx.x` (`small_t_bf16.cuh:53`), grade
   `(KVHeads, splits, batch)` (`small_t.cu:102`), linhas mapeadas por
   `q_head = kv_head * GroupSize + local_q` (`small_t.cuh:157-163`), com o comentário de
   projeto explícito *"A CTA owns one KV head and all GQA query heads, so each persistent
   K/V byte is streamed once"* (`small_t_k8v4.cuh:3-4`). **É exatamente a fusão de GQA que o
   `nosso README.md:498-507` lista como ganho não reclamado.** No **prefill**, ao
   contrário, ele volta a uma cabeça de consulta por CTA e relê 6× (`prompt_bf16.cuh:103`)
   — logo o nosso problema de prefill é dele também. E a razão 6:1 é a mesma
   (`geometry.cuh:15`: 24 cabeças de consulta / 4 de KV, igual ao nosso `n_head_kv = 4`).

---

## 1. Mapa da arquitetura

### 1.1 Tamanho e forma

| medida | valor | onde |
|---|---:|---|
| linhas de código rastreadas (`.cu/.cuh/.cpp/.h/.hpp/.py`) | 331 275 | `git ls-files` + `wc -l` |
| arquivos `.cu` / `.cuh` / `.cpp` / `.h` | 216 / 127 / 292 / 407 | idem |
| `src/ops` (kernels) | 476 arquivos, 64 303 linhas | idem |
| `src/targets` (execução do modelo) | 85 arquivos, 40 579 linhas | idem |
| `tests` | 164 arquivos, 47 936 linhas | idem |
| `bench` | 146 arquivos, 21 621 linhas | idem |
| `tools` | 80 arquivos, 22 680 linhas | idem |
| maior arquivo de fonte próprio | `src/targets/qwen3_6/impl/runtime/program_impl.h`, **12 378 linhas** | `wc -l` |
| arquitetura CUDA aceita | **só `120a`** — o build falha em qualquer outra | `CMakeLists.txt:7-14` |
| CUDA mínima | 13.1 | `CMakeLists.txt:36-40` |
| modelos | 5 identidades: Qwen3.6-27B e 3.8-27B (`groupwise-int`, `nvfp4`), 3.6-35B-A3B (`groupwise-int`) | `ninfer README.md:12-18` |

A proporção que interessa: **kernels 64 k linhas contra "produto" (serialização, servidor,
mídia) 17 k** — `src/serve` 11 152 + `src/product` 1 626 + `src/media` 727 + `src/text` 155
+ `apps` 1 476. O motor é um projeto de kernels com um servidor em volta, não o contrário.

### 1.2 As quatro camadas (e a razão de existirem)

`docs/maintainer/engine-architecture.md:54-68` fixa o caminho do pedido em quatro fronteiras:

```
Gateway  (protocolo, transporte, aquisição de mídia)   src/serve/, apps/
   ↓ PreparedPrompt / OutputSession
Frontend (tokenizer, template, VLM, detokenização)      src/targets/qwen3_6/impl/frontend/
   ↓
Engine   (FIFO, admission, lifecycle, publicação)       src/runtime/engine/
   ↓
Program  (recursos físicos, execução do modelo)         src/targets/<pkg>/impl/runtime/
```

A tabela de posse única (`ninfer docs/maintainer/engine-architecture.md:128-139`) é o ponto do desenho que vale
copiar conceitualmente: `Scheduler` só decide *quem roda quando* (`:142-152`),
`ResourceManager` só decide *o que fica retido* (`:154-165`), `Program` é a **única**
autoridade sobre bytes físicos, páginas e allocator (`:167-176`). Nenhum dos três mantém uma
cópia mutável do estado do outro.

### 1.3 Componente → onde vive → o que faz

| componente | onde | o que faz |
|---|---|---|
| kernels de matmul por formato | `src/ops/linear/{bf16,fp8,nvfp4,q4,q5,q6,w8}/` | 7 formatos × até 4 regimes (`gemv`, `gemm_simt`, `gemm_mma`, `small_t_mma`) |
| seleção de rota por forma | `src/ops/linear/q4/q4_dispatch.cpp:7-90` | `switch (k) { case 5120: switch (n) { ... } }` — tabela **escrita à mão por forma registrada** |
| atenção densa (decode) | `src/ops/softmax_attention/dense/causal_cache/small_t*.cuh` | split-KV com política tiered de splits |
| atenção densa (prefill) | `.../causal_cache/prompt_{bf16,fp8,i8,k8v4,nvfp4}.cuh` | online softmax, `Br=64`, MMA |
| atenção janela / empacotada / contexto | `.../dense/{sliding_window,packed,context}/` | variantes separadas, não `if` dentro do kernel |
| GDN / linear attention | `src/ops/linear_attention/gated_delta_net/` | `recurrent.cuh` (cauda) + `chunked/` (3 estágios) |
| projeções+conv do GDN | `src/ops/gdn_input_proj/{q4_q5,fp8,nvfp4,w8}/` | fusão projeção+`conv1d`+snapshot por formato |
| gating do GDN | `src/ops/gdn_gating_proj/bf16/` | GEMM com `cooperative_groups::this_grid().sync()` (`:296`) |
| KV: escrita + codecs | `src/ops/kv_cache/append/`, `.../{int8_g64,nvfp4_group16,fp8_e4m3}_codec.cuh` | append por formato; `hadamard_d256.cuh` faz rotação |
| KV: paginação | `src/core/paged_kv_cache.cpp`, `src/ops/kernel/paged_kv_address.cuh` | página = **64 tokens** (`kPagedKVPageSize = 64`, `src/core/paged_kv_cache.h:18`; `kPagedKVPageShift = 6`, `paged_kv_address.cuh:9`) |
| MoE | `src/ops/sparse_moe/{decode,prefill,small_t}/` | roteamento + especialistas; PDL entre os estágios |
| amostragem | `src/ops/kernel/sampling{,_device}.cuh` | top-k por `cub::BlockMergeSort`/`WarpMergeSort` (`sampling_device.cuh:24,32`) |
| spec decoding | `src/ops/kernel/speculative_round.cuh` (34 645 B), `mtp_round.cuh` | verificação, aceitação, reciclagem da janela |
| runtime / scheduler | `src/runtime/engine/` (`engine.cpp`, `scheduler.h`, `admission_policy.cpp`, `resource_search.h`) | FIFO com backfill, admission, planejamento de contexto |
| servidor | `src/serve/` (37 arquivos) | OpenAI Chat/Responses + Anthropic Messages, streaming, tools, contagem de tokens |
| app de perplexidade | `apps/perplexity/` | `CausalScoreCore` — scoring offline pelo mesmo `Engine` |
### 1.4 Como um token flui

**Decode de um round** (o caminho que o AGENTS.md chama de "compact decode batch"):
1. `Scheduler` forma **um** lote compacto com todos os pedidos prontos
   (`ninfer docs/maintainer/engine-architecture.md:28`, `:134`).
2. `Program` reconstrói o grafo do alvo; o replay acontece por **CUDA Graph** —
   `cudaStreamBeginCapture`/`cudaGraphInstantiate` em **1 arquivo cada**
   (`src/core/decode_graph.cpp`, confirmado por grep).
3. Os kernels rodam na ordem do grafo; entre estágios dependentes há **PDL** nos caminhos de
   MoE, `q4/q5 gemv`, `q4/q5 gemm_simt` e `gdn_input_proj` (§3.7).
4. A amostragem roda no device (`sampling_device.cuh`), com `top_k` limitado por
   `kSamplerCandidateCap` (`:148-151`).
5. A aceitação de draft, se houver MTP/DFlash, é um Op separado com contrato escrito
   (`include/ninfer/ops/speculative_round.h:36-70`) que devolve *tokens de alvo*, não
   logits — o estado provisório é commitado ou revertido pelo `Program`
   (`ninfer docs/maintainer/engine-architecture.md:116`).

### 1.5 KV cache, quantização, lote, spec decoding — o resumo factual

| item | ninfer | onde |
|---|---|---|
| KV | **paginado**, página de 64 tokens, block table de `int32` | `paged_kv_address.cuh:9-12,44-52` |
| formatos de KV | `BF16`, `FP8 E4M3 row-256`, `INT8 group-64`, `NVFP4 group-16`, `K8V4` — cada um com kernel **próprio** | `prompt_{bf16,fp8,i8,k8v4,nvfp4}.cuh` |
| rotação no KV | `hadamard_d256.cuh` — Hadamard de 256 aplicado no Q/K antes de quantizar | `src/ops/kv_cache/hadamard_d256.cuh` |
| pesos | `Q4G64_F16S` (4,25 b/w), `Q5G64_F16S` (5,25), `Q6G64_F16S` (6,25), `W8G32_F16S` (8,5), `NVFP4` (k16 + E4M3FN), `FP8_E4M3FN_ROW_BF16S` | `ninfer docs/maintainer/tensor-formats.md:22-39` |
| layout físico | **dois planos separados: códigos e escalas** — `codes[row][group][32 B]`, `scales[row][group][2 B]` | `q4_rowsplit_storage.cuh:13-15` |
| lote | concorrência **1..8 fixada no startup**, FIFO com limite, **sem preempção**, um lote por round | `AGENTS.md:35-36`, `ninfer docs/maintainer/engine-architecture.md:21-28` |
| chunk de prefill | **1 024 tokens** (medido no perfil publicado) | `ninfer README.md:139` |
| MTP | janela de draft **1 a 5** | `ninfer README.md:214`, `mtp_round.h:28` (`1<=K<=5`) |
| DFlash / DFlash2 | drafters alternativos; DFlash2 com K=1..15, "masked-draft", **contexto local de 5 camadas no StateImage, sem KV de backend completo** | `ninfer docs/maintainer/engine-architecture.md:32-35` |
| aceitação publicada | 68,2 % (C=1) a 69,3 % no 27B `groupwise-int`; 37,8 % (Story) a 89,8 % (Structured) por categoria | `ninfer docs/performance/qwen3.6-27b.md:79-90,120-127` |

---

## 2. O que depende de hardware NVIDIA — e o que sobra no gfx1201

Método: contagem por grep em `src include apps` sobre `*.cu/*.cuh/*.h/*.cpp`. Números são
ocorrências (arquivos distintos entre parênteses quando relevante).

| # | recurso NVIDIA | onde / quantas | para que serve | veredito no gfx1201 |
|---|---|---|---|---|
| 1 | `cuBLAS` / `cuBLASLt` | **0** ocorrências | — | **não se aplica: não existe** |
| 2 | `CUTLASS` | **0** | — | não se aplica |
| 3 | `Triton` | **0** | — | não se aplica |
| 4 | `wgmma` (Hopper) | **0** | — | não se aplica |
| 5 | `tcgen05` (Blackwell 5ª ger.) | **0** | — | não se aplica |
| 6 | `nvcuda::wmma` | **0** | — | não se aplica |
| 7 | **`mma.sync`** | **6 instruções, 1 arquivo**: `src/ops/common/mma.cuh:36,45,53,62,71,89` | m16n8k16 bf16/f16, m16n8k32 s8, m16n8k32 e4m3, m16n8k8 tf32, m16n8k64 mxf4nvf4 | **não portável como instrução**; há análogo direto para 2 dos 6: `wmma f32_16x16x16_f16` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`) e `wmma i32_16x16x16_iu8`. O **empacotamento de fragmento** (m16n8 com 4 acumuladores por lane) não é o nosso, mas a *forma* do laço (K em passos de 16, fragmentos em registrador, acumulador f32) é |
| 8 | **`ldmatrix`** (+ `.trans`) | 4 funções, `src/ops/common/mma.cuh:7-31`, chamadas em `q4_rowsplit_gemm_mma.cuh:316,323` e `q4_small_t_mma.cuh:173` | mover operando LDS→registrador **com o swizzle resolvido pelo hardware** | **não portável**; substituível por `ds_read_b128`/`ds_read2_b64` com o **mesmo swizzle XOR calculado à mão** — que o ninfer já calcula explicitamente (`q4_mma_swizzle_k64`, `q4_rowsplit_gemm_mma.cuh:88-90`), então o endereço já está no código e a parte NVIDIA é só o "faz de graça" |
| 9 | **`cp.async`** | **33 ocorrências em 15 arquivos**, incluindo `src/ops/common/memory.cuh` | staging global→LDS **assíncrono e multi-estágio**: `cp_commit()`/`cp_wait<S-1>()` (`q4_rowsplit_gemm_mma.cuh:298-305`) | **não portável como instrução** (RDNA não tem `cp.async`); **portável como mecanismo**: o ganho vem de separar *emissão* de *consumo* com `STAGES ≥ 2` (`rowsplit_mma.cuh:41`), e isso se reproduz com `buffer_load_dwordx4` para VGPR + `ds_write_b128`, ou com duplo buffer LDS emitido um estágio à frente |
| 10 | `cp.async.bulk` / `cp.async.bulk.tensor` / `cuTensorMapEncodeTiled` | **1 ocorrência de cada, num único arquivo**: `src/ops/linear/nvfp4/nvfp4_w4a4_tma.cuh:44,155` | TMA para o kernel W4A4 de NVFP4 | **não portável**; e é **1 kernel em 216 `.cu`** — irrelevante para a nossa decisão |
| 11 | `cooperative_groups` | **2 ocorrências, 1 arquivo**: `gdn_gating_proj/bf16/bf16_gdn_gating_proj_gemm_mma.cuh:20,296` (`this_grid().sync()`) | sincronização de grade num GEMM de gating | **não portável** (HIP tem `grid.sync()` só com lançamento cooperativo, e o custo/limite é outro). O mecanismo alternativo é o que o ninfer usa em todo o resto: **dividir em dois kernels + PDL**, ou reduzir por LDS |
| 12 | **`__shfl_sync`** | usado em `recurrent.cuh` (`normalize_qk_lane`, `warp_reduce_sum`, `:64-75`) e em `common/warp.cuh` | redução intra-warp (norma L2 do Q/K, reduções do GDN) | **portável**: `__shfl_xor` existe em HIP. É o mesmo recurso que já usamos 5× por chave na nossa atenção (`docs/estudo-prefill-j-atencao-kv.md:410`) |
| 13 | **`bar.sync` / `__syncthreads`** | em todos os kernels de GEMM | separar estágio de consumo | **portável**, mas na AMD o `s_barrier` de um workgroup de 256 threads tem custo diferente — medir, não assumir |
| 14 | **CUDA Graphs** | `cudaStreamBeginCapture`, `cudaGraphInstantiate`: **1 arquivo** (`src/core/decode_graph.cpp`) | replay do passo de decode | **portável** (`hipStreamBeginCapture`); e **nós já medimos que vale 1,06×** (`nosso README.md:258-262`) — não é alavanca |
| 15 | **PDL** (`cudaLaunchAttributeProgrammaticStreamSerialization`, `cudaTriggerProgrammaticLaunchCompletion`, `cudaGridDependencySynchronize`) | `src/core/pdl.cuh:24,40,43`; **28 chamadas em 9 arquivos** | iniciar o kernel consumidor antes do produtor terminar; o produtor sinaliza cedo (`trigger_dependents` logo no começo do kernel, ex. `q4_rowsplit_gemm_simt.cuh:208`) e o consumidor espera só na junção (`:319`) | **não portável** — `INFERIDO`: HIP/ROCm 7.2 não expõe equivalente; **confirmaria** com `grep -rn "ProgrammaticLaunch\|griddepcontrol" /opt/rocm/include`. A **ideia** (graduar quais arestas do grafo toleram começo antecipado) é portável e é o que as fusões do Vulkan já fazem por outro caminho |
| 16 | `__dp4a` | **0** | — | não se aplica (o ninfer **não usa dot de inteiro**; usa MMA s8 ou FMA em bf16/fp32) |
| 17 | `__nv_fp8` / `cuda_fp8.h` | `__nv_fp8` em **6 arquivos, 7 ocorrências** | formatos FP8 dos pesos e do KV | **não portável como tipo**; `INFERIDO`: gfx1201 tem `v_cvt_pk_fp8_f32` e ops de fp8 para *conversão*, não para MAC. Não temos fp8 hoje e o modelo não pede |
| 18 | `L2` hints: `cudaStreamSetAttribute`, `accessPolicyWindow`, prefetch explícito | `bench/ops` tem `__builtin_prefetch`-like; **ver §4.6 para o *warm L2* no host** | — | **portável com ressalva**: o gfx1201 tem Infinity Cache (medido por nós: atenção a 93-97 % dele), mas o mecanismo de hint é outro |
| 19 | `cub` (`cub::BlockMergeSort`, `cub::WarpMergeSort`) | `src/ops/kernel/sampling_device.cuh:24,32` | top-k do sampler no device | **não portável** (CCCL); o *mecanismo* — top-k por merge sort em vez de por threshold — é portável à mão |
| 20 | `nvtx` | `src/core/nvtx.h` | profiling | portável (ROCTx) |
| 21 | `__launch_bounds__(threads, minBlocks)` | em todo kernel (`q4_small_t_mma.cuh:61` → `(256, 6)`) | **fixar a ocupação mínima em tempo de compilação** | **portável com ressalva**: HIP aceita `__launch_bounds__`, mas o mapeamento para limites de VGPR/LDS na AMD é diferente do "minBlocks" da NVIDIA. Nós já fazemos isso por `__launch_bounds__` no `gemm_i8_kernel` (`include/rdna4/gemm.cuh:276`) |

**Resumo da §2**: dos 21 itens, **7 são "não se aplica"** (o ninfer simplesmente não usa
cuBLAS/CUTLASS/Triton/wgmma/tcgen05/`nvcuda::wmma`/`__dp4a`), **7 são não portáveis**
(`mma.sync`, `ldmatrix`, `cp.async`, TMA, `cooperative_groups`, PDL, `cub`, `__nv_fp8`) e
**6 são portáveis** (`__shfl`, `__syncthreads`, CUDA Graphs, `__launch_bounds__`, nvtx,
hints de cache). O que decide o nosso projeto é o primeiro grupo: **um motor de 3 218 tok/s
num 27B que nunca chama cuBLAS e usa 6 instruções de tensor core em 64 303 linhas de
kernel.** O caminho que temos é o caminho que ele trilhou.

---

## 3. Técnicas a salvar, ordenadas pelos NOSSOS gargalos medidos

Cada item: **(a)** o que o ninfer faz e onde, **(b)** o gargalo nosso que ataca, **(c)** o que
toca na nossa árvore, **(d)** se é medível.

### 3.1 Estágio: mover **código comprimido** e decodificar no laço de consumo (LDS→registrador, nunca LDS→LDS)

**(a)** O GEMM SIMT de q4 tem duas metades separadas por um `__syncthreads` que **não**
inclui a decodificação:
- `q4_simt_issue_stage` (`q4_rowsplit_gemm_simt.cuh:77-110`) emite `cp_async<16>` dos
  **bytes de código crus** (32 B por grupo de 64 pesos) e `cp_async<4>` dos **pares de
  escala** alinhados, e chama `cp_commit()`. Nada é decodificado aqui.
- `q4_simt_consume_stage` (`:112-159`) lê `shared_codes[phase*32 + lane]` como **um
  `uint32` = 8 nibbles**, chama `Q4SimtDecodeAtom::decode_eight` (`q4_rowsplit_storage.cuh:20-32`)
  — que é **4 `__hsub2` + `__byte_perm` sobre bf16, sem LUT e sem passar por fp32** — e
  consome `float weights[8]` **em registrador**, direto nos `fmaf` (`:147-154`).
- As escalas vêm em pares (`kScalePairsPerStage = kGroupsPerStage/2`,
  `q4_rowsplit_gemm_simt.cuh:44`) e o grupo ímpar é extraído por deslocamento
  (`:129-131`).

Ou seja: **o caminho do peso é global → LDS (4 bits/peso) → registrador+decode → FMA**. Nosso
`gemm_i8_kernel` faz **global → registrador (`Pf pf`) → decode+repack → LDS (int8) →
registrador (operando WMMA)** (`include/rdna4/gemm.cuh:307-332` e `:329`), isto é, **dois giros pela LDS** e
a decodificação **antes** da barreira, no caminho crítico do estágio.

**(b)** *staging* (50-61 % do kernel; 132-275 GB/s de fonte contra 633 de roofline, 
`plano-prefill.md:244-249`) **e** a correção de escala (53-56 % do miolo). São o mesmo
conserto: o que está caro não é "a LDS é lenta", é que o nosso formato de estágio
(`int8` na LDS, `include/rdna4/gemm.cuh:329`) **exige um passo de decodificação antes que a conta possa
começar**, e esse passo (i) trafega 2× mais bytes do que o código comprimido (8 bits contra
4,25) e (ii) fica entre a barreira e o laço, sem FMA para escondê-lo.

**(c)** `include/rdna4/gemm.cuh`: trocar `TR::store` (decode→LDS) por um estágio que copie
`qs`/`qh`/`scales` crus para a LDS em vetores de 16 B e mova a decodificação para dentro do
laço de `kb` (`:384`), alimentando o operando do `wmma` direto de registrador. Mantém a
correção de escala onde ela está (`s_dwsc`, `:330`) — o que muda é **onde** a decodificação
acontece, não a aritmética, então o gate bit-exato de `docs/plano-prefill.md:238-239`
continua valendo. `INFERIDO`: o custo por bloco de 32 cai de 590 para ~396 instruções — o
mesmo número que a frente G já mediu para o staging int8 contra o f16 (`plano-prefill.md:247`)
— **confirmaria** com `--save-temps` + contagem de instruções no corpo do laço, antes de
cronometrar.

**(d)** Sim, e é **bit-exato**: mesmo gate do D2 (`check-matmul-gpu`, `check-batch-gpu`,
`check-graph-gpu`, golden). Mede-se T-MAC/s em M=64/128/512 sobre `blk.3.ffn_up` iq3_s, que
é a bancada que a frente G já usa.

### 3.2 Micro-lote: o tile de registro é em **colunas de token**, com teto duro de 8

**(a)** `Q4RowSplitSimtGemmSchedule<RowsPerCta, ColsPerTile, GroupsPerStage, PipelineStages,
Cache, LaunchBoundsMinBlocks>` com **`static_assert(kColsPerTile <= 8)`**
(`q4_rowsplit_gemm_simt.cuh:52`) e as duas instanciações que embarcam:
`<8, 4, 16, 2, Cache::ca, 1>` e `<8, 8, 16, 2, Cache::ca, 1>`
(`q4_rowsplit_gemm_simt.cu:13-14`). Um warp é **dono de uma linha de peso** (`kCtaWarps =
kRowsPerCta`, `:39`); os acumuladores são `float acc[ColsPerTile]` — **8 no máximo** — e o
peso decodificado (`float weights[8]`) é reusado nas `ColsPerTile` colunas
(`:133-155`). A LDS gasta **8 704 B** (`kSharedBytes = 8×2×(16×32 + 8×4)`, `q4_rowsplit_gemm_simt.cuh:46-49`).

**(b)** **Micro-lote** — o fator de 2,50× do gap. Nosso D1 subiu o chunk mantendo o kernel e
**regrediu** (−18 % em N=32, −71 % em N=64) porque `acc[N][ILP]` custa N VGPRs por thread: o
`iq3_s` foi de 62 para 133 VGPRs e a ocupação caiu pela metade (`plano-prefill.md:29-40`).
O ninfer resolve o mesmo problema **pelo eixo oposto**: o paralelismo vai para
`RowsPerCta` (warps) e o *tile de registro* tem teto de 8, então a pressão de registrador
**não escala com o lote**. É a regra de projeto que faltava ao D1.

**E o limiar dele confirma o nosso número, de forma independente.** Em `q4_dispatch.cpp`,
para a forma `k = 5120, n = 1024` (`:12-17`):

```cpp
if (t == 1)  { return launch_q4_gemv_r1_w8_direct; }   // :14
if (t <= 15) { return launch_q4_simt_r8_c4; }          // :15
if (t == 16) { return launch_q4_simt_r8_c8; }          // :16
return launch_q4_mma_r64_c128;                          // :17
```

**`T ≤ 16 → SIMT; T > 16 → unidade de matriz`.** Um motor que tem `mma.sync` disponível e o
usa em 14 escalonamentos escolhe **não** usar unidade de matriz no nosso `kMaxBatch = 16`.
Isso é o mesmo resultado que nós medimos (o D2 em M=16 dá 0,96×, perde —
`docs/plano-prefill.md:131`) e o mesmo que o nosso plano fixou em "alvo de chunk = 128, não
512" (`docs/plano-prefill.md:166-168`). O que o ninfer acrescenta é **o que fazer nos dois
lados do limiar**: abaixo dele, um GEMM SIMT com tile de 8 colunas e pipeline de 2 estágios
(§3.2); acima, os 14 escalonamentos de `BN` (§3.3).

**(c)** `include/rdna4/gemm.cuh` (a instanciação `BM=16, BN=128, BK=64, RM=1, RN=8` em
`include/rdna4/gemm.cuh:544` já tem a forma certa em N; o que muda é o eixo M) e
`include/rdna4/graph.cuh:191` (`kMaxBatch`). É o mesmo trabalho do D2, mas com a ordem
corrigida: **primeiro o teto de 8 no tile de registro, depois subir o chunk.**

**(d)** Sim, com a bancada que já existe: `bench-matvec-shapes-gpu --batch 8,16,32,64` sobre
o inventário real de 497 tensores (`docs/plano-prefill.md:27`), reportando ms/token **e**
registradores/`localSizeBytes` por instanciação — os dois números que reprovaram o D1.

### 3.3 Micro-lote pequeno e MTP: um lançador **por valor exato de T**, com K dividido entre warps

**(a)** `q4_small_t_mma.cu:14-16,34-44` gera, em tempo de compilação, um **array de lançadores
indexado pelo T exato**: `make_launchers<FullGeometry, 2>(index_sequence<0..6>)` cobre
T = 2..8 e `make_launchers<OptimizedGeometry, 2>(index_sequence<0..18>)` cobre T = 2..20,
com `TileTokens = ((T+7)/8)*8` e `ActiveTokens = T`
(`make_launchers`, `:32-35`). O kernel (`q4_small_t_mma.cuh:32-38,61,82-90`):
`kKWarps = 8`, `kTileKPerWarp = 64` (então `kGroupK = 512`), `kRowsPerCta = 16`,
`__launch_bounds__(256, 6)`, grade de `131072/16 = 8 192` CTAs; o K é dividido entre os 8
warps e as parciais somadas **na LDS** ao final (`:201-260`, com a
`union SharedStorage { staging; float partial[...] }` em `:82-90` — o mesmo espaço serve de
estágio e de redução, para caber nos 48 KB).
O despacho usa a **mesma especialização por forma**: `if (t <= 20) return
launch_q4_draft_head_small_t; if (t <= 32) ... c32; if (t <= 48) ... c48;` —
`q4_dispatch.cpp:49-65`.

**(b)** **Micro-lote** (verificação de MTP com 2-3 tokens; `docs/README` limitação 3:
`pass_ms ≈ 13,9 + 6,04·N`, o que faz o break-even de aceitação ser 56 %) **e MTP** — a nossa
medição diz que o multiplicador do MTP é `2·aceitação / 1,12` e que o que falta é o custo
**marginal** por token (`nosso README.md:483-497`). Um kernel cujo tile é exatamente o T da
rodada, com K dividido em 8 warps, ataca exatamente esse custo marginal: em T=2-3 a
paralelização vem de K e de linhas de saída, **não** de colunas de token (que não existem).

**(c)** Novo, ao lado de `include/rdna4/gemm.cuh`: um `gemm_small_t` com `KWarps=8`,
`RowsPerCta=16` e redução por LDS. Chama-se em `graph.cuh` no caminho do MTP (verificação de
2-3 tokens) e no LM head quando N ≤ 8 — **nota**: o nosso LM head já roda a 98 % do roofline
(`nosso README.md:252`), então o alvo aqui é a **verificação**, não a projeção final.

**(d)** Sim: `bench --mtp --draft 2` contra `--mtp --draft 3` na mesma prompt de 3 683/3 996
tokens, medindo ms **por linha verificada** (hoje 8,0 ms contra 34,4 ms de um passo inteiro,
`nosso README.md:471-473`). O número a mover é o coeficiente 6,04 ms/token.

### 3.4 GDN: blocos de 64 com três estágios paralelos em vez de uma recorrência sequencial

**(a)** Existem **dois** caminhos, e o chunked é o que importa:
- `kChunkSize = 64`, `kStateDim = 128`, hard-coded e verificado por `static_assert` nos três
  kernels (`gated_delta_net/common.h:7-8`; `chunked/output.cuh:35-36`,
  `chunked/prepare_wy_wu.cuh:28`, `chunked/state_passing.cuh:21`).
- Três estágios, três lançamentos, com workspace explícito (`chunked/launch.cu:31-73`):
  1. `prepare_wy_wu` → produz `g_cumsum`, `W`, `U` (a representação WY por bloco);
  2. `state_passing` → a **única** passagem sequencial, sobre `L/64` estados de bloco, e
     escreve `h_chunk` e `v_new`;
  3. `output` → paralelo por bloco, usando `h_chunk`.
- O workspace é **alocado e tipado pelo host** e passado no contrato
  (`chunked/launch.h:22-30,32-48`: `g_cumsum [H_v, T] f32`, `W/U/v_new [128, H_v, T] bf16`,
  `h_chunk [128, 128, H_v, T/64] bf16`).
- O caminho **recurrente** fica para a cauda: `check_full_chunks()` recusa T não múltiplo de
  64 com a mensagem literal *"route tail tokens through AR instead"* (`chunked/launch.h:119-125`).
  Ele é `block = (32, 4)` com `kDvPerWarp = 4` e grade `(H_v, 1, kStateDim/kBlockDv)`
  (`recurrent.cu:29-30`), isto é **`H_v × 8` CTAs**, com o estado em registrador:
  `kQkPerLane = 4` floats por lane carregados como um `float4`
  (`recurrent.cuh:13-16,22-25`).

**(b)** **Recorrência do GDN — 0,852 ms/token**, o único item do andaime com tamanho para
importar; no degrau D4 ele passa a ser ~52 % do que sobra (`plano-prefill.md:100,115-119,253-255`).
O nosso `delta_rule_batch_rows_kernel` tem um `for (t = 0; t < n; ++t)` sequencial com
grade `(n_v_heads, ny)` (`include/rdna4/gdn.cuh:364,380,435`), e o caminho por token roda com **48 CTAs =
9 % de ocupação** (`nosso README.md:250`). O paralelismo de `H_v × 8` CTAs com o estado em
registrador é a mesma ideia da "reescrita de 4 threads por linha" que o nosso README estima em
até ~3 ms — mas o **ganho grande é o chunked**, porque troca `O(T)` passos sequenciais por
`O(T/64)`.

**(c)** `include/rdna4/gdn.cuh`: dividir `delta_rule`/`delta_rule_batch_rows_kernel` em três.
Os estágios 1 e 3 são trabalho paralelo sobre blocos e cabem no formato que o
`forward_batch` já tem (B=16, `include/rdna4/graph.cuh:191`) — **atenção**: `kChunkSize = 64` do ninfer é
maior que o nosso `kMaxBatch = 16`, então o nosso bloco natural hoje é 16 (ou 128 se o D2
virar default, que é o `RD_PREFILL_CHUNK=128` de `plano-prefill.md:138`). O workspace é
memória nova e precisa entrar no orçamento de VRAM (temos 1,68 GiB livres em `q5_0`/`q4_1` a
131K, `nosso README.md:438`) — `INFERIDO`: para `H_v = 8`, T=512, o
`W+U+v_new` do ninfer custa `3 × 128 × 8 × 512 × 2 B = 3,1 MB` por camada; **confirmaria**
com a conta exata do nosso `n_v_heads`.

**(d)** Sim e é o item mais fácil de medir isolado: existe `bench/ops/gated_delta_net_bench.cu`
no repo dele, e do nosso lado o caminho é isolar o `delta_rule` como a frente C já fez
(0,852 ms/token medidos, `plano-prefill.md:100`). O número a mover é ms/token do GDN com
T=16 e T=128.

### 3.5 Atenção no **decode**: um CTA por cabeça de KV, dono do grupo inteiro de GQA

Este é o item mais valioso do repo, e é o único que ataca diretamente uma limitação que nós
já temos medida e nomeada.

**(a)** O caminho `small_t` (decode e verificação de draft) tem **grade `(KVHeads, splits,
batch)`** (`small_t.cu:102`) e o índice de cabeça de KV é o eixo x:

```cpp
const int kv_head = static_cast<int>(blockIdx.x);      // small_t_bf16.cuh:53
```

E as linhas de consulta do CTA são o **produto cartesiano** (token × cabeças do grupo),
decodificado por (`small_t.cuh:157-163`):

```cpp
token             = row / Geometry::GroupSize;
const int local_q = row - token * Geometry::GroupSize;
q_head            = kv_head * Geometry::GroupSize + local_q;
```

com o teto imposto em tempo de compilação: `static_assert(TokenTile >= 1 && TokenTile *
Geometry::GroupSize <= 48)` (`small_t_bf16.cuh:27`) — ou seja, **um CTA cobre até 48 linhas
de consulta de 4 cabeças de KV**, e o K/V é carregado uma vez por tile para todas elas. O
comentário de projeto é explícito (`small_t_k8v4.cuh:3-4`):

> *"A CTA owns one KV head and all GQA query heads, so each persistent K/V byte is streamed
> once."*

O `GroupSize` é 6 para `CausalD256H24Kv4` (`geometry.cuh:15` + `head_mapping.cuh:14`) — **a
nossa mesma razão 24/4**. E o decode fica com uma grade minúscula e deliberada: `T=1` usa
`TokenTile = 1`, `WarpsPerCta = 2` (64 threads), grade `(4, splits, 1)` mais um kernel de
redução de 24 CTAs.

**(b)** **Atenção/KV no decode longo — a releitura de 6×.** O nosso número: as 6 cabeças de
um grupo leem as mesmas linhas, e a 64K isso são **25,77 GB lógicos contra 4,295 GB únicos**,
com o kernel a saturar ~1,35 TB/s de L2 (`nosso README.md:498-507`). O ninfer **elimina a
multiplicação por 6** por construção: cada byte de K/V persistente é lido **uma vez** por
cabeça de KV. Do lado do prefill a história é outra — ver §4.1.

**(c)** `include/rdna4/attn.cuh`: o nosso `attn_batch_kernel` tem grade `(n_head, n_tok)` =
(24, 16) = 384 CTAs (`include/rdna4/attn.cuh:320,416`) e `kvh = h / (n_head / n_head_kv)`
por **cabeça de consulta** (`include/rdna4/attn.cuh:331`). O port é trocar o eixo x de
`n_head` para `n_head_kv` e levar as 6 cabeças do grupo para dentro da CTA (eixo y ou
tile de linhas), mantendo a soma de linhas do grupo ≤ 48 como o ninfer.
**Nota de escopo**: isto é o mesmo movimento que o `attn_split_kernel`
(`include/rdna4/attn.cuh:496-505`) já repete; a mudança é no mapeamento, não na matemática.

**(d)** Sim, e o instrumento já existe: `scripts/check_attn_split.sh` mede o desvio do split
contra o não-dividido em texto real (a escada medida é 5,1989 unsplit → 5,2054 em 4 splits →
5,2114 em 16, `nosso README.md:503-507`), e
`python3 -c` sobre o log/`--phase`... — na prática: `bench --ctx-size 65536 --cache-type-k
q5_0 --start-pos 65408 --fill-cache` antes e depois, com a atenção isolada por
`docs/journal-kv.md` §6.1. O número a mover é o tráfego lógico de KV (**25,77 → ~4,3 GB**) e
a banda efetiva (165-172 GB/s, 27-29 % do pico, `nosso README.md:444-446`).

### 3.5b Atenção no **prefill**: o tile `Br = 64` com K/V na LDS

**(a)** O caminho `prompt`: `kCausalPromptBr = 64`, `kCausalPromptBc = 64`, 128 threads, LDS =
**`(Br + 2·Bc) · head_dim · sizeof(bf16)`** (`prompt_common.cuh:21-26`) — com
`head_dim = 256` dá **98 304 B**, isto é, um CTA por SM (o teto por CTA no `sm_120a` é 99 KB,
ver §3.9). Grade `(ceil(tokens/64), q_heads)`, K/V estagiado por CTA
(`prompt_bf16.cuh:97-103,195,206,362`), online softmax com `m`/`l` correntes, MMA.

**(b)** Aqui a leitura correta é a do nosso próprio estudo: a atenção é **2,1 %** do nosso
prefill e zerá-la levaria 122,66 → 125,30 tok/s
(`docs/estudo-prefill-j-atencao-kv.md:471-472,492-495`). O que **não** é nota de rodapé é
que `Br = 16` linhas de consulta por workgroup é o **pré-requisito do lote grande**
(`docs/estudo-prefill-j-atencao-kv.md:497-501`), e é o port nº 1 daquela frente.

**(c)** `include/rdna4/attn.cuh` (`attn_batch_kernel` em `:320-380`, grade em `:416`). O
nosso kernel hoje tem 8 256 B de LDS e **sem reúso de tile** — cada warp relê a linha de K/V
da global e a reduz com 5 `shfl_xor` + 2 `expf` **por chave**
(`docs/estudo-prefill-j-atencao-kv.md:404-420`). O port converte isso em tile: `Br` linhas de
consulta × `Bc` colunas de KV na LDS, redução **por coluna de tile**.

**(d)** Sim, e a conta já está feita: com `Br = Bc = 64` e `head_dim = 128` (o nosso, contra
256 do ninfer) a LDS pede `(64+128)·128·2 = 49 152 B` — **cabe nos nossos 64 KB** por
workgroup, com folga para o duplo buffer. Mede-se com `bench --prefill 512` e com
`RD_ATTN_SPLITS` para isolar a atenção, contra o alvo de `0,031 → ?` ms/token em 64/128
chaves (`docs/estudo-prefill-j-atencao-kv.md:450-451`).

### 3.6 Política de split da atenção em **degraus**, com teto alto

**(a)** `causal_small_t_split_upper_bound` (`small_t.cu:25-43`) calcula os splits por
**faixas de janela**, não por uma lei linear: 64 chaves/split até 4 096; 128 de 4 096 a
8 198; 256 de 8 198 a 16 390; 480 acima — com mínimo de 4 e teto
`SmallTMaximumSplits = 85 · SmallTSplitScale` (`geometry.cuh:11-12`). Há ainda casos
especiais por `(formato, T)`: `Int8Group64 && tokens == 5 && 128 < window <= 512` usa
32 chaves/split (`small_t.cu:57-62`), com a justificativa no comentário: *"T=5 usa um tile de
32 chaves por split"*.

**(b)** **Atenção/KV no decode longo.** A nossa política é `attn_splits_for(keys)`: `if (keys
< kAttnSplitMin) return 1; sp = keys/kAttnSplitMin` com `kAttnSplitMin = 512` e **teto 16**
(`include/rdna4/graph.cuh:432-444`, `include/rdna4/tuning.h:101-102`). O teto de 16 é o ponto: a 64K chaves a nossa
atenção custa **19,0 ms/token** e é issue-bound no dequant do KV
(`docs/estudo-prefill-j-atencao-kv.md:453`, `nosso README.md:253-254`); o ninfer deixa 85 splits na
mesma janela.

**(c)** `include/rdna4/graph.cuh:432-444` + `include/rdna4/tuning.h:101` — só a função de
política, mais `attn_split_launch` para aceitar o teto maior. É a mudança de menor risco
desta lista: nenhuma aritmética de kernel muda.

**(d)** Sim, e já temos o instrumento: `scripts/check_attn_split.sh` mede o desvio do split
contra o não-dividido em texto real (a escada medida é 5,1989 unsplit → 5,2054 em 4 splits →
5,2114 em 16 com CTA largo, `nosso README.md:503-507`). Estender a escada a 32/64 splits dá o
número que falta para escolher o teto.

### 3.7 Despacho: PDL é o mecanismo; a **ideia** portável é graduar as arestas

**(a)** `src/core/pdl.cuh` é um wrapper de 45 linhas: `launch_dependent` liga
`cudaLaunchAttributeProgrammaticStreamSerialization` (`:24`), `trigger_dependents()` chama
`cudaTriggerProgrammaticLaunchCompletion()` (`:40`) e `wait_for_dependencies()` chama
`cudaGridDependencySynchronize()` (`:43`). Uso real: **28 pontos em 9 arquivos** —
`sparse_moe/{small_t,decode}/`, `q4/q5_rowsplit_gemv.cuh`, `q4/q5_rowsplit_gemm_simt.cuh`,
`gdn_input_proj/q4_q5/`. O padrão é constante: `if (threadIdx.x == 0) pdl::trigger_dependents();`
**cedo** (ex. `q4_rowsplit_gemm_simt.cuh:208`) e `if constexpr (JoinPdl)
pdl::wait_for_dependencies();` **na junção** (`:319`) — com `JoinPdl` como parâmetro de
template, isto é, **opt-in por aresta do grafo**.

**(b)** **Despacho.** Nossas ~1 940 partidas/token custam ~4,0 ms (11 %), e o que sobra é
"a ~1 440 *pequenos* kernels, que estão no piso (2,77-3,16 µs contra 2,20 µs de um kernel
vazio)" (`nosso README.md:258-262`). Reproduzir o *efeito* do PDL no gfx1201 é a alternativa à
fusão de kernels — e a fusão já foi medida do nosso lado e vale **1,1 %**
(`plano-prefill.md:256`).

**(c)** Nada hoje; seria um mecanismo novo em `include/rdna4/stream.h` ou no `graph.cuh`. O
candidato portável é um **sinalizador por kernel**: o produtor escreve uma flag com
`__threadfence()` depois do último store útil, o consumidor gira nela antes do primeiro
load — o que é essencialmente feito à mão o que o PDL faz no hardware.
`INFERIDO`: **não portável diretamente** e o custo do spin pode comer o ganho; **confirmaria**
primeiro lendo `/opt/rocm/include` por um equivalente de `cudaGridDependencySynchronize`, e
só depois prototipando numa aresta só (a mais pesada do grafo).

**(d)** Parcialmente: dá para medir o **teto** sem implementar nada — contar, no grafo, as
arestas em que o consumidor só depende de uma fração pequena da saída do produtor. Se esse
número for próximo de zero, o item morre antes de custar uma linha de código.

### 3.8 L2: esquentar o cache da próxima projeção a partir da cauda do kernel anterior

**(a)** O HEAD do clone é literalmente esse tipo de mudança: `perf(ops): warm L2 for the next
projection from the MoE down tail, issuing the hint before the block barrier` (commit
`7f14d96`), e o anterior é `perf(ops): prefetch the shared-expert down weights from the D1
tail` (`ce95491`). Eu li só as mensagens de commit e **não** abri o diff, então o *mecanismo*
exato do hint está `INFERIDO` — o que a mensagem fixa é o **lugar** (a cauda do kernel
produtor, antes da barreira de bloco) e o **alvo** (os pesos da projeção seguinte).
**Confirmaria** lendo `git show 7f14d96`.

**(b)** **VRAM/banda** — o nosso matvec está a 436 GB/s de um roofline medido de 633
(`nosso README.md:209-210`), e a atenção roda a 93-97 % do Infinity Cache.

**(c)** `include/rdna4/gemm.cuh`, na cauda do laço de `k`, e o mesmo em `matvec.cuh`. O
gfx1201 tem Infinity Cache, mas o hint é outro — o caminho portável é emitir um
`buffer_load`/`global_load` de antecipação (ou `s_prefetch`) para a primeira linha do peso
seguinte.

**(d)** Sim: é a mesma bancada de `bench-matvec-shapes-gpu --batch`, medindo GB/s efetivos
na fonte com e sem o hint.

### 3.9 O que **não** vale a pena copiar de infraestrutura

- **A tabela de despacho escrita à mão por forma** (`q4_dispatch.cpp:7-91`, ~90 linhas de
  `switch (k) { case 5120: switch (n) { case 1024: ... } }`) é o que permite ao ninfer
  especializar `BN` no T exato — mas o custo é que **cada forma nova é código novo**, e o
  próprio arquivo joga `throw std::invalid_argument("q4 linear: unsupported shape or T")`
  (`:90`) para tudo que não está na lista. Para nós, o análogo útil é **uma tabela de
  limiares por T** (como a `mul_mat_vec_max_cols` que o Vulkan tem e nós não,
  `docs/plano-prefill.md:169-171`), não a enumeração por forma.
- **`48 KiB` como teto de LDS é orçamento autoimposto, não o limite do hardware.** O
  `sm_120a` tem **99 KB por CTA** — o próprio ninfer usa esse número como teto real em vários
  kernels (`bf16_gemm_mma.cuh:72`, `fp8_a8_mma.cuh:72`, `w8_rowsplit_gemm_mma.cuh:58`, todos
  `static_assert(... <= 99 * 1024, "sm_120a per-CTA shared memory limit")`), e os 48 KiB
  aparecem só na família Q4/Q5 e no `GemmCfg` compartilhado (`rowsplit_mma.cuh:46`,
  `q4_rowsplit_gemm_mma.cuh:81-82`). Ou seja: **a família de GEMM quantizado escolheu caber em
  48 KiB para garantir 2 CTAs/SM; não é restrição da placa.** O que vale copiar é o
  `static_assert` — o teto virar erro de compilação em vez de queda de ocupação silenciosa,
  que é exatamente o modo de falha que a frente G mediu (256 VGPR + 56 B de spill no `BM=128`,
  `docs/plano-prefill.md:134`). E vale o contraste: o caminho de atenção do prefill deles
  **gasta 98 304 B num CTA** (`prompt_common.cuh:24-26`) e paga 1 CTA/SM como preço.

---

### 3.10 MTP: não fotografar o estado — **gravar as entradas e reexecutar**

Este item não estava na minha lista inicial e é o que ataca mais diretamente a nossa
dificuldade medida com MTP.

**(a)** O ninfer **não** tira snapshot do estado recorrente por posição durante a verificação.
Ele roda a recorrência normalmente e, em paralelo, grava as **entradas cruas da transição** em
quatro planos (`src/core/gdn_replay_records.h:36-40`):

```
conv  : BF16 [conv_channels, width, rows]          // uma coluna por token de verificação
key   : BF16 [key_dim=128, qk_heads, width, rows]  // k CRU, pré-normalização
value : BF16 [value_dim=128, value_heads, width, rows]
gate  : FP32 [2, value_heads, width, rows]         // {g, beta} na ordem
```

`q` **não** está no registro, deliberadamente (`replayssm-gdn.md:183`: não afeta o estado).
Depois que o comprimento aceito `m` é conhecido, um kernel de *fold* reexecuta os `m` primeiros
passos a partir do checkpoint cometido e publica **só o estado final**
(`recurrent_fold_kernel`, `recurrent.cuh:686-697`), mais a história de conv
(`publish_final_conv_history`, `recurrent.cuh:549-589`). O fold reusa literalmente o mesmo
corpo de laço do kernel de verificação — `run_recurrent_sequence` é chamado nos dois
(`recurrent.cuh:682` e `:694`) — e os gates são lidos de volta como **bits exatos**, não
recalculados (`load_record_gate`, `recurrent.cuh:94-97`). É isso que torna o replay fechado
bit-exato.

A **razão** é o tamanho: 146,8125 MiB de snapshot contra 1,705078 MiB de registro por posição
no 27B = **86,1×**; 74,2× no 35B-A3B (`replayssm-gdn.md:59-66`). E a grade do fold é
`(kValueHeads, active_rows, kLayers·8)` (`recurrent.cu:122-124`) — todas as camadas, cabeças e
linhas em paralelo, com só `m` transições seriais por CTA.

**(b)** **MTP.** O nosso estado: o rollback do estado recorrente é bit-exato e custa **0,54 ms
por par snapshot+restore** (`nosso README.md:471-473`), e o MTP não paga porque o custo
marginal do matvec em lote é 6,04 ms/token (`nosso README.md:483-497`). O ninfer ataca uma
parte **diferente** do mesmo problema: em vez de baratear a cópia de estado, ele **elimina a
cópia** — escreve 4 planos pequenos por token e paga uma reexecução de `m` passos uma única
vez por rodada. **86,1× menos tráfego de estado na escrita**, e a leitura deixa de ser `m`
restaurações.

**(c)** `include/rdna4/mtp.cuh` / `mtp_gen.h` e o rollback em `include/rdna4/gdn.cuh`. O que
falta: os planos de registro (4 tensores, `width = draft+1` posições), um kernel de fold que
reuse o corpo do `delta_rule`, e o gate bit-exato — que já existe no nosso protocolo
(`--mtp` é byte-idêntico ao greedy, `nosso README.md:470-471`). **Atenção**: ao contrário do
snapshot, isto muda o *algoritmo de commit*, não um layout — o gate de md5 continua sendo o
juiz.

**(d)** Sim, e é o mais direto de todos: `--mtp --draft 2` e `--draft 3` na mesma prompt de
3 683/3 996 tokens, comparando os **0,54 ms** do par snapshot+restore contra o custo do fold.
O número a mover é `nosso README.md:475-478`: 31,03 = 0,96× na árvore final.

### 3.11 Layout: o peso nunca é pré-desquantizado nem reempacotado no carregamento

**(a)** Verificado em três pontos: o carregamento é **cópia de bytes**
(`cudaMemcpyAsync` do payload direto para o device, `src/artifact/materializer.cpp:233-238`), o
plano carrega só `{object, offset, bytes, alignment}` (`src/artifact/binder.h:22-27`) e o
*binding* é aritmética de ponteiro dentro do payload copiado
(`src/artifact/typed_binding.cpp:100-103`: `out.qdata = bytes; out.qhigh = ... ;
out.scales = bytes + geometry.scale_plane_offset;`). O layout persistente é **row-major com
planos separados**: código base → pad de 256 B → plano de bits altos → pad de 256 B → plano de
escalas fp16 (`ninfer docs/maintainer/storage-layouts.md:115-134`, `:196-206`). O **único**
swizzle que existe no artefato é o plano de escalas do NVFP4 (`blockscale-k16-m128x4-v1`);
**todo o resto é swizzle de LDS, aplicado na hora do `cp.async` e nunca armazenado**
(`rowsplit_mma.cuh:58-60`; `q4_mma_swizzle_k64` em `q4_rowsplit_gemm_mma.cuh:88-90`).

**(b)** **Staging (item 3.1).** Esta é a metade que faltava do §3.1: não basta decodificar
tarde — é preciso que **o artefato já esteja no formato que o `cp_async<16>` copia**, isto é,
32 B contíguos por grupo de 64 pesos, num plano que só tem códigos. O nosso IQ3_S tem `qs`,
`qh` e `scales` intercalados por bloco de 256 (`include/rdna4/gemm.cuh:104,280-282,304-332`),
e é por isso que o nosso estágio precisa de um passo de decodificação **antes** de escrever a
LDS. **Isto não é um problema de kernel, é de layout** — e nós já temos precedente de mexer
nessa camada (`include/rdna4/dequant_row.cuh`, o `quantize_batch`).

**(c)** `include/rdna4/quants.h`, `include/rdna4/quant_tables.h`, `include/rdna4/dequant.cuh`
e o loader (`include/rdna4/loader.h`). **O custo é alto e é honesto dizê-lo**: um layout
próprio significa um passo de conversão offline (como o `tools/convert/*/convert.py` deles) e
abandonar a leitura direta de GGUF. O caminho barato equivalente é **escrever um *espelho* de
códigos contíguos no device no startup**, a partir do GGUF — não é reempacotamento em tempo de
execução, é transformação de carga. `INFERIDO`: o custo é 1× o tamanho dos tipos cobertos em
VRAM, e isso provavelmente **não cabe** nos 1,68 GiB livres em `q5_0`/`q4_1` a 131K
(`nosso README.md:438`); **confirmaria** medindo o espelho para os três tipos que o D2 cobre
(`iq3_s` + `iq3_xxs` + `iq4_xs` = **68,0 %** dos bytes de peso, `docs/plano-prefill.md:132`).

**(d)** Parcialmente, e dá para matar o item barato: instrumentar o estágio do
`gemm_i8_kernel` para separar **bytes lidos da global** de **bytes escritos na LDS**. Se a
razão for ~2× (8 bits de int8 contra 4,25 de código), o §3.1 é o gargalo e o layout vale a
pena; se for ~1×, o gargalo é outro e o item morre antes de custar um conversor.

---

### 3.12 Amostragem: RNG **sem estado**, chaveado por (seed, posição, propósito)

Barato, pequeno e diretamente ligado a um gate que já temos.

**(a)** O sampler roda inteiro no device e o RNG é **splitmix64 contra-baseado**, sem estado
mutável (`sampling_device.cuh:83-88`), com a chave `(seed, posição lógica, propósito)`
(`include/ninfer/ops/sampling.h:68-70`) e 6 propósitos enumerados (`:14-21`: prefill, decode,
aceitação, correção, bônus, proposta DFlash2). O comentário do código diz a razão
(`sampling_device.cuh:90-91`): *"Pure function of its inputs so it is safe under CUDA-graph
replay (no mutable RNG state)"*. E o top-k do caminho rápido é limitado a **20 candidatos**
(`kSamplerCandidateCap = kSamplerCandidateFast = 20`, `src/ops/common/sampling_workspace.h:24-25`),
com clamp do `top_k` do usuário a esse teto (`sampling_device.cuh:146-154`) — **não há
ordenação do vocabulário inteiro**.

**(b)** **Dispatch/VRAM e reprodutibilidade.** O nosso caminho de logits copia **993 KB por
token** e o *sampler* custa 0,65 ms/token (`nosso README.md:255`), e nós já levamos o argmax
greedy para o device. Um RNG sem estado é o que permite manter o sampler **dentro** de um
grafo replayed sem carregar estado entre execuções — que é exatamente o que o nosso
`check_golden_run.sh` (geração reproduzível run a run, `nosso README.md:362`) exige, e o que
fica frágil se o grafo for capturado com estado de RNG.

**(c)** `include/rdna4/sampler.h`. É a mudança de menor risco desta lista inteira: não toca
nenhum kernel de modelo, e o gate é o que já existe (`./build/check-sampler`,
`scripts/check_golden_run.sh`).

**(d)** Sim: `./build/check-sampler` mais duas execuções da mesma prompt com `--seed` fixo,
comparando md5. Se o md5 já bate hoje, o ganho é de arquitetura (grafo + MTP), não de
correção — e é assim que deve ser reportado.

---

### 3.13 O drafter em bloco muda *por que* a especulação paga — e é o caminho que falta para o nosso MTP

Este item não é uma técnica de kernel; é uma correção de enquadramento, e é o que eu levaria
para a próxima sessão se só pudesse levar uma coisa depois do §3.5.

**(a)** O ninfer tem dois drafters com **formas diferentes de custo**:

- **MTP** sorteia autoregressivamente: `for (std::uint32_t step = 0; step + 1 < k; ++step)`
  chamando `mtp_forward_decode_batch` **mais** `mtp_propose_batch` a cada passo
  (`mtp_impl.h:169,183,185`). Ou seja, **cada draft custa um forward** — exatamente a nossa
  situação.
- **DFlash2** faz **um** forward *masked-block* sobre a largura inteira: `const int width =
  k + 1` (`dflash_impl.h:242`), `prepare_masked_block` (`:255`) e **um** laço de 5 camadas
  (`:261`) produzem os `K` candidatos de uma vez. A documentação é explícita: *"DFlash2 不是多步
  扩散采样器；它在一次五层 masked-block forward 中为运行配置指定的 K 个位置并行生成候选"*
  (`ninfer docs/maintainer/qwen3.8-27b-dflash2.md:24-26`) e *"没有扩散 timestep 或
  iterative-refinement state"* (`ninfer docs/maintainer/qwen3.8-27b-dflash2.md:427`).

**(b)** **MTP.** O efeito está medido e publicado na mesma tabela, na mesma fixture
(`long_decode_aime26_01`, Qwen3.8-27B `groupwise-int`):

| | aceitação | tokens/rodada | tok/s |
|---|---:|---:|---:|
| MTP3 (`ninfer docs/performance/qwen3.8-27b.md:89`) | **72,5 %** | 3,17 | 193,4 |
| DFlash2 K=7 (`ninfer docs/performance/qwen3.8-27b.md:108`) | **64,6 %** | **5,52** | **224,2** |

**A aceitação por token é MAIOR no MTP3 e ele é mais lento.** O que decide não é a taxa de
aceitação — é **quantos drafts cabem num forward**. Com 7 candidatos por forward contra 3
candidatos por 3 forwards, o DFlash2 ganha mesmo aceitando menos por posição.

Isto reenquadra exatamente a nossa limitação 3. O nosso diagnóstico atual é:
`multiplicador = 2·aceitação / 1,12` com break-even em **56 %** de aceitação, e a conclusão
escrita é que *"o multiplicador do MTP é refém do custo marginal por token do matvec em lote,
não da maquinaria do MTP"* (`nosso README.md:483-497`). A leitura do ninfer acrescenta um
segundo caminho para o mesmo muro: **se o custo por draft cai de "um forward" para
"1/K de um forward", o break-even de aceitação cai junto** — e aí o nosso MTP com 68 % de
aceitação em prosa deixa de estar no fio da navalha.

**(c)** Não é um port: é um drafter novo (um forward por bloco, com máscara). Ordem de
trabalho honesta: (1) §3.10 (ReplaySSM) e o item §3.3, que são **baratos** e atacam o custo que
já temos; (2) só então, se o MTP ainda não pagar, estudar um drafter em bloco — e aí o
`include/rdna4/mtp.cuh` / `mtp_gen.h` precisam de pesos que nós não temos no GGUF, o que é um
**bloqueio de artefato**, não de kernel.

**(d)** Parcialmente, e a medição que decide é barata: medir **ms por draft proposto** do nosso
MTP hoje (é o coeficiente 6,04 ms/token dividido pela janela) contra `1/K` do mesmo. Se a razão
não for próxima de `K`, o drafter em bloco é a explicação; se for, o gargalo é só o matvec e o
§3.3/§3.10 bastam.

---

## 4. O que o ninfer **não** faz, ou faz pior que nós

Esta seção é a lista de "não copie isto", e é onde o repositório é mais útil por ausência.

### 4.1 O **prefill** dele não funde o GQA (o decode funde) — e ali ele é tão redundante quanto nós

Isto é o complemento do §3.5, e é a assimetria mais instrutiva do repo. O caminho de
**decode** funde o grupo (§3.5); o de **prefill** não. `prompt_bf16.cuh:97-103`:

```cpp
const int q_block = static_cast<int>(blockIdx.x);
const int q_head  = static_cast<int>(blockIdx.y);
...
const int kv_head = q_head / Geometry::GroupSize;
```

**Uma cabeça de consulta por CTA**, e o K/V é estagiado por CTA com o `kv_head` derivado
(`causal_prompt_stage_kv(k_s, cache_k, kv_head, ...)`, `:195`). Com
`CausalD256H24Kv4 = AttentionHeadMapping<24, 4, 1>` (`geometry.cuh:15`) o `GroupSize` é **6**
— a **mesma razão 6:1** do nosso modelo (`n_head_kv = 4`, `nosso README.md:172`). Então as 6
cabeças de um grupo estagiam o **mesmo** tile de K/V seis vezes, exatamente como nós
(`docs/estudo-prefill-j-atencao-kv.md:424-428`).

O que ele ganha no prefill e nós não é outra coisa: `Br = 64` linhas de consulta por CTA
amortizam cada linha de KV sobre 64 consultas e trocam a redução por chave por uma redução por
coluna. Ou seja, **o nosso problema de prefill é dele também** — e o fato de o mesmo autor ter
feito a fusão no decode e não no prefill é informação: a fusão é fácil onde o número de linhas
de consulta por CTA já é o grupo inteiro (decode, `T` pequeno) e cara onde o tile é grande
(`Br = 64` linhas × 6 cabeças = 384 linhas de consulta, que estouraria o teto de 48 do
`small_t_bf16.cuh:27` e a LDS). **Para nós, então, o alvo certo é o decode** — que é onde a
nossa limitação 4 está medida (25,77 GB lógicos a 64K).

### 4.2 Não há caminho de KV quantizado *barato* no prefill

Cada formato de KV tem kernel **próprio** para decode e para prefill:
`prompt_{bf16,fp8,i8,k8v4,nvfp4}.cuh` e `small_t{,_bf16,_fp8,_i8,_k8v4,_nvfp4}.cuh` — 12
arquivos pares. É superfície de manutenção grande, e a nossa medição já mostrou que no
**prefill o formato quantizado é custo de instrução, não economia de banda** (e que `f16` no
caminho deles nem passa por LDS, `plano-prefill.md:198-203`). Ou seja: o ninfer paga a
multiplicação de kernels sem colher velocidade no regime que nos importa.

### 4.3 A especialização é por **forma registrada**, não por classe de forma

`q4_dispatch.cpp` cobre `k ∈ {5120, 2048, 1152}` e um punhado de `n`; tudo o mais é
`throw` (`:90`). `q4_small_t_mma.cu:47-50` checa `weight.n == 131072 && weight.k ==
5120|2048` explicitamente. É um desenho válido para "cinco artefatos registrados"
(`AGENTS.md:30-33`) e **incompatível com o nosso** propósito de ler qualquer GGUF: nós
carregamos 14 tipos de quantização por descoberta de arquivo (`nosso README.md:114-138`). Copiar a
tabela seria trocar generalidade por um ganho que só existe nas formas que já temos — o que
para o nosso par de GGUFs até seria defensável, mas é uma decisão explícita, não um port.

### 4.4 Sem preempção, sem prioridade, sem QoS, sem multi-GPU, sem offload

Explícito em `AGENTS.md:35-40` e `nosso README.md:229-236`: concorrência **fixada no startup**
(1..8), FIFO limitada, sem preempção, sem *swapping* de pedido ativo, sem offload de pesos,
sem multi-GPU. Nosso servidor é single-request sem keep-alive e sem prefix-cache
(`nosso README.md:523-526`) — ou seja, aqui **nós somos piores**, mas a lição é o inverso: o
`ResourceManager` deles (`ninfer docs/maintainer/engine-architecture.md:154-176`) existe para gerenciar um cache de
prefixo em duas camadas (Device/Host), e é um subsistema com 3 388 linhas de teste próprio
(`tests/test_resource_manager.cpp`, 3 531 linhas). **Não é um item para copiar agora**; é a
confirmação de que reúso de prefixo é um projeto em si.

### 4.5 O `program_impl.h` de 12 378 linhas

O maior arquivo do projeto é um único header com 12 378 linhas
(`src/targets/qwen3_6/impl/runtime/program_impl.h`) — 19 % de toda a camada de execução do
modelo. É o oposto da nossa estrutura (`graph.cuh`, 2 000-3 000 linhas, com kernels em
headers separados por assunto), e é um aviso sobre até onde a especialização por alvo pode
ser empurrada antes de virar um custo de manutenção.

### 4.6 Onde nós somos melhores (e portanto não devemos mudar por causa deste repo)

| eixo | nós | ninfer |
|---|---|---|
| variedade de quantização | **14 tipos** por descoberta de arquivo (`nosso README.md:114-138`) | 6 formatos registrados, forma fixa |
| gate de exatidão | bit-exatidão em caminhos rápidos **e** PPL ≤ 0,5 % contra llama.cpp (`nosso README.md:342-367`) | tolerância por Op: `relative_l2 = 4.1e-3`, `gross_absolute = 5.0e-6` (`tests/ops/test_gated_delta_net.cpp:26`), sem oráculo externo de PPL no repo |
| medição de quantização | erro **medido** separando aritmética (6,4e-8) de quantização de ativação (3,6e-3) (`nosso README.md:264-283`) | conformance por oráculo por Op (`ninfer docs/maintainer/op-development.md:356-433`), sem estudo de erro acumulado ponta a ponta publicado |
| KV quantizado no 131K | **implementado e medido** (`q5_0`/`q4_1`, 13,73 tok/s, 1,68 GiB livres, `nosso README.md:436-442`) | oferece formatos, mas sem número de contexto extremo equivalente |

---

### 4.7 Ausências que delimitam o que **não** se copia

Verificadas por grep negativo sobre `src include` ou por leitura direta:

- **Sem `preempt` em lugar nenhum do código de produção**: `grep -rin "preempt" src include` →
  **0** ocorrências (a palavra só existe em `AGENTS.md:36,39`, `ninfer README.md:231` e
  `engine-architecture.md:46-47`, sempre como exclusão declarada).
- **Sem grafo de CUDA no prefill.** Só decodificação/MTP/DFlash são capturados
  (`program_impl.h:11130-11226`); `grep -rn "capture_prefill\|prefill_graph" src/targets` →
  **0**. `INFERIDO`: o prefill é sempre eager; **confirmaria** procurando um símbolo
  `capture_prefill`. Para nós isso é informação útil: nós já medimos que replay de grafo vale
  **1,06×** (`nosso README.md:258-262`), então a ausência dele no prefill do ninfer é
  consistente com o nosso resultado, não uma lacuna deles.
- **Sem backend de CPU e sem portabilidade de plataforma.** O build recusa qualquer
  arquitetura que não seja `120a` (`CMakeLists.txt:6-13`) — a mesma escolha que nós fizemos
  (só `gfx1201`, `nosso README.md:529`).
- **Sem quantização de peso em tempo de execução.** O `AGENTS.md:61` proíbe "runtime weight
  repacking" explicitamente; a conversão é 100 % Python offline (`tools/convert/*/convert.py`).
  **A ativação, ao contrário, é quantizada dinamicamente por token** (`src/ops/linear/fp8/fp8_a8.cu:29-67`)
  — a mesma assimetria do nosso motor.
- **Sem quantização assimétrica / com zero-point, sem Q2/Q3, sem NF4, sem dialetos
  GGUF/GPTQ/AWQ** (`ninfer docs/maintainer/tensor-formats.md:727-742`). Isto é o oposto do
  nosso propósito: nós lemos **14 tipos** de GGUF por descoberta de arquivo
  (`nosso README.md:114-138`). É a razão pela qual a tabela de despacho por forma (§4.3) e o
  layout próprio de pesos (§3.11) são **caros** para nós e baratos para ele.
- **Sem `repetition_penalty`**, por decisão escrita no código: *"vLLM exposes
  repetition_penalty, but NInfer's Engine intentionally has no such sampler"*
  (`src/serve/openai_chat_request.cpp:225`), e só `= 1` é aceito (`:234-236`).
- **Sem `/v1/completions`** (rotas enumeradas em `src/serve/http_server.cpp:428-477`) e sem
  execução de tools — as chamadas parseadas voltam ao cliente (`ninfer README.md:235`).

### 4.8 Onde o ninfer está à frente por *processo*, não por kernel

Dois números que valem como referência de método, não de código:

1. **Ele mede a banda efetiva contra uma sonda própria, não contra a especificação.** O
   relatório de benchmark imprime `dram_spec=1792 GB/s` e `sustained_read=1674,5 GB/s`, sendo
   o segundo o resultado medido de uma leitura pura de 4 GiB (`tools/hbm_bandwidth_probe.cu`),
   e reporta o ponto BF16 exato a **89,91-91,80 %** dessa sonda, com a contagem de DRAM do NCU
   conferindo contra o passe lógico único (14 682 572 800 bytes reais contra 14 681 088 000
   lógicos, **+14 848 bytes, sem reprise de peso** —
   `ninfer docs/maintainer/linear-benchmark.md:472-479`). Nós temos o roofline medido (633
   GB/s, `nosso README.md:209`) e estamos a **51 %** dele ponta a ponta — a diferença de
   método é que eles fecham a contabilidade em bytes contados, e nós reportamos % de uma
   medida.
2. **As tolerâncias numéricas são um artefato declarado por perfil**, não um número por teste:
   atenção KV `bf16` = rel-L2 `2,8e-3` / abs `1,0e-3`, `int8` = `3,15e-3` / `1,1e-3`, `fp8` =
   `1,2e-2` / `4,0e-3`, `nvfp4` = `1,5e-2` / `5,0e-3` (`tests/ops/softmax_attention/causal_cache.cpp:43-71`),
   com a regra explícita de que *"token count, geometry, execution envelope, and private launch
   route do not select or relax it"* (`:41-42`), e para Linear um subsídio de quantização por
   perfil de ativação: A16 `1/256`, A8 `0,04`, A4 `0,16` (`tests/ops/linear/linear_test_common.cpp:34-51`).
   Isto é exatamente o que o nosso `docs/plano-prefill.md:205-215` está pedindo para o caminho
   f16 do GEMM ("precisão do f16 precisa de um número, não de uma promessa") — e o ninfer é uma
   implementação de referência de como escrever esse número.

---

## 5. Experimentos concretos para a próxima sessão

Ordenados por (ganho esperado × risco). **Nenhum foi executado**: a GPU é compartilhada e
esta tarefa é leitura de código.

| # | experimento | comando / kernel a escrever | número que move |
|---|---|---|---|
| 1 | **GQA fundido no decode: um CTA por cabeça de KV** (§3.5) — *o item de maior valor* | trocar o eixo x do `attn_split_kernel`/`attn_kernel` de `n_head` para `n_head_kv` e levar as 6 cabeças do grupo para dentro da CTA (`include/rdna4/attn.cuh:331,496-505`), com teto de linhas por CTA | **25,77 GB lógicos → ~4,3 GB** de tráfego de KV a 64K (`nosso README.md:498-507`); a banda efetiva hoje é 165-172 GB/s, 27-29 % do pico (`nosso README.md:444-446`) |
| 2 | **Registro de entradas + fold em vez de snapshot+restore no MTP** (§3.10) | planos `conv/key/value/gate` com `width = draft+1` + kernel de fold que reuse o corpo de `delta_rule`; gate: md5 do `--mtp` == greedy | os **0,54 ms** por par snapshot+restore (`nosso README.md:471-473`) e o 0,96× do `--draft 2` na árvore final (`nosso README.md:475-478`); a razão de tráfego de estado do ninfer é **86,1×** |
| 3 | **Dequant LDS→registrador no laço de consumo** (§3.1) | reescrever o estágio de `gemm_i8_kernel` (`include/rdna4/gemm.cuh:304-332`) para copiar `qs`/`qh`/`scales` em vetores de 16 B e mover `TR::store` para dentro do laço de `kb` (`:384`); gate `check-matmul-gpu`/`check-batch-gpu`/`check-graph-gpu` (bit-exato) | congelar o **50-61 %** de staging (`docs/plano-prefill.md:244`) e a fonte de **132-275 → ?** GB/s contra o roofline de 633 |
| 4 | **Separar bytes-de-global de bytes-para-LDS no estágio** (§3.11d) — barato, e decide o item 3 | contadores (ou um modo de build) no estágio do `gemm_i8_kernel`; comparar a razão com 8/4,25 bits | se a razão for ~2×, o item 3 e o layout valem; se ~1×, os dois morrem **antes** de custar um conversor |
| 5 | **Teto de 8 no tile de registro, e só então subir o chunk** (§3.2) | `bench-matvec-shapes-gpu --batch 8,16,32,64` sobre as 497 formas, reportando ms/token **+ VGPR + `localSizeBytes`** por instanciação | o **+18 % em N=32 / +71 % em N=64** da regressão do D1 (`docs/plano-prefill.md:29-34`); alvo é manter os 62 VGPRs do N=16 |
| 6 | **GDN em blocos com 3 estágios** (§3.4) | três kernels novos em `include/rdna4/gdn.cuh`: `prepare_wy_wu`, `state_passing`, `output`, com bloco 16 (nosso chunk) e workspace alocado no host | **0,852 ms/token** (`docs/plano-prefill.md:100`); no degrau D4 vale ~52 % do que sobra (`:253-255`) |
| 7 | **Política de split em degraus com teto maior** (§3.6) | editar `attn_splits_for` (`include/rdna4/graph.cuh:432-445`) para 4 faixas tipo ninfer (64/128/256/480 chaves por split) com teto 32 e 64; medir com `RD_ATTN_SPLITS` e `scripts/check_attn_split.sh` | **19,0 ms/token** da atenção a 64K (`docs/estudo-prefill-j-atencao-kv.md:453`); desvio na escada 5,1989 → 5,2114 (`nosso README.md:503-507`) |
| 8 | **Tile de atenção de prefill `Br=Bc=64` com K/V na LDS** (§3.5b) | reescrever `attn_batch_kernel` (`include/rdna4/attn.cuh:320-380`) para grade `(ceil(n/64), n_head)` com 49 152 B de LDS | **0,031 → ?** ms/token em 64 chaves e **0,051 → ?** em 128 (`docs/estudo-prefill-j-atencao-kv.md:450-451`); pré-requisito do chunk de 128 |
| 9 | **Kernel small-T dedicado para a verificação de MTP** (§3.3) | `gemm_small_t` com `KWarps=8`, `RowsPerCta=16`, redução por LDS, instanciado para T=2 e T=3 | o coeficiente **6,04 ms/token** de `pass_ms ≈ 13,9 + 6,04·N` (`nosso README.md:483-485`); hoje 8,0 ms por linha verificada (`:471`) |
| 10 | **Ler o diff de `7f14d96` e `ce95491`** (§3.8) | `git show 7f14d96 --stat` e `git log -p` no clone | decide se o hint de L2 na cauda vale 1 dia de trabalho; hoje estamos a **436 GB/s** de 633 (`nosso README.md:209`) |
| 11 | **Medir ms por draft proposto do nosso MTP** (§3.13) — decide se o problema é o matvec ou a forma do drafter | cronometrar o laço de draft do `--mtp` e dividir pela janela; comparar com `1/K` de um forward | decide entre investir em §3.3/§3.10 (barato) ou num drafter em bloco (que exige **pesos novos**, bloqueio de artefato) |
| 12 | **Contar as arestas do grafo com dependência parcial** (§3.7) | script que, por kernel, compara o conjunto de buffers escritos pelo produtor com os lidos pelo consumidor | **~4,0 ms/token** de despacho (`nosso README.md:258`); decide se o PDL vale um protótipo |

**O que este repo sugere que NÃO se faça**, com o número que já foi medido por nós: fusão de
kernels para o prefill (1,1 %, `plano-prefill.md:256`), duplo buffer de ativação (−17 %,
`plano-prefill.md:257`), ampliar o tile em M para matar a cauda de onda (−13 %,
`:257-258`), staging LDS da ativação (−5 %, `:258-259`). Nada no ninfer contradiz esses
resultados: ele **não** funde kernels em massa (usa PDL), e o duplo buffer dele é **só do
peso** (`q4_rowsplit_gemm_simt.cuh:35,46-49`).

---

## 6. O que não foi determinado

Marcado explicitamente, porque uma leitura de código tem limites — e porque a diferença entre
"li" e "medi" é a diferença entre um estudo e um palpite.

### 6.1 Limites de método (valem para o documento inteiro)

1. **Nenhum número foi executado.** Isto é leitura de código; a GPU é compartilhada e não foi
   tocada. Os 3 218 tok/s são a medição **publicada** do autor
   (`ninfer docs/performance/qwen3.6-27b.md:49`), não medida nossa. O hardware dele é um
   RTX 5090 (`ninfer README.md:25-28`) e o nosso um RX 9070 XT: qualquer razão entre os dois é
   comparação de **arquitetura**, nunca de máquina, e nenhuma razão de tok/s aparece neste
   documento.
2. **O custo de cada técnica no gfx1201 não é estimável a partir daqui.** O caso agudo é o
   §3.1: eu não medi quantas instruções o `TR::store` custa hoje nem quantas custaria o estágio
   de código cru. Os números 396/590 são a medição da frente G para **outra** comparação
   (`docs/plano-prefill.md:247`) e estão citados como tal.
3. **Toda contagem de ocorrência é de `grep`**, não de um build. Os "0 ocorrências" de cuBLAS,
   CUTLASS, Triton, `wgmma`, `tcgen05`, `nvcuda::wmma`, `__dp4a` e `preempt` são greps sobre
   `src include apps` — não compilei o projeto (não tenho CUDA 13.1 nem `sm_120a`). Um
   `#include` transitivo ainda poderia puxar alguma coisa; **confirmaria** com
   `cmake --build` e `ldd`/`nm` nos binários.
4. **Duas afirmações estruturais vieram de subagentes e eu reverifiquei as que sustentam
   conclusão** (o GQA do decode, o `kChunkSize = 64`, o limiar `T ≤ 16`, os 99 KB de LDS, o
   ReplaySSM, o RNG sem estado, as tolerâncias). As que eu **não** reverifiquei linha a linha
   são as de infraestrutura de servidor e de teste (§4.7, §4.8) — são descritivas e não
   carregam nenhuma recomendação.

### 6.2 O que ficou aberto, e o que confirmaria cada um

| # | aberto | o que confirmaria |
|---|---|---|
| 1 | **PDL no HIP.** Não verifiquei se o ROCm 7.2 expõe equivalente a `cudaGridDependencySynchronize`. `INFERIDO` que não. | `grep -rn "griddepcontrol\|ProgrammaticLaunch" /opt/rocm/include` |
| 2 | **A matemática do caminho `chunked` do GDN.** Li a decomposição em três estágios, o workspace e os nomes, mas **não** li `prepare_wy_wu.cuh` / `state_passing.cuh` / `output.cuh` termo a termo. `INFERIDO` que é a `chunk_delta_rule` padrão (WY por bloco). | ler `prepare_wy_wu.cuh:354-722` e comparar com a nossa `delta_rule` (`include/rdna4/gdn.cuh:57-108`) |
| 3 | **O épilogo do `small_t_mma`.** Li a estrutura, a geração de lançadores e a redução por LDS, mas não conferi `Q4SmallTMmaStoreEpilogue` nem o `RowPolicy` (`q4_small_t_mma.cuh:14-22`). | leitura do epílogo + `tests/ops/linear/` |
| 4 | **DFlash / DFlash2 não foram estudados nos kernels.** Sei o que a documentação afirma (K=1..15, *masked-block forward* único, sem refinamento iterativo, contexto local cíclico de 5 camadas, seletor de top-16) e vi os números de aceitação publicados, mas **não li** `speculative_round.cuh` (34 645 B) nem `dflash_impl.h`. Se o nosso MTP continuar sem pagar depois do §3.10, é o próximo lugar a olhar — e é um estudo separado. | ler `dflash_impl.h:242-330` e `candidate_selector_path.cu` |
| 5 | **O contêiner `.ninfer`.** Sei que não é um header C++ com tabela de seções, mas **não** abri um arquivo. O que decide para nós de qualquer forma é o layout dos planos (§3.11), e esse eu confirmei no código. | `python3 tools/artifact/inspect.py <artefato>` |
| 6 | **Nada sobre o MoE.** O 35B-A3B tem caminhos dedicados (`src/ops/sparse_moe/{decode,prefill,small_t}/`) com PDL e pesos estagiados, e é onde o ninfer publica os números mais altos (642,5 tok/s a C=1, `ninfer README.md:134`; 17 705,4 tok/s de prefill a 7 680 tokens, `ninfer docs/performance/qwen3.6-35b-a3b.md:66`). O nosso modelo não é MoE, então não abri a frente. | — |
| 7 | **Nenhum contador de tráfego de KV existe no repo** (a única instrumentação de atenção são ranges NVTX e um *timing recorder*; não há `atomicAdd` de bytes nem buffer de contador no workspace da atenção, que só tem `partial_acc/m/l`). Isto é um **fato negativo verificado**, e é relevante: significa que a alegação "cada byte de K/V é lido uma vez" do §3.5 é **estrutural (do mapeamento de grade)**, não uma medição publicada por eles. Do nosso lado, a medição equivalente existe e é a que cito (25,77 GB contra 4,295 GB, `nosso README.md:498-507`). | ncu `dram__bytes`/`lts__t_sectors` sobre `causal_attention_small_t_*` num 5090 |

### 6.3 O que **não** fica em aberto

Para não deixar dúvida sobre o que está firme: os cinco itens do topo do §3 têm, cada um, o
mecanismo lido no código e o gargalo nosso medido no nosso próprio repo. O item de maior valor
(§3.5, GQA fundido no decode) tem a grade, o índice de cabeça, o mapeamento de linhas e o
comentário de projeto do autor, todos citados — e a razão 6:1 dele é **idêntica** à nossa
(`geometry.cuh:15` contra o nosso `n_head_kv = 4`, `nosso README.md:172`), o que faz dele uma
comparação de igual para igual em vez de uma analogia.
