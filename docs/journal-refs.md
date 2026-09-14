# Diário — frente `refs` (levantamento de referências)

Nota de formato: a regra §5 de `docs/noite-regras.md` pede, por experimento,
**Referência / Hipótese / Comando / Resultado / Veredito**. Esta frente é de **leitura** — não
toca na GPU e não muda o motor —, então o campo **Comando** registra "nenhum" (com o comando que
*outra* frente deve rodar, quando existe) e o campo **Resultado** registra o que foi extraído,
com a etiqueta de confiança (**[V]** li o código/doc, **[R]** fonte externa relata, **[E]**
extrapolação minha). O campo **Veredito** é o veredito de **aplicabilidade** ao nosso perfil de
restrição (APLICA / APLICA COM VOLTA / NÃO APLICA) e o que ele implica para a noite.

O levantamento consolidado está em `docs/referencias-noturnas.md`.

---

## 1. llama.cpp — quantização do KV: layout dos blocos e kernels de store/load

- **Referência**: `ggml/src/ggml-common.h:195-256` (`block_q4_0/q4_1/q5_0/q8_0`),
  `ggml/src/ggml-quants.c:113,150,187,222-226,276,479,500` (refs quant/dequant),
  `ggml/src/ggml-cuda/cpy.cu:299-346,515-525` (store f32 → `q4_1`/`q5_0`).
- **Hipótese**: o `q5_0`/`q4_1` pode ser acrescentado ao `kv.h` mantendo **byte-identidade**
  com o llama.cpp (como já foi feito com `q4_0`/`q8_0`), preservando oráculo e PPL como rede.
- **Comando**: nenhum (leitura). Gate que a frente KV deve rodar depois:
  `./scripts/compare_llama_greedy.sh 32` com `--kv-k q5_0 --kv-v q4_1`.
- **Resultado** **[V]**: `q4_0` 18 B/bloco, `q4_1` 20 B (`d`+`m` fp16, **afim**), `q5_0` 22 B
  (`d` + `qh[4]` + nibbles), `q8_0` 34 B — uma escala por 32 elementos em todos.
  `quantize_row_q5_0_ref` usa `d = max/-16` com o sinal do maior `|x|` e
  `xi = MIN(31,(int8_t)(x*id+16.5))`; o 5º bit vai para `qh` por `memcpy` de um `uint32_t`
  (ordem little-endian — é a armadilha de transplante). O store do upstream é um kernel
  "bloco de 32 por thread", igual ao nosso `kv_store_row_kernel` (`kv.h:186`).
- **Veredito**: **APLICA** (implementação direta, risco baixo). Armadilha registrada para a frente
  KV: o `qh` do `q5_0` tem ordem de bits que um transplante "equivalente" erra por 1 bit.

## 2. llama.cpp + medição própria — `q8_0` vs `q4_0`/`q5_0` na atenção: a política de KV

- **Referência**: `docs/medicoes-banda-e-gargalos.md:485-493` (medido na GPU, `bench-attn-gpu`,
  64K, 16 camadas); `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl:22-25,57-77,126-138`
  (`FA_DEQUANT4_Q4_1`/`FA_DEQUANT4_Q5_0`, K e V com tipos independentes);
  `src/llama-context.cpp:3697-3730` (K ≠ V é permitido fora de MLA/DeepSeek4; `head_dim % 32`).
- **Hipótese**: a hipótese da noite (`K q5_0` / `V q4_1`) é a melhor troca de VRAM por qualidade
  **e** a mais rápida das opções de 4-5 bits.
- **Comando**: nenhum (leitura). Para as frentes:
  `for kv in q4_0 q4_1 q5_0 q8_0; do timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu $kv 16384 65536 131072 --no-sweep --ab-rounds 15; done`
  e, no contexto real, `./build/bench --start-pos 131000 --tokens 8 --kv-k q8_0 --kv-v q4_0`
  contra `--kv-k q5_0 --kv-v q4_1`.
- **Resultado** **[V]**: a 64K, `q8_0` faz a atenção em **19,87 ms (690 GB/s emitidos)** contra
  **21,03 ms** do `q4_0` (346 GB/s), com **quase o dobro do tráfego**; o token inteiro fica
  54,45 × 55,02 ms. Ou seja: **o gargalo é o desempacotamento por elemento, não os bytes** — e o
  `q5_0` carrega o mesmo desempacotamento **mais** um gather de 1 bit do `qh`.
  **[E]** ⇒ `K q8_0 / V q4_0` (3,25 GiB a 131K) tende a ser **igual ou mais rápido** que
  `K q5_0 / V q4_1` (2,625 GiB) e é mais preciso no K; o preço é +0,6 GiB.
- **Veredito**: **APLICA COM VOLTA** — implementar `q5_0`/`q4_1` (é a única forma de medir a
  hipótese e o par é de primeira classe no upstream), mas **medir `q8_0` no K como candidato de
  referência**, não tratá-lo como exótico. A escolha final é de número, não de opinião.

## 3. llama.cpp — MMQ/prefill: a condição de seleção e a fronteira `ne11 > 8`

- **Referência**: `ggml/src/ggml-cuda/mmq.cu:225-249,266-392` (`ggml_cuda_should_use_mmq`),
  `mmq.cuh:8,111-119`, `mmvq.cu:318-411`, `mmvq.cuh:3`.
- **Hipótese**: MMQ tiled é o caminho certo para o prefill e o `ne11 > 8` do briefing é a
  fronteira GEMV → GEMM.
- **Comando**: nenhum (leitura). Protótipo autorizado por `docs/noite-regras.md:60-61`; gates
  `./build/check-batch-gpu` (bit-exato) e `./scripts/check_regression.sh`.
- **Resultado** **[V]**: em **RDNA4 o `should_use_mmq` devolve `true` incondicional**
  (`mmq.cu:380-383`, com o link do PR #18537) — o upstream **não** usa "desquantizar + hipBLAS"
  em RDNA4. A fronteira dos 8 é do **GEMV**: `MMVQ_MAX_BATCH_SIZE 8` (`mmvq.cuh:3`),
  `ne11 <= 8` (`mmvq.cu:411`); `MMQ_DP4A_MAX_BATCH_SIZE 64` é só NVIDIA-com-tensor-core. O tile
  é `NE_K 32` com ativações **`q8_1`** quantizadas uma vez por chunk.
  **Correção de alvo**: a 70 tok/s o passe de pesos do prefill é ~1,1 ms/token
  (11,12 GB / 16), enquanto o prefill medido é **13,7 ms/token** (`docs/medicoes-m8.md:44-49`)
  ⇒ o gargalo do prefill **não é o matvec**, é o andaime por token.
- **Veredito**: **APLICA COM VOLTA** — copiar o desenho do tile (ativação `q8_1` 1×/chunk), mas
  o ganho grande do prefill está no item 12 (andaime batelado), não no MMQ isolado.

## 4. llama.cpp — MMVQ multi-coluna: o desenho de referência do *verify* batelado

- **Referência**: `ggml/src/ggml-cuda/mmvq.cu:726-735` (laço `for j < ncols_dst` reusando o bloco
  de pesos `vx`), `mmvq.cu:436-490` (tabela de ocupação por N: **RDNA4 quer `nwarps=8` com
  `ncols=1`** e 1 warp com N > 1), `include/rdna4/matvec.cuh:590-635` (o nosso
  `matvec_launch_batch`, N = 2..16, já existe e é bit-exato).
- **Hipótese**: verificar D rascunhos custa muito menos que D passes, porque a desquantização do
  peso é amortizada entre as colunas.
- **Comando**: nenhum (leitura). O número já está medido em `tests/check_batch_gpu.hip`; o
  complemento no modelo real é o item 12.
- **Resultado** **[V]**: o upstream tem exatamente o desenho que queremos (`ncols_dst` como
  parâmetro de template) e **retuna a ocupação por N**; nós já temos a peça e a curva (§12).
  Pista de graça para o decode: a tabela do RDNA4 pede **8 warps/CTA** para `q4_0/q5_0/q8_0` com
  uma coluna, e o nosso matvec mede 436/633 GB/s (issue-bound).
- **Veredito**: **APLICA** — o laço multi-coluna é o coração do ganho do MTP e do prefill; a
  varredura `--rows/--minb/--ilp` do nosso bench é o primeiro passo barato.

## 5. llama.cpp — fusões nomeadas do Vulkan + HIP graph

- **Referência**: `ggml/src/ggml-vulkan/ggml-vulkan.cpp:689-691,794-798,18153-18315`;
  `tests/bench_matvec_shapes_gpu.hip:359-404` (o nosso instrumento, já pronto, que imprime
  "ms/token of launch tax recoverable"); `docs/medicoes-banda-e-gargalos.md:523-524`
  (~1440 lançamentos pequenos ≈ 4,0 ms + 1,1 ms de latência de norma).
- **Hipótese**: fundir na ordem da lista do upstream recupera 4-7% do decode e mais no prefill.
- **Comando**: `timeout 900 ./scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MOD --reps 3`
  (janela limpa + `mem_info_vram_used` antes/depois, por §1.6 das regras).
- **Resultado** **[V]**: a lista aplicável é `SSM_CONV_BIAS_SILU`/`SSM_CONV_SILU` (`:18240,18248`),
  `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` (`:18188`, funde norm+rope **+ escrita das linhas de KV**),
  `SILU_MUL`/`SIGMOID_MUL` (`:18231-18234`), `MULTI_ADD` (`:18153`), `MUL_MAT_ADD` (`:18157`).
  **[R]** ressalva medida: HIP graph move custo de **submissão**, e os gaps parecem ser
  GPU-side (llama.cpp #6763) — o PR #11867 mede só **+3,6%** num 8B no MI100.
- **Veredito**: **APLICA** (baixo risco, bit-exato se a ordem for preservada), com a ressalva de
  que o valor do grafo deve ser **medido** antes de apostar nele; a fusão de kernels tem valor
  próprio (a barreira que se apaga).

## 6. llama.cpp — FA com KV quantizado: a armadilha do f16 e o shader `FA_DEQUANT4`

- **Referência**: `ggml/src/ggml-cuda/fattn.cu:459,583-695,707-722`,
  `ggml/src/ggml-cuda/fattn-common.cuh:53-82,1010`; shader citado no item 2.
- **Hipótese**: nós já estamos no desenho certo (desquantizar inline) e não devemos regredir.
- **Comando**: nenhum (leitura).
- **Resultado** **[V]**: `TILE` e `MMA_F16` **exigem K/V em f16**; com cache quantizado o backend
  escolhe `VEC` só para `Q->ne[1] <= 2` e, quando escolhe `TILE`/`MMA`, **materializa uma cópia
  f16 do cache inteiro** por chamada. **[E]** a 131K isso é `2 × 131072 × 1024 × 2 B = 536 MB`
  por camada ⇒ **~8,6 GB por passe de prefill** em 16 camadas. O shader do Vulkan mostra o padrão
  certo (`FA_DEQUANT4_Q5_0`: `hb = ((qh >> iqs) & 1) * 16`, valor `d*(nib + hb - 16)`).
- **Veredito**: **APLICA** como aviso — a frente de prefill **não** deve materializar KV em f16;
  desquantizar dentro do laço (é o que já fazemos) também no caminho batelado.

## 7. llama.cpp — MTP do `qwen35` e o driver `draft-mtp` (incl. planos de estado)

- **Referência**: `src/models/qwen35.cpp:485-644`; `common/speculative.cpp:750,1330-1770,2903-2990`;
  `common/common.h:326-332,394-400`; `src/llama-memory-recurrent.cpp:101,193-203`;
  `src/llama-arch.cpp:1104-1120`.
- **Hipótese**: o nosso `MtpHead` já espelha o grafo do upstream, e o que falta é
  **instrumentação e rollback**, não arquitetura.
- **Comando**: nenhum (leitura).
- **Resultado** **[V]**: grafo = `concat([e_norm; h_norm]) → eh_proj → 1 bloco de atenção plena
  com KV próprio → norma compartilhada → head compartilhada`, com `h_nextn` exposto (`:543-547,
  626-639`), exatamente o nosso (`docs/mtp.md:43`: `[e;h]` = 96,9% × `[h;e]` = 0,0%).
  Do driver: `n_max` **default 3**, `n_min` 0, **`p_min`** corta o encadeamento (`:1690-1700`);
  o `accept()` lê a linha `min(n_accepted, n_rows-1)` da `h` da verificação (`:1753-1766`) ⇒
  **o tronco tem de devolver a `h` pré-norma de TODAS as posições verificadas**; a truncagem de
  KV na rejeição é `llama_memory_seq_rm` (`:1560-1568,1652-1660`); a estatística por posição
  (`#acc rate/pos`, `:2903-2990`) é o que **nos falta**. Rollback do estado:
  **`n_rs_seq` planos + movimento de índice, sem `memcpy`** (`llama-memory-recurrent.cpp:101,
  193-203`), dimensionado por `draft.n_max` (`common/common.h:394-400`).
- **Veredito**: **APLICA** — é a lista de tarefas concreta do agente da verificação batelada
  (estatística por posição, `p_min`, exportar a `h` de todas as posições, planos por índice).

## 8. vLLM — PagedAttention, chunked prefill, prefix caching, continuous batching

- **Referência**: Kwon et al., SOSP'23 (<https://arxiv.org/abs/2309.06180>), docs do vLLM;
  Sarathi-Serve.
- **Hipótese**: a maior parte disso é máquina de multi-inquilino.
- **Comando**: nenhum (leitura).
- **Resultado** **[V]**: o próprio paper mede o custo da indireção — *"+20-26% higher attention
  kernel latency"* (§7.1) — e a sua correção é **fundir para cortar lançamentos** (§5.1), que é o
  nosso problema. Blocos de 16 tokens; fragmentação externa é **0** com uma sequência **[E]**.
  Chunked prefill: ganhos são de concorrência sob SLO, com custo explícito em TTFT, e cada chunk
  é outra passada sobre 11,12 GB.
- **Veredito**: **NÃO APLICA** para PagedAttention/continuous batching/chunked prefill (vazão).
  **APLICA** só o prefix caching (= §9, com o twist do GDN) e a lição de fusão.

## 9. ExLlamaV2 / SGLang / TRT-LLM — o que sobra para um usuário só

- **Referência**: `exllamav2/exllamav2_ext/cuda/cache_q.cuh` + `cache.cu` + `doc/qcache_eval.md`;
  <https://arxiv.org/abs/2312.07104> + `mamba_radix_cache.py`; docs do TRT-LLM (XQA,
  `fuse_gemms_mixed_children`, multi-CTA MHA).
- **Hipótese**: sobra a rotação de Hadamard, o prefixo de 1 slot e as fusões de contagem de
  lançamento.
- **Comando**: nenhum (leitura).
- **Resultado** **[V]**: **ExLlamaV2 não usa E4M3 (escalas são fp16) e não tem cache Q5**; o
  achado é o **`HADAMARD_Q`** (borboleta de 5 estágios antes de quantizar; `doc/qcache_eval.md`:
  Mistral-7B 13,41 → 13,37; "Q4 é mais preciso que FP8"). O llama.cpp adotou o mesmo
  (PR #21038; verificado em `src/llama-graph.cpp:2855-2861,2891` e `src/llama-kv-cache.cpp:23,
  315-334`). O **RadixAttention** tem overhead desprezível (<0,3%) mas é hostil ao nosso
  híbrido: `new_node.mamba_value = None # mamba cache can not be split`, COW do estado, e um
  checkpoint bf16 = 78,4 MB ≈ **4 356 tokens do KV `q4_0` inteiro**. No TRT-LLM, in-flight
  batching e XQA não aplicam; as fusões de contagem de lançamento valem ~3-4% combinadas; o
  **MHA multi-CTA** é o único item que mapeia no nosso gargalo (e coincide com o item 4 do top-5).
- **Veredito**: **APLICA** (rotação de Hadamard — maior valor esperado para o KV; fusões de
  lançamento; GQA/CTA), **APLICA COM VOLTA** (prefixo de 1 slot, só no servidor),
  **NÃO APLICA** (in-flight batching, XQA, fusions de tensor-parallel).

## 10. Literatura especulativa: DeepSeek MTP, EAGLE-1/2/3, Medusa, árvore + rollback

- **Referência**: <https://arxiv.org/abs/2412.19437> (§2.2, §5.4.3); EAGLE 2401.15077 /
  2406.16858 / 2503.01840; Medusa 2401.10774; Leviathan 2211.17192; vLLM RFC #49232; MagicDec
  2408.11049; AdaEDL 2410.18351; Sequoia 2402.12374.
- **Hipótese**: a nossa cabeça MTP é a linha "heads" da Medusa (~1,5×) e o que falta é a
  verificação batelada; a árvore é um multiplicador que **pode não pagar** no nosso caso.
- **Comando**: nenhum (leitura).
- **Resultado** **[V]**: DeepSeek-V3: 2 módulos **sequenciais**, aceitação do 2º token **85-90%**,
  **1,8× TPS** — e é uma afirmação de **D=2**, não de cadeia longa. Medusa tabela 3:
  **heads ~1,5× → +árvore ~1,9× → +árvore otimizada ~2,2× → +treino ~2,8×**. Medusa varre
  **γ = 4 (Vicuna-7B) / 3 (13B)** e Chen et al. dizem que o ganho *"plateaus or even regresses"*
  além de K = 4 — casa com a nossa varredura (D=4 melhor, D=8 pior, §12). Leviathan: se **α > c**
  há ganho garantido (Cor. 3.9); com os nossos medidos **α₁ = 0,867 > c ≈ 0,58**, logo o ganho
  existe, mas é modesto. Rollback: **planos `1+k` com movimento de índice** (llama.cpp) × **cópia
  de volta** (vLLM) × **cache de entradas / ReplaySSM** (SGLang, **só cadeia linear**); o
  ReplaySSM mede **~1,0× em B=1** — o nosso regime. **Cada nó de árvore custa ~65 KB de KV
  (barato) e 151 MB de plano de estado (fatal).**
- **Veredito**: **APLICA** (MTP, planos por índice, estatística por posição, `p_min` como
  "confiança ≈ aceitação"); **NÃO APLICA nesta noite** (fusão de features do EAGLE-3 — exigiria
  treinar; árvore antes de medir o cf real).

## 11. Literatura de qualidade de KV: K vs V, e por que PPL engana

- **Referência**: KIVI 2402.02750; KVQuant 2401.18079 (+ Apêndice N); KVTuner 2502.04420;
  OTT 2505.10938; KV-AdaQuant 2502.15075; AsymKV 2410.13212; Atom 2310.19102 (negativo);
  tabela de KLD do PR #21038 (Qwen3.5-9B); discussão #21297; avaliação AIME25 (ggerganov).
- **Hipótese**: "K precisa de mais bits que V" e a nossa hipótese (`K q5_0`/`V q4_1`) é o melhor
  ponto por byte.
- **Comando**: nenhum (leitura). Para a frente KV:
  `./scripts/compare_ppl.sh $MOD 10` **e** uma medida de **KLD média** (a criar) a 8K/32K, mais
  uma sonda needle a ≥ 64K.
- **Resultado** **[V]**: a tabela de KLD do upstream em **Qwen3.5-9B** dá `V q4_1` < `V q4_0` em
  **6/6 linhas** e `K q5_0` < `K q4_1` em **todas** (25-40%); a 4-5 bits **os dois lados
  importam** e o K é ~1,5× mais sensível (não 10× — isso é fenômeno de 2 bits). O achado que
  muda o nosso cálculo: a KIVI é sobre **granularidade**, e o Apêndice N do KVQuant mostra que,
  para quantização **por token** (o nosso caso), **pós-RoPE é melhor** que pré-RoPE ⇒ a história
  de outlier por canal **não se transfere** para nós. **A PPL é cega**: no AIME25, `q4_0` no KV
  cai de **37,9% para 2,0%** enquanto a PPL anda ~0,4%. E o llama.cpp **roda rotação de Hadamard**
  desde 2026-04 ⇒ os números publicados de qualidade de KV **não são o nosso regime**.
- **Veredito**: **APLICA** (manter `K q5_0`/`V q4_1` como default; **não** usar `q4_1` no K;
  subir para `K q8_0`/`V q4_1` se a VRAM sobrar; **medir a rotação de Hadamard**; aceitar por
  KLD/needle, não por PPL). **NÃO APLICA nesta noite**: K por canal e pré-RoPE (quebram o layout
  e o oráculo, e o próprio KVQuant mostra que perdem no regime por token).

## 12. Cross-check interno — a curva de custo do batch **já está medida** (e a peça compartilhada)

- **Referência**: `tests/check_batch_gpu.hip` + `docs/medicoes-m8.md:25-34` (curva, **bit-exata**,
  com o **modelo real**: `scripts/check_all.sh:21,79`); `docs/autotuning-gfx1201.md:157-161`
  (mesma curva em ms/token); `docs/medicoes-m8.md:18-21,47-49,75-80,90-97` (o que ainda é
  sequencial, projeção 1,3-1,5×, e a variante simples de rollback medida em 1,0-1,1×);
  `docs/mtp.md:189-197` (a projeção de 2,5×); `include/rdna4/matvec.cuh:590-635`.
- **Hipótese**: o ganho do MTP batelado é ~2,5× (como `docs/mtp.md` projeta) e o limite é a
  atenção/GDN.
- **Comando**: nenhum (leitura). Experimento que fecha a conta no modelo real: estender o bloco
  `REPLAY` de `tests/bench_matvec_shapes_gpu.hip:342-357` para `matvec_launch_batch(..., N)` com
  N = 2,3,4,8 e rodar
  `timeout 900 ./scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MOD --budget-mib 12000 --reps 3`.
- **Resultado** **[V]**: cf(N) = N/speedup medido: **N=2 → 1,16; N=3 → 1,68; N=4 → 1,75;
  N=8 → 3,27; N=16 → 6,11** (2 tokens por 1,16× o preço de um; 4 por 1,75×); em ms/token:
  16,3 / 7,7 / 4,9 / 1,9 / 0,87. **Ressalva medida, não suposta:** o gate roda em **contexto
  curto** e no `forward_batch` a atenção, o KV write e a recorrência do GDN continuam
  sequenciais (`docs/medicoes-m8.md:18-21`) ⇒ esses cf valem para a parte batelada e são
  **limite inferior** para a passada de tronco inteira. Aplicando-os à projeção do próprio repo
  (que já inclui atenção/GDN): **D=2 → 1,31×; D=3 → 1,26×; D=4 → 1,51×; D=8 → 1,34×**. Ou seja,
  a projeção de **2,5×** de `docs/mtp.md:196` assume cf = 1,0 e é **otimista por ~1,7×**; o
  melhor ponto é **D=4**, e o D=8 é pior que o D=4. **E a peça que falta é a mesma para o MTP e
  para o prefill**: atenção multi-linha com máscara causal + scan de GDN multi-token
  (`docs/medicoes-m8.md:18-21` lista exatamente os três itens sequenciais; a 70 tok/s o peso por
  token do prefill é 1,1 ms e o medido é 13,7 ms ⇒ o prefill é andaime).
- **Veredito**: **APLICA — e é a conclusão mais importante deste levantamento para a
  coordenação.** (a) Ajustar a expectativa do MTP de 2,5× para **1,3-1,5×** e apontar as frentes
  para D=4; (b) **a árvore especulativa não paga antes de baixar o cf**; (c) **dar dono único** à
  atenção multi-linha e ao scan de GDN multi-token, que são o caminho crítico de duas frentes ao
  mesmo tempo (MTP 1,5×; prefill de 73 para 200-400 tok/s **[E]**).
