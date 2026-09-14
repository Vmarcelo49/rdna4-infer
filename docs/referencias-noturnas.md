# Levantamento de referências — frente `refs` (noite 2026-09-14)

**Objetivo.** Tirar técnicas de outros motores de inferência e da literatura e, para cada uma,
dizer: o que é, de onde vem (URL e/ou `arquivo:linha` no checkout local do llama.cpp em
`/home/marcelo/Projetos/llama.cpp`, commit `df03399b8`), **se aplica ao nosso perfil de
restrição**, **ganho esperado** (número ou faixa, com a marca de medido vs extrapolado),
**custo/risco**, e **o experimento exato** que resolveria a dúvida.

**Esta frente não usou a GPU.** Nenhum comando com `gpu-lock.sh` foi executado. Todo número
citado como "medido" vem de um documento deste repositório (com `arquivo:linha`) ou de fonte
externa; os comandos propostos são para as outras frentes. Escrevi o diário enquanto lia
(`docs/journal-refs.md`), não no fim.

## Como ler as etiquetas

| etiqueta | significado |
|---|---|
| **[V]** | **verificado** — li o código/documento e a citação está transcrita aqui |
| **[R]** | **relatado** — uma fonte externa afirma; não reproduzi |
| **[E]** | **extrapolação minha** — aritmética ou inferência sobre números medidos |

Perfil de restrição que todo veredito usa (não repito em cada item):

- **1 GPU**: RX 9070 XT, 16 GB (15,92 GiB), gfx1201/RDNA4. **Sem MFMA**; `wmma i32_16x16x16_iu8`
  existe; dot inteiro = `sudot4`; `v_dot2_f32_f16`/`v_pk_fma_f16` a 1,5× o FMA fp32.
- DRAM medida **632,9-634,5 GB/s** (`docs/medicoes-banda-e-gargalos.md:526`); Infinity Cache
  ~1,5 TB/s; piso de despacho **2,2-3,6 µs/lançamento**.
- Qwen3.8-27B denso, **48 camadas GDN + 16 de atenção plena**, GQA **6:1** (24 cabeças de
  consulta, 4 de KV), `head_dim=256`; MTP/NextN (`blk.64.*`) implementado, aceitação 86,7%.
- Decode é GEMV (M=1) ⇒ **matriz não ajuda decode**; só prefill (N ≥ 8-16).
- Hoje: decode **29,3 tok/s** no fim de 4K; prefill **72,9 tok/s** a 512 (llama.cpp Vulkan:
  1143); 131K a **14,0 tok/s** com KV `q4_0`.
- Orçamento do token a 4K (`docs/medicoes-banda-e-gargalos.md:519-524`): matvec tronco
  23,51 + head 1,43 = **24,94 ms a 436 GB/s**; atenção 1,38; GDN 5,63 (3,97 de `delta_rule`);
  normas/rope 2,96; `act_quant` 0,88; sampler 0,44; cópia/dreno 0,23; **~1440 lançamentos
  pequenos ≈ 4,0 ms**; **1940 lançamentos/token**.

---

# 1. llama.cpp — o que copiar, o que evitar

## 1.1 Quantização do KV: layout dos blocos e kernels de store/load

**O que é.** O KV quantizado do llama.cpp é *stored quantized*: a linha de `head_dim` elementos
é guardada em blocos de 32 com **uma escala fp16 por bloco** (e, no `q4_1`/`q5_1`, também um
**min fp16**), e a atenção **desquantiza na hora** — exatamente o nosso desenho (`kv.h:2-11`).

**Layout exato [V]** (`ggml/src/ggml-common.h`):

| tipo | struct | bytes/bloco | bytes/elem | conteúdo |
|---|---|---|---|---|
| `q4_0` | `ggml-common.h:195-199` | 18 | 0,5625 | `d` fp16 + 16 B de nibbles, valor = `d*(nib-8)` |
| `q4_1` | `ggml-common.h:201-212` | 20 | 0,625 | `d`+`m` fp16 + 16 B nibbles, valor = `d*nib + m` (**afim**) |
| `q5_0` | `ggml-common.h:228-235` | 22 | 0,6875 | `d` fp16 + `qh[4]` (5º bit) + 16 B nibbles, valor = `d*(nib + 16*bit5 - 16)` |
| `q8_0` | `ggml-common.h:252-256` | 34 | 1,0625 | `d` fp16 + `int8 qs[32]`, valor = `d*q` |

**Quantizadores de referência [V]**: `quantize_row_q4_1_ref` (`ggml-quants.c:150`), escala
`d=(max-min)/15`, `m=min`; `quantize_row_q5_0_ref` (`ggml-quants.c:187`), `d=max/-16` com o
sinal vindo do maior `|x|`, `xi=MIN(31,(int8_t)(x*id+16.5))`, e o 5º bit empacotado em `qh`
(`qh |= ((xi&0x10)>>4) << j`, `ggml-quants.c:222-226`). Desquantizadores:
`dequantize_row_q4_1` (`ggml-quants.c:479`), `dequantize_row_q5_0` (`ggml-quants.c:500`).

**Kernels de store [V]**: `ggml_cpy_f32_q4_1_cuda` / `ggml_cpy_f32_q5_0_cuda`
(`ggml/src/ggml-cuda/cpy.cu:299-346`), despachados em `cpy.cu:515-525`. Um `cpy` = um kernel
"bloco de 32 por grupo de threads" — igual ao nosso `kv_store_row_kernel` (`kv.h:186`).

**Veredito: aplica direto.** Transcrevendo `quantize_row_q5_0_ref`/`quantize_row_q4_1_ref`
(como já foi feito para `q4_0`/`q8_0`, `kv.h:211-235`), a linha de cache continua
**byte-idêntica ao llama.cpp** e o oráculo/`PPL` continuam valendo. **Risco** baixo, com uma
armadilha conhecida: o `qh` do `q5_0` é copiado por `memcpy` de um `uint32_t`
(`ggml-quants.c:224-226`) — a ordem dos bits é a do inteiro little-endian, e é aí que um
transplante "equivalente" erra por 1 bit (o mesmo erro que `kv.h:211-214` documenta para o
`q4_0`). **Experimento:** estender o gate que compara linha armazenada × espelho host a
`Q5_0`/`Q4_1` e rodar `./scripts/compare_llama_greedy.sh 32` com `--kv-k q5_0 --kv-v q4_1`.

## 1.2 KV quantizado na atenção: o llama.cpp **paga** por isso em três lugares

**O que é [V].**

- `ggml_cuda_flash_attn_ext` escolhe `TILE`, `MMA_F16` ou `VEC` (`ggml/src/ggml-cuda/fattn.cu:707-722`);
  os dois primeiros **exigem K e V em f16** (`fattn.cu:683-695`).
- Quando o kernel é `TILE`/`MMA_F16` e o cache está quantizado, o backend **materializa uma
  cópia f16 do cache inteiro** por chamada (`fattn-common.cuh:53-82`, chamado de `fattn.cu:702`
  e `fattn-common.cuh:1010`).
- O caminho que lê o cache quantizado nativamente é o `VEC` (`ggml_cuda_get_fattn_vec_case`,
  `fattn.cu:459`), escolhido para KV quantizado só com **`Q->ne[1] <= 2`** (`fattn.cu:656-670`).

A `131K`, essa cópia f16 por camada é `2 × 131072 × 1024 × 2 B = 536 MB`; em 16 camadas,
**8,6 GB de tráfego extra por passe de prefill** (**[E]**, sobre `fattn-common.cuh:68-81`).

**Veredito: aplica com uma volta — e é uma armadilha para o nosso prefill.** Nossa atenção
já desquantiza inline (`kv.h:80-174`, `attn.cuh`), ou seja, já estamos no desenho do `VEC` sem
a limitação `ne[1] <= 2`. **Vantagem nossa**; mas se o prefill batelado precisar de f16 para o
`wmma i32`, ele reintroduz esse custo. **Não materializar f16: desquantizar dentro do laço.**

**Restrições do upstream [V]:** V quantizado **exige flash attention**
(`src/llama-context.cpp:3702-3710`); `head_dim % 32 == 0` para K e V
(`llama-context.cpp:3713-3730`; temos 256, OK); e **só MLA/DeepSeek4 recusam K ≠ V**
(`llama-context.cpp:3697-3700`) — Qwen3.5 **não** é MLA, então **K ≠ V é suportado**.

## 1.3 `q5_0` vs `q8_0` no KV: o que a nossa própria medição já respondeu

De **`docs/medicoes-banda-e-gargalos.md:485-493`** (medido na GPU, `bench-attn-gpu`, 64K, 16
camadas) **[V]**:

| KV a 64K | atenção | tráfego emitido | banda emitida | token inteiro |
|---|---|---|---|---|
| `q4_0` | 21,03 ms | 7,25 GB | 346 GB/s | 55,02 ms (18,17 tok/s) |
| `q8_0` | **19,87 ms** | 13,7 GB | **690 GB/s** | **54,45 ms (18,36 tok/s)** |

**O `q8_0` é 5,5% mais rápido que o `q4_0` na atenção e 1% mais rápido no token inteiro, com
quase o dobro do tráfego** — porque o gargalo é o **desempacotamento por elemento**, não os
bytes (`:491-493`). O `q5_0` tem o **mesmo desempacotamento do `q4_0` mais** um `qh`
(gather de 1 bit por elemento; ver `FA_DEQUANT4_Q5_0`, §1.4) ⇒ tende a rodar na faixa do
`q4_0` ou pior, **não** na faixa do `q8_0`.

Consequência para a política de KV (**[E]** sobre medições): `K q8_0 / V q4_0` custa **3,25 GiB**
a 131K e deve ser **igual ou mais rápido** que `K q5_0 / V q4_1` (2,625 GiB) — e é *mais*
preciso no K. Ver §5.5 para o que a tabela de KLD do upstream diz sobre isso.

## 1.4 `FA_DEQUANT4_*` do Vulkan: a referência de kernel de desquantização no laço de atenção

**O que é [V].** `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` declara **buffers
de K e de V separados por tipo** (`:22-25`) e desquantiza **4 elementos por vez** dentro do laço
de atenção: `FA_DEQUANT4_Q4_1` (`:57-67`) e `FA_DEQUANT4_Q5_0` (`:69-77`), selecionados
independentemente para K e V (`:126-138`). Há ainda um pipeline **dequant+transpose fundido
para KV quantizado** (`ggml-vulkan/ggml-vulkan.cpp:969`).

**Aplica direto.** É o nosso `kv_load8` (`kv.h:125-174`) com outra granularidade, e o
`FA_DEQUANT4_Q5_0` dá o padrão exato do 5º bit (`hb = ((qh >> iqs) & 1) * 16`, valor
`d*(nib + hb - 16)`) — **compare com `quantize_row_q5_0_ref` antes de escrever o seu**.
**Não copiar a granularidade de 4**: a nossa é 8 e foi ela que a medição de §1.3 mostrou ser o
gargalo.

## 1.5 MMQ / prefill: condição de seleção e o desenho do tile

**O que é [V].** `ggml_cuda_should_use_mmq` (`ggml/src/ggml-cuda/mmq.cu:266-392`):

- tipos suportados incluem `Q4_0/Q4_1/Q5_0/Q5_1/Q8_0` e K-quants/IQ (`mmq.cu:270-295`);
- exige ≥ 48 KiB de shared por bloco, senão cai para BLAS (`mmq.cu:299-306`);
- **para RDNA4 (`amd_wmma_available`, `mmq.cu:356`) o retorno é `true` incondicional**
  (`mmq.cu:380-383`): *"For RDNA4 MMQ is consistently faster than dequantization + hipBLAS"*
  (PR #18537). **Em RDNA4 o upstream não usa "desquantizar + hipBLAS" para GEMM quantizado.**
- a fronteira do **GEMV** é outra: `MMVQ_MAX_BATCH_SIZE 8` (`mmvq.cuh:3`) e
  `ggml_cuda_should_use_mmvq` devolve `ne11 <= 8` (`mmvq.cu:318-411`). O `ne11 > 8` do briefing
  é **esta** fronteira (GEMV dp4a → GEMM tiled dp4a), não uma condição do MMQ.
- `MMQ_DP4A_MAX_BATCH_SIZE 64` só vale para NVIDIA-com-tensor-core (`mmq.cuh:8`); a escolha de
  tile para RDNA3/RDNA4 em MoE usa tokens **por especialista** (`mmq.cu:248-249`);
  ativações em **`q8_1`** e `MMQ_TILE_NE_K 32` (`mmq.cuh:111-119`).

**Aplica com uma volta.** O que se copia é o **desenho**: ativação quantizada **uma vez por
chunk** (`quantize_mmq_q8_1_cuda`, `mmq.cu:225-247`) e o mesmo `sudot4` do nosso `matvec`, com
16-32 tokens por tile. A volta: **o matvec não é o gargalo de prefill** — a 70 tok/s o peso por
token é 11,12 GB/16 = 0,7 GB (1,1 ms), enquanto o prefill medido é 13,7 ms/token
(`docs/medicoes-m8.md:44-49`). O que trava é o **andaime por token** (atenção, GDN, normas,
KV write).

## 1.6 MMVQ: o desenho de referência para **verificar D rascunhos com uma leitura dos pesos**

**O que é [V].** `mul_mat_vec_q` tem o número de colunas (`ncols_dst`) como **parâmetro de
template** e o laço interno reusa o bloco de pesos:

```cuda
for (int j = 0; j < ncols_dst; ++j)          // mmvq.cu:726-735
    tmp[j][i] += vec_dot_q_cuda(vx, &y[j*stride_col_y + kby], ...);
```

É exatamente o que a verificação batelada precisa: N colunas pelo preço de uma leitura de
11,12 GB. O upstream **retuna a ocupação por N** para RDNA4 (`mmvq.cu:436-490`): `nwarps=8`
com `ncols_dst=1` para tipos de vec_dot simples (`q4_0/q4_1/q5_0/q5_1/q8_0/Q2_K/Q4_K/Q5_K/
Q6_K/IQ4_*`), e **`nwarps=1` para `ncols_dst>1`** ("regress due to register pressure").

**Aplica direto — e nós já temos a peça:** `matvec_launch_batch`, `matvec_batch_cap() == 16`,
instanciações N = 2,3,4,8,16 (`include/rdna4/matvec.cuh:590-635`), bit-exata no
`tests/check_batch_gpu.hip`. **E o número que decide a noite já está medido** — ver §6.

## 1.7 As fusões nomeadas do Vulkan (a lista que a frente de fusão deve usar)

**[V]** (`ggml/src/ggml-vulkan/ggml-vulkan.cpp`): padrões em `:689-691`, enum em `:794-798`,
nomes efetivos em `:18153-18315`.

| fusão | linha | por que importa para nós |
|---|---|---|
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` / `..._ROPE` | `18188`, `18198` | norm+escala+rope **e a escrita das linhas de KV** (`set_rows`) — 3-4 lançamentos por camada × 16 |
| `ROPE_VIEW_SET_ROWS` | `18255` | variante sem norma |
| `RMS_NORM_MUL_ADD`, `RMS_NORM_MUL_ADD_MUL`, `RMS_NORM_MUL` | `18206-18223` | residual + norma + escala |
| `SSM_CONV_BIAS_SILU`, `SSM_CONV_SILU` | `18240`, `18248` | **conv do GDN + silu** — hoje 2 lançamentos × 48 |
| `SILU_MUL`, `SIGMOID_MUL`, `GELU_MUL` | `18231-18234` | portão do FFN ×48 e **portão sigmoide da atenção do MTP** |
| `MULTI_ADD` | `18153` | até 4 somas de uma vez (residuais) |
| `MUL_MAT_ADD`, `MUL_MAT_ADD_ADD` | `18157`, `18163` | viés fundido no matvec; `MUL_MAT_ID_*` é MoE (**não aplica**) |

Também: a escolha de pipeline de matmul usa `aligned = ... && ne01 > 8 && ne11 > 8`
(`ggml-vulkan.cpp:9515`) — o "8" reaparece como granularidade mínima de tile.

**Aplica direto.** Nós temos 1940 lançamentos/token; ~1440 pequenos custam **~4,0 ms medidos**
mais ~1,1 ms de latência de `rms_norm` acima do piso (`medicoes-banda-e-gargalos.md:523-524`).
**Enquadramento que vale roubar** (de um relato de engenharia sobre decode em RDNA4 **[R]**):
*"o valor de fundir dois kernels de decode não é principalmente o lançamento que você economiza;
é a barreira que você apaga entre eles."*

## 1.8 HIP graph: o custo de despacho **já está instrumentado** no nosso bench

**[V]** `tests/bench_matvec_shapes_gpu.hip:359-404` captura os 497 matvecs num `hipGraph` e
imprime `"... -> Z ms/token of launch tax recoverable"`, com o comentário de que o llama.cpp
sai com `GGML_HIP_GRAPHS` ligado (`:366`).

**Aplica — com um aviso honesto [R]:** um HIP graph move custo de **submissão**, não de
**dispatch na GPU**; o llama.cpp registra (issue #6763) que os gaps são *"mostly due to GPU-side
launch overheads rather than CPU API calls"*, e o PR #11867 mede **+3,6%** num 8B no MI100
ROCm. Como o nosso gap medido é 2,2-3,6 µs **por kernel na fila**, o recuperável pelo grafo é
provavelmente bem menor que os 4 ms nominais. **Medir antes de comemorar.**

## 1.9 MTP do `qwen35`: o grafo, e o motor especulativo com **planos de estado por índice**

**Grafo do MTP [V]** (`src/models/qwen35.cpp:485-644`): exige `n_layer_nextn == 1` (`:488-489`);
entrada = par `(token, h)`; `h_norm=RMSNorm(h, nextn.hnorm)`, `e_norm=RMSNorm(embed(tok),
nextn.enorm)`, `concat=[e_norm; h_norm]` (**nessa ordem**, `:543-547`), `eh_proj`, **um bloco
decoder de atenção plena completo com KV próprio**, norma final compartilhada (`:626-630`) e
**head compartilhada** (`:636-639`); o nó `h_nextn` é exposto como saída (`:632-633`) e
realimenta o próximo rascunho.

**Driver [V]** (`common/speculative.cpp:1330-1770`, `common/common.h:326-332`):

- `n_max` = comprimento de rascunho (**default 3**), `n_min` (default 0), **`p_min`** (default
  0.0) = probabilidade mínima do token rascunhado; abaixo dela **o encadeamento para**
  (`speculative.cpp:1690-1700`) — versão barata da "confiança ≈ aceitação" do EAGLE-2;
- laço de rascunho **sequencial**, cada passo depende da `h` do anterior (`:1626-1750`);
- o ctx do MTP tem **memória própria**; `process()` faz "decode de recuperação" do **lote de
  verificação inteiro** com `h` deslocada em uma posição (`:1538-1556`), e o `accept()` escolhe
  a linha `min(n_accepted, n_rows-1)` (`:1753-1766`) ⇒ **o tronco precisa devolver a `h`
  (pré-norma) de *todas* as posições verificadas**, não só da última;
- truncagem de KV na rejeição por `llama_memory_seq_rm(mem, seq, pos, -1)` (`:1560-1568`,
  `:1652-1660`) — o padrão "escreve todos os candidatos, corta no prefixo aceito";
- **estatística por posição**: `n_acc_tokens_per_pos` / `#mean acc len = X, #acc rate/pos = (...)`
  (`:2903-2990`). **É o que falta no nosso motor** (medimos taxa agregada, `docs/mtp.md:143-149`).

**Rollback do estado recorrente — o mecanismo que o upstream usa [V]**:
`n_rs_seq` planos extras do estado completo, tensor alargado para `mem_size * (1 + n_rs_seq)`,
**rollback = movimento de índice (sem `memcpy`)**, dimensionado a partir da profundidade do
rascunho (`src/llama-memory-recurrent.cpp:101`, `:193-203`; `common/common.h:394-400`
`need_n_rs_seq()` devolve `draft.n_max` para MTP/EAGLE3/DFLASH/DSPARK e 0 caso contrário), e
`llm_arch_supports_rs_rollback` (`src/llama-arch.cpp:1104-1120`) **inclui `LLM_ARCH_QWEN35`**
(mas não `QWEN3NEXT`). No `server-context.cpp:3926-3928,3972-3973` o caminho rápido é
condicional (`n_rollback > n_rs_seq` ⇒ checkpoint + **devolve um token aceito**).

**Veredito: aplica direto.** Copiar (a) a estatística por posição, (b) o portão `p_min`,
(c) a `h` de todas as posições no lote de verificação, (d) **planos de estado por índice**.

---

# 2. vLLM — o que é máquina de multi-inquilino e o que sobra para um usuário só

| técnica | é máquina de multi-inquilino | sobra para um usuário | veredito |
|---|---|---|---|
| **PagedAttention** | fragmentação externa (**0** com uma sequência **[E]**), refcount/COW, preempção, troca | alocação sob demanda (o desperdício de reserva) | **aplica só como alocador, não como kernel** |
| **Chunked prefill** | mixar decode+prefill de outros pedidos, proteger o TBT alheio | teto de memória de ativação; TTFT | **não aplica para tok/s** |
| **Prefix caching (APC)** | agendamento ciente de cache entre pedidos | reuso exato de prefixo **entre chamadas** | **aplica — mas ver o GDN (§3.2)** |
| **Continuous/in-flight batching** | tudo | nada | **não aplica** |

**PagedAttention [V]** (Kwon et al., SOSP'23, [arXiv:2309.06180](https://arxiv.org/abs/2309.06180)):
blocos de **16 tokens** ("*block sizes from 16 to 128 lead to the best performance… block size
16 is large enough to efficiently utilize the GPU and small enough to avoid significant
internal fragmentation*"); os sistemas anteriores usavam só "*20.4% - 38.2% of the KV cache
memory… to store the actual token states*". **O custo da indireção é declarado** (§7.1):
*"our GPU kernels involve extra overheads of accessing the block table, executing extra
branches, and handling variable sequence lengths. As shown in Fig. 18a, this leads to **20–26%
higher attention kernel latency**."* E a correção que eles mesmos adotam (§5.1) é **fundir para
cortar lançamentos** — o que é o nosso problema, não o deles.

**Conta para o nosso caso [E]:** com uma sequência, a fragmentação externa é zero. A reserva
contígua desperdiça VRAM se o contexto pedido é muito maior que o uso (a 4K num cache de 131K,
2,35 GB reservados para 73,7 MB vivos), mas isso é **flexibilidade, não velocidade**: quem roda
`-c 4096` não desperdiça nada. **Veredito: não aplicar.** O único pedaço aproveitável
(allocator sob demanda) não muda o tok/s e interage mal com o nosso desenho de KV (linhas
`kv_row_bytes` com base por camada/sequência já são triviais de indexar).

**Chunked prefill [V]:** `chunked prefill is enabled by default whenever possible…`; os ganhos
do Sarathi-Serve são de **capacidade sob SLO com concorrência** (2,6× Mistral-7B/A100; 6,9×
Falcon-180B) e o próprio paper admite o custo: *"enables low TBT latency and high throughput at
the expense of a marginal increase in TTFT – this is due to the overhead of chunking"*.
Com um fluxo não há TBT alheio para proteger, e **cada chunk é outra passada sobre 11,12 GB de
pesos**. **Veredito: não aplicar para vazão.** Só faz sentido para *interatividade* (primeiro
token aparecer antes) e, aí, com chunk ≥ 2048 para amortizar o passe de pesos.

**APC [V]:** hash de bloco encadeado com o hash do pai, "*we only cache full blocks*", só blocos
completos. Para o **nosso CLI**, um prompt de sistema de 2K reusado economiza
2000/72,9 ≈ **27 s de TTFT por invocação** **[E]** — o maior item de valor de todo este
levantamento §2/§3, *desde que* se resolva o estado do GDN (§3.2). **Veredito: aplica para o
servidor, não para o CLI de uma pergunta.**

**Continuous batching:** puro multi-inquilino — **0** de ganho com batch 1. **Não aplicar.**
Agravante nosso: cada sequência precisa do **próprio estado GDN** (0,15 GiB) e o scan do GDN
deixa de ser 48 kernels e passa a ser 48 × n_seq.

---

# 3. ExLlamaV2 / SGLang / TensorRT-LLM

## 3.1 ExLlamaV2: cache Q4/Q6/Q8 e — o achado — a **rotação de Hadamard**

**Premissas corrigidas [V]:** `exllamav2/kernels/quant/cache_kernels.cu` **não existe** em tag
nenhuma; em v0.3.2 o cache está em `exllamav2/exllamav2_ext/cuda/cache_q.cuh` + `cache.cu`
(python: `cache.py`, paged: `ext_cache.cpp`, `attn.py:forward_paged`). **Não há E4M3** — as
escalas são `half` (fp16). **Não há cache Q5**; há Q4/Q6/Q8 e um cache FP8 que é **truncamento
de bits do fp16**, com um comentário explicando a rejeição do E4M3 (`cache.cu:18-24`).

**Layout [V]:** `Q_CACHE_BLOCKSIZE_Q = 512`, 2 elementos por thread (um `half2`); **uma escala
fp16 por 32 elementos consecutivos** dentro de cada cabeça (tensor de escalas
`(batch, seq, kv_heads, head_dim//32)`) ⇒ **mesmos bytes/elem do `q4_0`/`q8_0`** do llama.cpp
(0,5625 / 1,0625) **[E]**. Alinhamento exigido: `(kv_heads × head_dim × q_block) % 512 == 0`.

**Duas coisas melhores que o layout do llama.cpp:**
1. **Nibble linear**: elemento `e` → palavra `e/8`, nibble `e%8` — 8 elementos consecutivos num
   acesso de 32 bits, **pronto para `dp4a`**; o `q4_0` do llama.cpp intercala `j` com `j+16` no
   mesmo byte e exige shuffle antes do dot inteiro.
2. **Rotação de Hadamard antes de quantizar** (`#define HADAMARD_Q`): borboleta de 5 estágios
   `__shfl_xor_sync` antes de `fp16_to_q<4>` e desdobrada em `q_to_fp16` (reescala 1/32).
   Medido no `doc/qcache_eval.md`: Mistral-7B 3.0 bpw PPL 13,41 → **13,37** (fp16 13,33);
   Llama2-7B 4.0 bpw 11,74 → **11,60** (fp16 11,43). Surpresa do mesmo doc: **Q4 é mais preciso
   que FP8** (11,74 vs 11,92).

**O llama.cpp adotou a mesma ideia [V]** — PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038)
("llama : rotate activations for better quantization", 2026-04-01): rotação ortonormal de
Walsh-Hadamard em K e V sempre que o cache é quantizado (`attn_rot_k`/`attn_rot_v`), **custo
desprezível** ("*Rotation does not affect the dot product of 2 vectors… It adds 4 matrix
multiplication operators… at almost 0 cost*"), portão `head_dim % 64 == 0`, kill switch
`LLAMA_ATTN_ROT_DISABLE`. Verificado por mim no código local:
`src/llama-graph.cpp:2855-2861` (rotaciona **Q e K** com a mesma matriz e **V**, e aplica a
inversa na saída da atenção: `:2891`), `src/llama-kv-cache.cpp:23` (`ggml_gen_hadamard`),
`:315-334` (gate e aviso de desligamento).

**Veredito: aplica — e é o item de maior valor esperado deste levantamento para o KV.**
Custo em memória **zero**; ganho medido pelo upstream em **Qwen3-0.6B**: PPL com KV `q5_1`
61,70 → **14,15**, `q4_1` 212,48 → **22,28**, `q4_0` 62,02 → **46,25** (f16 13,67) **[R/V na
tabela do PR]** — redução de 4-10× no dano da quantização. **Consequência desconfortável para
nós:** o llama.cpp de hoje **rotaciona**, então *todos* os números de qualidade de KV
publicados depois de 2026-04 são de um cache **rotacionado**; o nosso motor **não roda
rotação**, e portanto o regime comparável é a coluna "sem rotação" — em que o KV `q4_0` é
muito pior do que os relatos de PPL sugerem (§5.5). **Risco:** não é bit-exato (introduz
arredondamento), então exige gate de PPL/KLD; está coberto pela autorização de "mudança na
política de atenção" (`docs/noite-regras.md:60-61`), com o desvio em % escrito no diário.
**Custo [E]:** a rotação de K na escrita e de Q por token são ~8 estágios de borboleta por
linha de 256 ⇒ ~0,02 ms/token em 16 camadas; a inversa na saída da atenção é aplicada **depois
do merge do split-KV**, sobre `24×256` valores por camada.

**Experimento:** A/B de PPL/KLD com e sem rotação, `q4_0/q4_0` e `q5_0/q4_1`, a 8K/32K; aceitar
se ΔPPL ≤ +0,5% contra o mesmo tipo **sem** rotação e a KLD média cair (ver §5.5 para limiares).

## 3.2 SGLang: RadixAttention — prefixo em árvore e o **estado recorrente**

**O que é [V]** (Zheng et al., [arXiv:2312.07104](https://arxiv.org/abs/2312.07104)):
*"we reintroduce a simple LRU eviction policy that evicts the least recently used leaf first"*;
nó com refcount, evictável em zero. Diferenças reais contra o APC do vLLM: casamento **por
token** com split (o vLLM arredonda para bloco de 16), uma travessia em vez de O(blocos) hashes
encadeados, e evicção por folha. Overhead declarado: *"the time used for managing the
RadixAttention data structures is only 0.2 seconds… less than 0.3%"*. Números do paper
(6,4× vazão / 3,7× latência) são explicitamente *"result from KV cache reuse, the exploitation
of parallelism within a single program, and faster constrained decoding"* — **não** da
RadixAttention sozinha. O único dado quase-single-user (Chatbot Arena, um worker por modelo):
**52,4%/74,1% de hit, TTFT 1,7× melhor**.

**O twist — e ele é nosso:** `lock_ref` só existe para concorrência (com 1 sequência é código
morto). Mas **as 48 camadas GDN carregam estado recorrente que não é reusável por token**:
um hit de prefixo no KV recupera no máximo os 16/64 = **~28% do trabalho de prefill**
**[E]**, não 7,6×. O SGLang construiu exatamente essa correção em `mamba_radix_cache.py` **[V]**:
`new_node.mamba_value = None  # mamba cache can not be split`, dois refcounts
(`full_lock_ref`/`mamba_lock_ref`), `mamba_branching_seqlen` arredondado **para baixo** ao
tamanho do chunk de cache, e COW ("the matched state must be fully copied as a snapshot").
**Custo de um checkpoint GDN = 153,9 MB fp32 / 78,4 MB bf16** [V, aritmética reproduzida] —
ou seja **um checkpoint bf16 ≈ 4 356 tokens do nosso KV `q4_0` inteiro**: depois de 11,12 GB
de pesos + 2,42 GB de KV cabem ~25 checkpoints, e **o pool de estados, não o KV, é o limite**.
**Veredito: aplica com o twist** — versão útil é **um slot** ("mantém o KV + estado da última
conversa"), e só para o servidor. **Experimento:** 10 requisições sequenciais `A + u_i`
(A = 4 000 tokens, u_i ≈ 400), concorrência 1: sem cache × 1-slot × KV-only; aceitar se a
mediana do TTFT ≤ 40% do baseline com |Δlogit| ≤ 1e-3.

## 3.3 TensorRT-LLM: in-flight batching e a fusão de decode em batch pequeno

**In-flight batching [V]: não aplica** ("*in-flight batching of requests (also known as
continuous batching or iteration-level batching) for higher serving throughput*" — é
escalonamento por iteração do Orca, com concorrência).

**XQA [V]: não aplica; a ideia, em parte.** Um kernel que faz "*adds the QKV bias, applies
RoPE, and performs dequantization and quantization*", mas com restrições de tensor core
(HMMA `SM>=80`, QGMMA só `SM==90`, MLA só `SM==120`) e **recusa MHA comum com beam_width==1**.
Ganho reportado (Llama-70B, 1×H200, ISL128/OSL2048): 1.227 → 2.941 tok/s/GPU — curva de
tensor core, não transferível para gfx1201. Para nós, o que se recupera é **contagem de
lançamentos**: ~3-4 × 16 = 48-64 de 1940 ≈ **0,3-0,4% do passo** **[E]**.

**O que realmente mapeia nos 1940 lançamentos @ 2,2 µs** (1940 × 2,2 µs = 4,27 ms ≈ 12,5% de
um passo de 34 ms **[E]**):

| fusão do TRT-LLM | existe por causa de | lançamentos salvos | % do passo |
|---|---|---|---|
| allreduce+residual+RMSNorm / oneshot AllReduce [V] | **paralelismo de tensor** | 0 | **irrelevante** |
| multi-CTA fused MHA "*when batch_size × num_heads is less than the number of multi-processors*" [V] | ocupação baixa em batch pequeno | 0 (adiciona trabalho) | **1-2,5% @131K** |
| concat de GEMM de QKV (`fuse_gemms_mixed_children`: "*Concatenates the weights for WDQ, WDKV, and WKR to reduce kernel launch overhead*") [V] | overhead de lançamento em batch pequeno | ~128 | 1,5-2,5% |
| gate+up+SwiGLU num FFN com portão [V] | idem | ~128 | 1,5-2,5% |
| add+RMSNorm (`fused_add_rms_norm_quant`) [V] | idem | ~64 | 0,4-0,8% |
| equivalente de CUDA graph (`hipGraph`) | overhead de **submissão** | 0 | **3-8% se os 2,2 µs forem de submissão** |

**O item que mapeia no nosso gargalo *declarado*** é o **MHA multi-CTA**: é a resposta do
TRT-LLM para ocupação baixa em batch pequeno e, em RDNA4, o análogo (uma CTA carrega a linha de
KV e calcula as **6 cabeças GQA**) mata a releitura 6× ao mesmo tempo (§4).

**Linha "megakernel" [V]:** Hazy, *Look Ma, No Bubbles!*, mede *"a launch cost of about 2.1
microseconds, and with CUDA graphs the launch cost only decreases to around 1.3 microseconds"* —
**corrobora os nossos 2,2 µs como piso de hardware/driver** e limita o que o grafo recupera.

## 3.4 Específico de AMD/RDNA4

- **Graphs no ROCm [V]:** a doc dá só "*graphs only provide a benefit for workloads that
  require many iterations*"; medido no llama.cpp: PR #11867 (MI100, 8B Q4_K_M) **85,00 → 88,04
  tok/s (+3,6%)**, +29% num 1,5B; issue #6763 relata +14% num 7B e o diagnóstico que importa
  (gaps são majoritariamente GPU-side).
- **Rates do RDNA4 [R]:** por CU/clk FP16 1024, INT8 2048, e **WMMA e ops vetoriais saem pelo
  mesmo pipe**; a mesma fonte mede que **dobrar o rate de matriz deixou o decode igual (39,6
  tok/s)** — confirmação independente de que matriz não ajuda em M=1. `v_dot4_i32_i8` **não
  aumenta o MAC/s** sobre FMA fp32; o ganho dele é **~4× menos slots de issue por byte de
  peso** — que é exatamente o nosso diagnóstico "issue-bound, não bandwidth-bound".
- **Orçamento de decode em RDNA4, forma parecida [R]:** 25,2 ms = 10,7 de streaming de pesos +
  0,6 de KV + 5,4 de kernels pequenos + **7,0 de dispatch/barreiras**; reuso de buffer → 22,2 ms
  (+14%); fusão → 18,0 ms.
- **Existe trabalho de terceiros sobre MTP em RDNA4 [R, metodologia não verificada]:** 9070 XT
  com Qwen3.6-27B IQ3_M + KV `q8_0` reporta **46,06 tok/s com 62,7% de aceitação**, 14,46/16 GB;
  outro com **KV `q4_0`** (a nossa config) reporta **10 → 35-40 tok/s**; um R9700 (mesmo
  gfx1201) fala em *"MTP provides a consistent ~2× speedup on RDNA4"*. É a única evidência
  RDNA4 que existe e está **acima** dos nossos 29,3 tok/s — vale rodar o llama.cpp
  `--spec-type draft-mtp --spec-draft-n-max 3` como **referência externa medida** antes de
  atribuir a diferença a algoritmo.
- **Negativo verificado:** *"Ripple: Accelerating LLM Inference on AMD GPUs"* **não existe**
  (o único Ripple LLM é o de smartphone, arXiv 2410.19274). Não citar.

---

# 4. Decodificação especulativa (literatura) — o que a verificação batelada deve saber

## 4.1 DeepSeek-V3/R1: o desenho do MTP e os números reportados

**[V]** ([arXiv:2412.19437v2](https://arxiv.org/abs/2412.19437), §2.2 + Fig. 3): D = **2 módulos
sequenciais**, `h'^k_i = M_k[RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))]`, head compartilhada,
perda = CE média por profundidade com λ = 0,3. Citação literal que descreve o nosso grafo:
*"Different from Gloeckle et al. (2024), which parallelly predicts D additional tokens using
independent output heads, we **sequentially** predict additional tokens and keep the complete
causal chain at each prediction depth."* E a avaliação (§5.4.3): *"the acceptance rate of the
second token prediction ranges between **85% and 90%**… This high acceptance rate enables
DeepSeek-V3 to achieve a significantly improved decoding speed, delivering **1.8 times TPS**."*

**Duas correções de enquadramento:** (a) **não há decomposição por posição** no relatório — os
85-90% são um agregado para o **segundo** token; (b) o 1,8× é uma afirmação de **D=2**, não de
cadeia longa. **O nosso 86,7% está na borda inferior da faixa** — é um número bom, e é o
mesmo regime que o do DeepSeek. **[E]:** o 1,8× vem de um MoE em H800 com MLA e 2 rascunhos;
usar como **teto**, não como previsão (a nossa previsão medida é §6: 1,3-1,5×).

## 4.2 EAGLE-1/2/3

| | o que acrescenta | ganho / τ (T=0) | fonte |
|---|---|---|---|
| EAGLE-1 | AR no nível de *feature* (penúltima camada) + head; **rascunho em árvore** | 3,05× / 3,96 (Vicuna-13B) | [2401.15077](https://arxiv.org/abs/2401.15077) |
| EAGLE-2 | **árvore dinâmica consciente do contexto**; confiança ≈ aceitação | 4,22× / 4,83 | [2406.16858](https://arxiv.org/abs/2406.16858) |
| EAGLE-3 | *training-time test* + fusão de features de várias camadas | 5,51× / 6,62 | [2503.01840](https://arxiv.org/abs/2503.01840) |

**[V]** a linhagem (EAGLE-3 §1): *"predicting the next-next token based solely on top-layer
features—which are inherently limited to the next token—poses a significant challenge"*; e
*"EAGLE inspired the multi-token prediction technique used in the pre-training of DeepSeek-v3,
which in turn inspired new architectural designs in EAGLE-3."* O ganho da árvore está isolado no
abstract do EAGLE-2: *"20%–40% faster than EAGLE-1"* **[V]**.

**Três coisas a copiar, por custo crescente:** (a) **confiança como aceitação** (o nosso head
já emite logits ⇒ margem do top-1 sai de graça; é o `p_min` do llama.cpp); (b) árvore em vez de
cadeia (§4.4 — e no nosso caso ela é **cara**); (c) fusão de features de várias camadas —
**exige treinar de novo o head** ⇒ **não nesta noite**.

## 4.3 Medusa: a decomposição que importa

**[V]** ([2401.10774](https://arxiv.org/abs/2401.10774)): *"we found that **five heads** are
sufficient at most"*; árvore esparsa de **64 nós, profundidade 4**; *"Medusa-1 can achieve over
2.2× speedup… while Medusa-2 further improves the speedup to 2.3-3.6×"*. Tabela 3:

| técnica | ganho |
|---|---|
| heads Medusa-1 **sem** tree attention | **~1,5×** |
| + tree attention | ~1,9× |
| + configuração de árvore otimizada | ~2,2× |
| + treino Medusa-2 | ~2,8× |

**A leitura honesta para nós:** o nosso head MTP **é** a linha "heads" ⇒ ~**1,5×** é o que a
peça que já temos vale, e a árvore é um multiplicador **que ainda não pagamos**. γ prático:
*"γ = 4 yielded the best performance for Vicuna-7B, while… γ = 3 [for] Vicuna-13B"* **[V]**, e
Chen et al. reportam que o ganho *"plateaus or even regresses"* além de K, com K = 4 — o que
casa com a nossa varredura (D=4 melhor, D=8 pior, §6).

## 4.4 Verificação batelada, árvore e rollback de estado — o que a literatura faz

**O truque do KV é padrão e nós já temos o equivalente [V]:** escrever todas as k linhas
candidatas e truncar depois — `llama_memory_seq_rm(mem_dft, seq, pos0, -1)`
(`common/speculative.cpp:750, 1559, 1647`).

**O estado recorrente é onde está a resposta. [R, verificado por mim na thread]** vLLM RFC
[#49232](https://github.com/vllm-project/vllm/issues/49232) (ReplaySSM) nomeia o nosso problema
exato: *"The SSM state update is irreversible, so today's implementation keeps a separate
recurrent state per draft token (~(1 + num_spec)× the state). Caching inputs instead makes
rollback a **ring-buffer pointer move — O(1) instead of O(T)** — and enables verifying a draft
window in one shot rather than by serial recurrence."* Ganhos reportados (B300, buffer 16):
**1,09/1,07/1,88×** (Mamba2) e **0,99/1,19/1,80×** (GDN, Qwen3.5-122B) para B=1/32/256.
**A coluna B=1 ≈ 1,0× é *o nosso regime*** — é o argumento mais forte contra sofisticar o
rollback nesta noite.

**Três implementações, três estratégias [V, todos verificados em codigo]:**

| impl | mecanismo | evidência |
|---|---|---|
| **llama.cpp** | `n_rs_seq` planos extras do estado completo, tensor alargado para `mem_size × (1 + n_rs_seq)`; rollback = **movimento de índice** (sem `memcpy`), dimensionado pela profundidade do rascunho | `src/llama-memory-recurrent.cpp:101,193-203`; `common/common.h:394-400` |
| **vLLM** | `num_speculative_blocks = num_speculative_tokens`; block table com 1+k colunas; estado aceito **copiado de volta** via `token_bias = num_accepted − 1` | `vllm/model_executor/layers/mamba/abstract.py:81-86`; `vllm/v1/worker/mamba_utils.py:219-223` |
| **SGLang** | baseline `intermediate_ssm = max_running×(γ+1)×[HV,V,K]`, chamado de *"the memory hog that collapses dspark concurrency"*; substituído por cache circular `(d,k,g)` + `h0` congelado, *"no state write-back"* — **só cadeia linear** (`speculative_eagle_topk <= 1`) | `kda_replayssm_spec_decode.py:7`; `gdn_replayssm_spec_decode.py` |

**Um detalhe do llama.cpp que deve moldar o desenho [V]:** `need_n_rs_seq()`
(`common/common.h:394-400`) devolve `draft.n_max` para MTP/EAGLE3/DFLASH/DSPARK e **0** para
drafter n-gram — **o número de planos é derivado da profundidade do rascunho, nunca escolhido
em separado**. E o caminho rápido é condicional (`server-context.cpp:3926-3928`); quando
`n_rollback > n_rs_seq`, o fallback **devolve um token aceito** (`:3972-3973`).

**Aplicado aos nossos números:** estado GDN = **3,15 MB/camada × 48 = 151 MB**
(`docs/medicoes-banda-e-gargalos.md:259`). O esquema `(1+k)` ingênuo com memcpy a k=4 custa
**755 MB** contra 11,2 GB de pesos + 2,6 GiB de KV — e **a nossa própria medição da variante
simples deu 1,0-1,1×, ou seja, não paga** (`docs/medicoes-m8.md:90-97`). Com **planos por
índice** (sem cópia por rodada) o custo vira VRAM, não tempo.

**Veredito:** planos `1+k` com movimento de índice = **aplica** (simples, igual ao llama.cpp e
ao vLLM); input-cache/ReplaySSM = **aplica e é melhor**, mas **não está entregue** no vLLM
(roadmap: "in progress"). **Risco:** bit-exatidão é obrigação dura — o port do SGLang exige
*"a BITWISE CLONE of `fused_recurrent_gated_delta_rule_fwd_kernel`… keep num_warps=1 so the
reduction trees match"*; nós temos a disciplina (`check_batch_gpu`, rel-L2 0,0).
**Experimento:** A/B de três estratégias de rollback a k=4 numa corrida fixa de 128 tokens —
(1) restore de checkpoint + replay do prefixo aceito, (2) planos `1+k` por índice, (3)
input-cache + replay — todas gated em ids bit-idênticos vs ganancioso puro.

**Árvore no nosso caso [E]:** cada nó da árvore custa ~65 KB de KV (**desprezível**) e
**151 MB de plano de estado** (**fatal**). É exatamente por isso que o SGLang restringe o
ReplaySSM a `topk <= 1`. **Veredito da árvore: não aplicar antes de medir o custo do batch**
(§6) — e, se for tentar, **um ramo de 2 nós na profundidade 2**, nada de 64 nós.

## 4.5 A matemática do k (com os nossos 86,7%)

**[V]** Leviathan et al. ([2211.17192](https://arxiv.org/abs/2211.17192)), Eq. 1:
`E[#tokens] = (1 − α^(γ+1)) / (1 − α)`; Teorema 3.8 (wall-clock)
`IF = (1 − α^(γ+1)) / ((1 − α)(γc + 1))`, com **c = razão entre o tempo de uma execução do
modelo alvo e uma do rascunho** — e a nota de que *"c depends on the hardware configuration and
software implementation details"*. Corolário 3.9: se **α > c**, existe ganho, de pelo menos
`(1+α)/(1+c)`. E, sobre o regime: *"in common situations where **memory bandwidth is the
bottleneck**, and compute resources are available, it may be a good default to accelerate
sampling"*; *"the target model's weights and KV cache can be read once per execution… the
number of memory accesses for reading them shrinks by a factor of (1−α^(γ+1))/(1−α)."*

**Com α variando por posição, usar a forma não-i.i.d.** `E = Σ_{i=1}^{γ+1} Π_{j<i} α_j`
(generalização direta **[E]**). Com os nossos medidos (`docs/mtp.md:156-165`: D=3 ⇒ 3,46
tokens/rodada; D=4 ⇒ 4,00; D=4 aceita 79/125), α₁ = 0,867 ⇒ E ≈ 3,75-3,84, batendo com os 4,00
medidos.

**A hipótese escondida do Leviathan — e é ela que não vale para nós.** O `c` do paper é o custo
de **uma** execução do rascunho sobre o custo de **uma** execução do alvo, e o alvo ali é **uma
passada batelada** que o paper assume custar ≈ o mesmo que um passo normal de decode (é o
argumento "os pesos são lidos uma vez"). **No nosso motor isso é falso:** o cf medido é
**1,75 (N=4) a 2,2 (a 4K, com atenção/GDN sequenciais)** (§6). A fórmula honesta para nós é

```
IF = E[tokens/rodada] / ( cf + γ · c_rascunho ) ,   c_rascunho = 2,28 ms / 36,1 ms = 0,063
```

que reproduz a tabela do repo: **D=2 → 2,62/(1,80+0,13) = 1,36×; D=4 → 3,84/(2,19+0,25) = 1,57×**
(contra 1,31× e 1,51× de `docs/medicoes-m8.md:75-80`, diferença de arredondamento). Com isso,
`α₁ = 0,867 > c` ⇒ **o Corolário 3.9 garante que existe ganho**, mas o ganho é **1,3-1,5×**, não
os 1,8× do DeepSeek nem os 2,6× do Medusa: **os números publicados supõem cf ≈ 1 e uma GPU
servidora; os nossos não.**

**Onde a literatura e nós discordamos [V]:** o argumento "os pesos são lidos uma vez" pressupõe
GEMV/GEMM **weight-stationary**. O nosso matvec roda a **436 GB/s = 69% do pico** e a
`medicoes-banda-e-gargalos.md` atribui os 31% restantes a **issue-bound** (171 instruções ISA por
220 bytes em `iq3_s`). **A nossa verificação é substancialmente issue-bound, não puramente
bandwidth-bound** — é por isso que o batch não entrega os 4× de economia de tráfego.
**MagicDec** ([2408.11049](https://arxiv.org/abs/2408.11049)) **[V]** é o único paper que
reenquadra contexto longo (acima de `S_inflection` o alvo vira memory-bound e o ganho **cresce**
com o batch, até 2,51× com batch 32-256 e drafter de KV esparso) — mas a premissa dele é o KV
como gargalo, e a 131K a **nossa** atenção é **compute/issue-bound** (346 GB/s emitidos, 55% do
pico, `medicoes-banda-e-gargalos.md:309-311`). **Não transferir sem medir.**

**Gaps explícitos:** não existe paper que analise o custo de releitura de pesos quantizados para
uma verificação de k linhas num motor **GEMV**; não existe microbenchmark publicado de
"salvar checkpoint de estado recorrente × recomputar k"; e **`2506.01206` (Mamba Drafters) é um
*drafter* Mamba com alvo Transformer — não citar como solução de rollback** **[V]**.

---

# 5. Qualidade da quantização de KV (literatura) — K vs V

## 5.1 K vs V: o que está estabelecido

**O achado da KIVI [V]** ([2402.02750](https://arxiv.org/abs/2402.02750)) é de **granularidade**,
não de número de bits: *"for key cache, there are a few fixed channels whose magnitudes are very
large… key cache should be quantized per-channel… value cache should be quantized per-token."*
Erro relativo: K por token 13,67 → erro do score de atenção **47,00**; K por canal 4,55 → **9,60**
(~5× melhor). Para V o erro é o oposto: por token 3,55, por canal 49,89 (**~15×**).

**KVQuant [V]** ([2401.18079](https://arxiv.org/abs/2401.18079)): K por canal + V por token
ganha **3,82 PPL** sobre por-token/por-token a 3 bits em LLaMA-7B (tabela 7: 10,87 → 7,05);
V por canal é catastrófico (223). Pré-RoPE: **7,05 → 6,23 (−0,82 PPL)** a 3 bits (tabela 8).
Por bit width: **4 bits < 0,02, 3 bits < 0,1, 2 bits < 0,5 PPL** de degradação no Wikitext-2 em
9 modelos; 1% de outliers por vetor compra −0,19.

**O teste de troca de bits mais limpo — KVTuner [V]** ([2502.04420](https://arxiv.org/abs/2502.04420)),
Llama-3-8B-Instruct, erro relativo da saída da atenção eₒ:

| | KV8 | K8V4 | K8V2 | K4V8 | KV4 | K4V2 | K2V8 | K2V4 | KV2 |
|---|---|---|---|---|---|---|---|---|---|
| PPL | 9,95 | 9,94 | 10,04 | 9,99 | 9,99 | 10,11 | **31,92** | **31,48** | 37,29 |
| eₒ | 0,014 | **0,100** | 0,401 | **0,168** | 0,207 | 0,453 | **0,882** | **0,892** | 0,962 |

A 6 bits médios, K8V4 (0,100) bate K4V8 (0,168); a 5 bits, K8V2 (0,401) bate K2V8 (0,882);
8→4 bits no K custa **13,9×** de erro de score, 4→2 custa 4,6×. **Ressalva [R]:** com K por
*canal* (estilo KIVI) a ordenação se inverte (K4V8 > K8V4) — **a assimetria depende da
granularidade**, e a nossa é por bloco de 32.

Espelhos: KV-AdaQuant ([2502.15075](https://arxiv.org/abs/2502.15075)) **[R]** K4V2 75,2% ×
K2V4 54,7% no GSM8K; AsymKV ([2410.13212](https://arxiv.org/abs/2410.13212)) **[R]** Llama-2-7b
2b-K/1b-V 38,77 × 1b-K/2b-V 12,81 no TruthfulQA (fp16 87,72); DiffKV
([2412.03131](https://arxiv.org/abs/2412.03131)) **[R]** K2V4 → "near-zero accuracy".
**Resultados negativos [R]:** **Atom** ([2310.19102](https://arxiv.org/abs/2310.19102)) **não**
tem separação K/V e afirma o contrário ("the quantization error of the K cache has less
influence on the output"); WKVQuant ([2402.12065](https://arxiv.org/abs/2402.12065)) só testa
KV4; QAQ ([2403.04643](https://arxiv.org/abs/2403.04643)) é 2 bits qualitativo; ZipCache
([2405.14256](https://arxiv.org/abs/2405.14256)) separa por granularidade, não por bits.

**O ponto mais afiado:** KVarN ([2606.03458](https://arxiv.org/abs/2606.03458)) **[R]** —
">98% dos top 5% erros de quantização sob o esquema KIVI estão numa matriz K".

## 5.2 Por quê — e a volta que o nosso layout impõe

**Mecanismo [V]:** chaves têm canais fixos de magnitude muito grande (KIVI); KVQuant mostra que
as chaves **pré-RoPE** têm outliers *"in specific channels **across different tokens**"* — ou
seja, **consistentes entre tokens**, que é exatamente por que a **escala por canal** funciona; e
que **pós-RoPE** *"the distribution becomes less structured and there are less consistent
magnitudes for outlier channels… as RoPE applies a rotation operation between pairs of
channels"*.

**A volta (o achado central desta seção) [V]:** o Apêndice N do KVQuant mostra que, para
quantização de chave **por token** (o nosso regime), **pós-RoPE é MELHOR** que pré-RoPE
(int3: 10,87 pós × 14,68 pré). Nós armazenamos K **pós-RoPE** com escala por bloco de 32 ⇒
**estamos no regime certo do ponto de vista do KVQuant**, e a história de "outlier por canal" —
que é o que justifica K em mais bits *na literatura* — **não se transfere diretamente**. O
"pré-RoPE" também não é a alavanca certa para nós (§5.3).

**QK-norm [V]:** o relatório do Qwen3 só diz *"we… introduce QK-Norm… to ensure stable training
for Qwen3"*; o ViT-22B ([2302.05442](https://arxiv.org/abs/2302.05442)) dá o mecanismo
(logits *"quickly grow to over 50000"* sem normalização) e o QK-Norm
([2010.04245](https://arxiv.org/abs/2010.04245)) **[R]** limita ao intervalo do cosseno.
**Resultado negativo [V]: não achei paper revisado que remova QK-norm e meça outlier de K**.
A pista empírica é da comunidade, mas é afiada ([discussão #21297](https://github.com/ggml-org/llama.cpp/discussions/21297))
**[R]**, K-only com V = f16, wikitext-2, 200 chunks: **Qwen2.5-7B** `q4_1` no K = PPL
**10131,7** e `q4_0` = **3846,9** ("garbage") com f16 = 8,910; **Qwen3-8B** `q4_1` = **10,993
(+0,86%)** e `q4_0` = **11,470 (+4,03%)** com f16 = 10,899. **Hipótese, não fato:** QK-norm
torna o nosso cache de K substancialmente mais quantizável do que os resultados da era LLaMA
sugerem — mas isso tem de ser confirmado por medição nossa.

**GQA 6:1 e head_dim 256 [E]:** não há teste em lugar nenhum. O único dado correlato da KIVI é
que o Falcon-7B (uma **única** cabeça KV) precisa de 4 bits enquanto LLaMA (multi-head) sobrevive
a 2 bits ⇒ **menos cabeças de KV = mais sensível**. Com 6 cabeças de consulta lendo uma cabeça KV
quantizada, o erro de um K chega a 6× mais tráfego de consulta; e com `head_dim=256` cada bloco
de 32 é 1/8 de uma cabeça, então um canal outlier dentro do bloco ainda distorce os 31 vizinhos.
**GQA e head_dim 256 argumentam a favor de gastar no K, não contra.**

## 5.3 Como se quantiza chave — e o que é usável para nós

| esquema | evidência | usável no nosso motor? |
|---|---|---|
| Escala de K por canal [V] | KVQuant −3,82 PPL | **Não** — blocos ggml = 1 escala por 32 contíguos |
| K por token, grupos de 32 (o nosso) [V] | KIVI: ok a 4 bits; catastrófico a 2 | Sim — é o `q4_0`/`q5_0`/`q4_1` |
| Pré-RoPE [V] | −0,82 PPL a 3 bits | **Não, e não é barato** — ver abaixo |
| 1% de outliers esparsos [V] | −0,19 PPL | Não (sem buffer esparso) |
| Primeiro token em f16 [V] | *"the model is disproportionately sensitive to quantization error in the first token… keeping only the first token in fp16"* | **Sim — barato; fazer** |
| **Rotação de Hadamard antes de quantizar** [V] | PR #21038 + ExLlamaV2 `HADAMARD_Q` | **Sim — a ideia de maior valor aqui** |

**Custo de pré-RoPE para nós [E]:** armazenar K **sem** RoPE e aplicar RoPE depois da
desquantização **dentro do kernel de atenção** ⇒ a 131K × 16384 são ~2 rotações por elemento de
chave **por passo de decode** (`4,3 GFLOP/passo` extra de fp32) e perde-se a propriedade
"rotaciona uma vez na escrita". Como o próprio Apêndice N do KVQuant mostra que pré-RoPE
**perde** para quantização por token, **não fazer**.

## 5.4 O custo a contexto longo: o que quebra primeiro

**Recuperação (retrieval) ≫ QA de contexto longo > perplexidade [V]:**

- **OTT** ([2505.10938](https://arxiv.org/abs/2505.10938)) tabela 14, Llama-3-8B-ProLong-512k,
  KIVI-2: RULER médio **64K: 88,60 → 84,98 (−3,62)**; **128K: 87,42 → 81,51 (−5,91)**;
  multi-key NIAH **99,8 → 78,4 (64K)** e **100 → 69 (128K)**. **A degradação ~dobra de 64K para
  128K** e mora no retrieval, não na média.
- KVQuant [V]: RULER a 32K, fp16 56,40 / KIVI-2 39,78 / KVQuant-3bit 53,65 / 2 bits 36,54;
  LongBench 31,96 / 30,04 / 31,21.
- KIVI [V]: LongBench 44,52 → 44,27 a 2 bits — **mas a KIVI nunca testa 64K-128K**; a GEAR
  ([2403.05527](https://arxiv.org/abs/2403.05527)) também não.
- **Acúmulo [V]:** KVTuner — *"error accumulation over the whole model and long context length
  is noticeable and may lead to token flipping"*; GEAR — decoding *"compounds the error of each
  step"*.
- **Negativo [V]:** **nenhum paper** plota magnitude/variância de outlier de K ou V como função
  do comprimento do contexto (2K→128K). A pergunta "a variância muda com o comprimento" está
  **em aberto** na literatura.

## 5.5 Veredito para a hipótese da noite (`K q5_0` / `V q4_1`)

**A evidência direta e decisiva é do próprio upstream [V]:** comentário em
[PR #21038](https://github.com/ggml-org/llama.cpp/pull/21038), modelo **Qwen3.5-9B**, wikitext-2,
**KLD média** contra cache f16 (baseline 0,000782):

| K \ V | q4_0 | q4_1 | q5_0 | q8_0 | f16 |
|---|---|---|---|---|---|
| q4_0 | 0,007979 | 0,007163 | 0,006165 | 0,005336 | 0,005426 |
| **q4_1** | 0,006612 | 0,006014 | 0,004827 | 0,004005 | 0,004174 |
| **q5_0** | **0,005025** | **0,004181** | 0,002849 | 0,002310 | 0,002223 |
| **q8_0** | 0,003642 | **0,002920** | 0,001715 | 0,000866 | 0,000923 |

Três leituras, todas **a favor da hipótese**, com correções:

1. **`V q4_1` bate `V q4_0` em 6/6 linhas** (e em 6/6 na re-execução com rotação) — a metade V
   da hipótese está **diretamente suportada** ao preço de +0,0625 B/elem.
2. **`K q5_0` bate `K q4_1` em todas as linhas** (25-40% menos KLD) — o 5º bit vale mais que o
   `min` afim. **Não testar `K q4_1` como "K mais barato".**
3. **`K q8_0` bate `K q5_0`** por ~1,4× na KLD com V `q4_1` (0,002920 × 0,004181) ao preço de
   **+0,75 GiB**. Se a VRAM sobrar, é estritamente melhor.
4. **Aditividade [V]:** só-K `q8_0` Δ0,000141; só-K `q5_0` Δ0,001441; só-V `q4_1` Δ0,002203;
   só-V `q4_0` Δ0,002965 ⇒ **a 4-5 bits OS DOIS LADOS importam**, e o K é ~1,5× mais sensível,
   não 10×. A história "K é 10× mais sensível" é um fenômeno de **2 bits**.

**Correções de aritmética de bytes (as minhas estavam erradas):** o tamanho exato por lado a
131 072 tokens (2³¹ elementos) é: `q5_0` 1,375 GiB, `q4_1` 1,250, `q4_0` 1,125, `q8_0` 2,125.
Logo **K `q8_0` + V `q4_0` = 3,250 GiB** (não 2,81) e **K `q5_0` + V `q4_0` = 2,500 GiB**
(não 2,44). Subir o K de `q5_0` para `q8_0` custa **+0,75 GiB**; subir o V de `q4_0` para `q4_1`
custa **+0,125 GiB** — o quantizador afim no V é 6× mais barato que o bit extra no K.

**PPL é cega — usar KLD.** Do dono do projeto, no mesmo PR **[V]**: *"I don't think it has ever
been clear how much it degrades the quality. How many people are probably using `Q4_0` KV cache
and thinking 'oh, PPL is not that bad'."* A avaliação AIME25-x8 dele (gpt-oss-20b, K = V):
F16 **37,9%**, Q8_0 31,7 → 37,1 (com rotação), Q5_1 30,8 → 32,5, Q5_0 25,4 → 32,5, Q4_1
18,3 → 28,3, **Q4_0 2,0% → 21,7** — enquanto a PPL andou ~0,4% em toda a faixa.

**Veredito final [V para as citações, E para o nosso caso]:**
**manter `K q5_0 + V q4_1` como default é defensável e é o melhor ponto por byte.** Subir para
`K q8_0 + V q4_1` (3,375 GiB) é estritamente melhor se os 0,75 GiB sobrarem — e a decisão
interage com a verificação batelada (§6), que quer ~0,75 GiB de planos de estado.
**A alavanca de maior valor esperado, porém, não é o tipo: é a rotação de Hadamard (§3.1)** —
custo zero de memória, dot product preservado, medida pelo upstream em 4-10× menos dano de
quantização, e ela pode tornar `q5_0` no K suficiente onde hoje não é. **O que ainda é
extrapolação nossa:** QK-norm, arquitetura híbrida (48 GDN + 16 atenção), GQA 6:1 e
`head_dim 256` — nenhum dos quatro foi testado com esse par de tipos. A arquitetura híbrida é
fator **atenuante** (`[E]`: a KVTuner mostra acúmulo de erro por camada, e nós só cacheamos 16
de 64).

**Experimento minimamente suficiente:** semente e corpus fixos, **f16-KV como referência**;
medir **PPL *e* KLD média** (a KLD é o discriminador); grade f16, `q8_0/q4_0`, `q8_0/q4_1`,
`q5_0/q4_0`, **`q5_0/q4_1`**, `q4_1/q4_0` a **8K e 32K**; mais uma sonda passkey/needle a
**≥ 64K** (não a 8K — o retrieval é o que quebra primeiro, e quebra ~2× mais forte a 128K).
**Aceitação:** a 8K e 32K, ΔPPL ≤ **+0,5%** e KLD média ≤ **0,005**; needle ≥ **90% a 64K** e
≥ **80% a 128K**. **Rejeitar** se ΔPPL > +2%, KLD > 0,01 ou needle < 70% a 64K.
Comandos: `./scripts/compare_ppl.sh $MOD 10` e `./scripts/compare_llama_greedy.sh 32`
(mais um alvo novo de KLD — ver §7).

---

# 6. O número que já está medido: `check-batch-gpu` dá a curva de custo do batch

`tests/check_batch_gpu.hip` compara, no mesmo processo, a passada batelada contra N passadas de
um token, **bit-exatas**, e imprime (`docs/medicoes-m8.md:25-34`) **[V]**:

| N | per-token ms | batched ms | speedup | **fator de custo cf(N) = N/speedup** |
|---|---|---|---|---|
| 2 | 56,102 | 32,530 | 1,72× | **1,16** |
| 3 | 49,776 | 27,860 | 1,79× | **1,68** |
| 4 | 44,879 | 19,683 | 2,28× | **1,75** |
| 8 | 37,639 | 15,386 | 2,45× | **3,27** |
| 16 | 36,645 | 13,974 | 2,62× | **6,11** |

**Leitura:** processar 2 tokens custa **1,16×** um token; 4 custam **1,75×**; 8 custam 3,27×.
Isto é o efeito de **amortizar a desquantização do peso entre as N colunas** (é por isso que o
ganho por token é grande mesmo sendo *issue-bound*): o mesmo bloco de pesos é desquantizado uma
vez e multiplicado N vezes, que é literalmente o desenho do `mmvq` do llama.cpp (§1.6). A mesma
curva, em ms/token, está em `docs/autotuning-gfx1201.md:157-161` **[V]**: **16,3 (N=2), 7,7 (3),
4,9 (4), 1,9 (8), 0,87 (16)** — e o teste roda com o **modelo real** (`scripts/check_all.sh:21,79`
passa o IQ3_S).

**Cuidado honesto (importante):** o gate roda num **contexto curto** (a atenção e o KV são
desprezíveis ali), e no `forward_batch` o KV write, a atenção e a recorrência do GDN
**continuam sequenciais** (`docs/medicoes-m8.md:18-21`). Numa passada de verificação a 4K/131K
esses três entram por token e **não** estão nesses cf. Ou seja: os cf acima valem para a parte
batelada; para a **passada de tronco inteira** eles são um **limite inferior de custo** — e é
por isso que a projeção de baixo (feita pelo próprio repo, com a atenção/GDN incluídas) dá
1,3-1,5×, e não 2,3×.

**A projeção do próprio repo (`docs/medicoes-m8.md:75-80`) [V], com esses cf aplicados:**

| D | tokens esperados/rodada | verificação | rodada | ms/token | **ganho** |
|---|---|---|---|---|---|
| 2 | 2,62 | 65 ms | 72 ms | 27,5 | 1,31× |
| 3 | 3,27 | 84 ms | 94 ms | 28,6 | 1,26× |
| 4 | 3,84 | 79 ms | 91 ms | 23,8 | **1,51×** |
| 8 | 5,44 | 123 ms | 146 ms | 26,8 | 1,34× |

**Isto contradiz (para baixo) a projeção de `docs/mtp.md:189-197` ("~2,5×")**, que assume um
passe batelado custando o **mesmo** que um passe de um token (cf = 1,0). Com os cf medidos:
**1,3-1,5×, e o melhor ponto é D = 4.** O teto do D=8 é **pior** que o do D=4 — a taxa por
posição cai mais rápido do que o tokens/rodada sobe. **E a variante simples de rollback
("aceita a rodada inteira só") mede 1,0-1,1×, ou seja, não paga** (`docs/medicoes-m8.md:90-97`).

**O que isso significa para a coordenação:** o mesmo número (cf) governa **três** decisões —
profundidade do rascunho no MTP, se o prefill batelado vale, e se a árvore especulativa paga.
**A árvore aumenta N; com cf(8) = 3,27 e cf(16) = 6,11, uma árvore de 8-16 nós já come o ganho
que traria** **[E]**. Medir a árvore só depois de saber cf no modelo real.

**Como fechar a conta no modelo real:** estender o bloco `REPLAY` de
`tests/bench_matvec_shapes_gpu.hip:342-357` para rodar o mesmo passe com
`matvec_launch_batch(..., N)` para N = 2,3,4,8 (`include/rdna4/matvec.cuh:590-635`) e imprimir
ms e GB/s por N; rodar com
`timeout 900 ./scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MOD --budget-mib 12000 --reps 3`.
O `check-batch-gpu` já dá o **bit-exato** e a curva no modelo real, mas **em contexto curto**;
o bench dá o **passe de pesos isolado com e sem batch**, que é o que separa "amortiza a
desquantização" de "custa N× o arithmetic".

## 6.1 A peça que o MTP e o prefill precisam é a **mesma**

`docs/medicoes-m8.md:18-21` diz o que continua sequencial no `forward_batch`: **escritas no KV,
atenção token a token e a recorrência do GDN**. E `:47-49` diz que o prefill parou em 70 tok/s
contra 440 da referência *"a distância que resta é o laço interno do `vec_dot`"*. Juntando com o
orçamento do token: a 70 tok/s o passe de pesos é ~1,1 ms/token (11,12 GB / 16), então os
**13,7 ms/token** do prefill são andaime sequencial — os mesmos três itens.

**Conclusão de coordenação [E]:** a **atenção multi-linha com máscara causal** e o **scan de GDN
multi-token** são o caminho crítico de **duas frentes ao mesmo tempo** (verificação batelada do
MTP ⇒ 1,5×; prefill ⇒ de 73 para 200-400 tok/s **[E]**). Feitos uma vez, servem às duas; feitos
duas vezes, é a noite perdida. O coordenador deveria **dar dono único** a essas duas peças e
fazer as outras frentes consumirem a interface.

---

# 7. Top-5 "fazer a seguir" (ordenado por ganho esperado ÷ custo ÷ risco)

### 1. Andaime batelado: atenção multi-linha causal + scan de GDN multi-token — **dono único**

- **Referência:** `docs/medicoes-m8.md:18-21,44-49,75-80` (medido: o que ainda é sequencial e o
  que ele custa); `include/rdna4/attn.cuh` (uma CTA por cabeça, uma linha de consulta);
  `include/rdna4/graph.cuh:993-1027` (laço sequencial por token dentro do `forward_batch`).
- **Ganho:** prefill **73 → 200-400 tok/s** **[E]** (o passe de pesos por chunk já é 1,1 ms/token
  a N=16; o resto é andaime) e, de brinde, verificação do MTP a **1,5×** (§6).
- **Custo/risco:** é a peça mais difícil da noite (máscara causal no split-KV, scan do GDN com
  dependência sequencial, bit-exatidão). **Faça primeiro no `bench-attn-gpu`** e valide com
  `check-batch-gpu` (bit-exato) + `check_regression.sh`.
- **Comando:** `timeout 900 ./scripts/gpu-lock.sh ./build/check-batch-gpu` e
  `timeout 900 ./scripts/gpu-lock.sh ./build/bench --prefill 2048`.

### 2. Laço interno mais barato por byte de peso (MMQ tiled dp4a + ocupação RDNA4)

- **Referência:** `mmq.cu:380-383` (RDNA4: MMQ sempre bate "dequantizar + hipBLAS"),
  `mmq.cuh:111-119` (tile NE_K=32 com ativações `q8_1`), `mmq.cu:225-247` (ativação quantizada 1×
  por chunk), `mmvq.cu:436-490` (**RDNA4 quer `nwarps=8` com 1 coluna**), §6 (o mesmo efeito
  medido no nosso `check-batch-gpu`: 2,28× a N=4 por amortizar a desquantização).
- **Ganho [E]:** decode **+5-10%** (o matvec é 69% do passo e roda a 436/633 GB/s, issue-bound);
  prefill: destrava o teto junto com o item 1; **e sobe o teto do MTP** (cf menor ⇒ D=4 de 1,5×
  para perto de 1,8×).
- **Custo/risco:** médio; protótipo já autorizado por `docs/noite-regras.md:60-61`; gate de
  bit-exatidão existe (`check-batch-gpu`, `check-regression.sh`). **Primeiro passo barato e sem
  risco:** A/B de `--rows/--minb/--ilp/--pf` no `bench-matvec-shapes-gpu` contra a tabela de
  ocupação do upstream.
- **Comando:** `timeout 900 ./scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MOD --rows 4,8 --minb 0,128 --ilp 1,2,4 --pf`.

### 3. KV: rotação de Hadamard em K/V (com Q espelhado) antes de quantizar o cache

- **Referência:** `src/llama-graph.cpp:2855-2861,2891` (rotaciona Q e K com a mesma matriz, V, e
  aplica a inversa na saída), `src/llama-kv-cache.cpp:23,315-334` (gate `head_dim % 64 == 0`,
  `LLAMA_ATTN_ROT_DISABLE`), PR [#21038](https://github.com/ggml-org/llama.cpp/pull/21038),
  ExLlamaV2 `HADAMARD_Q` (`cache_q.cuh`, borboleta de 5 estágios).
- **Ganho [R/V]:** PPL com KV `q4_1` **212,5 → 22,3** e `q4_0` **62,0 → 46,25** (Qwen3-0.6B);
  AIME25 com `q4_0` **2,0% → 21,7%**; **custo de memória zero**. **[E]** no nosso token:
  ~0,02 ms/token.
- **Custo/risco:** **não é bit-exato** (introduz arredondamento) — coberto pela autorização de
  "mudança na política de atenção", com o desvio em % no diário; exige verificar a inversa
  **depois do merge do split-KV** (senão o resultado muda por parcial).
- **Experimento:** A/B de PPL/KLD com e sem rotação para `q4_0/q4_0` e `q5_0/q4_1` a 8K/32K
  (§5.5 dá os limiares) + `./scripts/compare_llama_greedy.sh 32`.
- **Decisão de política anexa:** manter `K q5_0 + V q4_1` (2,625 GiB) como default; **não**
  usar `q4_1` no K; subir para `K q8_0 + V q4_1` (3,375 GiB) se a VRAM sobrar (§5.5).

### 4. Atenção GQA compartilhada: desquantizar cada elemento **uma vez** para as 6 cabeças

- **Referência:** `include/rdna4/attn.cuh:291` (`h = blockIdx.x`, 6× o mesmo trabalho);
  medições `docs/medicoes-banda-e-gargalos.md:297-311,346-384`; **protótipo já escrito** em
  `tests/bench_attn_gpu.hip:106,193-202,549-588` (imprime `gqa ms` × `split GB/s` × speedup);
  desenho em `docs/kv-memoria-desenho.md:398,405`; e o mesmo item na lista do TRT-LLM
  ("multi-CTA fused MHA"), §3.3.
- **Ganho medido-adjacente:** 4K ~1 ms/token (**3%**), 16K ~3 ms (**7%**)
  (`medicoes-banda-e-gargalos.md:379-381`); a 64K a estimativa da frente é **até +18%**
  (`kv-memoria-desenho.md:398`); a 131K **[E]** o teto é maior (atenção ≈ 40 dos 71 ms) —
  se cair para 10 ms, o token vai a ~41 ms (**14,0 → ~24 tok/s**).
- **Custo/risco:** pressão de registrador (6 q + 6 acc), `__syncthreads` por bloco de chaves,
  **sem `global_load_lds` em gfx1201**; o protótipo do M7 foi 8-12× mais lento por colapso de
  grade — **medir no bench antes de mexer no `attn.cuh`**.
- **Comando:** `timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu q8_0 4096 16384 65536 131072 --splits 2,4,8,16`.

### 5. Fusões nomeadas (§1.7) + HIP graph (§1.8): os ~4 ms de imposto de despacho

- **Referência:** `ggml-vulkan.cpp:689-691,18153-18315`; instrumento pronto em
  `tests/bench_matvec_shapes_gpu.hip:359-404`; inventário nosso
  (`medicoes-banda-e-gargalos.md:523-524`).
- **Ganho [E]:** **4-7% no decode** e **mais no prefill**; ordem por retorno:
  `SSM_CONV_*_SILU` (48×2→48), `RMS_NORM_MUL_ROPE_*_SET_ROWS` (16 × 3-4),
  `SILU_MUL`/`SIGMOID_MUL`, `MULTI_ADD`, `MUL_MAT_ADD`.
- **Custo/risco:** baixo; bit-exatidão é gate. **Ressalva medida:** HIP graph move custo de
  **submissão**, e os gaps aqui parecem **GPU-side** (§1.8) — medir o valor real imprimindo a
  linha `graph replay ... ms/token of launch tax recoverable` antes de apostar nele.
- **Comando:** `timeout 900 ./scripts/gpu-lock.sh ./build/bench-matvec-shapes-gpu $MOD --reps 3`.

---

# 8. "Não faça isto" (e por quê)

1. **PagedAttention / tabela de blocos.** Uma sequência ⇒ fragmentação externa **zero**; a
   indireção entra no laço mais quente da atenção (que já é issue-bound) e o paper mede o preço
   dela: **20-26% de latência a mais** no kernel de atenção (§2). O único pedaço útil (alocar
   sob demanda) é flexibilidade de VRAM, não velocidade.
2. **Continuous / in-flight batching (vLLM, TRT-LLM).** Puro multi-inquilino: ganho **0** com
   batch 1. Agravante nosso: cada sequência exige o **próprio estado GDN** (0,15 GiB) e o scan
   do GDN passa de 48 para 48×n_seq kernels. **Reabrir só se o servidor virar multiusuário, com
   número na mão.**
3. **Árvore especulativa (Medusa/EAGLE-2) antes de saber o cf real.** A árvore **aumenta N**, e
   cf(8) = 3,27 e cf(16) = 6,11 já comem o ganho (§6); pior, cada nó custa **151 MB de plano de
   estado** (o KV, 65 KB, é o barato). É por isso que o SGLang restringe o ReplaySSM a
   `topk <= 1` (§4.4). Se for tentar, **um ramo de 2 nós na profundidade 2**.
4. **Prefix cache em árvore / RadixAttention para o CLI.** Estruturalmente hostil ao nosso
   modelo: o estado recorrente (0,15 GiB) é função do prefixo, então um ponto de cache custa
   **~4 356 tokens do KV inteiro** em bf16 (§3.2) e o hit de KV sozinho recupera no máximo os
   16/64 das camadas. **Versão útil = 1 slot, no servidor, para prompt de sistema reusado** — e
   só depois de medir.
5. **Chunked prefill para vazão.** Cada chunk é outra passada sobre 11,12 GB de pesos; os ganhos
   do Sarathi-Serve são de concorrência sob SLO. Só vale para interatividade, com chunk ≥ 2048.
6. **Qualquer coisa com MFMA / coopmat / tensor core no decode.** Não existe MFMA em RDNA4 (não
   compila); o decode é GEMV; e uma medição independente em RDNA4 mostra que **dobrar o rate de
   matriz deixou o decode igual** (§3.4). `wmma i32` só entra em prefill com N ≥ 8-16.
7. **Layout de cache do ExLlamaV2 (escalas em fp16, packing linear).** O ganho é a **rotação**
   (§7.3), não o layout; trocar o layout **quebra a byte-identidade com o llama.cpp** e portanto
   o oráculo, o `compare_ppl` e o `compare_llama_greedy` — a nossa rede de segurança.
8. **Quantização de chave por canal / pré-RoPE (KIVI, KVQuant).** É a resposta certa *se* 4-5
   bits no K se mostrarem insuficientes, mas hoje: (a) exige layout que o llama.cpp não tem ⇒
   quebra a byte-identidade; (b) pré-RoPE exige rotacionar por elemento dentro da atenção
   (**~4,3 GFLOP/passo** extra a 131K **[E]**) — e o próprio Apêndice N do KVQuant mostra que,
   para quantização **por token**, pós-RoPE é **melhor**; (c) escala por canal adiciona trabalho
   por elemento num kernel já issue-bound. **Plano B medido, não plano A.**
9. **Copiar a materialização f16 do KV para a atenção (§1.2).** É o que o upstream faz quando o
   KV quantizado encontra um kernel que quer f16: **~8,6 GB por passe de prefill a 131K [E]**.
   Nós já desquantizamos inline — **não regredir** ao escrever o prefill batelado.
10. **Re-rodar o prefixo aceito para "desfazer" o GDN.** 3 tokens × 5,63 ms de GDN = 17 ms,
    metade de um passe de tronco. Use **planos de estado por índice** (§4.4), não replay.
11. **Acelerar o rollback do estado recorrente como se ele fosse o gargalo.** A própria
    referência do ReplaySSM mede **~1,0× em B=1** — o nosso regime; e a nossa medição da
    variante simples deu **1,0-1,1×** (`docs/medicoes-m8.md:90-97`). **Não super-engenheirar.**

---

# 9. Fontes

## 9.1 Código local (llama.cpp, `df03399b8`)

- `ggml/src/ggml-common.h:195-256` — layouts `block_q4_0/q4_1/q5_0/q8_0`
- `ggml/src/ggml-quants.c:113,150,187,222-226,276,479,500` — refs de quant/dequant
- `ggml/src/ggml-cuda/cpy.cu:299-346,515-525` — store f32 → `q4_1`/`q5_0`
- `ggml/src/ggml-cuda/mmq.cu:225-249,266-392`, `mmq.cuh:8,111-119` — MMQ, tile, `q8_1`
- `ggml/src/ggml-cuda/mmvq.cu:318-411,436-490,726-735`, `mmvq.cuh:3` — GEMV dp4a multi-coluna
- `ggml/src/ggml-cuda/fattn.cu:459,583-695,707-722`, `fattn-common.cuh:53-82,1010` — FA e KV quant
- `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl:22-25,57-77,126-138`
- `ggml/src/ggml-vulkan/ggml-vulkan.cpp:689-691,794-798,969,9515,18153-18315` — fusões
- `src/llama-context.cpp:3697-3730` — restrições de KV quantizado
- `src/llama-graph.cpp:2855-2861,2891` — rotação de Hadamard em Q/K/V e inversa na saída
- `src/llama-kv-cache.cpp:23,312-334` — `ggml_gen_hadamard`, gate de rotação
- `src/llama-memory-recurrent.cpp:101,193-203` — planos `n_rs_seq` e rollback por índice
- `src/llama-arch.cpp:1104-1120` — `llm_arch_supports_rs_rollback` (inclui `QWEN35`)
- `src/models/qwen35.cpp:485-644` — grafo MTP
- `common/speculative.cpp:750,1330-1770,2903-2990`, `common/common.h:326-332,394-400` — `draft-mtp`

## 9.2 Medições e código deste repositório

- `docs/medicoes-banda-e-gargalos.md:259,297-311,346-384,478-493,519-527`
- `docs/medicoes-m8.md:18-34,44-49,76-97` — `forward_batch`, **curva cf(N)**, projeção 1,3-1,5×
- `docs/mtp.md:141-197` — aceitação por D, custo do passo de rascunho
- `docs/kv-memoria-desenho.md:398,405,415-426,450-457`
- `include/rdna4/{kv.h,attn.cuh,matvec.cuh,gdn.cuh,graph.cuh}`; `tests/{bench_attn_gpu.hip,bench_matvec_shapes_gpu.hip,check_batch_gpu.hip}`

## 9.3 Papers e upstream (URLs)

- DeepSeek-V3: <https://arxiv.org/abs/2412.19437> (MTP §2.2; 85-90% e 1,8× em §5.4.3)
- EAGLE-1/2/3: <https://arxiv.org/abs/2401.15077>, <https://arxiv.org/abs/2406.16858>, <https://arxiv.org/abs/2503.01840>
- Medusa: <https://arxiv.org/abs/2401.10774> (tabela 3: 1,5× → 2,2× → 2,8×)
- Leviathan et al.: <https://arxiv.org/abs/2211.17192> (Eq. 1, Teorema 3.8, Corolário 3.9)
- MagicDec: <https://arxiv.org/abs/2408.11049>; AdaEDL: <https://arxiv.org/abs/2410.18351>; Sequoia: <https://arxiv.org/abs/2402.12374>
- ReplaySSM (vLLM RFC): <https://github.com/vllm-project/vllm/issues/49232>
- vLLM PagedAttention: <https://arxiv.org/abs/2309.06180> (bloco 16, §7.1: +20-26% de latência)
- SGLang / RadixAttention: <https://arxiv.org/abs/2312.07104> (+ `mamba_radix_cache.py`)
- KV: KIVI <https://arxiv.org/abs/2402.02750>; KVQuant <https://arxiv.org/abs/2401.18079>;
  KVTuner <https://arxiv.org/abs/2502.04420>; KV-AdaQuant <https://arxiv.org/abs/2502.15075>;
  AsymKV <https://arxiv.org/abs/2410.13212>; DiffKV <https://arxiv.org/abs/2412.03131>;
  GEAR <https://arxiv.org/abs/2403.05527>; OTT <https://arxiv.org/abs/2505.10938>;
  Atom <https://arxiv.org/abs/2310.19102> (negativo); WKVQuant <https://arxiv.org/abs/2402.12065>;
  QAQ <https://arxiv.org/abs/2403.04643>; ZipCache <https://arxiv.org/abs/2405.14256>;
  KVarN <https://arxiv.org/abs/2606.03458>; ViT-22B <https://arxiv.org/abs/2302.05442>;
  QK-Norm <https://arxiv.org/abs/2010.04245>
- llama.cpp: PR de rotação <https://github.com/ggml-org/llama.cpp/pull/21038>; MMQ em RDNA4
  <https://github.com/ggml-org/llama.cpp/pull/18537>; discussão de KV K-only
  <https://github.com/ggml-org/llama.cpp/discussions/21297>; graphs no ROCm
  <https://github.com/ggml-org/llama.cpp/pull/11867>
- ExLlamaV2: <https://github.com/turboderp-org/exllamav2> (`cache_q.cuh`, `cache.cu`, `doc/qcache_eval.md`)
- Hazy "Look Ma, No Bubbles!": <https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles>
