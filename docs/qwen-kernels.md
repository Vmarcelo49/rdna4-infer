# Qwen3.8-27B: o que o modelo pede vs. o que os kernels fazem

Metade de **leitura de código + auditoria de ISA** da tarefa 2. **Nenhuma execução na GPU.**
As únicas compilações foram de uma unidade de tradução *standalone* em `/tmp`
(`amdclang++ -x hip --offload-arch=gfx1201 -O3 -S`), instanciando os kernels reais deste
worktree — sem modelo, sem runtime, sem `hipMalloc`.

Onde eu afirmo "a ISA mostra X", o número vem da contagem de instruções por kernel na saída
de `-S` daquele TU (**488 funções de device** analisadas); onde afirmo medição, cito o
documento do repo.
**[minha conta]** marca aritmética minha. **[hipótese]** marca o que não foi medido.

---

## 1. As especificidades do Qwen3.8-27B, verificadas no modelo

Tudo abaixo saiu de um dumper CPU-only que abre só o cabeçalho GGUF e chama
`parse_qwen35_config()` + `validate_qwen35_layout()` **do motor**. Saída idêntica nos dois
arquivos UD (`LAYOUT_VALID 1`):

```
block_count=65  full_attention_interval=4  nextn_predict_layers=1
embedding_length=5120   feed_forward_length=17408   context_length=262144
head_count=24  head_count_kv=4  key_length=256  value_length=256
rope.dimension_count=64  rope.dimension_sections=11,11,10,0
rope.freq_base=10000000.0   attention.layer_norm_rms_epsilon=1e-06
ssm: conv_kernel=4  state_size=128  group_count=16  time_step_rank=48  inner_size=6144
```

| especificidade | valor | onde é lido / validado no motor |
|---|---|---|
| **GQA 24:4 = 6×** | `head_count=24`, `head_count_kv=4` | lido `src/backend/model.cpp:201-206`; validado `graph.cuh:411-413` (`NH % NKV != 0` → erro) |
| **head_dim 256** | `key_length=value_length=256` | lido `model.cpp:207-211`; igualdade K=V exigida em `graph.cuh:406-409`; `head_dim()` em `graph.cuh:211` |
| **`n_rot` 64, split-half (IMRoPE)** | `rope.dimension_count=64`; `sections=[11,11,10,0]` (2×soma = 64) | lido `model.cpp:228-234`; a invariante **`2*sum(sections) == dimension_count`** está em `model.cpp:252-264` e o split exato `[11,11,10,0]` é **assertado para os 65 blocos reais** em `model.cpp:271-282`; `n_rot` par e ≤ head_dim em `graph.cuh:415-417` |
| **RoPE freq base 1e7** | `1e7` | `model.cpp:289` (`kv_f64("qwen35.rope.freq_base")`); consumido em `graph.cuh:655` → `rope_launch(..., base, ...)` |
| **QK-norm em q e k** | `attn_q_norm`/`attn_k_norm`, F32, `[256]` | carregados `graph.cuh:467-468` (`upn_f32`); aplicados `graph.cuh:639-647` com `nrows = NH` / `NKV` e `ncols = HD` ⇒ **uma RMS por cabeça** |
| **MLP 17408** | `feed_forward_length=17408` | `model.cpp:198`; `graph.cuh:398`; os três tensores `ffn_gate/up/down` são validados por dim em `model.cpp:99-121` |
| **Layout híbrido** | 64 camadas de tronco = **48 GDN + 16 full-attention**; `is_full_attention_layer(i,4)` ⇒ `i ∈ {3,7,…,63}`; **+1 bloco MTP** (`blk.64`, 15 tensores) | `model.h:5-19` (spec), `model.h:59-61` (`(i+1) % interval == 0`), `graph.cuh:216` (`is_recr`: `(il+1)%4 != 0 && il < n_layer()`), `n_layer() = block_count - nextn_predict_layers`; validado por `validate_qwen35_layout()` (`model.cpp:126-157`) |
| **Par com q8_1 de 32** | toda ativação é `block_q8_1` de 32 elementos | `quants.h:22,78-81`; `quantize_q8_1_block` em `matvec.cuh:28-54`; 8 blocos q8_1 por bloco de 256 pesos |
| **GDN: 5·group·state = 10240** | `attn_qkv` `[5120,10240]` | `model.cpp:102-105` calcula `5*g*S` e valida a dim |
| **MTP preservado e ignorado** | `nextn_predict_layers=1` | `model.cpp:247-251` valida `< block_count`; `SPEC.md:22` "ler e ignorar"; `graph.cuh` nunca carrega `blk.64` |

Duas derivadas que os kernels usam e que vale explicitar [minha conta, confirmada pelas dims
do arquivo]:
`d_inner = time_step_rank × state_size = 48×128 = 6144` (`graph.cuh:399`),
`chan = 2×(group_count×state_size) + d_inner = 2×2048 + 6144 = 10240` (`graph.cuh:400-401`),
e `q` por cabeça = `2 × key_length = 512` (`[q(256) | gate(256)]`, `model.cpp:71-73`,
`attn.cuh:6-7`).

---

## 2. Cada kernel explora a especificidade, ou é genérico?

### 2.1 Atenção: o GQA 6:1 é usado para *indexar*, não para *compartilhar*

`attn_kernel` (sem split) e `attn_split_kernel` põem **uma CTA por cabeça de consulta**:

- `attn.cuh:190` — `attn_kernel<<<n_head, threads, smem, stream>>>` (24 CTAs);
- `attn.cuh:439` — `dim3 grid(n_head, n_splits)` (24×S CTAs no caminho com split);
- `attn.cuh:100` e `:300` — `const int kvh = h / (n_head / n_head_kv);`
- `attn.cuh:119` e `:317` — `kr = k + (j*n_head_kv + kvh) * krow`

O endereço lido depende **só de `kvh` e `j`**. Logo as 6 CTAs `h = 6·kvh … 6·kvh+5`
calculam endereços **idênticos** e cada uma percorre a KV inteira por conta própria: a
redundância de leitura é **6×**, exatamente o que o `docs/rocm-estudo.md:318` descreve
("a 64K a atenção está em ~1,56 TB/s de L2 por causa da redundância 6×; teto ~2,4× na
atenção ⇒ até +18% a 64K"). Volume [minha conta]: a 64K, f16, por camada de atenção,
`65536 × 4 × 256 × 2 B × 2 (K+V) = 268 MB`; ×16 camadas = **4,29 GB/token** — o mesmo
"4,3 GB de KV f16" de `docs/medicoes-m5.md:131`, que já é a cifra **com** as 6 leituras.
Compartilhar levaria a 0,72 GB ⇒ −3,6 GB/token, ou seja **~24% do tráfego do token**
(11,1 GB de pesos + 4,3 GB de KV): é a maior alavanca de tráfego isolada do repo acima de 32K.

Por que ainda não foi feito: o protótipo do M7 foi **8-12× mais lento**
(`docs/rocm-estudo.md:353-354`) — naquele momento o kernel era latência-bound com 24 CTAs
(24 de 64 CUs, `rocm-estudo.md:201`), e reduzir para 4 CTAs pioraria. O que mudou desde
então é a **grade larga**: com `kAttnSplitMin = 512` (`graph.cuh:308`, M8) são 8 splits a 4K e
16 a 8K+, ou seja 384 CTAs a 64K. Uma variante "CTA por (cabeça KV, split) com as 6 consultas"
daria 4×16 = 64 CTAs — que **enche as 64 CUs sem staging em LDS** só se cada CTA processar as
6 consultas em sequência reusando a mesma linha K/V lida uma vez. É a forma que o
`rocm-estudo.md:354` diz que falta tentar. **[hipótese]** — não foi medida.

### 2.2 RoPE: split-half correto, só nos primeiros 64 dims

`attn.cuh:24-53`. O kernel:

- `pairs = n_rot/2` (`:31`), `theta = pos[t] * powf(freq_base, -2p/n_rot)` (`:38`);
- pareia `p` com `p + n_rot/2` (`jc`, linhas `:48-52`) — **split-half**, não pares adjacentes;
- escreve só `base[p]` e `base[jc]`, com `base = x + (t*n_heads + h)*head_dim` (`:47`), ou seja
  **as dims `[0,64)` de cada cabeça**; `[64,256)` ficam intocadas.

O comentário `:42-46` registra por que split-half e não NEOX: `GGML_ROPE_TYPE_IMROPE` é
lowerado por `rotate_pairs(n_dims, n_dims/2, …)`. Aplicado a **q (24 cabeças) e k (4)** em
`graph.cuh:656-657`, depois das QK-norms e depois do `hipMemcpy` da posição (`:651`).
**Veredito: correto e específico.**

Custo/resíduo: `rope_kernel` tem **731 instruções** [ISA minha] para 768 threads úteis (q) e
128 (k) por lançamento, porque calcula `powf` + `cosf` + `sinf` **por elemento** (`:38-40`).
Só a RoPE lê `d_pos_`, e é a razão do `hipMemcpy` síncrono de 4 bytes por camada de atenção
(16/token, `graph.cuh:651`; `docs/rocm-estudo.md:159-165` mede ~0,05 ms = 0,15%). Passar `pos`
por valor ao kernel e pré-computar `cos/sin` por posição elimina os dois. **[hipótese]**

### 2.3 QK-norm: sim, por cabeça

`graph.cuh:639-647`: `rms_norm_launch(d_attnout_, attn_q_norm, d_attnout_, NH, HD, eps)` e o
mesmo para k com `NKV`. `nrows = 24`/`4`, `ncols = 256` ⇒ uma RMS independente por cabeça,
com peso de 256 elementos — que é o que o modelo pede (`attn_q_norm`/`attn_k_norm` são `[256]`).
Ordem correta: **projeção → deinterleave do gate (`:634`) → QK-norm → RoPE → gravação na KV**.
No caminho em batch é idêntico (`graph.cuh:884-891`), com `nrows = N`.

### 2.4 GDN / delta rule: usa os três números, mas ocupa metade da placa

`delta_rule_kernel` (`gdn.cuh:57-86`) e `delta_rule_launch` (`:88-95`):

```
delta_rule_kernel<<<n_v_heads, ((S + 31)/32)*32, 0, stream>>>(...)   // <<<48, 128>>>
const int h = blockIdx.x;      // cabeça de valor  -> time_step_rank = 48 = grid
const int j = threadIdx.x;     // linha da state    -> state_size = 128 = threads
const int kh = h % n_k_heads;  // repetição GQA     -> group_count = 16 = n_k_heads
```

Chamada em `graph.cuh:770`: `delta_rule_launch(q_c, k_c, v_c, d_gate_, d_beta_, state, v_c,
nvh, nkh, S)` com `nvh=48`, `nkh=16`, `S=128` (`graph.cuh:719-720`). Ou seja **os três números
são explorados**: a grade é `time_step_rank`, a largura do CTA é `state_size` (1 thread por
linha da state transposta, `gdn.cuh:8-10,68-69`, o que dispensa sincronização dentro da
cabeça) e a repetição 48/16 = **3×** vem do `%`.

Onde ele é genérico/ruim:

- **48 CTAs em 64 CUs**: 16 CUs ociosas, e 48×128 = 6144 threads = 192 warps em 128 SIMD32 =
  **1,5 warp por SIMD** [minha conta]. Nada no kernel divide a dimensão `j` para crescer a
  grade.
- Cada thread faz um laço serial de **2×S = 256 iterações** com dois percursos de memória
  (`row[i]`, `kd[i]`, `qd[i]`), com **carga escalar de 4 bytes** — as linhas são contíguas,
  `float4` cortaria as instruções de load 4×. [ISA minha: `delta_rule_kernel` = 145 instruções
  no total, sem nenhuma vetorização de memória]
- **A state é f32 e é o tráfego real do GDN** [minha conta]: `48 camadas × 48 cabeças ×
  128×128 × 4 B = 3,1 MB por camada`, lidos **e** escritos ⇒ **302 MB/token ≈ 0,9 ms a
  336 GB/s (≈2,5% do token)**. Isso não aparece em nenhum documento do repo. Não é conversão
  redundante (a recorrência tem de ler e escrever), mas é o item que explica o "resto" de
  `rocm-estudo` §A.1 além dos gaps de lançamento.

### 2.5 Cabeça LM: matvec normal, 874 MB por token, no teto de leitura

`graph.cuh:1206` — `proj(output_, d_x_, d_logits_, n_vocab, E, err)`, com `n_vocab = 248320`.
É literalmente o mesmo caminho de qualquer peso: `quantize_q8_1_kernel` + `matvec_launch`
com a forma `MtShape<5>` = `{rows=4, wpr=1}` e `MtIlp<5> = 2` (`matvec.cuh:449,468`) ⇒
`grid = 248320/4 = 62080` CTAs. Nada de especial: **sem fusão de argmax, sem softmax no
device, sem tratamento de vocabulário grande.** `output.weight` é `q5_k` com
**874.086.400 B** = **7,86% do tráfego do token** [gerado + minha conta], medido a 627 GB/s =
1,39 ms = **98% do teto de leitura** (`rocm-estudo.md:128-133`). **Veredito: está no limite;
só "não ler" (MTP/especulação) muda isso.**

### 2.6 Matvec GEMV/batch: genérico por tipo, afinado por medição

`matvec_kernel_gen` (`matvec.cuh:185-279`) é genérico sobre `T::dot`; o único ajuste por tipo
são as tabelas medidas `MtShape`/`MtIlp` (`matvec.cuh:444-477`) e o `RD_SHIP` de
`matvec_launch` (`:518-540`). O caminho em batch (`matvec_kernel_batch`, `:296-386`) é o
mesmo corpo com `N` acumuladores, bit-idêntico por construção (comentário `:286-291`).
**Genérico por design** — e é onde está a distância para o llama.cpp (§3.4).

---

## 3. Auditoria de matrix cores (a parte crítica)

### 3.1 O que existe no código: nada

Busca exaustiva (meu `grep` + levantamento independente em `include/ src/ tests/ scripts/` e
`/home/marcelo/Projetos/llama.cpp`):

| padrão | neste repo | em llama.cpp |
|---|---|---|
| `__builtin_amdgcn_mfma` | **0** | 14 sites, todos em `ggml/src/ggml-cuda/mma.cuh` |
| `__builtin_amdgcn_wmma` | **0** (1 hit, e é prosa: `docs/rocm-estudo.md:263`) | 13 sites, todos em `mma.cuh` |
| `amdgcn_mma`, `rocwmma`, `<rocwmma`, `v_mfma`, `v_wmma` | **0** | — |
| `cooperative_matrix` / `coopmat` / `VK_KHR_cooperative_matrix` | **0** | só em `ggml-vulkan/` |
| `sdot4` / `udot4` (código) | **0** | `common.cuh:717` |
| **inline asm (`__asm`)** | **0** | — |

O único intrínseco de aritmética de hardware do motor é
`__builtin_amdgcn_sudot4` em `vecdotq.cuh:88`, que serve **96 sítios** de `ggml_cuda_dp4a`
(`vecdotq.cuh:84-100`), dentro dos 14 `vec_dot_*_q8_1`. Mais `__builtin_amdgcn_perm`
(24 sítios) e `__builtin_amdgcn_s_prefetch_data` (`matvec.cuh:157`). Não há `kernels/`, não
há GEMM, não há caminho alternativo escondido.

### 3.2 O que o compilador aceita para gfx1201 (medido, compilando)

Cada linha é um `.hip` mínimo em `/tmp` compilado com
`amdclang++ -x hip --offload-arch=gfx1201 -O3 -S`:

| intrínseco | resultado | ISA emitida |
|---|---|---|
| `__builtin_amdgcn_mfma_f32_16x16x16f16` | **ERRO** `needs target feature mai-insts` | — |
| `__builtin_amdgcn_mfma_i32_16x16x16i8` | **ERRO** (mesma classe `mai-insts`) | — |
| `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(...)` | **OK** | **`v_wmma_i32_16x16x16_iu8`** |
| `__builtin_amdgcn_sdot4` | **ERRO** `needs target feature dot1-insts` | — |
| `__builtin_amdgcn_sudot4` | OK | **`v_dot4_i32_iu8`** |
| `__builtin_amdgcn_udot4` | OK | `v_dot4_u32_u8` |
| `__builtin_amdgcn_udot8` | OK | `v_dot8_u32_u4` |
| `__builtin_amdgcn_perm` | OK | `v_perm_b32` |
| `__builtin_amdgcn_s_prefetch_data` | OK | `s_prefetch_data` |
| `__builtin_prefetch` | OK, mas | **nenhuma instrução** (confirma `rocm-estudo.md:336-340`) |

**Três conclusões duras:**

1. **MFMA não existe em gfx1201.** Não é "não usado": é *inexistente* — o alvo não tem a
   feature `mai-insts`. Qualquer código com MFMA **não compila** para gfx1201. Portanto não
   pode haver "fallback silencioso para MFMA" neste motor, nem em nenhum outro: o modo de
   falha é erro de compilação, o mais alto possível. (No llama.cpp a mesma coisa é explícita:
   `AMD_MFMA_AVAILABLE` exige `defined(CDNA)` em `ggml-cuda/common.cuh:275-277`.)
2. **WMMA int8 existe e é alcançável** (`v_wmma_i32_16x16x16_iu8`). É o único matrix core
   utilizável para os nossos pesos quantizados, e o motor usa **zero** dele.
3. **O dp4a do motor é a instrução certa, em rate cheio.** `v_dot4_i32_iu8` é full-rate em
   gfx1201, igual a `v_fma_f32` (`rocm-estudo.md:206`), e `sdot4` (a variante que *não*
   compila aqui) é corretamente evitada pelo `#if defined(__gfx1201__)` de
   `vecdotq.cuh:86-88`. O único fallback escalar (`:90-98`) é para arquiteturas que o build
   nunca gera (`CMakeLists.txt:32`: só `--offload-arch=gfx1201`).

*Não confirmado*: não consegui estabelecer a assinatura dos WMMA de f16/bf16 para gfx12 (as
tentativas deram erro de aridade); o de int8, que é o que importa para pesos quantizados,
está confirmado. Os nomes de f16 existem no llama.cpp (`mma.cuh:1232,1267`), então o caminho
existe — só não o validei na ISA.

### 3.3 A ISA de **cada kernel quente** (contagem por kernel, `-S` deste worktree)

Contagem de instruções por corpo de kernel; `dp4a` = `v_dot4_i32_iu8`; `WMMA`/`MFMA` contam
`v_wmma*`/`v_mfma*`/`v_swmmac*`:

| kernel | instruções | **WMMA** | **MFMA** | dp4a | fma/fmac f32 |
|---|---|---|---|---|---|
| `matvec_kernel_gen<TIQ3S_S,8,1,1>` — **GEMV de 33,4% do tráfego** | 463 | **0** | **0** | 32 | 0 |
| `matvec_kernel_gen<TIQ4XS,8,1,2>` — 23,3% | 397 | **0** | **0** | 6 | 1 |
| `matvec_kernel_gen<TIQ3XXS_S,2,1,1>` — 16,8% | 461 | **0** | **0** | 32 | 0 |
| `matvec_kernel_gen<TQ5K,4,1,2>` — **cabeça LM** (7,9%) | 480 | **0** | **0** | 24 | 0 |
| `matvec_kernel_gen<TQ3K,1,8,2>` — o mais caro dos 14 | 873 | **0** | **0** | 12 | 0 |
| `matvec_kernel_batch<TIQ3S_S,8,1,1,16>` — **prefill** | 2016 | **0** | **0** | 512 | 0 |
| `matvec_kernel_batch<TQ5K,4,1,2,16>` — prefill | 3015 | **0** | **0** | 384 | 0 |
| `attn_kernel<F16,F16,8>` | 947 | **0** | **0** | 0 | 28 |
| `attn_split_kernel<F16,F16,8>` | 1025-1133 | **0** | **0** | 0 | 10-11 |
| `attn_merge_kernel` | 222 | **0** | **0** | 0 | 5 |
| `rope_kernel` | 731 | **0** | **0** | 0 | 19 |
| `rms_norm_kernel` | 256 | **0** | **0** | 0 | 4 |
| `l2_norm_kernel` | 205 | **0** | **0** | 0 | 1 |
| `delta_rule_kernel` (GDN) | 145 | **0** | **0** | 0 | 3 |
| `conv1d_state_kernel` | 145 | **0** | **0** | 0 | 4 |
| `quantize_q8_1_kernel` | 295 | **0** | **0** | 0 | 4 |
| `dequant_kernel_256<FnIQ3S>` | 157 | **0** | **0** | 0 | 0 |
| `mul_kernel` / `add_kernel` / `scale_kernel` | 31 / 31 / 27 | **0** | **0** | 0 | 0 |

Zero `v_wmma`, zero `v_mfma`, zero `v_swmmac` em **todos** os kernels do motor. Nenhum deles
usa `v_pk_fma_f16` nem `v_dot2_f32_f16` tampouco.

### 3.4 Veredito por kernel: houve "fallback silencioso"?

A pergunta do usuário (vLLM em gfx1201 "oficialmente suportado" mas caindo num caminho
genérico lento):

| kernel | veredito | evidência |
|---|---|---|
| **matvec GEMV (decode)** | **NÃO há fallback.** Roda na instrução certa, em rate cheio. | `v_dot4_i32_iu8` na ISA (32 sítios no `TQ3S_S`); full-rate em `rocm-estudo.md:206`; `sdot4` nem compila (erro, não degradação) |
| **matvec batch (prefill)** | **Não é fallback, mas é a lacuna real.** Faz o que promete (dp4a); o problema é que o *design* não usa WMMA, e o llama.cpp usa. | ISA do `matvec_kernel_batch<TIQ3S_S,…,16>`: 2016 instruções, 512 dp4a, **0 WMMA**. Contraste: `rocm-estudo.md:262-265` (llama.cpp, `ne11>8` → MMQ com WMMA int8 + LDS) |
| **atenção** | Sem matrix core por natureza (softmax+FMA). Mas **não usa f16 empacotado** mesmo com KV f16. | ISA: 947 instruções, 28 `fmac_f32`, 0 `v_pk_fma_f16`/`v_dot2_f32_f16`. `rocm-estudo.md:208`: esses são 1,5× o rate do FMA fp32 |
| **GDN / delta rule** | Sem matrix core aplicável (fp32, recorrência de 128 passos). Sub-paralelizado, não "lento por fallback". | ISA: 145 instruções, 3 fmac; 48 CTAs × 128 threads em 64 CUs |
| **RMSNorm / L2Norm** | Redução pura; matrix core não se aplica. | ISA: 256 / 205 instruções, 4 / 1 fmac, 9 loads LDS cada |
| **RoPE** | Sem matrix core aplicável. Custo está em `powf/cosf/sinf` por elemento. | ISA: 731 instruções, 19 fmac/fadd, 3 loads globais |
| **cabeça LM** | Mesmo matvec; **no teto de leitura** (627 GB/s = 98% do medido). | `rocm-estudo.md:128-133`; `graph.cuh:1206` usa o caminho comum |

**Resposta curta ao usuário: confirmei a *classe* de falha como risco, e refutei-a para os
nossos kernels quentes — com uma exceção qualificada.** Em nenhum kernel o compilador trocou
silenciosamente uma instrução boa por uma ruim: onde a instrução boa não existe (MFMA) o build
**quebra**, e onde existe e é a certa (dp4a) o motor a usa. A exceção qualificada é o
**prefill/batch**: ali existe uma instrução melhor disponível (`v_wmma_i32_16x16x16_iu8`) e o
motor não a usa — mas isso é uma ausência de design, não um fallback: o kernel não promete
WMMA em lugar nenhum, e `docs/rocm-estudo.md:317` já registra isso como o item 6 do ranking
("milestone, outro agente").

### 3.5 O design *poderia* usar matrix cores para os pesos IQ/K-quant?

Análise, não medição.

**Restrição de forma (decisiva para o decode):** WMMA int8 é `16×16×16`: consome um tile A
16×16 e um B 16×16 e produz 16×16. O GEMV de decode tem **M = 1** (um vetor de ativação).
Um WMMA com M=16 computaria 16 *linhas de peso* contra 16 *vetores de ativação* — isso é o
caminho em **batch**, não o decode. **Portanto matrix core não ajuda o decode de forma
alguma**, e a rota dp4a atual está certa para ele. Ela só ajuda prefill (N ≥ 8-16), que é
exatamente o corte do llama.cpp (`ne11 > 8`, `rocm-estudo.md:262`).

**Restrição de formato (o custo):** os blocos k-quant/i-quant **não são** os tiles
fp16/bf16/int8 que o WMMA quer. Há duas rotas e uma é inviável:

- **(a) MMQ — manter quantizado, desquantizar dentro do kernel para int8 e alimentar WMMA.**
  É o que o llama.cpp faz: `mmq.cuh:538-844` escolhe o corpo MMA por tipo, com layout de
  dados MMA (`use_mma_data_layout()`, `mmq.cuh:188-202`), staging dos pesos em LDS e
  exigência de **≥48 KB de shared por bloco** (`mmq.cu:310-317`, sem a qual cai para BLAS
  *em silêncio*). Custo para nós: uma **família de kernels nova para os 14 tipos**, com
  relayout dos pesos em tiles de 16×16 na LDS, a ativação quantizada num layout q8_1 próprio
  do MMA, e um caminho de dispatch separado para decode vs prefill. É trabalho de milestão
  (`rocm-estudo.md` §C.6).
- **(b) Desquantizar os pesos para f16 no load.** **Inviável em VRAM**: `iq3_s` tem
  3,4375 bpw; f16 tem 16 bpw, ou seja **4,65×** os bytes. O tronco de 11,1 GiB viraria
  ~52 GiB. Essa rota está morta nesta placa — e é o ponto que importa: num motor GEMV de
  16 GB, ler o peso **no formato quantizado** é o que faz o modelo caber; converter para um
  formato amigável a matmul não é grátis, é *o* custo.

**Custo de (a), quantificado:** só prefill. Medido hoje: **70,2 tok/s** de prefill em batch
(`docs/medicoes-m8.md:42-50`) contra os **440 tok/s** da referência (llama.cpp Vulkan em batch,
mesmo lugar, `medicoes-m8.md:48`), e a conclusão do próprio M8 é que "a distância que resta é o
laço interno do `vec_dot`". O M6 mediu o matvec batch em 8,97 ms/token de pesos ⇒ ~63-70 tok/s
de teto sem MMQ; o llama.cpp chega a 440 com WMMA+MMQ. **É o maior ganho absoluto disponível
no repo, e o único caminho para ele.**

### 3.6 O Vulkan do llama.cpp usa cooperative matrix para esses mesmos tensores?

**Sim, e nas duas gerações.** Confirmado em `/home/marcelo/Projetos/llama.cpp`:

- `VK_KHR_cooperative_matrix` (coopmat1) e `VK_NV_cooperative_matrix2` (coopmat2) são
  detectados e habilitados em `ggml-vulkan.cpp:6582-6597`, `:7054-7057`, `:7178-7281`
  (o de int8 é classificado em `:7238-7249` → `coopmat_int_support`).
- Escolha por dispositivo: **coopmat2 > coopmat1 > escalar**, cadeia mutuamente exclusiva em
  `ggml-vulkan.cpp:4925-5173`.
- **Os tipos de peso são exatamente os nossos**: `non_lut_quant_types` +
  `FOR_EACH_LUT_TYPE_NONFP4` em `ggml-vulkan.cpp:4834-4854` cobrem `Q2_K…Q6_K`, `IQ1_S`,
  `IQ2_XXS/XS/S`, `IQ3_XXS/S`, `IQ4_XS`, `IQ4_NL`.
- A desquantização acontece **dentro do load do coopmat**: coopmat2 via
  `coopMatLoadTensorNV(..., sliceTensorLayoutNV(...) DECODEFUNCA)` com `switch (MmTypeA)` em
  `mul_mm_cm2.comp:111-160,497-503`; coopmat1 desquantiza para `FLOAT_TYPE` na LDS e depois
  `coopMatLoad`/`coopMatMulAdd` (`mul_mm.comp:340-388`).

Ou seja: **a referência que perseguimos (440 tok/s de prefill) usa matrix cores nos mesmos
tensores quantizados que nós mantemos em dp4a.** Isso não é um detalhe de implementação do
llama.cpp — é a explicação estrutural da distância de 6× no prefill.

**Nota sobre fallback silencioso na referência** (relevante para calibrar a preocupação): o
llama.cpp **tem** fallbacks silenciosos nesse caminho — o desligamento do coopmat é
`GGML_LOG_DEBUG("WARNING: No suitable matrix core mode found")` (`ggml-vulkan.cpp:7272-7276`),
invisível por padrão; o `GGML_VK_DISABLE_COOPMAT*` não loga nada; e no HIP o desvio de MMQ
para hipBLAS (`mmq.cu:310-317`, `:337-383`) só existe como comentário. A classe de falha que o
usuário teme é **real no ecossistema** — e é justamente por isso que o nosso motor, que
**recusa** em vez de degradar (`SPEC.md:21`, e `kv.h:74` como a única brecha latente), está em
posição melhor para confiar nos próprios números.

---

## 4. Estado da fusão

### 4.1 O que já está fundido

| item | estado | linha |
|---|---|---|
| **softmax online na atenção** | **sim** — max/soma correntes, sem array de scores de O(t) | `attn.cuh:113-150` (e a justificativa em `:71-80`); versão com split em `:310-346` + `attn_merge_kernel` `:401-428` |
| **reuso da quantização da ativação** (`proj_qq`) | **sim (M8)** — eliminou 192 das 305 quantizações redundantes | `graph.cuh:586-602`; medido `docs/medicoes-m8.md:51-66` (+1,3%, bit-exato) |
| **quantização em batch** | **sim** — um warp por (linha, bloco de 32), mesma aritmética da versão por linha | `matvec.cuh:63-87`, `graph.cuh:819-829` |
| **norms e elementwise em N linhas** | **sim** — `rms_norm_launch` já recebe `nrows` | `graph.cuh:855-865` |
| **upload de posições em 1 cópia** (em batch) | **sim** | `docs/medicoes-m8.md:19-21` |
| **gate da atenção (sigmoid×saída)** | **não** — 2 lançamentos separados | `graph.cuh:680-682` |
| **silu+mul (SwiGLU)** | **não** — `unary_launch(Silu)` e depois `mul_launch` | FFN `graph.cuh:797-801`; GDN `graph.cuh:783-784` |
| **RMSNorm + quantize** | **não** — 2 lançamentos, e a ativação normalizada sai para a VRAM e volta | `graph.cuh:731-732` (norm) e depois `proj` (`:735`) |
| **residual add** | **não** — `add_launch` próprio | `graph.cuh:611-621` |
| **cadeia escalar do GDN** | **não** — 4 lançamentos de 48 elementos | `graph.cuh:745-752` |
| **`kv_write`** | **não** — 8 lançamentos de 128 threads por camada | `graph.cuh:706-711` |
| **argmax/sampler no device** | **não** — 993 KB de logits copiados + sync por token | `graph.cuh:1211`; `rocm-estudo.md:118-126` |

### 4.2 O que ainda custa uma ida-e-volta extra à VRAM por token

Os números de custo são do `docs/rocm-estudo.md` §A (medidos com `bench --layers`, eventos HIP
e replay), exceto onde marcado. **Contagem de lançamentos: 1876 por token** [minha conta, a
partir de `graph.cuh`: 48 camadas GDN × 20 + 64 FFN × 8 + 16 de atenção × 25 + 4 do final;
o estudo estima "~2200 antes do M8 → ~1800 depois", `medicoes-m8.md:64`, então bate].

| item | lançamentos/token | custo/token | fonte |
|---|---|---|---|
| cadeia escalar do GDN (`sigmoid`/`add`/`softplus`/`mul` × 48 camadas) | 192 | **~1,8 ms (5%)** [estimativa] | `rocm-estudo.md:142-150` |
| `rms_norm` em 1 CTA por linha | 129 | **~0,6 ms (1,7%)** | `rocm-estudo.md:167-175` |
| `quantize_q8_1` (restantes, já deduplicadas) | 113 | parte do teto de 1,5 ms | `medicoes-m5.md:52` |
| `kv_write` | 128 | **~0,5 ms (1,4%)** | `rocm-estudo.md:152-157` |
| `hipMemcpy` síncrono da posição | 16 | **~0,05 ms (0,15%)** | `rocm-estudo.md:159-165` |
| cópia de logits + sampler no host | 1 | **~0,5 ms (1,4%)** | `rocm-estudo.md:118-126` |
| silu+mul não fundidos | 112 | dentro dos "pequenos" | `graph.cuh:797-801` |
| gate da atenção não fundido | 32 | idem | `graph.cuh:680-682` |
| **teto dos pequenos e dependentes** | ~1250 | **~7 ms (20%)** — **teto superior, não medição** | `rocm-estudo.md:288-297` |
| **medido de fato (grafo HIP só na sequência de matvec)** | 497 | **1,38 ms (1,06×)** | `rocm-estudo.md:278-282` |

**A distinção que importa:** o custo dos kernels pequenos é o **gap de despacho** (3,5 µs
medidos com kernel vazio enfileirado), não o trabalho deles. O teto de 7 ms é uma
*extrapolação*; a única coisa medida é 1,38 ms, e só sobre os 497 matvecs grandes (que
sobrepõem o despacho com a execução). **A medição que falta é o grafo HIP sobre a sequência
inteira** — é ela que decide entre "fundir kernels" (grande) e "capturar um grafo por token"
(médio). Está registrado como o teste que falta em `rocm-estudo.md:293-297`.

Duas coisas que o estudo §A **não** contou e eu acrescento:

- **A state do GDN: 302 MB/token (≈0,9 ms, 2,5%)** — 48 camadas × 3,1 MB × (leitura+escrita)
  [minha conta]. É tráfego real que não aparece em nenhuma tabela.
- **A KV redundante 6×: 3,6 GB/token a 64K** (§2.1) [minha conta] — o maior item de tráfego
  evitável do repo, e ele *cresce* com o contexto enquanto os pesos são fixos.

---

## 5. Oportunidades ranqueadas por (impacto × confiança) / esforço

As 5 primeiras são as que eu defenderia; as demais ficam registradas para não se perderem.

| # | oportunidade | impacto estimado | confiança | esforço | arquivo que muda | gate que prova |
|---|---|---|---|---|---|---|
| **1** | **WMMA int8 + MMQ no prefill** (rota (a) de §3.5: manter quantizado, desquantizar para int8 dentro do kernel, staging em LDS, layout MMA por tipo) | **70 → 300-440 tok/s de prefill** (o maior ganho absoluto do repo) | **alta** — o llama.cpp faz exatamente isso com os mesmos 14 tipos (Vulkan coopmat E HIP MMQ), e o M8 já isolou o `vec_dot` como o limitador | **milestão** (família de kernels nova para 14 tipos + dispatch) | `matvec.cuh` (+ um `mmq.cuh` novo), `graph.cuh` (dispatch batch) | `check-batch-gpu` (bit-exatidão **não** se aplica: muda a ordem de soma → tolerância) + `compare_ppl.sh` + `bench --prefill` |
| **2** | **Reduzir ALU de índice/seletor nos `vec_dot` dos 3 i-quants dominantes** (`iq3_s` 33,4% + `iq4_xs` 23,3% + `iq3_xxs` 16,8% = **73% do tempo de matvec**) — o `rocm-estudo.md:229-237` mostra que ~57 das 171 instruções do `iq3_s` são aritmética de índice/seletor e propõe um LUT de seletor de sinais (16 entradas) trocando ~5 ALU por 1 load | matvec é 26,4-27,7 ms; −10% de instruções em 73% dele ≈ **−2 ms ⇒ +5-6% de decode** | média (a contagem de ISA é medida; o ganho não foi implementado) | médio-grande | `vecdotq.cuh` | **bit-exatidão**: `check-matvec-gpu --bench-ab` + oráculo por tipo |
| **3** | **Compartilhar K/V entre as 6 cabeças GQA**, agora que a grade de splits é larga (CTA por (cabeça KV, split) processando as 6 consultas, reusando a linha K/V lida uma vez) | **até +18% a 64K** (`rocm-estudo.md:318`); −3,6 GB/token de tráfego; pouco a 4K | média-baixa (o protótipo do M7 foi 8-12× mais lento; a premissa "grade larga + reuso em registrador" não foi testada) | grande | `attn.cuh` (+ `graph.cuh` para a grade) | `scripts/check_attn_split.sh` + `check-kvctx-gpu` + `compare_ppl.sh`; `bench --start-pos` para medir |
| **4** | **Fundir as cadeias pequenas** (as 4 ops escalares do GDN, `rms_norm`+`quantize`, `kv_write` numa chamada) **e/ou** grafo HIP da sequência inteira | **teto 7 ms (20%)** [estimativa]; medido de fato só **1,38 ms (4%)** | **baixa** — é exatamente a medição que falta (`rocm-estudo.md:293-297`). Fazer o grafo da sequência inteira **primeiro**: é barato e transforma o teto em número | grafo: médio; fusões: grande | `graph.cuh`, `kv.h`, `nn.cuh` | fusão preserva ordem ⇒ **bit-exatidão**: `check-graph-gpu` (oráculo por nó) + `check_golden_run.sh` |
| **5** | **`q3_K` por linearidade do `dp4a`** (`Σ(vil−vih)·u` exato em int32; a saturação de `__vsubss4` é código morto aqui) | `q3_k` = **3,61%** do tráfego (401 MB/token) a 307 GB/s = **1,31 ms**; 208 instruções é o kernel mais caro dos 14 [ISA minha]; esperado 1,31 → ~0,7 ms ⇒ **+1,4-1,7% de decode** | **alta** (o M2 mediu **2,0×** em `iq3_s` com a mesma troca, `rocm-estudo.md:245-249`) | pequeno-médio | `vecdotq.cuh` | **bit-exatidão**: `check-matvec-gpu` + oráculo por tipo |
| 6 | Atenção com f16 empacotado (`v_pk_fma_f16` / `v_dot2_f32_f16`, 1,5× o rate do FMA fp32) para o KV f16 | desconhecido; a atenção é *issue-bound na desquantização* (`medicoes-m5.md:130-131`), não no FMA — pode não pagar | baixa [hipótese] | médio | `attn.cuh`, `kv.h` | `bench-attn-gpu` + `scripts/check_attn_split.sh` |
| 7 | `delta_rule`: dividir a dimensão `j` para encher as 64 CUs (hoje 48 CTAs) e vetorizar as linhas com `float4` | ~0,9 ms de tráfego (2,5%) + as 16 CUs ociosas; ganho real desconhecido | média-baixa | pequeno-médio | `gdn.cuh` | **bit-exatidão** (mudar o número de threads não muda a aritmética) + `check-graph-gpu` |
| 8 | Argmax no device (caminho greedy) | ~0,3-0,5 ms (1%) | alta | pequeno-médio | `graph.cuh`, `main.hip` | `check_golden_run.sh` (ids idênticos) |
| 9 | `bench` descontar `token_embd` + MTP (e as shares da §3 do inventário) | 0 em tok/s; corrige **8%** do número de banda reportado | alta (provado por aritmética em `docs/quants-inventario.md` §3.3) | trivial | `main.hip`, `docs/medicoes-m6.md` | `info`/`bench` antes-e-depois |
| 10 | Fechar a brecha silenciosa de `kv_row_bytes` (`kv.h:74` devolve `0`) e alinhar `SPEC.md:21` com a whitelist real (F16/BF16) | 0 em tok/s; remove uma classe de falha silenciosa | alta | trivial | `kv.h`, `SPEC.md` | `check-kvctx-gpu` |

### Por onde eu começaria

**item 4 na forma de medição → item 5 → item 2.** O item 1 (WMMA/MMQ) é o de maior impacto
mas é um milestão e depende de outra frente; o item 4 na forma barata (*medir* o grafo HIP da
sequência inteira, não fundir nada) custa pouco e decide o orçamento dos itens seguintes; o
item 5 é o melhor (impacto × confiança)/esforço do repo; o item 2 é o único caminho para mexer
nos 73% de tempo que estão nos três i-quants dominantes.
