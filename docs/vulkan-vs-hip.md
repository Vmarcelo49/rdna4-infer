# Vulkan (llama.cpp) vs HIP (este motor) — comparação código-a-código

Por que o backend **Vulkan** do llama.cpp é competitivo (e às vezes mais rápido) numa GPU
AMD, e o que este motor deve roubar dele.

- **llama.cpp**: checkout `/home/marcelo/Projetos/llama.cpp`, revisão `df03399b8`
  (build `b10902`), backend Vulkan/RADV. Shaders em
  `ggml/src/ggml-vulkan/vulkan-shaders/`, host em `ggml/src/ggml-vulkan/ggml-vulkan.cpp`;
  os `*.spv` gerados que este relatório usa como evidência estão em
  `build/ggml/src/ggml-vulkan/vulkan-shaders.spv/` (1 519 arquivos).
- **este motor**: branch `feat/vulkan-vs-hip` (base `aa15eea`), `include/rdna4/*`.

## 0. Método, e o que é inferência

Isto é **leitura de código**. Nenhuma GPU foi usada, nada foi compilado nem executado — a
fila da GPU pertence a outro agente (`docs/gpu-queue.md`). Toda afirmação sobre o código
tem `arquivo:linha`. Onde eu **inferi** (tipicamente: o que um driver faz, ou quanto custa
um despacho do qual não medi o tempo), está marcado com **[inferência]**.

Os números deste motor e do baseline Vulkan são os **medidos** neste repo/máquina
(`docs/medicoes-m5.md`, `m6`, `m7`, `m8`, `docs/rocm-estudo.md`,
`docs/baseline-vulkan-iq3s.md`); nenhum número novo foi inventado. Onde eu faço aritmética
em cima de medições, digo qual é a conta.

---

## 1. Por que o Vulkan é competitivo em AMD

### 1.1 O caminho de decode não é int8 — é FMA em ponto flutuante

O fato mais importante, e o que contraria a intuição de quem conhece o backend HIP: **para
os tipos que dominam este modelo, o Vulkan não usa dot product inteiro nenhum.**

A lista de tipos que ganham shader `*_q8_1` (o caminho int8, MMQ) é fixa em
`vulkan-shaders-gen.cpp:626-630`:

```cpp
if (!f16acc && !coopmat && !coopmat2 && !dot2 &&
    (is_legacy_quant(tname) || is_k_quant(tname) || tname == "mxfp4")) {
    string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", ...);
}
```

com `is_legacy_quant` = `q2_0, q4_0, q4_1, q5_0, q5_1, q8_0`
(`vulkan-shaders-gen.cpp:237-239`), `is_k_quant` = qualquer `*_k` (`:241-243`) e
`is_lut_quant` = qualquer `iq*` + `mxfp4` + `nvfp4` (`:249-251`). Os `iq1_s`/`iq1_m`
entram no caminho int8 pela porta de trás da GEMV (`:810`), **os `iq2_*`/`iq3_*`/`iq4_xs`
não entram em lugar nenhum**. O artefato compilado confirma: existem
`matmul_q4_k_q8_1.spv`, `matmul_q3_k_q8_1.spv`, `mul_mat_vec_iq1_s_q8_1_f32*.spv` — e
**nenhum** `matmul_iq3_s_q8_1.spv` nem `mul_mat_vec_iq3_s_q8_1_*.spv`.

A composição de bytes deste arquivo (`docs/medicoes-m6.md`, tabela de escala, coluna
"share of model") é **77,7 % em tipos LUT** (`iq3_s` 30,9 %, `iq4_xs` 21,6 %, `iq3_xxs`
15,5 %, `iq2_s` 4,8 %, `iq2_xs` 2,4 %, `iq2_xxs` 2,2 %, `iq1_s` 0,3 %). Ou seja: **~78 %
dos bytes que este modelo lê por token passam pelo caminho sem int8** no Vulkan.

O que esse caminho faz, então? Na GEMV, `mul_mat_vec_iq3_s.comp:15-53` dequantiza o peso
(grid de 4 valores por lookup) e faz uma cadeia de **`fma` em `FLOAT_TYPE`** (que é
`float` neste caminho, `vulkan-shaders-gen.cpp:774`):

```glsl
// mul_mat_vec_iq3_s.comp:30-48
const vec4 grid0 = vec4(unpack8(iq3s_grid[qs.x | ((qh << (8 - 2*l)) & 0x100)]));
...
const FLOAT_TYPE sum = fma(FLOAT_TYPE(b0.x), FLOAT_TYPE((sign & 1) != 0 ? -grid0.x : grid0.x),
                        fma(...  ...  FLOAT_TYPE(0.0)))))))));
temp[j][n] = fma(dscale, sum, temp[j][n]);
```

Com `FLOAT_TYPE == float` as conversões são no-op: é **8 FMA fp32 + seleção de sinal por
elemento**. Não há `dotPacked4x8EXT`, não há quantização de ativação (a ativação entra em
**fp32**, `B_TYPE=float` na criação do pipeline, `ggml-vulkan.cpp:5523-5535`), não há
`dp4a`.

Contraste direto com o nosso kernel: `vecdotq.cuh:84-88` usa
`__builtin_amdgcn_sudot4` (= `v_dot4_i32_iu8`, full-rate em gfx1201,
`docs/rocm-estudo.md` B.1) e a ativação vem quantizada em `block_q8_1`
(`matvec.cuh:28-54`). Em decode, o nosso lado usa a instrução **certa**; o Vulkan usa FMA.

E mesmo assim ele anda mais rápido (39,7 vs ~29 tok/s). Isso é o que a seção 2 quantifica.

### 1.2 Organização do matmul: três famílias de shader, escolhidas por (tipo, n)

`ggml_vk_mul_mat` (`ggml-vulkan.cpp:10367-10438`) decide assim, em ordem:

| condição | caminho | shader |
|---|---|---|
| `dst->ne[1] == 1` (1 token) | GEMV (`:10433-10435`) | `mul_mat_vec_<tipo>.comp` |
| `2 ≤ n ≤ mul_mat_vec_max_cols` (=8, `:404`) | GEMV com `NUM_COLS` colunas | idem |
| resto (prefill) | GEMM (`:10436-10437`) | `mul_mmq.comp` **ou** `mul_mm.comp` |

No GEMM, `ggml_vk_mul_mat_q_f16` (`:9430`) tenta o MMQ primeiro (`:9490-9499`):

```cpp
bool quantize_y = ctx->device->integer_dot_product && src1->type == GGML_TYPE_F32 && ...;
const std::vector<vk_matmul_pipeline_pair>* mmp_map = quantize_y
    ? ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, GGML_TYPE_Q8_1, ...) : nullptr;
if (mmp_map == nullptr) {  // Fall back to f16 dequant mul mat
    mmp_map = ggml_vk_get_mul_mat_mat_pipeline_map(ctx, src0->type, y_non_contig ? f16_type : src1->type, ...);
    quantize_y = false;
}
```

Ou seja: se existe pipeline `(tipo, Q8_1)` → ativações quantizadas a `q8_1` + MMQ int8; se
não existe — o caso de `iq3_s`, `iq4_xs`, `iq3_xxs`…, porque nenhum `_q8_1` foi registrado
para os tipos LUT (`:5225-5229`), e na GEMV pelo mesmo motivo via
`ggml_vk_get_dequantize_mul_mat_vec` que devolve `nullptr` (`:8034-8053`) — cai no GEMM
fp16: pesos dequantizados para fp16 **dentro do shader** (`mul_mm_funcs.glsl:20-30` grava
em `buf_a` de LDS) e ativação fp16.

Quem registra o quê (`ggml-vulkan.cpp:5221-5229`):

```cpp
for (const auto type : non_lut_quant_types) {         // q*_0, q*_K, tq*_0
    sg_create_quant({type, GGML_TYPE_F32, ...}, tc_mmq, "matmul_quant_f32", ...);
}
#define X_SG(TYPE, tstr) sg_create({TYPE, GGML_TYPE_F32, ...}, tc_mmq, "matmul_" #tstr "_f32", ...);
FOR_EACH_LUT_TYPE(X_SG)                                // iq1_s … iq4_xs, iq4_nl, mxfp4, nvfp4
```

Duas arquiteturas de shader convivendo: os **LUT/IQ têm um shader por tipo**
(`matmul_iq3_s_f16`, com `DATA_A_IQ3_S` compilado dentro), e os **não-LUT compartilham um
SPIR-V único** com o tipo como *specialization constant* (`matmul_quant_f16/f32`,
`MmTypeA`, `constant_id = 12`; `ggml-vulkan.cpp:4823-4832`), cujo `switch` gigante de
dequant está em `mul_mm_funcs.glsl:342-630`. Nos dois casos o *driver* elimina os caminhos
não usados depois da especialização.

### 1.3 Tiles, workgroup, shared memory, coopmat

O `mul_mmq.comp` (int8) é o único com geometria explícita e ele é **explícito no arquivo**:
`BM=64, BN=64, BK=32, WM=32, WN=32, WMITER=2, TM=4, TN=2, WARP=32, BLOCK_SIZE=64`
(`mul_mmq.comp:72-84`, com `BK_STEP=4` em `:91-93`). O layout de staging é

```glsl
shared block_a_cache buf_a[BM * BK_STEP];   // mul_mmq.comp:97
shared block_b_cache buf_b[BN * BK_STEP];   // :98
block_a_cache cache_a[WMITER * TM];          // :100  (registradores)
```

e o laço principal (`:219-281`) faz: estágio A e B para LDS → barreira → LDS para
registradores → `sums[...] += mmq_dot_product(...)` (`:273`) → barreira. O
`block_a_to_shmem` de cada tipo **reorganiza** os quants no formato que o dp4a quer
(ex.: `Q4_K` junta dois nibbles em 0x0F0F0F0F, `mul_mmq_funcs.glsl:347-358`; `Q3_K`
"add the 3rd bit instead of subtracting it" para poder empacotar, `:283-294` e `:319`), e
`block_b_to_shmem` (`:457-481`) converte a ativação `q8_1` para o layout `int32`.

Para os GEMM "normais" (`mul_mm.comp`) as constantes de especialização têm default no
shader (`:161-170`) mas os valores reais vêm do host: o vetor *warptile* é
`[BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP]`
(`ggml-vulkan.cpp:4406-4410`) e tem **tuning por arquitetura** — em RADV/AMD com coopmat
(`:4524-4529`):

```cpp
// ggml-vulkan.cpp:4524-4528 (abreviado: o driver_id é vk::DriverId::eAmdProprietary)
} else if (device->vendor_id == VK_VENDOR_ID_AMD && device->coopmat_support && device->driver_id != eAmdProprietary) {
    l_warptile     = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
    l_warptile_mmq = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
```

(Os 11 números são, na ordem, `BLOCK_SIZE, BM, BN, BK, WM, WN, WMITER, TM, TN, TK, WARP`
— a legenda está no próprio código, `:4406-4410`. `wg_denoms` = `{128,128,1}` / `{64,64,1}`
/ `{32,32,1}` para os tiles l/m/s, `:4531-4533`.)

O tamanho do tile **não é livre**: `ggml_vk_matmul_shmem_support` (`:4150-4202`) soma o
staging (`(BM+BN)*(BK+pad)*type_size`), o palco do coopmat, *os row-ids do MoE* e — o
ponto interessante para nós — a **LUT do tipo**:

```cpp
case GGML_TYPE_IQ3_S: lut_size = 4*512;  break;   // ggml-vulkan.cpp:4172-4173 → 2 KB
...
const uint32_t total_size = load_bufs + mmid_row_ids + coopmat_stage + lut_size + ballots_sh;
```

e desliga tile médio/grande por tipo quando não cabe (`:4552-4565`). Isto é: **em Vulkan a
LUT do IQ3_S mora em LDS e é contabilizada no orçamento de shared memory do GEMM**.

### 1.4 Coopmat / `VK_KHR_cooperative_matrix`: quando é escolhido

- Exposição: `VK_KHR_cooperative_matrix` → `device->coopmat_support` (`:6583-6589`),
  `VK_NV_cooperative_matrix2` → `coopmat2_support` (`:6591-6594`). Cada um tem um
  `GGML_VK_DISABLE_COOPMAT`/`GGML_VK_DISABLE_COOPMAT2` para A/B.
- Gate por vendor: `ggml_vk_khr_cooperative_matrix_support` (`:20087-20102`) **rebaixa
  coopmat1 a RDNA3** quando o driver é AMD proprietary/AMDVLK (`:20093-20097`); para
  qualquer outro `driverID` (RADV/Mesa **não** é `eAmdProprietary` nem `eAmdOpenSource`)
  devolve `true` (`:20098`). Então, em RADV, coopmat1 fica habilitado se a extensão e a
  feature existirem. **[inferência]** a partir daí; o estado real nesta máquina sai na
  linha de log `ggml_vulkan: ... | matrix cores: KHR_coopmat|NV_coopmat2|none` (`:7662-7679`).
- coopmat2 é bem mais restritivo: exige `workgroupInvocations` 128 **e** 256 com
  `MGranularity/N/KGranularity ≤ 32` para fp16 e fp32 e
  `cooperativeMatrixFlexibleDimensionsMaxDimension ≥ 512` (`:7140-7156`).
- Efeito no GEMM: o coopmat só aparece no `mul_mm.comp` (`:20-23`, `:339-346`) via
  `coopMatLoad`/`coopMatMulAdd` (`:380-391`) — e **o MMQ int8 (q8_1) fica de fora quando
  coopmat está ativo** (`vulkan-shaders-gen.cpp:627`: `!coopmat && !coopmat2 && !dot2`).
  Ou seja, há um *trade-off* embutido: com coopmat ligado você perde o caminho int8 do
  GEMM e ganha o `coopMatMulAdd`.
- FA: `get_fa_tuning_params` (`:4061-4100`) prefere coopmat2 > coopmat1 > escalar, e
  `coopmat1_fa_support = coopmat_support && subgroup_require_full_support` (`:7056`).

### 1.5 Flash attention: o que o shader realmente faz

Parâmetros de tuning medidos/derivados por (head dim, n_rows, n_kv)
(`ggml-vulkan.cpp:3904-3922`, `:3927-4000`). Para AMD não-GCN (**o nosso caso**):

- decode (`n_rows == 1`): `subgroup_size = 32`, `row_split = 1`, `workgroup_size = 32*4 = 128`
  (`:3936-3953`), `block_rows = 1`, `block_cols = 64` (`:3959-3961`),
  `d_split = min(subgroup_size, 8, D_lsb/4)` (`:3973-3975`).
  Isto é: **1 linha de query por workgroup, 64 chaves por iteração, a dimensão do head
  fatiada entre subgrupos.**
- prefill (`n_rows ≥ 4`): `block_rows` 4-8, `block_cols` 32-64 (`:3962-3971`).
- `SHMEM_STAGING` só é ligado para NVIDIA (`:3977`) — no AMD o K/V é lido direto da global
  (ou dequantizado inline, `USE_DECODE_K` + `dequantize4`, `flash_attn.comp:315-321`).
- `limit_occupancy_shmem` (`:3986-3997`) é um array de shared memory **de propósito inútil**
  (`flash_attn.comp:75` e `:97-106` escrevem nele para o compilador não removê-lo) só para
  *baixar a ocupação* e mirar 4 subgrupos/SIMD. É um hack de driver, não um algoritmo.

Como as chaves são divididas (decode, nosso modelo: `N=1`, `neq2=24`, `nek2=4`,
`qk_ratio=6`):

```cpp
// ggml-vulkan.cpp:11251-11259
if (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa && ...) {
    gqa_ratio = qk_ratio;   // = 6
    N = gqa_ratio;
    workgroups_y /= gqa_ratio;   // 24/6 = 4
}
// :11347-11353
if (gqa_ratio > 1 && workgroups_x <= Br) {
    split_k = shader_core_count * 2 / (workgroups_x * workgroups_y * workgroups_z);
}
```

Com `Br = 1` e `workgroups_x = 1`: **`split_k = 64*2/(1*4*1) = 32`**, depois
`split_kv = ROUNDUP_POW2(max(1, KV/split_k), block_cols=64)` (`:11356-11361`), um
`flash_attn_split_k_reduce` por fim (`:11479-11506`). No espaço de 64K isso dá 32 splits
**por (head KV, grupo GQA)**, e a grade x empacota `(batch, split_k)` (`flash_attn_base.glsl:153-160`).

O detalhe que muda o jogo está em `flash_attn_base.glsl:184-186`:

```glsl
// When not using grouped query attention, all rows share the same iq2, equal to gl_WorkGroupID.y.
// When using grouped query attention, each workgroup does gqa_ratio consecutive values of iq2.
iq2 = gl_WorkGroupID.y * p.gqa_ratio;
```

com `Br = 1` e `row_split = 1`, `rows_per_thread = Br/row_split`... e `N = gqa_ratio = 6`
(`flash_attn.comp:31`), `i = 0`, `gqa_iq1 = workgroup.x / k_num`: **um workgroup calcula os
6 heads de query do grupo GQA de uma vez**, e o laço de score carrega o K uma única vez
por (coluna, bloco de dim) e reaproveita para as 6 linhas
(`flash_attn.comp:334-349`: `for c { for d { K_Tf = ...; for r { dot } } }`). É
exatamente a redundância 6× que o M7 mediu no nosso kernel — resolvida do lado de lá.

Variantes de FA que existem no build desta máquina: escalar fp16/fp32 (`flash_attn.comp`),
`_dot2` (fp16 empacotado, `:740-743`), `_int8`/MMQ (`:745-748`) e `cm1`/`cm2`
(coopmat). O caminho int8 **não é usado com KV f16**: `ggml_vk_fa_scalar_uses_mmq`
(`:4352-4365`) exige `integer_dot_product && subgroup_clustered` **e**
K ∈ {q4_0,q4_1,q5_0,q5_1,q8_0}; com KV f16 sobra o caminho escalar com `dot_product` fp16
(`dot_product_funcs.glsl:8-25`).

### 1.6 Dequantização: inline no matmul, e um passe separado só para o KV cache quantizado

- **Pesos**: *sempre* inline. Se existe shader para (tipo, ativação) o peso é dequantizado
  dentro do shader (`mul_mm_funcs.glsl:20-630` para o GEMM,
  `mul_mat_vec_iq3_s.comp:15-53` para a GEMV). O passe separado
  (`qx_needs_dequant`, `ggml-vulkan.cpp:9501-9507` + dispatch em `:9641-9645`) só existe
  para tipos **sem** shader de matmul — nenhum dos nossos 14.
- **KV cache**: aqui sim há um passe separado. Se K **e** V estão quantizados, `neq1 ≥ 64`,
  o cache é denso e cabe, o Vulkan dequantiza o cache inteiro para um *scratch* f16
  **transposto** e a FA lê sem stride (`:11223-11244`, dispatch em `:11416-11442`):

```cpp
const bool use_dequant_kv = k_quant && v_quant && neq1 >= 64 && is_dense_kv_cache(k) && ...;
```

  Hoje só `q8_0` tem o shader `dequant_q8_0_transpose` (`vulkan-shaders-gen.cpp:825-828`).
  Com KV em f16 (o default) nada disso roda e a leitura é direta.

### 1.7 O que isso explica do desempenho

Nada aqui é magia de driver: o ganho vem de (a) **um kernel GEMM/GEMV com staging em LDS e
reuso de ativação entre linhas**, (b) **menos trabalho por byte** no caso dos k-quants
(int8) e **reuso da linha de K/V entre os 6 heads** na atenção, (c) **fusão de operações**
(§2) e (d) uma **contagem de despachos menor** (§2). Note o que ele *não* faz: não tem
`dp4a` no caminho dos IQ, não tem prefetch (o `mmvq_prefetch_l2` é no-op em RDNA4 no
backend HIP também, `docs/rocm-estudo.md` B.4), não usa wave64.

---

## 2. Despacho, sincronização e o orçamento de um token

### 2.1 O que o Vulkan faz por operação

Um `vkCmdDispatch` custa (host): atualizar descriptor set, push constants, bind pipeline,
bind descriptor set, dispatch (`ggml-vulkan.cpp:8451-8464`). Nada disso é sincronização.
O que torna isso barato:

1. **Um command buffer para o grafo inteiro, reusado.** `ggml_vk_get_or_create_cmd_buffer`
   recicla buffers de um pool (`:8296-8305`) e `ggml_vk_begin_submission` reusa o
   `vk::CommandBuffer` (`:8307-8317`). O grafo de um token é gravado em um (ou poucos)
   command buffers; a submissão só é cortada a cada ~200 GFLOP ou 100 nós
   (`:18065-18083`, `:18120-18133`).
2. **Índices de descriptor pré-alocados**: `ctx->descriptor_sets[ctx->descriptor_set_idx++]`
   (`:8451`), com `ggml_pipeline_request_descriptor_sets` reservando na fase de *grafo*
   (`:9580-9592`).
3. **Push constants para os parâmetros** (`:8455`) — o equivalente ao argumento de kernel
   do HIP, sem buffer nem cópia.
4. **Barreira só onde há aliasing**, gravada *dentro* do command buffer
   (`ggml_vk_sync_buffers`, usada por ex. em `:9644`, `:11501`), nunca um sync de host.
5. **Um único sync por token**, e pequeno: com amostragem no device (§4 item 3), o que
   volta para o host são 4 bytes; sem ela, a linha de logits.

### 2.2 O que este motor faz por token — contado no código

Contagem feita à mão sobre `graph.cuh` (caminho por token, modelo de 64 camadas / 16 de
atenção / 48 GDN, `n_head_kv = 4`):

| por camada | lançamentos |
|---|---|
| `full_attn` (`graph.cuh:619-691`) + `attn_post_norm` (`:1166`) | 26 |
| `gdn_layer` (`graph.cuh:714-790`) + `attn_post_norm` | 21 |
| `ffn` (`graph.cuh:793-806`) | 8 |
| cabeça/rabo (`:1146`, `:1184`, `:1206-1211`) | 4 |

⇒ **16·26 + 48·21 + 64·8 + 4 = 1 940 lançamentos por token**, dos quais
**497 são `matvec_launch`** (uma por projeção: 4×16 + 5×48 + 3×64 + LM head) e
**257 são `quantize_q8_1`** (uma por projeção que *não* é `proj_qq`; `proj` em
`:557-584` faz quantize+matvec, `proj_qq` em `:586-603` faz só o matvec e os chamadores
`attn_k`/`attn_v` `:630-631`, `attn_gate`/`ssm_beta`/`ssm_alpha` `:738-740`,
`ffn_up` `:800` reusam os blocos `q8_1` da projeção anterior). Os outros 1 186 são
norms (209 `rms_norm` — inclui o `output_norm` — + 96 `l2_norm`), cadeia escalar do GDN
(192), `conv1d` + `delta_rule` (96), `kv_write` (**128** = 16 camadas × 4 cabeças ×
2 tensores, `:701-709`), `rope` (32), atenção (32 = 16 pares split+merge,
`attn.cuh:440-444`), `deinterleave` (16), gates sigmoid/mul (32), silu/mul (224),
resíduos (128 = 16+48+64), `dequant_row` do embedding (1).
Confere: 497+257+1 186 = 1 940.

Ainda por token:

- **`hipGetLastError()` depois de cada lançamento de `proj`/`proj_qq`** (`:575`, `:598`) e
  em todos os launchers (`nn.cuh:122`, `matvec.cuh:440`…). Medido: 1,04 µs → 1,39 µs por
  lançamento (`docs/medicoes-m5.md` §2) ⇒ **~0,1 ms/token**.
- **1 `hipMemcpy` D2H bloqueante de 993 KB** (`graph.cuh:1211`) — drena o pipeline; é o
  "tail" de 1,92 ms medido em `docs/rocm-estudo.md` §A.1.
- **16 `hipMemcpy` H2D síncronos de 4 bytes** para subir a posição
  (`graph.cuh:651`, um por camada de atenção).
- **Amostragem no host** (`include/rdna4/sampler.h`) sobre os logits copiados.

### 2.3 Quantos despachos cada lado faz por matmul

| | este motor | llama.cpp Vulkan |
|---|---|---|
| matmul **por projeção** | 1 `matvec_launch` + 0/1 `quantize_q8_1` | 1 (`iq*`) ou 2 (`q8_0`/k-quants, com `quantize_q8_1` antes — `:9659-9671`) |
| ativação | `block_q8_1` (int8, 1,0625 B/elem) | **fp32** para `iq*` (4 B/elem), `q8_1` para os demais |
| ativações quantizadas/token | 257 | 0 para 78 % dos bytes |
| matmuls/token | 497 | ~497 (mesmos tensores) |
| total de despachos/token | **1 940 (contado)** | **~1 300-1 400 [estimativa]** |

A estimativa do lado Vulkan: ~18 por camada de atenção (3 matmuls Q/K/V, 2 `rms_norm` de
Q/K, 2 `rope`, 1 cont do gate, 1-2 de FA, 1 matmul de saída, 1 `rms_norm` de post-norm,
4 do FFN com `SILU_MUL` fundido, mais 1-2 de cópia
`ggml_cont_2d` de `qwen35.cpp:293`), ~17 por camada GDN (`qwen35.cpp:335-470`: 2 matmuls
de `qkvz`, 1 de beta, 1 de alpha com `add` fundido, conv, 2 de L2, 1 `gated_delta_net`,
`norm_gated`, 1 matmul de saída) e ~4 por FFN (`qwen35.cpp:470-486` com `LLM_FFN_PAR`,
up/gate/down + `SILU_MUL` fundido). **Não é medição** — é contagem sobre o grafo, e o
número pode variar ±2 por camada conforme quais fusões casam.

### 2.4 A conta que fecha o `bench --prefill`

O gráfico acima junto com a lista de fusões (§2.5) explica por que o prefill está a 70
tok/s contra 440: no nosso lado o custo por byte de peso é o mesmo nos dois regimes
(o `vec_dot` é issue-bound, medido no M6), enquanto do lado Vulkan **o prefill usa outro
kernel** — GEMM tiled com staging em LDS e, para 22 % dos bytes, int8.

### 2.5 As fusões são explícitas e nomeadas (o que mais muda o jogo)

`ggml_backend_vk_graph_compute` casa *padrões de subgrafo* e funde o consumidor no
produtor, com um nome para cada (`ggml-vulkan.cpp:18149-18327`). A lista, filtrada para o
que existe no nosso grafo:

| padrão fundido | linha | onde isso aparece no nosso grafo |
|---|---|---|
| `MUL_MAT_ADD`, `MUL_MAT_ADD_ADD` | `:18155-18165` | resíduo depois de cada projeção (`graph.cuh:689`, `:787`, `:804`) |
| `SIGMOID_MUL`, `SILU_MUL` | `:18228-18237` | `sigmoid(attn_gate)*attn` (`:682-684`) e `silu(ffn_gate)*ffn_up` (`:798-801`) |
| `RMS_NORM_MUL`, `RMS_NORM_MUL_ADD_MUL` | `:18203-18227` | `norm_gated` do GDN (`:776-782`) |
| `SSM_CONV_SILU`, `SSM_CONV_BIAS_SILU` | `:18238-18250` | `conv1d` + `silu` (`:754`, `:781`) |
| `ROPE_VIEW_SET_ROWS` | `:18251-18258` | `rope` + `kv_write` do K (`:657`, `:663`) |
| `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` | `:18182-18193` | `k_norm` + `rope` + escrever no cache (`:645-663`) |

Com `GGML_VK_DISABLE_FUSION=1` (`:7421`) dá para A/B tudo isso.

### 2.6 Orçamento derivado (aritmética sobre medições, não medição nova)

Nosso decode: 29,26 tok/s (`docs/medicoes-m8.md`) = **34,2 ms/token**, com
matvec 26,4-27,7 ms para 11,122 GB (**402-421 GB/s**, `docs/rocm-estudo.md` §A.1),
LM head 1,39 ms (627 GB/s, 98 % do teto, §A.2.5), atenção ~1,2 ms, cópia+sampler ~0,5 ms,
"resto" (norms, GDN, gaps de despacho) 2-3 ms.

Vulkan: 39,73 tok/s (`docs/medicoes-m5.md` §1) = **25,2 ms/token**, e lê **os mesmos**
11,122 GB de pesos. Consequência aritmética: **a banda efetiva do Vulkan tem de ser
≥ 442 GB/s** (11,122 GB / 25,2 ms). Descontando o LM head (0,874 GB a 627 GB/s, o trecho
mais rápido) e ~2,5 ms de atenção+glue, o resto dos 10,25 GB teria de sair a **~480 GB/s** —
contra os nossos 402-421 GB/s. O resto do gap (1940 despachos × 3,5 µs ≈ 6,8 ms do nosso
lado, contra ~1 360 × custo desconhecido do lado deles) também joga a favor dele
**[inferência: não medi o custo por despacho em Vulkan]**.

Duas consequências práticas:

1. **O nosso `vec_dot` não é o único problema.** Mesmo com banda perfeita, os 1 940
   despachos custam ~6,8 ms (teto superior, `docs/rocm-estudo.md` §B.5: 3,5 µs com fila
   cheia) = 20 % do token. Fundir é a alavanca mais barata.
2. **O gap cresce com o contexto, e isso não é matvec.** 4K: 29,3 vs 39,7 = 74 %.
   16K: 24,3 vs ~39 = 62 %. 32K: ~22 vs 37-38 (`docs/baseline-vulkan-iq3s.md`) = **~58 %**.
   64K: 18,9 vs (não medido no baseline). Como o **nosso** fluxo de pesos é constante
   (~27 ms de matvec em qualquer contexto), a diferença que cresce é **a atenção** — e é
   aí que mora o item 4 do ranking (GQA/Vulkan-style).

---

## 3. Tabela técnica código-a-código

Legenda: **N** = nós (este motor) · **V** = Vulkan (llama.cpp).

### 3.1 Peso quantizado — GEMV (decode)

| aspecto | V: llama.cpp/Vulkan | N: este motor | quem é melhor e por quê |
|---|---|---|---|
| corpo do dot | `mul_mat_vec_iq3_s.comp:37-48` — 8 FMA fp32 + select de sinal | `vecdotq.cuh:875-916` — 4 `dp4a` + 2 `v_perm_b32` por 8 valores, linearidade `sumi = 2*sumi_pos − sumi_all` | **N** no papel: instrução int8 full-rate e 171 inst/220 B medidos (`docs/rocm-estudo.md` B.2) vs FMA fp32. **V** na prática — o que sugere que o gargalo do nosso lado não é a instrução e sim outra coisa (LUT fora de LDS, latência, banda) |
| ativação | fp32 direta (`ggml-vulkan.cpp:5523-5535`, `B_TYPE=float`), **sem quantizar** — `quantize_y` fica falso porque não existe pipeline `(IQ3_S, Q8_1)` (`:9822-9844` + `:8034-8053`) | `block_q8_1` por projeção (`matvec.cuh:28-54`, chamado de `graph.cuh:574`) | **V**: economiza 257 despachos e o trabalho de quantização (custo medido: 1,36-1,5 ms/token, teto, `docs/medicoes-m5.md`) — mas o int8 exige a ativação quantizada, então isso não é copiável sem trocar o corpo do dot |
| linhas por workgroup | `NUM_ROWS = rm_iq = 2` para IQ em RDNA4 (`ggml-vulkan.cpp:5481`, criação em `:5542-5544`) | `ROWS` por tipo, 1-8 (`matvec.cuh:445-458`) | empate; o nosso sweep do M2 mediu que a forma ideal depende do **formato da linha** (`PLAN.md` M2 passo 5) |
| redução | `subgroupAdd` + palco em LDS (`mul_mat_vec_base.glsl:94-128`, `:130-229`) | `__shfl_xor` e, se `WPR>1`, LDS (`matvec.cuh:264-278`) | empate |
| LUT do IQ | **`shared uint32_t iq3s_grid[512]`** + cópia no início do kernel (`types.glsl:1576`, `:1679`, `:1748-1756`), 2 KB contabilizados no orçamento de LDS (`ggml-vulkan.cpp:4172`) | `static const __device__ uint32_t iq3s_grid[512]` (`quant_tables.h:524`) — **memória global**, 8 `global_load_b32` por chamada | **V**. É o candidato #2 do ranking (§4) |
| tabela de decisão | `ggml_vk_should_use_mmvq` mede `k` e vendor (`:9703-9780`) | `matvec_default_config` por tipo, medido (`matvec.cuh:479-499`) | empate; a conclusão do M2 (`docs/rocm-estudo.md` D) de que a regra do HIP do llama.cpp não transfere continua válida |

### 3.2 Peso quantizado — GEMM (prefill em batch)

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| família de kernel | `mul_mmq.comp` (int8, 64×64×32, `BK_STEP=4`, staging A+B em LDS) para os 22 % de bytes que têm caminho `q8_1`; `matmul_<tipo>_f16` para os 78 % em `iq*` (mesmo `mul_mm.comp`, pesos dequantizados para fp16 em LDS, **sem int8**) | `matvec_kernel_batch<T,ROWS,WPR,ILP,N>` (`matvec.cuh:296-386`) — mesma GEMV com N acumuladores | **V**, por muito: 440 vs 70 tok/s |
| reuso do peso | tile BM×BN em LDS, A em registrador `WMITER*TM` (`mul_mmq.comp:100`, `:254-277`) | uma passada por linha, peso lido uma vez e usado N vezes (N ≤ 16, `matvec.cuh:333-352`) | **V**: o miolo do problema não é ler o peso N vezes, é o custo por byte |
| split-K | quando `m_tiles*n_tiles` é pequeno e `k ≥ 2048` (`:9023-9063`), com kernel de redução (`:9091-9104`) | não existe | **V** (não crítico: a nossa cabeça LM tem m enorme) |
| bit-exatidão | mudar de kernel muda a ordem de soma (o PPL do llama.cpp aceita isso) | **bit-exato por construção** (`docs/medicoes-m8.md`) | **N** — e isso é um ativo, ver §5 |

### 3.3 Atenção — decode

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| paralelismo | `Br=1` query por workgroup, `Bc=64` chaves, `split_k = 2*CUs/(wgs) = 32` (`:11347-11361`) | 1 CTA de 8 warps por (head, split), 1 token por CTA, `splits = keys/512` até 16 (`graph.cuh:308-325`) | **V** em contexto longo: 384 CTAs (24×16) contra 768 workgroups (192 em x, 4 em y) |
| reuso da linha de K/V entre os 6 heads GQA | **sim**, no workgroup: `iq2 = workgroup.y * gqa_ratio`, `N = gqa_ratio = 6` (`flash_attn_base.glsl:184-186`, `ggml-vulkan.cpp:11251-11259`), K carregado 1× por (col,bloco) e usado pelas 6 linhas (`flash_attn.comp:334-349`) | **não**: cada CTA re-lê a linha do seu head; o protótipo de compartilhar por *warp* mediu 8-12× mais lento (`docs/medicoes-m7.md`) | **V** — e o M7 já tinha escrito a pré-condição para refazer isso: grade larga por splits, que agora existe |
| dequant do KV | inline (`dequantize4`, `flash_attn_dequant.glsl:44-60`) com o tipo como spec constant; passe separado só se K **e** V quantizados e `neq1 ≥ 64` (`:11232-11244`) | inline no registrador, vetorizado, `kv_load8<CT>` (`kv.h:126-170`) | empate no f16; **V** tem a opção do passe separado (útil se voltarmos a KV q4_0) |
| redução | shuffle sobre `D_split` (`flash_attn.comp:432-438`) + merge dos splits | `__shfl_xor` por warp + merge em LDS do CTA (`attn.cuh:129-181`) + kernel de merge (`:401-428`) | empate |
| softmax online | `NEG_FLT_MAX_OVER_2 = 0xFEFFFFFF` em vez de `-inf` (`flash_attn.comp:163`), com `-3·ln2` de viés para caber em fp16 (`flash_attn_base.glsl:215`) | `-INFINITY` com guards explícitos para splits vazios (`attn.cuh:375-377`, `:410-413`) | **N** em exatidão (o truque do Vulkan *muda* a aritmética); **V** em robustez de fp16 |
| máscara | bitmap pré-computado + skip de blocos inteiros (`flash_attn.comp:199-244`, `flash_attn_mask_opt.comp`) | causal implícito (sem máscara, `j ≤ t`, `attn.cuh:118`) | **N**: o decode é causal puro, a máscara materializada do llama.cpp existe para batch/prefill |

### 3.4 Atenção — prefill

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| forma | `block_rows = 4-8`, `block_cols = 32-64` (`:3962-3971`), tudo num dispatch | atenção **token a token** mesmo no caminho batch (`docs/medicoes-m8.md`: "o que continua sequencial… a atenção token a token") | **V** — o nosso `forward_batch_layer` chama `attn_launch_split` por token (`graph.cuh:898-907`) |
| splits | `split_k` derivado dos CUs | idem, por contexto | empate |
| máscara/`mask_opt` | bitmap + `flash_attn_mask_opt.comp` | irrelevante (token a token) | **V** se formos a batch de verdade |

### 3.5 RoPE e a variante IMRoPE (split-half) deste modelo

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| tipo de par | split-half (`rope_neox`/`rope_multi`, `rope_funcs.glsl:108-113`, `:170-175`): `x0 = a[iw/2]`, `x1 = a[iw/2 + n_dims/2]` | split-half (`attn.cuh:42-52`): `jc = p + n_rot/2` | **empate — e numericamente idênticos** (ver abaixo) |
| ângulo | `pow(theta_scale, iw/2)` com `theta_scale = freq_base^(-2/n_dims)` (`rope_funcs.glsl:62`, `ggml-vulkan.cpp:13774`) | `powf(freq_base, -2*p/n_rot)` (`attn.cuh:38`) | idênticos: `theta_scale^p = freq_base^(-2p/n_rot)` |
| **IMRoPE / M-RoPE** | `rope_multi` com `is_imrope` (`rope_funcs.glsl:147-156`): setor = `(iw/2) % sect_dims`, `sector % 3` escolhe **um de quatro tensores de posição** (`rope_data_pos[i2 + ne02*k]`) | posição única: o kernel ignora as seções; o parser **valida** `[11,11,10,0]` e `2·Σ = dims` (`src/backend/model.cpp:231-275`) | **V em generalidade, empate no nosso caso**: texto puro tem as 4 posições iguais ⇒ mesmo resultado. Limitação documentada, não bug |
| escala (yarn) | `freq_scale`/`ext_factor`/`attn_factor` (`rope_funcs.glsl:20-36`) | ausente | **V**, mas inerte aqui (defaults do qwen35) |
| despacho | 2 dispatches por camada de atenção (um `ggml_rope_multi` para Q e um para K, `src/models/qwen35.cpp:299-309`), com o K podendo ser fundido no `set_rows` do cache | 2 lançamentos `rope_launch` por camada de atenção (`graph.cuh:656-657`) + 8 de `kv_write` | **V**: funde `rope + view + set_rows` (`:18251-18258`) e escreve direto no cache |

### 3.6 RMSNorm

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| forma | 1 workgroup de **512** threads por linha (`rms_norm.comp:42`, `:54`), redução em árvore com 1 barreira por nível, 9 níveis (`:87-97`) | 1 CTA de **256** threads por linha, 8 níveis (`nn.cuh:21-45`) | **V** marginal (512 > 256 no caso de 5120 elementos), nosso custo é dominado pelo gap de despacho (209 lançamentos ≈ 0,73 ms) |
| fusões | `RMS_NORM_MUL` (`:18220-18227`), `RMS_NORM_MUL_ADD_MUL` (`:18203-18207`), `RMS_NORM_ROPE_FUSION` / `RMS_NORM_SET_ROWS_FUSION` (`rms_norm.comp:6`, `:35`, `:140-150`) | nenhuma: `rms_norm` é sempre um lançamento (`graph.cuh:622`, `:730`, `:1166`, `:1184`) e o `add` do resíduo outro (`:689`, `:787`, `:804`) | **V**, claramente — é o item 1 do ranking |
| linhas largas | variante de dois passes com `subgroupAdd` (`rms_norm_partials.comp`) | não existe | indiferente aqui (E=5120, 1 linha) |

### 3.7 Cabeça LM e amostragem

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| LM head | GEMV `q5_K` **com q8_1 int8** (q5_k está na lista, `:810`) | `proj(output_, …)` → `vec_dot_q5_K_q8_1` (`matvec.cuh:107`) | empate na instrução; 1,39 ms = 627 GB/s = 98 % do teto medido (`docs/rocm-estudo.md` §A.2.5) |
| logits → host | **não copia no caminho greedy**: `llama_sampler_greedy_backend_apply` injeta um nó `ggml_argmax` no grafo (`src/llama-sampler.cpp:1076-1092`, habilitado em `src/llama-context.cpp:1236-1253`); volta 4 bytes | 993 KB de D2H bloqueante (`graph.cuh:1211`) + softmax/top-k no host (`sampler.h`) | **V**: ~0,3-0,5 ms/token (1-1,5 %, `docs/rocm-estudo.md` §A.2.4) |
| amostragem com temperatura | existe em GPU (`llama_sampler_dist_backend_apply`, `src/llama-sampler.cpp:1274`) | host | **N** para nós *se* considerarmos determinismo: o nosso RNG é próprio e reprodutível (`sampler.h:12-16`); copiar isso significaria reimplementar o RNG no device |

### 3.8 KV cache — layout e quantização

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| layout | por (head, token), strides em push constant; `is_dense_kv_cache` decide se a FA pode ler sem stride (`:11224-11229`) | `[max_ctx, n_head_kv]` por camada, `row_bytes` por tipo (`kv.h:66`), mesmo layout do llama.cpp | empate |
| tipos | f16 default; f32; q8_0; q4_0/q5_0/q5_1; iq4_nl | f32, f16, q8_0, q4_0 (`kv.h:22-27`) | **V** em variedade; irrelevante |
| escrita | `set_rows`, **fundido com rope/norm** (`:18251-18258`, `RMS_NORM_VIEW_SET_ROWS`) | 8 lançamentos por camada (`graph.cuh:701-709`: 4 cabeças × K,V) | **V**: 8 → 1 (ou 0, fundido) |
| dequant no consumo | inline (`dequantize4`) ou passe transposto para f16 (q8_0) | inline vetorizado (`kv_load8`, `kv.h:126-170`) | empate |
| medição nossa | — | f16 vs q4_0 a 64K: 18,91 vs 17,88 tok/s (`docs/medicoes-m7.md`) — q4_0 ainda é alavanca de VRAM, não de velocidade | — |

### 3.9 Roteiro de execução / despacho

| aspecto | V | N | quem é melhor |
|---|---|---|---|
| modelo de execução | grafo inteiro gravado num command buffer reusado, submetido em lotes de ~200 GFLOP/100 nós (`:18065-18083`) | ~1 940 `<<<>>>` num stream | **V** |
| sincronização por op | nenhuma (barreira dentro do CB) | nenhuma | empate |
| sync por token | 1 leitura pequena | 1 D2H de 993 KB (bloqueante) + 16 H2D de 4 B | **V** |
| check de erro | por submissão | `hipGetLastError()` por lançamento de projeção (`graph.cuh:575`, `:598`) | irrelevante em custo (~0,1 ms) |
| descritores | pool pré-alocado por pipeline (`:8446-8453`) | argumentos de kernel | **N** (o HIP não precisa disso) |
| observabilidade | `GGML_VK_PERF_LOGGER`, `GGML_VK_PIPELINE_STATS`, e um `GGML_VK_DISABLE_*` por caminho (`:6584`, `:6592`, `:6600`, `:6617`, `:7421`, `:7430`) | `RD_ATTN_SPLITS`, `--layers`, `bench-attn-gpu`, `bench-matvec-shapes-gpu` | **V** em granularidade: dá para desligar *cada* caminho e medir. O nosso não tem como A/B "matvec int8 vs FMA" |

---

## 4. O que roubar (ranking por ganho × esforço)

Cada item diz **o que muda**, **que medição prova** e **o que pode quebrar**. Nada aqui
foi medido — são propostas calibradas pelos números que já temos.

### 1. Fundir as cadeias escalares e de norma (impacto alto, esforço médio, **bit-exato**)

Copiar a *ideia* das fusões nomeadas do Vulkan (§2.5) como kernels fundidos à mão.

| fusão | arquivos | lançamentos salvos/token |
|---|---|---|
| `sigmoid(attn_gate)*attn` (1 kernel) | novo kernel em `nn.cuh` + `graph.cuh:682-684` | 16 |
| `silu(ffn_gate)*ffn_up` | `nn.cuh` + `graph.cuh:798-801` | 64 |
| `silu(z)*v` do GDN | `nn.cuh` + `graph.cuh:781-782` | 48 |
| cadeia escalar do GDN (`sigmoid`,`add`,`softplus`,`mul` → 1 kernel de 48 threads) | `nn.cuh` + `graph.cuh:746-751` | 144 |
| `rms_norm` + `quantize_q8_1` (escreve a linha normalizada **e** o `block_q8_1`) | `nn.cuh` + `matvec.cuh` + `graph.cuh:622-628`, `:730-736`, `:797` | 128 |
| `add_residual` + `attn_post_norm` (e o `output_norm` final) | `nn.cuh` + `graph.cuh:689`/`:1166`, `:787`, `:804`/`:622`·`:730`, `:1184` | 128 |
| `kv_write`: 1 lançamento para K+V × 4 cabeças | `kv.h:329-345` + `graph.cuh:701-709` | 112 |
| **total** | | **~640** (1 940 → ~1 300) |

- **Prova**: `docs/rocm-estudo.md` §B.5 mede o piso de 3,5 µs por lançamento enfileirado
  ⇒ 640 × 3,5 µs ≈ **2,2 ms/token = +6,5 %** de decode, mais o trabalho real removido
  (a cadeia escalar do GDN sozinha vale ~0,5 ms). Medir com `bench -n 32 --reps 3` A/B e
  `bench --layers N` para localizar o ganho por camada; `bench-matvec-shapes-gpu` cobre o
  replay do matvec (497 lançamentos), que não muda.
- **Gate**: todas as fusões preservam operações e ordem ⇒ `check-graph-gpu` (oráculo por
  nó) + `scripts/check_golden_run.sh` + `check-batch-gpu` têm de ficar **bit-idênticos**.
  A única que exige cuidado é `rms_norm`+`quantize`: manter a **ordem da redução em árvore**
  de `nn.cuh:37-40` e a fórmula exata de `quantize_q8_1_block` (`matvec.cuh:36-52`).
- **Risco**: baixo. Cada fusão é independente e cada uma tem seu próprio gate.

### 2. LUT dos IQ em LDS (impacto médio-alto, esforço **pequeno**, bit-exato)

`iq3_s` é 30,9 % dos bytes e o corpo do `vec_dot` faz 8 `global_load_b32` (uma por grupo de
4 valores) em `vecdotq.cuh:891-892`; `iq2_xxs`/`iq2_xs`/`iq2_s`/`iq3_xxs`/`iq1_s` idem.
O Vulkan copia a tabela para `shared` no início do kernel e lê de lá
(`types.glsl:1576`, `:1748-1756`) — e paga 2 KB no orçamento de LDS
(`ggml-vulkan.cpp:4172`). O `PLAN.md` M2 já registrou isso como candidato ("LUT `iq3s_grid`
em LDS (provado pelo Vulkan)").

- **Muda**: `include/rdna4/quant_tables.h` (ou um novo `lut_lds.cuh`) + os kernels de
  `matvec.cuh:190-279` e `:301-386` (cópia no início + `barrier`), e o `vec_dot` recebe o
  ponteiro da LDS em vez da global.
- **Prova**: `bench-matvec-shapes-gpu` por tipo (`iq3_s`, `iq2_*`, `iq3_xxs`, `iq1_s`) com
  o A/B intercalado que a ferramenta já faz, mais o piso de ruído que ela imprime.
  Alvo: > 5 % em `iq3_s`.
- **Cuidado**: 2-8 KB de LDS/CTA reduzem a ocupação; os nossos CTA são pequenos (1-8
  warps), então medir antes de generalizar para todos os tipos (a LUT de `iq1_s` é 4 KB).

### 3. Amostragem greedy no device (impacto ~1-1,5 %, esforço pequeno, bit-exato)

`ggml_argmax` como nó do grafo no Vulkan (`src/llama-sampler.cpp:1076-1092`). Aqui:
`proj(output_)` seguido de um `argmax_kernel` que devolve 1 `int` — sem copiar 993 KB, sem
drenar o pipeline para copiar logits.

- **Muda**: `graph.cuh:1206-1217` (caminho `greedy`: lançar argmax e copiar 4 bytes),
  um kernel em `nn.cuh`, e `src/main.hip`/`server` para usar o token do device.
- **Prova**: `--layers 0` no `bench` mede 1,95 ms, dos quais 1,39 ms é o LM head ⇒
  ~0,5 ms (1,5 %) para cópia+sync+softmax host (`docs/rocm-estudo.md` §A.2.4). Medir
  `bench -n 32 --reps 3` com o caminho novo; comparar com `tests/golden` (o argmax tem de
  dar o mesmo id em todos os passos).
- **Gate**: `check-sampler` (host) continua valendo para o caminho com temperatura; o
  greedy vira `check-graph-gpu`/golden (mesmo token, bit-exato).
- **Risco**: nenhum para temperatura/top-p (não mexemos neles).

### 4. Atenção estilo Vulkan: grupo GQA por workgroup + splits largos (impacto alto em contexto longo, esforço grande)

É o que mais explica a queda do nosso decode relativo: 74 % do baseline a 4K, **~58 % a
32K** (§2.6). O desenho do Vulkan está em `flash_attn_base.glsl:184-186` com
`N = gqa_ratio = 6` (`ggml-vulkan.cpp:11251-11259`): **um workgroup calcula os 6 heads de
query do grupo**, carrega cada linha de K/V **uma vez** (`flash_attn.comp:334-349`) e
recupera paralelismo por `split_k = 32` (§1.5). O M7 tinha medido um protótipo
*warp-level* 8-12× mais lento e concluído que só valeria "com grade larga por splits" —
essa condição agora existe (`graph.cuh:308-325`, 384 CTAs a 64K).

- **Muda**: `attn.cuh` (`attn_split_kernel`: `ROWS = n_head/n_head_kv` linhas por CTA,
  grade `(n_head_kv, splits)`, K/V lidos uma vez por (coluna, bloco de dim) e reusados nas
  6 linhas) + `attn_launch_split*` (`:431-446`) + `graph.cuh:672-680`/`:902-907`.
- **Prova**: `bench-attn-gpu f16 16384 65536` (a ferramenta mede ms/camada e rel-L2 contra
  o caminho sem split) e `bench --start-pos … --fill-cache` fim a fim. Alvo: 64K f16 de
  1,19 ms/camada para < 0,7 ms.
- **Gate**: **não é bit-exato** (muda a ordem de soma entre chaves). Usar
  `scripts/check_attn_split.sh` (PPL em texto real, limite 0,5 %) **e** `check-kvctx-gpu`
  (comparação split vs sem split), como o M7 fez; manter `RD_ATTN_SPLITS=1` para o caminho
  antigo.
- **Risco**: médio-alto (é o kernel onde o M7 já levou uma surra). Fazer como *bench*
  primeiro, com rel-L2 contra o caminho atual, e só depois plugar no grafo.

### 5. GEMM tiled para prefill (impacto máximo, esforço de milestão, **não bit-exato**)

`docs/medicoes-m8.md` já identificou o teto: 70 tok/s contra 440 é "o laço interno do
`vec_dot`, não o batching". O que roubar do Vulkan é a **estrutura** (não as constantes):
pesos e ativação escalonados em LDS, cache em registradores `WMITER*TM × WNITER*TN`,
tile BM×BN de saída por workgroup (`mul_mmq.comp:97-101`, `:219-281`), e a separação
**GEMV para decode / GEMM para prefill** (`ggml_vulkan.cpp:10433-10437`) em vez de tentar
um kernel que sirva aos dois.

- **Muda**: novo `include/rdna4/mmq.cuh` + `matvec.cuh` (mantendo a GEMV intacta) +
  `graph.cuh:820-1010` (escolher o kernel por N).
- **Prova**: `bench --prefill` (512 e 2048 tokens) — hoje 7,37 s e 29,18 s
  (`docs/medicoes-m8.md`). Alvo incremental: > 120 tok/s com o kernel tiled antes de
  qualquer fusão de precisão.
- **Gate**: **relaxa a bit-exatidão** — a ordem de soma muda dentro do tile. O gate passa a
  ser `scripts/compare_ppl.sh` (desvio por posição contra o llama.cpp) + `check_batch-gpu`
  com tolerância rel-L2 (não `0.0e+00`), e o `check_golden_run` só continua válido para o
  caminho de decode. **Decidir isso explicitamente antes de começar** — é o único item da
  lista que quebra o invariante do M6/M8.
- **Nota importante**: para `iq3_s`/`iq4_xs`/`iq3_xxs` o Vulkan **não** tem MMQ int8
  (§1.1), então não existe receita pronta para os 68 % mais pesados do nosso arquivo: o
  desenho do tile é nosso, com o `dp4a` que já temos, e a referência de estrutura é o
  `mul_mmq.comp` (que é int8 e faz exatamente esse trabalho para os k-quants).

### 6. KV cache quantizado via passe de dequant transposto (impacto médio, esforço médio)

O Vulkan tem um caminho que nós não temos: quando K **e** V estão quantizados e
`neq1 ≥ 64`, dequantiza o cache para um scratch f16 **transposto** uma vez e a FA lê sem
stride (`ggml-vulkan.cpp:11232-11244`, `:11416-11442`; shader `dequant_q8_0.comp` com
`DEQUANT_TRANSPOSE`, `vulkan-shaders-gen.cpp:825-828`). Do nosso lado, `q4_0` a 64K é
**5,8 % mais lento** que f16 (17,88 vs 18,91 tok/s, `docs/medicoes-m7.md`) porque a
dequantização é issue-bound dentro da atenção (`docs/medicoes-m5.md`).

- **Muda**: `kv.h` (um `kv_dequant_transpose_launch`) + `graph.cuh` (lançar o passe no
  início da atenção quando o tipo é quantizado) + `attn.cuh` (ler o scratch f16).
- **Prova**: `bench-attn-gpu q4_0 <t>` vs `f16 <t>` (o bench já toma o tipo) e
  `bench --start-pos … --fill-cache` com KV q4_0 a 64K/131K. Alvo: q4_0 ≥ f16 no mesmo
  contexto — o que compraria ~0,88 GiB a 64K e ~1,8 GiB a 131K (tabela de VRAM do M5).
- **Gate**: `check-kvctx-gpu` (bit-exatidão do cache) — a dequantização é a mesma função
  de `kv.h`, então o resultado deve ser idêntico.
- **Cuidado**: só paga quando o cache é grande (`neq1 ≥ 64` no critério deles); medir o
  ponto de virada antes de ligar por default.

### 7. Barato e útil: observar e poder desligar cada caminho (impacto indireto)

O Vulkan tem `GGML_VK_DISABLE_COOPMAT/COOPMAT2/INTEGER_DOT_PRODUCT/DOT2/DISABLE_MMVQ/FUSION`
(`:6584`, `:6592`, `:6600`, `:6617`, `:7430`, `:7421`) e um `GGML_VK_PERF_LOGGER` que
imprime tempo por nó via query pool (`:18023-18051`). Nós temos `RD_ATTN_SPLITS` e
`bench --layers`. Sugestões concretas, na ordem de valor:

1. `RD_FUSE=0` para desligar as fusões do item 1 (permite medir o ganho de cada uma).
2. `RD_MATVEC_F32=1`: uma variante do `vec_dot` de `iq3_s` **no estilo Vulkan** (FMA fp32
   sobre ativação fp32, `mul_mat_vec_iq3_s.comp:37-48`) para responder *empiricamente* se
   o `dp4a`+`perm` ganha mesmo — a ferramenta existe (`check-matvec-gpu --bench-ab`, que já
   compara variantes intercaladas) e é a pergunta aberta mais interessante desta leitura.
3. Instrumentar tempo por nó (equivalente ao `PERF_LOGGER`) — é o que falta para saber se
   o custo real por despacho neste motor é 3,5 µs ou menos.

---

## 5. O que **não** copiar

1. **Dequantizar pesos para um buffer f16/f32 antes do matmul** (`qx_needs_dequant`,
   `ggml-vulkan.cpp:9501-9507`). No Vulkan isso só existe para tipos sem shader de
   matmul; nós cobrimos os 14 tipos com `vec_dot` fundido. Fazer um passe extra custaria
   uma passada de 11 GB de leitura + escrita por token.
2. **O `occupancy_limiter`** (`flash_attn.comp:75`, `:97-106`): alocar shared memory inútil
   para *baixar* a ocupação é um ajuste para o escalonador do RADV. No nosso lado a
   alavanca equivalente (`__launch_bounds__`/`minb`) foi **medida e rejeitada**: 0,82×
   (`docs/rocm-estudo.md` D.2). Não repetir.
3. **Acumulação em fp16** (`ACC_TYPE = float16_t` nas variantes `f16acc`,
   `vulkan-shaders-gen.cpp:716-718`). Quebra os gates de 1e-6 do M2/M3 e a comparação de
   PPL por posição, que é a nossa prova de qualidade (`docs/medicoes-m5.md` §3). O
   `f32acc` explícito (`ggml-vulkan.cpp:11221`) mostra que nem eles confiam nisso em todo
   caso.
4. **O truque `-FLT_MAX/2` no softmax** (`flash_attn.comp:163`) e o viés `-3·ln2`
   (`flash_attn_base.glsl:215`): evitam NaN/overflow em fp16 **alterando o resultado**.
   O nosso `-INFINITY` com guards explícitos (`attn.cuh:375-377`, `:410-413`) é exato e já
   foi validado contra o oráculo; trocar por uma convenção numérica diferente é regressão
   disfarçada de robustez.
5. **Compartilhar K/V entre as cabeças GQA no nível do *warp*** — o protótipo do M7 mediu
   8-12× **mais lento** (`docs/medicoes-m7.md`). O que se copia é o desenho do Vulkan
   (§4 item 4: linhas por *workgroup* com grade larga), não a versão que já falhou.
6. **A maquinaria de descriptor set / push constant** (`ggml-vulkan.cpp:8446-8461`): ela
   existe porque o Vulkan não tem argumento de kernel. No HIP os argumentos do `<<<>>>`
   já fazem esse papel sem custo de host.
7. **O SPIR-V único com `switch` de tipo por spec constant** (`MULMAT_QUANT`,
   `mul_mm_funcs.glsl:342-630`, registrado só para os tipos não-LUT em
   `ggml-vulkan.cpp:5221-5224`): é a solução deles para não multiplicar variantes de
   shader; os nossos templates de C++ (`RD_MATVEC_TRAITS`, `matvec.cuh:93-140`) dão a
   mesma especialização de graça e com verificação de tipo.
8. **As constantes de tile do `mul_mmq`** (BM=BN=64, BK=32, BK_STEP=4) — copiar a
   *estrutura*, não os números: eles foram escolhidos para o orçamento de LDS daquele
   shader, que ainda carrega 2 KB de LUT (`ggml-vulkan.cpp:4172`). O nosso LDS é
   128 KB/WGP / 64 KB por workgroup (`docs/rocm-estudo.md` B.1), então o tile ótimo é
   outro e tem de ser medido.
9. **Amostragem no device para temperatura/top-p**: além de exigir um RNG no device, quebra
   o critério de aceite determinístico do M4 (`include/rdna4/sampler.h:12-16`). Vale só o
   `argmax` do caminho greedy.
10. **A unificação "um shader serve GEMV e GEMM"**: no Vulkan são **famílias separadas**
    (`mul_mat_vec_*.comp` vs `mul_mmq.comp`/`mul_mm.comp`, `ggml-vulkan.cpp:10433-10437`),
    e isso é uma escolha, não uma limitação. Continuar com `matvec_kernel_gen` para decode
    e um kernel próprio para prefill é seguir o desenho deles — tentar parametrizar o
    mesmo kernel para os dois é o erro que as duas famílias evitam.
11. **Especulação/MTP do llama.cpp**: não é específico do Vulkan, e o M8 já mediu o teto
    (1,3-1,5× por um trabalho considerável e arriscado num modelo híbrido recorrente).
12. **FA com KV `iq4_nl`** (`ggml_vk_fa_type_needs_shmem`, `:4343-4350`): não temos esse
    tipo no cache.

---

## 6. O que eu **não** consigo afirmar daqui (e como fechar)

1. **Qual caminho coopmat este build usa nesta máquina.** O `ggml_vulkan.cpp:20093-20098`
   deixa coopmat1 valer em RADV (o `driverID` do RADV não é nem `eAmdProprietary` nem
   `eAmdOpenSource`, então o workaround de "só RDNA3" não se aplica e a função devolve
   `true`), e coopmat2 depende de propriedades que não posso consultar sem abrir o device.
   **Fechar com** (precisa da fila da GPU, é só carregar o modelo e olhar o log):
   `scripts/gpu-lock.sh .../llama-bench -m <IQ3_S> -ngl 99 -p 32 -n 8 -r 1 2>&1 | grep -E "matrix cores|int dot|fp16:"`
   — a linha `ggml_vulkan: … | fp16: dot2 | … | matrix cores: NV_coopmat2|KHR_coopmat|none`
   (`:7662-7679`) resolve de uma vez.
2. **O custo por despacho do Vulkan.** Não medi; a conta do §2.6 assume que ele é menor que
   os nossos 3,5 µs. **Fechar com**: `GGML_VK_PERF_LOGGER=1` no `llama-bench`
   (`ggml-vulkan.cpp:18023-18051` imprime timestamp por nó) — a soma dos tempos de GPU dos
   ~1 300 nós contra o wall clock por token dá o gap real.
3. **Quantas instruções tem o corpo do `vec_dot` do Vulkan por 220 bytes.** A comparação
   do §3.1 é qualitativa. **Fechar sem GPU**: `spirv-dis` no
   `build/.../mul_mat_vec_iq3_s_f32_f32_subgroup.spv` e contar o bloco do laço — mesma
   metodologia do `docs/rocm-estudo.md` §B.2, só que do outro lado.
4. **Se `v_dot2_f32_f16` está ativo** (extensão Valve `dot2_f16`, `:6616-6618`,
   `:7027`). Se estiver, o caminho fp16 do GEMM faz 2 MACs por instrução e parte da
   desvantagem teórica do FMA desaparece. A mesma linha de log do item 1 responde
   (`fp16: dot2`).
5. **A contagem de ~1 300-1 400 despachos do lado Vulkan** é estimativa a partir do grafo
   `qwen35` + a lista de fusões; para virar número, contar os nós do grafo
   (`llama-eval-callback` ou o `PERF_LOGGER`).

## 7. Índice de leitura (o que foi lido, para auditoria)

**llama.cpp / Vulkan**
`ggml/src/ggml-vulkan/ggml-vulkan.cpp` — 404, 3904-4100, 4150-4208, 4330-4425, 4425-4569,
4700-4740, 4800-4900, 5455-5545, 5560-5600, 6510-6630, 7020-7060, 7140-7160, 7442-7480,
7655-7680, 8029-8115, 8296-8345, 8433-8465, 9023-9117, 9133-9166, 9430-9500, 9514-9700,
9703-9782, 9784-9850, 10367-10440, 11164-11520, 13265-13320, 13739-13848, 17090-17240,
17987-18140, 18142-18390, 20087-20102.
`vulkan-shaders/`: `mul_mmq.comp`, `mul_mmq_funcs.glsl`, `mul_mmq_shmem_types.glsl`,
`mul_mm.comp`, `mul_mm_funcs.glsl`, `dot_product_funcs.glsl`, `mul_mat_vec_base.glsl`,
`mul_mat_vec_iface.glsl`, `mul_mat_vec_iq3_s.comp`, `flash_attn.comp`, `flash_attn_base.glsl`,
`flash_attn_dequant.glsl`, `rms_norm.comp`, `rms_norm_partials.comp`, `rope_funcs.glsl`,
`rope_params.glsl`, `rope_head.glsl`, `rope_norm.comp`, `rope_multi.comp`, `types.glsl`
(1576, 1679, 1748-1756), `vulkan-shaders-gen.cpp` (49-78, 237-258, 461-670, 700-830,
1066-1080); artefatos em `build/ggml/src/ggml-vulkan/vulkan-shaders.spv/`.
`src/models/qwen35.cpp` (150-204, 254-336, 335-470, 470-486),
`src/models/delta-net-base.cpp` (415-450), `src/llama-sampler.cpp` (638-661, 1076-1092),
`src/llama-context.cpp` (1236-1253), `ggml/include/ggml.h` (1926-, 2639-2662).

**Este motor**
`include/rdna4/matvec.cuh` (28-87, 93-140, 155-279, 296-386, 417-541, 545-597),
`include/rdna4/vecdotq.cuh` (84-149, 692-716, 810-916), `include/rdna4/quant_tables.h`
(524), `include/rdna4/attn.cuh` (29-62, 82-181, 290-446, 497-519), `include/rdna4/kv.h`
(22-27, 66, 81-170, 329-345), `include/rdna4/nn.cuh` (21-72, 88-157),
`include/rdna4/graph.cuh` (295-330, 557-603, 619-711, 714-806, 820-1010, 1125-1219),
`include/rdna4/sampler.h`, `src/backend/model.cpp` (231-275), `PLAN.md` M2,
`docs/medicoes-m5.md`, `docs/medicoes-m6.md`, `docs/medicoes-m7.md`, `docs/medicoes-m8.md`,
`docs/rocm-estudo.md`, `docs/baseline-vulkan-iq3s.md`, `docs/gpu-queue.md`.

**Duas correções de contagem encontradas ao conferir o código** (não mudam conclusão
nenhuma, mas os números ficam certos):

- `docs/rocm-estudo.md` §A.2.1 diz "305 lançamentos [de quantização], 192 deles
  redundantes". Contando as chamadas em `graph.cuh`, as projeções são **497**
  (`4×16 + 3×64 + 5×48 + 1`) e as quantizações redundantes são **240**: 2 por camada de
  atenção (`attn_k`/`attn_v` reusam a de `attn_q`), 1 por FFN (`ffn_up` reusa a de
  `ffn_gate`) e 3 por camada GDN (`attn_gate`/`ssm_beta`/`ssm_alpha` reusam a de
  `attn_qkv`) ⇒ `16·2 + 64·1 + 48·3 = 240`. O "192" do doc vem de contar 3 redundantes por
  camada de atenção (são 2) e de esquecer o FFN. O código que roda faz **257**
  quantizações, e o total por token é **1 940** lançamentos.
- O assunto do commit `aa15eea` fala em "384 lançamentos a menos"; pelo código a mudança
  (`proj_qq`) remove **um** lançamento de quantização por chamada reusada, ou seja **240**.

---

## 8. Verificação (coordenador, com a placa livre)

A §6 listava como incerteza número 1 "qual caminho coopmat/`dot2` este build usa
nesta máquina". Fechado com **um** comando, o que a própria §6 propôs:

```bash
scripts/gpu-lock.sh env LD_LIBRARY_PATH=.../llama.cpp/build/bin \
  .../llama-bench -m Qwen3.8-27B-UD-IQ3_S.gguf -ngl 99 -p 32 -n 8 -r 1
```

Saída (llama.cpp b10902, RADV, RX 9070 XT):

```
ggml_vulkan: 0 = AMD Radeon RX 9070 XT (RADV GFX1201) (radv) | uma: 0 | fp16: dot2 |
  bf16: 1 | fp4: 0 | warp size: 64 | shared memory: 65536 | int dot: 1 | matrix cores: KHR_coopmat
| qwen35 27B IQ3_S - 3.4375 bpw | 11.20 GiB | 27.32 B | Vulkan | 99 | pp32 | 207.77 ± 0.00 |
| qwen35 27B IQ3_S - 3.4375 bpw | 11.20 GiB | 27.32 B | Vulkan | 99 |  tg8 |  26.03 ± 0.00 |
```

O que isso resolve:

1. **`matrix cores: KHR_coopmat`** — o gate de vendor (`ggml-vulkan.cpp:20087-20102`)
   **não** rebaixa coopmat1 no RADV: coopmat1 está ligado nesta máquina, como a §1.4
   suspeitava. Ou seja, o backend *pode* despachar GEMM por matriz cooperativa.
2. **`fp16: dot2`** — `v_dot2_f32_f16` existe e está habilitado (2 MACs por instrução
   para f16), e **`int dot: 1`** — `v_dot4_i32_iu8` também (o caminho `_q8_1`/MMQ).
3. Mas isso **não** muda a descoberta 1: para `iq3_s`/`iq4_xs`/`iq2_*` não existe shader
   `*_q8_1` nem coopmat (a lista de `.spv` do build é a prova), e `coopmat` de inteiros
   no caminho GEMV de decode só existe onde existe shader. Para 77,7 % dos bytes deste
   modelo o decode do Vulkan continua sendo **FMA fp32 sobre ativação fp32**.

Ressalva de método: os `tok/s` acima **não** são o baseline deste repositório. O baseline
de `docs/baseline-vulkan-iq3s.md` usa `-p 512 -n 128 -r 3` (modelo quente), e um `pp32` de
1 repetição mede majoritariamente sobrecarga de lançamento, não vazão — foi por isso que
o `tg8` deu 26,0 contra os 39,7 tok/s do baseline. A linha de capacidades é que é o
resultado deste comando; os `tok/s` do baseline devem ser re-medidos com as mesmas flags,
o que o `README.md` faz na seção de comparação.
