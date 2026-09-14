# Atenção e KV no prefill: o que o Vulkan faz (código a código)

Frente **J** do estudo do prefill. Escopo: **só** o caminho de atenção e o cache de KV durante o
**prefill** (512 a 4096 tokens) do llama.cpp Vulkan, comparado com `attn.cuh`/`kv.h`/`graph.cuh`
deste motor. Revisão lida: `/home/marcelo/Projetos/llama.cpp`, `df03399b8`. O irmão deste
documento (`docs/estudo-prefill-a-vulkan.md`) cobre seleção de pipeline do `mul_mm`, tiles do GEMM,
coopmat e fusões — **nada disso é repetido aqui**; o que estiver sem `file:line` aqui está lá.

Convenções: **[D]** = documentado por leitura de código, **[M]** = medido (digo por quem),
**DERIVADO** = aritmética minha sobre medições, **INFERIDO** = não verificado, com o que
confirmaria.

## Nota de integridade (primeiro, porque contamina referência)

`git status` da árvore lida: **uma** linha modificada, `ggml-vulkan.cpp:5462`
(`rm_kq = 2` → `1`, `git diff` completo conferido). É dentro de `ggml_vk_load_shaders`, na tabela
do **`mul_mat_vec`** (GEMV): não toca nenhum shader de attention, nenhum caminho de KV, e não é
citado por nenhuma linha deste documento. **Todo `file:line` daqui vale para `df03399b8` sem
ressalva de deslocamento** (a edição troca 1 linha por 1 linha). O número de atenção de 110,2 µs
por camada que eu uso do `docs/estudo-prefill-a-vulkan.md` §4.1 [M] vem de um binário com esse
patch — e o patch **não** mexe em FA, então aquele número sobrevive à nota de integridade do
`docs/estudo-prefill.md` §0.

## TL;DR (5 linhas)

1. Para o nosso shape (`HSK=HSV=256`, `n_head=24`, `n_head_kv=4`, N=512, KV=512) o Vulkan roda
   **`flash_attn_f32_f16_aligned_cm1`**: `Br=16` linhas de consulta × `Bc=64` chaves por workgroup,
   4 subgrupos de 64 lanes, `coopMatMulAdd` 16×16×16 **f16→f32**, `split_k=1`, grade **32×24 = 768
   workgroups**, LDS de **25,9 KB** (`ggml-vulkan.cpp:4014-4023`, `:11344-11361`, `:11132-11157`).
2. **A fusão GQA `N=gqa_ratio` NÃO vale no prefill**: a condição é `N <= 8`
   (`ggml-vulkan.cpp:11251-11259`), então com 512 tokens o backend tem **as mesmas 6 leituras
   redundantes de K/V que nós** — o §4 do `docs/estudo-prefill.md` ("1 workgroup por grupo kv,
   N=6 cabeças, split_k=32") descreve o **decode**, não o prefill.
3. **KV f16 não passa por LDS nenhuma**: com `aligned` o `coopMatLoad` lê K e V **direto da global**
   para o fragmento (`flash_attn_cm1.comp:269-303`, `:482-486`). KV quantizado ou desquantiza **no
   shader** e é obrigado a estagiar por LDS (`:236-245`, `:406-414`), ou — **só `q8_0`** — passa por
   um `dequant_q8_0_transpose` que copia **o cache inteiro** para um rascunho f16 (`:11230-11244`,
   `:11416-11442`, pipeline em `:5681`).
4. **O prefill paga 2× de trabalho causal até KV < 1024**: a máscara é um tensor `f16 [KV, N]` com
   `-inf` (`llama-kv-cache.cpp:1570`) e o atalho de bloco (`mask_opt`) só liga com
   `nem0 >= Bc*16 = 1024` (`ggml-vulkan.cpp:11309-11310`); abaixo disso todo workgroup percorre
   **todas** as `KV/Bc` blocos de chave.
5. **Atenção é nota de rodapé no prefill, e o número que decide é 2,1 %**: a nossa atenção custa
   **87,7 ms** dos 4174 ms do prompt de 512 (DERIVADO §4.4, validado contra a curva medida do
   prefill; piso de 38 ms pela rota do contexto longo); zerá-la levaria 122,66 → **125,30 tok/s** e
   o gap de 8,54× viraria 8,41×. Do lado deles o kernel inteiro de atenção são **1,76 ms por chunk
   de 512 tokens** = **0,39 %** do prefill deles [M] — mas por MAC emitido a FA deles é **99×** a
   nossa.
   O que **é** alavanca é a *forma* que essa atenção permite: 16 linhas de consulta por workgroup é
   o pré-requisito do micro-lote grande (o fator 2,50× do estudo).

---

## 1. Despacho e forma da flash attention no prefill

### 1.1 A cadeia de seleção, com os nossos números em cada passo

Entrada de `ggml_vk_flash_attn` (`ggml-vulkan.cpp:11164`): `q` permutado para
`[head_dim, n_tokens, n_head]` e `k`/`v` para `[head_dim, n_kv, n_head_kv]`
(`llama-graph.cpp:2609-2610` sobre a view de `llama-kv-cache.cpp:1278-1288`). Logo, para o
`pp512`:

| símbolo | valor | de onde |
|---|---|---|
| `HSK = nek0`, `HSV = nev0` | 256, 256 | `:11188-11189` |
| `N = neq1` | 512 (um ubatch) | `:11190` |
| `KV = nek1` | 512 (cache recém-escrito) | `:11191` |
| `neq2 / nek2` | 24 / 4 | `:11216` |
| `qk_ratio` | **6** | `:11216` |
| `workgroups_x, _y, _z` | 512, 24, 1 | `:11217-11219` |
| `f32acc` | **true** (`GGML_PREC_F32`) | `:11221` + `llama-graph.cpp:2639` |

1. **Caminho**: `device->coopmat2 ? FA_COOPMAT2 : coopmat1_fa_support ? FA_COOPMAT1 : FA_SCALAR`
   (`:4062-4063`). `coopmat2` é `VK_NV_cooperative_matrix2` (não é RADV) e `coopmat1_fa_support =
   coopmat_support && subgroup_require_full_support` (`:7056`) — o RADV aqui reporta
   `subgroupSize = 64` com subgrupos completos obrigatórios ⇒ **FA_COOPMAT1**.
   *(INFERIDO: que os dois gates estão ligados. Confirmaria com `GGML_VULKAN_DEBUG=1`, que loga
   `ggml_vk_create_pipeline(<nome>)` e nomeia `flash_attn_f32_f16_aligned_cm1`.)*
2. **Forma F32/f16 acerta?** `shape_ok = (f32acc && coopmat_support_16x16x16_f32acc) ||
   (!f32acc && coopmat_support_16x16x16_f16acc)` (`:4077-4079`), com a propriedade `f32acc` vindo de
   um `VkCooperativeMatrixPropertiesKHR` com A=B=f16 e C=Result=**float32** 16×16×16 (`:7218-7221`).
   O slop de LDS (`ggml_vk_flash_attn_coopmat_shmem_support`) tem de passar (`:4081-4085`): passa,
   §1.5. *(INFERIDO: `coopmat_support_16x16x16_f32acc` neste RADV. Se fosse falso, cairia em
   FA_SCALAR e o tile seria `Br=8, Bc=32` — §1.6 — com o mesmo `shmem_ok`. Confirmaria com o build
   de debug ou com um `vulkaninfo` que imprimisse as `VkCooperativeMatrixPropertiesKHR`.)*
3. **`N == 1` força SCALAR** (`:4089-4091`): é por isso que o decode não usa coopmat na atenção
   deles, apesar de usar no `mul_mm`.

### 1.2 O tile que sai, e a grade

`get_fa_tuning_params_coopmat1` (`:4002-4031`) **ignora `n_rows`** (`:4003`): o tile é constante:

```
:4014  coopmat_block_rows = 16; coopmat_block_cols = 16; num_subgroups = 4;
:4019  block_rows = 16;                 // Br
:4020  block_cols = 64;                 // Bc = 16 * 4 subgrupos
:4021  row_split  = 4;                  // uma fatia de 16 linhas por subgrupo
:4022  subgroup_size = device->subgroup_size;   // 64
:4023  workgroup_size = 4 * 64 = 256;
:4026  d_split = min(min(64, 8), D_lsb / 4) = min(8, 64) = 8;
```

As *specialization constants* são empurradas em `get_fa_spec_constants` (`:4129-4144`):
`WorkGroupSize=256, Br=16, Bc=64, HSK=256, HSV=256, Clamp=!aligned, D_split=8, row_split=4,
SubGroupSize=64, SHMEM_STAGING=0, Flags, LIMIT_OCCUPANCY_SHMEM=0, FaTypeK/V, FaBlockBytesK/V`.

`Flags` = `USE_MASK_OPT | MASK_ENABLE<<1 | LOGIT_SOFTCAP<<2 | OLD_AMD_WINDOWS<<3`
(`get_fa_pipeline_state`) ⇒ para nós **`Flags = 2`** (tem máscara, sem mask_opt, sem softcap).

**`aligned`** (`:11286-11289`): `KV % Bc = 512 % 64 = 0`, `q_stride = 256 & 7 = 0`,
`k_stride = nbk1/2 = 1024 & 7 = 0`, `v_stride = 1024 & 7 = 0` ⇒ **`aligned = true`** ⇒ variante
`_aligned_` e `Clamp = 0`, isto é **o shader não faz checagem de limites** (`KV_bounds_check =
false`, `flash_attn_base.glsl:32`).

**Grade**: `dispatch_pipeline` divide pelos `wg_denoms` (`:8434-8437`) e o denominador do eixo 0 é
`Br` (`:4770`):

```
x = CEIL_DIV(512, 16) = 32        y = neq2 = 24        z = 1
=> 768 workgroups de 256 threads (4 waves de 64 lanes) por camada de atenção
```

Trabalho por workgroup: `Br×Bc×HSK` = 16·64·256 = 262 144 MACs no `Q·Kᵀ` mais o mesmo no `P·V`;
por bloco de chaves `j` (8 deles, `KV/Bc`) ⇒ **4,19e6 MACs por workgroup**, **3,22e9 MACs por
camada**, **51,5e9 MACs por chunk de 512 tokens** (16 camadas). **DERIVADO**.

**Conferência contra a medição deles**: `FLASH_ATTN_EXT` = 16 × **110,2 µs** no chunk de 512
tokens ([M], `docs/estudo-prefill-a-vulkan.md` §4.1) ⇒ **29,2e12 MAC/s** = **30 %** do teto de
matriz f16 do cartão (97,4e12, §0.4 de lá). É praticamente a mesma taxa do `mul_mm` deles
(32,75e12): **a atenção deles roda tão perto do pico quanto o GEMM deles.** **DERIVADO**.

### 1.3 GQA: a fusão existe, mas os 8 tokens a escondem — e no prefill ela não roda

```
:11249  const uint32_t max_gqa = std::min(tuning_params.block_rows, 32u);      // = 16
:11251  if (N <= 8 && qk_ratio > 1 && qk_ratio <= max_gqa &&
:11252      qk_ratio * nek2 == neq2 && nek2 == nev2 && nem2 <= 1) {
:11256      gqa_ratio = qk_ratio;      // 6
:11257      N = gqa_ratio;
:11258      workgroups_y /= gqa_ratio; // 24/6 = 4
:11259  }
```

Com `N = 512` a primeira condição é falsa: **`gqa_ratio` fica 1, `workgroups_y` fica 24 e cada
workgroup atende **uma** cabeça de consulta**. No shader, `iq2 = gl_WorkGroupID.y * gqa_ratio`
(`flash_attn_base.glsl:182`) ⇒ 24 valores distintos de cabeça; e
`ik2 = iq2 / rk2` com `rk2 = neq2/nek2 = 6` (`:194`) ⇒ **6 workgroups leem as mesmas linhas de K/V
do mesmo kv head**. A correção que sai daqui:

> **No prefill o Vulkan tem exatamente a mesma releitura 6× de K/V que nós.** A "atenção com grupo
> GQA: 1 workgroup por grupo kv, N=6 cabeças, split_k=32" do `docs/estudo-prefill.md` §4 é o
> caminho de **1 token** (`N<=8`, e aí `workgroups_x = 1 ≤ Br` também dispara o `split_k = 32` de
> `:11347-11353`). Num ubatch de 512 tokens nada disso vale.

### 1.4 `split_k` no prefill: 1

```
:11344  Tr = CEIL_DIV(N, Br) = 32
:11349  } else if (gqa_ratio <= 1) {
:11350      total_wgs_no_split = Tr * workgroups_y * workgroups_z = 32*24 = 768
:11351      if (total_wgs_no_split < shader_core_count * 2)      // 768 < 128? NÃO
:11352          split_k = shader_core_count * 2 / total_wgs_no_split;
:11354  }
```

`shader_core_count = 64` (`activeComputeUnitCount`), ⇒ **`split_k = 1`, nenhum `_split_k_reduce`**,
**1 despacho de atenção por camada por chunk** (`:11507-11514`). O caminho `split_k > 1`
(`:11479-11506`, `flash_attn_split_k_reduce.comp`) só aparece em ubatch pequeno: com o **nosso**
chunk de 16 tokens, `Tr = 1`, `total_wgs = 24 < 128`, `split_k = 5 → split_kv = 128 → split_k = 4`.
Ou seja: no `-ub 16` (a comparação de micro-lote igual) eles **já** dividem K em 4.

### 1.5 LDS: 25,9 KB de 64 KB, e nenhum estorvo de ocupação

Modelo do host (`ggml_vk_flash_attn_coopmat_shmem_support`, `:11116-11162`) com os nossos
parâmetros (`HSK_pad=HSV_pad=256`, `row_split=4`, `f16vec4=8`):

| array | fórmula | valor |
|---|---|---:|
| `tmpsh` | `(Bc/MatBc)*4` = 4×4 | 16 B |
| `Qf[Br*qstride]` | `qstride = 256/4+2 = 66`, 16×66×8 | 8 448 B |
| `Psh[Bc*psh_stride]` | `psh_stride = 16/4+2 = 6`, 64×6×8 | 3 072 B |
| `sfsh[Bc*sfshstride]` | `sfshstride = Br/4 = 4` (`HSK>128`), `ACC_TYPEV4` f32 | 4 096 B |
| `kvsh` | `max(64×6, 64×16) = 1024` × `f16vec4` | 8 192 B |
| `pvsh` | `MatBc*osh_stride*f16vec4 = 16×16×8` | 2 048 B |
| `slope[Br]`, `iq_shmem` | `Br*acctype`, LUT IQ4_NL | 64 + 32 B |
| **total** | | **25 968 B** |

(`flash_attn_cm1.comp:42-63` declara os mesmos arrays; `ggml_vk_flash_attn_coopmat_shmem_support`
estima `osh_stride` igual e chega no mesmo total.) O limitador de ocupação
`LIMIT_OCCUPANCY_SHMEM` (`:3986-3997`) **não** se aplica: exige `n_rows >= 64 && hsk <= 128`
(`:3987`) e `hsk = 256`. Com 25,9 KB e 256 threads, a ocupação da FA não é limitada por LDS.

### 1.6 Se o coopmat não existisse (para ler o A/B `DISABLE_COOPMAT`)

`get_fa_tuning_params_scalar` (`:3927-4000`) com `n_rows = 512`: `row_split = 4`,
`subgroup_size = 64`, `workgroup_size = 256`, `reduce_block_rows = (256 & 8) || KV < 1024` =
**true** ⇒ `block_rows = Br = 8`, `block_cols = Bc = 32`, `d_split = 8`. A grade passa a
`CEIL_DIV(512,8) = 64 × 24 = 1536` workgroups, `Tr = 64`, `split_k` ainda 1. O shader é
`flash_attn.comp` (`:4718-4731`) com `Qf`/`masksh`/`kvsh` na LDS (`flash_attn.comp:49-60`) e, em
RADV (`dot2`), `v_dot2_f32_f16` (`dot_product_funcs.glsl:4-13`). **`GGML_VK_DISABLE_COOPMAT=1`
muda a atenção de 16×110,2 µs para outro kernel — o A/B de `pp512` (426,03 tok/s) mistura os dois
efeitos** e não isola a atenção.

---

## 2. O que ele faz com o cache de KV

### 2.1 KV f16: `coopMatLoad` direto da global, sem LDS

No laço de K (`flash_attn_cm1.comp:261-312`):

```
:269  const bool stage_k = USE_DECODE_K || KV_bounds_check || d * 16 + 16 > HSK;
:270  if (stage_k) { ...estagia Bc x MatBr em kvsh... }
:300  } else {
:301      const uint coord = k_offset/4 + (j*Bc + gl_SubgroupID*MatBc) * k_stride/4 + d*16/4;
:302      coopMatLoad(KMat, data_kv4, coord, k_stride / 4, gl_CooperativeMatrixLayoutRowMajor);
:303  }
```

Para f16 alinhado e sem checagem de limites, `stage_k` é **falso** (`USE_DECODE_K` =
`FaTypeK != F16`, `flash_attn_base.glsl:108`) ⇒ **zero instruções de staging**: o fragmento 16×16
de K vai do L2 direto para o registrador de matriz. Idem V (`:444`, `:482-486`). O `kvsh` de 8 KB
(§1.5) fica alocado mas **não é usado no caminho f16**.

Layout lido: `k_stride = nbk1/2` em elementos f16 = `nk_gqa = 4×256 = 1024` ⇒ **cada linha de K é
um pedaço de 16 f16 = 32 B a cada 2 048 B** (a linha de um kv head são 512 B dentro de um stride de
2 KB, porque o cache é `[kv_head][head_dim]` por token). O `coopMatLoad` faz `MatBr/4 = 4` acessos
de 16 B por linha, e o volume pedido por workgroup por bloco `j` é
`Bc×(HSK+HSV)×2 B = 64 KB`. Por camada: **768 × 8 × 64 KB = 402 MB pedidos** contra
**2 MB únicos** (K+V de 512 chaves) ⇒ **amplificação 192× = 6 (GQA) × 32 (blocos de 16 linhas de
consulta)**. **DERIVADO** — e é essa amplificação que exige que os 768 workgroups estejam vivos ao
mesmo tempo para o L1/L2 absorverem.

### 2.2 As duas multiplicações, e o tipo de acumulação

```
:257  coopmat<ACC_TYPE, Subgroup, MatBc, MatBr, Accumulator> SfMat = ...(0);
:258  coopmat<FLOAT_TYPE, Subgroup, MatBc, 16, MatrixA> KMat;   // K (e depois P)
:259  coopmat<FLOAT_TYPE, Subgroup, 16, MatBr, MatrixB> QMat;   // Q (e depois V)
:311  SfMat = coopMatMulAdd(KMat, QMat, SfMat);                 // K x Q^T, 16x16x16, HSK/16 vezes
:496  PVMat = coopMatMulAdd(KMat, QMat, PVMat);                 // P x V, 16x16x16
```

Ou seja, o `Q·Kᵀ` é escrito **transposto** (`Bc × Br`, `K` como A e `Q` como B carregado em
`ColumnMajor`, `:309`), para o caso `N=8` do GQA. Tipos, com as duas pontas do gerador
(`vulkan-shaders-gen.cpp:711-717`: `ACC_TYPE = fp16 && f16acc ? float16_t : float`):

| coisa | tipo | onde |
|---|---|---|
| `SfMat`, `PVMat` (acumulador do MMA) | **f32** (`f32acc`, forçado por `GGML_PREC_F32`) | `:11221`, `llama-graph.cpp:2639` |
| operandos A/B do MMA | **f16** | `:36-39`, `:258-259` |
| `sfsh` (escores + softmax) | f32 | `:52`, `ACC_TYPEV4` |
| `Psh` (P, entrada do 2º MMA) | **f16** | `:48`, `FLOAT_TYPEV4` |
| `Of` (acumulador do laço `j`) | **f16** | `:105`, `O_TYPEV4 = FLOAT_TYPEV4` |
| saída/`L`/`M` finais | f16 com `1/L` | `:607-620` |

Ou seja: **`Q·Kᵀ` e `P·V` acumulam em f32; P e o acumulador que atravessa os blocos de chave são
f16.** (É o oposto do `mul_mm` deles, que roda `f16acc` por não haver correção depois —
`docs/estudo-prefill-a-vulkan.md` §2.6.)

### 2.3 KV quantizado: dois caminhos, e o custo por elemento

**(a) `q8_0` no prefill: o cache inteiro é copiado para f16.** É o achado que mais muda a leitura
do "formato do KV no prefill":

```
:11230  k_quant = k->type != F16 && != BF16 && != F32;
:11232  use_dequant_kv = k_quant && v_quant && neq1 >= 64 &&       // N >= 64: SÓ PREFILL
                         is_dense_kv_cache(k) && is_dense_kv_cache(v) && ... &&
:11236                  pipeline_dequant_transpose[k->type] != nullptr &&
:11237                  pipeline_dequant_transpose[v->type] != nullptr && !coopmat2;
:11243  k_type_eff = use_dequant_kv ? GGML_TYPE_F16 : k->type;
...
:11424  vk_pipeline tr_k = ctx->device->pipeline_dequant_transpose[k->type];
:11436  ... dispatch tr_k sobre k_nel elementos ...
```

`pipeline_dequant_transpose` é criada para **um único tipo** — `GGML_TYPE_Q8_0` (`:5681`), porque o
gerador só emite a variante `DEQUANT_TRANSPOSE` para `q8_0`
(`vulkan-shaders-gen.cpp:825-828`). O shader (`dequant_q8_0.comp:19-33`) lê `[HS, NH, KV, NS]` e
escreve `[HS, KV, NH, NS]` — ele **transpõe**, para a FA ler K sem stride de cabeça. Depois disso a
FA roda o caminho f16 da §2.1 sobre o rascunho.

**(b) tudo o mais (`q4_0`, `q5_0`, `q4_1`, `iq4_nl`, ...): dequantização dentro do shader**, e ela
**obriga o staging por LDS** — `USE_DECODE_K/V` verdadeiro torna `stage_k`/`stage_v` verdadeiros
(`:236-245` para K já estagiado, `:279-288` para K pelo palco de `MatBr` colunas, `:406-414` para
V). As macros estão em `flash_attn_dequant.glsl` e o custo por elemento é este:

| tipo | instruções por 4 elementos | por elemento | linha |
|---|---:|---:|---|
| `q4_0` | 2 cargas de byte + 2 shifts + montagem de `vec4` + `d`→float + sub + mul ≈ **9** | 2,25 | `:46-55` |
| `q8_0` | 2 `unpack8` + 1 mul + `d` ≈ **5** | 1,25 | `:101-105` |
| `q5_0` | ≈ 12 (mais a montagem do 5º bit) | 3,0 | `:69-83` |
| `q4_1` | ≈ 10 (mais o `m` do bloco) | 2,5 | `:57-67` |

E o que isso custa **no tile**: por subgrupo, por bloco `j`, os elementos a desquantizar são
`16 linhas × 256 dims × 2 (K e V) = 8 192`; a `q4_0` a 2,25 instr/elemento isso dá
**18 400 instruções de lane = ~288 instruções de wave64**, contra **32 `coopMatMulAdd`** no mesmo
bloco (16 para `Q·Kᵀ` + 16 para `P·V`). **≈ 9× mais instrução de staging do que de MMA.**
**DERIVADO** — é a razão numérica pela qual o `q8_0` deles sai do shader e vira cópia de cache
inteiro (§2.3a), e a razão pela qual **f16 é o formato do prefill**.

**Resposta direta à pergunta "o caminho quantizado é alcançável no prefill com coopmat?":** sim,
os dois: `q8_0` pelo atalho do rascunho f16 (que só existe porque `N >= 64`), e `q4_0/q5_0/q4_1`
por desquantização no shader + LDS. O que **não** existe com coopmat é o caminho MMQ/dp4a da FA
escalar: `ggml_vk_fa_scalar_uses_mmq` (`:4352-4362`) exige `FA_SCALAR`, e ele é o que gera
`flash_attn_f32_f16_int8` (`vulkan-shaders-gen.cpp:741-746`). No caminho cm1 a multiplicação é
sempre f16.

### 2.4 O custo do `q8_0` em números nossos

Sem medição, só aritmética **DERIVADO**: por camada, por chunk, o `dequant_q8_0_transpose` move
`KV × 4 × 256` elementos — a 512 chaves são 524 288 elementos = **0,56 MB lidos (q8_0) e 1,05 MB
escritos (f16)**, mais 2 despachos (K e V). Irrelevante a 512. A **4096 chaves por chunk** já são
4,2 M elementos por tensor (13,9 MB de leitura + 16,8 MB de escrita **por camada**, ~490 MB de
tráfego por chunk de 512 tokens somando 16 camadas). O custo **cresce com o contexto** e é pago
**inteiro a cada chunk** — é o oposto de um cache.

---

## 3. Máscaras, causal e a escrita do KV

### 3.1 A máscara é dado, não fluxo de controle

`build_attn_inp_kq_mask` (`llama-graph.cpp:29-45`): tensor `f16 [n_kv, n_tokens, 1, n_stream]`; com
`causal_attn` as posições futuras recebem `-INFINITY` e as válidas `0`
(`llama-kv-cache.cpp:1565-1570`, `mask_keep = 0`, `mask_drop = -INFINITY`). Esse tensor entra na
FA (`llama-graph.cpp:2632`) e vira `Flags |= MASK_ENABLE` (`:11311-11312`). No shader, o tile
`Br×Bc` é lido para `mask_cache` (`flash_attn_cm1.comp:151`, `:160-226`) e somado a `sfsh`
(`:329-343`).

Consequência **estrutural**, e é a que interessa: como a causalidade é máscara, **os 16 tokens de
consulta de um workgroup podem viver no mesmo CTA** — cada um com um `t` diferente —, porque o que
separa um do outro é o valor da máscara, não o laço de chaves. É exatamente o oposto do nosso
`attn_batch_kernel`, onde `t = pos[qt]` é *iteração* (`attn.cuh:328`, `:348`).

### 3.2 O atalho causal só existe a partir de KV ≥ 1024 (e antes disso o prefill paga 2×)

```
:11309  bool use_mask_opt = mask && nem1 >= 32 && nem0*nem1 > 32768 &&
:11310                      nem0 >= tuning_params.block_cols * 16 &&
```

Para o nosso shape, `block_cols = 64` ⇒ o limiar é `nem0 = KV >= 1024`. Em KV = 512
**`use_mask_opt = false`** e o shader **não tem como pular bloco nenhum**: as condições que
pulariam (`if (mask_opt_bits == MASK_OPT_ALL_NEG_INF) continue;`, `:166-169`, e
`max_mask <= NEG_FLT_MAX_OVER_2 → continue`, `:222-224`) dependem do bitmask que só o pré-passe
escreve. Com N = 512 e KV = 512, os 32 blocos de consulta percorrem **os 8 blocos de chave
inteiros** ⇒ **o dobro dos MACs causais** (51,5e9 contra 25,9e9 úteis).

De KV ≥ 1024 em diante o pré-passe `flash_attn_mask_opt` (`:11446-11464`, shader
`flash_attn_mask_opt.comp`) escreve **2 bits por bloco (Br × 16·Bc)**: `1 = ALL_NEG_INF`,
`2 = ALL_ZERO`, `0 = misto` (`:29-30`, `:46-60`) e a FA volta a pular o que é futuro. Custa
**1 despacho a mais por camada** e um buffer de `CEIL_DIV(KV, 16·Bc) × CEIL_DIV(N, Br) × 4 B`
(`:11376-11377`) = 512 B a 4096 chaves.

### 3.3 A escrita do chunk no cache: fusão quando f16, `copy_to_quant` quando quantizado

- **f16**: a fusão `ROPE_VIEW_SET_ROWS` / `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS`
  (`ggml-vulkan.cpp:18251-18258` e `:18182-18193`) faz **um** despacho que calcula o rope e escreve
  a linha **dentro do cache** (`ggml_vk_rms_norm_mul_rope`, `set_rows_stride` em `:13924-13936`).
  Para o nosso grafo não há `RMS_NORM` antes do rope do K (a qk-norm é do qwen35 — o nosso motor
  tem `rms_norm` de k e v também: `graph.cuh:1069-1076`), então o padrão é o de 3 nós:
  `ROPE + VIEW + SET_ROWS` ⇒ **1 despacho por camada**.
- **quantizado**: a fusão é **desligada por tipo** — `if (set_rows->type != GGML_TYPE_F32 &&
  set_rows->type != GGML_TYPE_F16) return false;` (`:17750`). A linha então vai por
  `copy_to_quant.comp` com `SET_ROWS` (`:9285` seleciona `pipeline_cpy_f32_quant[dst]`;
  `vulkan-shaders-gen.cpp:887-891` gera `set_rows_<src>_<quant>_i64`; o shader muda para
  `local_size_x = 512` e `BLOCK_SIZE = 512`, `copy_to_quant.comp:5-13`) ⇒ **1 despacho por tensor**
  (K e V separados) **mais** o rope. É o mesmo trabalho que o nosso `kv_write_batch` faz
  (`graph.cuh:872-882`, 2 lançamentos: K e V, cada um com `width = n_tok*NKV*HD` elementos).

Nosso caminho **já é o barato aqui**: `kv_write_batch` = 2 lançamentos por camada por chunk contra
1 (f16) ou 2-3 (quantizado) deles, e a diferença total é lançamento, que é 1,6 % do prefill
(`docs/estudo-prefill-c-nosso.md` §4.2).

### 3.4 Contagem de despachos por camada de atenção no prefill

| peça | nós | eles (512 tokens, KV f16) |
|---|---:|---:|
| atenção | 1 por camada por chunk (`attn.cuh:423-425` via `forward_batch_layer`) | **1 por camada** (`split_k=1`, `:11512-11514`) |
| máscara | 0 (o `t` é o laço) | 0 com KV < 1024; **1** com KV ≥ 1024 |
| escrita do KV | 2 (`graph.cuh:873-874`) | **1** (`ROPE_VIEW_SET_ROWS`) |
| dequant do cache | 0 | 0 (f16); **2** com `q8_0` |
| rope do q/k | 2 (`graph.cuh:1075-1076`) | fundido no de cima / no `RMS_NORM_MUL_ROPE` |
| **por camada de atenção** | **~11** (o resto é norma/qk-norm/projeções) | ~7 |

A 512 tokens o nosso prefill faz **32 chunks** por camada ⇒ **32 despachos de atenção por camada**
num kernel que aceita `gridDim.y = 16` tokens; eles fazem **1** que aceita N = 512. A diferença de
despacho é ruído (1,6 %); a diferença de **forma paralela** não é (§4).

---

## 4. O nosso lado, e o número que decide

### 4.1 `attn_batch_kernel`: grade, paralelismo e o que roda por chave

```
attn.cuh:416   dim3 grid((unsigned)n_head, (unsigned)n_tok);      // (24, 16) = 384 CTAs
attn.cuh:414   threads = WPB*32 = 8*32 = 256                     // tuning.h:98 (kAttnWarpsPerBlock=8)
attn.cuh:415   smem = 8 * (2+256) * 4 = 8 256 B                  // por CTA
attn.cuh:348   for (int j = w; j <= t; j += kAttnWarpsPerBlock)  // cada warp anda (t+1)/8 chaves
attn.cuh:352   kv_load8<KT>  -> 16 B por lane por linha          // f16 = 1 uint4
attn.cuh:357   8 fmaf + 5 shfl_xor + 5 add  -> score            // redução POR CHAVE
attn.cuh:363-368  2 expf (correção do max + peso)                 // por chave, por warp
attn.cuh:377   8 fmaf (P*V)
```

Uma CTA por **(cabeça, token)**; o eixo dos tokens é `blockIdx.y`. Não existe reúso de tile: cada
linha de K e de V é relida (e re-desquantizada) por **cada** token de consulta que a vê, e a
redução do `q·k` é uma árvore de 5 `shfl` **por chave** com dois `expf` atrás — numa cadeia de
dependência serial por chave, sem paralelismo entre chaves dentro do warp. A FA deles faz o mesmo
trabalho com 32 instruções de MMA por 64 chaves e uma redução por **coluna de tile**
(`subgroupMax`/`subgroupAdd` em `flash_attn_cm1.comp:361`, `:531`), não por chave.

### 4.2 A releitura de 6× (nossa) e 192× (deles)

`kvh = h / (n_head / n_head_kv)` (`attn.cuh:331`): as 6 cabeças de consulta de um grupo leem as
mesmas linhas. Medido/derivado no estudo de contexto longo: **25,77 GB lógicos contra 4,295 GB
únicos a 64K, f16** (`docs/kv-memoria-desenho.md:303`, `docs/journal-longctx.md:197`). A taxa
efetiva do kernel é **~1,36 TB/s de L2** e **~225 GB/s de DRAM** a 4K e a 64K — ou seja, ele
satura pedidos, não banda (`docs/kv-memoria-desenho.md:355-362`).

Do lado deles a amplificação é **maior**, não menor: 6 (GQA) × 32 (blocos de 16 linhas) = **192×**
o byte único (§2.1). A diferença é que **eles pedem 16× menos por MAC** (0,125 B/MAC contra 2 B/MAC
nossos — o tile `Br=16` divide o pedido por 16), e o resto é computação: MMA contra
`fmaf+shfl+expf`.

### 4.3 A política de splits

`attn_splits_for(keys)` = `keys / kAttnSplitMin` com teto 16 (`graph.cuh:426-428`,
`tuning.h:101-102`), decidida por `pos0 + n` — o **último** token do chunk (`graph.cuh:1086`).
Num prompt de 512 tokens o último chunk tem `pos0+n = 512` ⇒ **splits = 1 em todos os 32 chunks**:
o `attn_batch_kernel` sem split, **1 despacho por camada por chunk**, `attn_split_batch` nunca
entra (`docs/estudo-prefill-c-nosso.md` §4.1 já registra o mesmo). O `split_k` deles a 512 tokens
também é 1 (§1.4) — a diferença é o tamanho do `Br`, não a divisão de K.

### 4.4 A conta que decide: 2,1 %

Medições nossas da atenção (16 camadas, o kernel puro):

| regime | valor | fonte |
|---|---|---|
| 64 tokens de prompt (chunk 16, contexto ≤ 64, `splits=1`) | **0,031 ms/token** | [M] `docs/estudo-prefill-c-nosso.md` §2.1 |
| 128 tokens (contexto ≤ 128, `splits=1`) | **0,051 ms/token** | [M] idem |
| 4K chaves (8 splits, benchmark por token) | **1,2 ms/token** | [M] `docs/rocm-estudo.md:63` |
| 64K chaves (8-16 splits, benchmark por token) | **19,0 ms/token** | [M] `docs/medicoes-banda-e-gargalos.md:370-371` (1,19 ms/camada × 16) |

Duas rotas para 512 chaves, e elas **não** concordam — o regime é diferente:

1. **Ajuste nos dois buckets do prefill, que é o regime certo** (`splits = 1`, o kernel em lote):
   `0,031 = a + 32b`, `0,051 = a + 64b` ⇒ `b = 0,000625 ms/token por chave`, `a = 0,011`. Somando
   sobre os 512 tokens (`Σ(t+1) = 131 328`):
   `512×0,011 + 0,000625×131 328 = 5,6 + 82,1` ⇒ **87,7 ms por prompt = 0,171 ms/token**.
   **Validação independente desta rota**: ela prevê **+0,140 ms/token** de N=64 para N=512; a curva
   medida do prefill dá **+0,130** (8,02 → 8,15 ms/token, `docs/estudo-prefill-c-nosso.md` §1.1) —
   dentro de 8 %, e é a medição *dentro* do prefill, não uma extrapolação.
2. **Lei linear das medições de contexto longo** (`0,29 µs por chave por token`, ×16 camadas):
   `0,29e-3 × 131 328 = 38,1 ms = 0,074 ms/token`. É **piso**, não estimativa: aquela lei vem do
   regime *com 8-16 splits* (384+ CTAs, latência escondida), e num prompt de 512 tokens o
   `attn_splits_for` devolve **1 em todos os 32 chunks** (§4.3). Confirmaria o piso com o prefill
   de 512 medido com `RD_ATTN_SPLITS` forçado — que **muda a aritmética** e por isso não serve como
   medição direta.

⇒ **DERIVADO: a nossa atenção custa 0,171 ms/token num prompt de 512 tokens = 87,7 ms dos
4174 ms medidos = 2,1 %** (piso 0,9 % pela rota 2).

Do lado deles, medido: **16 × 110,2 µs = 1,76 ms por chunk de 512 tokens** = 0,39 % dos 447,5 ms
deles, **3,44 µs/token** [M].

| | nosso | deles | razão |
|---|---:|---:|---:|
| ms por prompt de 512 tokens | **87,7** (DERIVADO) | **1,76** [M] | **50×** |
| µs/token | 171 | 3,44 | 50× |
| % do prefill | **2,1 %** | 0,39 % | — |
| MACs/s (úteis nos dois lados) | 0,294e12 | 14,6e12 | **50×** |
| MACs/s emitidos (eles fazem 2× o trabalho causal) | 0,294e12 | 29,2e12 | **99×** |

O `99×` é o número que dói: **por MAC emitido, a FA deles é ~100× a nossa** — 17,5 ps por par
(consulta, chave, cabeça, camada) contra 1 740 ps. É pior que os 8,5× do prefill inteiro e pior que
os ~9× do matvec, e a razão é a forma: 512 MACs por par feitos com 2 pedidos de 16 B + um `fmaf` de
8 elementos + 5 `shfl` + 2 `expf` (§4.1), contra 1 `coopMatMulAdd` por 16 pares × 16 chaves.

**A resposta à pergunta: atenção é nota de rodapé no prefill, e o número que decide é 2,1 %.** Se a
nossa atenção custasse **zero**, o prefill de 512 iria de **122,66 para 125,30 tok/s** (+2,2 %) e o
gap de 8,54× viraria **8,41×** — a atenção explica **1,5 % do gap**. Mesmo igualando-a exatamente à
deles, sobra ≤ 2 %. O que **não** é nota de rodapé é o que a forma da FA deles habilita:

- **16 linhas de consulta por workgroup** é o mecanismo que torna um micro-lote grande barato na
  atenção, e o micro-lote grande é **2,50×** do gap medido (`docs/estudo-prefill.md` §0). O nosso
  `attn_batch_kernel` já aceita 16 tokens em `gridDim.y`, mas com uma CTA por token: subir o chunk
  para 128/512 sem mudar a forma só multiplica CTAs, não reúso. **Atenção não é alavanca de
  *tempo*; é pré-requisito do *lote*.**
- **A rota 1 do §4.4 não vale acima de 512 chaves, e isso está medido.** Ela prevê
  `0,625 µs/chave`; a curva do prefill de 64 a 4096 tokens cresce só **+0,55 ms/token**
  (`docs/estudo-prefill-c-nosso.md` §1.1), isto é `0,27 µs/chave` em média — **2,3× menos do que a
  rota 1 prevê**. O motivo está no código: de 512 chaves para cima o `attn_splits_for` deixa de
  devolver 1 (`graph.cuh:426-428`) e o kernel ganha CTAs (menos latência exposta por chave), o que
  é coerente com os `0,29 µs/chave` medidos a 4K/64K. Ou seja: **a atenção é 2,1 % a 512 tokens e
  o próprio crescimento dela é o que a curva de 64→4096 mede**; a 4096 ela é
  **≤ 0,031 + 0,55 = 0,58 ms/token ≤ 6,8 %** (cota superior por diferença). Continua não sendo a
  alavanca; começa a ser um item de 2º nível.

---

## 5. O formato do KV no prefill: a mesma conclusão vale?

**O que foi medido (noite, 131K, decode)**: o formato do KV **não** compra velocidade — spread de
1,9 % entre 2,25 e 4,25 GiB de cache, e o **maior** é o mais rápido — porque a atenção é limitada
por latência/ocupação (**165-172 GB/s efetivos**, 27-29 % do pico) e não por banda
(`docs/journal-kv.md:26-28`, §6.1).

**O que o código diz para o prefill — a razão muda de lugar, e a resposta muda:**

1. **A intensidade aritmética por byte de KV é 16× maior.** `Br = 16` linhas de consulta por
   workgroup (§1.2) ⇒ 0,125 B de KV por MAC contra ~2 B/MAC nossos (§4.2). Quando o kernel deixa de
   ser limitado por byte lido, o formato do KV **para de pagar** — e passa a **custar**.
2. **O caminho f16 no prefill não paga desquantização nenhuma**: `coopMatLoad` direto da global
   para o fragmento (§2.1). O `q4_0` paga ~2,25 instruções por elemento **e** uma ida e volta pela
   LDS (§2.3): ~9× mais instrução de staging do que de MMA no mesmo tile. **No prefill o formato
   quantizado é um custo de instrução, não uma economia de banda** — exatamente o inverso do
   decode, onde ele economiza byte e a instrução é de graça (o kernel está esperando latência).
3. **O `q8_0` deles resolve isso pior**: sai do shader e vira uma **cópia do cache inteiro** por
   camada por chunk (§2.3a, `:11232-11442`), cujo custo cresce com o contexto e é pago por chunk.
   É uma escolha defensável a 512 tokens (0,56 MB lidos por camada) e indefensável a 4096
   (≥ 490 MB de tráfego extra por chunk de 512 tokens, DERIVADO).
4. **Do nosso lado o custo da desquantização é hoje 16× maior do que seria com tile**: nós
   desquantizamos por par (consulta, chave) — `kv_load8<Q4_0>` roda uma vez por lane por linha por
   token (`attn.cuh:373`, `kv.h:281-296`) — enquanto a forma tiled desquantiza uma vez por tile de
   16 linhas. Ou seja: **a nossa medição noturna de que o formato não importa seria reproduzida no
   prefill por um motivo errado** (a atenção é 2,1 % do tempo, então nada nela importa), e a
   conclusão "o formato não importa" deixaria de valer no dia em que a atenção fosse tiled.
5. **O que teria de ser medido** (nenhum dos dois existe hoje):
   - do lado deles: `llama-bench -p 512 -n 0 -r 3 --cache-type-k q8_0 --cache-type-v q8_0` e
     `q4_0/q4_0` contra `f16/f16`, **e o mesmo com `-p 4096`** — a previsão do código é ~0 a 512 e
     **negativo** no `q8_0` acima de ~1024 chaves por causa do `dequant_transpose`;
   - do nosso lado: o kernel de atenção **em lote** com `4K/16K/64K` chaves × `{f16,q8_0,q4_0}`
     (o `bench-attn-gpu` já existe para isso, mas mede o caminho por token) — é o que separa
     "instrução de desquant" de "byte lido" no nosso caso.

---

## 6. O que portar (ordenado por ganho esperado)

> Ordem por **ganho no prefill**, e o ganho da atenção é ≤ 2,1 % (§4.4). O item 1 é primeiro
> porque é o **pré-requisito** dos 2,50× do micro-lote, não pelos 2,1 % dele.

1. **Tile de `Br = 16` linhas de consulta por CTA, com a causalidade como máscara `f16 [KV, N]`.**
   *Portável.* Vem de `ggml-vulkan.cpp:4014-4023` (`Br=16`, `Bc=64`, `row_split=4`, 256 threads) e
   de `flash_attn_cm1.comp:95-102` (Q estagiado em `Qf`), `:257-312` (`SfMat` = `Bc×Br` em 16
   passos de 16 em `HSK`), `:329-343` (máscara somada a `sfsh`), `:372-396` (P em `Psh`).
   Toca `attn.cuh`: `attn_batch_kernel` ganha um eixo de **bloco de tokens** (não de token) e um
   laço de blocos de chave com a P em LDS; `graph.cuh` passa a construir a máscara `f16` do chunk
   (hoje a causalidade é `t = pos[qt]`, `attn.cuh:328`). **Por que é o item 1**: é o único que
   ataca as duas causas medidas ao mesmo tempo — 16× menos pedido de KV/desquant (0,125 B/MAC) e a
   morte da redução `shfl`+`2 expf` por chave (a redução passa a ser por coluna de tile,
   `flash_attn_cm1.comp:361`/`:531`) — e é o que torna o chunk de 128/512 possível na atenção.
   Ganho próprio ≤ 2,1 %; **ganho habilitado**: é a metade "atenção" do P0 (`docs/estudo-prefill.md`
   §5).
2. **A P em `f16` na LDS e o acumulador do laço de chaves em `f16`, com `Q·Kᵀ`/`P·V` em f32.**
   *Portável.* `flash_attn_cm1.comp:48` (`Psh` = `FLOAT_TYPEV4`), `:105` (`O_TYPEV4`), `:257`
   (`ACC_TYPE`), `:496`; o porquê do `f32acc` está em `llama-graph.cpp:2639` + `ggml-vulkan.cpp:11221`.
   Toca `attn.cuh` (a P hoje nunca é materializada: `p` é escalar por chave, `:368`) e é o que
   **barateia a P**: 16 consultas × 1 valor por chave em vez de 256 lanes recalculando o mesmo
   `expf`. Custo zero em precisão nova: eles já aceitam P em f16.
3. **Desquantizar para a LDS uma vez por tile (e não por par consulta×chave).**
   *Portável com ressalva.* Vem de `flash_attn_cm1.comp:236-245`/`:406-414` (`dequantize4` →
   `kvsh`) e `flash_attn_dequant.glsl:46-105`; o preço medido por elemento está na tabela da §2.3.
   Toca `kv.h` (as macros `kv_load8<Q4_0>`/`Q5_0`/`Q4_1` viram "desquantize um tile de 16×32 para a
   LDS"), não os bytes do cache — os nossos bytes já são idênticos aos do llama.cpp
   (`docs/journal-kv.md` §1). **Ressalva**: só vale depois do item 1; sem o tile não há onde
   amortizar, e a nossa medição noturna (formato não compra velocidade no decode) continua válida
   para o decode.
4. **Atalho de bloco causal por bitmask (2 bits por bloco `Br × 16·Bc`), com a máscara só até
   `t`.** *Portável.* Vem de `ggml-vulkan.cpp:11309-11310` (limiar `KV >= Bc*16`) e `:11446-11464`
   (o pré-passe), `flash_attn_mask_opt.comp:29-30`, e do consumo em `flash_attn_cm1.comp:161-169`.
   Serve **ao contrário** para nós: o item 1 introduz o desperdício que eles têm (com um bloco de 16
   consultas, o `t` do bloco é o maior `t`, então as linhas iniciais pagam chaves futuras); o
   bitmask devolve isso a partir de ~1024 chaves, e **abaixo disso nós podemos fazer melhor que
   eles** (pular por `t` real dentro do bloco custa zero e eles não fazem).
5. **Escrever o K com o rope direto no cache (`ROPE_VIEW_SET_ROWS`).** *Portável.*
   `ggml-vulkan.cpp:18251-18258`, `:13924-13936`; o teto do ganho é o custo de lançamento: **1,6 %**
   do prefill, dos quais isto é uma fração (`docs/estudo-prefill-c-nosso.md` §4.2). Toca
   `graph.cuh:1076`+`:1080` (fundir `rope_launch` de k com `kv_write_batch`).
6. **`dequant_transpose` do cache inteiro para `q8_0`.** *Não portável como ganho* — é o
   contrário: é o que o llama.cpp faz para **não** desquantizar no shader no prefill, e o custo
   cresce com o contexto (§2.3a, §5). Serve só como A/B de referência.
7. **A fusão GQA `N = gqa_ratio = 6` (1 workgroup para 6 cabeças).** *Não portável para o
   prefill.* Está atrás de `N <= 8` (`ggml-vulkan.cpp:11251`) e é o caminho de **1 token**
   (decode/MTP), junto com o `split_k=32`. Nada aqui muda o prefill; se algum dia a atenção virar
   alavanca será no **decode** (64K: 19 ms/token, `docs/medicoes-banda-e-gargalos.md:370-371`), e
   aí a fusão GQA é o item.
8. **O `coopMatMulAdd` 16×16×16 como instrução.** *Portável com ressalva* — a ressalva é a mesma
   que o irmão deste documento já registra para o `mul_mm` (`docs/estudo-prefill-a-vulkan.md` §5): o
   fragmento coopmat é opaco e o layout de WMMA da RDNA4 não é o da RDNA3, então em HIP isso se
   escreve com `__builtin_amdgcn_wmma_*_w32_gfx12`. Detalhe que **este** documento acrescenta: a
   pipeline cm1 da FA exige `require_full_subgroups` com **subgroup de 64**
   (`ggml-vulkan.cpp:4769-4772`, `device->subgroup_size = 64` [M]), enquanto o nosso motor é
   **wave32** (`docs/rdna4-gfx1201-hardware-brief.md:9`) e os nossos kernels de atenção são
   wave32 por construção (`lane = tid & 31`, `dpw = head_dim/32`, `attn.cuh:330-332`) — o porte do
   tile é para wave32 (`_w32_gfx12`), **não** para o `row_split = 4` deles.

---

## 7. O que não ficou determinado

1. **Qual variante cm1 exata roda** — `flash_attn_f32_f16_cm1` (f32acc, que é o que a cadeia de
   código indica: `GGML_PREC_F32` em `llama-graph.cpp:2639` + `:11221` + `:4078`) ou a `_f16acc`.
   Não desmontei SPIR-V de FA nem rodei com `GGML_VULKAN_DEBUG`. **INFERIDO**; confirmaria com o log
   de criação de pipeline (o nome sai de `:4767`, `aligned ? "_aligned_cm1" : "_cm1"`).
2. **Se `coopmat_support_16x16x16_f32acc` é verdadeiro neste RADV** (§1.1, passo 2). Se for falso a
   FA cai em FA_SCALAR (`Br=8, Bc=32`) e **todo o §1.2/§1.5 muda**, embora a conclusão do §4.4 não
   (a FA escalar a 4K/64K medida do lado deles continua sendo o mesmo tempo). Confirmaria com
   `vulkaninfo`/`GGML_VULKAN_DEBUG`.
3. **Os 402 MB de pedidos de KV por camada** (§2.1) são aritmética minha, não medição: eu não
   instrumentei contadores de L2. **DERIVADO**; a taxa implicada (3,7 TB/s a 110 µs) é o número
   que eu gostaria de ver confirmado — se ela não fechar com o hardware, alguma premissa do meu
   modelo de leitura está errada (a mais provável: o número de workgroups simultâneos que
   compartilham a linha). Confirmaria com `rocprof`/RGP **no lado Vulkan**, que não temos.
4. **O custo real da atenção a 512 chaves no nosso lado** (§4.4): duas rotas de extrapolação
   dentro de 15 %, nenhuma medição direta. Mediria com o `bench-phases-gpu --prefill 512 --level 2`
   — que **não é executável** a 512 por limite de `hipEvent_t` (`docs/estudo-prefill-c-nosso.md`
   §8.3), ou seja: precisaria de uma corrida a 256 tokens (contexto ≤ 256) para fechar o ajuste,
   ou de um bucket novo no nível 1.
5. **O efeito do formato do KV no prefill de ponta a ponta**, dos dois lados (§5.5). Nenhuma
   medição existe; a previsão do código (0 a 512 tokens, negativo acima de 1024 para `q8_0`) está
   escrita com o `file:line` que a sustenta, mas é previsão.
6. **`n_kv_max`** é setado pelo grafo (`llama-graph.cpp:2638`) e **não é lido** pelo backend Vulkan
   (nenhuma ocorrência em `ggml-vulkan.cpp`) — o `KV` da FA é `nek1`, o tamanho da view. Registro
   porque me custou uma busca: não há caminho de "FA sobre um pedaço do cache" no Vulkan.
