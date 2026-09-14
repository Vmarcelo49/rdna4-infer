# Como o Vulkan faz o prefill (código a código)

Leitura do backend **Vulkan** do llama.cpp (`/home/marcelo/Projetos/llama.cpp`, revisão
`df03399b885831b2a1603b3abb0d8c156808e363`, backend `ggml/src/ggml-vulkan/`), comparado com o
kernel em lote deste motor. Toda referência `arquivo:linha` vale para essa revisão.
**[D]** = documentado por leitura de código, **[M]** = medido nesta máquina,
**INFERIDO** = conclusão minha não verificada, com o que a confirmaria. O irmão deste
documento (`docs/estudo-prefill-b-mmq.md`, o caminho MMQ/WMMA do `ggml-cuda`) já estabeleceu
que **o Vulkan não tem caminho int8 para os tipos IQ** (§7.3 de lá); aqui eu mostro *qual*
shader roda no lugar, com que tile, e o que isso custa em MAC por instrução e em bytes de
ativação por MAC.

## TL;DR (5 linhas)

1. Para este modelo nesta GPU o prefill **não** passa por MMQ int8: para IQ3_S o
   `mul_mmq.comp` **não é nem gerado** (`vulkan-shaders-gen.cpp:627-630` só cobre
   legacy/K-quants/mxfp4) e o caminho que roda é o **coopmat1 fp16**
   `matmul_iq3_s_f16_f16acc_cm1` — pesos dequantizados para `float16_t` na LDS e
   `coopMatMulAdd` 16×16×16 (`mul_mm.comp:339-391`, confirmado no SPIR-V).
2. O tiling é **BM=128 × BN=128 × BK=32** por workgroup de 256 threads (4 subgrupos de 64
   lanes), LDS de **24 KB**, grade de 136×4 workgroups para `m=17408 n=512 k=5120`
   (`ggml-vulkan.cpp:4524-4532`, `4922-4923`, `4809-4821`).
3. Isso dá **4096 MACs por instrução** contra **128 MACs por issue wave32 de dp4a** — 32×
   menos instruções por MAC — mas a **mesma taxa de pico** (512 MAC/CU/clk nos dois, §0.4):
   o ganho é de contagem de instruções, não de aritmética, e é isso que explica a distância.
4. Medido com `GGML_VK_PERF_LOGGER=1`: **1811 despachos por chunk de 512 tokens**, 497 deles
   `MUL_MAT` = **85,4 % do tempo**; 24,935e12 FLOPs de matmul em 380,7 ms = **32,7e12 MAC/s** =
   **33,6 %** do teto de matriz fp16 do cartão, com só 24,8 GB/s de leitura de peso.
5. Nós: **3,02e12 MAC/s** ponta a ponta (3,10 % do teto de dp4a), **36 MACs por instrução
   emitida** contra ~840-1100 do caminho coopmat, e **1,125 byte de ativação por MAC** contra
   0,0156 B — a diferença está na *forma* (tile M + LDS), não na instrução de multiplicação.

---

## 0. Método, convenções e duas correções de aritmética

### 0.1 O que foi medido aqui, e como

Nada foi compilado nem modificado. Três corridas **dentro do lock** (`scripts/gpu-lock.sh`,
`timeout` por dentro):

| comando | o que deu |
|---|---|
| `vulkaninfo` | `subgroupSize = 64` (min 32, max 64), `maxComputeSharedMemorySize = 65536`, `VK_KHR_cooperative_matrix = true`, RADV Mesa 26.2.2, RX 9070 XT gfx1201 |
| `llama-bench -p 512 -n 0 -r 1 -ngl 99 -v` | `sched_reserve: graph nodes = 3751`, `n_ubatch = 512`, 66/66 camadas na GPU |
| `GGML_VK_PERF_LOGGER=1 llama-bench -p 512 -n 0 -r 1 -ngl 99` | tabela por nó: **1811 despachos**, 447,5 ms no chunk (2ª tabela, pós-aquecimento) |

O A/B do coopmat é do coordenador (mesma máquina, `-r 2`): **baseline 1196,49 ± 1,72 tok/s**;
`GGML_VK_DISABLE_COOPMAT=1` + `GGML_VK_DISABLE_COOPMAT2=1` → **478,38 ± 0,64**;
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` → **1179,51 ± 2,88**. Números deste motor:
`docs/journal-lote.md` (Medidas 1-4), `docs/journal-kernels.md` §10 e `docs/rocm-estudo.md` §B.

### 0.2 Correção 1: são 24,35e9 MACs por token, não 30,2e9

Duas derivações independentes, e elas batem:

* **formas dos tensores do GGUF** (`Qwen3.8-27B-UD-IQ3_S.gguf`, arquitetura `qwen35`, 65 blocos):
  497 tensores 2-D de matmul no tronco somam **24,3532e9 pesos em 11,108 GB** (3,649 bit/peso);
* **tabela de despachos medida**: os 496 `MUL_MAT` do chunk cobrem exatamente essas formas
  (`Σ m·k = 24,3506e9`) mais o head, que é despachado como `MUL_MAT_VEC` com **n = 1**, não 512.

Os 30,2e9 que circulam em `docs/journal-lote.md` e `docs/estudo-prefill-b-mmq.md` estão 24 %
altos (implicariam 2,95 bit/peso num inventário de 3,65). Consequência para todo o resto deste
documento — os "TOPS" derivados caem 24 %, a razão entre os dois lados não muda:

| | MACs/token | MAC/s medido | 2×MAC (convenção dos "TOPS") |
|---|---:|---:|---:|
| llama.cpp Vulkan, só a fase de matmul (380,7 ms) | 24,3506e9 | **32,75e12** | 65,5e12 |
| llama.cpp Vulkan, chunk inteiro (447,5 ms) | 24,3506e9 | 27,86e12 | 55,7e12 |
| llama.cpp Vulkan a 1196,49 tok/s (projeção) | 24,3506e9 | 29,14e12 | 58,3e12 |
| este motor, pp512 = 123,9 tok/s | 24,3506e9 | **3,017e12** | 6,03e12 |
| este motor, só a fatia de matvec (83,7 % do tempo) | 24,3506e9 | 3,606e12 | 7,21e12 |

Ou seja: os "67 TOPS" do enunciado são **65,5e12 ops/s = 32,75e12 MAC/s**, e o nosso
"7,5 TOPS" é **6,03e12 ops/s = 3,02e12 MAC/s**. A razão é 9,1× na fase de matmul (32,75/3,61) e
9,7× ponta a ponta.

### 0.3 Correção 2: não é int8

A correção que muda a leitura: **a referência não faz um único MAC int8 nos 78 % dos bytes
deste modelo que são tipos IQ**. O
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` mexe 1,4 % porque não havia o que desligar nesses tipos
(§1.2) — o mesmo fato que o irmão deste documento achou por outro caminho
(`docs/estudo-prefill-b-mmq.md` §7.3). O que roda é fp16 em unidade de matriz. Os "TOPS" da
tabela acima são MACs, não int8.

### 0.4 Tetos do cartão (RX 9070 XT, 64 CU, 2,97 GHz de boost)

| recurso | MAC/CU/clk | MAC/s | de onde vem |
|---|---:|---:|---|
| FP32 vetorial (FMA) | 128 | 24,3e12 | 48,7 TFLOPS publicados |
| FP16 vetorial (`v_pk_fma_f16`) | 256 | 48,7e12 | 97,3 TFLOPS publicados |
| **INT8 vetorial (`v_dot4_i32_iu8`, dp4a)** | **512** | **97,4e12** | 4 MAC/lane × 128 lanes; `v_dot4` = 1,00× `v_fma_f32` [M] no *brief* §2 |
| **FP16 matriz (WMMA 16×16×16)** | **512** | **97,4e12** | 1024 FLOP/CU/clk na tabela oficial da AMD para a RDNA4 (195 TFLOPS na 9070 XT) |
| INT8 matriz (WMMA iu8) | 1024 | 194,6e12 | 2048 OP/CU/clk (389 TOPS na 9070 XT) |

O 512 MAC/CU/clk do fp16-WMMA também sai de um modelo de ciclos independente que fecha com o
mesmo número (um tile 16×64 de saída sobre 32 de K = 32 768 MACs em "64 ciclos de matmul" num
CU). **A leitura que interessa: fp16-WMMA e dp4a têm o MESMO teto de MAC neste cartão; o int8
WMMA é 2× (não 4×, não 32×).** A vantagem do WMMA está na contagem de instruções (§3.3).

---

## 1. Seleção de despacho

### 1.1 A decisão de topo: `ggml_vk_mul_mat` (`ggml-vulkan.cpp:10367-10439`)

Ordem exata das condições, com os valores deste modelo (`M` = linhas de saída, `N` = tokens):

| linha | condição | vale aqui? |
|---|---|---|
| 10378 | `needs_split = dst->ne[2]==1 && dst->ne[3]==1 && nbytes > maxStorageBufferRange` | não (maior tensor ≤ 90 MB) |
| 10400 | `ggml_vk_can_use_fwht` | não |
| 10402 | `F16 && permutado && dst->ne[1]==1` (padrão 0213) | não (peso é quantizado) |
| 10415 | `F16 && !contíguo && dst->ne[1]==1` | não |
| 10422 | `src0 ∈ {F32,F16} && dst->ne[0]==1 && dst->ne[1] > mul_mat_vec_max_cols` (AᵀB com 1 linha) | não (`M` ≫ 1) |
| **10433** | **`dst->ne[1] == 1 \|\| (dst->ne[1] <= mul_mat_vec_max_cols && src1->ne[2]*src1->ne[3] == 1)`** → `ggml_vk_mul_mat_vec_q_f16` | **não: N = 512** |
| **10436-10437** | **senão → `ggml_vk_mul_mat_q_f16(..., disable_split_k=false)`** | **sim** |

`mul_mat_vec_max_cols = 8` (`:404`). É *todo* o limiar do GEMV batelado: **até 8 tokens ele usa
o mesmo `mul_mat_vecq.comp` do decode** (peso como A, ativação como B, uma linha de saída por
warp); acima disso cai no caminho de matriz-matriz. O heurístico fino `ggml_vk_should_use_mmvq`
(`:9703-9782`, "MMVQ is generally good for batches", `:9717-9720`) **não é consultado neste
caminho** — só em `ggml_vk_mul_mat_vec_q_f16` (`:9823`) e na variante `_id` (`:10846`). Logo, a
pergunta "o que muda em n=512" tem uma resposta única: sai do `mul_mat_vecq` e entra no
`mul_mm`.

### 1.2 `ggml_vk_mul_mat_q_f16` (`:9430-9700`): qual pipeline, exatamente

```
:9476  x_non_contig = (coopmat2 && src0==F32) || !dim01_contiguous(src0)
:9478  y_non_contig = (coopmat2 && src1==F32)
                    || (coopmat_support && !coopmat2 && is_quantized(src0) && src1==F32)   // <- VALE AQUI
                    || (src0==BF16 && ...) || !dim01_contiguous(src1)
:9490  quantize_y = integer_dot_product && src1==F32 && contíguo && !y_non_contig && (ne11*ne10)%4==0
:9493  mmp_map = quantize_y ? mapa({src0->type, Q8_1}, prec) : nullptr
:9497  if (!mmp_map) mmp_map = mapa({src0->type, y_non_contig ? f16_type : src1->type}, prec)
:9514  kpad = align_size(ne10, align)          // align = 128 para o tile L (l_align, :4534)
:9515  aligned = !quantize_y && ne10 == kpad && ne01 > 8 && ne11 > 8
:9517  pipeline = guess_map(*mmp_map, ne01=17408, ne11=512, aligned)
```

Para **IQ3_S** acontece isto, e cada passo tem prova:

1. **`quantize_y` é calculado verdadeiro, mas o mapa {IQ3_S, Q8_1} está VAZIO.** A geração de
   shaders só cria `mul_mmq.comp` para legacy/K-quants/mxfp4 (`vulkan-shaders-gen.cpp:627-630`),
   e o mesmo filtro vale para o GEMV (`:1325`). O artefato compilado confirma: existem
   `matmul_iq3_s_f16*.spv` e `mul_mat_vec_iq3_s_f32_f32*.spv`, e **nenhum** `*iq3_s*q8_1*.spv`
   (`build/ggml/src/ggml-vulkan/vulkan-shaders.spv/`). Então `mmp_map = nullptr` e
   `quantize_y` volta a `false` (`:9498`) — é exatamente o que o A/B do coordenador mede.
2. **`y_non_contig` é verdadeiro**, por `:9480-9481`: `coopmat_support && !coopmat2 &&
   is_quantized(src0)` com `src1 == F32`. O comentário na linha é literal: *"coopmat1: force
   f32->f16 conversion so the f16 B-type quant pipeline is used"* — **as ativações são
   convertidas para fp16 por um despacho separado** (`ggml_vk_cpy_to_contiguous`, `:9653`,
   cacheado por `prealloc_y_last_tensor_used`, `:9647-9657`) antes da matmul.
3. Com `y_non_contig` e `src0` quantizado, o mapa passa a ser **{IQ3_S, F16}** (`:9497`).
4. `f16acc` (`ggml_vk_get_mul_mat_mat_f16acc`, `:9117-9131`): tipo quantizado com
   `coopmat_support && !coopmat2` → `fp16 && coopmat_acc_f16_support && prec==DEFAULT` (`:9127-9129`).
   A chave vira `{IQ3_S, F16, mul_mat_id=false, f16acc=true}`.
5. `aligned`: `k = 5120` e `17408` são múltiplos de `l_align = 128` ⇒ `kpad == ne10`; `M > 8`,
   `N = 512 > 8` ⇒ **`aligned = true`** ⇒ pipeline *aligned* (`LOAD_VEC_B = 8`, 16 B por carga).
6. `split_k` (`ggml_vk_guess_split_k`, `:9023-9062`): `k ≥ 2048` mas `m_tiles*n_tiles = 136*4 = 544`
   > `shader_core_count` ⇒ **`split_k = 1`**; o despacho é
   `{CEIL_DIV(m,128), CEIL_DIV(n,128), 1}` (`ggml_vk_dispatch_pipeline` divide pelos `wg_denoms`
   da pipeline) = **544 workgroups** por matmul de 17408×512×5120.

### 1.3 O tile que sai: `l_warptile_mmq` com o override AMD+RADV (`:4524-4528`)

```
:4524  } else if (vendor_id == VK_VENDOR_ID_AMD && coopmat_support && driver_id != AmdProprietary) {
:4526      l_warptile     = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
:4527      l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
:4528      l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
```

É o único ponto do arquivo que muda o tiling por "AMD com coopmat em driver livre", e é o que
dá os números do prefill. Campos (`:4407-4410`): `[0]=BLOCK_SIZE, [1]=BM, [2]=BN, [3]=BK,
[4]=WM, [5]=WN, [6]=WMITER, [7]=TM, [8]=TN, [9]=TK, [10]=WARP`. Para esta GPU:
`subgroup_size = 64` (vulkaninfo [M]) ⇒ `mm_warp_8 = min(max(64,8),64) = 64` (`:4393`);
`coopmat_m/n/k = 16/16/16` (**INFERIDO**, §7) ⇒ `tm_*=tn_*=tk_*=16`:

| warptile | BLOCK_SIZE | BM | BN | BK | WM | WN | WMITER | TM | TN | TK | WARP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **`l_warptile_mmq` (o que roda)** | 256 | 128 | 128 | 32 | 64 | 64 | 2 | 16 | 16 | 16 | 64 |
| `m_warptile_mmq` | 128 | 64 | 64 | 32 | 64 | 32 | 2 | 16 | 16 | 16 | 64 |
| `s_warptile_mmq` | 64 | 32 | 32 | 32 | 32 | 32 | 2 | 16 | 16 | 16 | 64 |

E a escolha s/m/l é a função em `:5447-5454`:

```
:5449  if (configs.size() <= 1) return 0;
:5450  if (m <= 32 || n <= 32) return 0;
:5451  if (configs.size() == 2) return 1;
:5452  if (m <= 64 || n <= 64) return 1;
:5453  return configs.size() - 1;          // -> o tile L
```

Com `m ∈ {1024..17408}` e `n = 512`, e as três configurações existindo (o teste de LDS em
`:4557-4565` passa — §2.3), o índice é **2 = L**, e os `wg_denoms` do L são `{128,128,1}` (`:4531`).

### 1.4 Quando o coopmat NÃO existe

Com `GGML_VK_DISABLE_COOPMAT=1` (que reproduz o que acontece sem `VK_KHR_cooperative_matrix`):

* `coopmat_support = false` (`:6559`, `:6585`, desligado em `:7275`), `coopmat2 = false`;
* o `else` de `:4472` monta `tm_*=4, tn_*=4, tk_*=1` (`:4474-4482`), e o override AMD de `:4524`
  **não se aplica** (exige `coopmat_support`) ⇒ `l_warptile_mmq = {128, 128, 128, 32, 128, 64, 2,
  4, 4, 1, 64}`;
* `ggml_vk_get_mul_mat_mat_f16acc` cai em `:9130` (só `fp16 && prec`) ⇒ a chave continua
  `{IQ3_S, F16, false, true}`;
* as pipelines vêm de `sg_create`/`sg_create_mmq` (`:5175-5230`) e, como este cartão expõe
  `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2` no log do backend, `:6616-6618`), a
  variante é **`matmul_iq3_s_f16_dot2_f16acc.spv`**;
* no shader, `#ifdef COOPMAT` é falso ⇒ roda `mul_mm.comp:392-427`: cargas da LDS em registrador
  e `dot_product()` (`dot_product_funcs.glsl:4-13` → `v_dot2_f32_f16`, 2 MACs por instrução)
  sobre `ACC_TYPEV2 sums[WMITER*TM*WNITER*TN/2]`, que com esses warptiles dá `sums[2*4*4*4/2] = 64`
  acumuladores `f16vec2` = **128 valores de saída por thread**.

Medido: **478,38 tok/s contra 1196,49** — os matrix cores valem **2,5×** neste prefill.

### 1.5 Resumo da seleção para este modelo/GPU

| tipo do peso | n | chave da pipeline | shader |
|---|---:|---|---|
| IQ3_S / IQ4_XS / IQ3_XXS / IQ2_* / IQ1_S (77,6 % dos bytes) | 512 | `{tipo, F16, f16acc=true}` tile L | `matmul_<tipo>_f16_f16acc_cm1.spv` (`mul_mm.comp` + `COOPMAT`) |
| Q*_K (quantizado, e `mul_mmq` **é** gerado para eles) | 512 | `{tipo, Q8_1}` tile L | `matmul_q4_k_q8_1.spv` (`mul_mmq.comp`, int8 de verdade) |
| `token_embd` (saída, 1 token) | 1 | mapa mul_mat_vec | `mul_mat_vec_q5_K_...` (`MUL_MAT_VEC` na tabela medida) |

(Não há nenhum `MUL_MAT` de F32/F16 neste grafo: as normas e o `ssm_a` são 1-D e as projeções são
todas quantizadas.)

**O mesmo chunk mistura os dois mundos**: os k-quants vão por int8/MMQ e os LUT quants (78 % dos
bytes deste modelo) vão por fp16/coopmat. É por isso que
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` não é um controle limpo neste modelo — ele só desliga o
caminho minoritário (o irmão deste documento quantifica isso em §7.3 de lá: ~25 % dos bytes têm
as duas famílias).

---

## 2. O shader que roda para IQ3_S em n=512

### 2.1 Identificação (cadeia de evidências)

1. `mul_mmq.comp` não existe para `iq3_s` (§1.2) ⇒ não há caminho int8;
2. as pipelines cm1 são criadas em `:5085-5094` (`X_CM1` → `FOR_EACH_LUT_TYPE_NONFP4`) com
   `tc_mmq` e `create_aligned=true, require_full_subgroups=true` (`:5050`);
3. o nome do artefato é montado em `vulkan-shaders-gen.cpp:437`:
   `name + (f16acc?"_f16acc":"") + (coopmat?"_cm1":"")` ⇒ `matmul_iq3_s_f16_f16acc_cm1.spv`;
4. o SPIR-V desse arquivo comprova o tipo da multiplicação:

```
$ spirv-dis matmul_iq3_s_f16_f16acc_cm1.spv | grep -E "OpTypeCooperativeMatrix|MulAdd"
%1167 = OpTypeCooperativeMatrixKHR %half %uint_3 %TM %TN %uint_2   ; C = half, Use=Accumulator
%1266 = OpTypeCooperativeMatrixKHR %half %uint_3 %TM %TK %uint_0   ; A = half, Use=MatrixA
%1300 = OpTypeCooperativeMatrixKHR %half %uint_3 %TK %TN %uint_1   ; B = half, Use=MatrixB
%1341 = OpCooperativeMatrixMulAddKHR %1167 %1333 %1334 %1340
```

`%uint_3` é `Subgroup` e `%half` é o tipo dos **dois operandos** (e do acumulador, na variante
`f16acc`). Se `coopmat_acc_f16_support` fosse falso, a pipeline carregada seria
`matmul_iq3_s_f16_cm1.spv` (acumulador `float`): os dois são a MESMA multiplicação 16×16×16
fp16×fp16. **(INFERIDO**: a variante exata depende das propriedades KHR que o RADV reporta;
confirmaria com um build `GGML_VULKAN_DEBUG` lendo o `VK_LOG_DEBUG` de `ggml_vk_create_pipeline`,
ou com um capture de debug-utils, que nomeia cada pipeline.)

### 2.2 Geometria, grade e trabalho por thread

| grandeza | valor | onde |
|---|---|---|
| workgroup | 256 threads = 4 subgrupos | `l_warptile_mmq[0]=256`, `[10]=64` |
| tile do workgroup | 128 (M) × 128 (N) × 32 (K) | `[1],[2],[3]` |
| tile por subgrupo | 64×64 saídas = `(WM/TM)×(WN/TN)` = 4×4 coopmats de 16×16 | `mul_mm.comp:248-249`, `:342` |
| grade | `(m/128) × (n/128)`; para 17408×512 = **136×4 = 544 workgroups** | `:4531`, `ggml_vk_dispatch_pipeline` |
| laço K | 5120/32 = **160 iterações**; 17408×512×5120 = 45,6e9 MACs por despacho | `mul_mm.comp:362` |
| por thread | 16 384 saídas/256 = 64 saídas; 16 coopmats no registrador do subgrupo | `:342` |
| MACs por workgroup | 128×128×5120 = 83,9e6 | — |

### 2.3 LDS: 24 KB, byte a byte

Do SPIR-V (os tamanhos são `OpSpecConstantOp`, avaliados nos valores do §1.3):

| array | fórmula | valor |
|---|---|---:|
| `buf_a` | `BM × (BK/2 + SHMEM_STRIDE_PAD)` × 4 B = 128 × 20 × 4 | 10 240 B |
| `buf_b` | `BN × (BK/2 + SHMEM_STRIDE_PAD)` × 4 B = 128 × 20 × 4 | 10 240 B |
| `coopmat_stage` | `TM × TN × (BLOCK_SIZE/WARP)` × 2 B = 16×16×4×2 | 2 048 B |
| `iq3s_grid` | `uint32_t[512]` | 2 048 B |
| **total** | | **24 576 B** (de 65 536) |

`SHMEM_STRIDE_PAD = 4` para coopmat não-Intel (`mul_mm.comp:182-187`; o host só empurra as
constantes 12/13 para Intel, `:4813-4821`) ⇒ `SHMEM_STRIDE = BK/2 + 4 = 20` (`:192`), o padding
que desalinha os bancos nos acessos de 128 bits da coopmat. O `coopmat_stage` é o palco na LDS
para o store; o caminho alinhado nem o usa (`:481-484`). **Nota**: a checagem do host
(`:4196-4201`) estima `coopmat_stage = TM*TN/warps*sizeof(float) = 256 B` e portanto subestima o
shader em 1 792 B; não muda nada aqui (24,6 KB < 65,5 KB), mas é uma divergência real entre o
modelo de LDS do host e o shader.

### 2.4 Dequantização do peso: grid de 512 entradas na LDS, 4 pesos por lookup

`mul_mm_funcs.glsl:238-260` é o corpo para `DATA_A_IQ3_S`:

```glsl
const uint ib = idx / 64;            // bloco IQ3_S = 256 pesos = 64 grupos de 4
const uint iqs = idx % 64;
const uint iqh = iqs / 8;
const float d = float(data_a[ib].d);
const uint qs = data_a[ib].qs[iqs];
const uint qh = data_a[ib].qh[iqh];
const int8_t sign = int8_t(data_a[ib].signs[iqs / 2] >> (4 * (idx % 2)));
const uint scale = data_a[ib].scales[iqs / 16];
const float db = d * (1 + 2 * ((scale >> (4 * (iqh & 1))) & 0xf));   // escala do grupo de 32
const uint32_t grid = iq3s_grid[qs | ((qh << (8 - (iqs % 8))) & 256)];
const vec4 v = db * vec4(unpack8(grid));                            // 4 pesos de uma vez
store_a(col, k_pair,     FLOAT_TYPEV2(sign&1?-v.x:v.x, sign&2?-v.y:v.y));
store_a(col, k_pair + 1, FLOAT_TYPEV2(sign&4?-v.z:v.z, sign&8?-v.w:v.w));
```

* A tabela `iq3s_grid[512]` mora **na LDS** (`types.glsl:1576`: `shared uint32_t iq3s_grid[512]`),
  copiada de `iq3s_grid_const[512]` (`types.glsl:1679-1745`) por `init_iq_shmem`
  (`types.glsl:1746-1758`, chamada em `mul_mm.comp:219-221`): **2 048 B por workgroup**,
  amortizados por 160 iterações de K. O `iq_shmem_init.glsl` deste checkout é um stub vazio, mas
  ele só é incluído no caminho `MULMAT_QUANT` (`mul_mm.comp:206-208`) — nos shaders por tipo
  (o nosso caso) quem vale é a definição do `types.glsl`. **Verificado no artefato**: o SPIR-V
  tem `OpStore` com `MakePointerAvailable` para `%iq3s_grid` (a cópia) e `OpLoad` com
  `MakePointerVisible` no `iq3s_grid[...]` do `mul_mm_funcs.glsl:253` (o lookup).
* **4 pesos por lookup**: índice de 9 bits (8 de `qs` + 1 de `qh`) → um `uint32` → `unpack8` →
  `vec4`.
* Custo por thread por tile de K: `loadstride_a = 256 × LOAD_VEC_A_EFF(4) / BK(32) = 32` ⇒ o laço
  `for (l = 0; l < BM; l += loadstride_a)` (`:363`) dá **4 iterações**, 4 pesos cada ⇒ **16 pesos
  por thread por tile de K** (≈ 6,9 B lidos do global e 8 stores de `v2half` na LDS).
* **A escala do grupo de 32 é dobrada no valor** (`db = d*(1+2*scale)`), não corrigida depois: o
  que entra na LDS já é o peso em fp16 com a escala aplicada. É o oposto do MMQ int8 (que
  acumula em int32 e corrige no epílogo) e é o que dispensa todo o aparato de escalas — ao custo
  do arredondamento para fp16.

### 2.5 Ativação: fp16, 16 B por carga

* `LOAD_VEC_B = load_vec = fp16 ? "8" : "4"` (`vulkan-shaders-gen.cpp:462`) e, com `ALIGNED=1`,
  `LOAD_VEC_B_EFF = 8` (`mul_mm.comp:276`) ⇒ `load_b_to_shmem` faz
  `FLOAT_TYPEV8 bb = FLOAT_TYPEV8(data_b[idx])` = **8 halves = 16 B por carga**
  (`mul_mm_funcs.glsl:635-646`) e despeja 4 `v2half` na LDS.
* `B_TYPE = f16mat2x4` (`:463`): a ativação é fp16 **na memória**; a conversão f32→f16 é um
  despacho separado, uma vez por tensor de ativação (§1.2).
* `LOAD_VEC_A_EFF = LOAD_VEC_A = lut_load_vec_a("iq3_s") = 4` (`:633`, `:258-262`).
* Tráfego de B por workgroup: `BN × K × 2 B` = 128 × 5120 × 2 = 1,31 MB; como há 136 tiles em M,
  **cada elemento de ativação é lido 136 vezes por matmul** ⇒ 0,0156 byte por MAC (§3.6).

### 2.6 Acumulador, store e escalas

* Acumulador: `coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[16]`
  (`mul_mm.comp:342`). Com `f16acc`, `ACC_TYPE = float16_t` (`vulkan-shaders-gen.cpp:483`), e o
  `OpCooperativeMatrixMulAddKHR` do SPIR-V acima tem tipo `%half`. Sem `f16acc`, seria `float`.
  **Não há acumulador fp32 na variante carregada** — se isso importa para a precisão do nosso
  prefill, é um A/B a medir (a referência aceita, e o irmão deste documento discute o assunto do
  lado f16 em §7.4 de lá).
* Store: o caminho `is_aligned && is_in_bounds` (`:481-484`) converte o coopmat para `D_TYPE`
  (float) e faz `coopMatStore` direto no buffer de saída com `stride_d`, sem passar pela LDS. Os
  outros dois caminhos (stride não alinhado; tile parcial) usam `coopmat_stage` +
  `controlBarrier(Subgroup, ...)`.
* Escala pós-mma: **nenhuma** — a escala do grupo de 32 e o `d` do super-bloco já estão no valor
  fp16 de `buf_a` (§2.4).

---

## 3. Nível de instrução

### 3.1 De GLSL/SPIR-V para gfx12

`OpCooperativeMatrixMulAddKHR` com `Scope=Subgroup` e operandos fp16 de 16×16×16 é baixado pelo
LLVM/AMDGPU para a família `v_wmma_*_16x16x16_*`; o RADV expõe `VK_KHR_cooperative_matrix` em
RDNA3+ apoiado em WMMA, e a AMD chama a instrução de "Wave Matrix Multiply Accumulate". Para
este shader o candidato é `v_wmma_f16_16x16x16_f16` (acumulador fp16) ou
`v_wmma_f32_16x16x16_f16`. **INFERIDO** (não desmontei a ISA do shader do RADV): confirmaria com
`RADV_DEBUG=spirv` + disassembly do pipeline, ou com `rocprof`/RGP; o irmão HIP já tem a
evidência do lado HIP (`docs/estudo-prefill-b-mmq.md:224`: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
em `mma.cuh:1324-1325` no gfx1201).

No lado vectorial, o caminho **sem** coopmat usa `v_dot2_f32_f16` (`dot_product_funcs.glsl:4-13`,
habilitado por `DOT2_F16` = `VK_VALVE_shader_mixed_float_dot_product`, que este driver expõe), e o
nosso kernel usa `v_dot4_i32_iu8` via `__builtin_amdgcn_sudot4` (`vecdotq.cuh:84-88`).

### 3.2 MACs por instrução

| instrução | MACs por instrução (por wave) | MACs por lane |
|---|---:|---:|
| `v_dot4_i32_iu8` (wave32) | 4 × 32 = **128** | 4 |
| `v_dot4_i32_iu8` (wave64) | 4 × 64 = **256** | 4 |
| `v_dot2_f32_f16` (wave32) | 2 × 32 = 64 | 2 |
| `v_wmma_f16_16x16x16_f16` | 16·16·16 = **4096** | 64 (= 4096/64) |

Razão **32× por issue wave32** (128 → 4096) e 16× por issue wave64 — e 1024× se alguém comparar
"4 MACs do dp4a" (por lane) com "4096 do WMMA" (por wave), que é a comparação errada que aparece
em alguns lugares do repo.

### 3.3 O que isso vale em ciclos (e por que 32× não é 32×)

Pelos tetos do §0.4, em ciclos de CU por 4096 MACs:

| caminho | MAC/CU/clk | ciclos de CU por 4096 MACs | instruções por 4096 MACs |
|---|---:|---:|---:|
| dp4a (wave32) | 512 | 8 | 32 |
| dp4a (wave64) | 512 | 8 | 16 |
| **fp16 WMMA 16×16×16** | **512** | **8** | **1** |
| int8 WMMA 16×16×16 | 1024 | 4 | 1 |

**O WMMA fp16 tem exatamente a mesma taxa de MAC que o dp4a neste cartão** (512 MAC/CU/clk). O
ganho é de **contagem de instruções** — 1 contra 32 para o mesmo trabalho — e portanto de quanto
overhead (endereço, carga, espera) cabe no mesmo orçamento de issue. O caminho int8 (novo na
RDNA4) é 2× em taxa e continua 1 instrução por 4096 MACs.

**INFERIDO**: se o WMMA e o caminho vectorial compartilham a porta de issue (é o que a leitura de
terceiros sobre a RDNA4 afirma), então 8 ciclos de CU por instrução de matriz são 8 ciclos em que
a unidade vectorial não emite — e a vantagem líquida de *taxa* cairia a zero, sobrando o efeito
de contagem de instruções. O que separa as duas leituras é um microbench de WMMA no cartão
(medir dp4a/s com e sem WMMA intercalado), que `docs/journal-lote.md` §"o que sobra" já nomeia
como pré-requisito.

### 3.4 A aritmética dos dois lados

A base do enunciado, `64 CU × 4 SIMD × 2,5 GHz = 640e9` slots de issue/s, é "256 SIMD32 × 2,5 GHz",
isto é **slots de wave32**. (O *brief* mede `rocminfo` dando 128 SIMD32 × 2,4 GHz = 307e9; as duas
contagens dão o mesmo teto de MAC — muda o percentual, não a conclusão.)

* 1 slot wave32 saturado de dp4a = 128 MACs ⇒ teto **81,9e12 MAC/s a 2,5 GHz** = **97,3e12 a
  2,97 GHz**, que é exatamente o número publicado da placa. A conta do enunciado fecha.
* Nós: 3,61e12 MAC/s na fatia de matvec (3,02e12 ponta a ponta) ⇒ **3,7 %** do teto de dp4a
  (3,1 % ponta a ponta); em slots, `3,61e12/128 = 28,2e9` de 640e9 = **4,4 %**.
* A referência: 32,75e12 MAC/s ÷ 512 MAC/CU/clk = 64,0e9 ciclos de CU/s = **33,6 %** do teto de
  matriz fp16 (que é o mesmo número do teto de dp4a); se fosse int8 WMMA, 16,8 %.
* Sem olhar taxa de pico, **MACs por instrução emitida**: nós, `675e6` instruções de warp por
  token (`journal-kernels.md` §10, medido na ISA) para 24,35e9 MACs = **36 MACs/instrução**
  (o irmão deste documento usou 30,2e9 e chegou a 44,7; o número corrigido é 36,1). A referência,
  por tile de K e por subgrupo: 32 `coopMatMulAdd` (32 × 4096 = 131 072 MACs) contra ~72
  instruções do laço cooperativo (`mul_mm.comp:380-391`: 4 cargas de A + 4×(carga de B + mma) por
  metade de K) mais ~78 de staging (`:362-372`: 4 iterações de A de ~17 instruções, 2 de B de ~5)
  ⇒ **~840-1 100 MACs por instrução emitida**. Razão: **23-30×**. É aí, e não no pico de MAC, que
  estão os 9,1×.

### 3.5 Por que a nossa conta não fecha com "issue saturado"

No lote, `docs/journal-lote.md` mede o custo marginal de 6,04 ms/token e
`docs/journal-kernels.md` §10 conta `25,3e6` iterações de warp × ~428 instruções = `10,8e9`
instruções de warp por chunk de 16 tokens (por token: `675e6`). Isso dá `98-112e9` instruções/s
(15 % se contadas contra a passagem inteira de 110,7 ms, 17,5 % contra o custo marginal de
6,04 ms/token) contra 640e9 slots. O dp4a sozinho: `24,35e9/4 = 6,09e9` dp4a por lane por token
= `95,1e6` dp4a de warp (wave64) por token = **14 % das 675e6 instruções**. Não é saturação de
issue, não é DRAM (o lado do peso roda a 100 GB/s de 633), não é registrador
(`localSizeBytes = 0` em 14 tipos × 4 N) e não é o UNROLL (medido, refutado). O que sobra é
latência não coberta — e uma conta de tráfego que a referência simplesmente não tem.

### 3.6 A conta que separa os dois: bytes de ativação por MAC

| | bytes de ativação por MAC | por token |
|---|---:|---:|
| llama.cpp Vulkan (tile de M = 128) | 2 B / 128 = **0,0156** | 0,38 GB |
| este motor (1 linha de saída por warp, WPR=1) | 36 B / 32 = **1,125** | 27,4 GB |

Contas: nós lemos o bloco `q8_1` inteiro (36 B por 32 valores) **uma vez por linha de saída** ⇒
`m×n×k/32 × 36 B` por passagem = 1,125 B por MAC; 1,125 × 24,3506e9 × 16 tokens = **438 GB por
chunk de 16 tokens**, em 110,7 ms ⇒ **3,96 TB/s** de pedidos de carga (o peso, no mesmo
intervalo, é 100 GB/s). A referência lê a ativação **uma vez por tile de 128 linhas de saída** ⇒
0,0156 B/MAC = `24,935e12 FLOPs/128` = **194,8 GB por chunk de 512 tokens**, em 380,7 ms ⇒
**512 GB/s**. Razão por MAC: **72×**.

A nossa própria Medida 2 (`act_stride = 0`, 95 % do tempo) mostra que *tornar a ativação
residente* vale só 5 %, então a leitura honesta é a que o `journal-lote.md` já registrou: a conta
não fecha com nenhum recurso saturado, e o candidato do tamanho certo é L2/Infinity Cache a
~4 TB/s de pedidos (3× o que a IC entrega). O que este documento acrescenta é o número de
referência: **a forma que a referência usa reduz esse tráfego 72× sem mudar uma linha de
aritmética** — o tile de M, não a instrução.

---

## 4. Despachos e fusão por chunk de 512 tokens

### 4.1 Medido (`GGML_VK_PERF_LOGGER=1`, 2ª tabela = sem compilação de pipeline no meio)

* **1811 despachos** para o chunk de 512 tokens inteiro (um único grafo; `n_ubatch = 512`).
* **85,4 % do tempo em `MUL_MAT`**: 497 despachos = 382,2 ms de 447,5 ms.
* Repartição do resto (ms): `GATED_DELTA_NET` 12,3 · `GLU` 12,0 · `CONCAT` 10,2 ·
  `RMS_NORM_MUL` 9,9 (209 despachos em 4 formas) · `ADD` 5,1 · `SSM_CONV_SILU` 3,0 ·
  `RMS_NORM` 2,8 · `SILU_MUL` 2,1 · `FLASH_ATTN_EXT` 1,8 · `ROPE` 1,6 · `SCALE` 1,4 ·
  `MUL_MAT_VEC` 1,4 (o head, 1 token) · `SIGMOID_MUL` 0,7 · `GET_ROWS` 0,6 · `CPY` 0,5 ·
  `SOFTPLUS_MUL` 0,3 · `SET_ROWS` 0,2 · `SIGMOID` 0,1.
* Por tipo de peso, dentro do `MUL_MAT` (ms): **iq3_s 134,5 (127 despachos)** · iq3_xxs 78,1 (77) ·
  iq4_xs 77,4 (88) · iq2_s 23,9 (21) · q3_K 20,8 (15) · iq2_xxs 14,0 (12) · iq2_xs 13,5 (12) ·
  q4_K 7,3 (23) · q2_K 4,5 (6) · q5_K 2,6 (15) · iq1_s 2,1 (2) · q8_0 1,8 (96) · q6_K/iq4_nl 0,2.
* GFLOPS/s por despacho (2ª tabela): `iq3_s` **53,4-74,0e12** (o menor é o `k=17408`, o maior o
  `12288×512×5120`); `iq4_xs` 52,4-74,6e12; `iq3_xxs` 57,1-73,3e12; `q4_K` 48,3-63,2e12;
  `q3_K` 43,6-52,3e12 (é o k-quant mais lento, consistente com os 208 instruções por `vec_dot`
  que `docs/rocm-estudo.md` §B.3 mede do nosso lado).

Duas leituras que interessam para a nossa fila: **`FLASH_ATTN_EXT` custa 0,4 %** e
`GATED_DELTA_NET` 2,8 % — todo o resto do prefill desta referência é matmul. E a taxa de peso é
baixíssima: 11,108 GB em 447,5 ms = **24,8 GB/s** (29,2 GB/s contando só o tempo de matmul),
contra os 633 GB/s de roofline do cartão.

### 4.2 O que a fusão remove

O pass está em `ggml_backend_vk_graph_compute` (`:18144-18340`), com as decisões em
`:18150-18314`. Padrões existentes e o que a tabela medida mostra de fato:

| padrão do llama.cpp | linha | despachos medidos | no nosso grafo? |
|---|---|---:|---|
| `MUL_MAT + ADD [ + ADD]` → `MUL_MAT_ADD[_ADD]` | `:18155-18165` | dentro de `MUL_MAT` | **não**: matvec + `add_launch` (`graph.cuh:1316`, `:1334`) |
| `RMS_NORM + MUL` → `RMS_NORM_MUL` | `:18220-18227` | 209 (4 formas) | **sim**: a norma aplica o peso num lançamento só |
| `UNARY(SILU) + MUL` → `SILU_MUL` | `:18228-18237` | 48 | **não**: 2 launches (SwiGLU do FFN, `graph.cuh:1329-1330`) |
| `UNARY(SIGMOID) + MUL` → `SIGMOID_MUL` | `:18232` | 16 | **não**: 2 launches (gate da atenção, `:1137-1138`) |
| `SOFTPLUS + MUL` → `SOFTPLUS_MUL` | `:18234` | 48 | **não**: 2 launches (alpha do GDN, `:1225-1226`) |
| `SSM_CONV (+ADD+SILU)` → `SSM_CONV_SILU` | `:18238-18248` | 48 | **parcial**: `conv1d_state_batch_launch` (`:1232`) |
| `RMS_NORM+MUL+ROPE(+VIEW+SET_ROWS)` | `:18182-18202` | 16 + 16 | **não**: norma e rope em lançamentos separados (`:1320`, `:1075-1076`) |
| `MUL_MAT_ID + …`, `TOPK_MOE…`, `TOPK_QSA` | `:18166-18314` | 0 | não (não temos MoE) |
| `MULTI_ADD` (`ggml_vk_fuse_multi_add`) | `:18150-18154` | — | não |
| `RMS_NORM + MUL + ADD_MUL` | `:18203-18212` | — | não |

A fusão leva de **3751 nós de grafo** (medido: `sched_reserve: graph nodes = 3751`) a **1811
despachos**, removendo ~52 % dos nós. Quatro dos padrões que a referência usa e nós não
(`SILU_MUL`, `SIGMOID_MUL`, `SOFTPLUS_MUL`, `MUL_MAT+ADD`) são exatamente a diferença de
contagem da §4.3.

### 4.3 A nossa contagem, para comparar

Contagem estática de call-sites de lançamento no caminho em lote (`include/rdna4/graph.cuh`):

* **camada GDN** (48): `rms_norm` (1040) + `proj_batch` com `act_ready=false` no primeiro e no
  último (1205 → 2, 1262/1310 → 2) + `quantize_batch` (1206) + 3 `proj_batch` com
  `act_ready=true` (1207-1209) + `unary`/`add_bcast`/`unary`/`mul_bcast` (1221-1226) +
  `conv1d_state_batch` (1232) + `l2_norm` (1238) + `delta_rule_batch` (1243) + `rms_norm` (1249)
  + `unary`+`mul` (1254-1255) + `add` (1316) + `rms_norm` (1320) + gate/up (1327-1328) +
  `unary`+`mul` (1329-1330) + `proj_batch` down (1332 → 2) + `add` (1334) ⇒ **29 lançamentos**;
* **camada de atenção** (16): os mesmos 10 da cauda + norma, 3 projeções, quantização,
  deinterleave, 2 normas de q/k, 2 ropes, `kv_write_batch` (que faz 2 launches, `:873-874`),
  atenção, sigmoid + mul ⇒ **28**;
* por chunk de 16 tokens: `48×29 + 16×28 + cabeça` ≈ **1 860 lançamentos**;
* o CLI processa o prompt em chunks de **16** (`kMaxBatch = 16`, `graph.cuh:189`; `prefill_ids`
  em `src/main.hip:165-180` monta chunks de 16/8/4/3/2) ⇒ **512 tokens = 32 chunks ≈ 59 500
  lançamentos** para o mesmo trabalho que a referência faz em **1 811**.

Por token: **116 lançamentos** contra **3,5**. Com o gap de despacho medido neste cartão
(**3,5 µs**, `docs/rocm-estudo.md` §B.1), são `59 500 × 3,5 µs = 208 ms` = **5,0 %** dos nossos
4,13 s. Não é a causa dos 9×, mas é 5 % que a referência não paga — e o "~2200 lançamentos/token"
que circula no repo é do caminho **por token**; no caminho em lote são ~116/token.

---

## 5. O que NÃO é transferível

| peça | por quê |
|---|---|
| **`OpCooperativeMatrixMulAddKHR` / a abstração coopmat** | O KHR coopmat entrega um fragmento **opaco**: você escreve `coopMatLoad(cache_a, buf_a, offset, stride, RowMajor)` e o driver decide qual lane tem qual elemento (`mul_mm.comp:383-388`). Em HIP não existe essa camada: `__builtin_amdgcn_wmma_*_w32_gfx12` recebe registradores com **layout explícito**, e na RDNA4 esse layout **mudou** em relação à RDNA3 (a AMD avisa que não há compatibilidade). Porta-se a *forma* do tile, não o shader. |
| **`ALIGNED`, `SHMEM_STRIDE_PAD`, `APPLY_SLM_A_RESHAPE` como specialization constants** | `mul_mm.comp:51`, `:182-187` — um shader com N variantes resolvidas na criação da pipeline (`ggml-vulkan.cpp:4813-4832`). Em HIP isso vira template/`#define`; e a variante `APPLY_SLM_A_RESHAPE` existe só para o driver proprietário da Intel (`:4815-4819`). |
| **O tuning por `vendor_id`/`driver_id`** | O override que dá o tile 128×128 existe **apenas** para `AMD && coopmat && driver != AMD-proprietary` (`:4524`), e a pinagem de subgrupo é Intel-only (`:5039`). Não há equivalente em HIP: é tabela por arquitetura e driver. |
| **`require_full_subgroups` implicando 4 subgrupos** | `mul_mm.comp:244` usa `gl_SubgroupID`, e `:263-264` calcula `warp_r = warp_i % (BM/WM) = %2`, `warp_c = warp_i / 2`; com `BM/WM = 2` e `WN/TN = 4` a conta só fecha com **exatamente 4 subgrupos** ⇒ o shader exige `BLOCK_SIZE/64 = 4`, isto é, **wave64** (o `subgroupSize = 64` do vulkaninfo [M]). O nosso motor é wave32 (o *brief* mede wave32 no `rocminfo`). O mesmo shader em wave32 daria `gl_SubgroupID ∈ [0,8)` e escreveria fora do tile. Não é um detalhe de porte: é a razão pela qual o caminho WMMA do `ggml-cuda` usa explicitamente a variante `_w32_gfx12` e um `tile<>` com layout de wave32 (`mma.cuh:1324-1325`, coberto em `docs/estudo-prefill-b-mmq.md` §3.2). |
| **A tolerância a fp16 nos pesos** | O caminho coopmat **dequantiza para fp16 e guarda a escala no valor** (`mul_mm_funcs.glsl:246-254`); isso só é aceitável porque não há correção depois. Quem quiser int8 (para pegar os 2× da RDNA4) tem de reintroduzir a correção por bloco de 32 — o "descale tax" que o irmão deste documento mede do lado HIP (`docs/estudo-prefill-b-mmq.md` §6.4 e §7.6). Os dois documentos chegam à mesma conclusão por caminhos diferentes. |
| **`GGML_VK_DISABLE_*` como controle experimental** | Neste modelo o `DISABLE_INTEGER_DOT_PRODUCT` mexe 1,4 % porque **os tipos IQ não têm pipeline int8** (§1.2), não porque "int8 não ajuda no prefill". Ler esse A/B como uma medida do valor do int8 seria um erro. |

O que **é** transferível sem ressalva: a estrutura de três níveis (tile de saída na LDS → carga em
registrador de fragmento → uma instrução de matriz), o staging do peso dequantizado com a LUT
compartilhada na LDS, o tile de K = 32 (o bloco de quantização), o padding de 4 `v2half` na
stride da LDS, e o fato de a ativação ser lida uma vez por **tile de M** em vez de uma vez por
linha de saída.

---

## 6. O que precisaríamos ter no HIP para empatar

Ordenado por ganho esperado. Cada item diz de que linha do llama.cpp ele vem e o que custa.

1. **Tile de saída com o peso dequantizado na LDS e a ativação lida uma vez por tile de M**
   (a forma `mul_mm`, não a forma `mul_mat_vec`). *Portável.*
   Vem de `mul_mm.comp:362-427` (staging de A e B na LDS + laço cooperativo) e de
   `ggml-vulkan.cpp:4524-4532` (BM=128, BN=128, BK=32). No nosso lado,
   `matvec_kernel_gen`/`matvec_kernel_batch` (`matvec.cuh:322-430`, `:447-543`) hoje têm
   **1 elemento de saída por thread por passo de k**, `ROWS=8` linhas por CTA com `WPR=1`, e leem
   o bloco `q8_1` da ativação **uma vez por linha de saída** — 1,125 B/MAC contra 0,0156 B da
   referência (§3.6), 72×. É uma família de kernel nova (buffer LDS para o peso dequantizado +
   acumulador por thread + redução no fim), não um ajuste de parâmetro. Ganho esperado: é o item
   que ataca os ~4 TB/s de pedidos de ativação e o custo marginal de 6,04 ms/token
   (`journal-lote.md`); o próprio backlog estima 70 → 300-440 tok/s para o par
   "MMQ/WMMA + staging" (`backlog-noite.md` itens 48 e 50), e a maior parte dessa fração é
   atribuível só à forma, sem trocar instrução.
2. **Instrução de matriz (`v_wmma_i32_16x16x16_iu8`)** com o peso em int8 na LDS e a correção de
   escala por bloco de 32. *Portável com ressalva.*
   Vem do SPIR-V do §2.1 (`OpCooperativeMatrixMulAddKHR`) e da tabela de taxas do §0.4: 4096 MACs
   por instrução contra 128 de um issue wave32 de dp4a (32× menos instruções por MAC) e **2× a
   taxa de MAC do dp4a** (1024 contra 512 MAC/CU/clk). Ressalvas: (a) o layout de fragmento da
   RDNA4 não é o do KHR coopmat nem o da RDNA3 — em HIP ele se escreve à mão; (b) a escala por
   bloco de 32 volta a ser problema (a referência Vulkan não a tem porque virou fp16); (c) exige
   o quantizador de ativação no layout MMQ (blocos de 128 com 16 B de escala/soma) em vez dos
   nossos `q8_1` de 32. O primeiro passo nomeado continua sendo um microbench de WMMA i8 isolado
   (`docs/estudo-prefill-b-mmq.md` §6, que já tem o protótipo de GEMM int8 com dp4a medindo
   12,74e12 MAC/s).
3. **Processar o chunk inteiro em uma passagem** (n = 512, não 16). *Portável.*
   A referência monta um grafo de 512 tokens (`n_ubatch = 512` [M]) e lê os 11,108 GB **uma vez
   por prompt** (24,8 GB/s, §4.1); nós lemos 32 vezes (uma por chunk de 16, `graph.cuh:189`), o que
   custa a constante de 13,9 ms do ajuste `pass_ms ≈ 13,9 + 6,04·N` (`journal-lote.md`) × 32 =
   **445 ms = 10,8 % do nosso prefill**, e 59 500 lançamentos em vez de ~1 900. Custo: memória
   para as ativações do chunk (512 × 17408 × 4 B = 35 MB no maior tensor) e um caminho de
   atenção/GDN que aceite n grande.
4. **Fundir o epílogo** — `SILU_MUL`, `SIGMOID_MUL`, `SOFTPLUS_MUL`, `MUL_MAT+ADD`,
   `RMS_NORM+MUL`, `SSM_CONV+ADD+SILU`. *Portável.*
   Padrões em `ggml-vulkan.cpp:18150-18314`; a referência executa cada um em **um** despacho. Nós
   usamos 2 lançamentos em quatro deles (`graph.cuh:1329-1330`, `:1137-1138`, `:1225-1226`,
   `:1316`) ⇒ ~10 dos 29 lançamentos por camada desaparecem (−35 % de lançamentos ≈ −1,7 % do
   tempo) mais o tráfego intermediário (cada elemento visitado 3× em vez de 1×). Ganho esperado:
   2-4 %, e é o item mais barato da lista.
5. **Escolher o formato da ativação pelo caminho de instrução.** *Portável com ressalva.*
   A referência força f32→f16 nas ativações (`:9478-9483`) porque o único pipeline que existe
   para iq3_s é o fp16 — é decisão de **caminho de instrução**, não de precisão. Nós já temos a
   ativação em `q8_1` (`matvec.cuh:57-95`), que é o formato certo para o WMMA int8 (item 2) e o
   errado para o fp16. Não é trabalho independente: é a escolha que o item 2 impõe.
6. **LUT da tabela na LDS** — *já temos* (`matvec.cuh:359-364`, mesma semântica de "copia uma vez
   por CTA"; `docs/vulkan-vs-hip.md` §4.2 já registrava). Não é item de trabalho.

O que **não** vale como caminho: (a) afinar o dp4a atual — `journal-lote.md` Medidas 2-4 já
refutaram tráfego de ativação, UNROLL e registrador, e o §3.5 mostra que a ocupação de issue é
17,5 %; (b) perseguir banda de DRAM — a referência faz o prefill inteiro a 24,8 GB/s de peso.

---

## 7. Medições brutas, e o que ficou sem determinar

```
# A/B do coopmat (coordenador, mesma máquina, llama-bench -p 512 -n 0 -r 2)
baseline                                   1196,49 ± 1,72 tok/s
GGML_VK_DISABLE_COOPMAT=1 + ...COOPMAT2=1   478,38 ± 0,64
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1      1179,51 ± 2,88

# tabela por nó (este documento)
GGML_VK_PERF_LOGGER=1 llama-bench -m Qwen3.8-27B-UD-IQ3_S.gguf -p 512 -n 0 -r 1 -ngl 99
  2ª tabela: 1811 despachos, Total time: 447473 us
  MUL_MAT: 497 despachos, 382163 us (85,4 %)   |   FLASH_ATTN_EXT: 16 × 110,2 us (0,4 %)
  MUL_MAT iq3_s m=17408 n=512 k=5120: 45 × 1340,95 us (68055,5 GFLOPS/s)
  MUL_MAT iq3_s m=5120  n=512 k=17408: 23 × 1535,38 us (59441,5 GFLOPS/s)

# contagem do inventário (GGUF, python sobre o header)
497 tensores 2-D de matmul no tronco, 11,108 GB, 24,3532e9 pesos, 3,649 bit/peso
496 despachos MUL_MAT cobrem exatamente essas formas; o head sai como MUL_MAT_VEC com n=1

# capacidades do dispositivo (vulkaninfo, RADV Mesa 26.2.2)
subgroupSize = 64 (min 32, max 64) · maxComputeSharedMemorySize = 65536
VK_KHR_cooperative_matrix = true (revision 2) · cooperativeMatrix = true
ggml_vulkan: RX 9070 XT (RADV GFX1201) | fp16: dot2 | warp size: 64 | int dot: 1 | matrix cores: KHR_coopmat
```

O que **não** ficou determinado:

1. **A lista de formas KHR coopmat que o RADV reporta.** O `vulkaninfo` desta versão não imprime o
   array `VkCooperativeMatrixPropertiesKHR`; por isso `coopmat_m/n/k = 16/16/16` está **INFERIDO**
   (ainda que `matmul_iq3_s_f16_f16acc_cm1.spv` só possa rodar se as formas 16×16×16 existirem, e o
   `BLOCK_SIZE=256` com `WARP=64` só feche com 4 subgrupos).
2. **Qual das duas variantes cm1** (`f16acc`, acumulador fp16, ou fp32) é a carregada — depende de
   `coopmat_acc_f16_support`. Confirmaria com um build `GGML_VULKAN_DEBUG` ou um capture de
   debug-utils que nomeie a pipeline.
3. **O throughput real da `v_wmma_f16_16x16x16_f16` no gfx1201** (a placa não está documentada no
   ISA que temos, e não medi): é dele que depende a leitura do §3.3 sobre porta de issue
   compartilhada, e é o microbench que falta.
4. **A repartição L1/L2/IC dos ~4 TB/s de pedidos de carga** do nosso kernel em lote — `rocprof`
   e `omniperf` não estão instalados (`journal-lote.md` já registra isso como o que faltou).

---

## Nota do coordenador (14/09, tarde) — correção de integridade

Os números de `pp512` que eu (coordenador) passei para esta frente e que aparecem no §0.1 vinham
de um binário do llama.cpp **modificado localmente** (`ggml-vulkan.cpp:5462`, `rm_kq = 2 -> 1`,
patch de 2026-09-10, binário linkado 73 s depois). O worktree limpo (`/tmp/llama-clean`, commit
`df03399b8`) dá: **pp512 default 1054,29 ± 11,93** (não 1196,49), **coopmat desligado 426,03 ±
0,72** (não 478,38). A **razão** do coopmat praticamente não muda (2,47× contra 2,50×), e é a
razão que sustenta o §1.3 — mas qualquer leitura de valor absoluto neste documento deve usar os
números do binário limpo. Detalhe completo: `docs/estudo-prefill.md` §0.
