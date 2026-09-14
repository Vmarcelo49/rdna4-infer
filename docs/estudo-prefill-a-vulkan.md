# Como o Vulkan faz o prefill (código a código)

Leitura do backend **Vulkan** do llama.cpp (`/home/marcelo/Projetos/llama.cpp`, revisão
`df03399b885831b2a1603b3abb0d8c156808e363`, backend `ggml/src/ggml-vulkan/`), comparado com
o kernel em lote deste motor. Toda referência `arquivo:linha` vale para essa revisão.
**[D]** = documentado por leitura de código, **[M]** = medido nesta máquina,
**INFERIDO** = conclusão minha não verificada, com o que a confirmaria.

## TL;DR (5 linhas)

1. Para este modelo nesta GPU, o prefill **não** passa por `mul_mat_q`/MMQ int8: para
   `iq3_s` o shader `mul_mmq.comp` **não é nem gerado** (`vulkan-shaders-gen.cpp:627-630`
   só cobre legacy/K-quants/mxfp4), e o caminho que roda é o **coopmat1 fp16**
   `matmul_iq3_s_f16_f16acc_cm1` — pesos dequantizados para `float16_t` na LDS e
   `coopMatMulAdd` de 16×16×16 (`mul_mm.comp:339-391`).
2. O tiling é **BM=128 × BN=128 × BK=32** por workgroup de 256 threads (4 subgrupos de 64),
   LDS de **24 KB**, grade de 136×4 workgroups para `m=17408 n=512 k=5120`
   (`ggml-vulkan.cpp:4524-4532`, `4922-4923`, `4809-4815`).
3. Isso despeja **4096 MACs por instrução** (`OpCooperativeMatrixMulAddKHR`, 16·16·16) contra
   **128 MACs por issue wave32 de `v_dot4_i32_iu8`** — 32× menos instruções por MAC —, com
   taxa de pico **igual** (512 MAC/CU/clk nos dois) e topo de **2×** para o WMMA int8.
4. Medido com `GGML_VK_PERF_LOGGER=1`: **1811 despachos por chunk de 512 tokens**, 497 deles
   `MUL_MAT` = **85,4 % do tempo**; 24,935e12 FLOPs de matmul em 380,7 ms = **32,7e12 MAC/s**,
   ou **33,6 %** do pico de matriz fp16 do cartão.
5. Nós: **3,79e12 MAC/s** no matvec (3,9 % do pico de dp4a), 38 MACs por instrução emitida
   contra ~840-1100 do caminho coopmat, e **1,125 byte de ativação por MAC** contra 0,0156 B —
   a diferença está na *forma* (tile M + LDS), não na instrução de multiplicação.

---

## 0. Método, convenções e duas correções de aritmética

### 0.1 O que foi medido aqui, e como

Nada foi compilado nem modificado. Três corridas **dentro do lock** (`scripts/gpu-lock.sh`,
`timeout` por dentro):

| comando | o que deu |
|---|---|
| `vulkaninfo` (2×) | `subgroupSize = 64` (min 32, max 64), `maxComputeSharedMemorySize = 65536`, `VK_KHR_cooperative_matrix = true`, Mesa 26.2.2 / RADV, RX 9070 XT gfx1201 |
| `llama-bench -p 512 -n 0 -r 1 -ngl 99 -v` | `sched_reserve: graph nodes = 3751`, `n_ubatch = 512`, 66/66 camadas na GPU |
| `GGML_VK_PERF_LOGGER=1 llama-bench -p 512 -n 0 -r 1 -ngl 99` | tabela por nó: 1811 despachos, 447,5 ms no chunk (2ª tabela, pós-aquecimento) |

O A/B do coopmat é do coordenador (mesma máquina, `-r 2`): **baseline 1196,49 ± 1,72 tok/s**,
`GGML_VK_DISABLE_COOPMAT=1` + `GGML_VK_DISABLE_COOPMAT2=1` → **478,38 ± 0,64**,
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` → **1179,51 ± 2,88**. Números deste motor:
`docs/journal-lote.md` (Medidas 1-4) e `docs/rocm-estudo.md` §B.

### 0.2 Correção 1: são 25,622e9 pesos por token, não 30,2e9

Contagem direta do GGUF (`Qwen3.8-27B-UD-IQ3_S.gguf`, 866 tensores, arquitetura `qwen35`,
65 blocos): os tensores 2-D que entram em matmul no tronco somam **25,6220e9 pesos em
11,108 GB** (497 tensores) — o inventário que `bench-matvec-shapes-gpu` chama de "11,122 GB".
Os 30,2e9 usados em `docs/journal-lote.md` e `docs/estudo-prefill-b-mmq.md` incluem a
`token_embd` (1,42 GB) e o bloco MTP, que **não** são lidos por token no prefill.

Consequência: todos os "TOPS" derivados caem 15 %:

| | MACs/token | MAC/s medido | "TOPS" na convenção 2 ops/MAC |
|---|---:|---:|---:|
| llama.cpp Vulkan pp512 (matmul: 380,7 ms de 447,5 ms) | 25,622e9 | **32,7e12** | 65,5 |
| llama.cpp Vulkan pp512 (chunk inteiro) | 25,622e9 | 28,5e12 | 57,1 |
| este motor, pp512 = 123,9 tok/s | 25,622e9 | **3,18e12** (3,79e12 no matvec, 83,7 % do tempo) | 6,4 (7,6 no matvec) |

Ou seja: "67 TOPS" era 15 % alto; o valor é **61-65e12 ops/s** (30-33e12 MAC/s). E o nosso
"7,5 TOPS" era 7,5e12 ops/s = **3,75e12 MAC/s**, não 7,5e12 MAC/s. A razão entre os dois
(9,7× no matvec, 9,0× ponta a ponta) não muda.

### 0.3 Correção 2: não é int8

A correção que importa: **a referência não faz um único MAC int8 neste prefill**. O
`DISABLE_INTEGER_DOT_PRODUCT` mexeu 1,4 % porque não havia nada de int8 para desligar (§1.2).
O que roda é fp16 em unidade de matriz. Os "TOPS" da tabela acima são MACs, não int8.

### 0.4 Tetos do cartão (RX 9070 XT, 64 CU, 2,97 GHz de boost)

| recurso | MAC/CU/clk | MAC/s | fonte |
|---|---:|---:|---|
| FP32 vetorial (FMA) | 128 | 24,3e12 | 48,7 TFLOPS publicados |
| FP16 vetorial (`v_pk_fma_f16`) | 256 | 48,7e12 | 97,3 TFLOPS publicados |
| **INT8 vetorial (`v_dot4_i32_iu8`)** | **512** | **97,4e12** | 4 MAC/lane × 128 lanes; e `v_dot4` = 1,00× `v_fma_f32` medido no *brief* |
| **FP16 matriz (WMMA 16×16×16)** | **512** | **97,4e12** | 1024 FLOP/CU/clk (tabela da AMD, GPUOpen RDNA4) |
| INT8 matriz (WMMA iu8) | 1024 | 194,6e12 | 2048 OP/CU/clk, idem |

Os 512 MAC/CU/clk do fp16-WMMA saem também do modelo de ciclos de terceiros que fecha com o
mesmo número (32768 MACs de um tile 16×64×32 fp16 = "64 ciclos de matmul" em um CU).
**A leitura que interessa: fp16-WMMA e dp4a têm o MESMO teto de MAC neste cartão.** O int8
WMMA é 2× — não 4×, não 32×. A vantagem do WMMA é *por instrução*, não por ciclo (§3.3).

---

## 1. Seleção de despacho

### 1.1 A decisão de topo: `ggml_vk_mul_mat` (`ggml-vulkan.cpp:10367-10439`)

Ordem exata das condições, com os valores deste modelo (`M` = linhas de saída, `N` = tokens):

| linha | condição | quando é verdade aqui |
|---|---|---|
| 10378 | `needs_split = dst->ne[2]==1 && dst->ne[3]==1 && nbytes > maxStorageBufferRange` | não (tensores ≤ 90 MB) |
| 10400 | `ggml_vk_can_use_fwht` | não |
| 10402 | `F16 && permutado && dst->ne[1]==1` (0213) | não (peso é quantizado) |
| 10415 | `F16 && !contíguo && dst->ne[1]==1` | não |
| 10422 | `src0 ∈ {F32,F16} && dst->ne[0]==1 && dst->ne[1] > 8` (AᵀB com 1 linha) | não (`M` ≫ 1) |
| **10433** | **`dst->ne[1] == 1 \|\| (dst->ne[1] <= 8 && src1->ne[2]*src1->ne[3] == 1)`** → `ggml_vk_mul_mat_vec_q_f16` | **não: N = 512 > 8** |
| **10436-10437** | **senão: `ggml_vk_mul_mat_q_f16(..., disable_split_k=false)`** | **SIM** |

`mul_mat_vec_max_cols = 8` (`:404`). É *todo* o limiar do GEMV batelado: **até 8 tokens ele
usa o mesmo `mul_mat_vecq.comp`** que usaria em decode, com o peso como A e a ativação como B;
acima disso cai no caminho de matriz-matriz. O heurístico fino de GEMV-vs-MMQ
(`ggml_vk_should_use_mmvq`, `:9703-9782`; "MMVQ is generally good for batches", `:9717-9720`)
**não é consultado neste caminho** — só em `ggml_vk_mul_mat_vec_q_f16` (`:9823`) e na variante
`_id` (`:10846`). Logo, a pergunta "o que muda em n=512" tem uma resposta única: sai do
`mul_mat_vecq` (1 linha de saída por warp) e entra no `mul_mm` (tile 128×128).

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

Para **IQ3_S** acontece o seguinte, e cada passo tem prova:

1. **`quantize_y` é calculado verdadeiro, mas o mapa {IQ3_S, Q8_1} está VAZIO.** A geração
   de shaders só cria `mul_mmq.comp` para legacy/K-quants/mxfp4 (`vulkan-shaders-gen.cpp:627-630`),
   e o mesmo filtro vale para o GEMV (`:1325`). O artefato compilado confirma: existem
   `matmul_iq3_s_f16*.spv`, `mul_mat_vec_iq3_s_f32_f32*.spv` e **nenhum** `*_iq3_s_*q8_1*.spv`
   (`build/ggml/src/ggml-vulkan/vulkan-shaders.spv/`). Então `mmp_map = nullptr` e `quantize_y`
   volta a `false` (`:9498`) — **é exatamente o que a medição do coordenador mede**:
   desligar o dot product inteiro muda 1,4 %.
2. **`y_non_contig` é verdadeiro** por causa de `:9480-9481`: `coopmat_support && !coopmat2 &&
   is_quantized(src0)` com `src1 == F32`. O comentário na linha é literal: *"coopmat1: force
   f32->f16 conversion so the f16 B-type quant pipeline is used"*. Ou seja: **as ativações são
   convertidas para fp16 por um despacho separado** (`ggml_vk_cpy_to_contiguous`, `:9653`,
   cacheado por `prealloc_y_last_tensor_used`, `:9647-9657`) antes da matmul.
3. Com `y_non_contig` e `src0` quantizado, o mapa passa a ser **{IQ3_S, F16}** (`:9497`).
4. `f16acc` (`ggml_vk_get_mul_mat_mat_f16acc`, `:9117-9131`): para tipo quantizado com
   `coopmat_support && !coopmat2` → `ctx->device->fp16 && coopmat_acc_f16_support && prec==DEFAULT`
   (`:9127-9129`). A chave vira `{IQ3_S, F16, mul_mat_id=false, f16acc=true}`.
5. `aligned`: `k = 5120` e `17408` são múltiplos de `l_align = 128` ⇒ `kpad == ne10`, `M>8`, `N=512>8`
   ⇒ **`aligned = true`** ⇒ usa-se o pipeline *aligned* (o `LOAD_VEC_B=8`, 16 B por carga).
6. `split_k` (`ggml_vk_guess_split_k`, `:9023-9062`): `k ≥ 2048` mas `m_tiles*n_tiles = 136*4 = 544
   > shader_core_count` ⇒ **`split_k = 1`**, e o despacho é `{CEIL_DIV(m,128), CEIL_DIV(n,128), 1}`
   (`ggml_vk_dispatch_pipeline` divide pelos `wg_denoms` da pipeline) — 544 workgroups por
   matmul de 17408×512×5120.

### 1.3 O tile que sai: `l_warptile_mmq` com o override AMD+RADV (`:4524-4528`)

```
:4524  } else if (vendor_id == VK_VENDOR_ID_AMD && coopmat_support && driver_id != AmdProprietary) {
:4526      l_warptile     = { 256, 128, 128, 16, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
:4527      l_warptile_mmq = l_warptile_mmq_int = { 256, 128, 128, 32, mm_warp_8, 64, 2, tm_m, tn_m, tk_m, mm_warp_8 };
:4528      l_warptile_mmq_int_k = { 256, 128, 128, 32, mm_warp_16, 64, 1, 4, 2, 1, mm_warp_16 };
```

Este é o único ponto do arquivo que muda o tiling por causa de "AMD com coopmat via driver
livre" e é o que dá os números do prefill. Campos (`:4407-4410`):
`[0]=BLOCK_SIZE, [1]=BM, [2]=BN, [3]=BK, [4]=WM, [5]=WN, [6]=WMITER, [7]=TM, [8]=TN, [9]=TK, [10]=WARP`.

Para esta GPU: `subgroup_size = 64` (vulkaninfo) ⇒ `mm_warp_8 = min(max(64,8),64) = 64` (`:4393`);
`coopmat_m/n/k = 16/16/16` (as formas KHR que o RADV expõe — **INFERIDO**, ver §5) ⇒
`tm_m=tn_m=tk_m=16`. Resultado:

| | BLOCK_SIZE | BM | BN | BK | WM | WN | WMITER | TM | TN | TK | WARP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `l_warptile_mmq` (o que roda) | 256 | 128 | 128 | 32 | 64 | 64 | 2 | 16 | 16 | 16 | 64 |
| `m_warptile_mmq` | 128 | 64 | 64 | 32 | 32→64 | 32 | 2 | 16 | 16 | 16 | 64 |
| `s_warptile_mmq` | 64 | 32 | 32 | 32 | 32 | 32 | 2 | 16 | 16 | 16 | 64 |

E a escolha entre s/m/l é a função em `:5447-5454`:

```
:5449  if (configs.size() <= 1) return 0;
:5450  if (m <= 32 || n <= 32) return 0;
:5451  if (configs.size() == 2) return 1;
:5452  if (m <= 64 || n <= 64) return 1;
:5453  return configs.size() - 1;          // -> o tile L
```

Com `m ∈ {1024..17408}` e `n = 512`, e as três configurações existindo (o teste de LDS em
`:4557-4565` passa: §2.3), o índice é **2 = L**. Os `wg_denoms` do L são `{128,128,1}` (`:4531`).

### 1.4 Quando o coopmat NÃO existe

Sem `VK_KHR_cooperative_matrix` (`GGML_VK_DISABLE_COOPMAT=1` reproduz isso), o caminho é:

* `coopmat_support = false` (`:6559/6585`, desligado em `:7275`), `coopmat2 = false`;
* o `else` de `:4472` monta `tm_*=4, tn_*=4, tk_*=1` (`:4474-4482`) e o override AMD de
  `:4524` **não se aplica** (exige `coopmat_support`) ⇒ `l_warptile_mmq = {128, 128, 128, 32,
  mm_warp_8*2=128, 64, 2, 4, 4, 1, 64}`;
* `ggml_vk_get_mul_mat_mat_f16acc` cai em `:9130` (só `fp16 && prec`) ⇒ a chave continua
  `{IQ3_S, F16, false, true}`;
* as pipelines vêm de `sg_create`/`sg_create_mmq` (`:5175-5230`), e como este cartão expõe
  `VK_VALVE_shader_mixed_float_dot_product` (`fp16: dot2` no log do backend, `:6616-6618`),
  são as variantes **`matmul_iq3_s_f16_dot2_f16acc.spv`**;
* no shader, `#ifdef COOPMAT` é falso ⇒ roda `mul_mm.comp:392-427`: cargas da LDS em
  registrador e `dot_product()` (`dot_product_funcs.glsl:4-13` → `v_dot2_f32_f16`, 2 MACs por
  instrução) sobre `ACC_TYPEV2 sums[WMITER*TM*WNITER*TN/2]`, que com os warptiles acima dá
  `sums[2*4*4*4/2] = 64` acumuladores `f16vec2` = **128 valores de saída por thread**.

Medido: **478,38 tok/s contra 1196,49** — os matrix cores valem **2,5×** neste prefill.

### 1.5 Resumo da seleção para este modelo/GPU

| tipo do peso | n | pipeline | shader |
|---|---:|---|---|
| IQ3_S / IQ4_XS / IQ3_XXS / IQ2_* / IQ1_S (77,6 % dos bytes) | 512 | `{tipo, F16, f16acc=true}` tile L | `matmul_<tipo>_f16_f16acc_cm1.spv` (`mul_mm.comp` + `COOPMAT`) |
| Q*_K (também quantizado, `quantize_y` verdadeiro **e** `mul_mmq` gerado) | 512 | `{tipo, Q8_1}` tile L | `matmul_q4_k_q8_1.spv` (`mul_mmq.comp`, **int8**) |
| `token_embd` (saída, 1 token) | 1 | mapa mul_mat_vec | `mul_mat_vec_q5_K_...` (`MUL_MAT_VEC` na tabela medida) |
| F32/F16 | 512 | `{F16,F16,f16acc}` | `matmul_f16_f16acc_cm1.spv` |

Ou seja: **o mesmo chunk mistura os dois mundos** — os k-quants vão por int8/MMQ e os LUT
quants (78 % dos bytes deste modelo) vão por fp16/coopmat. É por isso que
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` não é um controle limpo para este modelo: ele só
desliga o caminho minoritário.

---

## 2. O shader que roda para IQ3_S em n=512

### 2.1 Identificação (cadeia de evidências)

1. `mul_mmq.comp` não existe para `iq3_s` (§1.2) ⇒ não há caminho int8;
2. as pipelines cm1 são criadas em `:5085-5094` (`X_CM1` → `FOR_EACH_LUT_TYPE_NONFP4`), com
   `tc_mmq` e `create_aligned=true, require_full_subgroups=true` (`:5050`);
3. o nome do artefato é montado em `vulkan-shaders-gen.cpp:437`:
   `name + (f16acc?"_f16acc":"") + (coopmat?"_cm1":"")` ⇒ `matmul_iq3_s_f16_f16acc_cm1.spv`;
4. o SPIR-V desse arquivo comprova o tipo da multiplicação:

```
$ spirv-dis matmul_iq3_s_f16_f16acc_cm1.spv | grep -E "CooperativeMatrixMulAdd|OpTypeCooperativeMatrix"
%1167 = OpTypeCooperativeMatrixKHR %half %uint_3 %TM %TN %uint_2   ; C = half,  Use=Accumulator
%1266 = OpTypeCooperativeMatrixKHR %half %uint_3 %TM %TK %uint_0   ; A = half,  Use=MatrixA
%1300 = OpTypeCooperativeMatrixKHR %half %uint_3 %TK %TN %uint_1   ; B = half,  Use=MatrixB
%1341 = OpCooperativeMatrixMulAddKHR %1167 %1333 %1334 %1340
```

`%uint_3` é `Subgroup` e `%half` é o tipo dos **dois** operandos e do acumulador.
Se `coopmat_acc_f16_support` fosse falso, o pipeline carregado seria
`matmul_iq3_s_f16_cm1.spv` (acumulador `float`); os dois são a MESMA multiplicação 16×16×16
fp16×fp16 (**INFERIDO**: a variante exata depende das propriedades KHR que o RADV reporta;
confirmaria com um build `GGML_VULKAN_DEBUG` lendo `VK_LOG_DEBUG("ggml_vk_create_pipeline")`
ou com um capture de debug-utils, que nomeia cada pipeline).

### 2.2 Geometria, grade e o que cada thread faz

| grandeza | valor | onde |
|---|---|---|
| workgroup | 256 threads = 4 subgrupos | `l_warptile_mmq[0]=256`, `[10]=64` |
| tile do workgroup | 128 (M) × 128 (N) × 32 (K) | `[1],[2],[3]` |
| tile por subgrupo | 64×64 saídas = `cms_per_row × cms_per_col = (WM/TM)×(WN/TN) = 4×4` coopmats de 16×16 | `mul_mm.comp:248-249`, `:342` |
| grade | `(m/128) × (n/128) × 1` workgroups; para 17408×512: **136×4 = 544** | `:4531`, `ggml_vk_dispatch_pipeline` |
| iterações do laço K | 5120/32 = **160**; 17408×512×5120 = 45,6e9 MACs por despacho | `mul_mm.comp:362` |
| trabalho por thread | 16384 saídas/256 threads = 64 saídas; 16 coopmats em registrador de subgrupo | `:342` |

### 2.3 LDS: 24 KB, byte a byte

Do SPIR-V (tamanhos são `OpSpecConstantOp`, avaliados nos valores do §1.3):

| array | fórmula | valor |
|---|---|---:|
| `buf_a` | `BM × (BK/2 + SHMEM_STRIDE_PAD)` × 4 B = 128 × 20 × 4 | 10 240 B |
| `buf_b` | `BN × (BK/2 + SHMEM_STRIDE_PAD)` × 4 B = 128 × 20 × 4 | 10 240 B |
| `coopmat_stage` | `TM × TN × (BLOCK_SIZE/WARP)` × 2 B = 16×16×4×2 | 2 048 B |
| `iq3s_grid` | `uint32_t[512]` | 2 048 B |
| **total** | | **24 576 B** (de 65 536) |

`SHMEM_STRIDE_PAD = 4` para coopmat não-Intel (`mul_mm.comp:182-187`; o host só empurra
constantes 12/13 para Intel, `:4813-4821`) ⇒ `SHMEM_STRIDE = BK/2 + 4 = 20` (`:192`), o padding
que evita conflito de banco nos acessos de 128 bits da coopmat. O `coopmat_stage` é o palco da
LDS para o store (o caminho "aligned" nem o usa: `:481-484`). **Nota**: a checagem do host
(`:4196-4201`) estima `coopmat_stage = TM*TN/warps*sizeof(float) = 128 B` e portanto subestima
o shader em 1 920 B; não muda nada aqui (24,6 KB < 64 KB), mas é uma divergência real entre o
modelo do host e o shader.

### 2.4 Dequantização do peso: grid de 512 entradas em LDS, 4 pesos por lookup

`mul_mm_funcs.glsl:238-260` é o corpo para `DATA_A_IQ3_S`:

```glsl
const uint ib = idx / 64;            // bloco IQ3_S = 256 pesos = 64 dwords de qs
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

* A tabela `iq3s_grid[512]` mora **na LDS** (`types.glsl:1576`, `shared uint32_t iq3s_grid[512]`),
  copiada de `iq3s_grid_const[512]` (`types.glsl:1679-1745`) por `init_iq_shmem`
  (`types.glsl:1746-1758`, chamada em `mul_mm.comp:219-221`). São **2 048 B** por workgroup, e a
  cópia é amortizada por 160 iterações de K.
* **4 pesos por lookup**: um índice de 9 bits (qs + 1 bit de qh) → um uint32 → `unpack8` →
  `vec4`. O `iq_shmem_init.glsl` do checkout é um stub vazio; quem define a função de verdade
  é o `types.glsl` (o stub não chega a ser usado porque o `#if defined(DATA_A_IQ3_S)` de
  `types.glsl:1746` vem antes do include, INFERIDO mas verificável compilando o shader com
  `glslc -DDATA_A_IQ3_S`).
* Custo por thread por tile de K: `loadstride_a = 256*LOAD_VEC_A_EFF/32 = 32` ⇒ o laço
  `for (l = 0; l < BM; l += loadstride_a)` dá **4 iterações**, cada uma com 4 pesos ⇒ **16 pesos
  por thread por tile de K** (≈ 6,9 B lidos da DRAM/L2 e ~8 stores na LDS).
* **A escala do bloco de 32 é dobrada no valor** (`db = d*(1+2*scale)`), não corrigida depois:
  o que entra na LDS já é o peso em fp16 com a escala aplicada. É o oposto do MMQ int8 (que
  acumula em int32 e corrige no epílogo) e é o que dispensa todo o aparato de escalas.

### 2.5 Ativação: fp16, 16 B por carga

* `LOAD_VEC_B = load_vec = fp16 ? "8" : "4"` (`vulkan-shaders-gen.cpp:462`) e, com `ALIGNED=1`,
  `LOAD_VEC_B_EFF = 8` (`mul_mm.comp:276`) ⇒ `load_b_to_shmem` faz `FLOAT_TYPEV8 bb =
  FLOAT_TYPEV8(data_b[idx])` = **8 halves (16 B) por carga** (`mul_mm_funcs.glsl:635-646`),
  e despeja 4 `v2half` na LDS.
* `B_TYPE = f16mat2x4` (`:463`), ou seja a ativação é fp16 *na memória*, não convertida no
  shader: a conversão f32→f16 é um despacho separado, uma vez por tensor de ativação (§1.2).
* `LOAD_VEC_A_EFF = LOAD_VEC_A = lva = lut_load_vec_a("iq3_s") = 4` (`:633`, `:258-262`).
* Todo o tráfego de B do workgroup é `BN × K × 2 B` = 128 × 5120 × 2 = 1,31 MB por workgroup;
  como há 136 tiles em M, **cada elemento de ativação é lido 136 vezes por matmul** ⇒
  0,0156 byte por MAC (§3.6).

### 2.6 Acumulador e store

* Acumulador: `coopmat<ACC_TYPE, gl_ScopeSubgroup, TM, TN, gl_MatrixUseAccumulator> sums[16]`
  (`mul_mm.comp:342`). Com `f16acc` (o que o dispositivo pede, `:9127-9129`), `ACC_TYPE =
  float16_t` (`vulkan-shaders-gen.cpp:483`) e o `OpCooperativeMatrixMulAddKHR` do SPIR-V acima
  tem tipo `%half`. Sem `f16acc`, seria `float`. **Não há acumulador fp32 na variante
  carregada** — se isso importa para a precisão do nosso prefill, é um A/B a medir.
* Store: o caminho `is_aligned && is_in_bounds` (`:481-484`) converte o coopmat para `D_TYPE`
  (float) e faz `coopMatStore` direto para o buffer de saída com `stride_d` — sem passar pela
  LDS. Os outros dois caminhos (stride não alinhado; tile parcial) usam `coopmat_stage` +
  `controlBarrier(Subgroup,...)`.
* `ACC_TYPE_MAX`/clamp (`:432-445`) não se aplica a este tipo.

### 2.7 Resumo do que caracteriza esta shader

| característica | valor |
|---|---|
| multiplicação | fp16 × fp16 → fp16 (WMMA 16×16×16 do subgrupo) |
| peso | dequantizado do IQ3_S para fp16 na LDS, escala do grupo de 32 embutida |
| LUT | 512×u32 na LDS, 4 pesos por índice, 2 KB por workgroup |
| ativação | fp16 na memória (convertida antes, em outro despacho), 16 B por carga |
| acumulador | registrador de coopmat do subgrupo, 16×16, tipo fp16 (f16acc) |
| escala pós-mma | **nenhuma** (já está no valor fp16) |
| laneamento | 256 threads, 4 subgrupos reais (WARP=64), grade M/128 × N/128 |

---

## 3. Nível de instrução

### 3.1 De GLSL/SPIR-V para gfx12

`OpCooperativeMatrixMulAddKHR` com `Scope=Subgroup` e operandos fp16 de 16×16×16 é baixado
pelo LLVM/AMDGPU para a família `v_wmma_*_16x16x16_*`. O RADV expõe `VK_KHR_cooperative_matrix`
em RDNA3+ justamente apoiado em WMMA (o commit do RADV e o guia da AMD chamam a instrução de
"Wave Matrix Multiply Accumulate"). Para este shader, o candidato é
`v_wmma_f16_16x16x16_f16` (acumulador fp16; `v_wmma_f32_16x16x16_f16` se a variante carregada
for a `_cm1` sem `f16acc`). **INFERIDO** (não desmontei ISA do shader RADV): confirmaria com
`rocprof`/RGP ou com `RADV_DEBUG=spirv` + disassembly do pipeline — e o irmão HIP já tem a
evidência do lado HIP (`docs/estudo-prefill-b-mmq.md:224`: `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`
em `mma.cuh:1324-1325` no gfx1201).

No lado vectorial, o caminho **sem** coopmat usa `v_dot2_f32_f16` (`dot_product_funcs.glsl:4-13`,
habilitado por `DOT2_F16` = `VK_VALVE_shader_mixed_float_dot_product`, que este driver expõe),
e o nosso kernel usa `v_dot4_i32_iu8` via `__builtin_amdgcn_sudot4` (`vecdotq.cuh:84-88`).

### 3.2 MACs por instrução

| instrução | MACs por instrução (por wave) | MACs por lane |
|---|---:|---:|
| `v_dot4_i32_iu8` (wave32) | 4 × 32 = **128** | 4 |
| `v_dot4_i32_iu8` (wave64) | 4 × 64 = **256** | 4 |
| `v_dot2_f32_f16` (wave32) | 2 × 32 = 64 | 2 |
| `v_wmma_f16_16x16x16_f16` | 16·16·16 = **4096** | 64 (= 4096/64) |

Razão **32× por issue wave32** (128 → 4096) ou 16× por issue wave64 — e **1024×** se alguém
comparar "4 MACs do dp4a" (por lane) com "4096 do WMMA" (por wave), que é a comparação errada.

### 3.3 O que isso vale em ciclos (e por que 32× não é 32×)

Pelos tetos do §0.4, em ciclos de CU por 4096 MACs:

| caminho | MAC/CU/clk | ciclos de CU por 4096 MACs | instruções por 4096 MACs |
|---|---:|---:|---:|
| dp4a (wave32) | 512 | 8 | 32 |
| dp4a (wave64) | 512 | 8 | 16 |
| **fp16 WMMA 16×16×16** | **512** | **8** | **1** |
| int8 WMMA 16×16×16 | 1024 | 4 | 1 |

Ou seja: **o WMMA fp16 tem exatamente a mesma taxa de MAC que o dp4a neste cartão** (512
MAC/CU/clk). O ganho é de **contagem de instruções** (1 contra 32 para o mesmo trabalho),
não de taxa aritmética. O caminho int8 (que a AMD adicionou na RDNA4 — `v_wmma_i32_16x16x16_iu8`)
é 2× em taxa e continua 1 instrução por 4096 MACs.

**INFERIDO**: se o WMMA e o caminho vectorial compartilham a porta de issue (é o que a leitura
de terceiros sobre a RDNA4 afirma: *"WMMA and ordinary vector instructions issue down the same
pipe"*), então 8 ciclos de CU por instrução de matriz significam 8 ciclos em que a unidade
vectorial não pode emitir — e a vantagem líquida do WMMA sobre dp4a na *taxa de MAC* seria
nula, sobrando só o efeito de "caberem 32× menos instruções não-matrix no mesmo orçamento".
O que separa as duas leituras é um microbench de WMMA no cartão (medir dp4a/s com e sem WMMA
intercalado), que `docs/journal-lote.md` §"o que sobra" já nomeia como pré-requisito.

### 3.4 A aritmética do nosso lado, com os números corrigidos

Base do enunciado: `64 CU × 4 SIMD × 2,5 GHz = 640e9` slots de issue/s — isso é
"256 SIMD32 × 2,5 GHz", ou seja **slots de wave32**. (O *brief* mede `rocminfo` dando 128
SIMD32 × 2,4 GHz = 307e9; as duas contagens dão o mesmo teto de MAC, muda o percentual.
Uso a do enunciado e digo onde ela entra.)

* 1 slot wave32 saturado de dp4a = 128 MACs ⇒ teto **81,9e12 MACs/s a 2,5 GHz** =
  **97,3e12 a 2,97 GHz**, que é exatamente o número publicado da placa. A conta do enunciado
  fecha.
* Nós: 3,75e12 MAC/s (matvec) ⇒ **4,6 %** do teto de dp4a; 3,18e12 ponta a ponta ⇒ 3,9 %.
* Em slots: 3,75e12 / 128 = 29,3e9 slots de dp4a/s contra 640e9 disponíveis = **4,6 %**.
* A referência: 32,7e12 MAC/s divisão por 512 MAC/CU/clk = 63,9e9 ciclos de CU/s = **33,6 %**
  do teto de matriz fp16 (e do teto de dp4a, que é o mesmo número); se fosse int8 WMMA seria
  16,8 %.
* Sem olhar taxa de pico: **MACs por instrução emitida**. Nós: `675e6` instruções de warp por
  token (`journal-kernels.md` §10) para 25,622e9 MACs = **38 MACs/instrução**. A referência:
  por tile de K e por subgrupo, 32 `coopMatMulAdd` (32×4096 = 131 072 MACs) contra ~72
  instruções do laço cooperativo (`mul_mm.comp:380-391`: 4+16 cargas + 16 mma por metade de K)
  mais ~78 de staging (`:362-372` com 4 iterações de A e 2 de B) ⇒ ~**840-1 100 MACs por
  instrução emitida**. Razão: **22-29×**.

### 3.5 Por que a nossa conta não fecha com "issue saturado"

No lote, `docs/journal-lote.md` mede `675e6` instruções de warp por token em 6,04 ms de custo
marginal ⇒ `112e9` instruções/s contra 640e9 slots = **17,5 % de ocupação de issue**, e o
dp4a ocupa 128/… = 4,6 % dos slots. Não é saturação de issue, não é DRAM (o lado do peso roda
a 100 GB/s em 633), não é registrador (localSizeBytes = 0) e não é o UNROLL (medido, refutado).
O que sobra é latência não coberta — e uma conta de tráfego que a referência não tem (§3.6).

### 3.6 A conta que separa os dois: bytes de ativação por MAC

| | bytes de ativação por MAC | por token |
|---|---:|---:|
| llama.cpp Vulkan (BM = 128) | 2 B / 128 = **0,0156** | 0,38 GB |
| este motor (1 linha de saída por warp) | 36 B / 32 = **1,125** | 28,8 GB |

Contas: o nosso kernel lê o bloco `q8_1` inteiro (36 B por 32 valores) **uma vez por linha de
saída** ⇒ `m × n × k/32 × 36 B` por passagem = 1,125 B por MAC. A referência lê a ativação
**uma vez por tile de 128 linhas de saída** ⇒ `m×n×k×2/128 B` = 0,0156 B por MAC. Razão: **72×**.

Em números absolutos: a referência lê `24,935e12 FLOPs / 128 = 194,8 GB` de ativação por chunk
de 512 tokens em 380,7 ms = **512 GB/s**; nós lemos 1,125 × 25,622e9 × 16 = **461 GB por chunk
de 16 tokens** em 110,7 ms = **4,2 TB/s** de pedidos de carga (o peso, no mesmo intervalo,
é 100 GB/s). A nossa própria Medida 2 (`act_stride = 0`, 95 % do tempo) mostra que *tornar a
ativação residente* vale só 5 %, então a leitura correta é a que o `journal-lote.md` já
registrou: a conta não fecha com nenhum recurso saturado, e o candidato do tamanho certo é
L2/Infinity Cache a ~4 TB/s de pedidos (3× o que a IC entrega). O que este documento
acrescenta é o **número de referência**: a forma que a referência usa reduz esse tráfego
**72×** sem mudar uma linha de aritmética.

---

## 4. Despachos e fusão por chunk de 512 tokens

### 4.1 Medido (`GGML_VK_PERF_LOGGER=1`, 2ª tabela = sem compilação de pipeline no meio)

* **1811 despachos** para o chunk inteiro de 512 tokens (um único grafo; `n_ubatch = 512`).
* **85,4 % do tempo em `MUL_MAT`**: 497 despachos = 382,2 ms de 447,5 ms.
* Repartição do resto (ms): `GATED_DELTA_NET` 12,3 · `GLU` 12,0 · `CONCAT` 10,2 ·
  `RMS_NORM_MUL` 9,9 (305 despachos somando as 4 formas) · `ADD` 5,1 · `SSM_CONV_SILU` 3,0 ·
  `RMS_NORM` 2,8 · `SILU_MUL` 2,1 · `FLASH_ATTN_EXT` 1,8 · `ROPE` 1,6 · `SCALE` 1,4 ·
  `MUL_MAT_VEC` 1,4 (o head, 1 token) · `GET_ROWS` 0,6 · `SIGMOID_MUL` 0,7 · `CPY` 0,5 ·
  `SOFTPLUS_MUL` 0,3 · `SET_ROWS` 0,2 · `SIGMOID` 0,1.
* Por tipo de peso, no `MUL_MAT` (ms): **iq3_s 134,5 (127 despachos)** · iq3_xxs 78,1 (77) ·
  iq4_xs 77,4 (88) · iq2_s 23,9 (21) · q3_K 20,8 (15) · iq2_xxs 14,0 (12) · iq2_xs 13,5 (12) ·
  q4_K 7,3 (23) · q2_K 4,5 (6) · q5_K 2,6 (15) · iq1_s 2,1 (2) · q8_0 1,8 (96) · q6_K/iq4_nl 0,2.
* GFLOPS/s medidos por despacho: 49,1-66,5e12 para `iq3_s` (o menor é o `k=17408`, o maior o
  `12288×512×5120`); `iq4_xs` chega a 75,9e12; `q3_K` cai a 29,9e12.

Uma observação que vale para a nossa fila: `FLASH_ATTN_EXT` custa **0,4 %** do prefill e
`GATED_DELTA_NET` 2,8 %. Todo o resto do prefill desta referência é matmul.

### 4.2 O que a fusão remove

O pass de fusão está em `ggml_backend_vk_graph_compute` (`:18144-18340`), com as decisões em
`:18150-18314`. Padrões existentes e o que a tabela medida mostra de fato:

| padrão do llama.cpp | linha | despachos medidos | no nosso grafo? |
|---|---|---:|---|
| `MUL_MAT + ADD [ + ADD]` → `MUL_MAT_ADD[_ADD]` | `:18155-18165` | dentro de `MUL_MAT` | **sim**: matvec + `add_launch` (resíduo), `graph.cuh:1316/1333` |
| `RMS_NORM + MUL` → `RMS_NORM_MUL` | `:18220-18227` | 209 | **sim**: nós já fundimos (a norma aplica o peso) |
| `UNARY(SILU) + MUL` → `SILU_MUL` | `:18228-18237` | 48 | **não**: 2 launches (Swiglu do FFN, `graph.cuh:1329-1330`) |
| `UNARY(SIGMOID) + MUL` → `SIGMOID_MUL` | `:18232` | 16 | **não**: 2 launches (gate da atenção, `:1137-1138`) |
| `SOFTPLUS + MUL` → `SOFTPLUS_MUL` | `:18234` | 48 | **não**: 2 launches (alpha do GDN, `:1225-1226`) |
| `SSM_CONV (+ADD+SILU)` → `SSM_CONV_SILU` | `:18238-18248` | 48 | **parcial**: `conv1d_state_batch_launch` (`:1232`) |
| `RMS_NORM+MUL+ROPE(+VIEW+SET_ROWS)` | `:18182-18202` | 16 + 16 | **não**: norma e rope separados (`:1320`, `:1075-1076`) |
| `MUL_MAT_ID + …`, `TOPK_MOE…`, `TOPK_QSA` | `:18166-18314` | 0 | não (não temos MoE) |
| `MULTI_ADD` (`ggml_vk_fuse_multi_add`) | `:18150-18154` | — | não |
| `RMS_NORM + MUL + ADD_MUL` | `:18203-18212` | — | não |

A fusão leva de **3751 nós de grafo** (medido, `sched_reserve: graph nodes = 3751`) a **1811
despachos** — remove ~52 % dos nós. Os 4 padrões que a referência usa e nós não
(`SILU_MUL`, `SIGMOID_MUL`, `SOFTPLUS_MUL`, `MUL_MAT+ADD`) são exatamente os que explicam a
diferença de contagem abaixo.

### 4.3 A nossa contagem, para comparar

Contagem estática de call-sites de lançamento por camada no caminho em lote
(`include/rdna4/graph.cuh`):

* **camada GDN** (48): `rms_norm` (1040) + `proj_batch`×4 sendo o primeiro e o último com
  `act_ready=false` ⇒ 2 launches cada (1205, 1207-1209, 1262) + `quantize_batch` (1206) +
  `unary`/`add_bcast`/`unary`/`mul_bcast` (1221-1226) + `conv1d_state_batch` (1232) +
  `l2_norm` (1238) + `delta_rule_batch` (1243) + `rms_norm` (1249) + `unary`+`mul` (1254-1255)
  + `add` (1316) + `rms_norm` (1320) + gate/up (1327-1328) + `unary`+`mul` (1329-1330) +
  `proj_batch` down (1332) + `add` ⇒ **≈ 29-30 lançamentos**;
* **camada de atenção** (16): os mesmos 10 da cauda + norma, 3 projeções, quantização,
  deinterleave, 2 normas de q/k, 2 ropes, `kv_write_batch` (que faz 2 launches, `:873-874`),
  atenção, sigmoid+mul ⇒ **≈ 28**;
* total por chunk de 16 tokens: 48×29 + 16×28 + cabeça ≈ **1 850-1 900 lançamentos**;
* o CLI processa o prompt em chunks de **16** (`kMaxBatch = 16`, `graph.cuh:189`;
  `prefill_ids` em `src/main.hip:165-180` monta chunks de 16/8/4/3/2) ⇒ **512 tokens = 32 chunks
  ⇒ ~59 500 lançamentos** para o mesmo trabalho que a referência faz em **1 811**.

Por token: **116 lançamentos** contra **3,5**. Com o gap de despacho medido neste cartão
(**3,5 µs**, `docs/rocm-estudo.md` §B.1), isso é `59 500 × 3,5 µs = 208 ms` = **5,0 %** dos
4,13 s do nosso prefill. Não é a causa dos 9×, mas é 5 % que a referência não paga — e o
número "~2200 lançamentos/token" que circula no repo é do caminho **por token**; no caminho em
lote são ~116/token.

---

## 5. O que NÃO é transferável

| peça | por quê |
|---|---|
| **`OpCooperativeMatrixMulAddKHR` / a abstração coopmat** | O KHR coopmat entrega um fragmento **opaco**: você diz `coopMatLoad(cache_a, buf_a, offset, stride, RowMajor)` e o driver decide qual lane tem qual elemento (`mul_mm.comp:383-388`). Em HIP não existe essa camada: `__builtin_amdgcn_wmma_*_w32_gfx12` recebe **registradores com layout explícito**, e na RDNA4 esse layout **mudou** em relação à RDNA3 (a AMD avisa que não há compatibilidade). O que se porta é a *forma* do tile, não o shader. |
| **`ALIGNED`, `SHMEM_STRIDE_PAD`, `APPLY_SLM_A_RESHAPE` via specialization constants** | `mul_mm.comp:182-187`, `:51` — um shader com N variantes resolvidas na criação da pipeline (`ggml-vulkan.cpp:4813-4832`). HIP resolve isso com template/`#define`; a variante `APPLY_SLM_A_RESHAPE` existe só para o driver proprietário da Intel (`:4815-4819`). |
| **O tuning por `vendor_id`/`driver_id`** | O override que dá o tile 128×128 existe **apenas** para `AMD && coopmat && driver != AMD-proprietary` (`:4524`), e a pinagem de subgrupo é Intel-only (`:5039`). Não há equivalente em HIP: é tabela por arquitetura e driver. |
| **`require_full_subgroups` = "4 subgrupos"** | `mul_mm.comp:244` usa `gl_SubgroupID` e `:263-264` calcula `warp_r = warp_i % (BM/WM) = %2`, `warp_c = warp_i / 2`; com `BM/WM = 2` e `WN/TN = 4`, a conta só fecha com **exatamente 4 subgrupos** ⇒ o shader exige `BLOCK_SIZE/64 = 4`, isto é, **wave64**. O nosso motor é wave32 medido (`docs/rocm-gfx1201-hardware-brief.md`, rocminfo). O mesmo shader em wave32 daria `gl_SubgroupID ∈ [0,8)` e escreveria fora do tile. Em HIP isso não é um detalhe de portabilidade: é a razão pela qual o caminho WMMA do `ggml-cuda` usa explicitamente a variante `_w32_gfx12` (`mma.cuh:1324-1325`) e um `tile<>` cujo layout é o do wave32. |
| **A tolerância a fp16 nos pesos** | O caminho coopmat **dequantiza para fp16 e guarda a escala no valor** (`mul_mm_funcs.glsl:246-254`). Isso só é aceitável porque não há correção de escala depois. Quem quiser int8 (para pegar os 2× da RDNA4) tem de reintroduzir a correção por bloco de 32 — que é exatamente o "descale tax" que o irmão deste documento mede do lado HIP (`docs/estudo-prefill-b-mmq.md` §6.3). Os dois documentos apontam para a mesma conclusão por caminhos diferentes. |
| **O `GGML_VK_DISABLE_*` como controle experimental** | No nosso caso `DISABLE_INTEGER_DOT_PRODUCT` mexe 1,4 % **porque o tipo não tem pipeline int8** (§1.2), não porque "int8 não ajuda". Usar esse A/B como evidência sobre o valor do int8 no prefill deste modelo seria um erro de leitura. |

O que **é** transferível sem ressalva: a estrutura de três níveis (tile de saída na LDS →
carga em registrador de fragmento → uma instrução de matriz), o staging do peso dequantizado
na LDS com a LUT compartilhada, o tile de K = 32 (o bloco de quantização), o padding de 4
`v2half` na stride da LDS, e o fato de a ativação ser lida uma vez por tile de M em vez de
uma vez por linha de saída.

---

## 6. O que precisaríamos ter no HIP para empatar

Ordenado por ganho esperado. Cada item diz de que linha do llama.cpp ele vem e o que custa.

1. **Tile de saída com o peso dequantizado na LDS e a ativação lida uma vez por tile de M**
   (a forma `mul_mm`, não a forma `mul_mat_vec`). *Portável.*
   Vem de `mul_mm.comp:362-427` (estágio de A e B na LDS + laço cooperativo) e de
   `ggml-vulkan.cpp:4524-4532` (BM=128, BN=128, BK=32). O que muda no nosso lado:
   `matvec_kernel_gen`/`matvec_kernel_batch` (`matvec.cuh:322-430`, `:447-543`) hoje têm
   **1 elemento de saída por thread por passo de k**, `ROWS=8` linhas por CTA com `WPR=1`, e
   leem o bloco `q8_1` da ativação **uma vez por linha de saída** (1,125 B/MAC contra
   0,0156 B da referência, §3.6). É uma família de kernel nova (buffer LDS para o peso
   dequantizado + acumulador por thread + redução no fim), não um ajuste. Ganho esperado: é o
   item que ataca os 4,2 TB/s de pedidos de ativação e o custo marginal de 6,04 ms/token
   (`journal-lote.md`) — o próprio backlog do repo estima 70 → 300-440 tok/s para o par
   "MMQ/WMMA + staging" (`backlog-noite.md` itens 48/50); sem trocar instrução, só com a forma,
   a fração atribuível a este item é a maior parte do que resta antes do teto de dp4a.
2. **Instrução de matriz (`v_wmma_i32_16x16x16_iu8`)** com o peso em int8 na LDS e a correção
   de escala por bloco de 32. *Portável com ressalva.*
   Vem do SPIR-V do §2.1 (`OpCooperativeMatrixMulAddKHR`) e da tabela de taxas da AMD (§0.4):
   4096 MACs por instrução contra 128 de um issue wave32 de dp4a (32× menos instruções por
   MAC) e **2× a taxa de MAC do dp4a** (1024 contra 512 MAC/CU/clk). Ressalvas: (a) o layout de
   fragmento da RDNA4 não é o mesmo da RDNA3 e não é o do KHR coopmat — em HIP ele é escrito à
   mão (`mma.cuh`, coberto em `docs/estudo-prefill-b-mmq.md` §3.2); (b) a escala por bloco de
   32 volta a ser problema (a referência Vulkan **não** a tem porque virou fp16); (c) exige o
   quantizador de ativação no layout MMQ (blocos de 128 com 16 B de escala/soma) em vez dos
   nossos `q8_1` de 32. O primeiro passo nomeado continua sendo um microbench de WMMA i8 no
   cartão, isolado do motor.
3. **Processar o chunk inteiro em uma passagem** (n = 512, não 16). *Portável.*
   A referência monta um grafo de 512 tokens (`n_ubatch = 512` medido) e lê os 11,108 GB **uma
   vez** por prompt (24,2 GB/s, §0.2); nós lemos 32 vezes (uma por chunk de 16,
   `graph.cuh:189`), o que custa a constante de 13,9 ms do ajuste `pass_ms ≈ 13,9 + 6,04·N`
   (`journal-lote.md`) × 32 = **445 ms = 10,8 % do nosso prefill**, além de 59 500 lançamentos
   em vez de ~1 900. Custo: memória para as ativações do chunk (512 × 17408 × 4 B = 35 MB no
   maior tensor) e um caminho de atenção/GDN que aceite n grande.
4. **Fundir o epílogo** — `SILU_MUL`, `SIGMOID_MUL`, `SOFTPLUS_MUL`, `MUL_MAT+ADD`,
   `RMS_NORM+MUL`, `SSM_CONV+ADD+SILU`. *Portável.*
   Padrões em `ggml-vulkan.cpp:18150-18314`; a referência os executa em **um** despacho cada.
   Nós fazemos 2 launches em cada um dos 4 primeiros (`graph.cuh:1329-1330`, `:1137-1138`,
   `:1225-1226`, `:1316`) ⇒ ~10 dos ~29 lançamentos por camada desaparecem (−35 % de
   lançamentos = ~1,7 % do tempo) mais o tráfego intermediário (cada elemento visitado 3× em
   vez de 1). Ganho esperado: 2-4 %, e é o item mais barato da lista.
5. **Escolher o formato da ativação pelo caminho de instrução.** *Portável com ressalva.*
   A referência força f32→f16 nas ativações (`:9478-9483`) porque o único pipeline que existe
   para iq3_s é o fp16 — é uma decisão **do caminho de instrução**, não de precisão. Nós já
   temos a ativação em `q8_1` (`matvec.cuh:57-95`), que é o formato certo para o WMMA int8
   (item 2) e o formato errado para o fp16. Não é trabalho independente: é a escolha que o
   item 2 impõe.
6. **LUT da tabela na LDS** — *já temos* (`matvec.cuh:359-364`, com a mesma semântica de
   "copia uma vez por CTA"; `docs/vulkan-vs-hip.md` §4.2 já registrava). Não é item de trabalho.

Itens que **não** valem como caminho: (a) afinar o dp4a atual — `journal-lote.md` Medidas 2-4
já refutaram tráfego de ativação, UNROLL e registrador, e o §3.5 mostra que a ocupação de issue
é 17,5 %; (b) perseguir banda de DRAM — a referência faz o prefill inteiro a 24 GB/s de peso.

---

## 7. Medições brutas (para quem quiser reconferir)

```
# A/B do coopmat (coordenador, mesma máquina, llama-bench -p 512 -n 0 -r 2)
baseline                                 1196,49 ± 1,72 tok/s
GGML_VK_DISABLE_COOPMAT=1 + ...COOPMAT2=1  478,38 ± 0,64
GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1     1179,51 ± 2,88

# tabela por nó (este documento)
GGML_VK_PERF_LOGGER=1 llama-bench -m Qwen3.8-27B-UD-IQ3_S.gguf -p 512 -n 0 -r 1 -ngl 99
  2ª tabela: 1811 despachos, Total time: 447473 us
  MUL_MAT: 497 despachos, 382163 us (85,4 %)  |  FLASH_ATTN_EXT: 16 × 110,2 us (0,4 %)
  MUL_MAT iq3_s m=17408 n=512 k=5120: 45 × 1340,95 us (68055,5 GFLOPS/s)
  MUL_MAT iq3_s m=5120  n=512 k=17408: 23 × 1535,38 us (59441,5 GFLOPS/s)

# capacidades do dispositivo (vulkaninfo, RADV Mesa 26.2.2)
subgroupSize = 64 (min 32, max 64) · maxComputeSharedMemorySize = 65536
VK_KHR_cooperative_matrix = true (revision 2) · cooperativeMatrix = true
ggml_vulkan: RX 9070 XT (RADV GFX1201) | fp16: dot2 | warp size: 64 | int dot: 1 | matrix cores: KHR_coopmat
```

O que **ficou sem determinar**: (a) a lista de formas KHR coopmat que o RADV reporta
(`vulkaninfo` desta versão não imprime o array `VkCooperativeMatrixPropertyKHR`) — por isso
`coopmat_m/n/k = 16/16/16` está marcado INFERIDO, ainda que `matmul_iq3_s_f16_f16acc_cm1.spv`
só possa rodar se as formas 16×16×16 existirem; (b) qual das duas variantes cm1
(`f16acc` com acumulador fp16 ou fp32) é a carregada; (c) o *throughput* de fato da
`v_wmma_f16_16x16x16_f16` no gfx1201 (não medido, não documentado no ISA que temos) — e é dele
que depende a leitura do §3.3 sobre porta de issue compartilhada; (d) a repartição exata de
L1/L2/IC dos 4,2 TB/s de pedidos de carga do nosso kernel em lote.
