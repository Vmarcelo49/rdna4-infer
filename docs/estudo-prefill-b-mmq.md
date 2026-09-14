# O caminho MMQ do ggml-cuda e o que dele cabe no gfx1201

Leitura estática de `/home/marcelo/Projetos/llama.cpp` na revisão `df03399b885831b2a1603b3abb0d8c156808e363`
(10/09/2026), backend `ggml/src/ggml-cuda/`. **Toda referência `arquivo:linha` vale para essa revisão.**
Nada foi modificado nem compilado; os números medidos vêm de `docs/journal-lote.md`,
`docs/journal-kernels.md`, `docs/medicoes-m8.md`, `docs/rdna4-gfx1201-hardware-brief.md` (citado como
*brief*) e do baseline Vulkan (`docs/baseline-vulkan-iq3s.md`). Convenções: **[D]** = documentado por
leitura de código, **[M]** = medido (fonte citada), **INFERIDO** = conclusão minha não verificada,
com o que a confirmaria.

Nota de revisão: neste checkout o MMQ já foi refatorado — os corpos de `vec_dot` **não** estão mais em
`mmq.cuh` (que os inclui em `mmq.cuh:431-432`), e sim em `mmq-vec-dot.cuh`; os `load_tiles_*` estão em
`mmq-load-tiles.cuh`; as tabelas de tile por arquitetura estão em `mmq-config-*.cuh`. As macros
`mmq_q8_1_ds_s`/`_ds_sumq`/`mmq_get_dm*` citadas em análises antigas **não existem** nesta revisão: o que
existe é o enum `MMQ_Q8_1_DS_LAYOUT_{D4,DS4,D2S6}` (`mmq.cuh:18-22`) e a união `d4/ds4/d2s6` de
`block_q8_1_mmq` (`mmq.cuh:39-44`).

## TL;DR (5 linhas)

1. O MMQ é um GEMM em tiles **com dequantização para `int8` em LDS**: tile de `I` linhas de peso × `J`
   tokens, `MMQ_ITER_K = 256` valores de K por iteração (`mmq.cuh:9`), y (ativação) já quantizado em blocos
   de **128** valores com 16 B de escala/soma na cauda (`mmq.cuh:24-27`, `quantize.cu:457-556`).
2. O produto interno tem duas famílias: `*_dp4a` (via `ggml_cuda_dp4a` → `__builtin_amdgcn_sudot4` no
   gfx12, `common.cuh:718-719`) e `*_mma` (via `mma()` → **`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12`**,
   `mma.cuh:1320-1325`), escolhidas em tempo de compilação por `AMD_WMMA_AVAILABLE` (`common.cuh:279-281`).
3. O gfx1201 **satisfaz** esse guardo (`RDNA4` ← `__GFX12__`, `vendors/hip.h:211-213`), tem tabela de tiles
   própria (`mmq-config-rdna4.cuh`) e `should_use_mmq` retorna `true` sem condição (`mmq.cu:380-382`):
   no gfx1201 o MMQ **é** o caminho WMMA int8, não o dp4a.
4. Nós estamos 9× atrás pela **forma aritmética**, não por banda nem por LDS: nosso kernel faz ~45 MAC por
   instrução de warp (medido: `675e6` instruções por token para `30,2e9` MACs) e teto de issue disso é
   **27 TOPS** nos `3,07e11` slots/s do cartão; o tile WMMA rende ~340 MAC/instrução (estimativa INFERIDO).
5. Os 67 TOPS do llama.cpp **não são alcançáveis com dp4a**: exigiriam `2,6e11` dp4a/s = 85 % de *todos* os
   slots de issue do cartão antes de qualquer carga de ativação ou conta de endereço — e o pico vetorial
   f16/fp32 (39-49 TOPS) também fica abaixo disso, logo a referência usa unidades de matriz; entre int8 e
   f16 a diferença **medida** neste cartão é 3,5 % (§7.3), então a aposta é o pipe de matriz, com
   **f16 + acumulador f32** como primeiro passo e o int8 (WMMA iu8) como teto e caminho de decode (§7.6).

---

## 1. Estrutura do tile: quem corta M, N e K, e qual tile sai para 512×17408×5120

### 1.1 As constantes

| constante | valor | onde |
|---|---|---|
| `MMQ_ITER_K` (K por iteração, **lógico**) | 256 | `mmq.cuh:9` |
| `MMQ_TILE_NE_K` (unidade de K do tile, em `int32` = 128 `int8`) | 32 | `mmq.cuh:116` |
| `MMQ_TILE_Y_K` (ints por token no tile y) | 36 (=144 B) | `mmq.cuh:119` |
| `QK8_1_MMQ` (valores por bloco y) | 128 | `mmq.cuh:24` |
| `MMQ_NWARPS` | 8 | `mmq.cuh:11` — **definida e nunca usada** (o número de warps vem de `nthreads`) |
| `rows_per_warp()` | **16** sob `AMD_MFMA_AVAILABLE`/`AMD_WMMA_AVAILABLE` | `mmq.cuh:180-186` |
| `tile_C` (acumulador) | `tile<16,16,int,DATA_LAYOUT_J_MAJOR>` → **8 int32/lane** | `mmq.cuh:479` |

O tile vive numa struct de configuração (`mmq.cuh:165-204`): `nthreads`, `occupancy`, `I` (largura em
`src0->ne[1]`, isto é **linhas de saída**), `J` (largura em `src1->ne[1]`, isto é **tokens**), `sram_layout`
(geometria da linha do tile em `int32`), `K_vram` (= `MMQ_ITER_K`), `stream_k`, `fallback`
(`mmq.cuh:166-174`).

### 1.2 Como o workgroup corta

- **N (linhas de peso = saída)**: `blockIdx.x`, `nty = ceil(nrows_x/I)` — `mmq.cuh:969`, `mmq.cuh:1406`.
- **M (tokens)**: `blockIdx.y`, `ntx = ceil(ncols_max/J)` — `mmq.cuh:1407`, grade em `mmq.cuh:1409`.
- **K**: laço interno `for (kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter)` com
  `blocks_per_iter = K_vram/qk` — `mmq.cuh:895-896`, `mmq.cuh:902`.
- **Dentro do workgroup**: cada warp pega `rows_per_warp/tile_C::I = 16/16 = 1` minitile em i
  (`ntx = 1`, `mmq.cuh:485`), ou seja **16 linhas de peso por warp**, e varre os `J` tokens em passos de
  `tile_C::J = 16` (`mmq.cuh:492-497`).
- **Duplo buffer do y**: por iteração de K o y é copiado para LDS em dois pedaços de 128 valores e cada
  pedaço é consumido por uma chamada de `vec_dot` (`mmq.cuh:904-934`), com `__syncthreads()` entre eles.

### 1.3 Como o tile é escolhido

`ggml_cuda_mul_mat_q` (`mmq.cu:85-264`) → `mul_mat_q_case<type>` (`mmq.cuh:1554-1563`) escolhe
`fallback = (nrows_x % 128 != 0)`; depois `mul_mat_q_switch_J` (`mmq.cuh:1471-1552`) varre
`J = 8,16,...,128` e fica com o **menor número de tiles em M** que caiba em LDS
(`mmq.cuh:1480-1496`):

```cpp
for (int J = 8; J <= 128 && ntiles_J_best > 1; J += 8) {
    config = ggml_cuda_mmq_get_config(type, J, fallback, cc);
    if (config.type == GGML_TYPE_COUNT) continue;           // (type,J) sem instância
    if (mmq_get_nbytes_shared(config, cc) > smpbo) continue; // não cabe em LDS
    ntiles_x = (ncols_opt + config.J - 1) / config.J;
    if (ntiles_x < ntiles_J_best) { J_best = J; ntiles_J_best = ntiles_x; }
}
```
`I` e `nthreads` **não** são escolhidos: vêm da tabela para aquele `(type, J)` (`mmq.cuh:1396`).

### 1.4 O tile de M=512 × N=17408 × K=5120 com IQ3_S

`ncols_opt = ncols_max = ne11 = 512` (caso denso, `mmq.cu:174`). Nenhum `J ≤ 128` dá 1 tile
(`ceil(512/128) = 4`), então o vencedor é o maior `J`, desde que exista entrada na tabela e caiba em LDS:
`CASE(GGML_TYPE_IQ3_S, 256, 2, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, false)`
(`mmq-config-rdna4.cuh:233`). Logo **[D]**:

| campo | valor |
|---|---|
| `nthreads` | 256 = 8 warps de 32 (warp físico 32 no gfx12: `common.cuh:384-390`) |
| `occupancy` (min. blocos/WGP) | 2 |
| `I` × `J` | 128 linhas × 128 tokens |
| `K_vram` | 256 → **20 iterações** para K=5120 |
| grade | `(nty=136, ntx=4, 1)` = **544 CTAs**, 8,5 ondas em 32 WGP com ocupação 2 |
| LDS | ids 512 B + y 18 432 B + x 38 912 B = **57 856 B** de 65 536 B (ver §2.2) |

Por que esse tile: `J` grande reduz o número de blocos de coluna (`ntx = M/J`), e **é `ntx` que multiplica
o custo de staging do peso** (o peso é dequantizado de novo para cada bloco de coluna). Com `I = 128`,
cada CTA cobre 16 linhas por warp — o máximo que o `tile<16,16>` permite sem `ntx > 1` — e o `int32` de
`sum[]` por thread vai a 64 (`float sum[J*I/(nwarps*warp_size)] = {0}` → `128*128/(8*32) = 64`, `mmq.cuh:898`),
que é o teto prático de registrador desse formato.

---

## 2. Caminho do peso: de quantizado para `int8` em LDS e a contabilidade de escala

### 2.1 O layout de LDS

`mmq.cuh:884-886`:
```cpp
extern __shared__ int data_mul_mat_q[];
int * tile_y = data_mul_mat_q + J;                                  // [0,J) = ids_dst
int * tile_x = tile_y + GGML_PAD(J*MMQ_TILE_Y_K, nwarps*warp_size);
```
Três regiões, tamanho total em `mmq.cuh:1382-1387` (`nbs_ids + nbs_x + GGML_PAD(nbs_y, nthreads*4)`), com
`nbs_x = I * sram_stride * 4` no caminho MMA (`mmq.cuh:421-424`). Não há padding extra: o padding está
**embutido no stride da linha** do tile x.

`ggml_cuda_mmq_get_sram_stride` (`mmq.cuh:132-151`) dá o stride em `int32` por linha:
`Q8_0 = 2*32 + 2*32/8 + 4 = 76` (`QI8_0 = QI8_1 = 8`, `ggml-common.h:121-125`) e
`Q3_K = 2*32 + 32/2 + 4 = 84`; os `static_assert` exigem `stride % 8 == 4` (`mmq.cuh:153-159`), que é a
regra de banco de LDS declarada em `mmq.cuh:114-115`: *"K % 2 == 1 for dp4a or K % 8 == 4 for mma"*.
O caminho dp4a usa `2*MMQ_TILE_NE_K + 1 = 65` ints (`mmq-load-tiles.cuh:644`) com swizzle diagonal da área
de escala (`+ i/4`, `+ i/8`).

**Byte count, IQ3_S, I=128, J=128, nthreads=256** (`mmq-config-rdna4.cuh:233` + stride Q8_0 = 76):
ids `128*4 = 512` B; y `128*144 = 18 432` B (já múltiplo de `nthreads*4`, o `GGML_PAD` não acrescenta);
x `128*76*4 = 38 912` B → **57 856 B**. A linha do tile x tem 304 B = 256 B de payload `int8` (256 valores)
+ 32 B de escalas (8 `float`) + 16 B de padding.

### 2.2 O staging: `load_tiles_iq3_s`

`mmq-load-tiles.cuh:1359-1426`, com `threads_per_row = (MMQ_ITER_K/(4*QR3_S))/2 = 8` e `nrows = 4`
(`mmq-load-tiles.cuh:1375-1376`), portanto `kqsx = threadIdx.x % 8`:

1. **Leituras** (`:1389-1395`): 8 B de `qs` (`int2`), 1 B de `qh` (índice de 9 bits: 8 de `qs` + 1 de `qh`)
   e 4 B de `signs` = **13 B por thread** para 32 pesos.
2. **LUT** (`:1399-1401`): `iq3s_grid[...]` — tabela de 512 `uint32` (2 KB, residente em L1) com 4
   magnitudes `int8` empacotadas; mesma indexação da referência CPU (`ggml-common.h:1052`).
3. **Sinal** (`:1403-1407`): `__vcmpne4` monta máscara `0x00/0xFF` por byte e `grid = (grid ^ mask) - mask`
   (`__vsub4` mapeado para `__vsubss4` em `vendors/hip.h:275`) = negação em complemento de dois por byte.
4. **Escrita em LDS** (`:1410-1411`): `x_qs[i*sram_stride + 8*kqsx + (2*l+0/1)] = grid_l/grid_h` — 8
   `int32` por thread, **escrita de 4 B por instrução** (`LDS.32`), 64 ints = 256 `int8` por linha.
5. **Escala** (`:1418-1423`): `ls = 1 + 2*((scales[kqsx/2] >> ...) & 0x0F)`, `d = bxi->d` e
   `x_df[i*sram_stride + kqsx] = ls*d` → **8 `float` por linha**, um por bloco de 32 pesos, exatamente a
   álgebra da CPU (`d * (1 + 2*scales)`).

O ganho estrutural: **este trabalho é feito uma vez por linha de peso por bloco de coluna**, e o resultado
serve os `J = 128` tokens do tile. No caminho por elemento (nosso), a mesma dequantização seria refeita por
token.

Contagem de instruções do staging (INFERIDO, leitura linha a linha): ~40 instruções por thread por linha de
256 pesos (`5` cargas + `4 l × 8` + escala), e as 8 threads de uma linha são lanes do mesmo warp → 4 linhas
por iteração de warp ⇒ ~1024 pesos por 40 instruções de warp ≈ **26 pesos por instrução de warp**
(`0,039 instrução por byte estagiado`).

### 2.3 A contabilidade de escala (o que faz o produto inteiro reproduzir a quantização)

**Lado x (peso)** — a escala entra **já multiplicada** no tile, nunca crua:

| tipo | o que vai para a área de escala (`x_df`) | onde |
|---|---|---|
| IQ3_S / IQ3_XXS / IQ4_XS / IQ4_NL | 8 `float`/linha (`ls*d`) = uma por 32 valores | `mmq-load-tiles.cuh:1421`, `:1352`, `:1488`, `:1557` |
| IQ2_XS / IQ2_S | 8 `float`/linha (`(ls·d + d/2)/4`) | `mmq-load-tiles.cuh:1218-1219`, `:1286-1287` |
| IQ2_XXS | 4 `float`/linha (uma por 64 valores) | `mmq-load-tiles.cuh:1155` |
| Q3_K | 16 `float`/linha (`d*(sc-32)`) | `mmq-load-tiles.cuh:673-679` |
| Q4_K / Q5_K | 8 `half2`/linha = (`d*sc`, **`-dmin*m`**) | `mmq-load-tiles.cuh:894-908` |
| Q6_K | 1 `float` (`d`) + 4 `int32` (16 escalas `int8`) em área `x_sc` própria | `mmq-load-tiles.cuh:955-956`, `:1009`, `:1026` |

**Lado y (ativação)** — `block_q8_1_mmq` (`mmq.cuh:27-46`) é a fusão de 4 blocos q8_1 de 32 valores:
`int8_t qs[128]` + 16 B de cauda que guardam **escalas e somas parciais**
(*"To avoid shared memory bank conflicts each block is padded with 16 bytes. This padding is also used to
store block scales/partial sums"*, `mmq.cuh:32-33`). O layout é escolhido **pelo tipo do peso**
(`mmq.cuh:60-101`): `D4` (uma `float` por 32 valores) para IQ3_S/IQ3_XXS/IQ4_XS/IQ4_NL/Q3_K/Q6_K;
`DS4` (`half2 (d, s)` por 32) para Q4_K/Q5_K/IQ4_1; `D2S6` para Q2_K (escala por 64, soma por 16 nos
primeiros 96). O quantizador preenche isso em `quantize.cu:457-556`: a soma é a **soma dos valores float
antes da quantização** (`quantize.cu:506-512`), gravada em `ds4`/`d2s6` (`quantize.cu:540-552`).

**A correção**: para formatos com mínimo subtraído, `x = d_x·q_x + m_x` e `y = d_y·q_y`, então
`Σ x·y = d_x·d_y·Σq_x q_y + m_x·(d_y·Σq_y)`. O segundo termo é exatamente `dmA.y * dsB.y` em
`mmq-vec-dot.cuh:359-361` (caminho MMA) e `sumf_m += ds8f.y * m[i]`, `return dm4f.x*sumf_d - dm4f.y*sumf_m`
(`vecdotq.cuh:614-620`, caminho dp4a). Tipos **sem** offset (IQ3_S, IQ3_XXS, IQ4_XS, IQ4_NL, Q3_K, Q6_K)
não têm correção: o resultado é `d_x·d_y·Σq_x q_y` com `Σq_x q_y` inteiro exato
(`mmq-vec-dot.cuh:195-196` no MMA, `vecdotq.cuh:249-257` no dp4a).

### 2.4 O epílogo

Acumulador por thread é `float sum[64]` (`mmq.cuh:898`), alimentado por duas chamadas de `vec_dot` por
iteração de K (`mmq.cuh:916` e `:932`). No caminho MMA cada `mma()` (duas WMMA, K=32) termina com
**1 carga de `dA` + 8 × 2 FLOP** (`sum += C.x[l]*dA*dB`, `mmq-vec-dot.cuh:193-197`); com
`get_i(l) = threadIdx.x % 16` constante em `l` no RDNA4 (`mma.cuh:213-214`) o `dA` é içado. A escrita final é
`dst[ids_dst[j]*stride + i] = sum[...]` (`mmq.cuh:509-518`; dp4a em `mmq.cuh:459-468`).

---

## 3. Produto interno: qual instrução cada ramo usa

### 3.1 Ramo dp4a

`ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` (`mmq-vec-dot.cuh:110-140`) → `vec_dot_q8_0_q8_1_impl`
(`vecdotq.cuh:246-257`) → **`ggml_cuda_dp4a`** (`common.cuh:714-752`):

| arquitetura | instrução | linha |
|---|---|---|
| HIP CDNA / RDNA2 / gfx906 | `__builtin_amdgcn_sdot4` | `common.cuh:716-717` |
| **HIP RDNA3 / RDNA4 (gfx1201)** | **`__builtin_amdgcn_sudot4(true,a,true,b,c,false)`** | `common.cuh:718-719` |
| HIP RDNA1 / gfx900 | asm `v_mul_i32_i24` + `v_add3_u32` (4 pares) | `common.cuh:720-733` |
| HIP restante (GCN antigo) | multiplicação escalar `int8` | `common.cuh:734-738` |
| CUDA ≥ sm_61 | `__dp4a` | `common.cuh:743-744` |
| CUDA < sm_61 | multiplicação escalar | `common.cuh:746-748` |

### 3.2 Ramo MMA — e ele **existe** para o gfx1201

`ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma` (`mmq-vec-dot.cuh:142-278`), com `tile_A/B = tile<16,8,int,I_MAJOR>` e
`tile_C = tile<16,16,int,J_MAJOR>`, e o miolo em `mma.cuh:1306-1337`:

| arquitetura | instrução | linha |
|---|---|---|
| CDNA3/4 | `__builtin_amdgcn_mfma_i32_16x16x32_i8` | `mma.cuh:1312` |
| CDNA1/2 | `__builtin_amdgcn_mfma_i32_16x16x16i8` (×2) | `mma.cuh:1314-1315` |
| **RDNA4 (gfx12)** | **`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a_vec[0], true, b_vec[0], acc[0], true)`** (×2) | `mma.cuh:1324-1325` |
| RDNA3 | `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32` (×2) | `mma.cuh:1330-1331` |
| NVIDIA Turing+ | asm `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` | `mma.cuh:924` |

A variante transposta do B usa o mesmo builtin com o último argumento `false` (`mma.cuh:1442`).
No RDNA4 o layout de entrada é `DATA_LAYOUT_I_MAJOR` (`mma.cuh:90-96`) e a carga do fragmento **não usa
`ldmatrix`**: é um copy vetorial de 16 B por lane (`mma.cuh:848-850`), o que importa para a portabilidade
(§5).

### 3.3 Cadeia de fallback sem MFMA e sem WMMA

1. Config do tile: host `ggml_cuda_mmq_get_config` → CDNA → RDNA4 → RDNA3_5 → RDNA3 → **senão RDNA2**
   (`mmq.cuh:229-255`); no device o mesmo por macro (`mmq.cuh:257-282`).
2. Corpos: `ggml_cuda_mmq_get_util_funcs` (`mmq.cuh:538-844`) — se `use_mma_data_layout()` é falso, cai no
   `switch` dp4a (`mmq.cuh:541-679`, `load_tiles_*` + `*_dp4a` + `write_back_dp4a`); se é verdadeiro, no
   `switch` MMA (`mmq.cuh:705-843`).
3. `mma()` sem nenhuma das duas macros → `NO_DEVICE_CODE` (`mma.cuh:1333-1336`), que é `printf` + `__trap()`
   (`common.cuh:427-431`) — não é fallback funcional.
4. Seleção: `ggml_cuda_should_use_mmq` (`mmq.cu:266-393`) — exige `smpbo ≥ 48 KiB` (`mmq.cu:310-317`),
   `amd_mfma_available` (`common.cuh:342-348`), depois `amd_wmma_available` (**RDNA3/RDNA4 → `true` sem
   condição**, `mmq.cu:356-382`), depois Vega/GCN-APU só MoE (`mmq.cu:388-390`), senão dp4a/hipBLAS
   (`mmq.cu:392`). Ordem do dispatch de `mul_mat`: `mmf` → `mmvq` → `mmq` → hipBLAS
   (`ggml-cuda.cu:1856-1868`).
5. `GGML_CUDA_FORCE_MMQ` força o MMQ mesmo sem dp4a (`mmq.cu:329-331`).

### 3.4 O ggml-cuda compila para o gfx1201? Qual caminho ele toma?

**Sim, por leitura de código e CMake** (não compilado aqui; o `build-hip/` do projeto está configurado com
`--offload-arch=gfx1201` e **não tem nenhum `.o`** — `docs/baseline-vulkan-iq3s.md`, seção "Próximo"):

- `cc` vem do `gcnArchName` (`ggml-cuda.cu:323`), parseado em `ggml-cuda.cu:171-214` → `gfx1201` = major
  `0x12`, minor `0x01` → **`cc = 0x1001201`**, que satisfaz `GGML_CUDA_CC_IS_RDNA4` (`cc ≥ 0x1001200`,
  `common.cuh:84,93`).
- No device pass, `--offload-arch=gfx1201` define `__GFX12__` → `RDNA4` (`vendors/hip.h:211-213`) →
  `AMD_WMMA_AVAILABLE` (`common.cuh:279-281`); `AMD_MFMA_AVAILABLE` **não** é definido (exige `CDNA`,
  `common.cuh:275-277`).
- Consequências: `amd_wmma_available(cc)` verdadeiro (`common.cuh:350-352`) ⇒ `use_mma_data_layout()`
  verdadeiro (`mmq.cuh:189-202`) ⇒ tile/RDNA4 (`mmq.cuh:234-236` host, `mmq.cuh:261-262` device) e corpos
  `*_mma` com o builtin `_gfx12`; `should_use_mmq` devolve `true` (`mmq.cu:382`).
- CMake: `ggml/src/ggml-hip/CMakeLists.txt:35-41` encaminha `GPU_TARGETS`/`AMDGPU_TARGETS` para
  `CMAKE_HIP_ARCHITECTURES`; **não existe lista default de arquiteturas AMD** no repositório (se não for
  passada, o CMake usa o `rocm_agent_enumerator` nativo). Defines AMD: `GGML_USE_HIP`, `GGML_HIP_GRAPHS`,
  `GGML_HIP_NO_VMM` (`ggml-hip/CMakeLists.txt:86,100-106`); `GGML_CUDA_F16` **não existe** nesta revisão.
- **Não há opção para desligar o caminho WMMA de RDNA3/RDNA4**: a única flag é `GGML_HIP_MMQ_MFMA`
  (`ggml/CMakeLists.txt:219` → `GGML_HIP_NO_MMQ_MFMA`, `common.cuh:275`), que só afeta CDNA.
- Limiar MMVQ→MMQ afinado **para RDNA4**: IQ3_S/IQ3_XXS/IQ2_*/Q3_K/Q4_K passam a MMQ acima de **4 tokens**
  (`mmvq.cu:258-282`); host genérico AMD = `ne11 ≤ 8` (`mmvq.cu:412`, `mmvq.cuh:3`).

---

## 4. Por que uma instrução classe-MMA muda o teto (aritmética)

Base de issue: **128 SIMD32 × 2,4 GHz = 3,07e11 instruções de warp/s** ([M] *brief* §1 e §6: `rocminfo` dá
64 CU, 2 SIMD/CU, wave32, `multiProcessorCount` 32 = WGP, `clockRate` 2400 MHz). VOPD dá 2 VALU por slot,
mas **não existe par VOPD para `DOT4`/`DOT8`** ([D] *brief* §2, ISA §7.8) — logo dp4a tem no máximo
1 por SIMD por ciclo. O número de `640e9` slots usado em `docs/journal-lote.md` conta 64 CU × **4** SIMD ×
2,5 GHz = 256 SIMDs e fica 2,1× acima deste; uso `3,07e11` daqui para frente (a diferença não muda nenhuma
conclusão, muda o percentual).

MACs por instrução: `v_dot4_i32_iu8` = 4 MAC/lane × 32 lanes = **128 MAC/instrução**;
`v_wmma_i32_16x16x16_iu8` = 16·16·16 = **4096 MAC/instrução** (o acumulador é `int32x8_t`, 8 int32 por lane,
`mma.cuh:1318-1319`) — razão de **32×**.

| cenário | MAC/s | "TOPS" (2·MAC, a convenção dos 67 do llama.cpp) |
|---|---|---|
| dp4a saturando issue, modelo do enunciado (64 CU × 4 SIMD × 2,5 GHz) | 640e9 × 128 = 8,2e13 | **164** |
| dp4a saturando issue, modelo do *brief* (128 SIMD32 × 2,4 GHz) | 3,07e11 × 128 = 3,9e13 | **79** |
| WMMA iu8 saturando issue (modelo do enunciado) | 2,62e15 | **5 243** (ficção: unidade de matriz não é full-rate) |
| WMMA iu8 saturando issue (modelo do *brief*) | 1,26e15 | **2 517** (idem) |
| pico **vetorial** f16/fp32 (2 FMA/slot, [D] *brief* §2: 48,7 TFLOPS oficiais, 39,3 a 2,4 GHz) | 2,0-2,4e13 | **39-49** |
| nosso motor, medido (pp512 = 123,9 tok/s, 512 × 30,2e9 MACs em 4,13 s) | 3,74e12 | **7,5** |
| llama.cpp Vulkan, medido (pp512 = 1 113,96 tok/s; 512 × 30,2e9 MACs em 459,6 ms) | 3,36e13 | **67** |

Três leituras, nesta ordem:

1. **O que o dp4a exigiria para chegar aos 67 TOPS**: `3,36e13 / 128 = 2,62e11` dp4a/s = **85 % de todos os
   slots de issue do cartão**, com zero instrução de carga, de endereço ou de staging. Nossa própria leitura
   de ISA (`docs/journal-kernels.md` §10) mostra **2,3 instruções não-dot por dp4a** no lote em N=16, o que
   levaria a `8,6e11` instruções/s = **2,8× a capacidade total** do cartão. Não fecha por construção.
2. **O teto dp4a com a *nossa* mistura**: medimos `675e6` instruções de warp por token para `30,2e9` MACs
   (`docs/journal-kernels.md` §10: `25,3e6` iterações × ~428 instruções por chunk de 16 tokens = `10,8e9`
   para 16 tokens) = **44,7 MAC por instrução de warp**. A 100 % de issue: `3,07e11 × 44,7 = 1,37e13` MAC/s =
   **27,4 TOPS**. Estamos a 7,5 → **27 % do nosso próprio teto**. Para chegar a 67 TOPS com essa mistura
   seria preciso `7,5e11` instruções/s = **2,4× todo o issue do cartão**. Ou seja: a mistura atual não é
   "dp4a-bound", ela é **grande demais em instruções por MAC** — e nenhuma afinidade de dp4a conserta isso.
3. **O que o WMMA exigiria**: `3,36e13 / 4096 = 8,2e9` WMMA/s = **2,7 % dos slots** (uma WMMA a cada ~37
   ciclos por SIMD). O miolo MMA do MMQ não é o gargalo — quem passa a mandar é o epílogo de escalas e o
   staging. **INFERIDO**: o *throughput* real da `v_wmma_i32_16x16x16_iu8` no gfx1201 não está documentado
   nem medido aqui; o que a aritmética mostra é que **não precisa ser rápido**: até a 1 instrução por 32
   ciclos por SIMD (intervalo pessimista, do tamanho do MFMA do CDNA) ela ainda entrega 67 TOPS. Confirmar
   com um microbench de WMMA i8 no cartão — que é o primeiro experimento que `docs/journal-lote.md` já
   nomeia como pré-requisito.

Duas consequências que caem direto no nosso caso:

- **O pico vetorial do cartão (39-49 TOPS) está abaixo dos 67 TOPS medidos** no Vulkan. Logo a referência
  **não** usa o pipe vetorial: ela usa unidades de matriz. No shader isso é visível — `mul_mm.comp:340-342`
  usa `coopmat<FLOAT_TYPE,...>` com estágio em LDS, e para tipos quantizados o caminho `cm2` dequantiza
  para `float16_t` (`dequant_funcs_cm2.glsl:16`) antes do MMA. **INFERIDO**: que o *runtime* esteja de fato
  no caminho coopmat e não no FMA; confirma com o log de inicialização do backend (detecção em
  `ggml-vulkan.cpp:506-518, 907-925`) ou comparando pp512 com e sem coopmat.
- **O teto do nosso motor não é banda**: no caminho por token já rodamos 500-522 GB/s de 633 GB/s de
  roofline (`docs/journal-lote.md`, Medida 1), e a 67 TOPS o tráfego de peso por passagem seria
  24 GB/s (11,122 GB / 459,6 ms) — folga de 26×. Também não é LDS (§6.2). É a forma aritmética.

---

## 5. Veredito de portabilidade para o gfx1201

| peça | veredito | por quê (com o código) |
|---|---|---|
| **Esqueleto de tile do MMQ** (`mul_mat_q`, duplo buffer de y, laço de K, grade x/y, `write_back_mma`) | **portável** | É C++ CUDA/HIP puro: `extern __shared__`, `__syncthreads()` (vira `s_barrier_signal/wait`, *brief* §3), `#pragma unroll`, `__launch_bounds__`. Nada de PTX. `mmq.cuh:868-942`, `:947-1234`, `:1389-1469`; host `mmq.cu:85-264`. |
| **Staging `int8` em LDS** (`load_tiles_*`) | **portável** | Só aritmética de inteiros e LDS. No HIP os shims existem (`__vcmpne4` = `vendors/hip.h:291`, `__vsub4` = `:275`), o caminho LUT usa `v_perm_b32` via `__builtin_amdgcn_perm` (`vecdotq.cuh:34-95`) e `v_perm_b32` é **full-rate** no gfx1201 ([M] *brief* §2). As escritas em LDS são `int32` (4 B) — sem dependência de largura de vetor. |
| **Correção de escala** (`d_x·d_y·Σ`, `(d,m)`+soma parcial para tipos com offset) | **portável** | Álgebra pura sobre a cauda de 16 B do bloco y, com `dmA.y*dsB.y` (`mmq-vec-dot.cuh:359-361`) ou `dm4f.y*sumf_m` (`vecdotq.cuh:620`). Exige escrever o **quantizador y no layout MMQ** (§6.3) — é o custo real. |
| **Corpos `vec_dot_*_dp4a`** | **portável, com ressalva** | Já é o que o nosso motor usa — `include/rdna4/vecdotq.cuh:88` tem o mesmo `sudot4`. Ressalva: no gfx1201 eles **não são o caminho do MMQ** (o `use_mma_data_layout()` desvia para o MMA, `mmq.cuh:190-194`); continuam úteis como fallback de cauda (`fallback=true`) e para `J` pequeno. |
| **Corpos `vec_dot_*_mma` no ramo NVIDIA (`mma.sync`)** | **não portável** | Asm PTX `mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32` e `ldmatrix.sync.aligned.m8n8.x4.b16` (`mma.cuh:924`, `:832-837`), mais os `get_i/get_j` do `tile<>` sem `AMD_WMMA_AVAILABLE` (`mma.cuh:226-271`). Não existe `ldmatrix` equivalente em RDNA4; nada disso compila para gfx12. |
| **Caminho irmão baseado em WMMA (o "MMQ variante wmma")** | **portável — e é literalmente o mesmo arquivo** | Não há `template-instances` separadas para WMMA: `template-instances/mmq-instance-iq3_s.cu` só faz `DECL_MMQ_CASE(GGML_TYPE_IQ3_S)`, e a seleção do corpo é por macro (`AMD_WMMA_AVAILABLE`, `common.cuh:279-281`) dentro de `mma.cuh`/`mmq-vec-dot.cuh`. O builtin usado no gfx1201 é **o mesmo que já foi confirmado compilando no nosso cartão**. |
| **Flash-attention com MMA** | **não portável / não aplicável** | No RDNA4 o kernel MMA de FA é habilitado por `amd_wmma_available(cc) && Q->ne[0] <= 128 && ... Q->ne[1]*gqa_ratio_eff > 8` (`fattn.cu:649-651`) — com `head_dim = 256` do Qwen3.8-27B **ele nem é selecionado**; cai em `BEST_FATTN_KERNEL_TILE` (`fattn.cu:652-671`). A maquinaria f16 do gfx12 existe (`mma.cuh:1029`, `wmma_f16_16x16x16_f16_w32_gfx12`), mas usá-la exigiria estender o kernel de FA, não portar. |

---

## 6. O plano de porte concreto

### 6.1 Tile e orçamento de LDS

Copiar a tabela do RDNA4 (`mmq-config-rdna4.cuh:222-233`) em vez de inventar geometria:

| passo | `I × J` | `nthreads` | LDS (ids+y+x) | quando usar |
|---|---|---|---|---|
| 1º (estrear) | 64 × 64 | 128 | 256 + 9 216 + 19 456 = **28 928 B** | M=16 (chunk de prefill, MTP) — `mmq-config-rdna4.cuh:229` |
| 2º (alvo) | 128 × 128 | 256 | 512 + 18 432 + 38 912 = **57 856 B** | M=512 — `mmq-config-rdna4.cuh:233` |
| 3º (só se M crescer) | 128 × J>128 | — | não existe na tabela; `J ≤ 128` (`mmq.cuh:1480`) | — |

O gfx1201 reporta **65 536 B de LDS por workgroup** (e a ISA permite ≤64 kB por work-group, *brief* §1);
o WGP tem 128 kB, então 2 CTAs de 57,9 kB cabem — é exatamente o `occupancy = 2` da tabela
(`mmq-config-rdna4.cuh:233`, aplicado em `__launch_bounds__` em `mmq.cuh:948`). Atenção ao registrador:
`I=J=128` dá **64 acumuladores `float` por thread** (`mmq.cuh:898`) mais os fragmentos; a regra medida de
≤96 VGPRs/thread para ocupação máxima (*brief* §4) provavelmente não se sustenta nesse ponto — o
`I=64/J=64` de 28 928 B é a configuração que cabe na curva de ocupação.

Além disso, o piso de `smpbo ≥ 48 KiB` (`mmq.cu:310-317`) é satisfeito sem opt-in: no HIP `smpbo =
prop.sharedMemPerBlock` (`ggml-cuda.cu:320-321`).

### 6.2 Custo de staging por byte de peso — e se LDS vira o novo muro

**Premissa de banda de LDS (ASSUMPTION)**: 64 bancos × 4 B × 32 WGP × 2,4 GHz = **1,97e13 B/s ≈ 19,7 TB/s**.
O número de bancos (64 bancos de 4 B por WGP) é [D] (*brief* §1, ISA §1.2.2.1/§3.3.5); o "4 B por banco por
ciclo" é a leitura padrão, **não medida aqui**. Confirmaria com um microbench de LDS (cópia `ds_read_b128`
em cadeia) — o mesmo tipo de contador `SHADER_CYCLES` que o *brief* §2 já usou.

Aritmética para IQ3_S, M=512, J=128 (⇒ `ntx = 4` blocos de coluna) e a passagem a 67 TOPS
(1,55e13 MACs / 3,36e13 MAC s⁻¹ = **0,46 s**):

| tráfego | conta | total por passagem | taxa |
|---|---|---|---|
| **escrita de staging em LDS** (1 B por byte de peso, `mmq-load-tiles.cuh:1410-1411`) | 4 × 11,122 GB | 44,5 GB | 97 GB/s |
| **leitura do tile x pelo `load_ldmatrix`** (1 leitura por byte por passo `k01`; cada byte pertence a exatamente um passo) | 4 × 11,122 GB | 44,5 GB | 97 GB/s |
| **y → LDS** (2 cópias de `J×144 B` por iteração; `= 1,125·M·pesos/I`) | 1,125 × 512 × 30,2e9/128 | 136 GB | 296 GB/s |
| **piso** |  | **≈225 GB** | **≈490 GB/s = 2,5 % de 19,7 TB/s** |

**Resposta direta: LDS não é o muro.** Mesmo dobrando a premissa (ou metade dela), o staging fica na casa de
1-5 % da banda de LDS. O que o staging custa é **instrução**: ~0,039 instrução de warp por byte estagiado,
contra ~340 MAC por instrução de warp no miolo MMA (INFERIDO, §6.5) — ou seja, o staging é ~6-10 % das
instruções do laço. O que sobra como gargalo real, na ordem: (a) o epílogo de escala (1 FMA por acumulador
por `mma()`, `mmq-vec-dot.cuh:193-197`), (b) o próprio staging, (c) a latência — que é o que hoje nos limita
(15-27 % de issue).

### 6.3 Layout da ativação

O MMQ **não** usa o layout q8_1 padrão do ggml. Ele quer, em VRAM, blocos de **128** valores com 16 B de
cauda, ordenados **k_block-major / token-minor** (`ib = ib0 + k_block*ne1 + blockIdx.x`, `quantize.cu:526-534`),
para que o tile y seja uma cópia contígua de `J×36` ints (`mmq.cuh:905-911`). O stride por token vem de
`mmq.cu:163-166`. Nosso `quantize_q8_1_batch_kernel` produz o layout de 32 valores por bloco com `(d, s)` —
é preciso escrever um quantizador novo (~120 linhas) que:
emita `int8` (não `int8` + `s` no formato antigo), ponha `d` (uma `float` por 32 valores, layout `D4`) ou
`(d, s)` (`DS4`) na cauda de 16 B, e transponha para `[k_block][token]`. Vale para qualquer motor que queira
o tile MMQ, inclusive um tile dp4a puro.

### 6.4 Como as escalas por bloco de 32 se dobram

- **Peso**: a escala já entra multiplicada no tile (`ls*d`, `mmq-load-tiles.cuh:1421`), 8 `float` por linha de
  256 pesos para IQ3_S (32 B de 304 B de linha). Para Q4_K/Q5_K, 8 `half2` `(d·sc, -dmin·m)`
  (`mmq-load-tiles.cuh:894-908`); a negação do mínimo é feita no staging, de propósito, para o epílogo
  poder **somar** o termo de correção.
- **Ativação**: `d` (ou `(d,s)`) por 32 valores na cauda do bloco de 128 (`quantize.cu:547-552`).
- **Epílogo**: `sum[...] += C.x[l]*dA*dB` (`mmq-vec-dot.cuh:196`) e, para tipos com offset,
  `sum[...] += dmA.y*dsB.y` (`mmq-vec-dot.cuh:361`). Custo: 2 FLOP por acumulador por `mma()` — 16 FLOP por
  8192 MACs (8 acumuladores × 2), ou 1 instrução por ~512 MACs. É o preço fixo do formato.

### 6.5 O que o porte dá, em número

Ordem de grandeza do miolo MMA, contado linha a linha (**INFERIDO**, e é a conta que decide o projeto):
por warp, por passo `k01` (K=32): 1 carga de `tile_A` + 8 iterações de `j0` × (1 `tile_B` + 1 `dB` + 2 WMMA +
1 `dA` + 16 FLOP) ≈ **169 instruções para 65 536 MACs**; com 8 passos `k01` por iteração de K e o staging
(~160) e as cópias de y (~40) por cima, dá **~300-350 MAC por instrução de warp** contra os **44,7 medidos**
no nosso kernel. A 67 TOPS isso pede `9,5e10` a `1,1e11` instruções/s = **31-36 % dos slots** — dentro do
que o cartão entrega, e da mesma ordem da ocupação que já atingimos (27-32 %). Confirmação: compilar o
caminho RDNA4 e contar a ISA (`--save-temps`/`llvm-objdump`) como já foi feito em `docs/journal-kernels.md`
§10; o microbench de WMMA i8 mede o outro fator.

### 6.6 Duas ressalvas de projeto

1. **Bit-exatidão**: o MMA acumula o `int32` exato por 32 valores (2 WMMA de K=16) e depois aplica
   `dA·dB` — a mesma granularidade e a mesma ordem crescente de K do nosso `vec_dot` atual, o que torna a
   bit-exatidão *plausível*. **INFERIDO**: só um teste diferencial (padrão de bits, nos tensores reais)
   decide; o gate desta frente talvez tenha de ser `rel-L2` em vez de `memcmp`.
2. **MMQ não substitui o GEMV do decode**: com M=1 os 16 tokens do MMA ficam 15/16 ociosos; o ggml mantém
   `mmvq` até 4-8 tokens (`mmvq.cu:258-282, 412`). No nosso motor o decode já está a 500-522 GB/s
   (banda-bound, `docs/journal-lote.md`), então o MMQ entra no **lote** (M ≥ 4), não em tudo.

---

## 7. As duas famílias de implementação (int8 vs f16) para pesos de 3-4 bits

Pergunta do coordenador: **para IQ3_S/IQ3_XXS/IQ4_XS (59 % dos bytes), qual é a aposta certa no
gfx1201 — int8 (a família do MMQ do ggml-cuda) ou f16 (a família que o Vulkan usa)?** Medições novas
no mesmo cartão/modelo (`llama-bench -p 512 -n 0 -r 2`, backend Vulkan, feitas pelo coordenador):
baseline **1 196,49 tok/s**; `GGML_VK_DISABLE_COOPMAT=1` → **478,38**; coopmat OFF +
`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` → **461,78**; `GGML_VK_DISABLE_DOT2=1` → **1 121,81**.

### 7.1 O que cada família faz no laço interno (verificado no código)

| | **(a) int8** | **(b) f16** |
|---|---|---|
| dequantização do peso | LUT + sinal + empacotamento em `int8` na LDS — `mmq-load-tiles.cuh:1359-1426` (IQ3_S), `:1295-1357` (IQ3_XXS), `:1428-1494` (IQ4_XS) | LUT + escala em fp32 + arredondamento para f16 na escrita — `mul_mm_funcs.glsl:238-261` (IQ3_S), com a LDS em `shared FLOAT_TYPEV2 buf_a[...]`, `mul_mm.comp:194` |
| ativação | precisa ser **quantizada**: `block_q8_1_mmq` de 128 valores + 16 B de cauda — `mmq.cuh:27-46`; quantizador em `quantize.cu:457-556` | **não** precisa: B entra como f16 (de f32 no palco, `mul_mm.comp:378-413`) |
| instrução de MAC (vetorial) | `v_dot4_i32_iu8` = **128 MAC/instrução** (`common.cuh:718-719`) | `v_dot2_f32_f16` = **64 MAC/instrução** (`dot_product_funcs.glsl:4-14`, SPIR-V id 6916) ou 2 `v_fma_f32` (idem, `#else`) |
| instrução de MAC (matriz) | `v_wmma_i32_16x16x16_iu8` = **4 096 MAC** (`mma.cuh:1324`) | `v_wmma_f32_16x16x16_f16` = **4 096 MAC** (`mma.cuh:1232`; no Vulkan, `coopmat<FLOAT_TYPE,...>`, `mul_mm.comp:340-342`) |
| correção de escala | obrigatória para tipos com mínimo: `dmA.y*dsB.y` (`mmq-vec-dot.cuh:359-361`), `mul_mmq_funcs.glsl:33-41` (`dm*(q_sum*ds.x - ds.y)`) | nenhuma: a escala entra multiplicada no próprio f16 estagiado |
| arredondamento do peso | **nenhum**: o inteiro da grade entra exato e a escala é aplicada em fp32 no epílogo | cada peso é arredondado para f16 (mantissa de 11 bits) |

### 7.2 MAC por slot de issue e instruções por 4 MACs

Base de issue: `3,07e11` instruções de warp/s (§4). "Por 4 MACs" = por lane, para comparar com o dp4a,
que vale 1 instrução por 4 MACs por lane.

| caminho | MAC por instrução (warp) | instruções por 4 MACs | MAC por instrução de warp, com o overhead | onde vem o número |
|---|---|---|---|---|
| **int8 dp4a, o nosso kernel em lote** | 128 | **3,31** (1 dp4a + 2,3 não-dot, 33 % de esperas) | **44,7** (medido) | `docs/journal-kernels.md` §10 |
| **int8 dp4a, num tile bom** (4×4 de registrador, LDS) | 128 | ~2,5-3,0 | ~45-50 | INFERIDO da estrutura de `mul_mmq_funcs.glsl:33-41` + `l_warptile_mmq_int` (`ggml-vulkan.cpp:4495`, TM=TN=4) |
| **int8 WMMA (MMQ-RDNA4)** | 4 096 | **0,0103** (16 WMMA + 25 cargas LDS + 128 FMA de epílogo por 65 536 MACs) | **~300-350** | INFERIDO, §6.5 |
| **f16 `v_dot2_f32_f16`, tile do `mul_mm.comp`** | 64 | **~2,56** (por lane por passo `BK_STEP`: 36 cargas LDS + 128 `dot_product` para 256 MACs; `mul_mm.comp:395-424`) | ~50 | INFERIDO, contado no shader |
| **f16 sem DOT2** (2 fma + 2 conversões por `dot_product`) | 32 | ~4,8-6,0 | ~22-27 | idem, ramo `#else` de `dot_product_funcs.glsl:18-25` |
| **f16 coopmat / WMMA f16** | 4 096 | ~0,01 | ~300+ | idem |

Peak de cada família, para referência: int8 dp4a **78,6 TOPS** (100 % dos slots, zero outra instrução);
f16/fp32 vetorial **39-49 TOPS** (2 FMA/slot, *brief* §2); matriz (i8 ou f16) ≥ **2 500 TOPS** de issue.
Ou seja: **o teto da família int8 é 1,6-2× o da família f16 no pipe vetorial, e as duas estão ~30-50×
abaixo do pipe de matriz.**

**Checagem cruzada com o protótipo do P0** (`tests/bench_gemm_gpu.hip`, commit `4c88cad`, medido pelo
coordenador — GEMM int8 com `sudot4` em `tests/bench_gemm_gpu.hip:42`, tile de registrador `RM×RN` em
`:96-107`): **12,74e12 MAC/s em M=512** = 25,5 TOPS, ou `12,74e12/4/32 = 9,95e10` dp4a de warp/s. Com as
2,5-3,0 instruções por dp4a da linha "int8 dp4a, num tile bom" acima, isso é `2,5-3,0e11` instruções de
warp/s = **65-78 % dos 3,80e11 slots a 2,97 GHz** (o protótipo usa o clock de boost; a minha base de
`3,07e11` é o clock medido de 2,4 GHz — as duas batem: o próprio bench calcula o pico dp4a como
`4096 lanes × 2,97 GHz × 4 MAC = 48,7 T MACs/s = 97,3 TOPS`, exatamente 1,24× o meu 78,6 TOPS a 2,4 GHz).
Leitura: **o protótipo já está a 2/3-3/4 do muro de issue da família dp4a** — e é por isso que ele para
perto do fallback do Vulkan e não se aproxima dos 67 TOPS. (O "104 % do llama.cpp sem coopmat" do commit
compara com a **fatia de matvec** do fallback — `478,38 × 30,2e9 × 0,837 = 1,21e13` MAC/s, contra os quais
12,74e12 = 105 % — e não com o total do modelo, que seria 88 %; as duas contas fecham.) Daqui para cima as
saídas são só duas, e as duas são o pipe de matriz: menos instruções por MAC (WMMA: 0,0103 por 4 MACs) ou
mais MAC por instrução (4 096).

### 7.3 O que o A/B do Vulkan mede — e o que ele **não** mede

Fato de código que muda a leitura das medições: **o Vulkan não tem caminho int8 para os tipos IQ.**
As pipelines de `q8_1` (inteiro, dp4a) são criadas uma a uma e a lista é fechada
(`ggml-vulkan.cpp:5235-5246`): Q2_0, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, MXFP4, Q2_K, Q3_K, Q4_K, Q5_K,
Q6_K — **nenhum IQ1/IQ2/IQ3/IQ4**. O seletor pede primeiro o mapa `(type_a, Q8_1)` e só cai no f16 se
ele estiver vazio (`ggml-vulkan.cpp:9490-9501`), então para IQ3_S/IQ3_XXS/IQ4_XS o Vulkan usa **sempre
f16**, com ou sem `integer_dot_product`.

Consequências para ler as quatro medições:

- O `-3,5 %` de `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT` só pode afetar os tipos que **têm** as duas
  famílias: q5_K 9,4 % + q3_K 9,0 % + q6_K 3,3 % + q4_K 2,2 % + q2_K 1,1 % + q8_0 0,3 % ≈ **25 % dos
  bytes** (`docs/quants-inventario.md`). É, portanto, um A/B direto **int8-dp4a × f16** num kernel
  bem-tileado deste cartão: **3,5 %**, ou seja empate técnico.
- Os 59 % de bytes IQ rodam em f16, coopmat ou não. Os 2,50× do coopmat (1 196 / 478) são, para esses
  tipos, o ganho de **matriz f16 × vetor f16** — não de "int8 × f16".
- `GGML_VK_DISABLE_DOT2` custando 6 % mostra que trocar `v_dot2_f32_f16` por 2 `v_fma_f32` quase não
  muda nada: **o caminho f16 vetorial não está limitado pelo dot**, está limitado por carga/issue.

**Calibração de quanto vale o empate de 3,5 %.** O fallback f16 a 478 tok/s = 28,9 TOPS exige, pela
contagem de §7.2, `1,44e13/4 × 2,56/32 = 2,9e11` instruções de warp/s = **~94 % dos 3,07e11 slots**
(com a incerteza da minha contagem, entre 70 % e 95 %). Ou seja: o kernel f16 vetorial do Vulkan está
saturado de issue. Nesse regime, trocar 1 dp4a (4 MAC por lane) por 2 dot2 (2 MAC cada) só mexe na
*fatia* que o MAC ocupa dos slots — e como boa parte dos slots vai para cargas LDS, staging e endereço,
o ganho medido é 3,5 %. Mesma leitura do `DISABLE_DOT2` (6 %): **nestes kernels o MAC não é o gargalo,
a carga é.** Consequência para nós: o nosso teto de família (78,6 TOPS com dp4a) não se alcança
"trocando a instrução", e sim cortando instruções de carga/espera por MAC — que é onde os nossos 3,31
por 4 MACs (contra ~2,5-2,6 do f16 tileado) moram.

### 7.4 Precisão: o peso de 3-4 bits cabe exato em f16?

| | peso | ativação | erro dominante |
|---|---|---|---|
| **int8** | exato: a grade do IQ3_S são inteiros ímpares ≤127 (cabe em `int8`), a escala por 32 vai em fp32 e é aplicada no epílogo (`mmq-load-tiles.cuh:1421`, `mmq-vec-dot.cuh:196`) | **q8_1**: 8 bits com absmax por bloco de 32 (`quantize.cu:499-521`) → erro relativo até 1/254 no maior elemento do bloco e pior nos pequenos | **a ativação** (≈4e-3) |
| **f16** | arredondado para f16: erro relativo ≤ 2⁻¹¹ = **4,9e-4** — o IQ3_S quantiza o peso original com erro de ~1-5 %, logo o arredondamento f16 é **20-100× menor que o erro da própria quantização** | f16: 11 bits, **sem** escala por bloco e sem penalidade de outlier | nenhum dos dois domina; ~4,9e-4 em ambos |

Ressalva séria do lado f16: **o acumulador**. `v_pk_fma_f16` acumula em f16 (inviável para K=5120);
é obrigatório usar a variante de acumulador f32 — `v_dot2_f32_f16` (`dot_product_funcs.glsl:12-15`)
ou coopmat com `f32acc`/`SPV_DOT2` (`ggml-vulkan.cpp:5225-5227`). O Vulkan tem as duas variantes
(`_f32` e `_f32_f16acc`) e escolhe por `ggml_vk_get_mul_mat_mat_f16acc` (`ggml-vulkan.cpp:9135`), com
fallback para f32acc se a pipeline f16acc não existir (idem, `:9136-9146`) — **INFERIDO**: qual das duas
rodou nos 1 196 tok/s não está dito na medição; confirmar com `GGML_VK_...`/log ou comparando a precisão
da saída. Se for f16acc, o número é 2,5× mas a precisão não é a do caminho f32.

### 7.5 Custo de dequantização e de LDS por byte de peso

| | instruções por byte de peso estagiado | bytes de LDS por byte de peso | correção de escala |
|---|---|---|---|
| int8 (IQ3_S, `load_tiles_iq3_s`) | ~0,039 de warp (≈40 por 1024 pesos, §6.2) = **1,2 por lane por peso** | 1 B escrito + 1 B lido = 2 | sim (só para tipos com mínimo) |
| f16 (mesma LUT, `mul_mm_funcs.glsl:238-261`) | INFERIDO: mesma ordem de grandeza (a escala entra em fp32 no próprio staging e o sinal é aplicado por elemento, sem `__vcmpne4`/`__vsub4`) | 2 B escritos + 2 B lidos = **4** | não |

O custo em LDS dobra no f16, mas o total continua na casa de 1-5 % da banda (§6.2) — **não é o
critério**. O critério é a *largura* do tile: com f16 os operandos ocupam 2× por valor, então o tile
`I=128 × K=256` do MMQ int8 (38 912 B só de x) viraria ~78 KB e **não cabe nos 65 536 B** do gfx1201.
Na prática o caminho f16 fatia o K (BK=32-64 com duplo buffer, como o Vulkan faz: `mul_mm.comp:194-195`
usa `SHMEM_STRIDE = BK/2 + 4`), o que resolve o orçamento de LDS mas multiplica os pontos de
`__syncthreads()` e a matemática de índice por passo de K.

### 7.6 Veredito para IQ3_S / IQ3_XXS / IQ4_XS

1. **Entre as duas famílias vetoriais, a decisão não move o ponteiro:** 3,31 (nosso dp4a) / ~2,5-3
   (dp4a tileado) / 2,56 (f16 com dot2) instruções por 4 MACs, e o A/B medido neste cartão dá 3,5 %.
   O teto do int8 no papel é 1,6-2× maior (78,6 × 39-49 TOPS), mas **nós não estamos perto de nenhum dos
   dois tetos**: 7,5 TOPS = 9,5 % do teto int8, enquanto o fallback f16 do Vulkan (478 tok/s = 28,9 TOPS)
   está a 59-74 % do teto f16. O nosso problema é estrutural, não de família.
2. **A aposta certa é o pipe de matriz, não a família.** 2,50× medido (coopmat), e ambos os WMMA do
   gfx12 existem no compilador deste host. A 4 096 MAC por instrução, a diferença i8 × f16 no MAC/instrução
   (a matriz i8 costuma ser 2× a f16) é irrelevante: os 67-72 TOPS pedem <3 % dos slots em qualquer um
   dos dois (§4).
3. **Como primeiro passo, f16 (com acumulador f32)** — pelas três razões que decidem de fato:
   (i) **precisão**: para pesos de 3-4 bits o arredondamento f16 do peso (4,9e-4) é 20-100× menor que o
   erro da própria quantização, enquanto o caminho int8 *introduz* uma quantização nova de 8 bits na
   ativação — f16 é mais preciso ponta a ponta nestes tipos; (ii) **código**: f16 não precisa do
   quantizador q8_1 no layout MMQ (§6.3, ~120 linhas + kernel), nem dos termos de correção, nem do
   truque de sinal `__vcmpne4`/`__vsub4` do `load_tiles_iq3_s`; (iii) **existe implementação medida
   neste cartão e neste modelo** (1 196 tok/s com coopmat f16), enquanto o caminho int8 não tem
   *nenhuma* medição no gfx1201 (Vulkan não tem pipeline int8 para IQ; o backend HIP do llama.cpp não
   está construído).
4. **Manter o int8 onde ele é estruturalmente melhor**: (a) no **decode/M=1**, onde `v_dot4` faz 4 MAC
   por lane contra 2 do dot2 e não há dimensão de matriz para amortizar (é o caminho que já temos,
   `include/rdna4/vecdotq.cuh`); (b) como **fallback de cauda** (`J` pequeno, `fallback=true`) via
   `mmq.cuh:541-679`; (c) para tipos em que o arredondamento f16 do peso for inaceitável — **INFERIDO**:
   nos três tipos do P0 ele não é (§7.4); só uma medição de perplexidade/KL com pesos em f16 confirma.
5. **O que mediria a decisão em vez de inferi-la** (nesta ordem, tudo no nosso motor): um microbench de
   `v_wmma_i32_16x16x16_iu8` e de `v_wmma_f32_16x16x16_f16` com acumulador f32 no gfx1201 (MAC/s por
   SIMD e intervalo de issue de cada um); depois o mesmo par de microbenches para `v_dot4_i32_iu8` ×
   `v_dot2_f32_f16` com **o mesmo tile de registrador** (é o que separa "família" de "estrutura"); e só
   então o kernel de IQ3_S nas duas versões, com o gate de precisão (`rel-L2` contra o caminho atual,
   mais KL no prompt de referência).

## Portar ou não portar (ordenado por ganho esperado)

1. **O caminho de matriz (WMMA) para os tipos que dominam os bytes** — `mma.cuh:1306-1337` (int8:
   builtin `_gfx12` em `:1324-1325`; f16: `:1232`), `mmq-vec-dot.cuh:142-278` (`vec_dot` com escalas),
   `mmq-load-tiles.cuh:1359-1426` (IQ3_S) e `:1295-1357` (IQ3_XXS), ligados em `mmq.cuh:810-815` (IQ3_S)
   e `:804-809` (IQ3_XXS), com o tile `mmq-config-rdna4.cuh:222-233`. Qual das duas variantes de matriz
   primeiro — **§7.6: f16 com acumulador f32**, e o int8 como teto e caminho de decode**. Cobre **52,6 %** dos bytes do nosso inventário (iq3_s 34,8 % +
   iq3_xxs 17,8 %, `docs/quants-inventario.md`) e é o único item que muda a ordem de grandeza: leva o
   teto de 27 para >250 TOPS de issue e a nossa execução de 7,5 para a casa dos 40-70.
2. **O staging de peso em LDS com a LUT** — `mmq-load-tiles.cuh` (os 5 tipos: `:1359-1426`, `:1295-1357`,
   `:1428-1494`, `:822-945`, `:598-700`). É o que amortiza a dequantização sobre os 128 tokens do tile; no
   nosso kernel ela é amortizada sobre 16 (ISA, `docs/journal-kernels.md` §10). Ganho isolado (em cima de um
   tile dp4a, sem WMMA): ~1,3-2×, porque corta instruções não-dot sem cortar o dp4a.
3. **O quantizador y no layout MMQ** (`quantize.cu:457-556`, stride em `mmq.cu:163-166`) — pré-requisito
   duro do item 1 e reaproveitável por qualquer tile (inclusive dp4a). ~120 linhas, sem GPU nova.
4. **O esqueleto de tile + duplo buffer de y, sem MMA** (`mmq.cuh:868-942`, `:1394-1469`) — útil como
   degrau intermediário e como caminho de cauda (`fallback=true`, `J` pequeno), mas **não** resolve o
   problema: teto de 27 TOPS com a mistura atual, 56 TOPS mesmo com uma mistura idealizada
   (0,4 instrução não-dot por dp4a) — abaixo dos 67 TOPS da referência.
5. **Os corpos `*_dp4a`** (`mmq-vec-dot.cuh:110-140` e `vecdotq.cuh`) — **manter**, mas só como fallback
   (`mmq.cuh:541-679`) e para o decode/GEMV; já temos versão vendorizada (`include/rdna4/vecdotq.cuh`).
   Não investir mais neles como caminho principal.
6. **Não portar**: os corpos `mma.sync`/`ldmatrix` da NVIDIA (`mma.cuh:832-837`, `:924`, `:946`), o caminho
   MFMA/CDNA (`mma.cuh:1308-1316`), e o FA-MMA (`fattn.cu:649-651`) — este último nem selecionado com
   `head_dim = 256`.

## O que não foi determinado

- **Throughput real da `v_wmma_i32_16x16x16_iu8` no gfx1201**: nenhum dado aqui. A aritmética mostra que
  67 TOPS só exigem 1 WMMA a cada ~37 ciclos por SIMD, mas é medição que falta (microbench).
- **Se o ggml-cuda compila limpo para gfx1201**: respondido por leitura (sim, sem bloqueio de código);
  nenhum objeto compilado foi inspecionado (`build-hip/` do projeto tem 0 `.o`).
- **Banda de LDS medida**: usei 19,7 TB/s como premissa derivada do número de bancos da ISA, não medida.
- **Se o Vulkan está de fato em coopmat** e não em FMA: inferido do pico vetorial (< 67 TOPS) e do shader
  (`mul_mm.comp:340-342`); confirmação é log de inicialização ou A/B com coopmat desligado.
- **Bit-exatidão entre o nosso `vec_dot` e o MMA**: plausível (mesma granularidade de 32 e mesma ordem de
  K), não verificada.
