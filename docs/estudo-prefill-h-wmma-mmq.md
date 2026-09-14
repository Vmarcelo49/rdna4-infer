# MMQ int8 com WMMA sobre os blocos reais do modelo (frente H, degrau D4)

Frente H do `docs/plano-prefill.md`. Bancada nova: `tests/bench_mmq_wmma_gpu.hip`
(alvo `bench-mmq-wmma-gpu` do CMakeLists, que já existia). É a pergunta que a
frente D deixou aberta: o pico do WMMA int8 é **4,2× o do dp4a** (§2 do
`docs/estudo-prefill-d-wmma.md`) e o caminho int8 **pode ser bit-exato** (§5 do
mesmo doc) — mas ninguém tinha rodado o WMMA int8 sobre os **blocos reais** do
modelo dentro de um GEMM tilejado. É isso que decide se o teto de 4,2× vira
prefill.

Tensor: `blk.3.ffn_up.weight` de
`/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`,
**N=17408 × K=5120, iq3_s, 38,3 MB, 0,4297 B/peso** — o mesmo tensor da frente F,
então os números são comparáveis. Ativação quantizada em `q8_1` **pelo kernel do
próprio motor** (`rdna4::quantize_q8_1_batch_launch`, `matvec.cuh:78`).

**TL;DR**

1. **Bit-exato, e é o resultado mais forte deste documento.** O WMMA int8 com o
   mesmo staging e a mesma correção do dp4a dá **0 elementos diferentes** em
   todos os shapes: `0 de 278 528` (M=16), `0 de 1 114 112` (M=64), `0 de
   2 228 224` (M=128) e **`0 de 8 912 896` (M=512, N=17408)**; contra o
   `vec_dot_iq3_s_q8_1` do motor, `0 de 4096`; contra o oráculo de CPU,
   `0 de 256`. A ISA confirma que o motor e esta bancada emitem a **mesma
   sequência** por bloco de 32: `v_mul_lo_u32` (sumi·(1+2·sc) em inteiro),
   `v_cvt_f32_i32`, `v_mul_f32` (d_w·d_a), `v_fmac_f32`.
2. **Velocidade: 7,5 / 16,7 / 19,7 / 22,8 T-MAC/s em M=16/64/128/512** = 3,8 /
   8,4 / 10,0 / **11,8 %** do pico de **198 T-MAC/s remedido na mesma janela**
   (a frente D mediu 182 numa janela diferente; contra 182 os mesmos números dão
   4,1 / 9,2 / 10,8 / 12,5 %).
3. **O ganho real não é 4,2×, é 1,56-1,83×.** Contra o dp4a tilejado com o MESMO
   staging e a MESMA correção, na mesma janela: 2,82× em M=16, 1,56× em M=64,
   1,68× em M=128, **1,83× em M=512**. A diferença entre o pico (4,1×) e o kernel
   (1,8×) tem nome: **a correção de escala bit-exata custa 53-56 % do miolo**
   (miolo isolado: 84 T-MAC/s sem correção contra 37 T-MAC/s com ela).
4. **Onde trava é o mesmo lugar de sempre: o staging.** 50-61 % do tempo do
   kernel em M≥64 (1,0-1,4 ms de 2,0-2,4 ms), a 132-275 GB/s de fonte contra os
   633 GB/s de roofline do cartão — o diagnóstico da frente F (§1/§4) se
   reproduz sem mudança. O miolo, quando roda sozinho, faz 37 T-MAC/s (19 % do
   pico) e 84 T-MAC/s sem a correção (42 %).
5. **Duplo buffer piora 35-43 %** (mesma janela, mesmo binário: 14,16 contra
   21,68 T-MAC/s em M=512/BM128 BN64): o mesmo resultado negativo da frente F,
   agora medido no kernel do WMMA.

---

## 1. A bancada, e o que ela verifica antes de cronometrar

`bench-mmq-wmma-gpu` tem três kernels de GEMM sobre o **mesmo staging** (bytes
int8 na LDS + `d_w`/`(1+2·sc)`/`d_a` por bloco de 32), e a diferença entre eles é
só a instrução de MAC:

| kernel | laço interno | para que serve |
|---|---|---|
| `gemm_wmma_i8_kernel` | `v_wmma_i32_16x16x16_iu8` (128 MAC/lane) | o objeto do estudo |
| `gemm_dp4a_i8_kernel` | `sudot4` = `ggml_cuda_dp4a` (4 MAC/lane) | o A/B bit a bit e de taxa |
| `ref_engine_kernel` | `rdna4::vec_dot_iq3_s_q8_1` do motor, 1 thread por saída | a referência do MOTOR |

Layout da LDS por janela de `BK` (stride de linha `BK+8`, 8 B alinhado; com
`BK=64` são 18 palavras, e 18 é par, então os 16 lanes de linha caem em 16 bancos
distintos nas duas leituras de fragmento):

```
s_w[BN][BK+8]  int8   peso   (linha = n, coluna = k)  -> operando B do WMMA
s_a[BM][BK+8]  int8   ativação q8_1                   -> operando A
s_dwsc[NKB*BN] int2   x = bits de d_w, y = 1+2*sc     (por n, por bloco de 32)
s_da[NKB*BM]   float  d_a = fp16(ds & 0xffff)         (por m, por bloco de 32)
```

`d_w`/`d_a` ficam **transpostos** (bloco de k maior, linha menor) de propósito: é
o que faz a correção custar 8 floats consecutivos + 1 `int2` por lane, e não um
gather. É a consequência prática de D ser a transposta de A/B (§5 abaixo).

Receita do operando: a de `docs/estudo-prefill-d-wmma.md` §1 (A0/B0/D0), com os
dois bools de sinalidade em `true` e o `clamp` em `true`; os acumuladores são
zerados a cada bloco de 32, então `|sumi| ≤ 32·15·127 = 60 960` e não existe
saturação possível.

**Cadeia de verificação (tudo antes de qualquer cronometragem):**

1. **staging int8 contra o oráculo de CPU** — 2048 pesos da linha 0, byte a byte,
   com a álgebra do `vec_dot_iq3_s_q8_1` escrita em CPU sobre os bytes crus do
   GGUF e a LUT `iq3s_grid` **lida do device**: `0 de 2048` bytes diferentes,
   `0 de 64` erros em `d_w`/`(1+2·sc)`;
2. **GEMM num caso pequeno (M=4, N=64) contra o oráculo de CPU** (produto em
   int32 exato, `×(1+2·sc)` inteiro, `d_w·d_a` em fp32, acumulador fp32 com
   `std::fma`): `0 de 256` elementos com bits diferentes, `max|gpu-cpu| = 0`;
3. **GEMM contra o `vec_dot_iq3_s_q8_1` DO MOTOR** (M=16, N=256, 4096 saídas):
   `0` bits diferentes para o WMMA, `0` para o dp4a e `0` entre os dois;
4. **contagem de bits diferentes entre WMMA e dp4a em todos os shapes medidos**;
5. **`hipGetLastError()` depois de todo lançamento**, contagem de zeros e de
   não-finitos na saída (um kernel que não roda aparece como "toda zero").

O item 5 não é cerimônia: a **primeira** versão desta bancada escrevia fora do
array de LDS (com `DBUF=false` o índice de buffer alternava entre 0 e 1 num array
de tamanho 1) e a verificação pegou **24 de 256 saídas não-finitas** antes de
qualquer número ser reportado. Foi o item 2 que apontou o lugar.

Reprodução da cadeia de verificação (tudo dentro do lock, `timeout` **dentro**):

```bash
source scripts/rocm-env.sh
cmake --build build -j8 --target bench-mmq-wmma-gpu
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
./scripts/gpu-lock.sh timeout 900 ./build/bench-mmq-wmma-gpu $M --m 16 --cfg 3 --reps 3
```

---

## 2. Bit-exatidão: 0 elementos diferentes, e por quê

Janela A (`--m 16,64,128,512 --cfg 0,3,6,7 --reps 5`; pico remedido na janela:
WMMA 198,04 T-MAC/s, dp4a 48,12, razão 4,12×). Saída literal, por shape:

```
  check M=16   BM64  BN64  BK64  W4x1 MT1 NT4 WMMA x dp4a: 0 de 278528 bits diferentes (max ulp 0, rel-L2 0, zeros 0)
  check M=64   BM64  BN64  BK64  W4x1 MT1 NT4 WMMA x dp4a: 0 de 1114112 bits diferentes (max ulp 0, rel-L2 0, zeros 0)
  check M=128  BM128 BN64  BK64  W8x1 MT1 NT4 WMMA x dp4a: 0 de 2228224 bits diferentes (max ulp 0, rel-L2 0, zeros 0)
  check M=512  BM128 BN64  BK64  W8x1 MT1 NT4 WMMA x dp4a: 0 de 8912896 bits diferentes (max ulp 0, rel-L2 0, zeros 0)
```

`max ulp 0` e `rel-L2 0` são o que distingue "bit-exato" de "perto": não há
1 ulp de diferença em nenhum elemento. A janela A tem 16 células verificadas
(4 geometrias × 4 valores de M), de 278 528 a 8 912 896 elementos cada (~50
milhões no total); somando as outras janelas desta frente (B, C, D, G e os dois
A/B de duplo buffer) são 55 células — **todas com 0 elementos diferentes**.

**A prova de que isso é a aritmética do motor, e não uma convenção desta
bancada.** Compilando o arquivo para ISA e olhando o laço interno do
`ref_engine_kernel` (que é o `vec_dot_iq3_s_q8_1` inlinado com `acc +=`), a
sequência por bloco de 32 é:

```
    v_mul_lo_u32 v5, v5, v9        ; sumi *= 1+2*sc      (INTEIRO, vecdotq.cuh:725)
    v_cvt_f32_i32 v5, v5           ; (float) do inteiro
    v_mul_f32   v1, v9, v1         ; d = d_w * d_a       (fp32, vecdotq.cuh:727)
    v_dual_fmac_f32 v52, v1, v5    ; acc = fma(d, sumi_f, acc)
```

O kernel desta bancada emite exatamente as mesmas quatro instruções
(`__fmaf_rn` fixado no lugar do `fmac` que o compilador escolheria), e é por isso
que a comparação com o motor dá `0 de 4096` e não "alguns ulps". **A ordem
importa**: aplicar `(1+2·sc)` depois do `d_w·d_a` em fp32, ou acumular o int32 por
vários blocos e escalar uma vez no fim, muda o arredondamento — é o que o §5 da
frente D já dizia e o que esta medição confirma com a instrução na mão.

ISA e contagem de instruções (comando exato):

```bash
/opt/rocm/bin/amdclang++ -x hip --offload-arch=gfx1201 -O3 -DNDEBUG -Iinclude \
    --cuda-device-only -S tests/bench_mmq_wmma_gpu.hip -o /tmp/h-dev.s
grep -o "gemm_wmma_i8_kernelILi[0-9]*ELi[0-9]*ELi[0-9]*ELi[0-9]*ELi[0-9]*ELb[01]EEE[^.]*\.num_vgpr, [0-9]*" /tmp/h-dev.s
```

---

## 3. Velocidade: T-MAC/s por M, e a fração do pico

Janela A e B/C (picos remedidos na mesma janela: **198,04 / 198,77 / 198,40
T-MAC/s** de WMMA int8; 48,12 / — / 48,03 de dp4a; razão 4,12× / — / 4,13×, contra
4,2× da frente D). Todas as linhas são **mínimo de 5 repetições** depois de
150 ms de aquecimento por modo, com a mesma grade nos três modos.

| M | melhor tile | t_GEMM ms | **T-MAC/s** | % do pico (198) | % de 182 (frente D) | dp4a mesma janela | **dp4a/WMMA** | × prefill hoje (3,18 T) |
|---|---|---|---|---|---|---|---|---|
| 16 | BM16 BN64 BK64 W1×4 | 0,189 | **7,54** | 3,8 % | 4,1 % | 2,67 | **2,82×** | 2,37× |
| 64 | BM64 BN128 BK64 W4×2 | 0,341 | **16,71** | 8,4 % | 9,2 % | 10,69 | **1,56×** | 5,25× |
| 128 | BM128 BN64 BK64 W8×1 | 0,578 | **19,74** | 10,0 % | 10,8 % | 11,76 | **1,68×** | 6,21× |
| 512 | BM128 BN128 BK64 W8×2 | 1,999 | **22,83** | 11,8 % | 12,5 % | 12,47 | **1,83×** | 7,18× |

Comandos (a linha de M=512 da melhor geometria sai da janela B,
`--m 512 --cfg 8,9 --reps 5`; as outras três da janela A):

```bash
./scripts/gpu-lock.sh timeout 900 ./build/bench-mmq-wmma-gpu $M --m 16,64,128,512 --cfg 0,3,6,7 --reps 5
./scripts/gpu-lock.sh timeout 900 ./build/bench-mmq-wmma-gpu $M --m 512 --cfg 8,9 --reps 5
./scripts/gpu-lock.sh timeout 900 ./build/bench-mmq-wmma-gpu $M --m 64,128 --cfg 9 --reps 5
```

**Varredura de geometria** (janela D, `--m 16,64 --cfg 0,1,2,3,4,5,6 --reps 5`,
pico da janela 116,43 T-MAC/s — a janela estava com a placa compartilhada; os
números absolutos são dessa janela, o **ranking** se reproduz na janela A):

| M | tile | t_GEMM ms | T-MAC/s | dp4a | dp4a/WMMA | LDS/CTA | VGPR/lane |
|---|---|---|---|---|---|---|---|
| 16 | **BM16 BN64 BK64 W1×4** | 0,187 | **7,63** | 2,69 | 0,35 | 6 912 B | 95 |
| 16 | BM16 BN128 BK64 W1×4 | 0,204 | 7,00 | 2,68 | 0,38 | 12 544 B | 110 |
| 16 | BM32 BN64 BK64 W2×2 | 0,249 | 5,74 | 2,67 | 0,47 | 8 192 B | 104 |
| 16 | BM64 BN128 BK64 W4×2 | 0,346 | 4,13 | 2,67 | 0,65 | 16 384 B | 129 |
| 16 | BM64 BN64 BK64 W4×1 | 0,363 | 3,93 | 2,68 | 0,68 | 10 752 B | 129 |
| 64 | **BM64 BN128 BK64 W4×2** | 0,347 | **16,43** | 10,73 | 0,65 | 16 384 B | 129 |
| 64 | BM64 BN64 BK64 W2×2 | 0,360 | 15,83 | 10,81 | 0,68 | 10 752 B | 128 |
| 64 | BM64 BN64 BK64 W4×1 | 0,365 | 15,64 | 10,63 | 0,68 | 10 752 B | 129 |
| 64 | BM32 BN64 BK64 W2×2 | 0,446 | 12,80 | 10,72 | 0,84 | 8 192 B | 104 |
| 64 | BM64 BN64 BK32 W4×1 | 0,485 | 11,77 | 10,66 | 0,91 | 5 888 B | 125 |
| 64 | BM16 BN64 BK64 W1×4 | 0,590 | 9,66 | 10,62 | 1,10 | 6 912 B | 95 |
| 64 | BM128 BN64 BK64 W8×1 | 0,588 | 9,71 | 10,73 | 1,11 | 15 872 B | 129 |

Três leituras:

1. **Em M=16 o WMMA ganha de verdade**: 7,63 contra 2,69 T-MAC/s do dp4a
   (**2,8×**), porque o tile estreito (BM=16, 4 warps de 1 tile de 16 em N) não
   desperdiça linhas e o staging é o único custo — 80-82 % do tempo, a 586 G
   pesos/s de fonte (252 GB/s).
   É a forma que o motor usa hoje (M=16 por sub-lote): **2,4× o prefill atual**
   sem sair de M=16.
2. **O tile de M tem de acompanhar o M do lote**: BM=16 em M=64/128/512 reestagia
   o peso M/16 vezes e o "staging" passa a ser 143-164 % do tempo do GEMM (a
   contabilidade da frente F §2, mesma convenção). BM=128 em M=64 desperdiça
   metade do tile (9,19 T contra 16,43 do BM=64).
3. **`BK=128` e `BN=128` não pagam o que prometiam**: BK=128 em BM=128 (LDS
   30 KB ⇒ 2 CTAs/CU) dá 21,20 T em M=512 contra 21,87 do BK=64, e o staging
   piora (61 % contra 55 %; 116 contra 133 GB/s de fonte) — a hipótese "linha de
   cache cheia conserta o staging" do §4 da frente D **não se confirmou** aqui.
   Já `BN=128` (com BM=128, 8 warps, W8×2) é o melhor tile em M=512: 22,83 T com
   **50 % de staging** (1,009 ms de 1,999 ms) a 353 G pesos/s.

---

## 4. Onde vai o tempo: staging, correção, matrix core

O mesmo kernel, na mesma grade, em quatro modos (é o que permite atribuir a
diferença ao staging e não a outro knob): modo 0 = GEMM; modo 1 = staging
(ativação + peso) sem conta; **modo 4 = miolo isolado** (o staging acontece uma
vez e a conta roda `K/BK` vezes sobre a mesma LDS, sem barreira — o total de MAC
é o mesmo do GEMM, então a taxa é comparável direto e **não depende de
subtração**). Linhas de M=512, janela A/B:

| tile | t_GEMM | staging (modo 1) | % staging | sem correção | miolo (modo 4) | miolo sem correção | correção = % do miolo |
|---|---|---|---|---|---|---|---|
| BM128 BN64 BK64 W8×1 | 2,087 ms | 1,156 ms | **55 %** | 1,509 ms | 1,233 ms → **37,0 T** | 0,544 ms → 83,9 T | **56 %** |
| BM128 BN128 BK64 W8×2 | 1,999 ms | 1,009 ms | **50 %** | 1,422 ms | 1,270 ms → 35,9 T | 0,591 ms → 77,3 T | **53 %** |
| BM64 BN128 BK64 W4×2 | 2,162 ms | 1,113 ms | 51 % | 1,509 ms | 1,238 ms → 36,9 T | 0,570 ms → 80,0 T | 54 % |
| BM64 BN64 BK64 W4×1 | 2,389 ms | 1,429 ms | 60 % | 1,803 ms | 1,214 ms → 37,6 T | 0,566 ms → 80,6 T | 53 % |

Em M=128 (janela A, BM128 BN64 W8×1): t_GEMM 0,578 ms, staging 0,335 ms (58 %),
miolo 0,340 ms → 33,5 T-MAC/s (16,9 % do pico), miolo sem correção 0,155 ms →
73,7 T-MAC/s (37,2 %).

**Leitura, com o número que aponta para o lugar:**

1. **O staging é o maior pedaço: 50-61 % do kernel em M≥64.** Ele move
   132-275 GB/s de fonte (o roofline medido do cartão é 633 GB/s,
   `docs/medicoes-banda-e-gargalos.md`) e escreve 1,25 B/peso na LDS (contra 2 B
   do caminho f16). É o mesmo diagnóstico da frente F §1/§4 — **latência, não
   banda nem instrução** — e a melhora de 173,5 para 308-353 G pesos/s vem do
   staging mais barato (int8 em vez de f16) e de 3-6 CTAs/CU em vez de 1.
2. **A correção de escala é o segundo maior pedaço, e é nova**: 25-30 % do GEMM e
   **53-56 % do miolo**. Ela não se esconde atrás do matrix core: o miolo com ela
   faz 37 T-MAC/s e sem ela 84 T-MAC/s (2,3×). É a mesma disputa de issue que o §2
   da frente D mediu no laço misto WMMA+dp4a — o WMMA não tem um pipe de VALU
   paralelo livre.
3. **O matrix core não é o limite.** O miolo sem correção chega a 84 T-MAC/s
   (42 % do pico) e o pico em si é reprodutível (198,0 no início e 197,98 no fim
   da mesma corrida, 1,00×). Os 11,8 % do pico que o kernel inteiro alcança são
   **staging (50 %) + correção (28 %) + miolo saturado no resto**.
4. **Mesmo com a correção de graça, o teto é ~32 T-MAC/s** (M=512, BM128 BN128:
   t "sem correção" = 1,422 ms = 32,09 T-MAC/s = 16,2 % do pico). Ou seja: se
   alguém achar uma correção que não custe nada **e mantenha a bit-exatidão**, o
   caminho int8-WMMA empata com o prefill do llama.cpp (32,7 T) — e não passa
   disso, porque metade do tempo é staging.

---

## 5. A consequência da transposição de D: a correção, instrução por instrução

A frente D §1 previu que D ser a transposta de A/B faria a correção custar
"2 LDS.128 + 1 LDS.32 + 8 FMA por lane por bloco de 32". Medido na ISA, para um
warp com MT=1 × NT=4 (32 acumuladores por lane, o desenho do BM128/BN64 W8×1):

| por lane, por bloco de 32 | instruções | o que é |
|---|---|---|
| `v_mul_lo_u32` | **32** | `sumi *= 1+2·sc` (inteiro, um por acumulador) |
| `v_mul_f32` | **32** | `d = d_w · d_a` (um por acumulador) |
| `v_cvt_f32_i32` | **32** | `(float)sumi` (um por acumulador) |
| `v_fmac_f32` | **32** | `acc = fma(d, sumi_f, acc)` (um por acumulador) |
| `ds_load_b128` | **2** | os 8 floats de `d_a` = **8 floats consecutivos por lane** |
| `ds_load_2addr_b64` | **2** | os 4 `int2` (`d_w`, `1+2·sc`) — 1 por tile de N, que o compilador funde 2 a 2 |
| **total** | **132** | 128 VALU + 4 LDS |

Contado na ISA do binário (o corpo do kernel contém exatamente 2 blocos de 32,
por causa do desenrolar de `BK=64`): `v_mul_lo_u32` = 64, `v_mul_f32` = 64,
`v_cvt_f32_i32` = 64 (+64 do caminho `nocorr`), `v_fmac_f32` = 64, `ds_load_b128`
= 4, `ds_load_2addr_b64` = 14 (= 20 leituras de fragmento de A/B + 8 de
`d_w`/`sc`, que o compilador fundiu duas a duas). Ou seja: **a receita da frente D
está certa na forma (8 floats consecutivos, 1 valor de `d_w` por lane, sem
gather) e subestima o custo em 3,2×** — 35 instruções por 8 acumuladores contra
as 11 previstas, porque faltavam na conta o `× (1+2·sc)` inteiro, o `d_w·d_a` em
fp32 e a conversão int→float, que são justamente as três instruções que a
bit-exatidão obriga a manter.

Custo por MAC: 128 VALU por 1024 MAC por lane por bloco de 32 = **0,125
instrução por MAC**, contra 8 WMMA (= 0,0078 issue/lane-MAC). Em contagem de
instrução o miolo é 94 % correção; em tempo ele é 53-56 %, porque o WMMA custa
~7-8 slots de issue cada (o §2 da frente D mediu 8,6).

O que a transposição de fato entrega, medido: a leitura de `d_a` sai como **2
`ds_load_b128` por lane por bloco** e a de `d_w`/`sc` como **1 `int2` por tile de
N** — 4 instruções de LDS para 32 acumuladores (64 B), e o modo 1 da frente F já
tinha mostrado que a
LDS não é o gargalo (as leituras de fragmento custam <1 % do kernel). A
transposição resolveu o que prometia; o que ela não resolve é o número de
instruções de VALU.

---

## 6. O que falhou, e o que quase virou número errado

- **Duplo buffer: −35 a −43 %.** A/B na MESMA janela, duas invocações do MESMO
  binário em sequência dentro de um único lock (picos da janela 190,74/202,44
  sem e 196,07/194,35 com):

  | M | tile | sem dbuf | com dbuf | razão | staging sem → com | fonte GB/s sem → com |
  |---|---|---|---|---|---|---|
  | 128 | BM64 BN64 W4×1 | 17,69 T | **10,52 T** | 0,59× | 61 % → 85 % | 194 → 83 |
  | 128 | BM128 BN64 W8×1 | 19,61 T | **12,71 T** | 0,65× | 57 % → 86 % | 115 → 50 |
  | 512 | BM64 BN64 W4×1 | 19,16 T | **10,86 T** | 0,57× | 60 % → 80 % | 216 → 91 |
  | 512 | BM128 BN64 W8×1 | 21,68 T | **14,16 T** | 0,65× | 55 % → 73 % | 133 → 65 |

  O duplo buffer dobra a LDS por CTA (15,9 → 31,7 KB em BM128/BN64) e derruba a
  ocupação; o staging piora em todas as células. **É o mesmo resultado da frente
  F §4** ("sobreposição entre CTAs é mais barata que pipeline dentro do CTA"),
  agora medido no kernel do WMMA — e é a razão pela qual o melhor caminho medido
  é BN grande com 3-4 CTAs/CU, e não pipeline. *(A primeira medição desta célula,
  feita antes de o pico ser corrigido, comparava janelas diferentes e dava −32 %;
  o número acima é o A/B dentro da mesma janela.)*
- **O pico medido estava 2× baixo, e isso inflava o "% do pico".** A primeira
  versão media o pico logo depois da verificação, em janelas de 0,3-0,6 ms: deu
  99,8 T-MAC/s e a tabela saiu com "21,6 % do pico" em M=512. Com aquecimento de
  pico (358 ms), janela forçada para ~4 ms e calibração iterativa, o mesmo
  binário mede **198,04 T-MAC/s** (e 197,98 no fim da mesma corrida, 1,00×) — o
  número real é **11,0 %**, não 21,6 %. A lição vale para qualquer bancada desta
  casa: o pico precisa do mesmo aquecimento de DPM que o objeto medido, senão a
  razão entre os dois mede a rampa de clock.
- **O modo "staging sozinho" não é comparável quando BM < M.** Com BM=16 em
  M=512 o modo 1 dá 7,788 ms contra 4,753 ms do GEMM (164 %), porque a
  contabilidade do staging inclui a releitura do peso por fatia de M
  (`N·K·M/BM` pesos) enquanto o GEMM só paga o tempo. Não é erro de medição, é
  uma taxa que precisa da ressalva — e é o motivo pelo qual o "conta pura =
  GEMM − staging" fica negativo nessas linhas.
- **A primeira versão escrevia fora da LDS.** Com `DBUF=false` o array tem um
  buffer e o índice alternava; o resultado foram 24 de 256 saídas não-finitas e
  `max|gpu-cpu| = 2,8` no caso pequeno. Corrigido antes de qualquer número, e é a
  razão pela qual a verificação vem antes do cronômetro nesta bancada.
- **A hipótese "BK=128 conserta o staging" não se confirmou** (§3, item 3):
  21,20 contra 21,87 T-MAC/s em M=512 e a fonte cai de 133 para 116 GB/s. A
  frente D §4 apontava BK=128/linha de cache cheia como "candidato nº 1 a
  conserto"; no kernel do WMMA ele não pagou.

---

## 7. Veredito para o degrau D4

| | T-MAC/s | % do pico de 182 | contra o llama.cpp (32,7 T) | contra o prefill de hoje (3,18 T) |
|---|---|---|---|---|
| prefill de hoje (matvec em lote) | 3,18 | 1,7 % | 0,10× | 1,00× |
| dp4a tilejado, blocos reais (esta bancada, mesma janela) | 2,67-12,56 | 1,5-6,9 % | 0,08-0,38× | 0,84-3,95× |
| f16 dot2, blocos reais (frente F, M=128/512) | 9,2-11,7 | 5,1-6,4 % | 0,28-0,36× | 2,9-3,7× |
| **WMMA int8, blocos reais (esta frente, M=128/512)** | **19,7-22,8** | **10,8-12,5 %** | **0,60-0,70×** | **6,2-7,2×** |
| WMMA int8, miolo isolado (sem staging) | 33,5-37,6 | 18,4-20,7 % | 1,03-1,15× | 10,5-11,8× |
| WMMA int8, miolo isolado sem a correção | 73,7-83,9 | 40,5-46,1 % | 2,25-2,57× | 23-26× |
| pico de WMMA int8 (mesma janela / frente D) | 198 / 182 | 109 % / 100 % | 6,1× / 5,6× | 62× / 57× |

**1. A resposta à pergunta da frente D é: sim, o WMMA int8 funciona sobre os
blocos reais e é bit-exato — e não, ele não entrega os 4,2×.** Ele entrega
**1,56-1,83× o dp4a tilejado** na mesma janela (2,82× em M=16) e **1,9-2,1× o
f16** da frente F, porque a correção de escala que a bit-exatidão exige custa
53-56 % do miolo. O teto de 4,2× do pico é real (§2 da frente D, reproduzido aqui
em 4,12× na mesma janela), mas ele é um teto de *instrução de MAC*, não de
kernel: 8 WMMA contra 132 instruções de correção por bloco de 32.

**2. O que decide o D4 não é a família, é o staging — de novo.** Ele é 50-61 % do
kernel em M≥64 e o miolo saturado é ~16-19 %. Mesmo uma correção gratuita levaria
o kernel a 32 T-MAC/s (16 % do pico), ou seja, ao empate com o llama.cpp. O
ganho de 6,2-7,2× contra o prefill de hoje vem de **M≥64 por lançamento + staging
int8 + WMMA**, e a ordem de trabalho que os números impõem é: (i) staging
(especialização de warps produtor/consumidor, que é a única saída que a frente F
apontou e que esta bancada não testou); (ii) só depois, o matrix core.

**3. O caminho barato continua sendo o de M=16.** Com BM=16 e 4 warps de um tile
de 16 em N, o WMMA int8 faz **7,63 T-MAC/s = 2,4× o prefill de hoje sem mudar o
M do lote**, com 80-82 % do tempo em staging. Se o motor não puder processar 64-128
tokens por GEMM, este é o número que ele pode embarcar; se puder, o alvo é
**19,7-22,8 T-MAC/s (0,60-0,70× o llama.cpp)** com os tiles da tabela do §3.

**4. O que os dois lados medidos juntos dão (EXTRAPOLAÇÃO, não medição).** A
frente G (`docs/estudo-prefill-g-staging2.md`) atacou exatamente o gargalo que
esta frente isolou: o staging dela caiu de 41-43 % para **17,0-18,3 % do kernel**
(0,51 → 0,16 ms em M=128; 1,47 → ~0,53 ms em M=512, ~2,8× menos tempo) e a cópia
da ativação deixou de existir, com o caminho **f16** a 12,5-14,9 T-MAC/s e a
**95 % do muro de issue** (3,60e11 de 3,80e11 slots/s). Ou seja: a frente G
removeu o staging e mostrou que o caminho vetorial f16 chegou ao fim da linha —
*o que sobra lá é remover instrução de MAC*, que é precisamente o que o WMMA faz
(8 WMMA + 132 instruções de correção por bloco de 32 por warp, contra ~512
`v_dot2_f32_f16` do mesmo trabalho).

Com os dois números na mão, o kernel combinado (staging da G + miolo desta
frente) fica em **27-34 T-MAC/s** em M=512:

```
staging da G (M=512, BM128 BN128) ......... ~0,36 ms   (2,8x menos que os 1,00 ms daqui)
miolo desta frente, sem staging ........... 0,99 ms (medido no GEMM, com sobreposição entre CTAs)
                                        ou 1,33 ms (modo 4, miolo isolado, sem sobreposição)
total ..................................... 1,35-1,69 ms -> 33,8-27,0 T-MAC/s
```

Isto é: **0,83-1,03× o prefill do llama.cpp (32,7 T)** — o empate, não a vitória.
E, nessa configuração, o próximo muro passa a ser a correção de escala, que hoje
é 0,589 ms = **30 % do kernel** e 53-56 % do miolo. A ordem que os números impõem
para o D4 é: (i) juntar as duas frentes (o staging da G com o miolo daqui);
(ii) atacar a correção, que é o único item que sobra entre o empate e o ganho.

**5. O que esta bancada não mediu** (fica para a próxima frente): a
especialização de warps no staging (produtor/consumidor), que é a única alavanca
apontada e não testada; o `ffn_down` (K=17408) e os tensores de atenção
(5120×5120), onde o paralelismo de N cai; os outros tipos do inventário
(`iq3_xxs` 17,8 % dos bytes, `iq4_xs` 9,5 %, `q5_K` 9,4 %) — o staging desta
bancada só existe para `iq3_s`; e o gate numérico ponta-a-ponta (KL/PPL), que
para o caminho int8 **não é necessário** justamente porque a bit-exatidão foi
medida (é o que a frente D §5 prometia e esta frente confirmou).
