# GEMM com os blocos reais e staging em f16

Frente F do `docs/plano-prefill.md` (degrau **D2**), 14/09 noite. Bancada nova:
`tests/bench_gemm_quant_gpu.hip`, alvo `bench-gemm-quant-gpu` (o alvo já existia no
`CMakeLists.txt`). O que ela cobre e que a `bench-gemm-gpu` (frente P0, dado sintético int8) não
cobria: **os blocos reais do modelo** dequantizados **para f16 na LDS** e o laço interno em
`v_dot2_f32_f16` com **acumulador f32** — o veredito da frente B §7.6.

Tensor medido (escolhido automaticamente pela forma que o plano cita, K=5120 × N=17408):
`blk.3.ffn_up.weight`, iq3_s, 38,3 MB, **0,4297 B/peso** (110 B por 256 pesos). K=5120 = 20
super-blocos; N=17408 = 136 colunas de 128.

## TL;DR (5 linhas)

1. **9,23 T MAC/s em M=128** com os pesos reais (BM=BN=128, BK=64, RM=RN=8, 256 threads, dot2 f32) =
   **2,90× o nosso prefill (3,18 T)**; melhor tile medido em M=128: **9,79 T** (BM=64, 1,165 ms). Em
   M=16: **4,45 T** (0,320 ms); em M=64: **8,29 T**; em M=512: **11,73 T** (3,69×).
2. **O staging real custa 0,514 ms por passada do `ffn_up` em M=128 = 42 % do kernel** (173,5 G
   pesos/s, 74,6 GB/s de fonte contra 633 GB/s medidos do cartão) e **não** é banda nem instrução:
   pelos números ele ocupa ~2 % dos slots de issue — é **latência**.
3. **f16-dot2 NÃO ganha do int8-dp4a no caminho vetorial** (A/B na mesma janela): 9,23 contra 10,11 T
   em M=128 (−8,7 %), 11,73 contra 12,76 em M=512 (−8,1 %), 4,45 contra 3,62 em M=16 (+23 %).
4. **O miolo f16 está no muro dele**: a conta pura isolada dá 18,84 T em M=512 = **97 % dos 19,5 T**
   do teto de issue do `v_dot2_f32_f16` neste tile (0,625 instrução/MAC); o dp4a entrega 12,76 T =
   **33 %** do teto dele (39,0 T) — ou seja, o headroom que existe está no dp4a, não no f16.
5. **Precisão medida**: o staging reproduz o dequant do motor **bit a bit** depois do arredondamento
   f16 (0 de 2048 bits diferentes), erro relativo máximo 4,32e-4 = 2⁻¹¹; ponta a ponta o GEMM fica
   com **rel-L2 = 2,07e-4**, igual em todos os M.

Reprodução completa (é a sequência exata dos números deste arquivo):

```bash
source scripts/rocm-env.sh
cmake --build build -j8 --target bench-gemm-quant-gpu
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf

# itens 1-4 numa passada (com verificação): staging, GEMM, correção e o split
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 16,64,128,512 --bk 64 --reps 5

# variações de tile em M=128 (ocupação de LDS x paralelismo, duplo buffer)
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 128 --bm 64 --bk 64  --reps 5
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 128      --bk 32  --reps 5
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 128      --bk 32 --dbuf --reps 5
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 128 --bm 64 --bk 32 --reps 5

# o A/B int8+dp4a, na MESMA janela e com as mesmas formas (frente P0)
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-gpu --m 16,64,128,512 --bk 64 \
    --n 17408 --k 5120 --reps 5
```

## 1. Staging sozinho: iq3_s real → f16 na LDS

Três modos do MESMO kernel (mesma grade, mesmos threads, mesmo código de staging; só o que vem
depois muda), o que é o que permite atribuir a diferença ao staging e não a outro knob:

| modo | o que faz |
|---|---|
| 1 | ativação f16 → LDS **e** peso real → LDS (é o "staging sozinho" do item 1) |
| 2 | só a cópia da ativação f16 (isola a outra metade) |
| 0 | staging + conta (o GEMM) |

Consumo mínimo no lugar da conta: 8 leituras de LDS com índice dependente de `tid`, somadas num
registrador que só é escrito sob uma guarda impossível — assim o compilador não pode provar que a
LDS é desnecessária e apagar o staging (era o risco desta medição virar ficção).

| M | tile | t_staging (modo 1) | G pesos/s | GB/s fonte | GB/s LDS (escrita) | G pesos/s por CTA | B/peso no LDS |
|---|---|---|---|---|---|---|---|
| 16 | BM16 BN128 BK64 | 0,214 ms | **417,2** | 179,3 | 834,4 | 3,07 | 2,0 |
| 64 | BM64 BN128 BK64 | 0,255 ms | **349,2** | 150,1 | 698,5 | 2,57 | 2,0 |
| 128 | BM128 BN128 BK64 | 0,514 ms | **173,5** | 74,6 | 347,0 | 1,28 | 2,0 |
| 128 | BM64 × 2 fatias, BK64 | 0,441 ms | **404,6** (2× o peso) | 173,8 | 1 394,0 | — | 4,0 |
| 128 | BM128, BK32 | 0,370 ms | **240,2** | 103,2 | 461,5 | — | 2,0 |
| 512 | BM128 × 4 fatias, BK64 | 1,468 ms | **242,8** (4× o peso) | 104,3 | 485,6 | 0,45 | 8,0 |

Leituras (todas com o comando da tabela acima, `--reps 5`, mínimo de 5):

1. **O staging não está perto de nenhum teto de recurso.** Fonte: 74,6-179,3 GB/s contra **633 GB/s**
   medidos do cartão (`docs/medicoes-banda-e-gargalos.md`) — 12-28 %. LDS escrita: 347-1 394 GB/s
   contra a premissa de 19,7 TB/s da frente B §6.2 — 2-7 %. Instrução: a conta é ~55 instruções de
   lane por sub-bloco de 32 pesos (`stage_iq3s_32`: 2 cargas de `qs`, 1 de `qh`, 1 de `signs`, 1 de
   escala, 8 cargas na LUT, 2 `__vcmpne4`, 2 `__vsub4`, 16 `cvt`, 16 `mul`, 16 `pack`, 16 escritas de
   4 B), ou seja **4,7e6 instruções de warp para 8,9e7 pesos ≈ 12 µs a 3,8e11 slots/s**: o medido é
   **0,514 ms, 43× o teto de issue**.
2. **O que limita é a latência da cadeia** `qs → índice da LUT → carga na LUT (L1) → cvt → mul →
   pack → store`, com poucas cadeias independentes por thread (uma, no tile BM=128: 256 threads para
   256 sub-blocos por iteração de K ⇒ exatamente 1 sub-bloco por thread por iteração). Com 1 CTA por
   CU em BM=128 (LDS 33,8 KB de 65,5 KB) não há outro CTA para preencher os buracos: por iteração de K
   são ~21 000 ciclos para 440 instruções de warp.
3. **A alavanca medida é a ocupação, não a largura de carga**: com BM=64 (LDS 25,3 KB ⇒ 2 CTAs/CU e
   272 CTAs em vez de 136) o staging passa de 173,5 para **404,6 G pesos/s**; com BK=32 em BM=128
   (LDS 17,4 KB ⇒ 3 CTAs/CU) passa de 173,5 para 240,2 G pesos/s. Em todos os casos o byte por peso
   estagiado é o mesmo (0,43 B lidos da global, 2 B escritos no LDS).
4. **A cópia da ativação não é de graça**: no modo 2 (só ativação, BM=128) o tempo é 0,178-0,190 ms —
   35-37 % do staging total. Ela move 136 CTAs × 128×5120×2 B = **178 MB por passada** (o tensor de
   ativação tem 1,3 MB, então vem de L2) a ~940 GB/s. No caminho novo isso substitui o `act_quant`
   (`docs/estudo-prefill-c-nosso.md`: 0,227 ms/token), mas não é zero. *Ressalva de método: o split
   A × W depende do codegen de um micro-kernel limitado por latência — duas versões do mesmo código
   deram 0,010 ms e 0,190 ms para a mesma cópia de ativação; o número robusto é o total do modo 1.*

## 2. GEMM f16 tilejado com os blocos reais

`C[M][N] = A_f16[M][K] × W_dequant[N][K]`, N=17408, K=5120 (a forma real do `ffn_gate/up`), tile por
CTA, `RM × RN` acumuladores f32 por thread, A e W na LDS, laço interno em `v_dot2_f32_f16`
(`__builtin_amdgcn_fdot2`, confirmado no `.s`: uma instrução por dot2, acumulador f32 no terceiro
operando). Grade: 136 CTAs por fatia de M.

| M | tile | ms | **T MAC/s** | TOPS | × o nosso prefill (3,18 T) | × llama.cpp s/ coopmat (12,26 T) |
|---|---|---|---|---|---|---|
| 16 | BM16 BN128 BK64 RM2 RN8 (128 th) | 0,320 | **4,45** | 8,9 | 1,40× | 36 % |
| 64 | BM64 BN128 BK64 RM4 RN8 | 0,688 | **8,29** | 16,6 | 2,61× | 68 % |
| **128** | BM128 BN128 BK64 RM8 RN8 | 1,236 | **9,23** | 18,5 | **2,90×** | 75 % |
| 128 | BM64 BN128 BK64 (2 fatias) | **1,165** | **9,79** | 19,6 | **3,08×** | 80 % |
| 128 | BM128 BK32 | 1,242 | 9,19 | 18,4 | 2,89× | 75 % |
| 128 | BM128 BK32 + duplo buffer | 1,178 | 9,69 | 19,4 | 3,05× | 79 % |
| 512 | BM128 BN128 BK64 (4 fatias) | 3,891 | **11,73** | 23,5 | 3,69× | 96 % |

O melhor tile em M=128 é BM=64 e ele se repete entre janelas dentro de 1 % (9,79 T com verificação
completa; 9,89 T numa janela posterior, com o mesmo binário). A geometria menor tem teto de issue
menor — BM=64/RM=4/RN=8 gasta 44 instruções por 2 048 MAC (46,5 MAC/instrução ⇒ 17,7 T) contra 51,2
do BM=128/RM=RN=8 (⇒ 19,5 T) — mas realiza uma fração maior dele: **55 % contra 47 %**, que é o
efeito da ocupação descrito no item 1.

A/B com o protótipo int8+dp4a rodado **na mesma janela de lock, mesmas formas** (a coluna int8 é
`./build/bench-gemm-gpu --m 16,64,128,512 --bk 64 --n 17408 --k 5120 --reps 5`, que verifica cada
configuração contra oráculo antes de cronometrar; ele usa dado sintético e por isso é o teto do
int8, não a previsão do motor):

| M | int8 dp4a (mesma janela) | f16 dot2 (real) | f16/int8 | doc. da frente P0 (outra janela) |
|---|---|---|---|---|
| 16 | 3,62 T | **4,45 T** | **1,23×** | 3,46 T |
| 64 | 8,86 T | 8,29 T | 0,94× | 7,23 T |
| 128 | 10,11 T | 9,23 T (BM=64: 9,79) | **0,91×** | 10,34 T |
| 512 | 12,76 T | 11,73 T | 0,92× | 12,74 T |

Três leituras:

1. **Em M=128 o f16 com os blocos reais é 9 % mais lento que o int8 sintético** (0,91×), e 3 % mais
   lento que o próprio número da frente P0 (10,34 T) — que é o número que o `docs/estudo-prefill.md`
   tabelou como "84 % do llama.cpp sem coopmat". O ganho de 2,9-3,1× contra o nosso prefill se
   reproduz; a troca de família, não.
2. **A aritmética explica**: com `RM=RN=8` o tile gasta 80 instruções de warp por 4 096 MAC (16
   cargas de half2 + 64 dot2), ou seja **51,2 MAC por instrução**; como o `v_dot2_f32_f16` faz 2
   MAC/lane contra os 4 do `v_dot4_i32_iu8`, o teto de issue do f16 é **19,5 T** contra **39,0 T** do
   int8 a 2,97 GHz (3,80e11 slots/s). A "conta pura" medida (GEMM − staging, §4) dá **18,84 T em
   M=512 = 97 % do teto do f16**: o miolo f16 está no muro. O protótipo int8, com 12,76 T, está a
   33 % do teto dele — está limitado por outra coisa (carga/LDS/latência), não por issue.
3. **M=16 não é o alvo, mas o f16 é o melhor caminho lá**: 4,45 contra 3,62 T (+23 %). O motivo é o
   mesmo do item 1 — em BM=16 o staging tem 2 sub-blocos por thread (mais ILP por thread) e o tile
   é estreito o bastante para 2-3 CTAs por CU.

## 3. Correção (dois oráculos, e a cadeia entre eles)

A verificação é encadeada, e cada elo é medido contra o seguinte. Nada é redigitado: a LUT
`iq3s_grid` é **copiada do device** (`copy_lut_kernel` lê `rdna4::iq3s_grid`) e usada pelo oráculo de
CPU, o dequant do oráculo de GPU é o `rdna4::dequantize_iq3_s` **do motor** (`include/rdna4/dequant.cuh`),
e o staging usa a mesma álgebra do `vec_dot_iq3_s_q8_1` (`__vcmpne4`/`__vsub4`, `scales[s/2]`,
`qh[s]`, `signs[4s..4s+4)`).

**(a) oráculo de CPU (M=4, N=64, K=5120; bytes crus do GGUF, dequant do motor, acumulador em fp64)
contra o oráculo de GPU (um warp por saída, `dequantize_iq3_s`, acumulador fp32):**

```
max |ref| = 3,63593 ; max |gpu-cpu| = 9,47e-07 ; max rel = 1,55e-04
```

9,5e-07 absoluto sobre somas de 5120 termos (o CPU soma em fp64 e o GPU em fp32, em outra ordem):
é ruído de arredondamento fp32, não divergência de algoritmo.

**(b) staging isolado contra o oráculo de CPU (2048 pesos da linha 0, comparados um a um, não em
soma):**

```
max |w| = 0,03362 ; max |dw| = 1,07e-05 ; max rel = 4,32e-04   (2^-11 = 4,88e-04)
bits diferentes do arredondamento f16 esperado (RN, feito no host): 0 de 2048
```

Ou seja: **o valor que entra na LDS é exatamente `half_rn(dequant_do_motor(peso))`** — o staging não
introduz nenhum erro além do arredondamento para f16 da frente B §7.1, e o erro relativo máximo
coincide com o 2⁻¹¹ = 4,9e-4 previsto lá.

**(c) kernel tilejado contra o oráculo de GPU, a matriz C INTEIRA (M×17408), em todos os M:**

| M | rel-L2 | max abs | max rel (só onde \|ref\| > 5 % do máximo) | max \|ref\| |
|---|---|---|---|---|
| 16 | **2,07e-04** | 1,34e-03 | 3,23e-03 | 7,08 |
| 64 | **2,07e-04** | 1,48e-03 | 3,68e-03 | 7,11 |
| 128 | **2,07e-04** | 1,48e-03 | 3,68e-03 | 7,11 |
| 512 | **2,07e-04** | 1,48e-03 | 3,68e-03 | 7,27 |

O **rel-L2 é o número a usar** (é o gate que a frente B §7.6 propõe): 2,07e-4, idêntico em todos os
M, o que confirma que ele é dominado pelo arredondamento do peso (aleatório, ~4,9e-4/√3 ≈ 2,8e-4) e
não pela ordem das somas. O "max rel" de 3,7e-3 é o pior caso de cancelamento (um elemento cuja soma
quase se anula) e não deve ser lido como precisão. Contra o erro da própria quantização (1-5 % nos
tipos de 3-4 bits, frente B §7.4) o f16 continua 1-2 ordens de grandeza abaixo; a confirmação
ponta-a-ponta (KL/PPL) continua sendo o gate do `PLAN.md` M2/M3, e não está medida aqui.

**Dois erros de método que esta bancada pegou, e que valem como aviso** (regra da casa: "verificar
antes de cronometrar"):

1. `rdna4::dequantize_iq3_s` escreve nas posições `32*(lane%8) + 8*(lane/8) + [0,8)` **do super-bloco
   inteiro** — o `yy` tem de ser um buffer de 256 floats, não um `float w[8]` por lane. Com 8 floats
   o kernel estourava a pilha e o oráculo devolvia **NaN em 59 961 de 69 632 saídas**. Sem o oráculo
   de CPU e sem o contador de não-finitos ("não bate" não diz nada) isso passaria como "erro de f16".
2. O **primeiro** caso medido num processo sai ~47 % mais lento que os seguintes na mesma
   configuração (1,757 contra 1,195 ms com `--m 128,128,128`); do segundo em diante a repetição
   concorda em 0,1 %. A bancada agora faz **600 ms de aquecimento global antes do laço** e 250 ms por
   modo. *Clock: a amostragem de `/sys/class/drm/card1/device/pp_dpm_sclk` durante a janela deu
   3 200-3 400 MHz (a tabela DPM do mesmo arquivo lista 500/1798/2400 — a amostra não bate com a
   tabela e serve só como indicação de clock alto); o teto de 19,5 T a 2,97 GHz usado aqui é
   confirmado por baixo pelo medido (18,84 T de conta pura).*

## 4. Onde vai o tempo, e o veredito f16 × int8-dp4a

Medido no mesmo run, com o mesmo kernel em três modos (item 1). "Conta" = GEMM − staging.

| M | t_GEMM | t_staging | conta (por subtração) | staging % do GEMM | T MAC/s de conta pura |
|---|---|---|---|---|---|
| 16 | 0,320 ms | 0,214 ms | 0,107 ms | **67 %** | 13,35 |
| 64 | 0,688 ms | 0,255 ms | 0,433 ms | 37 % | 13,18 |
| 128 | 1,236 ms | 0,514 ms | 0,723 ms | **42 %** | 15,79 |
| 128 (BM=64) | 1,165 ms | 0,441 ms | 0,724 ms | 38 % | 15,75 |
| 512 | 3,891 ms | 1,468 ms | 2,423 ms | 38 % | **18,84** |

1. **Staging e conta se SERIALIZAM** na forma de buffer único: 0,514 + 0,723 = 1,237 contra 1,236 ms
   medidos — a soma bate com o total, não há sobreposição nenhuma.
2. **Duplo buffer não resolve** (e isso é medido, não inferido): com `--dbuf` o staging de k+1 é
   emitido antes da conta de k, com uma barreira por iteração em vez de duas — o pipeline clássico de
   software. Par a par na mesma janela: em **BM=128/BK=32** o ganho é **+5 %** (1,239 → 1,178 ms;
   9,21 → 9,69 T); em **BM=64/BK=64** é **−17 %** (1,153 → 1,383 ms; 9,89 → 8,25 T), porque o duplo
   buffer dobra a LDS por CTA (50,7 KB) e derruba a ocupação de 2 para 1 CTA/CU. **Sobreposição
   entre CTAs é mais barata que pipeline dentro do CTA**: é a mesma LDS que serve às duas coisas, e
   com 2 CTAs/CU o staging de um CTA cai dentro da conta do outro.
3. **Em M=128, o f16-staged é MAIS LENTO que o protótipo int8-dp4a: 9,23 contra 10,11 T** (e o melhor
   tile f16, BM=64, empata em 9,79). A diferença é pequena mas o sinal é consistente em M=64, 128 e
   512; o f16 só ganha em M=16. Isso não contradiz a frente B §7.6 — ela decidiu por f16 **por
   precisão, por código e por ser o layout que o coopmat consome**, não por TOPS; e o §7.6 diz
   explicitamente que "entre as duas famílias vetoriais a decisão não move o ponteiro". A medição
   confirma isso e mostra o muro: **o dot2 satura issue a 19,5 T e nós chegamos a 18,84 T; o dp4a
   tem teto de 39,0 T e o protótipo para em 12,76 T.**

Extrapolação para o motor no degrau D2 (o `ffn_up` medido é 1 dos 497 tensores do tronco; tensores
menores têm menos paralelismo, então isto é o **teto otimista**):

| | taxa | 25,622e9 pesos por passada | prefill (128 tokens) |
|---|---|---|---|
| nosso matvec em lote hoje | 3,18 T | 1 031 ms | 124 tok/s (medido) |
| **f16-staged (este trabalho, M=128)** | **9,79 T** | **335 ms** | **~382 tok/s** |
| int8-dp4a tilejado (protótipo, M=128) | 10,11 T | 324 ms | ~395 tok/s |
| f16-staged em M=512 | 11,73 T | 279 ms | — |

O alvo do D2 no `docs/plano-prefill.md` era ~400-480 tok/s assumindo 10,34 T; com o que se mede
aqui (9,2-9,8 T em M=128) o alvo realista é **~380-395 tok/s**, ou **3,0-3,2× o prefill de hoje** —
e o chunk de 128 é pré-requisito, porque em M=16 o mesmo kernel dá 4,45 T (1,40×).

## Veredito para o degrau D2

**1. f16-staged contra int8-dp4a: empate técnico com o f16 ligeiramente atrás — o número que
decide é 9,23 contra 10,11 T MAC/s em M=128, na mesma janela e com as mesmas formas.** Não há ganho
de velocidade em trocar a família no caminho vetorial; o teto de issue do `v_dot2_f32_f16` (19,5 T)
é metade do `v_dot4_i32_iu8` (39,0 T) e o miolo f16 já está em 97 % do dele. **Escolher f16 pelo
desempenho seria trocar um muro de 39 T por um muro de 19,5 T já encostado.**

**2. O que justifica f16 mesmo assim (e é decisão de arquitetura, não de TOPS):** (i) é o formato que
o `coopmat`/WMMA f16→f32 consome, então o D3 não precisa de um segundo staging nem de um segundo
tile — o mesmo layout que a frente F mediu é o que a unidade de matriz come; (ii) mata o
`act_quant` medido pela frente C (0,227 ms/token = ~29 ms por passada de 128 tokens ≈ 9 % dos 335 ms
do GEMM); (iii) precisão: 0 erro de arredondamento além do f16 e rel-L2 = 2,07e-4 no GEMM real.

**3. O que decide a ordem do trabalho não é a família, é o staging.** Ele custa 38-42 % do kernel em
M≥64, é limitado por **latência** (~2 % dos slots de issue, 12-28 % da banda), e o duplo buffer não
o esconde (+5 % no melhor caso, −17 % quando consome LDS demais). O que o melhora, medido, é
**paralelismo**: BM=64 → 2 CTAs/CU → 404,6 G pesos/s contra 173,5 G em BM=128 (1 CTA/CU).

**4. O que o motor precisa ter para embarcar o D2 (tudo medido aqui):**

| peça | valor medido/recomendado |
|---|---|
| **tile** | `BM=64 × BN=128 × BK=64` (RM=4, RN=8, 256 threads): **9,79 T** em M=128, o melhor medido. `BM=128 × BN=128 × BK=64` (RM=RN=8) dá 9,23 T e é o mesmo desenho do protótipo int8 (comparação direta) |
| **orçamento de LDS** | **≤ 32 KB por CTA** para caber 2 CTAs/CU: BM=64/BK=64 = 25,3 KB (2 CTAs); BM=128/BK=64 = 33,8 KB (1 CTA, e é onde o staging desaba); BM=128/BK=64 **não cabe** em duplo buffer (67,6 KB > 65 536 B) |
| **staging** | um sub-bloco de 32 pesos por thread por iteração de K (`stage_iq3s_32`), com a LUT `iq3s_grid` (2 KB, residente em L1) e a escala `d*(1+2*s)` aplicada **em fp32** com um único arredondamento na conversão; **não** gastar LDS em duplo buffer (medido: não paga); se der, 2 sub-blocos independentes por thread (mais ILP) ou especialização de warps, porque a cadeia `qs → LUT → cvt → pack → store` é o que trava |
| **escala** | `scales[s/2] >> 4*(s%2)` e `qh[s]`, `signs[4s..4s+4)` — a convenção do `dequantize_iq3_s`/`dequantize_row_iq3_s`; `signs` entra pelo truque `__vcmpne4`+`__vsub4` do `vecdotq.cuh` (2 instruções por 4 bytes assinados) |
| **laço interno** | `v_dot2_f32_f16` via `__builtin_amdgcn_fdot2` (acumulador **f32**; `v_pk_fma_f16` está fora por acumular em f16), padding de 1 half2 por linha da LDS (stride ímpar) e `tr = tid/TN, tc = tid%TN`, que é o que evita conflito de banco nas duas cargas |
| **ativação** | f16 (não `q8_1`): deixa o `act_quant` fora e é o formato do coopmat. A cópia da ativação para a LDS custa 178 MB por passada (L2) e aparece como 35 % do tempo de staging em BM=128 |
| **gate numérico** | rel-L2 ≤ ~1e-3 contra o caminho atual (`check-regression-gpu` + `compare_ppl.sh` do `PLAN.md` M2/M3): o medido é 2,07e-4 |

**5. Onde o D2 para, e o que sobra para o D3.** Com os 9,2-9,8 T medidos, o D2 entrega 3,0-3,2×
(≈124 → 380-395 tok/s) e chega a 75-80 % do llama.cpp **sem coopmat**. Passar disso não é ajuste de
tile: o caminho vetorial f16 está a 97 % do teto dele e o int8 a 33 % do dele — as duas rotas
terminam na mesma ordem de grandeza (~10-13 T), e o fator que falta é o **2,50× da unidade de
matriz** já medido no llama.cpp. A boa notícia é que o staging que a frente F mediu (f16 na LDS, 0
bit de erro além do arredondamento) é exatamente o que o `coopmat` f16→f32 consome, então o D3
reaproveita o D2 inteiro e só troca o corpo do laço interno.

**O que não foi medido aqui (e ficaria para a próxima frente):** o `ffn_down` (K=17408, 68
super-blocos, 5120 linhas — N menor, menos colunas de CTA) e os tensores pequenos de atenção
(5120×5120, 26,2e6 pesos), onde o paralelismo cai; os outros tipos de peso do inventário (iq3_xxs
17,8 % dos bytes, iq4_xs 9,5 %, q5_K 9,4 % — a frente F só tem o staging de iq3_s); a KL/PPL
ponta-a-ponta do prefill em f16 contra o de hoje (o gate do `PLAN.md` M2/M3); e a especialização de
warps no staging (produtor/consumidor), que é a única saída que a medição aponta para os 42 % de
staging sem pagar LDS por um duplo buffer.
