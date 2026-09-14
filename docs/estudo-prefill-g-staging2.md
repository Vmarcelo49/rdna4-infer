# Staging, segunda volta: o que ele custa de verdade e como tirá-lo do caminho

Frente G, 14/09 (mesma bancada da frente F: `tests/bench_gemm_quant_gpu.hip`, alvo
`bench-gemm-quant-gpu`, agora com `--ab`). A frente F mediu **42 % do kernel no staging**, atribuiu
isso a **latência** (≈2 % dos slots de issue) e deixou a pergunta: *como tirar isso do caminho?*
Esta frente responde com uma tabela A/B de sete variantes do MESMO kernel, medidas na MESMA janela,
e com a contagem de instruções da ISA — que **corrige o 2 % e muda o diagnóstico**.

## TL;DR (6 linhas)

1. **12,49 T MAC/s em M=128, 14,89 T em M=512 e 9,82 T em M=64** com a melhor configuração
   (V6 = `BM128 BN128 BK64 RM=RN=8` + cópia da ativação em `int4` + prefetch do peso em registrador +
   duplo buffer só do W + ativação fora da LDS). Contra o mesmo kernel sem nenhuma dessas quatro
   mudanças, **na mesma janela**: 8,05 / 8,82-10,21 / 8,52 T — ou seja **1,55× / 1,40-1,69× / 1,15×**.
   Contra o prefill de hoje (3,18 T): **3,93× / 4,68× / 3,09×**.
2. **O staging do peso caiu de 35,3 % para 18,0 % do kernel em M=128** (0,500 → 0,165 ms) e a
   **cópia da ativação deixou de existir** (a ativação é lida direto da global pelo laço interno).
3. **Toda variante dá C BIT A BIT igual** (`bitid = 0`) e rel-L2 = 2,07e-4 contra o oráculo: nenhum
   ganho desta frente custou precisão.
4. **A ISA corrige a frente F**: o dequant f16 custa **590 instruções por sub-bloco de 32 pesos por
   lane** (18,4 por peso), não as ~55 que a frente F estimou — o staging é **~22 % dos slots de
   issue, não 2 %**, e o "43× acima do teto" era artefato dessa subestimativa de 10,7×.
5. Com essa contagem, **V6 entrega 3,60e11 instruções de warp/s = 95 % dos 3,80e11 slots/s** que as
   frentes D/F mediram como teto de issue do cartão num laço puro de `v_dot2_f32_f16`. **O caminho
   f16, nesta forma, está no muro de issue** — o que sobra não é escalonamento, é remover instrução.
6. **O staging int8 é 1,31× mais rápido que o f16 na mesma grade e com os mesmos blocos** (0,122
   contra 0,159 ms em M=128, 730 contra 561 G pesos/s, com 1,49× menos instruções e metade dos bytes
   de LDS). O GEMM int8+dp4a ponta a ponta **não** foi medido aqui — só o staging dele.

Reprodução (é a sequência exata dos números deste arquivo):

```bash
source scripts/rocm-env.sh
cmake --build build -j8 --target bench-gemm-quant-gpu
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf

# a tabela A/B inteira (tres M, verificacao + cronometragem numa passada)
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 64,128,512 --ab --reps 7

# o caminho antigo (frente F), para o absoluto de referencia
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 64,128,512 --bk 64 --reps 5

# a ISA (instrucoes por variante, spills, ordem do prefetch)
hipcc -S --offload-arch=gfx1201 -Iinclude -O3 -DNDEBUG -std=c++17 \
    tests/bench_gemm_quant_gpu.hip -o /tmp/bqg.s
```

## 1. O que a ISA diz do staging (e a correção do 2 %)

O kernel da frente F, modo 1 (*só staging*, mesma grade e mesmos threads), tem **680 instruções antes
do primeiro `s_barrier`**: 92 de prólogo, 9 do corpo do laço da ativação (16 iterações) e **590 do
staging do peso** — que é **um sub-bloco de 32 pesos por thread**. A mistura dessas 590:

| instruções | o que são |
|---|---|
| 85 `v_and_b32`, 60 `v_lshlrev_b32`, 28 `v_lshrrev_b32`, 18 `v_mad_u32_u24`, 41 `v_or_b32` | índice da LUT (`qs[2i] \| ((qh << k) & 0x100)`), endereços, extração de bytes |
| 48 `v_bfe_i32`, 32 `v_cvt_f32_i32`, 32 `v_fma_mixlo_f16` | `(float)(signed char)byte * d` e o arredondamento para f16 (o `mul`+`cvt` saem fundidos num `v_fma_mixlo_f16` por valor) |
| 48 `v_lshlrev_b16`, 32 `v_sub_nc_i16`, 16 `v_cmp_ne_u16`, 16 `v_cndmask_b32` | **a aplicação de sinal** (`__vcmpne4`/`__vsub4` são emulados por byte no gfx1201): ~112 instruções, 19 % do staging |
| 8 `global_load_b32` da LUT + 5 cargas de campo + 8 `ds_store_2addr_b32` | o acesso em si: 21 instruções de 590 |

Ou seja: **18,4 instruções por peso**, e só 21 de 590 têm a ver com memória. O staging não é um
acesso caro, é **aritmética de bit** — e o teto dele é de issue, não de banda:

| | frente F (estimado) | medido na ISA |
|---|---|---|
| instruções de lane por sub-bloco de 32 pesos | ~55 | **590** |
| instruções de warp para 8,9e7 pesos (M=128) | 4,7e6 | **5,13e7** |
| tempo a 3,80e11 slots/s | ~12 µs | **135 µs** |
| medido (staging do peso sozinho, `ag`, M=128) | 0,326 ms* | **0,159 ms** = 85 % do teto de issue |

*0,326 ms é a subtração `modo 1 − modo 2` do arquivo da frente F, não uma medida direta; o 0,159 ms
é a mesma coisa medida com `ag` (que tira a cópia da ativação da conta) e ainda inclui o
`__syncthreads` e o consumo mínimo da LDS — ou seja, o teto de 135 µs está sendo comparado com um
número que é um limite superior do trabalho de staging.

A frente F aplicou a conta ao staging **total** (0,514 ms) e concluiu 43× o teto de issue; com a
contagem da ISA o número honesto é **1,2× o teto** para o staging do peso. O que a frente F leu como
"latência pura" é, em boa parte, **issue**: o kernel estava apenas mal escalonado em cima de um custo
de instrução que já era grande.

## 2. A/B: as sete variantes, a mesma janela

Bits do template `STG` (`bench_gemm_quant_gpu.hip`, `gemm_v2_kernel`), todos no mesmo tile
`BM128 BN128 BK64 RM=RN=8` (256 threads), grade 136×1 em M=128:

| bit | nome | o que muda |
|---|---|---|
| 1 | `avec` | cópia da ativação com `int4` (16 B) em vez de `half2` (4 B): 16 → 4 iterações |
| 2 | `pf` | prefetch dos campos do peso (qs/qh/signs/d/scales) de k+BK para **registradores**, emitido antes da conta de k |
| 4 | `db` | duplo buffer **só do W** (o A fica com um buffer): a dequantização+STS de k+BK roda durante a conta de k |
| 8 | `ag` | a ativação **não vai para a LDS**; o laço interno lê direto da global (broadcast de 2 endereços por warp) |

Medido (`--ab --reps 7`; a run de `--reps 5` concorda dentro de 1-2 % em todas as linhas, exceto onde
anotado):

| M | variante | th | regs | LDS KB | CTA/CU | ms | **T MAC/s** | t_stg ms | staging % | bitid | rel-L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 0,670 | 8,52 | 0,257 | 38,4 | 0 | 2,07e-4 |
| 64 | **V6 (BM64)** | 256 | 201 | 33,0 | 1 | **0,581** | **9,82** | 0,167 | 28,7 | 0 | 2,07e-4 |
| 128 | V0 base | 256 | 256 | 33,0 | 1 | 1,418 | 8,05 | 0,500 | 35,3 | 0 | 2,07e-4 |
| 128 | V0 base (repete no fim) | — | — | — | — | 1,363 | 8,37 | 0,485 | 35,6 | — | — |
| 128 | V1 = avec | 256 | 256 | 34,5 | 1 | 1,106 | 10,31 | 0,235 | 21,2 | 0 | 2,07e-4 |
| 128 | V4 = avec+ag | 256 | 256 | 16,5 | 3 | 1,338 | 8,53 | 0,159 | 11,9 | 0 | 2,07e-4 |
| 128 | **V6 = avec+pf+db+ag** | 256 | 220 | 33,0 | 1 | **0,914** | **12,49** | 0,165 | 18,0 | 0 | 2,07e-4 |
| 128 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 1,158 | 9,85 | 0,432 | 37,3 | 0 | 2,07e-4 |
| 128 | V6 (BM64) | 256 | 201 | 33,0 | 1 | 1,014 | 11,25 | 0,287 | 28,3 | 0 | 2,07e-4 |
| 512 | V0 base | 256 | 256 | 33,0 | 1 | 5,176 / 4,303¹ | 8,82 / 10,60¹ | 1,490 | 28,8 | 0 | 2,07e-4 |
| 512 | V1 = avec | 256 | 256 | 34,5 | 1 | 3,642 | 12,53 | 0,768 | 21,1 | 0 | 2,07e-4 |
| 512 | **V6** | 256 | 220 | 33,0 | 1 | **3,065** | **14,89** | 0,527 | 17,2 | 0 | 2,07e-4 |
| 512 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 5,015 | 9,10 | 1,517 | 30,3 | 0 | 2,07e-4 |
| 512 | V6 (BM64) | 256 | 201 | 33,0 | 1 | 3,897 | 11,71 | 1,050 | 26,9 | 0 | 2,07e-4 |

¹ **A primeira célula de M=512 é a menos confiável deste arquivo**: a linha V0 medida primeiro deu
5,176 ms e a MESMA configuração repetida no fim da corrida deu 4,303 ms (na corrida de `--reps 5`:
4,469 contra 4,581). É DPM/janela compartilhada, não código — por isso a razão V6/V0 em M=512 se
reporta como **1,40× a 1,69×**. V6 (3,065 / 3,119) é estável entre as duas corridas (1,8 %).

**Leituras:**

1. **`avec` sozinho já vale 31 %** (1,418 → 1,106 ms em M=128) e derruba o staging de 35,3 % para
   21,2 %. É o ganho mais barato da tabela: 4 iterações de cópia em vez de 16, 4 pedidos de 16 B em
   vez de 16 de 4 B. O `AS` passa de `BK+2` para `BK+8` halves (16-B alinhado) e **não** cria conflito
   de banco: dentro de uma warp o A só tem 2 endereços (`tr` e `tr+1`), cada um broadcast para 16
   lanes — quem precisa de stride ímpar é o W (`W2 = BK/2+1`), e esse não mudou.
2. **`ag` sozinho não ganha nada** (8,53 contra 8,05 T) mas **libera 16,5 KB de LDS e sobe a ocupação
   para 3 CTAs/CU** e mata a cópia da ativação (que era 35 % do staging). É a peça que faz o `db`
   caber: sem ele o duplo buffer do W não tem LDS.
3. **O que decide é a combinação `pf+db+ag`**: 12,49 T contra 8,05 do V0 (mesma janela) e 10,31 do
   V1. O `pf` emite as cargas dos campos depois do `__syncthreads` e antes da conta — o compilador
   **não pode afundar a carga através da barreira**, então a janela de voo é uma iteração inteira
   (carga → conta → barreira → dequantização que a consome). A ISA do V6 confirma: a primeira
   `global_load` está no offset 45 do corpo do laço, o primeiro `v_dot2_f32_f16` em 1 367 e a
   `s_barrier` em 582 — a carga é emitida **antes** da conta e consumida depois dela, ~1 800
   instruções de vida útil. O `db` deixa a dequantização+STS de k+BK rodar durante a conta de k.
4. **BM=64 com 2 CTAs/CU deixou de ser o melhor**: com o pipeline dentro do CTA (V6) o BM=64 ganha
   de 11,25 contra 12,49 T em M=128 e de 11,71 contra 14,89 em M=512. A frente F tinha medido o
   contrário (BM=64 9,79 contra BM=128 9,23) porque lá **não havia pipeline**: a sobreposição vinha
   de fora do CTA. Com `db` a sobreposição vem de dentro e o tile maior, que gasta menos instrução
   por MAC, volta a ganhar.

## 3. Onde o kernel está agora: no muro de issue

Contabilidade por warp por iteração de K (BM=128, BK=64, RM=RN=8), toda ela contada na ISA:

| | V0 (frente F) | V6 |
|---|---|---|
| conta (`v_dot2` + cargas de A/W do laço interno) | 2 580 | 2 580 |
| staging do peso | 590 | 590 |
| cópia da ativação (16 → 4 iterações de `int4`) | 144 | 0 (ag) || **total por warp por iteração de K** | **3 314** | **3 170** |
| instruções executadas na passada de M=512 (544 CTAs × 80 iterações) | 1,154e9 | 1,104e9 |
| medido (M=512) | 4,469 ms | **3,065 ms** |
| **instruções de warp/s realizadas** | **2,58e11 (68 % do teto)** | **3,60e11 (95 % do teto)** |

(A célula de 4,469 ms é a corrida de `--reps 5`; com o 4,303 ms da repetição da corrida de
`--reps 7` o V0 fica em 2,68e11 = 71 %. A conclusão não muda: **68-71 % contra 95 %**.)

O teto de 3,80e11 slots/s é o das frentes D/F (medido num laço puro de dot2, não remedido aqui).
A leitura é direta: **V0 desperdiça 32 % dos slots de issue em stalls; V6 desperdiça 5 %**. Não
porque tem muito menos instrução (só 4,3 % menos: o `ag` tira a cópia da ativação, não a
dequantização), mas porque **para de esperar**. E é por isso que o ganho de 1,55× é maior do que a
queda de instrução — o kernel saiu de um regime dominado por latência e encostou no de issue.

**Consequência para a próxima frente: o que sobra não se resolve com escalonamento.** Os 590 do
staging do peso são 18,6 % de V6 e a conta (2 580) é 81 %; para ir além de ~15 T é preciso **remover
instrução de MAC** — a via é o dp4a (4 MAC/instrução contra 2 do `v_dot2_f32_f16`), exatamente como
a frente F concluiu pelo teto (39,0 contra 19,5 T), agora com o número de issue que diz *onde* isso
aparece: 1 024 dp4a + correção no lugar de 2 048 dot2.

## 4. Staging int8 × staging f16 (a regra 5 do pedido)

O MESMO sub-bloco, a MESMA grade, os MESMOS bytes de fonte (38,3 MB por passada em M=128) e o mesmo
consumo mínimo na LDS; só o destino muda: `f16` (64 B por sub-bloco, com `cvt`+`mul`+`pack`) contra
`int8` (32 B, magnitude 1..15 do grid com o sinal por `__vsub4` e o `1+2*sc` **fora** do byte, que é
a convenção do `vec_dot_iq3_s_q8_1` — `vecdotq.cuh:725` multiplica em int32 e trocar a ordem muda o
arredondamento). As duas linhas com `ag` para que a cópia da ativação não entre na conta:

| M | f16 na LDS | int8 na LDS | ganho | G pesos/s f16 | G pesos/s int8 | GB/s de fonte f16 → int8 | LDS escrita f16 → int8 |
|---|---|---|---|---|---|---|---|
| 128 | 0,159 ms | **0,122 ms** | **1,31×** | 561 | 730 | 241 → 314 | 1 121 → 730 GB/s |
| 512 | 0,495 ms | **0,368 ms** | **1,35×** | 720 | 969 | 309 → 416 | 1 440 → 969 GB/s |

Na ISA a diferença é maior do que no relógio — **396 instruções contra 590** para o mesmo sub-bloco
(1,49×), com 3 cargas de campo em vez de 5 (o `d` e a escala não são necessários no byte), **2
`ds_store_b128` em vez de 8 stores** e **zero** `v_cvt_f32_i32`/`v_fma_mixlo_f16`. O que sobra nos
dois é a mesma emulação de sinal (~112 instruções): ela é 19 % do staging f16 e 28 % do int8.

Contra o roofline de 633 GB/s medido do cartão (`docs/medicoes-banda-e-gargalos.md`), o staging int8
em M=512 roda a **416 GB/s = 66 %** — ou seja, o staging int8 está mais perto de ser banda do que o
f16 (309 GB/s = 49 %), e ao mesmo tempo é 1,35× mais rápido.

**O que isto NÃO é:** não é o GEMM int8 ponta a ponta. Falta o laço interno em dp4a com a correção
por bloco de 32 (`acc += (d_w*d_a) * (sumi * (1+2*sc))`, com o `d_a` vindo de uma ativação Q8_1) —
que é o que faria o número final subir, e não foi medido nesta frente.

## 5. O que falhou (e o número que mostra por quê)

| variante | M=128 | veredito |
|---|---|---|
| V2 = `avec+pf` (sem `ag`, sem `db`) | **8,745 ms (1,30 T)** | −87 % contra o V0 |
| V3 = `avec+pf+db` (sem `ag`) | **8,337 ms (1,37 T)** | −83 % |
| V5 = `avec+pf+ag` (sem `db`) | 2,059 ms (5,54 T) | −46 % contra o V1 |

A causa está na ISA e não é sutil: **spills**. O campo prefetchado (`wf`, ~6-8 registradores) fica
vivo através do laço interno de ~2 300 instruções e o alocador de registradores **estoura o teto de
256 VGPRs** — `RM=RN=8` são 64 acumuladores + 16 operandos + endereços:

| variante (BM128 RM8) | instruções com `scratch_*` | regs |
|---|---|---|
| V0 base | 4 | 256 (teto) |
| V1 `avec` | 4 | 256 |
| V2 `avec+pf` | **413** | 256 |
| V3 `avec+pf+db` | **412** | 256 |
| V5 `avec+pf+ag` | 158 | 256 |
| **V6 `avec+pf+db+ag`** | **0** | 220 |

O `ag` é o que abre espaço: sem o tile de A não existem os endereços por linha (`tr + r*TM`), e o
`db` encurta a vida do `wf` (ele é consumido no topo da iteração seguinte, logo depois da barreira).
Com os dois, o alocador cabe em 220 registradores e **não há um único acesso a scratch**. É um
resultado de método que vale registrar: *a mesma transformação que dá +55 % numa forma dá −87 % na
outra, e a diferença é pressão de registrador* — a razão pela qual a frente F não viu ganho nenhum
do duplo buffer (`--dbuf`: +5 % em BK=32, −17 % em BM=64) é que lá o duplo buffer dobrava a LDS e
derrubava a ocupação; aqui ele não dobra o A e ainda libera registrador.

Falhou também, e vale como resultado: **mais paralelismo clássico não era a alavanca**. A linha V4
(`ag`, 3 CTAs/CU contra 1) anda *menos* que o V1 de 1 CTA/CU (8,53 contra 10,31 T) — a ativação lida
da global dentro do laço interno custa mais do que a cópia economizada; ela só passa a valer quando
o `db` a transforma em sobreposição (V6). E o BM=64 com 2 CTAs/CU, que era o achado da frente F,
perde para o BM=128 em V6 (item 4 do §2).

## 6. O que não foi medido

- **O GEMM int8+dp4a com os blocos reais** (o §4 mede só o staging dele). É a única via que a
  contabilidade do §3 aponta para passar de ~15 T.
- **O split A × W por modo**: o modo 2 (só a cópia da ativação) continua sendo a medição frágil que a
  frente F já tinha sinalizado (0,190 ms aqui); o número robusto é o staging do peso sozinho, medido
  com `ag` no modo 1 (0,159 ms em M=128) — e mesmo ele inclui o `__syncthreads` e o consumo mínimo.
- **O teto de issue por SIMD deste cartão não foi remedido**: os 3,80e11 slots/s são herdados das
  frentes D/F e toda a leitura do §3 depende deles [DERIVADO].
- **`ffn_down` (K=17408) e os tensores de atenção** (5120×5120), onde o paralelismo cai: a frente F
  também não mediu, e nada aqui muda isso.
- **A ordem de comparação entre variantes dentro de uma corrida**: a primeira célula de M=512 saiu
  20 % mais lenta que a repetição da MESMA configuração no fim da mesma corrida (§2, nota 1). Todas
  as razões deste arquivo usam linhas da mesma corrida, e as duas corridas independentes concordam
  dentro de 1-2 % em todas as linhas menos essa.

## Veredito para o motor

1. **O que embarcar, medido:** `BM=128 × BN=128 × BK=64` (RM=RN=8, 256 threads) com (i) **ativação
   lida da global no laço interno** (não estagiar A na LDS), (ii) **duplo buffer só do W**, (iii)
   **prefetch dos campos do peso em registrador** com o `__syncthreads` entre a carga e o uso, (iv)
   cópia da ativação em `int4` quando ela ainda existir (outros tensores, `act_quant`). Isso dá
   **12,5 T em M=128 e 14,9 T em M=512**, contra 3,18 T do prefill de hoje: **3,9× a 4,7×**.
2. **O staging deixa de ser 42 %:** em M=128 ele é **18,0 %** (0,165 ms de 0,914 ms) e é só o peso —
   a metade da ativação simplesmente não existe mais nesse desenho. Em M=64 sobra 28,7 % e em M=512
   17,2 %.
3. **A ordem do trabalho, agora com número:** o kernel f16 está a **95 % do teto de issue** do
   cartão. Não há mais escalonamento a ganhar; o que resta é (a) o dp4a, que dobra MAC/instrução e
   é o único caminho para além de ~15 T, e (b) o staging int8, que é 1,31× mais rápido que o f16 e
   já foi medido aqui. As duas peças juntas — staging int8 + laço dp4a com a correção inteira por
   bloco de 32 — são a frente seguinte, e nenhuma delas é ajuste de tile.
4. **O que NÃO fazer:** prefetch de peso sem tirar a ativação da LDS (V2/V3: −87 %, por spill), e
   acrescentar CTAs por CU confiando só na ocupação (V4: 3 CTAs/CU e *menos* taxa que 1 CTA/CU com
   pipeline).
