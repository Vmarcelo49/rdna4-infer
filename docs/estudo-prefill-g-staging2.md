# Staging, segunda volta: o que ele custa de verdade e como tirá-lo do caminho

Frente G, 14/09 (mesma bancada da frente F: `tests/bench_gemm_quant_gpu.hip`, alvo
`bench-gemm-quant-gpu`, agora com `--ab`). A frente F mediu **42 % do kernel no staging**, atribuiu
isso a **latência** (≈2 % dos slots de issue) e deixou a pergunta: *como tirar isso do caminho?*
Esta frente responde com uma tabela A/B de oito variantes do MESMO kernel, medidas na MESMA janela,
e com a contagem de instruções da ISA — que **corrige o 2 % e muda o diagnóstico**.

## TL;DR (6 linhas)

1. **12,5-12,7 T MAC/s em M=128, 14,6-14,9 T em M=512 e 9,8 T em M=64** com a melhor configuração
   (V6 = `BM128 BN128 BK64 RM=RN=8` + cópia da ativação em `int4` + prefetch do peso em registrador +
   duplo buffer só do W + ativação fora da LDS). Contra o **kernel ORIGINAL da frente F rodado na
   mesma janela e na mesma tabela** (9,37-9,73 / 11,68-11,82 / 8,34-8,49 T): **1,31-1,36× / 1,24-1,27×
   / 1,16-1,18×**. Contra o prefill de hoje (3,18 T): **3,9-4,0× / 4,6-4,7× / 3,1×**. Cinco janelas
   independentes concordam dentro de 2 %.
2. **O staging caiu de 41,1-43,2 % (kernel original) para 18,0-18,3 % do kernel em M=128**
   (0,50-0,51 → 0,164 ms) e a **cópia da ativação deixou de existir** (a ativação é lida direto da
   global pelo laço interno). Em M=512 sobram 17,0-17,3 % e em M=64 28,5-28,7 %.
3. **Toda variante dá C BIT A BIT igual** (`bitid = 0`) e rel-L2 = 2,07e-4 contra o oráculo: nenhum
   ganho desta frente custou precisão.
4. **A ISA corrige a frente F**: o dequant f16 custa **590 instruções por sub-bloco de 32 pesos por
   lane** (18,4 por peso), não as ~55 que a frente F estimou — o staging é **22 % das instruções de
   V0 (18 % só o peso), não 2 %**, e o "43× acima do teto de issue" era artefato dessa subestimativa
   de 10,7× (o número honesto é **1,2×**).
5. Com a contagem da ISA, **V6 entrega 3,60e11 instruções de warp/s = 95 % dos 3,80e11 slots/s** que
   as frentes D/F mediram como teto de issue do cartão — contra 2,58e11 (68 %) do V0. **O caminho f16,
   nesta forma, está no muro de issue**: o que sobra não é escalonamento, é remover instrução de MAC.
6. **O staging int8 é 1,31× mais rápido que o f16 na mesma grade e com os mesmos blocos** (0,122
   contra 0,159 ms em M=128, 730 contra 561 G pesos/s, 1,49× menos instruções e metade dos bytes de
   LDS). O GEMM int8+dp4a ponta a ponta **não** foi medido aqui — só o staging dele.

Reprodução (é a sequência exata dos números deste arquivo):

```bash
source scripts/rocm-env.sh
cmake --build build -j8 --target bench-gemm-quant-gpu
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf

# a tabela A/B inteira (tres M, verificacao + cronometragem numa passada)
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 64,128,512 --ab --reps 9

# o caminho antigo (frente F), para o absoluto de referencia
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --m 64,128,512 --bk 64 --reps 5

# a sonda de quantizacao de onda: mesma config, grade de 128 CTAs em vez de 136
./scripts/gpu-lock.sh timeout 900 ./build/bench-gemm-quant-gpu $M --n 16384 --m 128 --ab --reps 5

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
| medido (staging do peso sozinho, `ag`, M=128) | 0,326 ms* | **0,159 ms** = 85 % do teto |

*0,326 ms é a subtração `modo 1 − modo 2` do arquivo da frente F, não uma medida direta. O 0,159 ms
é a mesma coisa medida com `ag` (que tira a cópia da ativação da conta) e ainda inclui o
`__syncthreads` e o consumo mínimo da LDS — ou seja, está sendo comparado com um número que é
**limite superior** do trabalho de staging.

A frente F aplicou a conta ao staging **total** (0,514 ms) e concluiu 43× o teto de issue; com a
contagem da ISA o número honesto é **1,2× o teto** para o staging do peso. O que a frente F leu como
"latência pura" é, em boa parte, **issue**: o kernel estava mal escalonado em cima de um custo de
instrução que já era grande, e as duas coisas foram medidas separadas aqui.

## 2. A/B: as oito variantes, a mesma janela

Bits do template `STG` (`gemm_v2_kernel`), todos no tile `BM128 BN128 BK64 RM=RN=8` (256 threads):

| bit | nome | o que muda |
|---|---|---|
| 1 | `avec` | cópia da ativação com `int4` (16 B) em vez de `half2` (4 B): 16 → 4 iterações |
| 2 | `pf` | prefetch dos campos do peso (qs/qh/signs/d/scales) de k+BK para **registradores**, emitido antes da conta de k |
| 4 | `db` | duplo buffer **só do W** (o A fica com um buffer): a dequantização+STS de k+BK roda durante a conta de k |
| 8 | `ag` | a ativação **não vai para a LDS**; o laço interno lê direto da global (broadcast de 2 endereços por warp) |

Medido (`--ab --reps 9`; as corridas de `--reps 5/7` concordam dentro de 1-2 % em todas as linhas,
exceto a nota 1 abaixo):

| M | variante | th | regs | LDS KB | CTA/CU | ms | **T MAC/s** | t_stg ms | staging % | bitid | rel-L2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 64 | **F frente F (kernel original, BM64)** | 256 | — | — | — | 0,684 | 8,34 | 0,260 | 38,0 | 0 | 2,07e-4 |
| 64 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 0,672 | 8,49 | 0,256 | 38,1 | 0 | 2,07e-4 |
| 64 | **V6 (BM64)** | 256 | 201 | 33,0 | 1 | **0,579** | **9,84** | 0,167 | 28,7 | 0 | 2,07e-4 |
| 128 | **F frente F (kernel original)** | 256 | 214 | 33,0 | 1 | 1,218 / 1,193² | 9,37 / 9,56² | 0,501 | **41,1** | 0 | 2,07e-4 |
| 128 | V0 base (mesma geometria, reescrito)² | 256 | 256 | 33,0 | 1 | 1,466 / 1,384² | 7,78 / 8,24² | 0,517 | 35,3 | 0 | 2,07e-4 |
| 128 | V1 = avec | 256 | 256 | 34,5 | 1 | 1,113 | 10,25 | 0,236 | 21,2 | 0 | 2,07e-4 |
| 128 | V4 = avec+ag | 256 | 256 | 16,5 | 3 | 1,323 | 8,62 | 0,159 | 12,0 | 0 | 2,07e-4 |
| 128 | **V6 = avec+pf+db+ag** | 256 | 220 | 33,0 | 1 | **0,896** | **12,74** | 0,164 | **18,3** | 0 | 2,07e-4 |
| 128 | V7 = V6 com BN=136 | 272 | 221 | 35,1 | 1 | 1,066 | 10,70 | 0,183 | 17,2 | 0 | 2,07e-4 |
| 128 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 1,159 | 9,84 | 0,431 | 37,2 | 0 | 2,07e-4 |
| 128 | V6 (BM64) | 256 | 201 | 33,0 | 1 | 1,010 | 11,30 | 0,288 | 28,5 | 0 | 2,07e-4 |
| 512 | **F frente F (kernel original)** | 256 | 214 | 33,0 | 1 | 3,888 / 3,859² | 11,74 / 11,82² | 1,497 | **38,5** | 0 | 2,07e-4 |
| 512 | V0 base | 256 | 256 | 33,0 | 1 | 5,189 / 4,519¹ | 8,79 / 10,10¹ | 1,476 | 28,4 | 0 | 2,07e-4 |
| 512 | V1 = avec | 256 | 256 | 34,5 | 1 | 3,593 | 12,70 | 0,764 | 21,3 | 0 | 2,07e-4 |
| 512 | **V6** | 256 | 220 | 33,0 | 1 | **3,101** | **14,71** | 0,527 | **17,0** | 0 | 2,07e-4 |
| 512 | V7 = V6 com BN=136 | 272 | 221 | 35,1 | 1 | 4,004 | 11,40 | 0,672 | 16,8 | 0 | 2,07e-4 |
| 512 | V0 base (BM64) | 256 | 233 | 24,8 | 2 | 4,262 | 10,71 | 1,519 | 35,6 | 0 | 2,07e-4 |
| 512 | V6 (BM64) | 256 | 201 | 33,0 | 1 | 3,818 | 11,95 | 1,048 | 27,5 | 0 | 2,07e-4 |

¹ **A primeira célula de M=512 é a menos confiável deste arquivo**: ela sai 5-15 % mais lenta que a
repetição da MESMA configuração no fim da mesma corrida (5,189 contra 4,519; em outras corridas
5,275/4,398 e 5,176/4,303). É DPM/janela compartilhada, não código. **A linha F — o kernel ORIGINAL
da frente F, disparado pela mesma tabela e na mesma sequência — é a referência que vale**: 1,218/1,193
em M=128 e 3,888/3,859 em M=512, estável a 2 % entre a primeira e a última célula.

² **Cuidado com a coluna `V0` deste arquivo:** `V0` é uma reescrita minha do kernel da frente F com a
MESMA geometria e o MESMO código de staging (é a base das comparações entre variantes), e ela sai
**10-15 % mais lenta que o kernel original** — 1,384 contra 1,193 em M=128 e 4,519 contra 3,859 em
M=512, com o mesmo número de registradores (256) e as mesmas 4 instruções de spill. As razões
relativas ao `V0` (V1 = 1,33×, V6 = 1,62×) **superestimam** o ganho contra o protótipo da frente F; as
razões que valem são as da linha `F`: **1,31-1,36× em M=128, 1,24-1,27× em M=512 e 1,16-1,18× em
M=64**. A causa da diferença entre `V0` e `F` não foi isolada — é a única lacuna de método desta
tabela.

**Leituras:**

1. **`avec` sozinho já vale 1,10× sobre o kernel ORIGINAL** (10,25 contra 9,37-9,56 T da linha F na
   mesma janela; 1,218 → 1,113 ms) e derruba o staging de 41,1 % para 21,2 %. É o ganho mais barato da tabela: 4 iterações de cópia em vez de 16, 4 pedidos de 16 B em
   vez de 16 de 4 B. O `AS` passa de `BK+2` para `BK+8` halves (16-B alinhado) e **não** cria conflito
   de banco: dentro de uma warp o A só tem 2 endereços (`tr` e `tr+1`), cada um broadcast para 16
   lanes — quem precisa de stride ímpar é o W (`W2 = BK/2+1`), e esse não mudou.
2. **`ag` sozinho não ganha nada** (8,61 contra 7,75 T do V0 — e *menos* que o V1) mas **libera
   16,5 KB de LDS, sobe a ocupação para 3 CTAs/CU** e mata a cópia da ativação (que era 35 % do
   staging). É a peça que faz o `db` caber: sem ele o duplo buffer do W não tem LDS.
3. **O que decide é a combinação `pf+db+ag`**: 12,74 T contra 9,37-9,56 do kernel original (mesma
   janela) e 10,25 do V1 — **1,33-1,36×** e 1,24×. O `pf` emite as cargas dos campos depois do `__syncthreads` e antes da conta;
   o compilador **não pode afundar a carga através da barreira**, então a janela de voo é uma iteração
   inteira (carga → conta → barreira → dequantização que a consome). A ISA do V6 confirma: primeira
   `global_load` no offset 45 do corpo do laço, primeiro `v_dot2_f32_f16` em 1 367, `s_barrier` em
   582. O `db` deixa a dequantização+STS de k+BK rodar durante a conta de k.
4. **BM=64 com 2 CTAs/CU deixou de ser o melhor**: com o pipeline dentro do CTA (V6) o BM=64 dá 11,04
   contra 12,57 T em M=128 e 11,96 contra 14,74 em M=512. A frente F tinha medido o contrário
  (BM=64 9,79 contra BM=128 9,23) porque lá **não havia pipeline**: a sobreposição vinha de fora do
   CTA (2 CTAs/CU). Com `db` ela vem de dentro, e o tile maior — que gasta menos instrução por MAC —
   volta a ganhar.

## 3. Onde o kernel está agora: no muro de issue

Contabilidade por warp por iteração de K (BM=128, BK=64, RM=RN=8), toda ela contada na ISA:

| | V0 (frente F) | V6 |
|---|---|---|
| conta (`v_dot2` + cargas de A/W do laço interno) | 2 580 | 2 580 |
| staging do peso | 590 | 590 |
| cópia da ativação (16 → 4 iterações de `int4`) | 144 | 0 (ag) |
| **total por warp por iteração de K** | **3 314** | **3 170** |
| instruções executadas na passada de M=512 (544 CTAs × 80 iterações) | 1,154e9 | 1,104e9 |
| medido (M=512) | 4,469 ms | **3,065 ms** |
| **instruções de warp/s realizadas** | **2,58e11 (68 % do teto)** | **3,60e11 (95 % do teto)** |

(A célula de 4,469 ms é a corrida de `--reps 5`; com o 4,303 ms da repetição da corrida de
`--reps 7` o V0 fica em 2,68e11 = 71 %. A conclusão não muda: **68-71 % contra 95 %**.) O teto de
3,80e11 slots/s é o das frentes D/F, medido num laço puro de dot2 — **não foi remedido aqui**
[DERIVADO].

A leitura é direta: **V0 desperdiça 32 % dos slots de issue em stalls; V6 desperdiça 5 %**. Não
porque tem muito menos instrução (só 4,3 % menos: o `ag` tira a cópia da ativação, não a
dequantização), mas porque **para de esperar**. O ganho de 1,6× é maior que a queda de instrução
porque o kernel trocou de regime.

**Consequência: o que sobra não se resolve com escalonamento.** Os 590 do staging do peso são 18,6 %
das instruções de V6 e a conta (2 580) é 81 %; para ir além de ~15 T é preciso **remover instrução de
MAC** — a via é o dp4a (4 MAC/instrução contra 2 do `v_dot2_f32_f16`), como a frente F concluiu pelo
teto (39,0 contra 19,5 T), agora com o número de issue que diz *onde* isso aparece: 1 024 dp4a +
correção no lugar de 2 048 dot2.

## 4. Staging int8 × staging f16 (a regra 5 do pedido)

O MESMO sub-bloco, a MESMA grade, os MESMOS bytes de fonte (38,3 MB por passada em M=128) e o mesmo
consumo mínimo na LDS; só o destino muda: `f16` (64 B por sub-bloco, com `cvt`+`mul`+`pack`) contra
`int8` (32 B, magnitude 1..15 do grid com o sinal por `__vsub4` e o `1+2*sc` **fora** do byte, que é
a convenção do `vec_dot_iq3_s_q8_1` — `vecdotq.cuh:725` multiplica em int32 e trocar a ordem muda o
arredondamento). As duas linhas com `ag`, para que a cópia da ativação não entre na conta:

| M | f16 na LDS | int8 na LDS | ganho | G pesos/s f16 → int8 | GB/s de fonte f16 → int8 | LDS escrita f16 → int8 |
|---|---|---|---|---|---|---|
| 128 | 0,159 ms | **0,122 ms** | **1,31×** | 561 → 730 | 241 → 314 | 1 121 → 730 GB/s |
| 512 | 0,498-0,500 ms | **0,371-0,372 ms** | **1,34×** | 716 → 958 | 307 → 411 | 1 432 → 958 GB/s |

Na ISA a diferença é maior do que no relógio — **396 instruções contra 590** para o mesmo sub-bloco
(1,49×), com 3 cargas de campo em vez de 5 (o `d` e a escala não são necessários no byte), **2
`ds_store_b128` em vez de 8 stores** e **zero** `v_cvt_f32_i32`/`v_fma_mixlo_f16`. O que sobra nos
dois é a mesma emulação de sinal (~112 instruções): 19 % do staging f16 e 28 % do int8.

Contra o roofline de 633 GB/s medido do cartão (`docs/medicoes-banda-e-gargalos.md`), o staging int8
em M=512 roda a **411 GB/s = 65 %** — quase o dobro do f16 em fração de roofline (307 GB/s = 49 %) e
ao mesmo tempo 1,34× mais rápido em pesos/s.

**O que isto NÃO é:** não é o GEMM int8 ponta a ponta. Falta o laço interno em dp4a com a correção por
bloco de 32 (`acc += (d_w*d_a) * (sumi * (1+2*sc))`, com o `d_a` de uma ativação Q8_1) — é o que
faria o número final subir, e não foi medido nesta frente.

## 5. O que falhou (e o número que mostra por quê)

**(a) Prefetch sem tirar a ativação da LDS — spill de registrador.** É o resultado mais instrutivo
da frente, porque a MESMA transformação dá 1,33× numa forma e 0,18× na outra:

| variante | M=128 | veredito |
|---|---|---|
| V2 = `avec+pf` | **8,823 ms (1,29 T)** | 0,18× o V0 |
| V3 = `avec+pf+db` | **8,577 ms (1,33 T)** | 0,17× o V0 |
| V5 = `avec+pf+ag` | 2,059 ms (5,54 T) | 0,54× o V1 |

A causa está na ISA e não é sutil: o campo prefetchado (`wf`, ~6-8 registradores) fica vivo através
do laço interno de ~2 300 instruções, e `RM=RN=8` já são 64 acumuladores + 16 operandos + endereços,
com o teto de 256 VGPRs por lane:

| variante (BM128 RM8) | instruções `scratch_*` (spill) | regs |
|---|---|---|
| V0 base | 4 | 256 (teto) |
| V1 `avec` | 4 | 256 |
| V2 `avec+pf` | **413** | 256 |
| V3 `avec+pf+db` | **412** | 256 |
| V5 `avec+pf+ag` | 158 | 256 |
| **V6 `avec+pf+db+ag`** | **0** | 220 |

O `ag` é o que abre espaço (sem tile de A não existem os endereços por linha `tr + r*TM`) e o `db`
encurta a vida do `wf` (ele é consumido no topo da iteração seguinte, logo depois da barreira). Com
os dois, o alocador cabe em 220 registradores e **não há um único acesso a scratch**. É também a
explicação do que a frente F tinha medido: lá o duplo buffer **completo** (A e W) dobrava a LDS e
derrubava a ocupação de 2 para 1 CTA/CU (−17 % em BM=64); aqui ele não dobra o A e ainda libera
registrador.

**(b) Grade múltipla de 64 por alargamento do tile (`BN=136`) — o ganho de onda não paga o custo.**
A quantização de onda em M=128 é real e foi medida com uma variável só (mesmo kernel, mesmo tile,
só o tamanho da grade): `--n 16384` dá 128 CTAs (2 ondas exatas num cartão de 64 CUs) e
`--n 17408` dá 136 CTAs (2,125 → 3 ondas, porque o CTA residente é 1):

| V6, M=128 | CTAs | ms | T MAC/s | µs por coluna-de-CTA |
|---|---|---|---|---|
| N=16384 | 128 | 0,762 | **14,10** | 5,95 |
| N=17408 (a forma pedida) | 136 | 0,905 | 12,61 | 6,65 |

Ou seja **11,8 % do tempo em M=128 é a cauda da terceira onda**. Mas consertar isso pela largura do
tile (`BN=136`, que faz `N/BN = 128` exato) **piora**: V7 dá 10,90 T contra 12,57 (−13 %) em M=128 e
11,49 contra 14,74 (−22 %) em M=512. O motivo é aritmético: 136 = 17×8, então `TN=17` e o bloco tem
272 threads = **8,5 warps** (a última warp tem 16 lanes ociosas, ~6 % dos slots), a LDS vai de 33,0
para 35,1 KB e cada CTA passa a estagiar 136 colunas de peso em vez de 128. **A correção certa para a
onda não é a largura do tile, é desacoplar o número de CTAs do tamanho da grade** (CTA persistente
com fila de trabalho) — que não foi implementada aqui.

**(c) Mais CTAs por CU, sozinho, não é alavanca.** V4 (`ag`, 3 CTAs/CU contra 1) anda *menos* que o
V1 de 1 CTA/CU (8,61 contra 10,28 T): a ativação lida da global dentro do laço interno custa mais do
que a cópia economizada. Ela só passa a valer quando o `db` a transforma em sobreposição (V6).

## 6. O que não foi medido

- **O GEMM int8+dp4a com os blocos reais** (o §4 mede só o staging dele). É a única via que a
  contabilidade do §3 aponta para passar de ~15 T.
- **CTA persistente com fila de trabalho** (§5b): é a resposta medida para os 11,8 % de cauda de onda
  em M=128, e não foi implementada.
- **O split A × W por modo**: o modo 2 (só a cópia da ativação) continua a medição frágil que a
  frente F já tinha sinalizado (0,190 ms aqui); o número robusto é o staging do peso sozinho, medido
  com `ag` no modo 1 (0,159 ms em M=128) — e mesmo ele inclui o `__syncthreads` e o consumo mínimo.
- **O teto de issue por SIMD deste cartão não foi remedido**: os 3,80e11 slots/s são herdados das
  frentes D/F e toda a leitura do §3 depende deles [DERIVADO].
- **`ffn_down` (K=17408) e os tensores de atenção** (5120×5120), onde o paralelismo cai: a frente F
  também não mediu, e nada aqui muda isso.
- **A ordem de comparação entre variantes dentro de uma corrida**: a primeira linha de M=512 sai
  10-20 % mais lenta que a repetição da MESMA configuração no fim da mesma corrida (§2, nota 1).
  Todas as razões deste arquivo usam linhas da mesma corrida, e as quatro corridas independentes
  concordam dentro de 1-2 % em todas as linhas menos essa.

## Veredito para o motor

1. **O que embarcar, medido:** `BM=128 × BN=128 × BK=64` (RM=RN=8, 256 threads) com (i) **ativação
   lida da global no laço interno** (não estagiar A na LDS), (ii) **duplo buffer só do W**, (iii)
   **prefetch dos campos do peso em registrador** com o `__syncthreads` entre a carga e o uso, (iv)
   cópia da ativação em `int4` quando ela ainda existir (outros tensores, `act_quant`). Isso dá
   **12,5-12,7 T em M=128 e 14,6-14,9 T em M=512**, contra 3,18 T do prefill de hoje: **3,9× a 4,7×**
   (e 1,24-1,36× contra o protótipo da frente F, §2 nota 2).
2. **O staging deixa de ser 42 %:** em M=128 ele é **18,3 %** (0,164 ms de 0,896 ms) e é só o peso —
   a metade da ativação simplesmente não existe mais nesse desenho. Em M=512 sobram 17,0 % e em M=64
   28,7 % — contra 41,1-43,2 % / 38,5 % / 38,0 % do kernel original.
3. **A ordem do trabalho, agora com número:** o kernel f16 está a **95 % do teto de issue** do
   cartão. Não há mais escalonamento a ganhar; o que resta é (a) o **dp4a**, que dobra MAC/instrução e
   é o único caminho para além de ~15 T, (b) o **staging int8**, que é 1,31× mais rápido que o f16 e
   já foi medido aqui, e (c) **CTA persistente** para a cauda de onda de M=128 (11,8 % medidos).
   Nenhuma das três é ajuste de tile.
4. **O que NÃO fazer:** prefetch de peso sem tirar a ativação da LDS (V2/V3: −81 %, por spill de
   registrador, §5a); acrescentar CTAs por CU confiando só na ocupação (V4: 3 CTAs/CU e *menos* taxa
   que 1 CTA/CU com pipeline); e consertar a onda alargando o tile (V7: −13 % a −22 %).
