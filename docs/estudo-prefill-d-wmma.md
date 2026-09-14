# WMMA int8 no gfx1201: quanto vale e onde trava

Frente D do estudo de prefill. Ferramenta: `tests/bench_wmma_gpu.hip` (alvo
`bench-wmma-gpu` do CMakeLists, que ja' existia). Todas as corridas sob
`scripts/gpu-lock.sh` com o `timeout` dentro do lock; a placa estava
**compartilhada** durante a janela (outro processo com ~62 % de `gpu_busy`), o que
aparece no §4 como variacao de 1,5-2x entre corridas — por isso todo numero
usado para decidir e' uma **razao medida dentro da mesma corrida**.

**TL;DR**

1. `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` **funciona** neste cartao:
   compila, roda, e reproduz a referencia de CPU **com 0 divergencias em 256**
   depois de acertar o empacotamento (secao 1). O acumulador e' **int32 exato**
   (16x127x127 = 258064 e 16x(-128)x127 = -260096 saem exatos).
2. **Pico medido: WMMA int8 = 182 T-MACs/s = 364 TOPS; dp4a = 44 T-MACs/s =
   88 TOPS; razao 4,2x** (mesma celula: 4,66x). WMMA f16: 89,6 T-MACs/s (acc
   f32) e 95,1 (acc f16).
3. O llama.cpp faz **67 TOPS-equivalente = 33,7 T-MACs/s** no prefill (frente
   COM: e' coopmat **f16**, o caminho int8 deles nem e' compilado com coopmat
   ligado). **O pico de dp4a deste cartao (88 TOPS) ja' e' 1,3x isso.**
4. O motor esta' em ~5,0 T-MACs/s = **12,8 % do pico de dp4a que ele ja' usa**.
   O 9x que falta **nao e' o matrix core** — e' alimentacao (cargas + latencia):
   um minigemm WMMA de primeira tentativa chega a 4,2-13,7 T-MACs/s (12-41 % do
   llama.cpp) e **58-94 % do tempo dele e' staging, nao WMMA**.
5. WMMA int8 **pode ser bit-exato** (int32 por bloco de 32 identico ao do dp4a:
   0/40960 divergencias; e o GEMM int8 bate a referencia de CPU **bit a bit,
   0/2048 elementos**). A via f16 **nao** e' bit-exata: 2,1e-04 de rel-L2.

---

## 1. O builtin funciona, e o empacotamento dos operandos e' este

O bench varre **32 hipoteses** de empacotamento (4 variantes para A x 4 para B x 2
para o acumulador D) contra uma referencia de CPU em int32, com dados aleatorios
em [-100, 100]. Uma unica combinacao zera as divergencias:

```
    A0 B0 D0  ->    0   max|d| 0   <== EXATA        (as outras 31: 240 a 256
    A2 B2 D0  ->    0   max|d| 0   <== EXATA         divergencias em 256)
    A3 B3 D0  ->    0   max|d| 0   <== EXATA
```

Tres combinacoes empatam porque A0/A2/A3 diferem so' na ordem dos k dentro do lane
(A2 e A3 sao permutacoes de A0 **dentro do proprio lane**), e o produto interno e'
soma — ou seja, o resultado nao distingue. O probe da secao 2 resolve a
permutacao, e a receita final e' a de A0 B0 D0.

### Receita verificada (o que o autor do motor precisa)

Para um tile 16x16x16, **wave32 (32 lanes)**, com o operando A = `A[m][k]` e o
operando B guardado como `Bn[n][k]` (linha = n, coluna = k — a transposta do
`B[k][n]` do produto, que e' como o llama.cpp e o hipfire guardam o B):

```
A: lane L, byte p (p=0..7)  =  A[m = L%16][k = 8*(L/16) + p]
B: lane L, byte p (p=0..7)  =  Bn[n = L%16][k = 8*(L/16) + p]
D: slot j do lane L (j=0..7) =  C[m = 8*(L/16) + j][n = L%16]
```

Isto e', em palavras: **cada lane carrega 8 k consecutivos de UMA linha**; os
lanes 0-15 levam `k = 0..7` e os lanes 16-31 levam `k = 8..15` da mesma linha
(`m = L%16` para A, `n = L%16` para B). O acumulador **nao** segue o mesmo
empacotamento: o slot `j` do lane L vale `C[8*(L/16)+j][L%16]`, ou seja o
acumulador e' a **transposta** do operando A (o `DATA_LAYOUT_J_MAJOR` do
llama.cpp, `mma.cuh:213`: "matrix C is the transposed matrix A&B on RDNA4").
Consequencia pratica: o `d_w` da correcao e' **1 valor por lane** (o `n` do lane) e
o `d_a` sao **8 floats consecutivos** (os 8 `m` do lane) — o que faz a correcao
custar 2 LDS.128 + 1 LDS.32 + 8 FMA por lane por bloco de 32, e nao um gather.

Em registradores: A e B sao `int32x2` (8 int8) por lane; D e' `int32x8` (8 int32)
por lane. O builtin toma 6 argumentos,
`(bool a_signed, int32x2 a, bool b_signed, int32x2 b, int32x8 c, bool clamp)`, e o
bool de A foi medido: com `A=0xFF` e `B=+1` o resultado e' **-16** (16 x -1), logo
**`true` = leitura com sinal** (o llama.cpp passa `true, true` para os dois; o
hipfire passa `false` em A quando o nibble e' 0..15 sem sinal).

### Probe independente (um byte aceso por vez)

Acende **um** byte do fragmento (lane L0, posicao p0) com o outro operando valendo
`k+1`; o resultado nao nulo devolve o indice **real** daquele byte. O mapa inteiro
(32 lanes x 8 bytes, para A e para B) sai como **bijecao sobre o tile 16x16, com 0
duplicados e 0 fora do tile**, e a tabela confirma byte a byte a receita acima:

```
   lane byte ->  A: (m,k)      B: (n,k)
      0    0 ->  m= 0 k= 0       n= 0 k= 0
      0    7 ->  m= 0 k= 7       n= 0 k= 7
     16    0 ->  m= 0 k= 8       n= 0 k= 8
     17    3 ->  m= 1 k=11       n= 1 k=11
     31    7 ->  m=15 k=15       n=15 k=15
```

### f16 e o que mais o builtin aceita

`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` (8 halves = 16 B por lane em
cada operando, `f32x8` de acumulador) usa **exatamente o mesmo empacotamento**
(varredura das 32 hipoteses: vencedora A0 B0 D0, 0 divergencias; "mesma receita do
int8: SIM"), com acumulacao f32 dentro do k-tile de 16: **erro relativo maximo
5,96e-08**, `|d|` maximo 0,25 em somas de ate ~1e6. Existe tambem a variante de
acumulador f16 (`wmma_f16_16x16x16_f16_w32_gfx12`), que e' o que o llama.cpp usa
para FA; ela tem pico medido no §2.

---

## 2. Pico: WMMA int8 x WMMA f16 x dp4a

Lacos apertados com **8 acumuladores independentes** (sem cadeia de dependencia) e
desenrolar U=8; o numero de iteracoes e' argumento de kernel. Cada celula abaixo e'
o **minimo de 3 rodadas de 5 lancamentos** depois do aquecimento de DPM. A ISA foi
conferida: o corpo do laco `U=8` tem **64 `v_wmma_i32_16x16x16_iu8`** (um por
chamada do builtin) e 64 `v_dot4_i32_iu8` no laco de dp4a — nao ha' desvio entre o
que foi escrito e o que foi emitido.

| laco | G-instr-warp/s | T-MACs/s | TOPS | MACs por instrucao |
|---|---|---|---|---|
| WMMA int8 (CTAs=32, U=8) | 44,42 | **181,95** | **363,9** | 4096 |
| WMMA int8 (CTAs=32, U=1) | 44,91 | 183,97 | 367,9 | 4096 |
| WMMA int8 (repetido, CTAs=32) | 44,75 | 183,31 | 366,6 | 4096 |
| WMMA f16 acc f32 | 21,87 | 89,57 | 179,1 | 4096 |
| WMMA f16 acc f16 | 23,21 | 95,05 | 190,1 | 4096 |
| dp4a (CTAs=32) | 305,36 | 39,09 | 78,2 | 128 |
| dp4a (CTAs=256) | 369,51 | **47,30** | **94,6** | 128 |
| dp4a (repetido, CTAs=32) | 303,84 | 38,89 | 77,8 | 128 |

Piso de ruido (a mesma celula medida de novo, em **taxa**, que e' o que nao depende
do `iters` calibrado): WMMA int8 181,95 vs 183,31 T-MACs/s = **1,007x**; dp4a 39,09
vs 38,89 = **0,995x**. Ou seja, dentro de uma corrida a medicao e' solida a ~1 %.

- **Razao WMMA int8 / dp4a: 4,2x** (melhor celula de cada: 182/44 = 4,1x; mesma
  celula, 32 CTAs: 4,66x). Em MACs por instrucao a razao e' 32x (4096 contra 128);
  a taxa de instrucao vai 6,9x **no sentido contrario** (305 G de dp4a contra 44 G
  de WMMA), logo **cada WMMA custa ~6,9x os ciclos de um dp4a** e ainda ganha 4,2x
  em MACs/s.
- **f16 e' 1,9-2,0x mais lento que int8** neste cartao (89,6 contra 182 T-MACs/s
  com acumulador f32; 95,1 com acumulador f16, que ganha so' 6 %). Se o motor for
  construir um caminho de matrix cores, **int8 e' a escolha certa aqui** — e e' a
  unica que pode ser bit-exata (§5).
- **Ciclos por instrucao por SIMD**: 9,6-15,6 para o WMMA int8 e 1,4-2,0 para o
  dp4a. A faixa existe porque a referencia e' o proprio laco de FMA fp32 medido na
  mesma corrida (13,7-19,5 T-FMA/s = 56-80 % do pico de 24,35 T-FMA/s a 2,97 GHz),
  ou seja a **DPM varia durante a janela** [SUPOSICAO: FMA fp32 = 1 instrucao por
  ciclo por SIMD32; essa e' a unica suposicao da conversao, e o que importa — as
  razoes — nao depende dela]. Por CU (4 SIMD): 2,4-3,9 ciclos por WMMA.
- **O matrix core nao e' um pipe separado**: um laco misto de 1 WMMA + 8 dp4a por
  passo, com acumuladores disjuntos, custa **1,16x a soma** dos dois lacos puros
  (misto/max = 1,33x). Eles **dividem o mesmo recurso de issue/execucao**; nao ha'
  "dp4a de graca ao lado do WMMA".
- Registradores: 97 por lane no laco de WMMA int8 (com 8 acumuladores), o que
  **limita a ocupacao a 2 warps por SIMD** (teto de 256 VGPRS/lane). Foi preciso
  128-256 CTAs para a taxa saturar; com 32 CTAs (1 warp/SIMD) ja' se chega a 97 %
  do pico, o que mostra que o laco e' limite de **vazao do pipe de matriz**, nao de
  latencia.

### O que isso diz do caminho f16 que o llama.cpp usa (frente COM)

Confirmado no codigo, com `file:line`:

- `vulkan-shaders-gen.cpp:693`: `matmul_shaders(true, ..., coopmat=true, coopmat2=false, f16acc=false)`
  — e' essa a variante gerada para coopmat1;
- `vulkan-shaders-gen.cpp:514` (`if (coopmat2 || fp16) return "float16_t"`) fixa
  `FLOAT_TYPE = float16_t`, e `:483` fixa `ACC_TYPE = "float"` (**acumulador
  f32**);
- `mul_mm.comp:340-342` usa `coopmat<FLOAT_TYPE, ..., gl_MatrixUseA/B>` e
  `coopmat<ACC_TYPE, ..., gl_MatrixUseAccumulator>`; `:194` declara
  `shared FLOAT_TYPEV2 buf_a[...]` — isto e', **o peso k-quant e' dequantizado
  para f16 na LDS** e `:383` faz `coopMatLoad` de la';
- `vulkan-shaders-gen.cpp:627`: o shader MMQ int8 (`mul_mmq.comp`, o caminho
  dp4a/WMMA int8) so' e' gerado quando `!coopmat && !coopmat2` — **com coopmat
  ligado o caminho int8 dos k-quants nem existe**. E' exatamente o que a medicao
  da frente COM mostra (`GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1`: 1179 contra 1196
  tok/s, 1,4 %).

Ou seja: no prefill deles a aritmetica e' **f16 x f16 -> f32 por coopmat**, e o
teto dela neste cartao (medido aqui) e' **89,6 T-MACs/s = 179 TOPS**, contra
**182 T-MACs/s = 364 TOPS** do int8. Os 33,7 T-MACs/s que eles **alcancam**
ficam em 38 % do pico de f16 e 19 % do pico de int8 — nao e' o matrix core que
os limita, e' o mesmo caminho de dados de que o §4 fala.

---

## 3. O minigemm realista

`C[M][N] = A[M][K] . W[N][K]` int8, pesos e ativacao estagiados em LDS, correcao
de escala fp32 **por bloco de 32** (`d_w * d_a`, o esquema k-quant/Q8_1: o
acumulador int32 e' zerado a cada bloco de 32, multiplicado em fp32 e somado no
acumulador fp32 — a mesma sequencia do motor), K = 5120 (a forma real do tronco),
M = 16 e 64, N = 1024 e 4096. E a variante f16 (o staging converte int8\*d para
f16 na LDS, como o llama.cpp faz). Vida dupla: staging ping-pong com BK = 64.

| M | N | K | int8 ms | T-MACs/s | TOPS | % do llama.cpp | f16 ms | T-MACs/s | TOPS | % do llama.cpp |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 1024 | 5120 | 0,0784 | 1,07 | 2,1 | 3,2 % | 0,1250 | 0,67 | 1,3 | 2,0 % |
| 16 | 4096 | 5120 | 0,0804 | **4,17** | 8,3 | **12,4 %** | 0,1245 | 2,70 | 5,4 | 8,0 % |
| 64 | 1024 | 5120 | 0,1195 | 2,81 | 5,6 | 8,3 % | 0,1992 | 1,68 | 3,4 | 5,0 % |
| 64 | 4096 | 5120 | 0,1306 | **10,28** | 20,6 | **30,5 %** | 0,2086 | 6,43 | 12,9 | 19,1 % |

Varredura de geometria (K=5120, N=4096, int8) — **a forma do CTA e' o knob que
mais vale**:

| config | CTAs | warps/CTA | ms | T-MACs/s | % do llama.cpp | % do pico WMMA |
|---|---|---|---|---|---|---|
| M=16 BM16 BN16 W1 | 256 | 1 | 0,1082 | 3,10 | 9 % | 1,7 % |
| M=16 BM16 BN32 W2 | 128 | 2 | 0,0871 | 3,85 | 11 % | 2,1 % |
| M=16 BM16 BN64 W4 | 64 | 4 | 0,0800 | 4,20 | 12 % | 2,3 % |
| M=64 BM64 BN16 W4 | 256 | 4 | 0,0980 | **13,69** | **41 %** | 7,5 % |
| M=64 BM64 BN32 W4 | 128 | 4 | 0,1108 | 12,11 | 36 % | 6,7 % |
| M=64 BM64 BN64 W4 | 64 | 4 | 0,1315 | 10,20 | 30 % | 5,6 % |

Corretude (M=16, N=128, K=5120, referencia de CPU que reproduz a mesma sequencia
aritmetica):

- **int8: 0/2048 elementos com bits diferentes da referencia** — bit-exato;
- f16: rel-L2 **3,79e-06** contra o fp64 dos pesos ja' arredondados para f16, e
  **2,09e-04** contra o resultado int8 dos mesmos dados. Este 2,1e-04 e' o desvio
  que a via f16 introduz (arredondamento do peso para 11 bits de mantissa), e e'
  a razao pela qual a via f16 precisa de gate numerico mesmo quando o int8 nao
  precisaria.

Leitura: um protótipo de primeira tentativa, sem split-K, sem `buffer_load`, com 4
warps por CTA, chega a **41 % do llama.cpp** na melhor geometria (M=64) e a **12 %**
na forma que o motor realmente usa hoje (M=16 por sub-lote). O pico de WMMA medido
no §2 e' 182 T-MACs/s, entao o protótipo esta' em **2-7,5 % do proprio pico**: o
gargalo nao e' a unidade de matriz.

---

## 4. Onde trava

O proprio bench responde, medindo **o mesmo kernel com a mesma grade** em tres
modos: (0) GEMM completo, (1) staging + as leituras de fragmento na LDS (sem WMMA,
sem correcao) e (2) so' o staging (global -> LDS):

| celula | GEMM int8 | staging puro | staging + leituras LDS | WMMA (por diferenca) |
|---|---|---|---|---|
| M=16 N=1024 | 0,0784 ms | 0,0733 ms = **94 %** | 0,0727 ms = 93 % | 0,0057 ms = 7 % |
| M=16 N=4096 | 0,0804 ms | 0,0754 ms = **94 %** | 0,0752 ms = 94 % | 0,0052 ms = 6 % |
| M=64 N=1024 | 0,1195 ms | 0,0716 ms = **60 %** | 0,0717 ms = 60 % | 0,0478 ms = 40 % |
| M=64 N=4096 | 0,1306 ms | 0,0752 ms = **58 %** | 0,0749 ms = 57 % | 0,0557 ms = 43 % |

**Onde trava, com o numero que aponta para o lugar: no staging (leitura dos pesos
da memoria global para a LDS) — 94 % do tempo em M=16 e 58 % em M=64. O matrix core
e' 6-43 %.** E as leituras de fragmento na LDS **nao custam nada** (0,0752 com
contra 0,0749 sem, dentro de 0,5 %), ou seja a LDS tem folga de sobra.

Os numeros que fecham essa leitura:

- O staging lê 21 MB (os pesos de N=4096 x K=5120) em 0,075 ms = **280 GB/s**. O
  roofline de DRAM medido deste cartao e' ~633 GB/s (`docs/medicoes-banda-e-gargalos.md`),
  logo o staging roda a **44 % do roofline** — e nao ha' sobreposicao nenhuma entre
  o staging e o compute quando o mesmo warp faz os dois.
- A LDS pedida pelo GEMM e' de 464 B por WMMA (A 2x8 B + B 8x8 B por lane + os 8
  floats de `d_a` + 1 de `d_w`, por bloco de 32, 8 WMMAs por bloco). A 10,28
  T-MACs/s = 2,51e9 WMMA/s isso da' **1,16 TB/s**; [SUPOSICAO: LDS de 32 bancos x
  4 B = 128 B/ciclo/CU] 128 B x 64 CU x 2,4 GHz = **19,7 TB/s** de modelo. A LDS
  esta' a **6 % do modelo** — e o modo 1 acima confirma na pratica.
- O acesso ao peso no staging e' **linha por linha com `BK=64`**: cada linha de 64 B
  e' meia linha de cache (128 B) e as linhas estao separadas por K = 5120 B. O
  padrao desperdica metade de cada linha e nao tem localidade de pagina de DRAM.
  E' o candidato nº 1 a conserto (BK=128, ou staging em ordem k-maior como o
  `block_a_to_shmem` do llama.cpp) e a razao mais provavel dos 280 contra 633 GB/s.
- A ocupacao e' o outro limite: com 97-138 registradores por lane cabem 2 warps por
  SIMD, e a grade de M=16 com BN=64 da' 64 CTAs de 4 warps = **1 warp por SIMD** —
  nao ha' ninguem para cobrir a latencia de ~1000 ciclos do `LDG`. A varredura de
  geometria mostra isso: mais CTAs (BN menor) **piora** em M=16 (mais A duplicado) e
  **melhora** em M=64 (13,69 contra 10,20 T-MACs/s).

### O que falhou (registrado, nao maquiado)

- **A medida de banda de LDS pura nao saiu.** O kernel de copia LDS->LDS
  (`int4`, 64 CTAs x 256 threads, 16 KB de LDS) mede tempo **constante de ~1,4 s por
  lancamento para `iters` = 100, 1000 e 5000** (e 57 ms com `iters=1`, 460 ms com
  `iters=10`), o que e' impossivel para uma copia de LDS: nao escala com o trabalho
  e nao tem ordem de grandeza plausivel. Foi testado com `extern __shared__` e com
  `__shared__` estatico, com o mesmo resultado. **Nao usei esse numero em lugar
  nenhum**; a folga da LDS no §4 vem do modo 1 do GEMM (mesmo kernel, mesma grade,
  diferenca dentro de 0,5 %), que e' uma comparacao dentro da mesma corrida.
- **A variacao entre corridas e' de 1,5-2x** no mesmo codigo (a mesma celula do
  GEMM deu 10,20 e 5,25 T-MACs/s em duas corridas), o que bate com a placa estar
  compartilhada (`gpu_busy` de 62 % com outro processo durante a janela; o lock
  serializa os agentes que o respeitam, nao os que nao respeitam). **Nenhuma
  decisao deste documento usa numero absoluto de corridas diferentes** — as razoes
  (WMMA/dp4a, staging/GEMM, protótipo/llama.cpp, medida/repeticao) sao todas
  dentro da mesma corrida.
- O `clock` nao e' medivel de forma estavel sob essa carga: o laco de FMA fp32 deu
  13,7 e 19,5 T-FMA/s em corridas diferentes (56 % e 80 % do pico). Por isso os
  "ciclos por WMMA" do §2 sao uma **faixa** e a comparacao principal e' em MACs/s.

---

## 5. Bit-exatidao

**Um caminho WMMA int8 pode reproduzir os resultados do `dp4a` bit a bit, e o
critério e' a granularidade da correcao, nao a aritmetica.** A soma de produtos
int8 em int32 e' exata e associativa — a ordem nao importa —, entao o que decide e'
se o acumulador int32 e' zerado no mesmo ponto: a secao EXATO compara, bloco de 32
por bloco de 32, o int32 do WMMA (2 chamadas de k16) contra o do dp4a (8 de k4) ao
longo de K=5120: **0 divergencias em 40960** (256 pares x 160 blocos), tanto com
dados aleatorios em [-100, 100] quanto no caso extremo -128/+127, onde as somas
chegam a 520208. Em cima disso, o GEMM int8 completo (com a correcao fp32 por bloco
de 32 e ordem sequencial em k) bate a referencia de CPU **em todos os 2048
elementos, bit a bit** (`memcmp` de cada float). Isto e': o esquema de correcao do
motor (`acc += (d_w*d_a) * sumi_int32`, bloco a bloco) e' reproduzivel, e a unica
coisa que quebra a exatidao e' acumular o int32 por varios blocos e aplicar uma
escala unica no fim (e' o que um kernel que "otimiza" a correcao faz). O unico
ponto de atencao especifico do IQ3_S (59 % dos bytes deste modelo) e' que o
multiplicador `1 + 2*sc` (1..31, `vecdotq.cuh:725`) tem que ser aplicado como
**multiplicacao inteira no int32 do bloco**, antes do `d_w * d_a` em fp32, porque
o motor faz exatamente `sumi *= 1+2*sc` em inteiro e so' depois `d * sumi`
(`vecdotq.cuh:725-728`) — trocar a ordem muda o arredondamento. A via **f16 nao
tem essa propriedade**: ela arredonda o peso para 11 bits de mantissa (rel-L2
medido de 2,1e-04 contra o int8, §3) e depende da ordem de soma dentro do
acumulador do matrix core (erro relativo de 5,96e-08 por k-tile de 16, §1). Ou
seja: int8-WMMA pode entrar com gate **bit-exato** (comparacao byte a byte contra a
saida atual, como o `check-matmul-gpu` faz); f16-WMMA **exige** gate numerico
(PPL/regressao), sem alternativa.

---

## Veredito: o matrix core **nao** e' o 9x que falta

O numero que decide, medido neste cartao:

| | T-MACs/s | TOPS-equivalente | contra o llama.cpp (33,7 T-MACs/s) |
|---|---|---|---|
| llama.cpp Vulkan pp512 (coopmat **f16**) | 33,7 | 67 | 1,00x |
| **pico de dp4a** (o que o motor ja' usa) | **39-47** | **78-95** | **1,16-1,40x** |
| motor hoje (matvec em lote, N=16) | 5,0 | 10 | 0,15x |
| pico de WMMA int8 | 182 | 364 | 5,4x |
| pico de WMMA f16 | 89,6 | 179 | 2,7x |
| minigemm WMMA int8 (M=64, melhor geometria) | 13,7 | 27 | 0,41x |
| minigemm WMMA int8 (M=16, a forma do motor) | 4,2 | 8,3 | 0,12x |

**Um kernel de dp4a rodando no pico deste cartao (88 TOPS) ja' seria 1,3x mais
rapido que o prefill do llama.cpp — e o motor usa 12,8 % desse pico.** O 9x nao
esta' na escolha da instrucao; esta' no que alimenta a instrucao. Isso e'
consistente com o que o `docs/journal-lote.md` ja' tinha medido (nao e' banda de
DRAM, nao e' transbordo de registrador, nao e' falta de MLP, e a contabilidade de
issue/dp4a nao fecha): o custo por token e' dominado por cargas de ativacao,
aritmetica de indice e as esperas entre elas, e **a mesma parede aparece no caminho
WMMA**: o minigemm de primeira tentativa fica em 2-7,5 % do pico de matrix core
porque 58-94 % do tempo dele e' staging de peso. Trocar dp4a por WMMA sem arrumar
o caminho de dados da' **12 % do llama.cpp em M=16 e 41 % em M=64** — ou seja, de
zero a 3,7x, nunca 9x.

O que WMMA **da'** de verdade e' teto: 364 TOPS contra 88, um fator 4,2x de pico, e
ele e' **alcancavel com bit-exatidao** (ao contrario do f16). Vale como aposta
depois de o caminho de dados estar resolvido — como substituto do gargalo atual,
nao como causa dele.

O que o motor teria que construir para usar isso (e' reescrita, nao ajuste):

1. **Um GEMM de verdade, com M >= 64 por lancamento.** O mapa atual e' GEMV com M=1
   por token e o lote compartilha so' a leitura do peso (M=16); com M=16 o
   protótipo empata com o motor de hoje (4,2 contra 3,75 T-MACs/s). Ou o prefill
   passa a processar 64-128 tokens por GEMM, ou entra split-K (fatia o K entre
   CTAs e reduz — o que custa a bit-exatidao) para ter warps suficientes.
2. **Dequantizacao para int8 na LDS, no staging.** O peso tem que virar bytes int8
   na LDS: para os k-quants o valor e' o nibble menos o offset; para o IQ3_S e' o
   byte do grid (1..15) com o `1+2*sc` **fora** do int8 (senao 15x31 = 465 nao
   cabe) e aplicado como multiplicacao inteira no int32 do bloco (§5). Isso e' o
   `block_a_to_shmem` do llama.cpp, tipo por tipo.
3. **Staging em ordem k-maior com linha de cache cheia** (BK=128) e/ou
   `buffer_load`, com 4+ warps por SIMD: e' o item que o §4 aponta como o gargalo
   real (280 GB/s contra 633 GB/s de roofline).
4. **Correcao de escala por bloco de 32**: acumulador int32 zerado a cada bloco,
   `facc += (d_w*d_a) * int32` com o `d_w` de 1 valor por lane e o `d_a` de 8
   valores consecutivos por lane (o layout D da §1), na mesma ordem em k do motor.
5. **Ativacao em Q8_1** (ja' existe: `quantize_q8_1_block`) e o mesmo cuidado de
   ordem de soma na reducao final.

E o caminho barato que os numeros deste estudo sugerem testar **antes** de tudo
isso: o motor esta' a 12,8 % do pico de dp4a que ja' tem. Um experimento de uma
tarde — medir o mesmo `matvec_kernel_batch` com a ativacao lida por linha inteira
(LDS) e mais warps por CU em vez de menos — vale mais que a reescrita para WMMA,
porque o teto do dp4a (88 TOPS) ja' passa do que o llama.cpp alcanca (67 TOPS).
