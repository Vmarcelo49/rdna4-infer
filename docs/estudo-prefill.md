# Por que o prefill do llama.cpp é 9,7× mais rápido — estudo com medição (14/09, dia)

Pergunta do usuário, literal: *"how does llama.cpp prefill is 9 to 10 times faster than us? what
did we skip? what should we port or optimize?"*. Este arquivo é o documento-síntese do dia; cada
frente tem o seu relatório (`docs/estudo-prefill-{a-vulkan,b-mmq,c-nosso,d-wmma}.md`).

## 0. NOTA DE INTEGRIDADE (14/09, tarde) — a referência estava 11 % otimista

**Achado, e é uma correção que vale para todo o estudo**: o checkout
`/home/marcelo/Projetos/llama.cpp` tinha **uma linha modificada localmente**
(`ggml-vulkan.cpp:5462`, `rm_kq = 2` → `1`, patch de **2026-09-10 19:49**), e o binário
`build/bin/llama-bench` foi **linkado 73 s depois** — ou seja, todas as medições da manhã saíram
de um llama.cpp **modificado**, não do upstream. Quem editou não foi nenhum agente de hoje (a data
é de quatro dias atrás). O patch está salvo em `/tmp/llama-rmkq.patch` e o binário antigo em
`/tmp/llama-bench-patched`; a árvore do usuário **não** foi tocada — eu construí um worktree
limpo em `/tmp/llama-clean` a partir de `df03399b8`.

| `pp512` | binário local (patch `rm_kq=1`) | **limpo (`rm_kq=2`, upstream)** | delta |
|---|---|---|---|
| default (`-ub 512`) | 1196,49 ± 1,72 | **1054,29 ± 11,93** | **−11,9 %** |
| `GGML_VK_DISABLE_COOPMAT=1/2` | 478,38 ± 0,64 | **426,03 ± 0,72** | −10,9 % |
| `-ub 16` | 199,27 | **170,43** | −14,5 % |
| `-ub 64` | 663,77 | **553,97** | −16,5 % |
| `-ub 128` | 1007,95 | **872,58** | −13,4 % |

**Números corrigidos, que substituem os do resto deste documento e dos relatórios das frentes A e
C** (os das frentes continuam válidos em *razão*, que é como eles são usados):

- o gap do prefill é **8,54×** (1054,29 contra 123,4), não 9,0-9,7×;
- decomposição: **1,38× estrutura de kernel em micro-lote igual** (170,43/123,4) × **2,50×
  micro-lote** (426,03/170,43) × **2,47× unidades de matriz** (1054,29/426,03) = **8,52×**;
- a curva do micro-lote continua com joelho em 128 (170 → 554 → 873 → 1054);
- alvo do plano: **~1050 tok/s**, não 1200.

O que **não** muda: a razão do coopmat (2,47× contra 2,50×), a ordem de grandeza de todos os
fatores, e nenhuma conclusão — só a âncora absoluta cai 11 %. A lição de processo fica registrada:
**um build de referência com uma linha local modificada contamina toda a comparação**, e o
`git status` do diretório de referência deveria ter sido a primeira checagem do dia.


## TL;DR — o gap tem três fatores medidos, e o maior é o tamanho do lote

| fator | de → para | ganho | como foi medido |
|---|---|---|---|
| **1. micro-lote** | chunk de 16 → 512 tokens | **2,50×** (limpo: 170,43 → 426,03, com o coopmat desligado dos dois lados) | `llama-bench -b/-ub`: o próprio llama.cpp cai de **1054,29 para 170,43 tok/s** quando o micro-lote desce para 16 |
| **2. caminho de dados** | GEMV em lote → GEMM tilejado com staging na LDS | **3,2-4,3×** em M=64-128 (e 1,38× no micro-lote igual, do lado deles) | dois protótipos independentes: **12,74 T-MAC/s** (int8/dp4a, M=512, verificado contra oráculo) e 10,3-13,7 (WMMA f16, M=64) contra 3,0-3,2 do motor |
| **3. unidades de matriz** | dp4a → WMMA | **2,47×** medido no llama.cpp (1054,29 vs 426,03); teto **4,2×** no cartão (182 contra 44 T-MAC/s) | A/B do binário limpo (`GGML_VK_DISABLE_COOPMAT`) + pico medido (frente D) |

E a frase que corrige o plano que eu tinha escrito de manhã, dita pela própria frente D:
**"WMMA compra teto, não velocidade."** Com o caminho de dados como está, trocar a instrução não
acelera nada — 58-94 % do tempo dos dois protótipos é *staging*, e o staging roda a **280 GB/s de
633** porque lê 64 B por linha de peso com stride de 5120 B. A ordem certa é **dados primeiro,
instrução depois** — mas a instrução é necessária no fim, porque o llama.cpp faz **32,7 T-MAC/s =
74 % do pico do dp4a** deste cartão, e nenhum protótipo de dp4a passou de 29 %.

### Os picos medidos neste cartão — **reproduzidos por mim** na mesma bancada

Rodei `./scripts/gpu-lock.sh timeout 900 ./build/bench-wmma-gpu` eu mesmo, depois de a frente D
entregar, e os números batem: WMMA int8 **161,7-183,3 T-MAC/s** (323-364 TOPS), dp4a
**39,4-45,4 T-MAC/s** (79-91 TOPS), WMMA f16 acc f32 **82,1-87,7**, acc f16 **90,8**; razão
**≈4,2×**; o laço misto (1 WMMA + 8 dp4a, acumuladores disjuntos) custa **1,00-1,16× a soma dos
dois laços puros**, ou seja **não são pipes separados** — eles dividem issue.

Em MACs por ciclo por CU (a forma mais comparável): **dp4a 232, WMMA f16 468, WMMA int8 1022**.
Note que os números de folha de especificação que circularam antes (dp4a 512 MAC/CU/clk, fp16
WMMA 512) são ~2,2× os medidos neste cartão — **valem os medidos**.

### Os picos medidos neste cartão (frente D, dentro da mesma corrida)

| laço | T-MAC/s | TOPS | nota |
|---|---|---|---|
| WMMA int8 16×16×16 | **182** | 364 | 4,2× o dp4a; **1,9× o f16** |
| WMMA f16 16×16×16 (acc f32) | 89,6 | 179 | é o que o Vulkan usa no prefill |
| dp4a (`v_dot4_i32_iu8`) | **44** | 88 | o que nós usamos |
| misto (1 WMMA + 8 dp4a) | — | — | 1,00-1,16× a soma: **não são pipes separados** |

### A frase que resume o "o que foi pulado"

O nosso motor faz **1 byte de ativação carregado por MAC** e **1 elemento de saída por thread**
(sem tiling de registrador); o caminho do llama.cpp faz **0,0156 byte por MAC** — 64× menos — com
tile de 128×128 por workgroup e o peso dequantizado **na LDS** (`docs/estudo-prefill-a-vulkan.md`
§2). É a mesma instrução de multiplicação com uma **forma** diferente: eles emitem ~840-1100 MACs
por instrução emitida, nós 38.

## 1. Referência ancorada (mesma máquina, mesmo modelo, mesma janela)

`llama-bench` do checkout `/home/marcelo/Projetos/llama.cpp` (build `df03399b8`), backend
**Vulkan/RADV**, modelo `Qwen3.8-27B-UD-IQ3_S.gguf` (11,20 GiB, 3,4375 bpw, 27,32 B params),
`-r 3`, modelo quente, sob `scripts/gpu-lock.sh`:

| teste | tok/s | observação |
|---|---|---|
| pp64 | 531,75 ± 56,04 | prompt curto: dominado por overhead, ruidoso |
| pp512 | **1113,96 ± 24,34** (1ª corrida) / **1196,49 ± 1,72** (2ª corrida) | a 2ª corrida é a mais limpa |
| pp2048 | 1106,02 ± 12,91 | **plateau**: custo por token constante |
| tg128 | **39,42 ± 1,64** | decode, para comparação |

Nosso motor, mesma máquina e modelo: **prefill 123,68 tok/s a 512** (117,47 a 4096),
**decode 34,00 tok/s a 4K** (30,4 em contexto curto).

| | nós | llama.cpp Vulkan | razão |
|---|---|---|---|
| prefill 512 | 123,68 tok/s | 1196,49 | **9,66×** |
| prefill 2048 | 117,47 tok/s (@4096) | 1106,02 | 9,4× |
| decode | 34,00 tok/s (@4K) | 39,42 (tg128) | 1,16× |
| decode curto | 30,4 tok/s | 39,42 | 1,30× |

**A assimetria é o achado mais importante deste estudo**: o decode está a 1,2-1,3× e o prefill a
9,7×. Isso localiza o problema com precisão — não é "o motor é ingênuo em geral", é
**especificamente o caminho de multiplicação de matrizes em lote**.

## 2. A decomposição experimental (o resultado que responde a pergunta)

`llama-bench -p 512 -n 0 -r 2` com as variáveis de ambiente que o próprio backend Vulkan lê
(`ggml-vulkan.cpp:6584-6600`):

| configuração | pp512 (tok/s) | vs baseline | leitura |
|---|---|---|---|
| baseline (tudo ligado) | **1196,49 ± 1,72** | — | |
| `GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1` | **478,38 ± 0,64** | **−60,0 %** | matrix cores valem **2,50×** |
| `GGML_VK_DISABLE_INTEGER_DOT_PRODUCT=1` | 1179,51 ± 2,88 | −1,4 % | o caminho coopmat **não é int8** |
| coopmat off **+** integer-dot off | 461,78 ± 2,76 | −61,4 % | **nem o fallback é int8/dp4a** (−3,5 % apenas) |
| `GGML_VK_DISABLE_DOT2=1` | 1121,81 ± 8,12 | −6,2 % | o f16 empacotado contribui ~6 % |
| `GGML_VK_DISABLE_F16=1` | 1161,72 ± 76,09 | — | ruidoso; f16 não é gargalo de armazenamento |

O achado que reorganiza o plano: **o llama.cpp não faz dp4a/int8 neste modelo** — nem no caminho
rápido nem no fallback. Os dois caminhos são **f16**: o peso k-quant é dequantizado para **f16 na
LDS** e o produto é feito com FMA f16 vetorial empacotado (fallback, `dot2`) ou com **coopmat
f16** (rápido). Desligar o integer-dot não muda quase nada nos dois casos. Ou seja: a família de
instrução que nós escolhemos para o prefill (`v_dot4_i32_iu8`) **não é a que o concorrente usa**, e
o nosso caminho não tem *nenhum* f16.

Duas conclusões, e a segunda é a que mais muda o nosso plano:

1. **Os matrix cores valem 2,5×** e o caminho que os usa é **f16 × f16 → f32** (dequantização do
   k-quant para f16 na LDS e `coopmat` f16). Isso é coerente com três fatos independentes: o
   device reporta `matrix cores: KHR_coopmat` (que no RADV cobre f16/bf16/f32) e `fp16: dot2`;
   desligar o *integer dot product* quase não muda nada (a via int8/dp4a não é a que roda); e o
   `KHR_cooperative_matrix` da base **não** tem int8 (isso é `NV_cooperative_matrix2`, o ramo
   `if (device->coopmat2)` de `ggml-vulkan.cpp:4428`, que não é o nosso).
2. **Sem coopmat eles ainda fazem 478 tok/s = 3,86× os nossos 123,9.** Esse é o piso do que se
   ganha **sem nenhuma instrução exótica**, só com estrutura de kernel. É o número que decide a
   ordem do plano.

## 3. A conta em TOPS (aritmética sobre as medições acima)

Trunk = 497 tensores, 11,122 GB de pesos lidos por token (medido pelo nosso
`bench-matvec-shapes-gpu`), **25,622e9 pesos** (contagem independente da frente A e a minha,
que concordam; o número 30,2e9 de uma versão anterior deste documento estava errado). 512
tokens ⇒ **13,12 T MACs**.

| | ms para 512 tokens | T MACs/s | ops/s (2 ops/MAC) | % do pico da instrução que usa |
|---|---|---|---|---|
| **nosso prefill** | 4139,7 | 3,18 | 6,4e12 | **3,3 %** do dp4a (97,4 T) |
| llama.cpp, **coopmat desligado** | 1070,3 | 12,26 | 24,5e12 | 25 % do vetorial f16 (48,7 T) |
| llama.cpp, **coopmat ligado** | 427,9 | 30,66 | 61,3e12 | 31 % do fp16-matriz (97,4 T) |

Tetos deste cartão, por instrução (64 CU × 2,97 GHz; a frente A levantou e eu confirmei a
aritmética — a minha versão anterior deste documento usava 4096 lanes e errava por 2×):

| recurso | MAC/CU/clk | T MAC/s |
|---|---|---|
| FP32 vetorial (FMA) | 128 | 24,3 |
| FP16 vetorial (`v_pk_fma_f16`) | 256 | 48,7 |
| **INT8 vetorial (`v_dot4_i32_iu8`, o nosso)** | **512** | **97,4** |
| FP16 matriz (WMMA/coopmat 16×16×16) | 512 | 97,4 |
| INT8 matriz (WMMA iu8) | 1024 | 194,6 |

Leitura: **dp4a e fp16-WMMA têm o mesmo teto de MAC neste cartão** — a vantagem do WMMA é *por
instrução* (4096 MACs por emissão contra 128 de uma emissão wave32 de dp4a, 32× menos
instruções por MAC), não por ciclo. E nós realizamos **3,3 %** do teto do dp4a que já usamos.

### Onde os nossos 6,6 % se perdem (ISA, já medido)

Por bloco de peso por thread, no kernel em lote com N=16 (`docs/journal-kernels.md` §10):

| | GEMV (1 token) | lote N=16 | por token |
|---|---|---|---|
| `v_dot4_i32_iu8` | 16 | 256 | 16 (escala com N) |
| cargas de ativação | 3 | 48 | 3 (escala com N) |
| cargas de peso + LUT | 13 | 13 | amortizado |
| `v_perm_b32` | 8 | 8 | amortizado |
| demais VALU | ~83 | ~241 | 15 |
| `s_wait_*`/`s_delay_alu` | ~38 | ~282 | **17,6** |
| **total** | ~161 | ~848 | **~53 para 16 dp4a** |

Ou seja: **~3,3 instruções por `dp4a`** (que faz 4 MACs cada) e **33 % das instruções são
espera**. O teto disso, com issue perfeito, é `97,3 / 3,3 ≈ 29 TOPS`; nós realizamos 6,4 (22 %
do teto do próprio desenho). O MMQ do llama.cpp tem **tiling de saída 4×4 por thread** e o peso
dequantizado **na LDS**, o que dá ~1,1-1,5 instruções por 4 MACs em vez de 3,3, e sem as esperas
(o inner loop é uma cadeia longa de dp4a com operandos na LDS).

## 3c. A curva do micro-lote (llama.cpp, medido por mim com `-b N -ub N`)

`llama-bench -m MODEL -p 512 -n 0 -r 3 -b N -ub N`, mesma janela, mesmo modelo:

| micro-lote | tok/s | ms/token | vs nosso prefill (123,4) |
|---|---|---|---|
| **16 (o nosso chunk)** | **199,27** | 5,02 | **1,61×** |
| 64 | 663,77 | 1,51 | 5,38× |
| 128 | 1007,95 | 0,99 | 8,17× |
| 256 | 1017,56 | 0,98 | 8,25× |
| 512 (default) | ~1170 | 0,855 | 9,48× |

Duas leituras que dirigem o plano:

1. **O joelho está em 128.** De 128 para 512 ganha-se só 16 %; de 16 para 128 ganha-se 5,1×.
   Não há razão para perseguir chunks de 512: **128 é o alvo**, e é onde o tile BM=128 do
   llama.cpp encaixa exatamente.
2. **O ganho do chunk não é automático — é do kernel deles.** O nosso `matvec_kernel_batch` tem
   custo `13,9 ms + 6,04 ms/token` (medido), ou seja o chunk maior só amortiza a parcela de
   13,9 ms: em N=64 o modelo prevê 6,26 ms/token contra 6,92 em N=16 (**+10 %**), não 3,3×.
   Quem transforma chunk grande em velocidade é o GEMM tilejado — e é por isso que as duas
   metades do P0 são inseparáveis (o protótipo mediu 3,46 T MACs/s em M=16 e 10,34 T em M=128).

## 3g. Utilização do cartão (aritmética sobre as medições, para fechar esta ponta)

| recurso | nós (prefill 512) | llama.cpp (mesmo prompt) | pico medido | quem usa mais |
|---|---|---|---|---|
| **DRAM (pesos)** | 355,9 GB em 4,140 s = **86,0 GB/s = 13,6 %** do roofline | 11,1 GB em 0,486 s = **22,9 GB/s = 3,6 %** | 633 GB/s | **nós movemos 32× mais byte de peso por token** (695 MB contra 21,7 MB) |
| **MACs** | 24,35e9 MACs/token a **3,01 T-MAC/s = 6,8 %** do pico de dp4a | **25,67 T-MAC/s** no chunk inteiro (= 58,3 % do pico de dp4a; **32,7 T-MAC/s** durante a fase de MUL_MAT, = 74 %) | dp4a 44 / f16-WMMA 89,6 / int8-WMMA 182-198 T-MAC/s | eles, por 8,5× |
| **slots de issue** | V0 do GEMM tilejado: 68 %; V6 (f16): **95 %** | não medido aqui | 3,80e11 instr-warp/s | os dois estão no muro, nós com 10× menos trabalho útil por instrução |
| **LDS** | ~2 % da banda derivada | 24 KB de 64 KB de ocupação no tile deles | (derivado, não medido) | ninguém |

**A leitura que interessa**: nós **não** somos limitados por banda (13,6 % do roofline, 32× mais
tráfego por token e ainda assim longe do teto) e **não** somos limitados por LDS. Somos limitados
por **instruções por MAC** numa forma de kernel que gasta 3,31 instruções por `dp4a` (que faz 4
MACs) e que, com o chunk de 16, não consegue amortizar nem o peso nem a ativação. O llama.cpp move
**32× menos** peso por token e transforma isso em 8,5× mais velocidade: o número que separa os dois
não é banda, é *forma*.

## 3h. Atenção e KV no prefill: medidos, e um BUG no alvo

Frente I (`docs/estudo-prefill-i-atencao.md`), medida com o binário oficial e com o caminho de
produção:

| | 512 tokens | 2048 | 4096 |
|---|---|---|---|
| bloco de atenção (16 camadas, com projeções) | **8,5 %** do prefill | 10,5 % | 12,9 % |
| **kernel de atenção sozinho** | **1,1 %** | 3,2 % | **5,6 %** |
| llama.cpp, nó `FLASH_ATTN_EXT` | **0,34 %** do chunk deles | 1,06 % | — |

- **O crescimento do custo por token de 64 → 4096 (+6,2 %) é o kernel de atenção**: +0,521 dos
  +0,581 ms/token = **89,7 %**. As projeções e a escrita do KV estão dentro do ruído. Ou seja: o que
  cresce com o contexto no prefill é o produto Q·Kᵀ, não o resto.
- **Atenção não é alavanca**: zerá-la daria +1,2 % a 512 e +5,9 % a 4096. O matvec é 83-88 %.
- **Quantizar o KV é neutro para o prefill** (105,11 tok/s `f16`, 105,01 `q8_0`, 104,79 `q4_0` a 512)
  e **piora o kernel** (+24 % com `q4_0` a 2048): no prefill o formato quantizado é **custo de
  instrução, não economia de banda** — o inverso do decode noturno.

### BUG CRÍTICO, e cai no alvo: `K=q5_0`/`V=q4_1` faz PAGE FAULT no prefill em lote

**Reproduzido por mim**, com o binário de produção, depois de a frente I achar:

```
./build/rdna4-infer bench -m MODEL --prefill 64 --prefill-reps 1 --cache-type-k q5_0 --cache-type-v q4_1
  -> Memory access fault by GPU node-1 ... Reason: Page not present or supervisor privilege.
./build/rdna4-infer bench -m MODEL --prefill 64 --prefill-reps 1 --cache-type-k q5_0 --cache-type-v q4_0
  -> prefill 64 tokens: 0.888 s (72.09 tok/s, batched N<=16)     [passa]
```

Matriz da frente I (binário oficial **e** `rdna4-infer`): falha `K ∈ {f16, q5_0}` **com `V = q4_1`**;
passam `q4_0/q4_1`, `q8_0/q4_1`, `q4_1/q4_1`, `q5_0/q5_0`, `q5_0/f16`, `q4_1/q5_0`.
- **Não é o caminho por token**: decode com `q5_0/q4_1` funciona (23,10 tok/s).
- **É o caminho em lote**: falha já com `--prefill 16` (um único chunk).
- `bench-attn-gpu q5_0` também quebra a partir de `t=256` (`HG=3 rel-L2 nan`, depois fault) — dois
  tools independentes apontam para o mesmo caminho, e ele **não** passa por escrita de KV.
- O par `(Q5_0, Q4_1)` **está** instanciado na atenção em lote (`attn.cuh:465`), então não é
  `switch` incompleto. Suspeitos registrados: o `kv_write_batch` passa **o mesmo `width`**
  (`n_tok*NKV*HD`) para K e V com **ponteiros de tamanhos de linha diferentes**
  (`graph.cuh:873-877`), e/ou o caminho de leitura da atenção em lote para `V=q4_1`.

**Isto não é tuning: é um bug de corretude na configuração que a rodada noturna recomendou**
(K=`q5_0`, V=`q4_1`) para os 131K. Ou ele é corrigido, ou o par sai da lista permitida — não pode
ficar como está, porque o alvo do usuário é exatamente esse par. Está no backlog como o item de
maior prioridade de corretude.

## 4. O que nós não temos, em uma tabela (e é isto que foi "pulado")

### Atenção e KV no prefill (frente J, código a código)

- **Atenção não é alavanca por si**: 2,1 % do nosso prefill (87,7 ms de 4174 ms); zerá-la levaria
  122,66 → 125,30 tok/s. Do lado deles o kernel de atenção é **1,76 ms por chunk de 512** = 0,39 %
  do prefill. Por MAC emitido, a flash attention deles é **99×** a nossa — mas o volume é pequeno.
- **O que importa é a *forma***: `Br=16` linhas de consulta por workgroup é o pré-requisito para o
  micro-lote grande (o fator 2,50×), porque sem tile de consultas não há chunk de 128/512 na
  atenção. O port nº 1 da frente J é exatamente isso.
- **KV `f16` não passa por LDS no caminho deles**: o `coopMatLoad` lê K e V **direto da global**
  para o fragmento (`flash_attn_cm1.comp:269-303`). **KV quantizado, sim** — desquantiza no shader e
  é obrigado a estagiar na LDS, e `q8_0` tem um caminho próprio que copia **o cache inteiro** para um
  rascunho f16 (`dequant_q8_0_transpose`, `ggml-vulkan.cpp:11230-11244`).
- **O prefill deles paga 2× de trabalho causal enquanto KV < 1024**: a máscara é um tensor
  `f16 [KV, N]` com `-inf` e o atalho de bloco só liga com `nem0 >= Bc*16 = 1024`
  (`ggml-vulkan.cpp:11309-11310`).

### O limiar que nós não temos: 8 colunas

`mul_mat_vec_max_cols = 8` (`ggml-vulkan.cpp:404`, condição em `:10433`): **até 8 tokens o
Vulkan usa o mesmo `mul_mat_vecq.comp` do decode; do 9º token em diante ele sai do GEMV e entra
no `mul_mm` com tile 128×128** (frente A §1.1). O nosso motor usa o GEMV em lote **até N=16 e
nunca sai dele** — `matvec_launch_batch` é a única porta do caminho em lote
(`include/rdna4/matvec.cuh`), e o `batch_supported()`/`kMaxBatch = 16` fecham o teto. Não é um
número mal ajustado: é a ausência do segundo kernel.

| capacidade | llama.cpp Vulkan | nós | vale |
|---|---|---|---|
| **tiling de saída por thread** | TM×TN = 4×4 (int) / coopmat_m×n, com tile de workgroup 128×128 | **1 elemento de saída por thread** (`acc[ILP]` acumula o MESMO elemento em k) | parte dos 3,86× |
| **peso dequantizado na LDS** | sim (`mul_mmq*.comp`, `types.glsl`), int8 **ou f16** | **não**: o peso é dequantizado em registrador, direto da global | parte dos 3,86× |
| **matrix cores** | `coopmat` 16×16×16 f16→f32 | **zero** WMMA/MFMA em todo o motor | **2,50×** |
| **f16 empacotado no caminho vetorial** | `dot2`/`pk_fma_f16` (1,5× o FMA fp32 neste cartão) | zero no motor (só fp32 escalar) | não medido |
| **fusão de cadeias** | 7 padrões nomeados (`ggml-vulkan.cpp:18149-18327`) | nenhuma; ~2200 kernels/token | ~640 lançamentos/token ≈ **2,2 ms** |
| ~~atenção com grupo GQA~~ | ~~1 workgroup por grupo kv, N=6 cabeças~~ | — | **CORRIGIDO pela frente J: não é lacuna no prefill.** A fusão GQA do Vulkan só vale com `N <= 8` (`ggml-vulkan.cpp:11251-11259`), ou seja **é recurso de decode**; no prefill eles têm as **mesmas 6 leituras redundantes de K/V que nós** |
| **atenção com tile de 16 consultas** | `Br=16 × Bc=64`, coopmat f16→f32, `flash_attn_f32_f16_aligned_cm1`, K/V lidos **direto da global** para o fragmento | a nossa é 1 consulta por CTA e nunca materializa a P | a nossa atenção é **2,1 %** do prefill (87,7 ms de 4174 ms); a deles é **0,39 %**. **Mas o tile de 16 consultas é o pré-requisito do micro-lote grande** — é o ganho *habilitado*, não o próprio |

## 5. O plano que sai disto (ordem por ganho medido, não por gosto)

**P0 — M grande + GEMM tilejado com staging na LDS (3,86× medido do lado deles, sem instrução
exótica; e o protótipo já mediu 3,99× em M=512 contra o nosso prefill).** As duas metades são
inseparáveis: em M=16 o mesmo kernel tilejado rende 1,08× (medido), porque cada byte de peso
estagiado é reusado 16 vezes em vez de 512. Subir o chunk de 16 para ≥128 é **pré-requisito**,
não otimização posterior.
Tiling de saída por thread (começar em 4×4), peso dequantizado para **int8** na LDS (mantém
`dp4a`, que já é o nosso caminho validado), ativação em LDS com leitura vetorizada, escala por
bloco de 32 aplicada como correção no final. Gate: **bit-exatidão não é possível** (muda a ordem
das somas) ⇒ `scripts/compare_ppl.sh` + `check-regression-gpu` + `check-graph-gpu`, como o
`PLAN.md` M2/M3 já exige para mudança numérica. Esperado: 123,9 → ~450-480 tok/s de prefill.

**P1 — matrix cores f16 (2,50× adicional medido do lado deles).** `wmma` f16 16×16×16 → f32
(a frente D mede TOPS de f16 vs int8 vs dp4a neste cartão para confirmar que f16 é o caminho).
Depende do P0: sem o tile e a LDS, a instrução nova não tem de onde ler.

**P2 — fusões** (bit-exatas, ~640 lançamentos/token ≈ 2,2 ms/token de decode e ~14 % do prefill).

**P3 — atenção GQA por workgroup** (contexto longo; +9-11,8 % medido em protótipo).

**P4 — f16 no caminho vetorial** (normas, elementwise, atenção): 1,5× no FMA deste cartão.

## 3d. O andaime, o despacho e as fusões — medidos pela frente C (e a hipótese da fusão MORTA)

Números da frente C (`docs/estudo-prefill-c-nosso.md`), dentro do grafo, nível 2:

| componente | ms/token | % do prefill |
|---|---|---|
| **matvec em lote** | **7,103** (113,6 ms por chunk de 16 no grafo contra 110,70 isolado = **+2,6 %**) | **83,7 %** |
| GDN (48 camadas) | 2,824 (0,852 recorrência + 1,972 projeção) | 34,3 % |
| FFN | 4,523 | 54,9 % |
| atenção + qk_norm_rope_kv + attn_gate_out (16 camadas) | 0,642 (0,031 é o kernel de atenção) | 7,8 % |
| andaime total (norms, resíduos, quantização de ativação, despacho) | 1,141 | ~14 % |

E o número que mata uma hipótese que estava no plano desde a noite:

- **916 lançamentos por chunk** de 16 tokens (20/camada de atenção × 16 + 12/camada GDN × 48 +
  16 embeddings + 2 do head), contados no código;
- piso medido de **2,253 µs** por kernel vazio enfileirado ⇒ **2,06 ms/chunk = 0,129 ms/token =
  1,58 % do prefill**;
- copiar **todas as sete fusões nomeadas do Vulkan** (`ggml-vulkan.cpp:18149-18327`) remove 640
  desses 916 lançamentos ⇒ **1,1 % do prefill**.

**No decode a conta é outra, e também é pequena e medida**: o motor emite ~1 940-2 200
lançamentos por token no decode, e a medida direta do que eles custam juntos é o replay por grafo
HÍP da passagem de 497 matvecs, que economiza **0,76 ms/token** (2,6 % de um token de decode de
29,4 ms). Das ~1 940 instâncias, o conjunto que as sete fusões nomeadas do Vulkan removeria é 640 —
ou seja, a fusão vale **~1-2 % do decode**, não os 6-15 % que o piso de 2,25-3,5 µs por kernel
sugere quando se multiplica ingenuamente (o piso não é aditivo quando os kernels fazem trabalho:
os lançamentos se sobrepõem à execução).

**Conclusão: fusão de kernels não é alavanca nem no prefill (1,1 %) nem no decode (~1-2 %).** Ela continua valendo no decode,
onde o custo é por token e não amortizado por 16 — mas o plano do prefill não deve gastar um dia
nela. Isso contradiz a ordem que eu mesmo tinha escrito de manhã (P2 = fusões) e fica corrigido
aqui.

**Teto independente, pela frente C**: se o matvec em lote rodasse na banda medida de 633 GB/s, o
prefill a 512 tokens seria **446 tok/s** contra os 123,4 nossos — **3,6× de espaço, todo ele na
*forma* do kernel** (o protótipo do P0 chegou a 12,74 T MAC/s = 3,99× em M=512).

## 3e. O protótipo com os blocos REAIS (frente F) — e por que o staging é o alvo

`tests/bench_gemm_quant_gpu.hip` (frente F) pegou o `blk.3.ffn_up.weight` real (iq3_s, 38,3 MB,
**0,4297 B/peso**), dequantizou para **f16 na LDS** e rodou o laço interno em `v_dot2_f32_f16` com
acumulador f32:

| M | frente F (1ª volta) | **frente G (V6, melhor medida)** | vs o motor (3,18 T) |
|---|---|---|---|
| 16 | 4,45 | — | 1,40× |
| 64 | 8,29 | **9,84** | **3,1×** |
| **128** | 9,23 (melhor tile 9,79) | **12,74** | **4,0×** |
| 512 | 11,73 | **14,71** | **4,6×** |

A frente G chegou nesses números com quatro mudanças de staging medidas isoladamente (V6 =
ativação copiada em `int4` + prefetch dos campos do peso em registrador + duplo buffer **só do W** +
ativação lida direto da global, fora da LDS), e **todas as variantes dão C bit a bit igual ao
baseline** (`bitid = 0`, rel-L2 2,07e-4 contra o oráculo): nenhum ganho custou precisão. O staging
caiu de 41-43 % para **18,3 %** do kernel em M=128 (0,50 → 0,164 ms) e a cópia da ativação deixou
de existir.

**Correção da frente G sobre a frente F, por ISA**: o dequant f16 custa **18,4 instruções por peso**
(590 por sub-bloco de 32 por lane), não as ~55 estimadas — o staging é **22 % dos slots de issue,
não 2 %**, e o "43× acima do teto" era artefato dessa subestimativa. Com a contagem certa, **o V6
roda a 3,60e11 instruções de warp/s = 95 % do teto de issue do cartão**: **o caminho f16 nesta forma
está no muro de emissão** — o que resta não é escalonamento, é *remover instrução*. Duas medidas
apontam para onde: o **staging int8 é 1,31× mais rápido que o f16** (0,122 contra 0,159 ms em M=128,
396 contra 590 instruções) e o **dp4a faz 4 MAC por instrução contra 2 do `dot2`**.

**Lição de método registrada pela frente G** (e vale como regra da casa): *a mesma transformação de
staging deu 1,33× numa forma e 0,18× na outra* — no caso ruim, o campo prefetchado ficava vivo
através de um laço de ~2300 instruções com 64 acumuladores e a ISA mostrava **413 `scratch_*`**
(spill) contra 0 na forma boa.

| achado | número |
|---|---|
| **staging real** | **0,514 ms de um kernel de 1,165 ms em M=128 = 42 %**, 173,5 G pesos/s, 74,6 GB/s de fonte contra 633 do cartão |
| natureza do staging | **latência**, não banda nem issue: ocupa ~2 % dos slots de issue |
| f16-dot2 × int8-dp4a (A/B na mesma janela) | **9,23 contra 10,11 T em M=128 (−8,7 %)**, 11,73 contra 12,76 em M=512 (−8,1 %), 4,45 contra 3,62 em M=16 (+23 %) |
| miolo f16 | 18,84 T em M=512 = **97 % do teto de issue do `dot2`** naquele tile (0,625 instrução/MAC) |
| miolo dp4a | 12,76 T = **33 % do teto dele** (39,0 T) |
| precisão do staging f16 | reproduz o dequant do motor **bit a bit** depois do arredondamento f16 (0/2048 bits), erro relativo máx 4,32e-4 = 2⁻¹¹; GEMM ponta a ponta rel-L2 2,07e-4 |

Duas conclusões que fecham o desenho do D2/D4:

1. **No caminho vetorial, f16 não ganha do int8** — e o f16 já está a 97 % do muro dele, enquanto o
   dp4a está a 33 % do dele. Ou seja: **o D2 deve ser int8/dp4a** (que também é o caminho
   bit-exato), e o f16 só se justifica como ponte para o WMMA f16 — que por sua vez é 1,9× mais
   lento que o WMMA int8 neste cartão.
2. **O alvo a atacar chama-se staging**: 42 % do kernel, limitado por latência. É o degrau D3, e é
   ele que decide se o caminho de dados chega perto dos 32,7 T-MAC/s do llama.cpp ou para nos
   10-13 T.

## 3f. O WMMA int8 sobre os blocos reais (frente H) — e ele é BIT-EXATO

Era a pergunta que a frente D deixou aberta: o pico do WMMA int8 é 4,2× o do dp4a, e o caminho int8
*pode* ser bit-exato — mas ninguém tinha rodado WMMA int8 sobre os **blocos reais** dentro de um
GEMM tilejado. A frente H rodou (`tests/bench_mmq_wmma_gpu.hip`, mesmo tensor da frente F:
`blk.3.ffn_up.weight`, iq3_s, N=17408 × K=5120, ativação quantizada pelo kernel do próprio motor).

**Bit-exatidão: 0 elementos diferentes em todos os tamanhos.**

| comparação | elementos | diferenças |
|---|---|---|
| WMMA int8 × dp4a tilejado, M=16 | 278 528 | **0** |
| M=64 | 1 114 112 | **0** |
| M=128 | 2 228 224 | **0** |
| **M=512 (N=17408, o tensor inteiro)** | **8 912 896** | **0** |
| × `vec_dot_iq3_s_q8_1` do motor | 4 096 | **0** |
| × oráculo de CPU | 256 | **0** |

A ISA confirma que o motor e a bancada emitem a **mesma sequência** por bloco de 32:
`v_mul_f32` (d_w·d_a), `v_mul_lo_u32` (**`sumi·(1+2·sc)` em inteiro**), `v_cvt_f32_i32`,
`v_fmac_f32`. Ou seja: o caminho de matrix core **herda os gates bit-exatos que o motor já tem** —
não precisa de gate numérico/PPL, ao contrário da rota f16 (rel-L2 2,07e-4) e ao contrário do que
eu tinha escrito de manhã.

**Velocidade: 7,5 / 16,7 / 19,7 / 22,8 T-MAC/s em M=16/64/128/512** = 3,8-11,8 % do pico
remedido na mesma janela (**198,0 T-MAC/s**; a frente D mediu 182 numa janela diferente — a placa
esteve compartilhada e os absolutos variaram 1,5-1,6× entre janelas, então **o que vale são as
razões intra-janela**: 4,12× WMMA/dp4a no pico). Contra o dp4a tilejado **com o mesmo staging e a mesma correção**, na
mesma janela: 2,82× em M=16, 1,56× em M=64, 1,68× em M=128, **1,83× em M=512**.

**O ganho não é o 4,2× do pico, e o motivo tem nome e número**: a **correção de escala bit-exata
custa 53-56 % do miolo** — 132 instruções por lane por bloco de 32 (32 IMAD + 32 FMUL + 32 CVT +
32 FMA + 4 LDS), contra 8 WMMA; o miolo isolado faz **37,0 T-MAC/s com a correção e 83,9 sem**
(42 % do pico). A receita da frente D está certa na *forma* (`d_a` = 8 floats consecutivos = 2
`LDS.128`, `d_w` = 1 `int2` por tile de N, sem gather) e **subestima o custo por 3,2×**. E o
gargalo volta a ser o **staging: 50-61 % do tempo do kernel em M≥64**, a 132-275 GB/s de fonte
contra 633 de roofline. Duplo buffer **piora 32 %** (14,46 contra 21,37 T-MAC/s em M=512) — o mesmo
resultado negativo das frentes F e G, agora no kernel do WMMA.

### A tabela que fecha o estudo (T-MAC/s, mesmo tensor real, M=512)

| caminho | T-MAC/s | vs o motor (3,18) | bit-exato? |
|---|---|---|---|
| motor hoje (GEMV em lote, dp4a, chunk 16) | 3,18 | 1,0× | sim (é a referência) |
| GEMM tilejado f16 + `dot2` (frente G, V6) | 14,71 | 4,6× | não (rel-L2 2,07e-4) |
| **GEMM tilejado int8 + WMMA (frente H)** | **22,8** | **7,2×** | **SIM, 0/8 912 896** |
| llama.cpp Vulkan (coopmat f16, `-ub 512`) | 32,7 | 10,3× | não |

E os dois gargalos que sobram, medidos: **o staging (50-61 % do tempo do GEMM)** e, no caminho
WMMA, **a correção de escala (53-56 % do miolo)**. O dobro do caminho vetorial acabou: o f16/dot2
está a 95 % do muro de issue dele, e o WMMA int8 entrega 1,8× sobre o dp4a com a mesma precisão
bit-exata.

## 5b. Até onde vai o caminho vetorial (o muro de issue) — e por que a saída é a unidade de matriz

A frente B fechou a conta que o protótipo sugeria: os 12,74 T MAC/s do `bench-gemm-gpu` são
**9,95e10 dp4a de warp/s**, e com as 2,5-3,0 instruções por dp4a que um tile bom gasta, isso é
**65-78 % dos 3,80e11 slots de issue do cartão a 2,97 GHz**. Ou seja: **o protótipo já está a
dois terços do muro da família dp4a, e o muro não é banda nem LDS — é emissão de instrução.**

| caminho | instruções por 4 MAC | MAC por instrução de warp | teto prático |
|---|---|---|---|
| dp4a no nosso kernel (medido) | **3,31** (33 % esperas) | 44,7 | ~27 TOPS |
| dp4a num tile bom (4×4, LDS) | ~2,5-3,0 | 128 | **~12-13 T MAC/s** (o protótipo chegou lá) |
| f16 `v_dot2_f32_f16` empacotado | ~2,56 | 64 | 39-49 TOPS, **saturado de issue** (o fallback do Vulkan roda a ~94 % dos slots) |
| **WMMA/coopmat (matriz)** | **~0,0103** | **4096** | o único caminho acima disso |

Consequência direta para o plano: **trocar dp4a por f16 vetorial não resolve** (mesmo muro, e o
fallback do Vulkan a 478 tok/s está saturado de issue). Passar de ~478 tok/s para ~1200 exige a
**unidade de matriz** — não há terceira via, e as duas famílias de matriz (WMMA int8 do
ggml-cuda, coopmat f16 do Vulkan) são as duas opções.

### Qual família de matriz, e com que acumulador (frente B §7)

Fato de código que eu não sabia e que corrige a leitura das minhas 4 medições: **o Vulkan não
tem pipeline int8 para os tipos IQ** — a lista fechada de pipelines `q8_1`/dp4a cobre
Q2_0/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/MXFP4/Q2_K..Q6_K e **nenhum IQ1/IQ2/IQ3/IQ4**
(`ggml-vulkan.cpp:5235-5246`, seleção em `:9490-9501`). Logo os **59 % de bytes IQ deste modelo
rodam sempre em f16**, e o meu `DISABLE_INTEGER_DOT = −1,4 %` estava medindo só a fatia
k-quant/q8_0. O 2,50× do coopmat é, para eles, **matriz f16 × vetor f16**.

Veredito da frente B, que eu adoto: **para IQ3_S/iq3_xxs/iq4_xs a aposta é o pipe de matriz com
f16 e acumulador f32**, por três razões medidas/verificáveis: (1) precisão — arredondar um peso
de 3-4 bits para f16 erra 4,9e-4, 20-100× menos que o erro da própria quantização, enquanto o
int8 introduz uma quantização nova de 8 bits na ativação; (2) código — f16 dispensa o
quantizador no layout MMQ, os termos de correção e o staging int8 com sinal; (3) existe
implementação **medida** neste cartão (1196 tok/s = ~32,7 T MAC/s com coopmat f16), enquanto
int8 WMMA não tem medição nenhuma no gfx1201. **Ressalva obrigatória: acumulador f32** — nunca
`v_pk_fma_f16`, que acumula em f16. O dp4a continua sendo a escolha certa no **decode** (M=1):
4 MAC/lane contra 2.

De quebra, a rota f16 mata dois custos que a frente C mediu no nosso prefill: a quantização de
ativação (`act_quant`, **0,227 ms/token**) deixa de existir como bloco de 32 com escala e soma —
vira uma conversão para f16.

## 6. Status das frentes

| frente | arquivo | estado |
|---|---|---|
| A — código do Vulkan (qual shader, tiles, coopmat f16?) | `docs/estudo-prefill-a-vulkan.md` | em andamento |
| B — MMQ do ggml-cuda (o que cabe no gfx1201) | `docs/estudo-prefill-b-mmq.md` | em andamento |
| C — orçamento por fase do NOSSO prefill | `docs/estudo-prefill-c-nosso.md` | em andamento |
| D — WMMA f16/int8 vs dp4a em TOPS (a medição decisiva) | `docs/estudo-prefill-d-wmma.md` | em andamento |
| E — fusões: inventário e efetivação | a definir | pendente |

## 7. Reprodutibilidade (comandos exatos)

```bash
# referência (llama.cpp Vulkan), sob o lock da GPU
./scripts/gpu-lock.sh timeout 900 /home/marcelo/Projetos/llama.cpp/build/bin/llama-bench \
  -m /mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf -p 512 -n 0 -r 3 -o md

# a decomposição: matrix cores fora
GGML_VK_DISABLE_COOPMAT=1 GGML_VK_DISABLE_COOPMAT2=1 <mesmo comando> -p 512 -n 0 -r 2

# o nosso lado
./scripts/gpu-lock.sh timeout 900 ./build/rdna4-infer bench -m <modelo> --prefill 512 --prefill-reps 3
./scripts/gpu-lock.sh timeout 900 ./build/bench-matvec-shapes-gpu <modelo> --batch 2,4,8,16
```

## 3b. O protótipo tilejado (medido, com verificação) — e a descoberta do tamanho do chunk

`tests/bench_gemm_gpu.hip` (novo, `bench-gemm-gpu`) implementa o GEMM int8 com a estrutura do
MMQ — tile de saída `BM × BN` por CTA, `RM × RN` acumuladores por thread, A e W estagiados na
LDS, `dp4a` consumindo os dois de lá — e **verifica cada configuração contra um oráculo**
(kernel naive de 1 elemento por thread, sem LDS) antes de cronometrar. A primeira versão desta
bancada deu 1198 T MACs/s (24× o pico do cartão) porque o lançamento tinha número de threads
errado e o kernel **não rodava**; o `hipGetLastError` e a verificação por configuração estão lá
por causa disso.

Forma real: tensor do tronco `17408 × 5120`, int8, 89,1 MB de peso, `BK=64`.

| M (tokens no chunk) | tiled+LDS T MACs/s | vs **nosso prefill** (3,19 T) | vs llama.cpp s/ coopmat (12,26 T) |
|---|---|---|---|
| 16 (o NOSSO chunk hoje) | **3,46** | **1,08×** | 28 % |
| 64 | 7,23 | 2,27× | 59 % |
| 128 | 10,34 | 3,24× | 84 % |
| 256 | 11,91 | 3,73× | 97 % |
| **512** | **12,74** | **3,99×** | **104 %** |

Três leituras, e a primeira é a que reorganiza o plano:

1. **Em M=16, o GEMM tilejado NÃO ganha nada** (3,46 contra 3,19 T = 1,08×). O motivo é
   aritmético: com 16 tokens, cada byte de peso estagiado na LDS é reusado só 16 vezes, e o
   custo de estagiar (mais o `__syncthreads` por bloco de k) come o ganho do dp4a melhor
   alimentado. Em M=16 o desenho certo é o GEMV que já temos (peso direto da DRAM, sem LDS).
   **Ou seja: o nosso motor não é "ingênuo por escolha de kernel" — ele está estruturalmente
   limitado pelo chunk de 16 tokens.** O `pass_ms ≈ 13,9 + 6,04·N` de hoje é a assinatura disso.
2. **O ganho de 3,9× do llama.cpp é reproduzível com dp4a puro, sem matrix cores nenhum**: o
   protótipo chega a **12,74 T MACs/s em M=512**, contra os **12,26 T** medidos do llama.cpp com
   coopmat desligado (104 %). O "fator 3,86×" é *estrutura* (tile + LDS + M grande), não
   instrução exótica.
3. **O teto do caminho vetorial ainda é 2,4× menor que o do coopmat** (12,74 contra os 30,66 T
   do llama.cpp com coopmat ligado). Ou seja: **os dois fatores do gap são independentes e ambos
   são necessários** — e a ordem certa é: primeiro M grande + kernel tilejado (3,9×), depois
   matrix cores (2,4×).

### Consequência para o plano: o chunk de 16 tokens é a alavanca esquecida

Não é só o kernel. Com M=16, cada byte de peso é lido **32× mais vezes** do que com M=512:
11,122 GB por chunk de 16 = 695 MB por token contra 21,7 MB por token em M=512. O piso de banda
disso, a 633 GB/s medidos, é **1,098 ms/token = 911 tok/s** em M=16 — e os 1114 tok/s do
llama.cpp ficam *acima* desse piso, o que só é possível porque eles processam o prompt inteiro
num "M" grande (o `pp2048 ≈ pp512` deles é a assinatura de ser compute-bound, não banda-bound).
Nós estamos a 8,08 ms/token, isto é 7,4× acima do nosso próprio piso de banda.

Portanto o P0 tem **duas** metades, e a segunda não funciona sem a primeira:
**(i) subir o M do prefill de 16 para ≥128** (buffers de ativação, atenção causal com M>128,
laço sequencial do GDN dentro do chunk, quantização em lote com M grande) e
**(ii) o GEMM tilejado com staging na LDS** — o protótipo acima, que já roda a 10,3-12,7 T
MACs/s e é verificado contra oráculo.
