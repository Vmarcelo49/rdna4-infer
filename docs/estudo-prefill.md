# Por que o prefill do llama.cpp é 9,7× mais rápido — estudo com medição (14/09, dia)

Pergunta do usuário, literal: *"how does llama.cpp prefill is 9 to 10 times faster than us? what
did we skip? what should we port or optimize?"*. Este arquivo é o documento-síntese do dia; cada
frente tem o seu relatório (`docs/estudo-prefill-{a-vulkan,b-mmq,c-nosso,d-wmma}.md`).

## TL;DR — o gap tem TRÊS fatores, todos medidos nesta máquina

| fator | de → para | ganho | como foi medido |
|---|---|---|---|
| **1. tamanho do micro-lote** | chunk de 16 → 512 tokens | **2,39×** | `llama-bench -b/-ub`: o **próprio llama.cpp** cai de 1168,49 para **200,25 tok/s** quando o micro-lote desce de 512 para 16 (frente C) |
| **2. estrutura do kernel** | GEMV em lote → GEMM tilejado com LDS | **1,62×** (em M=16) / **3,9×** (em M≥128) | protótipo `bench-gemm-gpu` verificado: 3,46 T MACs/s em M=16 (1,08× o nosso!) contra **12,74 T em M=512** (104 % do llama.cpp sem coopmat) |
| **3. unidades de matriz** | dp4a → coopmat/WMMA | **2,50×** | `GGML_VK_DISABLE_COOPMAT=1`: 1196,49 → **478,38 tok/s** (mesma forma de tile, mesma precisão de referência) |

1,62 × 2,39 × 2,50 = **9,68×**, contra os **9,66×** medidos de ponta a ponta (1196,49 contra
123,68). **A maior alavanca isolada não é o kernel: é o chunk de 16 tokens** — e ela é
*pré-requisito* da segunda, porque em M=16 um GEMM tilejado não ganha nada (medido: 1,08×).

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

## 4. O que nós não temos, em uma tabela (e é isto que foi "pulado")

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
| **atenção com grupo GQA** | 1 workgroup por grupo kv, N=6 cabeças, `split_k=32` | 6 leituras redundantes de K/V por token | já medido: +9,0-11,8 % a 64K/131K |

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

**Conclusão: para o prefill, fusão de kernels não é alavanca.** Ela continua valendo no decode,
onde o custo é por token e não amortizado por 16 — mas o plano do prefill não deve gastar um dia
nela. Isso contradiz a ordem que eu mesmo tinha escrito de manhã (P2 = fusões) e fica corrigido
aqui.

**Teto independente, pela frente C**: se o matvec em lote rodasse na banda medida de 633 GB/s, o
prefill a 512 tokens seria **446 tok/s** contra os 123,4 nossos — **3,6× de espaço, todo ele na
*forma* do kernel** (o protótipo do P0 chegou a 12,74 T MAC/s = 3,99× em M=512).

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
