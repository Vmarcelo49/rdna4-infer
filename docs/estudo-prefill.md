# Por que o prefill do llama.cpp é 9,7× mais rápido — estudo com medição (14/09, dia)

Pergunta do usuário, literal: *"how does llama.cpp prefill is 9 to 10 times faster than us? what
did we skip? what should we port or optimize?"*. Este arquivo é o documento-síntese do dia;
cada frente tem o seu próprio relatório (`docs/estudo-prefill-{a-vulkan,b-mmq,c-nosso,d-wmma}.md`).

## TL;DR — a resposta em três linhas

O gap é **multiplicativo e já está decomposto por medição**: **3,86×** vem de *tiling de saída +
staging na LDS* (llama.cpp faz esse número **com os matrix cores desligados** por variável de
ambiente) e **2,50×** adicionais vêm do **cooperative matrix** (f16, não int8 — desligar o
integer-dot não muda nada). 3,86 × 2,50 = **9,66×**, contra os **9,66×** medidos de ponta a ponta.
Nós não estamos a "9× de distância de um kernel bom": estamos a **3,9× de um kernel de GEMM
tilejado comum** e depois a 2,5× de usar os matrix cores.

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
`bench-matvec-shapes-gpu`), ≈25,8 G pesos (11,122 GB a 3,44 bpw). 512 tokens ⇒ **13,2 T MACs**.

| | ms para 512 tokens | T MACs/s | TOPS (int8-equivalente) | % do pico dp4a |
|---|---|---|---|---|
| **nosso prefill** | 4139,7 | 3,19 | **6,4** | **6,6 %** |
| llama.cpp, **coopmat desligado** | 1070,3 | 12,26 | **24,5** | 25,2 % |
| llama.cpp, **coopmat ligado** | 427,9 | 30,66 | **61,3** | 126 % |

Pico dp4a teórico deste cartão (`4096 lanes × 2,97 GHz × 4 MAC`): **48,7 T MACs/s = 97,3 TOPS**.
O llama.cpp passa de 100 % desse pico quando usa matrix cores — é a prova aritmética de que eles
não estão fazendo dp4a nesse caminho. E nós estamos a **6,6 %** do que o dp4a permite.

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

## 4. O que nós não temos, em uma tabela (e é isto que foi "pulado")

| capacidade | llama.cpp Vulkan | nós | vale |
|---|---|---|---|
| **tiling de saída por thread** | TM×TN = 4×4 (int) / coopmat_m×n, com tile de workgroup 128×128 | **1 elemento de saída por thread** (`acc[ILP]` acumula o MESMO elemento em k) | parte dos 3,86× |
| **peso dequantizado na LDS** | sim (`mul_mmq*.comp`, `types.glsl`), int8 **ou f16** | **não**: o peso é dequantizado em registrador, direto da global | parte dos 3,86× |
| **matrix cores** | `coopmat` 16×16×16 f16→f32 | **zero** WMMA/MFMA em todo o motor | **2,50×** |
| **f16 empacotado no caminho vetorial** | `dot2`/`pk_fma_f16` (1,5× o FMA fp32 neste cartão) | zero no motor (só fp32 escalar) | não medido |
| **fusão de cadeias** | 7 padrões nomeados (`ggml-vulkan.cpp:18149-18327`) | nenhuma; ~2200 kernels/token | ~640 lançamentos/token ≈ **2,2 ms** |
| **atenção com grupo GQA** | 1 workgroup por grupo kv, N=6 cabeças, `split_k=32` | 6 leituras redundantes de K/V por token | já medido: +9,0-11,8 % a 64K/131K |

## 5. O plano que sai disto (ordem por ganho medido, não por gosto)

**P0 — GEMM tilejado com staging na LDS (3,86× medido do lado deles, sem instrução exótica).**
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
