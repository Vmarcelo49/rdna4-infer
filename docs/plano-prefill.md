# Plano do prefill: do chunk de 16 ao tilejado com unidade de matriz

Este é o plano de implementação que sai do estudo de 14/09 (`docs/estudo-prefill.md` e as quatro
frentes `docs/estudo-prefill-{a-vulkan,b-mmq,c-nosso,d-wmma}.md`). Todo número aqui tem a medição
que o sustenta; nenhuma escolha é por preferência. O que **não** está medido está marcado
`INFERIDO`.

## 0. Alvo e degraus

| degrau | o que muda | ganho medido/derivado | prefill esperado |
|---|---|---|---|
| **hoje** | chunk 16 + GEMV em lote (dp4a, 1 saída/thread, sem LDS) | gap real **8,54×** (não 9,7×: ver nota de integridade em `estudo-prefill.md` §0) | **123,4 tok/s** = 3,0-3,2 T-MAC/s (7 % do pico de dp4a medido) |
| ~~D1~~ | chunk maior com o MESMO kernel | **MEDIDO E REPROVADO**: −18 % em N=32, −71 % em N=64 (§0.1) | **não fazer** |
| **D2** | GEMM tilejado com staging na LDS, **M=64-512** | **12,74 T em M=128 e 14,71 em M=512 (4,0-4,6×)**, todas as variantes bit-exatas | **~331-347 tok/s** |
| **D3** | D2 + **caminho de dados consertado** (BK=128/linha de cache cheia, ordem k-maior, mais warps) | não medido; é o que destrava os dois caminhos (hoje 58-94 % do tempo é staging, a 280 GB/s de 633) | a medir |
| **D4** | D2/D3 + laço interno **WMMA int8** | **MEDIDO pela frente H: 19,7 T em M=128 e 22,8 em M=512 = 6,2-7,2×, e BIT-EXATO (0/8 912 896)** | **~470 tok/s** (Amdahl §1b) |
| referência | llama.cpp Vulkan, `-ub 512` (build **limpo**) | 32,7 T-MAC/s (74 % do pico de dp4a, 37 % do teto f16-WMMA) | **1054,3 tok/s** |

Os degraus não são independentes: **D2 não rende nada sem D1** (medido: o mesmo GEMM tilejado em
M=16 dá 3,46 T MAC/s = 1,08× o kernel de hoje; em M=128 dá 10,34 T = 3,24×). E **D3 não existe
sem o staging de D2** — a unidade de matriz precisa ler os operandos da LDS.


### 0.1 O degrau D1 foi medido e REPROVOU (14/09)

A previsão do modelo de custo era +10 % em N=64 (só a parcela fixa de 13,9 ms amortiza). Medido
com `bench-matvec-shapes-gpu --batch 8,16,32,64` (inventário real, min de 3):

| N | ms/token | vs N=16 | registradores (`iq3_s`) | scratch |
|---|---|---|---|---|
| 8 | 8,074 | +16 % | — | 0 |
| **16** | **6,964** | — | **62** | 0 |
| 32 | 8,242 | **+18 %** | 78 | 0 |
| 64 | 11,916 | **+71 %** | **133** | **0 (sem spill)** |

**Mecanismo**: `acc[N][ILP]` custa N registradores por thread; em N=64 o `iq3_s` vai de 62 para
133 registradores, o que **derruba a ocupação pela metade** (133 × 256 threads = 34 k registradores
por CTA) — e como este kernel é limitado por latência, perder ocupação custa mais do que o peso
amortizado ganha. **Não há spill** (`localSizeBytes = 0` em todos), então não é transbordo: é
ocupação.

Consequências para o plano:
1. **O degrau D1 não existe como degrau.** Subir o chunk sem trocar o kernel é **regressão**.
2. Isso *reforça* a dependência do D2: quem dá reuse em M grande é o **staging na LDS** (o peso
   entra uma vez por tile e serve os 128 tokens), não mais acumuladores por thread. O tilejado
   usa 8×8 = 64 acumuladores *fixos* (independentes de M), com o M grande multiplicando o
   trabalho por CTA e não os registradores.
3. O `kMaxBatch = 16` do motor não é um número mal escolhido para o kernel que existe — é o ótimo
   dele (medido). Só faz sentido mudar junto com o kernel novo.

## 1. A ordem certa: caminho de dados primeiro, instrução depois

Isto é o que a frente D mediu e o que corrige o plano que eu tinha escrito de manhã:

| caminho | pico medido neste cartão | o que os protótipos **realmente** alcançam |
|---|---|---|
| dp4a (`v_dot4_i32_iu8`) | **44 T-MAC/s = 88 TOPS** | 12,74 T-MAC/s (o meu, int8, M=512) = **29 % do pico** |
| WMMA f16 16×16×16 (o que o Vulkan usa) | **89,6 T-MAC/s** (acc f32) / 95,1 (acc f16) | 10,3-13,7 T-MAC/s (minigemm da frente D, M=64) = **11-15 % do pico** |
| WMMA int8 16×16×16 | **182 T-MAC/s = 364 TOPS** | não testado ponta a ponta |

Dois protótipos escritos de forma independente, com instruções internas diferentes, chegam ao
**mesmo lugar (~10-14 T-MAC/s)** — e a frente D diagnosticou o porquê com o mesmo kernel em três
modos: **58-94 % do tempo é staging (global→LDS)**, rodando a **280 GB/s de 633** porque lê 64 B
por linha de peso com stride de 5120 B (meia linha de cache), e **sem sobreposição** entre
staging e compute quando o mesmo warp faz os dois.

Consequências, e são elas que ordenam o plano:

1. **Trocar dp4a por WMMA sem arrumar o caminho de dados não acelera nada** (o teto sobe de 44
   para 182 T-MAC/s; o tempo continua sendo o staging). A frente D foi explícita: *"WMMA compra
   teto, não velocidade"*.
2. **Mas o teto é necessário para empatar com o llama.cpp**: eles fazem 32,7 T-MAC/s, que é
   **74 % do pico do dp4a** — nenhum protótipo de dp4a chegou perto disso (29 % no melhor caso),
   enquanto é apenas **37 % do teto do f16-WMMA** e 18 % do int8-WMMA. Ou seja: **D2/D3
   (caminho de dados) primeiro, D4 (unidade de matriz) depois** — e não o contrário.
3. **A boa notícia do dia**: um MMQ int8 com WMMA **pode ser bit-exato**. A frente D mediu
   **0 divergências em 40960** comparações contra o dp4a ao longo de K=5120, com o GEMM int8
   batendo a referência de CPU **bit a bit (0/2048)**, desde que se acumule int32 por bloco de 32
   e a multiplicação `1+2*sc` do IQ3_S seja feita **em inteiro** antes do `d_w*d_a` em fp32
   (15×31 = 465 não cabe em int8 — é o mesmo cuidado que `vecdotq.cuh:725-728` já tem). Isso
   significa que a rota int8 mantém os gates bit-exatos, enquanto a rota f16 (a do Vulkan) exige
   gate numérico: **int8 é 1,9× mais rápido que f16 neste cartão e ainda é bit-exato.**

### A receita do operando (medida pela frente D, para não ser redescoberta)

```
A: lane L, byte p (0..7) = A[m = L%16][k = 8*(L/16) + p]
B: lane L, byte p (0..7) = B[n = L%16][k = 8*(L/16) + p]     (B guardado [n][k], linha = n)
D: slot j do lane L      = C[m = 8*(L/16) + j][n = L%16]    (C TRANSPOSTO em relação a A/B)
```
`__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(signed_a, a, signed_b, b, c, clamp)`;
f16 usa `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` com o **mesmo empacotamento**.
A transposição de D é boa notícia para a correção de escala: `d_w` é **1 valor por lane** e `d_a`
são **8 floats consecutivos** (os 8 m do lane) — 2 `LDS.128` + 1 `LDS.32` + 8 FMA por lane por
bloco de 32, sem gather.

## 1b. O teto real de cada degrau, por Amdahl (aritmética sobre o orçamento medido)

A frente C mediu o orçamento dentro do grafo a N=64: **matvec 7,103 ms/token**, andaime
**1,13 ms/token** (dos quais **GDN recorrência 0,852**), total 8,233 ms/token = 123,4 tok/s. Com
isso dá para dizer o que cada degrau entrega de verdade — e é menos do que a razão do kernel
sugere, porque o andaime não melhora junto:

| degrau | T-MAC/s do matvec | fator | matvec ms/token | **prefill total** | ganho real |
|---|---|---|---|---|---|
| hoje | 3,18 | 1,00× | 7,103 | **123,4 tok/s** | — |
| D2 (frente G, V6) | 12,74 (M=128) | 4,0× | 1,774 | **~331 tok/s** | **2,7×** |
| D2 em M=512 | 14,71 | 4,6× | 1,538 | **~347 tok/s** | 2,8× |
| D4 (frente H, medido) | 22,8 | 7,2× | 0,990 | **~471 tok/s** | **3,8×** |
| D4 com o staging da frente G (EXTRAPOLAÇÃO da frente H, não medida) | 27-34 | 8,5-10,7× | 0,66-0,84 | **~508-559 tok/s** | 4,1-4,5× |
| e o teto do próprio WMMA com correção grátis | ~32 | 10,1× | 0,70 | **~545 tok/s** | 4,4× |
| D4 no ritmo do llama.cpp | 32,7 | 10,3× | 0,690 | **~548 tok/s** | 4,4× |
| D4 + andaime no ritmo deles | 32,7 | 10,3× | 0,690 | **~1050 tok/s** | 8,5× |

**A lição de planejamento**: com o matvec resolvido, **o andaime passa a mandar** — 0,852 ms/token
só de recorrência do GDN é 29 % do que sobraria no melhor caso. O degrau que chega a ~550 tok/s
exige, além do GEMM, atacar a recorrência do GDN (é o item 5 da tabela do §3, e é o único item do
andaime com tamanho para importar). A frente C chegou ao mesmo teto por outro caminho (446 tok/s
com o matvec na banda de 633 GB/s).

## 2. Decisões fixadas pelo estudo

1. **f16, não int8, para os tipos IQ** (frente B §7, com argumento de precisão e de código):
   arredondar um peso de 3-4 bits para f16 erra 4,9e-4 — 20-100× menos que o erro da própria
   quantização —, enquanto o int8 introduz uma quantização nova de 8 bits na ativação; e f16
   dispensa o quantizador no layout MMQ, os termos de correção e o staging int8 com sinal.
   **Acumulador f32 sempre**: nunca `v_pk_fma_f16`.
   → Consequência de projeto: **a ativação do prefill passa a ser f16, não `q8_1`**, o que de
   quebra elimina o kernel `act_quant` que a frente C mediu em **0,227 ms/token**.
2. **O dp4a fica no decode** (M=1): 4 MAC/lane contra 2 do `dot2`. Não mexer no caminho por
   token, que está a 1,16-1,30× do llama.cpp e é bit-exato com gate.
3. **Alvo de chunk = 128, não 512.** A curva medida do llama.cpp tem o joelho em 128
   (16 → 199, 64 → 664, **128 → 1008**, 256 → 1018, 512 → 1170). 128 é também o `BM` do tile
   deles (`ggml-vulkan.cpp:4524-4532`), então o mesmo número serve para o tile e para o chunk.
4. **Limiar GEMV/GEMM explícito**, como o `mul_mat_vec_max_cols = 8` do Vulkan
   (`ggml-vulkan.cpp:404`, decisão em `:10433`): com N ≤ 8 usa o kernel por token; acima disso,
   o GEMM. Hoje `matvec_launch_batch` é a única porta e não há segunda opção.
5. **Tile**: `BM=128 × BN=128 × BK=32`, 256 threads, `RM=RN=8` por thread (o protótipo já usa
   exatamente isto e passa na verificação contra oráculo). LDS: A 128×32×2 B = 8 KB, W
   128×32×2 B = 8 KB, duplo buffer = 32 KB de 64 KB disponíveis (`maxComputeSharedMemorySize =
   65536` no cartão, medido pela frente A) — sobra para a escada de `coopmat` (24 KB na versão do
   Vulkan).

## 3. O que muda no motor, arquivo por arquivo (`INFERIDO` quanto ao esforço)

| # | mudança | arquivos | por que |
|---|---|---|---|
| 1 | chunk do prefill 16 → 128 | `include/rdna4/graph.cuh` (`kMaxBatch`, `batch_supported`, buffers `d_aqb_`/`d_xb_`), `src/main.hip` (`prefill_ids`) | sem isto o GEMM não tem M para amortizar (medido: 1,08× em M=16) |
| 2 | GEMM f16 tilejado + staging na LDS | novo `include/rdna4/gemm.cuh`; variante bench primeiro | o kernel do D2/D3 |
| 3 | ativação f16 no lote (substitui `quantize_q8_1_batch`) | `include/rdna4/matvec.cuh` (`quantize_batch`), `graph.cuh` | remove `act_quant` (0,227 ms/token) e é o formato que o `coopmat` consome |
| 4 | atenção causal em lote com M=128 | `include/rdna4/attn.cuh` (`attn_batch_kernel`, `attn_split_batch_launch`) | hoje o lote é por token; a frente C mediu `attention` + `qk_norm_rope_kv` + `attn_gate_out` em 0,088 ms/token, e ela precisa acompanhar o M |
| 5 | laço sequencial do GDN dentro de um chunk de 128 | `include/rdna4/gdn.cuh` (`delta_rule`, conv) | a recorrência continua sequencial; só a *projeção* vira GEMM (é o que o llama.cpp faz: matmul com M grande e o scan por dentro) |
| 6 | laço interno `coopmat` f16→f32 | `gemm.cuh` (só o corpo interno muda) | D3 |

### Atenção e KV no prefill (frente J) — o que entra no P0 e o que não entra

- **Atenção não é alavanca de tempo**: 2,1 % do nosso prefill (87,7 ms de 4174 ms); zerá-la levaria
  122,66 → 125,30 tok/s e o gap de 8,54× viraria 8,41×. Do lado deles, 1,76 ms por chunk de 512 =
  0,39 %.
- **Mas o *tile* de atenção entra no P0 como pré-requisito**: `Br=16` linhas de consulta por
  workgroup (`ggml-vulkan.cpp:4014-4023`) é o que permite o chunk de 128/512 na atenção; sem ele,
  subir o chunk só multiplica CTAs. É o port nº 1 da frente J
  (`docs/estudo-prefill-j-atencao-kv.md` §6).
- **KV no prefill**: `f16` no caminho deles **não passa por LDS** (o fragmento coopmat lê direto da
  global); quantizado **é obrigado** a estagiar na LDS, e `q8_0` tem um caminho que copia **o cache
  inteiro** para um rascunho f16 por camada por chunk. Ou seja: **no prefill o formato quantizado é
  custo de instrução, não economia de banda** — a razão inverte em relação ao decode noturno. Isso
  não contradiz o alvo de `q5_0`/`q4_1` (que é uma escolha de *memória*, para caber 131K), mas
  significa que não se deve esperar ganho de velocidade de prefill vindo do formato.

## 4. Gates (o caminho **não** é bit-exato, e isso é uma decisão consciente)

O GEMM muda a ordem das somas e o tipo da ativação, então **não vale bit-exatidão** — o
`PLAN.md` M2/M3 já fixou o protocolo para mudança numérica:

- `scripts/compare_ppl.sh` (PPL contra llama.cpp por posição) com limite declarado **0,5 %**;
- `check-regression-gpu` e `check-graph-gpu` (oráculo por nó) para a estrutura;
- `scripts/check_golden_run.sh` para a geração gananciosa;
- o caminho antigo fica atrás de um `RD_*` explícito (padrão da casa) para A/B intercalado;
- **precisão do f16 precisa de um número, não de uma promessa**: medir a KL/PPL do prefill em
  f16 contra o prefill em `q8_1`+dp4a na mesma prompt, e reportar as duas.

## 5. Riscos, e o que já os reduz

| risco | estado |
|---|---|
| staging da dequantização custa mais que o ganho | medido do lado deles: o mesmo staging é usado a 478 tok/s **sem** unidade de matriz; e o nosso protótipo com staging já dá 3,24× em M=128 |
| LDS estoura | 32 KB de 64 KB no duplo buffer; o tile do Vulkan usa 24 KB (frente A) |
| ganho some no andaime | a frente C mediu: o andaime é 9,7 % do prefill e o despacho 0,06 % — o ganho não some |
| chunk 128 quebra a atenção/GDN | é o item 4/5 da tabela; a frente C mediu os dois como 8,7 % do prefill somados, então há espaço para pagar |
| `coopmat` não ser alcançável do HIP/gfx1201 | é exatamente o que a frente D mede agora; se falhar, o plano para em D2 (~480 tok/s = 3,9×) |

## 6. Ordem de execução proposta (revisada com tudo o que foi medido hoje)

O caminho está medido de ponta a ponta: cada degrau tem número, e os dois gargalos que sobram estão
localizados.

1. **D2 — GEMM tilejado com os blocos reais, caminho int8.** Base medida: **12,74 T-MAC/s em M=128**
   (frente G). Configuração recomendada, toda medida: `BM=128 BN=128 BK=64 RM=RN=8`, 256 threads,
   **≤32 KB de LDS por CTA** (para 2 CTAs/CU), staging de 1 sub-bloco de 32 pesos por thread,
   ativação em `int4`, prefetch dos campos do peso em registrador, **duplo buffer só do W**, ativação
   lida direto da global, padding de 1 `half2` por linha. **Não** usar duplo buffer de ativação
   (medido: −17 %) e **cuidado com o prefetch sem liberar LDS** (medido: 413 `scratch_*` = 0,18×).
   Gate: como o caminho int8 é bit-exato, valem os gates que já existem (`check-matmul-gpu`,
   `check-batch-gpu`, `check-graph-gpu`, golden) — **não** é preciso PPL.
2. **D4 — trocar o miolo por WMMA int8** (mesmo staging, mesma correção): **19,7 T em M=128 e 22,8
   em M=512 (6,2-7,2×), bit-exato com 0 de 8 912 896 elementos** (frente H). A receita do operando
   está em `docs/estudo-prefill-d-wmma.md` §(c) e a da correção (D transposto → `d_w` 1 valor por
   lane, `d_a` 8 floats consecutivos) em §(a) da frente H.
3. **D3 — staging**, que é o gargalo comum e **o item nº 1 que sobra**: 50-61 % do tempo do GEMM em
   M≥64, a 132-275 GB/s de fonte contra 633 de roofline; e no WMMA a **correção de escala custa
   53-56 % do miolo**. As duas coisas são *remoção de instrução*, não escalonamento: o candidato
   medido é o staging int8 (1,31× mais rápido que o f16, 396 contra 590 instruções por sub-bloco),
   e o candidato não medido é especialização de warps (produtor/consumidor) e CTA persistente para
   a cauda de onda (11,8 % em M=128, medido).
4. **Chunk de 128** — pré-requisito de tudo acima (`kMaxBatch`, buffers, atenção causal com M>128,
   laço do GDN dentro do chunk). **Medido: subir o chunk com o kernel ATUAL regride** (−18 % em
   N=32, −71 % em N=64), então isto anda junto com o item 1, não antes.
5. **Recorrência do GDN (0,852 ms/token)** — é o único item do andaime com tamanho para importar
   depois que o matvec estiver resolvido: no degrau D4 ele passa a ser ~52 % do que sobra (Amdahl
   §1b).
6. **O que NÃO fazer** (todas refutadas por medição hoje): fusão de kernels para o prefill (1,1 %),
   subir o chunk com o kernel atual (regressão), duplo buffer de ativação (−17 %), ampliar o tile
   em M para matar a cauda de onda (−13 %), `UNROLL` no kernel em lote (+5,6 % pior), staging LDS da
   ativação (−5 %), persistir com o caminho f16 vetorial (95 % do muro de issue dele).
