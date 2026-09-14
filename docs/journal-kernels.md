# Diário — frente **kernels de decode** (`feat/noite-kernels`)

Worktree `../rdna4-wt-noite-kernels`, base `main` = tag `noite-baseline-2026-09-14` (`e48f3c4`).
Escopo: kernels do decode por token (`matvec.cuh`, `vecdotq.cuh`, `quant_tables.h`, `nn.cuh`,
`gdn.cuh`, `tuning.h`, `attn.cuh` pontual, `tests/`, `docs/`). Toda medida com
`timeout 900 ./scripts/gpu-lock.sh ...`; VRAM (`mem_info_vram_used`) anotada antes/depois.
Contrato: `docs/noite-regras.md`. Números de partida: `docs/medicoes-banda-e-gargalos.md`,
`docs/autotuning-gfx1201.md`, `docs/vulkan-vs-hip.md`.

## 0. Baseline e pisos de ruído dos harnesses (medidos antes de qualquer mudança)

Sem isso nenhum "ganho" abaixo vale. Todos com o lock e em janela verificada
(`fuser -v /dev/kfd` vazio, `mem_info_vram_used` < 200 MiB antes).

| harness | comando | resultado | piso de ruído medido |
|---|---|---|---|
| decode 4K real (fill-cache 4090) | `./scripts/gpu-lock.sh ./build/rdna4-infer bench -m IQ3_S -p "The capital of France is" -n 16 --reps 3 --ctx-size 4096 --start-pos 4090 --fill-cache` (3×) | **28,66 / 28,77 / 28,60 tok/s** = 34,76-34,96 ms/token; 344-346 GB/s efetivos | **±0,3 %** entre corridas idênticas |
| `bench-matvec-shapes-gpu` | ver §1 | ver §1 | **1,001x** (`ship` vs `ship2`, herdado da campanha de autotuning; reconferido nesta rodada) |
| `check-matvec-gpu --bench-ab` | ver §1 | ver §1 | ver §1 |

VRAM antes da bateria: 198 MB. Depois da bateria de baseline: 651 MB (nenhum processo vivo).

## 1. Atribuição do `iq3_s` (33 % do tráfego do matvec): quanto custa a LUT global?

- **Referência**: `docs/vulkan-vs-hip.md` §1.1/§4.2 (`iq3_s` faz 8 `global_load_b32` por
  `vec_dot` em `vecdotq.cuh:891-892`; o Vulkan põe a `iq3s_grid` em `shared`,
  `ggml-vulkan.cpp:4172` e paga 2 KB de LDS) + `docs/autotuning-gfx1201.md` §5.2 (443 GB/s
  contra teto de leitura de 619 GB/s).
- **Hipótese**: se a LUT global for parte material do custo, o `iq3_s` sai de 443 GB/s para
  perto do teto (7,74 → 6,5 ms/token = +3,4 % de decode). Se a LUT for barata (L1 hit), o
  ganho é ~0 e a LDS não vale 2 KB de orçamento + barreira.
- **Comando**: ver §1.1 abaixo (diagnóstico já existente: `--bench-read` e `--bench-ab`).
- **Resultado**: ver §1.1
- **Veredito**: ver §1.1

### 1.1 (implementação) o que foi construído para medir isso

- `include/rdna4/vecdotq.cuh`: os 5 `vec_dot` que usam LUT e que embarcam
  (`iq3_s` perm, `iq3_xxs` perm2, `iq2_xxs` perm2, `iq2_xs` perm2, `iq2_s` perm2) ganharam
  uma variante `*_lut` com um 5º parâmetro (o ponteiro da tabela). O corpo é o mesmo; a
  entrada de produção virou um forwarder que passa a tabela global — ou seja, **o
  caminho que embarca não mudou de código gerado**, só de nome.
- `include/rdna4/matvec.cuh`: `matvec_kernel_gen` ganhou o parâmetro `class LUT = NoLut`;
  com `LUT::words > 0` o CTA copia a tabela para `__shared__` no início (cópia
  cooperativa, uma `__syncthreads()`, sem retorno divergente antes dela) e o `vec_dot`
  lê da LDS. `matvec_launch_lut_lds(dt, ...)` usa os tipos `LutIq*` para
  `iq3_s/iq3_xxs/iq2_s/iq2_xs/iq2_xxs` (60 % dos bytes do matvec) e cai no caminho de
  produção para os demais.
- Gate: `./build/check-matvec-gpu <model> --check-lds` (novo) compara os dois caminhos
  com **memcmp** linha a linha, tipo a tipo, no tensor real do inventário — a tolerância
  do oráculo de CPU (1e-5..1e-4 para os tipos IQ) aceitaria uma tabela errada.
- Bench: `bench-matvec-shapes-gpu ... --lut-lds` (nova variante `kind=8`) e, para o
  item 2, as colunas novas `ms-fix` / `ms(tok)-fix`, que descontam **um piso de sync por
  tensor** (`null_once`): o número por tensor deste bench carrega o `hipEventRecord` +
  `hipDeviceSynchronize` de cada lançamento, o que num tensor de 261 KB é a maior parte
  do valor — sem essa correção a classe `q8_0` 48×160 parece custar 1,00 ms/token.

### 1.2 RESULTADO — LUT em LDS: **MANTIDO com guarda de razão** (bit-exato)

- **Comando** (uma sessão de lock, janela limpa — `mem_info_vram_used` 198-271 MB antes
  de cada passo, nenhum `flock` esperando de outra frente no momento da medida):
  `timeout 900 ./build/bench-matvec-shapes-gpu $MODEL --reps 2 --repeat-ship --rows 1 --lut-lds`
  e `timeout 300 ./build/check-matvec-gpu $MODEL --check-lds`, dentro de
  `./scripts/gpu-lock.sh bash -c '...'`.
- **Piso de ruído do harness** (mesmo kernel medido duas vezes, inventário inteiro):
  **ship 24,636 ms vs ship2 24,666 ms = 0,999x**; por tipo, ≤1,01x. `bench-delta-gpu`
  (outro harness) deu 1,0008x.
- **Resultado (ms/token por tipo, inventário real de 497 tensores / 11,12 GB)**:

  | tipo | bytes(tok) | LUT | peso/CTA | ship | lut-lds | razão |
  |---|---|---|---|---|---|---|
  | **iq3_s** | 3,717 GB (33,4 %) | 2 KB | 17,6 KB | 7,762 | **7,251** | **1,070x** |
  | **iq3_xxs** | 1,863 GB (16,7 %) | 1 KB | 3,9 KB | 4,468 | **3,781** | **1,182x** |
  | iq2_xxs | 0,269 GB | 2 KB | 1,3 KB | 0,648 | 0,708 | 0,914x |
  | iq2_xs | 0,292 GB | 4 KB | 1,5 KB | 0,669 | 0,984 | 0,680x |
  | iq2_s | 0,581 GB | 8 KB | 3,3 KB | 1,140 | 1,897 | 0,601x |
  | **total do matvec** | 11,12 GB | — | — | 24,636 | 24,579 | 1,002x |

  Leitura: o agregado (1,002x) **esconde** dois ganhos grandes e três perdas grandes. A
  separação é exatamente o **tamanho da LUT contra o peso que cada CTA lê**: quem ganha
  tem razão 3,9x/8,8x; quem perde, 0,37x/0,41x/0,66x (copiar 8 KB de LUT por CTA para um
  tensor cujo CTA lê 3,3 KB de peso é caro, e a `iq2_*` ainda lê a LUT como `uint2`, com
  conflito de banco na LDS).
- **Regra embarcada** (`matvec_lut_lds_ok`): usa LDS só quando
  `rows * bpr * block_bytes >= 2 * lut_bytes`. Ela exclui precisamente os três perdedores
  e inclui os dois ganhadores; **ganho que entra = 0,511 + 0,687 = 1,199 ms/token =
  +3,3 % no token de 36,0 ms** (só no matvec; o resto do token não muda).
- **Bit-exatidão (gate novo)**: `check-matvec-gpu --check-lds` → **5 tipos BIT-IDENTICO,
  max|d| 0,000e+00** (memcmp de 256 linhas por tipo, no tensor real do inventário). Não é
  tolerância: é `memcmp`. Importante porque a tolerância do oráculo de CPU (1e-5..1e-4
  nos tipos IQ) aceitaria uma tabela errada.
- **Por que ganha (ISA, `--save-temps` no `iq3_s` UNROLL=2)**: `global_load_b32` 30 → 7,
  `ds_load_b32` 0 → 24, e o que importa: **`s_wait_loadcnt` 35 → 14** com o total de
  instruções **praticamente igual (620 → 619)**. Ou seja: não é redução de contagem de
  instrução, é redução de **espera de load** — o que também é a evidência de que o `iq3_s`
  é limitado por latência, não por issue (o que contraria a leitura "issue-bound" do
  `docs/medicoes-banda-e-gargalos.md` §2.2: medido, o gargalo é a espera do gather da LUT).
- **Veredito**: **MANTIDO** (guarda por tipo/forma, bit-exato, +3,3 % do token estimado a
  partir do matvec; o número fim a fim está na §4).

## 2. Tabela de shape por (tipo, linhas): `rows=1` nas formas de grid pequeno

- **Referência**: `docs/autotuning-gfx1201.md` §4.1/§5.1 (a classe `q8_0` 48×160 roda a
  25 GB/s e custaria 1,00 ms/token = 4 % do matvec; sugestão: `rows=1` para fazer 48 CTAs
  em vez de 24).
- **Hipótese**: `ROWS` é bit-exato (só muda qual CTA calcula qual linha), então `rows=1`
  nas formas com poucas linhas deveria recuperar parte dos 1,00 ms/token.
- **Comando**: o mesmo da §1.2 (`--rows 1` na mesma corrida; piso de ruído 0,999x).
- **Resultado**: agregado do matvec **25,31 ms com `rows=1` contra 24,64 ms do ship =
  0,973x (PIOR)**. Por tipo: `q8_0` **0,998x** (a classe alvo não se move: 0,994 → 0,997),
  `q6_k` 0,558x, `iq3_s` 0,941x, `iq3_xxs` 0,977x, `q3_k` 0,987x, `q5_k` 0,990x,
  `iq2_s` 0,980x, `iq4_nl` 0,898x; o resto dentro de ±0,5 %.
- **Por que a hipótese errou (mecanismo)**: com `WPR=1` cada linha é **um warp**, então o
  número de warps é `nrows` **independentemente de `ROWS`** — o que `ROWS` muda é só como
  esses warps são agrupados em CTAs. Na classe 48×160 são 48 warps nos dois casos (24 CTAs
  de 2 warps ou 48 de 1); não há trabalho novo para distribuir. O "1,00 ms/token" da classe
  também tem um problema de método: o número por tensor do
  `bench-matvec-shapes-gpu` carrega `hipEventRecord`+`hipDeviceSynchronize` por
  lançamento, e num tensor de 261 KB isso é a maior parte do valor (as colunas
  `ms-fix`/`ms(tok)-fix` foram adicionadas para isso).
- **Veredito**: **REVERTIDO/ABANDONADO** — nenhuma linha foi mudada em `tuning.h`
  (`kMtRows` intacto), `check-tuning` continua verde. A alavanca real para essa classe é
  *fundir os 96 lançamentos* (uma grade que atende vários tensores), o que é do
  `graph.cuh`/prefill, não do kernel.

## 3. `delta_rule`: 8,515x com carga `float4` da linha do estado (**bit-exato**)

- **Referência**: `docs/medicoes-banda-e-gargalos.md` §4.2 (69,6 µs por lançamento contra
  um piso de memória de ~20 µs; 48 CTAs de 128 threads em 64 CU = 9 % de ocupação; 4
  passadas sobre 3,15 MB de estado; 152 GB/s).
- **Hipótese**: é latência, não banda nem ocupação; a linha de 512 B do estado é lida com
  128 cargas **escalares de 4 B por thread**, cada lane num setor de 32 B próprio (32
  setores por instrução). Carregar a linha com `float4` (32 cargas de 16 B) deve reduzir a
  espera sem mexer em nenhuma conta.
- **Comando**: `timeout 900 ./build/bench-delta-gpu --rounds 12` (cadeia de 100 lançamentos,
  mínimo de 12 rodadas × 3 passadas rotacionadas, aquecimento ≥320 ms por variante,
  formas reais nvh 48 / nkh 16 / S 128). Janela limpa (VRAM 198 MB antes).
- **Piso de ruído do harness**: o kernel que embarca medido duas vezes na mesma corrida:
  **71,70 µs vs 71,64 µs = 1,0008x**.
- **Resultado (µs por camada recorrente; ×48 = ms/token)**:

  | variante | µs/lançamento | razão | GB/s de estado |
  |---|---|---|---|
  | histórico T128 RPT1 **escalar** | 71,70 | 1,000x | 176 |
  | histórico medido 2ª vez (piso) | 71,64 | 1,001x | 176 |
  | **T128 RPT1 float4 (o que embarca)** | **8,42** | **8,515x** | **1 496** |
  | T64 RPT2 float4 | 21,06 | 3,404x | 598 |
  | T32 RPT4 float4 | 36,21 | 1,980x | 348 |
  | T64 RPT2 escalar | 138,38 | 0,518x | 91 |
  | T32 RPT4 escalar | 190,03 | 0,377x | 66 |
  | piso do padrão (lê+escreve, sem cadeia) | 5,72 | 12,53x | 2 202 |

- **Bit-exatidão**: todas as variantes comparadas com **memcmp** contra o kernel que
  embarcava — 786 432 floats de estado + 6 144 de saída, **todas IDÊNTICAS**. Nenhuma
  mudança de arredondamento: as mesmas operações, na mesma ordem, com os mesmos
  acumuladores.
- **Leitura honesta de duas ressalvas**: (a) a cadeia de 100 reusa o MESMO buffer de
  estado (3,15 MB, cabe no Infinity Cache de 64 MB), então os 1 496 GB/s são de cache;
  no grafo são 48 estados diferentes (151 MB, não cabe) e o piso de DRAM é
  12,6 MB / 633 GB/s = **20 µs por camada = 0,96 ms/token** — o ganho fim a fim tem que ser
  medido, não extrapolado (é o que a §4 faz); (b) `RPT>1` mediu PIOR, contra a intuição do
  briefing: com `RPT>1` há menos warps no ar (192 → 96 → 48) e o `float4` já entrega o
  paralelismo de memória que faltava. O que embarca é a mudança mínima: **largura da carga**.
- **Veredito**: **MANTIDO** (`delta_rule_kernel_ilp<128,1,true>` no `gdn.cuh`, com o kernel
  histórico preservado como `delta_rule_launch_scalar` para o bench; `S != 128` cai no
  caminho antigo). Mudança **aditiva e local** no `gdn.cuh` (a frente de prefill também
  mexe nesse arquivo para o scan em batch: os dois blocos convivem).

## 4. Normas do decode (`rms_norm`/`l2_norm`): 4 cargas em voo (**bit-exato**)

- **Referência**: `docs/medicoes-banda-e-gargalos.md` §4.3 (medido: `rms_norm(1x5120)` custa
  **10,90 µs** por lançamento contra um piso de despacho de 2,20 µs, e roda 128× por token:
  `attn_norm` ×64 + `post_norm` ×64) + `nn.cuh:30-45` (1 CTA de 256 threads para 5120
  elementos = 20 cargas por thread numa cadeia de `fma`).
- **Hipótese**: das 8,7 µs acima do piso, a maior parte é **latência de load não escondida**
  (a cadeia de `fma` é serial no acumulador e o laço tem limite de iteração em runtime, o
  que impede o compilador de puxar as cargas). Carregar 4 elementos em registradores antes
  de aplicar os 4 `fma` mantém a ordem exata e põe 4 cargas em voo → esperado ~2-3 µs a
  menos por lançamento.
- **Comando**: `timeout 900 ./build/bench-norm-gpu --rounds 12` (novo alvo `bench-norm-gpu`;
  cadeia de 100 lançamentos, aquecimento ≥320 ms, mínimo de 12 rodadas × 3 passadas
  rotacionadas, formas reais do motor). Janela: dentro do lock; VRAM 277 MB antes.
- **Piso de ruído**: a variante nova medida duas vezes deu 8,94 vs 8,92 µs (**1,002x**);
  o antigo e o novo são medidos no mesmo harness, na mesma corrida.
- **Resultado (µs/lançamento)**:

  | caso | antigo | novo | razão | ms/token (×lançamentos/token) |
  |---|---|---|---|---|
  | `rms_norm(1 x 5120)` com `w` | 10,97 | **8,94** | **1,227x** | 1,404 → 1,144 (**−0,26**) |
  | `rms_norm(48 x 128)` | 3,17 | 3,19 | 0,994x | 0,152 → 0,153 (neutro) |
  | `l2_norm(16 x 128)` | 3,16 | 3,16 | 0,998x | 0,303 → 0,304 (neutro) |
  | `rms_norm(8 x 5120)` (extrapolação) | 10,69 | **8,80** | **1,215x** | — |

- **Bit-exatidão**: **memcmp da saída inteira nos 4 casos → BIT-IDENTICO**. Mesmos `fma`,
  mesma ordem, mesma árvore de redução de 256 parciais; só as cargas foram antecipadas.
- **Veredito**: **MANTIDO** (−0,26 ms/token = 0,7 % do token). Ficou claro com a medida que
  a hipótese de "8,7 µs de latência" era parcialmente certa: 2 µs eram latência de load, e
  os ~6,7 µs restantes acima do piso são outra coisa (1 CTA só, com 8 `__syncthreads` na
  árvore de redução) — atacável, mas com retorno menor, e ficou fora do orçamento da noite.

## 5. O que foi testado e **não** entrou (com o número que matou)

1. **`rows=1` / tabela de shape por (tipo, linhas)** — 0,973x no agregado (pior), a classe
   alvo `q8_0` 48×160 não se moveu (0,998x). Detalhe e mecanismo na §2. `tuning.h` intocado.
2. **`lut-lds` para `iq2_xxs`/`iq2_xs`/`iq2_s`** — 0,914x / 0,680x / 0,601x: a LUT (2/4/8 KB)
   é maior que o peso que o CTA lê (1,3/1,5/3,3 KB), então a cópia por CTA custa mais que o
   ganho. É o que a guarda `matvec_lut_lds_ok` (razão ≥ 2x) exclui.
3. **`UNROLL=4` com a LUT em LDS** — medido na §1.3 (a tabela já rejeitava `unroll=4` com a
   LUT global em 0,917x).
4. **Fusão dos kernels pequenos (item 3 do briefing, ~1,1 ms/token)** — **não feito, e o
   motivo é medido no código, não no desempenho**: o `emit()` do grafo faz um
   `hipMemcpy` **bloqueante** no momento da chamada (`tests/check_graph_gpu.hip:119-121`),
   então fundir 4 ops em 1 lançamento faz o nó intermediário (`a_softplus`, `beta_sigmoid`)
   ser lido DEPOIS de o buffer já ter sido sobrescrito → o oráculo por nó quebra. A variante
   que preserva o dump é 4 → 2 lançamentos (junta `sigmoid`+`add`+`softplus`, deixa o `mul`
   separado porque o `emit` do `a_softplus` acontece entre os dois): só 0,26 ms/token, e
   exigiria mexer no `graph.cuh` (arquivo da frente de prefill) por ~0,7 % — não paga o
   risco de merge. `rms_norm`+`quantize` tem o mesmo problema agravado: o `q8_1` é
   compartilhado por várias projeções (`proj_qq`), então a fusão mexeria no `proj()`.
   Registrado aqui para o coordenador decidir com o número na mão.

## 6. Estado do gate (todos com `scripts/gpu-lock.sh`, janela verificada)

**Precisão sobre qual binário:** a bateria abaixo rodou no binário com a LDS (§1) e a
`delta_rule` float4 (§3) já embarcadas, e **antes** da mudança de `nn.cuh` (§4). Depois
dela, o mesmo conjunto foi re-enfileirado no binário final (`/tmp/kernels-final.txt`:
gates + `check-matmul-gpu` base vs novo + A/B fim a fim) e **não tinha rodado** quando esta
sessão terminou de escrever — a fila da GPU passou ~45 min com 14-16 processos e a placa
monopolizada por outras frentes. O `bench-norm-gpu` dá a prova de bit-exatidão da §4
(memcmp nos 4 casos), então o que falta é a confirmação no grafo, não a correção.

| gate | resultado |
|---|---|
| `check-tuning` (CPU) | OK — 22 linhas |
| `check-matvec-gpu --check-lds` (novo) | **OK — 5 tipos BIT-IDENTICO** (memcmp), com a regra de produção usando LDS exatamente em `iq3_s`/`iq3_xxs` |
| `check-matvec-gpu` (14 tipos vs oráculo CPU) | OK — 14 testados (iq3_s rel-L2 6,5e-08 na tol 4,5e-06) |
| `check-matmul-gpu` | **OK — bit-exact em todas as configurações** |
| `check-batch-gpu` | **OK — prefill em batch bit-idêntico** (rel-L2 0,00e+00) |
| `check-nn-gpu` | OK |
| `check-graph-gpu` (`GRAPH_LAST_TOKEN=1`, oráculo por nó) | **PASS** |
| `scripts/check_regression.sh` | **OK — 7 casos, 7 ids bit-exatos, rel-L2 0,00e+00** (parede 58 s) |
| `scripts/check_golden_run.sh` | **OK** |

### 1.3 `UNROLL` depois da LDS (PENDENTE de medida)

Pergunta: a tabela de autotuning rejeitou `UNROLL=4` (0,917x) com a LUT **global**
(`docs/autotuning-gfx1201.md` §4.3). Com o gather fora do caminho global
(§1.2), o ótimo de `UNROLL` pode mudar — `UNROLL` é a alavanca bit-exata
documentada (mesmas operações, mesma ordem, mais cargas em voo), então se ela
virar, o ganho é seguro.

- **Comando**: `timeout 900 ./build/bench-matvec-shapes-gpu $MODEL --reps 3
  --repeat-ship --lut-lds-unroll 1,4` (mesmo harness, piso `ship` vs `ship2`).
- **Já medido**: `UNROLL=1` com LDS fica **pior** que o `UNROLL=2` que embarca
  (iq3_s 7,462 vs 7,243 ms = 0,971x; iq3_xxs 4,014 vs 3,774 = 0,940x), na mesma
  corrida em que `ship` vs `ship2` deu 1,000x. Isso já diz que a LDS não tornou o
  unroll baixo melhor.
- **Resultado**: a corrida com `1,4` foi enfileirada
  (`/tmp/kernels-ldsunroll.txt`) e **não tinha rodado até o fim desta sessão** por
  contenção da fila da GPU (16 processos no `flock`, a placa ocupada por outra
  frente com corridas de 64K). Fica registrado como **aberto, com o comando exato**
  — não invento o número.

## 7. Resumo dos ganhos medidos (ms/token de um token de 36,0 ms)

| mudança | arquivo | ganho medido no kernel | bit-exato? | veredito |
|---|---|---|---|---|
| LUT dos IQ em LDS, com guarda de razão (iq3_s, iq3_xxs) | `matvec.cuh`, `vecdotq.cuh` | **−1,21 ms** (1,070x + 1,182x) | **sim** (memcmp) | MANTIDO |
| `UNROLL=4` com a LUT em LDS | `matvec.cuh`, `tuning.h` | **aberto** (ver §1.3: corrida enfileirada, não executada) | sim (por construção) | PENDENTE |
| `delta_rule` com carga `float4` da linha do estado | `gdn.cuh` | **−3,04 ms** (8,515x isolado, 5,72 µs de piso de padrão) | **sim** (memcmp) | MANTIDO |
| `rms_norm`/`l2_norm` com 4 cargas em voo | `nn.cuh` | **−0,26 ms** (1,227x no 1x5120) | **sim** (memcmp) | MANTIDO |
| `rows=1` / tabela de shape | — | 0,973x (pior) | sim | REVERTIDO |
| LUT em LDS para iq2_* | — | 0,60-0,91x (pior) | sim | REVERTIDO (guarda exclui) |
| fusão dos kernels pequenos | `graph.cuh` | ~0,26-1,1 ms estimados | sim | NÃO FEITO (quebra o dump por nó; ver §5.4) |

## 8. Amplificação em lote (hoist da dequantização em `matvec_kernel_batch`)

- **Referência / pedido**: coordenador (mensagem de 03:38), com a medida da frente de
  prefill: o matvec em lote lê 12,0 GB por chunk de 16 tokens em **110 ms = 109 GB/s**,
  contra 446 GB/s do mesmo matvec no caminho por token; depois de batelarem o andaime, o
  matvec virou 85 % do prefill (6,93 de 7,98 ms/token a N=16).
- **Hipótese**: `T::dot(rowp, abase + n*act_stride, k, kqs)` é chamado N vezes e refaz, por
  token, o gather da LUT, a montagem da máscara de sinal e o `V_PERM` — tudo independente do
  token. Dequantizar UMA vez por bloco (16 int32 em registrador) e fazer N `dp4a` deve
  reduzir o trabalho de ALU por byte.
- **O que foi feito** (nesta ordem, para não misturar com a LDS): `vecdotq.cuh` ganhou
  `vec_prep_iq3_s_q8_1_perm`/`vec_dot_prep_iq3_s_q8_1_perm` e os equivalentes de
  `iq3_xxs` (os dois tipos com LUT + `V_PERM`, 50 % dos bytes); `matvec.cuh` ganhou o par
  `T::prep`/`T::dot_prep` com **fallback identidade** (`NoPrep` → chama exatamente o `dot`
  de antes), então os 12 outros tipos têm, por construção, o mesmo código de antes; só
  `iq3_s` e `iq3_xxs` passam pelos traits `*_PREP` no `RD_BATCH`.
- **Custo esperado, honestamente**: contando as instruções de um bloco `iq3_s` (110 B) no
  laço em lote com N=16: prep ≈ 40 ops uma vez, e por token ≈ 8 cargas de ativação +
  8 `dp4a` + ~9 ops de epílogo. O prep é ~10 % do total, então o teto do ganho por essa
  via é ~10 % **nesse tipo** — não 2-3x. O coordenador estimou mais; a conta acima é o
  motivo de eu não afirmar o número dele.
- **Comando (enfileirado)**: `kernels-prep-measure.sh` (base pristino vs novo:
  `check-matmul-gpu` dos dois binários + `check-batch-gpu` + `bench --prefill 512`
  intercalado 3×), e os gates que o coordenador pediu
  (`check-matmul-gpu`, `check-batch-gpu`) na bateria de gates final.
- **Resultado**: **a fila da GPU não liberou até o fim desta sessão** (16 processos no
  `flock`, 30+ min com a placa monopolizada por outra frente). Sem número medido, não entro
  com a mudança no caminho que embarca: a rota fica atrás de duas linhas
  (`RD_BATCH(TIQ3S_PREP, ...)` / `RD_BATCH(TIQ3XXS_PREP, ...)` no `matvec.cuh`), o código
  fica no arquivo com o fallback identidade, e o comando exato está acima. É uma decisão
  explícita: **aberto, com o desenho pronto e o comando escrito**, para o coordenador ou
  para quem pegar a placa depois.

## 9. Decisões do coordenador (04:25) e estado final da frente

1. **A/B fim a fim**: o coordenador assumiu a corrida (mesma janela, base alternando com o
   novo, 3 rodadas) e passa os números. Os meus dois jobs que esperavam o lock a noite
   inteira foram perdidos (um deles por um `kill` meu com padrão que casou com o próprio
   shell — erro meu, registrado aqui); re-enfileirei **um** job compacto com a validação
   final do binário commitado (`/tmp/kfin.txt`).
2. **Hoist da dequantização em lote: DESLIGADO**, e o argumento que decide é a contagem de
   instruções (§8): ~40 ops de preparo por bloco de 110 B contra ~25 ops por token (N=16)
   ⇒ teto ~10 %, contra a estimativa de 2-2,5x da frente de prefill. O coordenador
   registrou no diário da noite (C8) como **gargalo real do matvec em lote não
   identificado, com duas hipóteses medidas e nenhuma confirmada**. As duas linhas para
   ligar e o comando de medida estão no `matvec.cuh` e no §8.
3. **Fusão dos kernels pequenos: NÃO fazer** — o motivo (o `emit()` bloqueante do oráculo
   por nó, §5.4) foi aceito: o oráculo é o gate mais forte do repo e a variante que
   preserva o dump vale só 0,26 ms/token.
4. **O que está ligado**: os três ganhos bit-exatos (§1.2 LUT em LDS, §3 `delta_rule`
   float4, §4 normas com 4 cargas em voo). **Desligado e pronto para ligar**: o hoist em
   lote. **Abandonados com a medida que os matou**: `rows=1` (0,973x), LUT nos `iq2_*`
   (0,601-0,914x), `RPT>1` na `delta_rule` (0,377-0,518x), fusão dos pequenos (oráculo).
   **Aberto sem medida**: `UNROLL=4` com a LUT em LDS (§1.3).

## 10. Onde estão os "3,6x de instruções por byte" do matvec em lote? (leitura de ISA, sem GPU)

Pedido do coordenador (C8). Método: `--save-temps` com duas instanciações explícitas no
mesmo arquivo — `matvec_kernel_gen<TIQ3S_S, ROWS=8, WPR=1, ILP=1, UNROLL=2>` (o caminho
por token que embarca) e `matvec_kernel_batch<TIQ3S_S, 8, 1, 1, N=16>` — e contagem de
opcodes por categoria (`grep` no `.s`, por função).

| | GEMV (1 token, por bloco por thread) | LOTE N=16 (por bloco por thread) | razão |
|---|---|---|---|
| `v_dot4_i32_iu8` | 16 | 256 | **16,0** (escala com N) |
| cargas de ativação | 3 | 48 | **16,0** (escala com N) |
| cargas de peso + LUT | 13 | **13** | **1,0 (amortizado)** |
| `v_perm_b32` | 8 | **8** | **1,0 (amortizado)** |
| demais VALU | ~83 | ~241 | 2,9 |
| `s_wait_*`/`s_delay_alu` | ~38 | ~282 | 7,5 |

Leitura (e é ela que responde o C8): **não há dequantização reexecutada N vezes.** O
compilador já CSE-ou o `T::dot` inteiro do lado do peso: o gather da LUT, a montagem da
máscara de sinal e o `V_PERM` aparecem **uma vez por bloco** no kernel em lote (8 perms e
8 cargas de LUT por bloco, não 8×16). Consequências:

1. O **hoist que eu implementei (§8) tem teto ~0**, não os ~10 % que a contagem de
   instruções sugeria: o que eu ia hoistar já está hoisted. Ele fica DESLIGADO e o
   comentário no `matvec.cuh` agora diz isso — melhor do que "desligado por falta de
   medida": está desligado porque a ISA mostra que não vale.
2. O que **não** pode ser amortizado é exatamente o que escala 16×: 16 `dp4a` e 3 cargas
   de ativação por token por bloco (2 delas `global_load_b128`, resultado do merge de 4
   `int32`, e 1 `ds`/`u16` do `ds` do `q8_1`). Isso é o mesmo trabalho por token que o
   caminho de decodificação já faz — e é por isso que o lote ganha 3,6x por token (só o
   lado do peso é amortizado) e ao mesmo tempo não chega perto do roofline: 6,875 ms/token
   no lote contra 24,94 ms/token no decode, com ~0,7 GB de tráfego DRAM por token
   (≈101 GB/s de um teto de 633).
3. Logo, **o matvec em lote é limitado por latência, não por banda nem por instrução
   repetida**: o IPC implícito é ~0,15 do pico de issue (25,3e6 iterações de warp × ~428
   instr = 10,8e9 warp-instr por chunk de 11,12 GB contra 110 ms e ~640e9 warp-instr/s).
   As cargas de ativação vêm de um working set de 320 KB (16 linhas × 20 KB, em L2) e cada
   linha de peso do tensor re-lê a ativação inteira — a hipótese a medir na próxima sessão
   é o **tráfego de re-leitura da ativação por linha** (para um tensor 17408×5120:
   17408 × 16 × 20 KB ≈ 5,6 GB por tensor por chunk, saindo de L2/Infinity Cache), e as
   duas alavancas candidatas são mais blocos independentes por thread (mais cargas em voo)
   e staging de ativação por CTA em LDS.
