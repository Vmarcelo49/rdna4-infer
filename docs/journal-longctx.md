# Diário da frente longctx — 2026-09-14 (não supervisionada)

Frente: **"o modelo ainda se sustenta a 131K, e o nosso caminho de contexto longo é
fiel à referência?"**
Worktree `rdna4-wt-noite-longctx`, branch `feat/noite-longctx`, base `e48f3c4`
(`noite-baseline-2026-09-14` + `ea0e670` regras).
Formato: `docs/noite-regras.md` §5. Todo número diz o comando exato e a janela.

## TL;DR (para quem só vai ler isto)

1. **RoPE em posição longa: fórmula certa, NENHUM scaling faltando. Sem P0.** Os dois GGUF não
   têm `rope.scaling.*`; o llama.cpp resolve `freq_scale=1,0`/`ext_factor=0,0` (nenhum
   YaRN/NTK/linear) e, para texto, o IMROPE do qwen35 **é** um RoPE split-half comum sobre
   `n_rot=64` (medido: `rel-L2 = 0,000e+00` contra o NEOX-64 do ggml, até 262 143). O nosso
   kernel bate com o `ggml_rope_multi` real a `3,6e-4` em 131 071 = **piso do fp32** (1 ulp de
   fase). Um YaRN×4 configurado daria 0,24-0,42: 3 ordens de grandeza acima. Gate novo:
   `check-rope-long-gpu` (§1, §2).
2. **Fim-a-fim em posição longa (17 639 tokens, prefill em lote nos dois lados): os 5 primeiros
   ids são os mesmos, na mesma ordem, argmax 16 nos dois**; |Δlogit| de 0,085-0,118, do qual
   −0,0995 é deslocamento uniforme (invariante no softmax) ⇒ espalhamento real de ±0,018 contra
   gaps de 0,16-0,64 (§6.2).
3. **Qualidade vs contexto (PPL por posição, mesma janela nos dois motores): o desvio do motor
   NÃO cresce com o contexto** — 0,185 % (512) → 0,163 % (2 048) → 0,108 % (14 336). A 14K o
   dNLL médio é **plano** por faixa de posição (−0,0017 / −0,0008 / −0,0010 / −0,0008) (§6.1).
4. **Curva da atenção (×16 camadas)**: 1,16 ms/token a 4K → 3,89 (16K) → 15,5 (64K f16) / 18,4
   (q8_0) / 19,9 (q4_0) → 30,8 (131K f16) / 36,2 (q8_0) / 38,1 (q4_0). A atenção sai de ~3 % do
   passo a 4K para ~53 % a 131K (§3.1, §3.3). **`q8_0` é mais rápido que `q4_0`** a 64K/131K
   apesar de mover 1,9× mais bytes (o `q4_0` paga desempacotamento de nibble).
5. **Política de splits**: a que embarca está a ≤5 % do ótimo a 4K/16K (piso de ruído 0,995-1,000)
   e **não é o ótimo a 131K q4_0**: 48×16 é 1,064× melhor (2,33 ms/token = 3,3 % do passo) —
   confirma a frente de autotuning de forma independente (§3.5).
6. **GQA 6:1**: com **f16** (4K-16K, onde o motor vive) compartilhar K/V **perde 1,1-1,8×** —
   a releitura 6× é servida pelo cache a 1,67 TB/s medidos e o gargalo é paralelidade (CTAs).
   Com **q4_0/q8_0 a 64K/131K** o protótipo `HG=3` (2 passadas, mesma grade, 6× menos CTAs) é
   **9-12 % mais rápido** que o kernel que embarca (2,1-3,3 ms/token ≈ 4 % do passo), com
   equivalência numérica medida. Recomendação: **não implementar para 4K-16K; levar adiante só
   para KV quantizado em contexto longo** (§4.2-4.3).
7. **Decode a 64K/131K com cache sintético** (o que o baseline reporta): 19,24 tok/s (64K q8_0,
   13,75 GiB), 19,92 (64K f16, 15,62 GiB), **14,09 (131K q4_0, 13,87 GiB)**; **zero derrame para
   GTT** nas três (§3.4).
8. **131K de verdade não foi medido**: o prefill em lote medido é 64,5 tok/s a 17,6K ⇒ ~33 min
   só para encher o KV de 131K, fora do `timeout 900`. Os números de 128K do README são cache
   sintético (§5.2, §5.4). **É o buraco a fechar na manhã.**

---

Ferramenta oficial de perfil (`rocprof`/`omniperf`) **não existe nesta máquina**
(`noite-regras.md` §3.5): tudo abaixo é evento HIP, `bench --layers`, A/B e subtração.

---


## 1. RoPE em posição longa: o que a referência realmente faz (leitura + medição)

- **Referência**: `/home/marcelo/Projetos/llama.cpp/src/models/qwen35.cpp:587-592`
  (`ggml_rope_multi`), `src/llama-model.cpp:1322-1334` (leitura das chaves de scaling),
  `src/llama-context.cpp:110-116,168-211` (resolução em runtime),
  `src/llama-batch.cpp:780-788` (posições por seção), `ggml/src/ggml-cpu/ops.cpp:5994-6050`
  (`ggml_mrope_cache_init`), `docs/qwen-kernels.md` §1.
- **Hipótese (P0 em potencial)**: se a referência aplicasse YaRN/NTK/linear a partir de
  algum ponto de contexto e nós não, todo o contexto longo seria lixo silencioso — e o
  sintoma seria um delta de PPL **crescendo com a posição**.
- **Comando (leitura, sem GPU)**: dump do cabeçalho GGUF dos dois modelos IQ3_S/IQ4_XS;
  `grep` nas chaves `rope.scaling.*` do llama.cpp.
- **Resultado (estático)**:
  1. Os dois GGUFs têm **50 chaves** e **nenhuma** `rope.scaling.*`
     (`rope.scaling.type`, `factor`, `original_context_length`, `yarn_log_multiplier`,
     `yarn_ext_factor`, `attn_factor`, `beta_fast/slow`, `rope.scale.linear` ausentes).
     Só existem `qwen35.rope.dimension_sections=[11,11,10,0]`,
     `qwen35.rope.dimension_count=64`, `qwen35.rope.freq_base=1e7`,
     `qwen35.context_length=262144`.
  2. Sem a chave, `llama-model.cpp:1322` usa o **default `"linear"`** → tipo de scaling
     `LINEAR` (não `NONE`), e `factor` ausente ⇒ `rope_freq_scale_train = 1.0`
     (`llama-model.cpp:1330-1334`).
  3. `llama-context.cpp:172-174`: `yarn_ext_factor` fica `-1` (default de
     `llama-hparams.h:159`) e vira `0.0` porque o tipo não é `YARN`;
     `yarn_attn_factor = 1.0 * rope_attn_factor(1.0)`.
  4. `mode` = `LLAMA_ROPE_TYPE_IMROPE` (`llama-model.cpp:3041`, `LLM_ARCH_QWEN35`).
  5. Texto puro: `llama-batch.cpp:780-788` **replica a mesma posição nas 4 seções**
     (`src_off = 0` quando `batch.token`), ou seja `p_t = p_h = p_w = p_e`; em
     `ggml_mrope_cache_init` isso faz `theta_h = theta_w = theta_t` para todo par, e o
     IMROPE interleaved **degenera no RoPE split-half comum sobre os 64 dims**.
- **Ressalva (medida, não suposta)**: o llama.cpp *pode* aplicar scaling se o usuário passar
  `--rope-scaling-type`/`--rope-scale`/`--yarn-*` na linha de comando. Nenhuma ferramenta que
  este repo usa passa (`capture_oracle.sh`, `compare_ppl.sh`, `oracle-next-token`,
  `compare_llama_greedy.sh` usam `llama_context_default_params()` e só mexem em
  `n_gpu_layers`), então a referência dos nossos gates é a **sem escala** — e é a que
  reproduz o modelo.
- **Veredito**: **nenhum scaling é esperado** para este modelo em nenhuma posição
  (freq_scale=1, ext_factor=0, attn_factor=1). O que resta verificar por medição é a
  *fórmula* (n_rot=64, split-half, todos os 32 pares) e a *precisão* em posição longa —
  feito no experimento 2.

---

## 2. `check-rope-long-gpu`: o nosso kernel de RoPE contra o `ggml_rope_multi` real

- **Referência**: `ggml/src/ggml-cpu/ops.cpp` (`ggml_mrope_cache_init`, `rotate_pairs`),
  `ggml/src/ggml.c:4336` (`ggml_rope_multi`); o gate existente
  `tests/check_rope_gpu.hip:65-71` só testa posições `3..34` **contra uma referência
  escrita à mão** (não contra o ggml).
- **Hipótese**: a fórmula do motor (`include/rdna4/attn.cuh:30-54`) é idêntica ao
  `ggml_rope_multi(IMROPE, sections=[11,11,10,0], freq_base=1e7, freq_scale=1.0,
  ext_factor=0.0)` com as 4 seções iguais, em posições até 262 143 — e a diferença
  cresce só pela precisão do `float` (ordem de operações: `powf` por par no nosso kernel
  vs. multiplicação iterativa por `theta_scale` no ggml).
- **Ferramenta nova**: `tests/check-rope-long-gpu.hip` (alvo `check-rope-long-gpu`). Ela
  linka o **ggml real** do llama.cpp (`libggml.so` + `libggml-cpu.so`) e chama
  `ggml_rope_multi`/`ggml_rope_ext` no backend CPU — não uma referência reescrita à mão,
  que é o furo do gate antigo. Também calcula a rotação exata em `double` como terceira
  coluna, e duas variantes com scaling (YaRN×4 e linear×4) para dimensionar o que uma
  escala ausente custaria.
- **Comando**: `timeout 900 ./scripts/gpu-lock.sh ./build/check-rope-long-gpu`
  (VRAM antes 1,157 GiB / depois 0,198 GiB; GTT 36,6 MB → 30,3 MB; janela limpa — só a
  minha frente, nenhum processo de modelo vivo).
- **Resultado** (posições 0 … 262 143, heads 24, head_dim 256, n_rot 64, entrada aleatória
  semeada; `rel-L2` = norma L2 da diferença / norma L2 da referência):

| pos | max\|Δ\| ours-ref | relL2 (ours,ref) | relL2 (ref,exato) | relL2 (ours,exato) | relL2 (NEOX ctrl,ref) | relL2 (YaRN×4,ref) | relL2 (linear×4,ref) |
|---|---|---|---|---|---|---|---|
| 0 | 0,0 | 0,0 | 0,0 | 0,0 | 0,0 | 2,39e-1 | 0,0 |
| 1 | 6,0e-8 | 1,04e-8 | 1,45e-8 | 1,42e-8 | **0,0** | 2,36e-1 | 7,7e-2 |
| 3 | 1,2e-7 | 1,97e-8 | 2,12e-8 | 1,78e-8 | **0,0** | 2,37e-1 | 2,16e-1 |
| 34 | 1,0e-6 | 8,56e-8 | 4,91e-8 | 9,12e-8 | **0,0** | 2,39e-1 | 3,70e-1 |
| 512 | 1,4e-5 | 1,59e-6 | 9,51e-7 | 8,10e-7 | **0,0** | 2,35e-1 | 3,92e-1 |
| 4 096 | 1,3e-4 | 1,37e-5 | 8,08e-6 | 6,93e-6 | **0,0** | 2,40e-1 | 5,24e-1 |
| 16 384 | 4,8e-4 | 5,37e-5 | 3,20e-5 | 2,71e-5 | **0,0** | 2,73e-1 | 5,94e-1 |
| 32 768 | 1,2e-3 | 1,11e-4 | 6,43e-5 | 5,73e-5 | **0,0** | 3,37e-1 | 6,34e-1 |
| 65 536 | 2,6e-3 | 2,21e-4 | 1,28e-4 | 1,14e-4 | **0,0** | 4,08e-1 | 6,01e-1 |
| **131 071** | **3,8e-3** | **3,61e-4** | 2,42e-4 | 4,19e-4 | **0,0** | 3,74e-1 | 6,51e-1 |
| 262 143 | 9,3e-3 | 1,04e-3 | 9,06e-4 | 7,28e-4 | **0,0** | 4,18e-1 | 5,87e-1 |

  - **Controle decisivo**: `ggml IMROPE+sections{11,11,10,0}` contra `ggml NEOX n_dims=64`
    dá **rel-L2 = 0,000e+00 exatamente** em todas as posições, até 262 143. Isto é a
    prova numérica de que, com as 4 seções de posição iguais (`llama-batch.cpp:780-788`),
    a chamada do qwen35 **é** um RoPE split-half comum sobre os 64 dims — a estrutura que
    o nosso kernel implementa.
  - `max |dtheta| = 7,8e-3` = **exatamente 1 ulp** do ângulo naquele ponto (o ulp de
    `theta ≈ 1e5` é `2^-7 = 7,8e-3`), no máximo **6 ulp**; `max|dcos| = 7,5e-3`,
    `max|dsin| = 2,6e-3`. Ou seja: a diferença nossa-vs-referência em posição longa é
    *inteiramente* explicada pela ordem de operações do cálculo do ângulo em fp32
    (`powf` por par vs. produto iterativo).
  - Leis medidas: `rel-L2(ours,ref) ≈ 4e-9 · pos` (média medida 4,4e-9) e
    `rel-L2(ref,exato) ≈ 3e-9 · pos` — as duas escalas são as mesmas; **nenhum dos dois
    motores é "o certo" em posição longa**, os dois estão a ≲1 ulp do ângulo exato.
  - Tamanho do erro que uma **configuração de scaling** causaria (as duas colunas de
    controle, ambas com `freq_scale = 0,25`, ou seja "×4"): tipo YaRN
    (`ext_factor=1, attn_factor=0,5`) → **rel-L2 0,24 a 0,42**; tipo linear (`ext_factor=0`)
    → 0 na pos 0 e 0,08-0,65 da pos 1 a 131K. Isto é: se o modelo pedisse qualquer escala e nós
    não aplicássemos (ou o contrário), a diferença seria de **24-65 %**, três ordens de
    grandeza acima dos 3,6e-4 medidos a 131K. O controle serve para dimensionar, não para
    afirmar qual escala o modelo pediria — isso está provado pelas chaves do GGUF e pela
    resolução do llama.cpp (§1).
- **Veredito**: **fórmula MANTIDA (sem mudança de código)**: o kernel está certo e a
  diferença residual em 131K é o piso de fp32, 3,6e-4 rel-L2, três/quatro ordens de
  grandeza abaixo de qualquer erro de fórmula ou de escala. **Nenhum bug de RoPE, nenhum
  scaling faltando.** O gate novo entrou com limite `rel-L2 < max(1e-7, 4e-8·pos)` (10× a
  lei medida): falha se a diferença crescer mais rápido que o orçamento de fase do fp32.

  Margem do gate (limite `max(1e-7, 4e-8·pos)` contra o medido): pos 1 → 1,0e-7 vs 1,0e-8;
  512 → 2,0e-5 vs 1,6e-6; 4 096 → 1,6e-4 vs 1,4e-5; 131 071 → 5,2e-3 vs 3,6e-4;
  262 143 → 1,0e-2 vs 1,0e-3. **Todas as 11 posições passam com 10-14× de margem**, e o
  controle NEOX mede 0,0 exato. (A primeira versão do gate usava 1e-5 fixo até 4096 e
  falhava por 1,4× em 4096 — foi a medida que corrigiu o gate, não o contrário.)

### Consequência para o resto da noite (número, não opinião)

- Em 131K, `Q`/`K` pós-RoPE **não podem** ser comparados bit-exatos com o llama.cpp: o
  próprio cálculo da fase em fp32 custa ~2,4e-4 (referência) a 4,2e-4 (nosso) de rel-L2
  contra a rotação exata. Qualquer diff de nó no nível de 1e-4 a 131K **não** é evidência
  de bug do motor.
- Para comparação: o KV `q8_0` introduz ~2e-3 relativo por elemento e o `q4_0` ~5e-2 —
  uma ordem de grandeza (q8_0) a duas (q4_0) **acima** do erro de RoPE em 131K. Ou seja:
  a 131K quem limita a qualidade é a quantização do KV, não a fase do RoPE.

---

## 3. Curva de custo da atenção em 16K/64K/131K, e a política de split

- **Referência**: `docs/autotuning-gfx1201.md` §3.2 (tabela de splits medida; `kAttnSplitWpbWide`
  embarcado = CTA larga a partir de 16 splits), `docs/medicoes-banda-e-gargalos.md` §3.2
  (a atenção a 64K q4_0 está limitada por issue/desquantização, 346 GB/s emitidos),
  `docs/kv-memoria-desenho.md` §4.2 (bytes emitidos vs únicos).
- **Hipótese**: (a) a fração do passo de decode gasta na atenção cresce com o contexto e é a
  diferença entre 16K e 131K; (b) a política `keys/kAttnSplitMin` (≤16 splits) continua sendo
  a escolha certa a 64K/131K, ou o ótimo depende do tipo de KV como mediu a frente de
  autotuning (24 splits ganham em q4_0/131K e perdem em f16).
- **Comandos**: `timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu <f16|q8_0|q4_0> 4096 16384 65536 131072`
  e `timeout 900 ./scripts/gpu-lock.sh ./build/bench-phases-gpu <modelo> --ctx <n> --pos <n-64> --tokens 24 --level 2 [--kv-k T --kv-v T]`.
  Janela: anotada em §6 (o log registra VRAM antes/depois de cada comando).
- **Modelo de bytes (conta minha, confere com `kv-memoria-desenho.md` §4.2)** — por token, 16
  camadas de atenção, 24 cabeças de consulta, 4 KV, head_dim 256:

| KV | ctx | emitido (GB/token) | único (GB/token) | DRAM a 640 GB/s | IC a 1,4 TB/s |
|---|---|---|---|---|---|
| f16 | 16 384 | 6,44 | 1,07 | 1,7 ms | 4,6 ms |
| f16 | 65 536 | **25,77** | **4,29** | 6,7 ms | 18,4 ms |
| f16 | 131 072 | **51,54** | **8,59** | 13,4 ms | 36,8 ms |
| q8_0 | 65 536 | 13,69 | 2,28 | 3,6 ms | 9,8 ms |
| q8_0 | 131 072 | 27,38 | 4,56 | 7,1 ms | 19,6 ms |
| q4_0 | 65 536 | 7,25 | 1,21 | 1,9 ms | 5,2 ms |
| q4_0 | 131 072 | **14,50** | **2,42** | 3,8 ms | 10,4 ms |

  O fator emitido/único é **6,0×** exatamente em todas as linhas (é o GQA 6:1), e a coluna
  "IC" é o que a atenção custaria se os bytes emitidos fossem servidos a 1,4 TB/s.
- **Resultado / veredito**: preenchido em §3.1 abaixo.

### 3.1 Resultado medido (curva + política) — `bench-attn-gpu`, KV f16

`./scripts/gpu-lock.sh timeout 900 ./build/bench-attn-gpu f16 4096 16384 65536 131072`
(varredura de `splits ∈ {2,4,8,16}` × `warps/CTA ∈ {8,16,32}`, mínimo de 3 rodadas por célula).
**Piso de ruído da corrida: 1,000× / 0,995× / 0,999× / 0,999×** (o mesmo kernel sem split
medido como 3 células independentes).

| ctx | kernel sem split | **política que embarca** (`keys/512`≤16) | melhor célula | melhor vs política | ms/token em jogo (×16 camadas) | atenção em ms/token |
|---|---|---|---|---|---|---|
| 4 096 | 0,3434 ms | 8×8 = **0,0723** | 4×16 = 0,0693 | 1,044× (piso 1,000) | 0,049 ms | **1,16** |
| 16 384 | 1,3433 | 16×8 = **0,2429** | 4×16 = 0,2318 | 1,048× (piso 0,995) | 0,178 ms | **3,89** |
| 65 536 | 6,5226 | 16×8 = **0,9671** | 4×16 = 0,9593 | 1,008× (**ruído**) | 0,125 ms | **15,47** |
| 131 072 | 13,0183 | 16×8 = **1,9330** | 4×16 = 1,9123 | 1,011× (**ruído**, piso 0,999) | 0,332 ms | **30,93** |

- **Confirma a frente de autotuning de forma independente**: "4 splits × 16 warps" é a melhor
  célula a 4K/16K (aqui 1,044×/1,048×, lá 1,046×) e a 64K/131K a diferença cai para o piso de
  ruído (aqui 1,008×/1,011× com piso 0,999×, lá "nada acima do ruído"). A política embarcada
  (16×8 a partir de 16K, CTA larga a partir de 16 splits) está **a ≤5 % do ótimo em todos os
  contextos medidos**, e o que está em jogo são 0,05-0,33 ms/token (0,1-0,9 % do passo).
- **Curva da atenção**: 1,16 → 3,89 → 15,47 → **30,93 ms/token** de 4K a 131K (×16 camadas),
  ou seja a atenção sai de ~3 % do passo a 4K para a maior parte dele a 131K (com o passo a
  131K medido em §3.2). É esta curva que a noite tinha que ter por escrito.

---

### 3.2 Onde o tempo do passo vai (motor real, `bench-phases-gpu --level 2`) — medido

**Comando**: `./scripts/gpu-lock.sh timeout 850 ./build/bench-phases-gpu <modelo> --ctx <n>
--pos <n-64> --tokens 24 --level 2 --kv-k T --kv-v T` (contexto semeado: posições n-64..n-1;
cache sintético). 24 tokens por célula, melhor da passada limpa (o `bench-phases` aquece com
`sync` no laço até ≥300 ms de GPU ocupada).

| ctx / KV | **passo** | tok/s | **atenção** | fatia do passo | matvec (trunk) | matvec GB/s | **emitido (GQA 6×)** | splits |
|---|---|---|---|---|---|---|---|---|
| 16 384 f16 | **37,10 ms/token** | 26,95 | **4,487 ms** | **12,1 %** | 23,96 ms | 464,7 | 1430 GB/s | 16 |
| 65 536 f16 | **49,13 ms/token** | 20,35 | **16,494 ms** | **33,6 %** | 24,03 ms | 463,3 | 1561 GB/s | 16 |
| 131 072 q4_0 | **70,74 ms/token** | 14,14 | **38,320 ms** | **54,2 %** | 24,17 ms | 460,6 | **378 GB/s** | 16 |

- **A curva que a noite devia ter por escrito**: a atenção sai de **12 % do passo a 16K** para
  **34 % a 64K e 54 % a 131K**; o matvec do tronco fica **constante em ~24 ms/token**
  (11,133 GB lidos a ~460 GB/s), que é o piso do motor em qualquer contexto; o head
  (`output.weight` 0,874 GB) custa 1,391 ms a 628 GB/s.
- **Banda emitida medida**: 1430 GB/s a 16K, **1561 GB/s a 64K** (o Infinity Cache servindo a
  releitura 6× — o meu número derivado em §4.2 era 1,67 TB/s, coerente) e **378 GB/s a 131K
  q4_0** contra 2,4 GB/token de bytes únicos (63 GB/s). Isto é a confirmação independente do
  diagnóstico da frente de banda ("a 64K+ com `q4_0` o kernel é limitado por
  desquantização/issue, não por banda"): 378 GB/s emitidos é **1/4** do que o cache entrega em
  f16, e o kernel está *mais lento* apesar de mover 4× menos bytes.
- O `14,14 tok/s` a 131K q4_0 bate com os 14,0 tok/s do baseline e com o meu `bench` em §3.4
  (14,09) — três medidas independentes do mesmo ponto.

### 3.3 Curva da atenção por **tipo de KV** (a comparação que faltava a 64K+)

Comandos: `./scripts/gpu-lock.sh timeout 900 ./build/bench-attn-gpu <tipo> <ctx...> --splits 16
--wpb 8` (job C) e a varredura completa em f16 (job A). Números por camada × 16 = ms/token do
passo; "fatia" usa o tok/s de decode do baseline (`docs/medicoes-m5.md`, cache sintético).

| ctx | f16 (ms/layer → ms/token) | q8_0 | q4_0 | passo medido (tok/s) | fatia da atenção (q4_0/q8_0) |
|---|---|---|---|---|---|
| 4 096 | 0,0723 → **1,16** | — | — | 28,6 (34,9 ms) | 3 % |
| 16 384 | 0,2429 → **3,89** | — | — | 26,7 (37,5 ms) | 10 % |
| 65 536 | 0,9671 → **15,47** | 1,1512 → **18,42** | 1,2428 → **19,88** | 19,3 q8_0 (51,8 ms) | **36 %** |
| 131 072 | 1,9234 → **30,77** | 2,2653 → **36,24** | 2,3837 → **38,14** | 14,0 q4_0 (71,4 ms) | **53 %** |

- **A atenção sai de ~3 % do passo a 4K para ~53 % a 131K** (q4_0), com os dois tipos
  quantizados medidos na mesma janela.
- **q8_0 é mais rápido que q4_0** na atenção, a 64K (7,4 %) e a 131K (5,0 %), apesar de mover
  ~1,9× mais bytes: o `q4_0` paga desempacotamento de nibble por elemento. Consequência
  prática para a noite: se o KV `q5_0`/`q4_1` da frente KV tiver um caminho de leitura
  vetorizado, ele deve ser **mais rápido e mais preciso** que o `q4_0` — e é isso que o
  `kv_load8` do `q4_0` não faz hoje.

### 3.4 Custo de decode a 64K/131K com cache sintético (o que o baseline reporta) — medido

**Comando**: `./scripts/gpu-lock.sh timeout 900 ./build/rdna4-infer bench -m <modelo>
--ctx-size <n> --start-pos <n-64> -n 32 --reps 3 --cache-type-k T --cache-type-v T`
(`--start-pos` semeia o KV por kernel: é **custo**, não qualidade — o próprio binário imprime
"[synthetic keys/values: cost measurement, not quality]").

| ctx | KV | VRAM em uso (de 15,92 GiB) | **decode** | ms/token | GTT | banda efetiva |
|---|---|---|---|---|---|---|
| 65 536 | q8_0/q8_0 | **13,75 GiB** (2,18 livres) | **19,24 tok/s** | 52,0 | 75 MB | 231 GB/s |
| 65 536 | f16/f16 | **15,62 GiB** (0,30 livres) | **19,92 tok/s** | 50,2 | 75 MB | 240 GB/s |
| 131 072 | q4_0/q4_0 | **13,87 GiB** (2,05 livres) | **14,09 tok/s** | 71,0 | 75 MB | 170 GB/s |

- **Nenhum derrame para GTT em nenhuma das três** (`mem_info_gtt_used` = 75 MB antes e depois;
  a VRAM volta a 4,3 GB ao fim). **Contradição medida** com `docs/medicoes-banda-e-gargalos.md`
  §2.4/§5 ("o caso 64K f16: 2,1 GB vão para GTT e o decode cai 8×, de 18,1 para 2,2-8,0 tok/s")
  e com `docs/journal-noite.md:148` ("com f16 não cabe"): nesta árvore e nesta janela, **64K com
  KV f16 CABE** — 15,62 GiB em uso de 15,92 (0,30 GiB livres), GTT inalterado em 75 MB e
  **19,92 tok/s**, ou seja *mais* rápido que o `q8_0` (19,24) e sem penhasco nenhum.
  Consequência para a decisão de default acima de 56K (`docs/backlog-noite.md` item 1): a troca
  para `q8_0` continua defensável — mesmo tempo, o **dobro da precisão** — mas **não** pelo
  motivo escrito ("matar o penhasco de 8× do GTT"), que não se reproduz aqui. Se alguém mediu o
  penhasco, ou era outra configuração (MTP/`--layers` alocados junto) ou a janela estava suja;
  fica como item a re-medir com o mesmo comando dos dois lados.
- Os 131K em q4_0 a 14,09 tok/s reproduzem o 14,0 do baseline ✓ (mesma metodologia de cache
  sintético, agora com a janela limpa documentada).

### 3.5 Política de splits a 64K/131K — confirmação independente (job H)

Comando: `./scripts/gpu-lock.sh timeout 850 ./build/bench-attn-gpu <tipo> 131072 --splits
8,16,24,32,48 --wpb 8,16 --smax 48` (e o mesmo a 64K). Piso de ruído 1,000-1,003×.

| ctx / KV | política que embarca (16×8) | melhor célula | ganho | ms/token em jogo |
|---|---|---|---|---|
| 131 072 q4_0 | 2,4315 ms | **48×16 = 2,2859** | **1,064× (6,4 %)** | 2,33 ms (3,3 % do passo) |
| 131 072 q8_0 | 2,3420 | **24×16 = 2,2721** | 1,031× (3,1 %) | 1,12 ms |
| 131 072 f16 | (0,991× medido no job A: 16×8 = 1,9330) | 24×16 = 1,9406 | 0,996× (nada) | — |

- **Confirma de forma independente a frente de autotuning** (`docs/autotuning-gfx1201.md` §3.2:
  "24/32/48/64 splits ≥ 8-16 splits, exceto no KV q4_0 a 131K", e o `kAttnSplitWpbWide` já
  embarcado): a 131K **q4_0** a política de 16×8 **não é o ótimo** — 48×16 é 6,4 % melhor. A
  131K **f16** os 24 splits não ajudam (0,996×), como o job A mediu na varredura completa.
- Isto é uma decisão de política que depende do **tipo de KV** (o número de splits ótimo cresce
  quando a linha é pequena) e está registrada como candidata em `docs/autotuning-gfx1201.md` §5.

---

## 4. GQA 6:1 — quanto custa a releitura, e o compartilhamento compensa?

- **Referência**: `docs/kv-memoria-desenho.md` §4.2 (tabela lógica vs única),
  `docs/rocm-estudo.md` §D.8 ("protótipo do M7: 8-12× mais lento; só reconsiderar com a
  grade larga dos splits e com staging em LDS"), `docs/autotuning-gfx1201.md` §5 item 3
  (a alavanca de GQA é a #7 do roadmap, "até +18% a 64K"), `docs/medicoes-banda-e-gargalos.md`
  §3.3 (a 64K q4_0 a atenção **não** é limitada por banda).
- **Hipótese**: o protótipo do M7 falhou por **grade estreita + pressão de registrador**, não
  porque compartilhar K/V seja inútil. Se for isso, um protótipo com a mesma grade do kernel
  com splits (`n_head_kv × S` CTAs) e o mesmo merge deve mostrar ganho onde o kernel está
  limitado por cache (f16 a 4K-16K) e nenhum ganho onde está limitado por issue (q4_0 a 64K+).
- **Desenho do experimento** (2×2, `tests/bench_attn_gpu.hip`, protótipo C novo):
  `attn_gqa_split_kernel<KT,VT,HG>` com grid `(n_head_kv, S)` e um parâmetro `HG` = cabeças de
  consulta por passada sobre a fatia de chaves:
  - `HG=1` → 6 passadas: **mesmo tráfego do kernel que embarca**, grade diferente;
  - `HG=3` → 2 passadas: metade do tráfego;
  - `HG=6` → 1 passada: 1/6 do tráfego.
  Comparar `HG=1` com `HG=6` **na mesma grade** separa "o ganho é o tráfego" de "o ganho é a
  grade"; comparar `HG=6` com o kernel que embarca diz se a grade larga sobrevive.
  Os parciais são escritos no **mesmo layout `[h][s]`** do kernel com splits, então o
  `attn_merge_kernel` que já existe combina os dois sem mudança.
  O bench também passou a imprimir `hipFuncGetAttributes` (registradores/spill) de cada kernel:
  um protótipo que derrama registrador explica a própria lentidão sem teoria de grade.
- **Comando**: `timeout 900 ./scripts/gpu-lock.sh ./build/bench-attn-gpu <tipo> <t...> --splits S --wpb 8 --gqa-hg 6,3,2,1`
  (com `--gqa-splits S` para forçar S diferente da política).
- **Evidência prévia que já responde metade da pergunta** (do próprio repo): a 64K o KV `f16`
  — que emite **3,5× mais bytes** que o `q4_0` (25,77 vs 7,25 GB/token) — é *mais rápido*
  (19,0 vs 17,8 tok/s, `README.md`/`docs/medicoes-m7.md` §107-108, citado em
  `docs/auditoria-qualidade.md`). Se o kernel a 64K fosse limitado por banda, o tipo que
  emite 3,5× menos bytes não poderia perder: o `q4_0` a 64K é **limitado por
  desquantização/issue**, e é exatamente por isso que reduzir tráfego (compartilhar K/V)
  não pode ajudar lá. A medida que falta — e que o protótipo C dá — é *quanto* isso vale
  em cada contexto.
- **Resultado / veredito**: §4.1.

---

### 4.1 Resultado medido (atenção sintética, `bench-attn-gpu`)

**Comando**: `./scripts/gpu-lock.sh timeout 900 ./build/bench-attn-gpu f16 4096 16384 --splits 8 --wpb 8 --gqa-hg 6,3,2,1`
(+ as corridas de 64K/131K e q4_0/q8_0 em §4.2). Coordenação: 24 cabeças de consulta / 4 KV,
head_dim 256, 1 token de consulta, **por camada** (× 16 camadas para ms/token).
Piso de ruído da própria corrida: **0,999× (4K) e 0,998× (16K)** (o mesmo kernel sem split
medido como 3 células independentes, mínimo de 3 rodadas).

**Recursos dos kernels** (`hipFuncGetAttributes`; `multiProcessorCount` = 32 = **32 WGP** =
64 CU, conforme `docs/rdna4-gfx1201-hardware-brief.md:9`; o joelho de ocupação é ~96 VGPR):

| kernel | VGPR/lane | derrame | leitura |
|---|---|---|---|
| `attn_kernel<f16>` (sem split) | 95 | 0 B | no joelho de ocupação |
| `attn_split_kernel<f16>` | 97 | 0 B | no joelho |
| `attn_gqa_kernel<f16>` (**protótipo B do M7**) | 42 | **1232 B** | derrama 1,2 KB por lane |
| `gqa_split HG=6` (protótipo C) | 68 | 400 B | derrama pouco |
| `gqa_split HG=3` | 125 | 0 B | sem derrame |
| `gqa_split HG=1` | 56 | 0 B | sem derrame |
| `attn_merge_kernel` | 10 | 0 B | — |

| t (chaves) | ship (split 8×8) ms | gqa-split HG=1 ms | HG=3 ms | HG=6 ms | HG=6 vs ship | HG=6 vs HG=1 | protótipo B (M7) ms | piso de ruído |
|---|---|---|---|---|---|---|---|---|
| 4 097 | **0,0715** | 0,1358 | 0,1293 | 0,1694 | 2,37× mais lento | 1,25× mais lento | 5,168 | 0,999× |
| 16 385 | **0,2343** | 0,2652 | 0,2657 | 0,4834 | 2,06× mais lento | 1,82× mais lento | 20,149 | 0,998× |

- **Equivalência numérica**: todos os HG medem `rel-L2 = 3,2e-07` (4K) e `6,3e-07` (16K)
  contra o kernel sem split — o mesmo valor que o kernel com splits que embarca (3,24e-07 /
  6,45e-07). O protótipo C está correto; e o `HG=1` (seis passadas, mesmo tráfego do kernel
  que embarca) mede o mesmo, o que fecha a checagem de que a grade `(4,S)` com passadas
  sequenciais não introduz erro.
- **O que isso diz, com o número**: com a **mesma grade** `(4,S)`, cortar o tráfego de K/V
  em 6× (`HG=1` → `HG=6`) deixa o kernel **1,25× (4K) e 1,82× (16K) MAIS LENTO**. Ou seja:
  a releitura 6× **não é o gargalo** nesses contextos — ela é absorvida pelo cache — e o
  custo de servir 6 cabeças por CTA (laço serial de 6 reduções/softmax por linha + pressão
  de registrador: 68 VGPR e 400 B de derrame no HG=6, contra 56 VGPR e 0 B no HG=1) é maior
  que os bytes economizados.
- **Correção de leitura do repo**: `docs/medicoes-banda-e-gargalos.md` §3.2 lê "1.315 GB/s
  emitidos a 16K = limitada por cache (88 % do IC)" como prova de que a atenção é limitada
  por cache. Essa banda é **derivada do tempo medido** (bytes emitidos ÷ tempo), então não é
  evidência independente do limitador; o experimento HG=1 vs HG=6 no mesmo grid é: se a
  banda fosse o limitador, 1/6 dos bytes teria que ser ≥1× mais rápido. Não é. O limitador
  mensurável é a **paralelidade (CTAs)**: o mesmo kernel com grade `(24,S)` ganha 4,73× (4K)
  e 5,62× (16K) do kernel sem split, e com grade `(4,S)` perde ~1,9×/1,1× para ele.
- **Sobre o protótipo B do M7**: os 8-12× de lentidão têm agora **duas** causas medidas — o
  derrame de **1232 B/lane** (42 VGPR alocados, o compilador preferiu derramar) e a grade de
  4 CTAs. A hipótese "grade estreita" que o `docs/rocm-estudo.md` §D.8 registrou estava
  incompleta: metade do problema é registrador.

### 4.2 GQA 6:1 — a resposta completa (f16, 4K→131K), e onde o compartilhamento ganha

Tabela de `bench ms` por **camada** (1 token de consulta, 24 cabeças / 4 KV); ×16 camadas =
ms/token. Piso de ruído medido em cada contexto: 0,995-1,000×.

| ctx | sem split | **embarca** (grid 24×S) | GQA HG=1 (6 passadas) | HG=3 (2 passadas) | HG=6 (1 passada) | GQA S=24 HG=3 | ganho do GQA vs embarca |
|---|---|---|---|---|---|---|---|
| 4 097 | 0,3434 | **0,0723** (S=8) | 0,1358 | 0,1293 | 0,1694 | — | **0,56× (perde 1,8×)** |
| 16 385 | 1,3433 | **0,2429** (S=16) | 0,2652 | 0,2657 | 0,4834 | — | **0,91× (perde 1,1×)** |
| 65 537 | 6,5226 | **0,9671** (S=16) | 2,5523 | 1,0378 | 1,5059 | — | 0,93× (perde 1,07×) |
| 131 073 | 13,0183 | 1,9234 (S=16) / 2,0866 (S=24) | 5,0749 | 2,0536 (S=16) | 3,9803 | **1,7793** (S=24) | **1,08× vs S=16; 1,17× vs S=24** |
| protótipo B (M7) | — | — | 5,168 / 20,149 / 79,2 / 160,7 | | | | 0,014-0,012× (72-86× mais lento) |

- **4K/16K (onde o motor passa a vida): compartilhar K/V não compensa.** Com a mesma grade,
  cortar o tráfego 6× (HG=1→HG=6) deixa o kernel **1,25× (4K) e 1,82× (16K) mais lento**: a
  releitura 6× é absorvida pelo Infinity Cache e o que custa é servir 6 cabeças por CTA
  (laço serial de reduções/softmax + registrador: HG=6 = 68 VGPR **e 400 B de derrame**,
  HG=1 = 56 VGPR sem derrame). A grade de 4 CTAs por split não substitui a de 24.
- **131K f16: aí sim o tráfego pesa** — HG=1 (6× tráfego) explode para 5,07 ms e o HG=6 (1× )
  fica em 3,98 ms; e com **S=24** o HG=3 chega a **1,7793 ms**, ou seja **1,08× mais rápido que
  a política de 16 splits** (2,3 ms/token × 16 camadas = ~3 % do passo a 131K) e 1,17× mais
  rápido que o mesmo kernel que embarca com 24 splits (2,0866). **Mas 131K f16 não cabe na
  VRAM** (8 GiB de KV + 11,2 de pesos): o tipo que a noite pode usar a 131K é q4_0 (§4.3).
- **Recomendação (com o número)**: **não implementar atenção agrupada por GQA para os
  contextos onde o motor vive (4K-16K): perde 1,1-1,8× medido.** A 131K f16 ela ganha 8 %, e o
  número que justifica ou mata isso é o de q4_0/q8_0 a 131K (§4.3) — se lá o kernel for
  limitado por desquantização (como a frente de banda mediu: 346-372 GB/s emitidos contra um
  teto de cache de ~1,6 TB/s), o ganho de tráfego não se converte e a resposta continua "não
  vale". Independentemente disso, o custo de implementação não é o de um ajuste: é um kernel
  novo com merge próprio (o protótipo C tem 130 linhas e um parâmetro HG).
- **O que a releitura 6× custa de fato, medido**: em 131K f16 uma execução do kernel que
  embarca move `24 cabeças × 131 072 chaves × 1 024 B (K+V) = 3,22 GB` de tráfego **emitido**
  em 1,9234 ms = **1,67 TB/s** de L2 (contra 279 GB/s de bytes *únicos*, `2 × 268 MB`). Ou
  seja: o 6× está sendo **servido pelo cache** a 1,67 TB/s, não indo à DRAM. Um kernel que
  elimina a releitura economiza bytes de L2 que não são o gargalo a 4K/16K (medido: HG=6 é
  mais lento que HG=1 lá) e que só passam a pesar a 131K (medido: HG=1 2,6× pior que HG=3).

### 4.3 q4_0 e q8_0 a 64K/131K — onde o compartilhamento **ganha** (e por quê)

Tabela atualizada com o **q8_0** (job C completo, `--splits 16 --wpb 8`, grade do protótipo
`4×16`, piso de ruído 1,000-1,009× em todas as linhas):

| ctx / KV | embarca 16×8 | **GQA HG=3** (S=16) | ganho | ms/token em jogo (×16) |
|---|---|---|---|---|
| 65 536 q4_0 | 1,2428 | **1,1115** | **11,8 %** | 2,10 ms |
| 131 072 q4_0 | 2,3837 | **2,1767** | **9,5 %** | 3,31 ms |
| 65 536 q8_0 | 1,1512 | **1,0558** | **9,0 %** | 1,53 ms |
| 131 072 q8_0 | 2,2653 | **2,0775** | **9,0 %** | 3,00 ms |
| 65 536 f16 | 0,9671 | 1,0378 | −7,3 % (perde) | −1,13 ms |
| 131 072 f16 | 1,9234 (S=16) | 2,0536 (S=16) / **1,7793** (S=24) | −6,8 % / **+8,1 %** | +2,30 ms (S=24) |

- **Efeito colateral útil para a escolha de KV**: a atenção com **q8_0 é mais rápida que com
  q4_0** nos dois contextos (64K: 1,1512 vs 1,2428 ms = 7,4 %; 131K: 2,2653 vs 2,3837 = 5,0 %)
  **apesar de mover ~1,9× mais bytes** — confirma de forma independente o que a frente de banda
  suspeitava (a 64K+ o `q4_0` é limitado pelo desempacotamento de nibble, não por banda) e
  reforça o alvo da noite (K/V quantizado mais largo é mais rápido *e* mais preciso).


Mesmo comando de §4.1, com `--splits 16 --wpb 8` (a política que embarca nesses contextos) e
`--gqa-hg 6,3,2,1` (grade `4×16` no protótipo C). Piso de ruído: **1,009× (64K q4_0)** e
**1,000× (131K q4_0)** — as diferenças abaixo estão muito acima dele.

| ctx / KV | sem split | **embarca** 16×8 | GQA HG=2 | **GQA HG=3** | GQA HG=6 | protótipo B (M7) | **HG=3 vs embarca** | ms/token em jogo (×16) |
|---|---|---|---|---|---|---|---|---|
| 65 536 q4_0 | 6,6315 | **1,2428** | 1,2753 | **1,1115** | 1,3933 | 82,473 | **1,118× (11,8 % mais rápido)** | 2,10 ms |
| 131 072 q4_0 | 13,6355 | **2,3837** | 2,5176 | **2,1767** | 3,7435 | 160,414 | **1,095× (9,5 % mais rápido)** | 3,31 ms |
| 131 072 f16 | 13,0183 | 1,9234 (16×8) | 2,5817 | 2,0536 (S=16) / **1,7793 (S=24)** | 3,9803 | 160,703 | 0,94× (S=16) / **1,08× (S=24)** | −3,3 ms (S=24) |
| 65 536 f16 | 6,5226 | 0,9671 | 1,3105 | 1,0378 | 1,5059 | 79,211 | 0,93× | −1,13 ms |
| 16 385 f16 | 1,3433 | 0,2429 | 0,2834 | 0,2657 | 0,4834 | 20,398 | 0,91× | −0,36 ms |
| 4 097 f16 | 0,3434 | 0,0723 | 0,1397 | 0,1293 | 0,1694 | 5,296 | 0,56× | −0,91 ms |

- **O padrão é coerente e tem mecanismo**: o protótipo C **ganha onde a linha de KV é
  pequena** (q4_0: 144 B por chave) e **perde onde é grande** (f16: 512 B). Com q4_0 o kernel
  que embarca paga desquantização/issue por linha lida 6× (a frente de banda já tinha medido
  346-372 GB/s emitidos contra um teto de cache de ~1,6 TB/s: é issue, não banda), e o
  agrupamento por GQA com `HG=3` reduz essas passadas de 6 para **2** — corta exatamente o
  trabalho que é o gargalo. Com f16 a linha é grande, o kernel fica limitado por banda de
  cache (1,67 TB/s medidos) e aí o que importa é a paralelidade (CTAs), que o agrupamento
  tira.
- **Recomendação (com o número)**: **não implementar para 4K-16K f16** (perde 1,1-1,8×
  medido — é onde o motor passa a vida) e **levar o protótipo adiante para q4_0 a 64K/131K**,
  onde ganha **9,5-11,8 % (2,1-3,3 ms/token, ~4 % do passo)** já com `S=16` (mesma grade do
  kernel que embarca, 6× menos CTAs) e com equivalência numérica medida (`rel-L2 1,1e-06` a
  64K e `1,6e-06` a 131K, o mesmo valor do caminho com split que embarca).
- **O que falta para fechar**: comparar contra a **melhor célula** do kernel que embarca (24/32
  splits, que a frente de autotuning mediu como melhores em q4_0 a 131K) — está na fila
  (`bench-attn-gpu q4_0 131072 --splits 8,16,24,32,48 --wpb 8,16`), e o q8_0 a 64K/131K
  (o tipo que a frente de banda deixou "não medido"), também na fila. Se a melhor célula do
  kernel que embarca chegar perto de 2,18 ms, o ganho do GQA a 131K some e a recomendação
  volta a "não vale".

## 5. Achados colaterais (bugs encontrados, com evidência)

### 5.1 `bench-attn-gpu`: a checagem numérica do protótipo GQA lia VRAM não inicializada

- **Como apareceu**: ao estender o bench para o protótipo C eu precisava do tensor de
  referência (`a`) antes da medida, e `grep -n "d_out\b"` mostrou que **nenhum kernel do
  bench escreve `d_out`** (`tests/bench_attn_gpu.hip:472-476` aloca, `:838` lê). A linha
  `prototype vs ship: rel-L2` comparava o protótipo do M7 contra lixo de VRAM.
- **Quando quebrou (evidência de git, não opinião)**: `d_out` **era** escrito pelo bloco de
  medição do kernel que embarca (`git show a830570:tests/bench_attn_gpu.hip:219`,
  `2236d62:219`, `4344d12:219` — `attn_launch(..., d_out, ...)`). No refactor da varredura do
  commit **`9ef1526`** ("Autotuning gfx1201 (task 6)", 2026-09-14, **ancestral do `main`**) esse
  `attn_launch` virou `d_out2` e a referência ficou pendurada: a linha passou a comparar contra
  VRAM não inicializada.
- **O que isso invalida**: a afirmação de `docs/medicoes-m7.md:65-66` ("o protótipo GQA do bench
  é *bit-idêntico* ao kernel que embarca, `rel-L2 0.0e+00`") **foi medida antes** do refactor
  (`2236d62`, com o `d_out` sendo escrito) — então o número do M7 estava certo quando foi
  escrito; o que ficou inválido foi **qualquer re-medição depois de `9ef1526`**: desde então a
  coluna não media nada. Nenhuma conclusão do repo depende dela (o tempo do protótipo continua
  válido — é tempo), mas quem re-mediu o protótipo nesse intervalo leu lixo.
- **Correção** (só em `tests/`): a referência agora é uma execução explícita do kernel sem
  split em `d_out2`, copiada para `a` antes das comparações.
- **Re-confirmação com a referência válida**: com o conserto, o protótipo B do M7 mede
  **`rel-L2 0,0e+00` contra o kernel sem split em todos os contextos (4K a 131K)** — é mesmo
  bit-idêntico (o protótipo só reagrupa CTAs; a aritmética por cabeça e a ordem do merge são as
  mesmas). A afirmação do M7 está de pé, agora medida de novo na árvore atual.
- **Para o coordenador**: se o merge do protótipo C entrar, esta correção entra junto (é o
  mesmo arquivo).

### 5.2 Alcançabilidade: o que dá para medir a 131K dentro da caixa de tempo da noite

Medido (ver §6 para os comandos e as janelas):

| etapa | número medido | consequência |
|---|---|---|
| prefill em lote (chunks de 16, ctx 4096) | **74,9 tok/s** (13,35 ms/token) — medido pelo `bench-phases-gpu` da frente de medições (worktree `rdna4-wt-noite-medicoes-gpu`, log `/tmp/base-prefill.log`), **a confirmar com a minha própria medida** em §6 | 131 072 tokens = **1 750 s** |
| decode a 4K (f16) | 34,9 ms/token (28,6 tok/s) | 131 072 tokens = 78 min |
| `ppl` do motor (1 forward por token) | ~24-28 tok/s | 131K = ~1,5-1,8 h |
| comando de GPU da noite | `timeout 900` | **1 500 s** |

Ou seja: **nenhuma medida fim-a-fim a 131K cabe no orçamento de 900 s por comando** — nem o
prefill em lote (1 750 s só para encher o KV). O teto honesto desta noite, com o prefill em
lote, fica em ~16-32K — o `elapsed` das sondagens de 16K e 32K em §6 dá o número real
(aqui a estimativa era 219 s e 437 s a 74,9 tok/s, **sem** a atenção, que cresce com o
contexto e é o termo dominante no prefill em lote: 16 consultas por chunk contra um cache
longo). Em 64K+ a extrapolação já passa de 2 000 s.

O que isso significa para o alvo da noite: os números de 128K do `README`/baseline são de
cache **sintético** (`bench --start-pos`), isto é, medem decode com um KV pré-semeado, não
um prefill real de 131K tokens. A 131K, hoje, o motor **não foi medido em texto real** por
nenhuma frente — este é o número que falta para a manhã.

---

### 5.3 `timeout` fora do `gpu-lock.sh` queima o orçamento esperando a fila (medido, com custo)

- **O contrato manda** `timeout 900 ./scripts/gpu-lock.sh <cmd>`, mas com sete frentes na
  mesma placa o `flock` espera. O `timeout` começa a contar **antes** do lock, então os 900 s
  são consumidos na fila: a minha primeira corrida do `compare_ppl.sh` de 512 morreu com
  **RC=124 no meio do prefill** depois de ~11 min de espera (log `/tmp/longctx-ppl512.log`:
  `RC=124` com a linha "== this engine:" já impressa e o motor ainda rodando).
- **Correção usada nesta frente**: `./scripts/gpu-lock.sh timeout 900 <cmd>` — o timeout passa
  a valer só para o trabalho, e a espera de fila fica limitada pelo `flock -w 3600` do próprio
  wrapper (que já existe para isso). Todos os comandos deste diário a partir de §4 usam essa
  forma. **Vale para as outras frentes**: quem rodar `timeout` por fora perde a corrida inteira
  quando a placa está ocupada.

### 5.4 O que **não** foi medido, com o motivo e a receita

- **Diff de nó (Qcur/Kcur pós-RoPE) numa posição longa** contra o dump do llama.cpp: é o que
  fecharia o item 1 do briefing com nó em vez de argmax. O dump só dá os valores de um grafo
  cujo último *ubatch* tem 1 token (`scripts/capture_oracle.sh` usa `UB=1` por isso), então são
  **N grafos de 1 token**; a 16 385 tokens isso é 16 385 decodes per-token (CPU ~3 tok/s =
  1,5 h; Vulkan ~15-25 tok/s = 11-18 min, acima do `timeout 900`) e o dump com filtro de
  tensores fica pequeno (~20 nós/grafo), mas o tempo não cabe na noite.
  Receita para a manhã: `UB=1 NGL=99` + filtro de nomes (`^Qcur$ ^Kcur$ ^attn_pregate$`) num
  prompt de ~16K, e então `GRAPH_LAST_TOKEN=1 ./build/check-graph-gpu <modelo> <dump> -`.
  Com `-ub 16` (barato) o último grafo tem 16 tokens e os "últimos 3 valores" do nó passam a
  ser do token 15, não do último — por isso `UB=1` é obrigatório.
- **Qualidade a 64K/131K**: fora do alcance por tempo (§5.2). O que existe hoje a 131K são
  medidas de *decode* com cache sintético (`bench --start-pos`), não texto real.

OBS de merge: `scripts/gpu-lock.sh` **não** está nos meus commits — a correção de
reentrância que o coordenador aplicou nos 5 worktrees está no meu working tree como alteração
não commitada (`if [ "${GPU_LOCK_HELD:-0}" = 1 ]; then exec "$@"; fi`). Meus commits são
`tests/` + `docs/` + `CMakeLists.txt` apenas (regra do briefing).

### 5.5 O lock da GPU ficou preso num auto-deadlock de outra frente (03:00)

Árvore medida às 03:00 (`pstree -p 27622`):
`flock(27622, segura) → bash /tmp/matrix-mtp.sh → timeout 3600 /tmp/mtp-matrix.sh → timeout 900
./scripts/gpu-lock.sh ./build/rdna4-infer run … → flock(30740), ESPERANDO o lock que o próprio
avô segura`. É a armadilha que o cabeçalho do `scripts/gpu-lock.sh` descreve (flock não é
reentrante entre processos; um script que trava por dentro de uma corrida travada espera o
próprio pai até o timeout). Consequência: ~2,25 h de espera improdutiva (9 configs × 900 s),
VRAM em 198 MB (nenhum kernel rodando) e **todas as frentes paradas atrás do lock**.
Reportado ao coordenador (não matei processo de outra frente, regra §1.5). Este é o motivo de
todo comando desta frente entre 02:43 e o fim do bloqueio aparecer como "esperando" no diário.

## 6. Qualidade vs contexto: PPL por posição, motor vs llama.cpp

- **Referência**: `scripts/compare_ppl.sh` (mesma tiling do `llama-perplexity --ppl-stride`:
  janela = ctx + stride/2, posições pontuadas `[window-stride-1, window-1)`),
  `docs/medicoes-m5.md:64-92` (baseline a ctx 512: pior chunk 0,250 %, pior posição 0,458 nats).
- **Hipótese**: o desvio motor↔llama.cpp **não** cresce com o contexto; ele é propriedade
  do motor (ordem de redução no matvec/atenção, ~0,1-0,25 %), não do comprimento. A
  degradação de PPL *absoluta* com o contexto é do modelo/quantização e aparece nos dois
  motores juntos.
- **Comandos**: `./scripts/gpu-lock.sh timeout 900 env CTX=<n> STRIDE=<n> ./scripts/compare_ppl.sh <modelo> <chunks>`
  e, para os contextos em que motor+referência não cabem em 900 s juntos, os dois lados em
  comandos separados (`ppl --nll-out` e `ORACLE_NLL_OUT=... oracle-next-token <ids>`), com a
  comparação feita por posição (o script destrutivo do `compare_ppl.sh` apaga os NLLs
  por posição; aqui eles são guardados para medir o desvio *por faixa de posição*).
- **Fidelidade fim-a-fim em posição longa** (mesmo ids, prefill em lote nos dois lados):
  `run -m <modelo> -p "<texto>" -c <ctx> -n 1 -v [--temp 1 --top-k 5 ...]` no motor e
  `ORACLE_NGL=99 ORACLE_NCTX=<ctx> ORACLE_NBATCH=<ctx> ORACLE_NUBATCH=512 ./build/oracle-next-token <modelo> <ids>`
  na referência; compara argmax e logits no topo (o `run -v` imprime os candidatos crus
  quando `--temp 1.0 --top-k 5`, `src/backend/sampler.cpp:82-136`).
- **Resultado**: §6.1 (PPL) e §6.2 (posição longa).

### 6.1 PPL por posição — resultado medido

**Tabela consolidada (mesma metodologia em todos os pontos: mesma janela e mesmos ids nos
dois motores, desvio por chunk e por posição; o critério do script é 1 % por chunk):**

| ctx (janela) | chunks | PPL motor | PPL llama.cpp | **pior desvio de chunk** | pior \|dNLL\| de uma posição |
|---|---|---|---|---|---|
| 512 (768) | 6 | 3,8231…7,7695 | 3,8160…7,7698 | **0,185 %** | 0,458 |
| 2 048 (3 072) | 2 | 5,9218 / 8,3859 | 5,9222 / 8,3722 | **0,163 %** | 0,434 |
| 4 096 (6 144) | 1 | 7,3900 | 7,3960 | **0,081 %** | 0,367 |
| 8 192 (10 240) | 1 | (job B, na fila) | | | |
| 14 336 | 1 | 3,9792 | 3,9835 | **0,108 %** | 0,929 |

- **O desvio do motor NÃO cresce com o contexto**: 0,185 % (512) → 0,163 % (2K) → **0,081 %
  (4K)** → 0,108 % (14K). Se houvesse um erro de posição/limite/overflow no caminho longo, este
  número subiria com o contexto; ele fica plano e *cai* — e o pior |dNLL| de uma posição
  acompanha (0,458 → 0,434 → 0,367 até 4K).
- (8 192: comando na fila; o valor entra aqui.)
- A cauda de uma posição isolada fica em 0,43-0,46 nats até 2K e 0,93 a 14K.

**Ponto de 14K (o mais longo que a noite alcança em PPL)**: janela única de **14 336 tokens**
(1 chunk), posições pontuadas **6 144-14 335** (8 192 posições, cada uma com 6 144 a 14 336
tokens de contexto), KV f16 nos dois lados, corpus wikitext-2-raw, mesmo id stream.
Comandos: motor `ppl --ctx-size 10240 --stride 8192 --chunks 1` (**505,5 s = 28,36 tok/s**);
referência `ORACLE_NLL_OUT` com os mesmos 14 336 ids (per-token, ~23 min).

| | motor | llama.cpp | desvio |
|---|---|---|---|
| PPL da janela | **3,9792** | 3,9835 | **0,108 %** |
| NLL médio | 1,3811 | 1,3822 | −0,00108 nats |
| pior posição \|dNLL\| | — | — | 0,929 nats |

**dNLL médio por faixa de posição** (é isto que responde "o motor piora com o contexto?"):

| posições | contexto | n | dNLL médio | max \|dNLL\| |
|---|---|---|---|---|
| 6 144-8 191 | ~6-8K | 2 048 | −0,00173 | 0,297 |
| 8 192-10 239 | ~8-10K | 2 048 | −0,00084 | 0,650 |
| 10 240-12 287 | ~10-12K | 2 048 | −0,00098 | 0,929 |
| 12 288-14 335 | ~12-14K | 2 048 | −0,00078 | 0,523 |

- **O desvio médio NÃO cresce com a posição** (fica em ~1e-3 nats, alternando de sinal, de 6K
  a 14K): não há sinal de falha do motor com o contexto. E o motor fica *ligeiramente melhor*
  (−0,001 nats) que a referência nesta janela.
- **Honestidade sobre a cauda**: o pior |dNLL| **de uma posição isolada** dobra de 0,458 (ctx
  512, `docs/medicoes-m5.md`) para **0,929** (ctx 14K). Média plana, cauda 2× maior — coerente
  com "matmul em ordem de redução diferente" num softmax de 248 320 vias, não com um erro
  sistemático que cresce com o contexto.

**Comando**: `./scripts/gpu-lock.sh timeout 900 env CTX=512 STRIDE=512 ./scripts/compare_ppl.sh <IQ3_S> 6`
(janela 768 tokens por chunk, 512 posições pontuadas por chunk, corpus wikitext-2-raw;
janela: `mem_info_vram_used` 12,9 GB no início — o lock foi esperado dentro do comando, ver §5.3).

| chunk | PPL motor | PPL llama.cpp | desvio rel | max \|dNLL\| | n |
|---|---|---|---|---|---|
| 0 | 3,8231 | 3,8160 | 0,185 % | 0,1497 | 512 |
| 1 | 8,7500 | 8,7591 | 0,105 % | 0,2379 | 512 |
| 2 | 7,2332 | 7,2326 | 0,008 % | 0,2775 | 512 |
| 3 | 6,3395 | 6,3407 | 0,018 % | 0,4581 | 512 |
| 4 | 7,2493 | 7,2422 | 0,097 % | 0,1566 | 512 |
| 5 | 7,7695 | 7,7698 | 0,004 % | 0,2013 | 512 |

- **pior chunk: 0,185 %** (baseline `docs/medicoes-m5.md` com 10 chunks: 0,250 %);
  pior posição 0,4581 nats (baseline: 0,458). Os números de 0 a 5 são **idênticos** aos do
  M5 — o harness e o binário deste worktree reproduzem o baseline dígito a dígito.
- `compare-ppl: OK` (o critério do script é 1 % por chunk).

### 6.2 Posição longa de verdade: motor vs llama.cpp no MESMO ids

**Sensibilidade do PRÓPRIO caminho da referência** (mesmo prompt, `ORACLE_NUBATCH` 512 vs 16 —
llama.cpp não é bit-idêntico entre os dois caminhos de MUL_MAT):

| top-5 na posição 17 638 | ub=512 | ub=16 | diferença |
|---|---|---|---|
| 16 | 16,8713 | 16,8158 | 0,0555 |
| 17 | 16,2301 | 16,1717 | 0,0584 |
| 18 | 15,6146 | 15,5248 | 0,0898 |
| 20 | 15,3875 | 15,3118 | 0,0757 |
| 19 | 15,2302 | 15,1482 | 0,0820 |

- **A discordância nossa-vs-referência (espalhamento ±0,018 depois de remover o deslocamento
  uniforme) é MENOR que a discordância da referência com ela mesma entre dois ubatchs
  (0,056-0,090).** Ou seja: em 17 639 tokens de contexto, o motor está dentro do ruído
  caminho-a-caminho do llama.cpp — não há sinal de erro de posição longa.


**Comando (motor)**: `./scripts/gpu-lock.sh timeout 850 ./build/rdna4-infer run -m <modelo>
-p "$(cat /tmp/prompt16k.txt)" --ctx-size 18048 -n 1 -v --temp 1.0 --top-k 5 --top-p 1.0
--min-p 0.0 --repeat-penalty 1.0 --seed 1 --cache-type-k f16 --cache-type-v f16`
(prompt de wikitext-2 medido pelo próprio tokenizer: **17 639 tokens**; o `-v` imprime os 5
candidatos crus, `src/backend/sampler.cpp:82-136`, e a lista de ids que a referência recebe).
**Comando (referência)**: `ORACLE_NGL=99 ORACLE_NCTX=18048 ORACLE_NBATCH=18048
ORACLE_NUBATCH=512 ./build/oracle-next-token <modelo> $(cat /tmp/lc-ids5-16k.txt)` (e a mesma
corrida com `ORACLE_NUBATCH=16` para medir a sensibilidade do próprio caminho da referência).

Medido até agora (o motor):
- **prefill em lote: 17 639 tokens em 273,54 s = 64,48 tok/s** (medido por mim, `--ctx-size
  18048`; o valor de 74,90 tok/s que a frente de medições mediu era a ctx 4096 — a diferença é
  a atenção, que cresce com o contexto). Extrapolando com a curva medida: ~25,7K ≈ 450 s,
  **131K ≈ 2 000 s (33 min)**, coerente com o teto de tempo de §5.2.
- decode do 1 token seguinte: 37,6 ms (26,1 tok/s) a 17 639 de contexto.
- **top-5 do motor na posição 17 638**: `16(16.786) 17(16.137) 18(15.497) 20(15.301) 19(15.115)`
  (o prompt termina numa sequência numérica; o modelo continua a lista).
- **Referência com o mesmo ids** (llama.cpp Vulkan, `ORACLE_NUBATCH=512`, mesmo id stream):

| top-5 (posição 17 638) | id | logit |
|---|---|---|
| motor | **16** | 16,786 |
| motor | 17 | 16,137 |
| motor | 18 | 15,497 |
| motor | 20 | 15,301 |
| motor | 19 | 15,115 |
| llama.cpp | **16** | 16,8713 |
| llama.cpp | 17 | 16,2301 |
| llama.cpp | 18 | 15,6146 |
| llama.cpp | 20 | 15,3875 |
| llama.cpp | 19 | 15,2302 |

- **Os cinco ids são os mesmos, na mesma ordem** (argmax 16 nos dois), em 17 639 tokens de
  contexto. |Δlogit| = 0,085-0,118; desses, **−0,0995 é um deslocamento uniforme** (o softmax é
  invariante a isso: não muda probabilidade nenhuma) e o que sobra é um espalhamento de
  **−0,018 a +0,014** contra gaps de 0,157-0,641 entre os cinco primeiros. Ou seja: a 17,6K de
  contexto o motor reproduz a distribuição da referência com erro relativo de ~2-3 % nos
  candidatos do topo — e acerta o argmax.
- Isto é a evidência fim-a-fim de posição longa que o briefing pedia no item 1 (no nível de
  logits em vez de nó; o diff de nó está bloqueado por tempo, §5.4).

## 7. Estado no fim da noite (o que rodou, o que não rodou)

**Rodou e está medido neste diário** (todos com comando exato, janela e piso de ruído):
- `check-rope-long-gpu` (**gate novo**): OK, 11 posições até 262 143, controle NEOX = 0,0 (§2).
- `bench-attn-gpu` f16: varredura completa 2/4/8/16 splits × 8/16/32 warps em 4K/16K/64K/131K
  (§3.1); q4_0 e q8_0 em 64K/131K (§3.3); protótipo C (GQA agrupado) nos três tipos e nos
  quatro contextos, com VGPR/derrame de cada kernel (§4.1-4.3).
- `bench-attn-gpu --splits 8..48`: política de splits a 131K q4_0/q8_0/f16 (§3.5).
- `bench-phases-gpu --level 2`: decomposição do passo no motor real a 16K/64K/131K (atenção
  12,1 % / 33,6 % / **54,2 %**; matvec constante em 24 ms/token; banda emitida 1430/1561/378 GB/s)
  (§3.2).
- `bench` com cache sintético (`--start-pos`): decode a 64K q8_0/f16 e 131K q4_0 com
  VRAM/GTT (§3.4).
- `compare_ppl.sh` a 512 (6 chunks) e 2 048 (2 chunks); janela única de 14 336 tokens com o
  desvio por faixa de posição (§6.1).
- Sondagem de posição longa a **17 639 tokens**: top-5 do motor vs llama.cpp (mesmos ids, mesma
  ordem, argmax igual) + a sensibilidade do próprio caminho da referência (ub512 vs ub16) (§6.2).
- `compare_llama_greedy.sh 16`: **IDS MATCH** (16 tokens idênticos) — a extensão do
  `oracle-next-token` não mudou o caminho default (job K).
- `check-rope-gpu` e `compare_ppl.sh` rodaram também na forma que já existia (inalterada).

**Ficou na fila quando a noite acabou** (com o motivo, para a manhã):
- Pontos de PPL a 4 096 e 8 192 e a sondagem a 25 742 tokens: comandos prontos, esperando a
  placa (a fila do lock teve blocos de 30-50 min de outras frentes; ver §5.3, §5.5).
- `check_all.sh --quick` no meu worktree: os meus commits são `tests/` + `docs/` +
  `CMakeLists.txt` e os gates afetados por eles foram rodados individualmente (acima); a
  bateria completa fica para o coordenador no merge.

**Não é possível nesta noite, com o motivo e a receita**:
- **Qualidade/decodificação a 131K com texto real**: o prefill em lote medido é **64,5 tok/s a
  17,6K** (§6.2) ⇒ ~2 000 s (33 min) só para encher o KV de 131K, fora do `timeout 900`; o
  `ppl` do motor é 1 forward por token (~1,5-1,8 h a 131K). Receita: um caminho `ppl` com
  logits por token em **lote** (hoje `Graph::forward_batch` devolve só o último token) — o
  matvec em lote mede 74,9 tok/s a 4K e o custo por token cai ~10×.
- **Diff de nó (Qcur/Kcur) em posição longa**: o dump do eval-callback só é comparável com
  `UB=1` (16 385 decodes de 1 token) — receita exata em §5.4.
