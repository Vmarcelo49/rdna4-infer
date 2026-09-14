# Diário da frente longctx — 2026-09-14 (não supervisionada)

Frente: **"o modelo ainda se sustenta a 131K, e o nosso caminho de contexto longo é
fiel à referência?"**
Worktree `rdna4-wt-noite-longctx`, branch `feat/noite-longctx`, base `e48f3c4`
(`noite-baseline-2026-09-14` + `ea0e670` regras).
Formato: `docs/noite-regras.md` §5. Todo número diz o comando exato e a janela.

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
  ext_factor=0.0)` com as 4 seções iguais, em posições até 131 071 — e a diferença
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
  - Tamanho do erro que uma escala ausente causaria: YaRN×4 → **rel-L2 0,24 a 0,42**
    (24-42 %, em *todas* as posições, inclusive pos 0 por causa do `mscale`); linear×4 →
    0,08 na pos 1 crescendo para 0,65 a 131K.
- **Veredito**: **fórmula MANTIDA (sem mudança de código)**: o kernel está certo e a
  diferença residual em 131K é o piso de fp32, 3,6e-4 rel-L2, três/quatro ordens de
  grandeza abaixo de qualquer erro de fórmula ou de escala. **Nenhum bug de RoPE, nenhum
  scaling faltando.** O gate novo entrou com limite `rel-L2 < max(1e-7, 4e-8·pos)` (10× a
  lei medida): falha se a diferença crescer mais rápido que o orçamento de fase do fp32.

### Consequência para o resto da noite (número, não opinião)

- Em 131K, `Q`/`K` pós-RoPE **não podem** ser comparados bit-exatos com o llama.cpp: o
  próprio cálculo da fase em fp32 custa ~2,4e-4 (referência) a 4,2e-4 (nosso) de rel-L2
  contra a rotação exata. Qualquer diff de nó no nível de 1e-4 a 131K **não** é evidência
  de bug do motor.
- Para comparação: o KV `q8_0` introduz ~2e-3 relativo por elemento e o `q4_0` ~5e-2 —
  uma ordem de grandeza (q8_0) a duas (q4_0) **acima** do erro de RoPE em 131K. Ou seja:
  a 131K quem limita a qualidade é a quantização do KV, não a fase do RoPE.
