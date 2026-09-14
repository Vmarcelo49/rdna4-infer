# Regras da rodada noturna (2026-09-14) — leia antes de qualquer coisa

Sessão **não supervisionada** até de manhã. Sete frentes em paralelo, **uma placa de GPU**,
um coordenador que faz os merges. Este arquivo é o contrato; o seu diário
(`docs/journal-<frente>.md`) é o entregável.

## 0. Estado de partida

`main` em `e48f3c4`, tag **`noite-baseline-2026-09-14`**: árvore mergeada das 9 tarefas
anteriores, `scripts/check_all.sh` **PASS** (10 gates de CPU + 15 de GPU), números finais no
`README.md`.

Alvo da noite: **131K de contexto**, **KV `q5_0` (K) / `q4_1` (V)**, **MTP entregando ganho
real**, e o máximo de desempenho que couber no resto da noite.

**Achado que precisa ser dito primeiro, com honestidade:** o motor **não tem** `q5_0` nem
`q4_1` para o KV hoje — `include/rdna4/kv.h` só tem `f32`, `f16`, `q8_0`, `q4_0`. Então
"K em q5_0 / V em q4_1" não é um resultado deste motor: é o que a frente KV vai
*implementar e medir*. Não trate como fato estabelecido; trate como hipótese a testar.
Conta preliminar para 131K: `q5_0` = 22 B/bloco de 32 (0,6875 B/elem), `q4_1` = 20 B/bloco
(0,625 B/elem), 16 384 elementos de K e 16 384 de V por token ⇒ 21 504 B/token ⇒ **2,63 GiB**
em 131 072 tokens, contra 2,25 GiB do `q4_0/q4_0` e 4,25 GiB do `q8_0/q8_0`. Some 11,20 GiB
de pesos + ~0,5 GiB de buffers: cabe, com folga pequena. **Meça antes de afirmar.**

## 1. GPU: uma placa, uma execução por vez

1. **Todo** comando que toca a GPU passa por `scripts/gpu-lock.sh <comando>`. Sem exceção.
   O wrapper agora exporta `GPU_LOCK_HELD=1` para o filho, então os scripts que já se travam
   sozinhos (`check_golden_run.sh`, `check_attn_split.sh`, `compare_ppl.sh`,
   `compare_llama_greedy.sh`, `check_server.sh`, `check_regression.sh`) não travam duas vezes.
2. **Sempre com `timeout`**: `timeout 900 ./scripts/gpu-lock.sh ./build/...`. Um comando que
   pendura não pode pendurar a noite.
3. **Nunca** rode dois processos que carregam o modelo. Uma execução mapeia 11,9-15,7 GiB de
   15,9 GiB.
4. Antes e depois de uma medição, olhe a VRAM:
   `cat /sys/class/drm/card*/device/mem_info_vram_used`. Se ficar acima de ~200 MiB sem
   processo vivo, **avise o coordenador** em vez de assumir que está tudo bem (um processo
   morto por SIGTERM pode segurar VRAM).
5. Se um comando seu pendurar: `kill` no **seu** processo, registre no diário e siga. Nunca
   mate processo de outra frente.
6. **Janela limpa**: número medido com outra frente rodando é lixo. Anote no diário, para cada
   número, se a janela estava limpa (`mem_info_vram_used` baixo, nenhum `flock` esperando).
   Na dúvida, reporte as duas corridas.

## 2. Correção não se negocia (e regressão se reverte)

1. Antes de **qualquer** mudança de kernel: `git tag` local da sua frente
   (`git tag noite-<frente>-antes-<n>`) e commit do estado atual. Nunca sobrescreva o último
   estado bom.
2. Depois de **qualquer** mudança de kernel ou de aritmética, rode os gates que existem:
   `timeout 900 ./scripts/gpu-lock.sh ./build/check-graph-gpu <modelo> reference/oracle_prompt6_ub1_tok7_cpu.txt -`
   (precisa de `GRAPH_LAST_TOKEN=1` antes do comando),
   `./scripts/check_regression.sh` (79 s, é o gate numérico mais forte),
   `./build/check-batch-gpu`, `./scripts/check_golden_run.sh`.
3. **Regressão medida ⇒ reverta na hora**, não deixe marcado para depois. Registre no diário
   o que reverteu e a evidência.
4. **Bit-exatidão é o padrão** onde já existe (matvec em batch, prefill em batch, `proj_qq`).
   Mudança que quebra bit-exatidão **só** entra se for medida e validada por PPL/regressão,
   autorizada explicitamente no seu briefing, e com o desvio em % escrito no diário.
   Autorizado nesta noite: protótipo MMQ/prefill tiled (frente prefill) e qualquer mudança na
   política de atenção; **não** autorizado mexer na aritmética do matvec de decode sem gate.
5. PPL e equivalência com o llama.cpp são a rede de segurança de qualidade:
   `./scripts/compare_llama_greedy.sh 32` e `./scripts/compare_ppl.sh <modelo> 10`.

## 3. Método: medir, não opinar

1. **Piso de ruído antes do ganho.** Meça a mesma configuração duas vezes (o A/B intercalado
   de `bench-attn-gpu` e `bench-matvec-shapes-gpu` já existe; use `--ab/--cand`). Ganho dentro
   do piso é "nada acima do ruído" no diário, não "melhoria".
2. Aqueça com `sync` **dentro** do laço até ≥300 ms de GPU ocupada (DPM cai para SCLK baixo
   entre kernels; sem isso você mede o relógio, não o kernel).
3. **Curto é melhor que longo**: prefira `--reps 2/3` e medições de segundos. Não gaste a
   noite num único experimento.
4. Toda tabela do diário diz o comando exato que a produziu e se a janela estava limpa.
5. `rocprof`/`rocprofv3`/`omniperf` **não estão instalados** (só `rocminfo`): use eventos HIP,
   `bench --layers`, contagem de lançamentos, contadores no kernel e subtração A/B — e diga no
   diário que a ferramenta oficial não existe.

## 4. Caixa de tempo por experimento

- **Leitura/desenho**: livre (não queima GPU).
- **Implementação de um experimento**: se em ~45 min não houver nem compilação nem número,
  pare, registre e vá para o próximo.
- **Direção**: no máximo 2 tentativas fracassadas; na terceira, registre "abandonado, com
  medida" e troque de direção. Direção abandonada com número vale mais que direção meia-pronta.
- **Gates**: rode os baratos sempre; `check_all.sh --quick` antes de pedir merge.

## 5. Diário — o entregável da manhã

Cada frente mantém `docs/journal-<frente>.md` **no seu worktree**, com uma linha por
experimento, no formato:

```
## <n>. <título curto do experimento>
- **Referência**: de onde veio a ideia (arquivo:linha deste repo, paper, outro motor, URL)
- **Hipótese**: o que deveria melhorar e por quê (com número esperado, se houver)
- **Comando**: o comando exato (com o lock) e a janela (limpa? VRAM antes/depois)
- **Resultado**: números medidos, com o piso de ruído daquele harness
- **Veredito**: MANTIDO (com o ganho em %) / REVERTIDO / ABANDONADO (com o porquê)
```

Regras do diário: escrito **enquanto** acontece, não no fim; nada de "deve melhorar" sem
número; hipótese rejeitada fica com a medida que a matou; se um experimento deu resultado
ambíguo, diga que é ambíguo. O coordenador consolida tudo em `docs/journal-noite.md` (o
relatório da manhã) e faz os merges.

## 6. Git

1. Cada frente trabalha **no seu worktree**, commita **na sua branch**, e para. O merge é do
   coordenador, em série.
2. `CMakeLists.txt`: **append no fim** (é o único arquivo que colide; a resolução é manter os
   dois blocos). Prefira alvos novos a mexer nos existentes.
3. `src/main.hip`: contrato de 3 linhas (include + dispatch). Implementação de subcomando novo
   vai em arquivo próprio.
4. Nada de commitar `.gguf`, `reference/`, `build/` ou artefato de build.
5. Se precisar de código que outra frente está mexendo (`graph.cuh`, `attn.cuh`,
   `matvec.cuh`), **diga no diário** e mantenha a mudança pequena e localizada: o coordenador
   resolve o merge.

## 7. Modelos (read-only, compartilhados)

```
/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf   (12,0 GB, 15 dtypes)
/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf  (14,2 GB, 13 dtypes)
```

`reference/` é um symlink para o repo principal (dumps de oráculo + corpus wikitext) e é
gitignored.
