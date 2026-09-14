# Três frentes paralelas (worktrees + branches)

Para trabalhar em três frentes ao mesmo tempo sem colisão de arquivos, cada agente
roda em um **git worktree** próprio, com branch próprio, e o merge é feito depois na
`main` (pelo agente coordenador).

| frente | worktree | branch | arquivos principais |
|---|---|---|---|
| servidor OpenAI-compatible | `../rdna4-infer-wt-server` | `feat/openai-server` | `src/server/**`, `tests/check_server*`, `docs/servidor-openai.md` |
| MTP (nextn / speculative) | `../rdna4-infer-wt-mtp` | `feat/mtp` | `include/rdna4/graph.cuh` (aditivo), `src/backend/model.*`, `tests/check_mtp_gpu.hip`, `docs/mtp.md` |
| estudo ROCm/HIP + otimizações desta placa | `../rdna4-infer-wt-rocm` | `feat/rocm-study` | kernels (`matvec.cuh`, `attn.cuh`, `kv.h`, `nn.cuh`, `dequant_row.cuh`), `docs/rocm-estudo.md` |

## Regras de convivência (para o merge ser trivial)

1. **GPU é serializada.** Toda execução que toca a GPU (teste, `bench`, `ppl`,
   ferramentas de oráculo) passa por `scripts/gpu-lock.sh <comando>`: uma execução
   mapeia 11,9-15,7 GiB de 15,9 GiB, então duas ao mesmo tempo falham com
   `hipMalloc failed`. O wrapper espera o lock e devolve o exit status do comando.
2. **`CMakeLists.txt`: só append.** Cada agente adiciona seus alvos no fim do arquivo.
3. **`src/main.hip`: contrato de 3 linhas.** A implementação de um subcomando novo vai
   em arquivo próprio (ex.: `src/server/serve.cpp`, declarado em
   `include/rdna4/cli_commands.h`); em `main.hip` só entram o `#include` e uma linha de
   dispatch antes do `return cmd_info(...)` final.
4. **Não rebasear nem mexer na `main`**: o agente commita na sua branch e para.
5. **Cada agente configura o próprio build** no seu worktree:
   `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j6`.
   O diretório `reference/` (dumps de oráculo + corpus wikitext) é um symlink para o
   repositório principal e é gitignored.
6. **Nada de commitar** `.gguf`, `reference/` ou artefatos de build.

Modelos (read-only, compartilhados):
`/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf` (11,2 GiB) e
`...-UD-IQ4_XS.gguf` (13,3 GiB).


---

# Resultado do merge (três frentes)

Todas as três frentes terminaram e foram **mergeadas na `main`** (merges `--no-ff`, com
os conflitos de `CMakeLists.txt` resolvidos mantendo os dois blocos append de cada
lado; nenhum conflito de código). O que cada uma entregou e o que ficou de fora:

| frente | commits | merge | estado |
|---|---|---|---|
| servidor OpenAI-compatible | 4 | `ea73450` | completo; 94 checagens de aceitação verdes nos dois modelos |
| ROCm/HIP + gfx1201 (estudo) | 5 | `78db0b3` | inventário completo + 1 otimização landada + política de splits (aplicada no merge) |
| MTP (NextN) | 9 | merge próprio | implementado com oráculo do llama.cpp; **ainda não compensa em velocidade** |

## O que foi verificado depois do merge (árvore final, um único build)

`check-graph-gpu` (oráculo por nó) **PASS** · `check-graph-gpu` argmax **PASS** ·
`check-mtp-gpu` **PASS (0 falhas)** · `check-kvctx-gpu 65536 q4_0` **OK** ·
`check-matvec-gpu` **OK (14 tipos)** · `check-matdequant-gpu`/`nn`/`rope`/`dequant` OK ·
`check-server-http` **70 checagens, 0 falhas** · `check-stream`/`sampler`/`chat`/
`tokenizer`/`eog`/`dtype`/`loader`/`model` OK · `check_golden_run.sh` **OK** ·
`check_attn_split.sh` **OK**.

Decode no fim do contexto, IQ3_S (árvore final): 4K f16 **26,8 tok/s**, 16K **24,3**,
64K f16 **18,9**, 64K q4_0 **17,9**, 131K q4_0 **13,0** (baseline do M5: 22,1 / 13,7 /
— / 4,2 / 2,3).

## Três descobertas que mudaram o que já estava escrito

1. **Meu briefing dos agentes tinha o comando errado do gate do grafo** (faltava
   `GRAPH_LAST_TOKEN=1`, obrigatório com dump `-ub 1`): dois agentes viram
   `check-graph-gpu` vermelho e o agente de ROCm reportou isso honestamente em vez de
   esconder. O gate sempre esteve verde com o comando correto.
2. **O commit `a830570` mente no assunto**: diz "32 warps" e a linha commitada é
   `kAttnWarpsPerBlock = 8` (o 32 foi revertido para medir baseline e não voltou).
   Corrigido em `docs/medicoes-m7.md`; com split-KV ativo, 8 warps mede melhor mesmo.
3. **O MTP tem oráculo**: o meu briefing dizia que o llama.cpp ignora o bloco 64; o
   agente mediu que o llama.cpp b10902 implementa `graph_mtp` para qwen35
   (`--spec-type draft-mtp`) e construiu o oráculo em cima disso — foi assim que a
   ordem do concat (`enorm` antes de `hnorm`, 96,9% vs 0,0% de aceitação) deixou de ser
   suposição.

## Pendências que sobraram (e de quem são)

- **Verificação em batch para o MTP** (~2,5× potencial): precisa de atenção causal
  multi-query e forward em batch no tronco — é o mesmo trabalho do prefill em batch
  (M6 passo 2), então as duas coisas devem ser feitas juntas.
- **`proj()` com `act_ready`** e fusão das cadeias pequenas (GDN, `rms_norm`+quantize,
  `kv_store_row`): descritas em `docs/rocm-estudo.md` §E, exigem `graph.cuh`.
- **Prefill em batch** continua sendo a maior lacuna de usabilidade para prompt longo.
