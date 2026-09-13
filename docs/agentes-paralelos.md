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
