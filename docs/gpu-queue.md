# Fila da GPU (uma placa, seis agentes)

Regras que valem para todo agente deste lote:

1. **Toda** execução que toca a GPU passa por `scripts/gpu-lock.sh <comando>` — uma
   execução mapeia 11,9-15,7 GiB de uma placa de 15,9 GiB, então duas ao mesmo tempo
   dão `hipMalloc failed` ou números sem sentido. O wrapper espera a vez e devolve o
   exit status do comando. Vale para testes, `bench`, `ppl`, ferramentas de oráculo e
   qualquer script que carregue o modelo.
2. **Não** rode GPU para tarefa que é de leitura/escrita de código: o lote tem duas
   filas de GPU (medições e autotuning) e elas precisam da placa inteira.
3. `rocprof`/`rocprofv3`/`omniperf` **não estão instalados** nesta máquina (só
   `rocminfo`). Quem precisar de profiling usa o que existe: eventos HIP, o
   `bench --layers`, os contadores de relógio via `s_memrealtime`/`SHADER_CYCLES`, ou
   mede por subtração de fases — e diz explicitamente no relatório que a ferramenta
   oficial não estava disponível.
4. Cada agente trabalha no seu worktree e commita na sua branch; o merge é feito
   depois, em série, pelo agente coordenador. Não rebasear, não mexer na `main`.

## O que aconteceu na prática (e o que mudou por causa disso)

A regra 1 valia para quem lembrasse de chamá-la, e no meio do lote alguém não lembrou: o
agente de autotuning rodou `ppl` sem `flock`, segurando **12,07 GiB** (`mem_info_vram_used`),
enquanto o agente de medições rodava seus `bench`. Consequências medidas:

- `scripts/check_golden_run.sh` do lado do autotuning deu **FAILED** em 5 casos de KV
  (`q4_0`, `f16/q8_0`, `q8_0/f16`, `f32/q4_0`) com "engine exited non-zero" e saída vazia —
  era `graph init failed: layer 29: hipMalloc failed`, ou seja **VRAM ocupada**, não o motor.
  O coordenador reproduziu a falha rodando o mesmo script fora do lock, e ela desapareceu
  rodando dentro do lock (`check-golden-run: OK`).
- As medianas das medições do outro agente ficaram 15-20% acima da melhor passada.

Conserto estrutural (não é mais uma regra que depende de disciplina): os gates que rodam GPU
**se travam sozinhos** — `check_golden_run.sh`, `compare_llama_greedy.sh`,
`check_attn_split.sh`, `compare_ppl.sh` e `check_server.sh` fazem
`exec gpu-lock.sh "$0" "$@"` quando `GPU_LOCK_HELD != 1`, e pulam o lock quando o chamador já
o tem (`flock` não é reentrante entre processos, então re-travar dentro de uma execução
travada faria o gate falhar por timeout em vez de rodar). O `scripts/check_all.sh` é quem
exporta `GPU_LOCK_HELD=1` na sua fase B, que roda a bateria inteira com **um** lock.

Lição de método que vale para o próximo lote: um gate vermelho por contenção é pior que um
gate não rodado, porque vira conclusão errada no relatório. Por isso todo número medido
neste lote precisa dizer se a janela estava limpa.
