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
