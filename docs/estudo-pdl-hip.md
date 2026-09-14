# PDL no HIP/ROCm: não existe equivalente (inspeção de headers, sem GPU)

Toolchain inspecionado nesta máquina: `/opt/rocm/.info/version` = **7.2.4**,
`/opt/rocm/bin/amdclang++ --version` = **AMD clang version 22.0.0git**, alvo de compilação
`gfx1201`. Nada foi executado na GPU: são greps em `/opt/rocm`, sondas de compilação de 5
linhas em `/tmp/pdlprobe/` e o assembler `llvm-mc` sobre o ISA. O lado CUDA foi lido do ninfer
em `/tmp/ninfer` HEAD `d492968`. Nenhum arquivo do motor ou do ninfer foi modificado.

## TL;DR

1. **Veredito: não portável.** O ROCm 7.2.4 não declara nenhuma das três peças do PDL:
   `hipTriggerProgrammaticLaunchCompletion`, `hipGridDependencySynchronize` e
   `hipLaunchAttributeProgrammaticStreamSerialization`. As três **não compilam**.
2. O que existe no header é compatibilidade de *fonte* com o CUDA, não mecanismo: as macros
   `hipGraphKernelNodePortProgrammatic` e `hipGraphDependencyTypeProgrammatic`
   (`/opt/rocm/include/hip/hip_runtime_api.h:1941,1945`) e mais nada — a única API capaz de
   *criar* uma aresta programática (`hipStreamBeginCaptureToGraph`, `:7946`) diz na própria
   documentação que `dependencyData` **"is currently not supported and has to be passed as
   nullptr"** (`:7941`), e `hipGraphAddDependencies` (`:8061`) nem tem esse parâmetro.
3. O ISA gfx1201 não tem instrução para isso: `llvm-mc --triple=amdgcn -mcpu=gfx1201` rejeita
   `s_griddepcontrol` e quatro variantes. Os nomes `GRIDDEPCONTROL_WAIT` /
   `GRIDDEPCONTROL_LAUNCH_DEPENDENTS` que aparecem no toolchain são do **NVPTX**
   (`llvm.nvvm.griddepcontrol.wait` em `/opt/rocm/lib/llvm/bin/opt`).
4. **PDL ataca outra coisa do que o nosso custo.** PDL esconde o *tail/head* na GPU (latência
   de rampa entre kernels dependentes); os nossos ~4,0 ms/token são enfileiramento na **CPU**
   (2,253 µs × 1 940), que o PDL não remove — quem remove é replay de grafo, já medido em
   1,06× = 1,38 ms/token.
5. **Fechar o item como não portável.** O mais próximo verificado é (a) replay de grafo HIP e
   (b) o sinalizador por kernel com `__threadfence()` + spin do consumidor, que é a mesma ideia
   feita à mão. A parte *portável* da ideia do PDL — graduar quais arestas toleram começo
   antecipado — continua aberta e não depende de hardware nenhum.

## 1. O mecanismo do CUDA, exatamente

`/tmp/ninfer/src/core/pdl.cuh` é um wrapper de 45 linhas com três peças:

**Lado do consumidor (host).** `launch_dependent` (`pdl.cuh:22-36`) monta um
`cudaLaunchConfig_t` com um único atributo: `cudaLaunchAttributeProgrammaticStreamSerialization`
(`pdl.cuh:24`) com `val.programmaticStreamSerializationAllowed = 1` (`pdl.cuh:25`), e lança com
`cudaLaunchKernelEx` (`pdl.cuh:35`). É o **consumidor** que carrega o atributo — o produtor é
lançado normalmente.

**Lado do produtor (device).** `trigger_dependents()` (`pdl.cuh:40`) chama
`cudaTriggerProgrammaticLaunchCompletion()`. O comentário do próprio arquivo diz o contrato
(`pdl.cuh:38-39`): *"Every producer CTA must call this at least once or exit. This enables
dependent scheduling but does not make producer writes visible to the consumer."* Ou seja: o
gatilho **só** libera o escalonamento; ele não publica memória.

**Lado do consumidor (device).** `wait_for_dependencies()` (`pdl.cuh:43`) chama
`cudaGridDependencySynchronize()`, e o comentário do arquivo (`pdl.cuh:42`) exige que seja
chamado *"on every consumer control path before its first access to producer-dependent data"*.
Essa espera é a barreira de memória: só depois dela os writes do produtor estão visíveis.

**Quem suporta.** A documentação da NVIDIA exige **compute capability 9.0 ou superior**
(Hopper) para haver sobreposição ([CUDA Programming Guide §4.5](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html));
em arquitetura anterior o atributo é no-op ou erro. O gatilho, por si só, *"provides no memory
visibility guarantee itself"* ([CUDA Runtime API, Execution Control](https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__EXECUTION.html)).
O uso é opt-in por aresta, com template parameters `TriggerPdl`/`JoinPdl`
(`ninfer src/ops/linear/q4/q4_rowsplit_gemm_simt.cuh:187`), gatilho cedo
(`.../q4_rowsplit_gemm_simt.cuh:208`) e espera só na junção (`.../q4_rowsplit_gemm_simt.cuh:319`).

**A contagem de sítios, conferida.** `grep -rn "pdl::" /tmp/ninfer/src --include=*.cu --include=*.cuh`
(excluindo `src/core/pdl.cuh`) dá **25 linhas de chamada em 8 arquivos**:
`pdl::launch_dependent` 9, `pdl::trigger_dependents` 8, `pdl::wait_for_dependencies` 8. Somando
as 3 definições dentro de `src/core/pdl.cuh` e o próprio header, chega-se aos **28 sítios em 9
arquivos** que `docs/estudo-ninfer.md:205` registra — as duas contagens descrevem a mesma coisa.
Por arquivo: `sparse_moe/decode/` 8, `sparse_moe/small_t/` 4,
`gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu` 4, e 2 cada em
`q4/q5_rowsplit_gemv.cuh` e `q4/q5_rowsplit_gemm_simt.cuh`, 1 em
`q4_q5_gdn_input_independent.cu`.

**A forma que precisaríamos casar:** três símbolos — um atributo de lançamento, um gatilho no
produtor e uma espera no consumidor, com a espera sendo a barreira de memória. Procuramos os
três.

## 2. O que o nosso toolchain tem

### 2.1 Os greps, e o que voltou

```
$ grep -rn "GridDependency" /opt/rocm/include/                     # 0 hits
$ grep -rn "griddepcontrol" /opt/rocm/include/                     # 0 hits
$ grep -rn "GridDependencySynchronize" /opt/rocm/                   # 0 hits (toda a árvore)
$ grep -rn "hipTriggerProgrammaticLaunchCompletion" /opt/rocm/include/
/opt/rocm/include/hip/hip_runtime_api.h:1938: * hipTriggerProgrammaticLaunchCompletion() or have terminated.
```

O único hit de `hipTriggerProgrammaticLaunchCompletion` em todo o `/opt/rocm/include` é **um
comentário**. A função não é declarada em lugar nenhum. O mesmo vale para a espera: 0 hits.

`grep -rni "programmatic" /opt/rocm/include/hip/` (todos os 6 hits):

```
/opt/rocm/include/hip/hip_runtime_api.h:1938: * hipTriggerProgrammaticLaunchCompletion() or have terminated.
/opt/rocm/include/hip/hip_runtime_api.h:1939: * It must be used with edge type hipGraphDependencyTypeProgrammatic.
/opt/rocm/include/hip/hip_runtime_api.h:1941:#define hipGraphKernelNodePortProgrammatic 1
/opt/rocm/include/hip/hip_runtime_api.h:1945:  hipGraphDependencyTypeProgrammatic = 1
/opt/rocm/include/hip/hip_runtime_api.h:1955:                  ///< hipGraphKernelNodePortDefault, hipGraphKernelNodePortProgrammatic, or
/opt/rocm/include/hip/hip_runtime_api.h:1956:                  ///< hipGraphKernelNodePortLaunchCompletion.
```

São **defines e um valor de enum de grafo**, com os comentários copiados do CUDA. O que o
header do CUDA tem a mais e o HIP não tem: as declarações das funções de device e o valor de
`cudaLaunchAttributeProgrammaticStreamSerialization`.

(O grep em todo o `/opt/rocm/include` devolve mais 4 hits em
`/opt/rocm/include/thrust/system/cuda/detail/core/triple_chevron_launch.h`, que é o backend
CUDA do thrust distribuído dentro do ROCm — não é HIP.)

### 2.2 As sondas de compilação (é o teste que decide)

Quatro arquivos de 2 linhas em `/tmp/pdlprobe/`, compilados com
`amdclang++ --offload-arch=gfx1201 --rocm-path=/opt/rocm -I/opt/rocm/include -c`:

```
########## probe_ok (baseline) ##########      [exit 0]
########## probe_trigger ##########
probe_trigger.hip:2:23: error: use of undeclared identifier 'hipTriggerProgrammaticLaunchCompletion'
########## probe_wait ##########
probe_wait.hip:2:23: error: use of undeclared identifier 'hipGridDependencySynchronize'
########## probe_attr ##########
probe_attr.hip:2:23: error: use of undeclared identifier 'hipLaunchAttributeProgrammaticStreamSerialization'
```

O baseline compila e produz bundle de offload
`probe_ok.o.0.hipv4-amdgcn-amd-amdhsa--gfx1201` (`llvm-objdump --offloading probe_ok.o`), então
o caminho de compilação HIP para gfx1201 está funcional — as três falhas são ausência de
símbolo, não ambiente quebrado. Reprodução:

```hip
#include <hip/hip_runtime.h>
__global__ void k() { hipTriggerProgrammaticLaunchCompletion(); }
```

### 2.3 Os atributos de lançamento que existem

`/opt/rocm/include/hip/hip_runtime_api.h:1570-1578`:

```c
typedef enum hipLaunchAttributeID {
  hipLaunchAttributeAccessPolicyWindow = 1,     ///< Valid for Streams, graph nodes, launches
  hipLaunchAttributeCooperative = 2,            ///< Valid for graph nodes, launches
  hipLaunchAttributeSynchronizationPolicy = 3,  ///< Valid for streams
  hipLaunchAttributePriority = 8,               ///< Valid for graph node, streams, launches
  hipLaunchAttributeMemSyncDomainMap = 9,       ///< Valid for streams, graph nodes, launches
  hipLaunchAttributeMemSyncDomain = 10,         ///< Valid for streams, graph nodes, launches
  hipLaunchAttributeMax
} hipLaunchAttributeID;
```

**Seis valores, nenhum de PDL.** A união de valores (`:1584-1599`) tem exatamente seis membros
(`accessPolicyWindow`, `cooperative`, `priority`, `syncPolicy`, `memSyncDomainMap`,
`memSyncDomain`) — não há campo tipo `programmaticStreamSerializationAllowed` para preencher.
`hipLaunchKernelEx` existe (macro em `:10227`, `hipLaunchKernelExC` em `:6774`) e é o análogo
estrutural de `cudaLaunchKernelEx` — mas sem o atributo, não há o que passar.

### 2.4 O grafo: as macros existem, a API para usá-las não

`hipGraphAddDependencies` (`:8061`) é a única função de aresta, e a assinatura é a v1 do CUDA:

```c
hipError_t hipGraphAddDependencies(hipGraph_t graph, const hipGraphNode_t* from,
                                   const hipGraphNode_t* to, size_t numDependencies);
```

Não há parâmetro de `hipGraphEdgeData` e não há `_v2` (`grep -n "_v2\b"` em todo o header só
devolve `hipMemPrefetchAsync_v2`, `hipMemAdvise_v2` e `hipStreamGetCaptureInfo_v2`). O CUDA
anexa a aresta programática por `cudaGraphAddDependencies_v2`/`cudaGraphAddNode` com edge data;
o HIP não tem esse caminho.

O único lugar do header que aceita `hipGraphEdgeData` é `hipStreamBeginCaptureToGraph`
(`:7946-7949`), e a documentação dele (`:7939-7943`) diz, literalmente:

```
* @warning param "const hipGraphEdgeData* dependencyData" is currently not supported and has to be
passed as nullptr.
```

`hipGraphAddNode` (`:8275`) recebe `hipGraphNodeParams` (`:1905-1924`), cujo union tem
`kernel`/`memset`/`memcpy`/`mem alloc`/`mem free` — nenhum campo de aresta. Ou seja: mesmo que o
runtime honrasse a porta programática, **não existe chamada capaz de criá-la**.

### 2.5 O runtime e o compilador

```
$ nm -D --defined-only /opt/rocm/lib/libamdhip64.so | grep -ci programmatic      # 0
$ strings /opt/rocm/lib/libamdhip64.so | grep -i programmatic                    # 2 linhas:
hipGraphDependencyTypeProgrammatic
hipGraphDependencyTypeProgrammatic
$ strings /opt/rocm/lib/libhsa-runtime64.so | grep -ci "griddep\|programmatic"   # 0
```

Duas ocorrências de uma **string de nome de enum** na lib HIP e zero símbolos exportados: é
tabela de nome/validação, não implementação. O ROCr não menciona nada disso.

Compilador:

```
$ /opt/rocm/bin/amdclang++ -dM -E -x hip /dev/null | wc -l        # 5834 macros
$ ... | grep -i grid                                               # 0 hits
$ strings /opt/rocm/lib/llvm/bin/clang | grep -o "__builtin_amdgcn_[a-z_0-9]*" | sort -u | wc -l
669
$ ... | grep -ci "dep\|grid_dep"                                   # 0
```

**669 builtins `__builtin_amdgcn_*` e nenhum de dependência de grade.** Os que existem nessa
vizinhança são de barreira de *workgroup* e de escalonamento
(`__builtin_amdgcn_s_barrier_signal`, `_wait`, `s_sched_barrier`), não de grade.

### 2.6 O ISA do gfx1201

```
$ printf 's_endpgm\n' | llvm-mc --triple=amdgcn -mcpu=gfx1201 -show-encoding
	s_endpgm                                ; encoding: [0x00,0x00,0xb0,0xbf]
```

O assembler está funcional para gfx1201. Com ele, cinco candidatos a mnemônico de PDL:

```
s_griddepcontrol 0                  => error: invalid instruction
s_grid_dependency_control 0         => error: invalid instruction
s_dependent_launch 0                => error: invalid instruction
s_launch_dependents                 => error: invalid instruction
s_griddepcontrol_launch_dependents  => error: invalid instruction
```

E a origem dos nomes que *aparecem* no toolchain (`strings /opt/rocm/lib/llvm/bin/opt`):

```
griddepcontrol.wait;
griddepcontrol.launch_dependents;
llvm.nvvm.griddepcontrol.wait
llvm.nvvm.griddepcontrol.launch.dependents
GRIDDEPCONTROL_WAIT
GRIDDEPCONTROL_LAUNCH_DEPENDENTS
```

O namespace de intrinsic é `llvm.nvvm.*` — **NVPTX**. É o mesmo LLVM (o registro de alvos do
`llvm-mc` inclui `nvptx`/`nvptx64`), então a string aparecer no binário não diz nada sobre
AMDGPU. O que diz é a rejeição acima.

## 3. As alternativas do gfx12

### 3.1 Verificado nos headers (existe e é usável hoje)

**(a) Replay de grafo HIP.** `hipStreamBeginCapture`, `hipGraphInstantiate`, `hipGraphLaunch`
existem e já estão em uso no nosso próprio benchmark
(`tests/bench_matvec_shapes_gpu.hip:402,414,420`). O motor **não** usa grafo HIP em lugar
nenhum: `grep -rn "hipGraph\|StreamCapture" src include kernels` dá 0 hits; o
`include/rdna4/graph.cuh` é o grafo do forward escrito à mão, com lançamentos `<<<>>>`
(`include/rdna4/attn.cuh:60,193,417,652,656`). O que ele compra, medido: **1,06× = 1,38
ms/token** para os mesmos 497 kernels (`docs/rocm-estudo.md:291,356`), e a sequência inteira de
lançamentos replayada como grafo comprou *"only 1.06× (1.4 ms)"* (`README.md:259-260`).

**(b) Atributos de lançamento disponíveis.** `hipLaunchKernelEx` + `hipLaunchAttributePriority`
(`hip_runtime_api.h:1574`), `hipLaunchAttributeCooperative` (`:1572`),
`AccessPolicyWindow` (`:1571`) e `MemSyncDomain` (`:1575`). Prioridade de fila existe também no
ROCr: `hsa_amd_queue_set_priority` (`/opt/rocm/include/hsa/hsa_ext_amd.h:2858`, enum em `:2827`).

**(c) Estrutura de dependência semântica no grafo.** Como o HIP graph é um DAG comum, arestas
que **não** existem já permitem sobreposição: dois ramos independentes de um grafo HIP podem
ser co-residentes se os recursos permitirem. Isso é o que `docs/estudo-ninfer.md:499,508`
descreve como a parte portável da ideia (opt-in por aresta) e é o que o `graph.cuh` já consegue
expressar hoje só mudando a topologia.

**(d) A mesmo-ideia-à-mão.** Sinalizador por kernel: o produtor escreve uma flag com
`__threadfence()` depois do último store útil, o consumidor gira nela antes do primeiro load.
Não depende de ISA nenhuma (store atômico com semântica de release/acquire existe em gfx12), e
já está registrado como o candidato em `docs/estudo-ninfer.md:517-520`.

### 3.2 `INFERIDO` — o que eu não pude verificar sem GPU

- **`INFERIDO`: dois kernels do mesmo stream não são co-residentes no gfx12.** Filas AQL são
  in-order e o HIP marca o bit de barreira por dispatch, então a sobreposição pediria streams
  separados ou pacotes sem barreira. Não achei isso em header nenhum — `grep -rni "interleave"`
  em `/opt/rocm/include/hsa/` dá **0 hits**, e não há opção de "sem barreira" exposta no HIP
  público. **Confirmaria** com uma medição de tempo sobreposto de dois kernels de longa duração
  em streams distintos contra o mesmo stream.
- **`INFERIDO`: as macros de porta do grafo são inertes no gfx12.** O runtime contém a string do
  enum, mas nenhuma API aceita o edge data, então isso é inobservável de qualquer forma. Nota: a
  porta `hipGraphKernelNodePortLaunchCompletion` ("all blocks of the kernel have begun
  execution", `:1934`) é o análogo CUDA-sem-gatilho; ela também só é alcançável por edge data.
  **Confirmaria** com `hipGraphDebugDotPrint` sobre um grafo com edge data — impossível hoje,
  já que `hipStreamBeginCaptureToGraph` rejeita `dependencyData != nullptr`.
- **`INFERIDO`: o AMDGPU backend não tem *nenhum* mecanismo de "começo de grade dependente".**
  Baseado em: `llvm-mc` rejeita os mnemônicos, 0 builtins, 0 menções em header, 0 símbolos no
  runtime. Não li o `.td` do backend AMDGPU (não está nesta máquina), então não afirmo a
  ausência no *backend*, só a ausência de superfície utilizável.
- **`INFERIDO`: o custo de qualquer emulação por spin em gfx1201.** Não medi. O risco é o
  consumidor ocupar CUs girando e ser escalonado antes do produtor terminar, criando deadlock
  em cenário de grade cheia — exatamente o que o PDL evita porque o hardware conhece a
  dependência entre as duas grades (o `docs/estudo-ninfer.md:900` já propõe contar as arestas
  antes de prototipar).

## 4. Veredito e o próximo passo

**Não portável.** O PDL do CUDA precisa de três coisas; o ROCm 7.2.4 tem zero das três
verificadas por compilação (§2.2), e o ISA gfx1201 não tem a instrução (§2.6). O que sobrou no
header (§2.1, §2.4) são nomes de compatibilidade sem caminho de uso: o gatilho de device não é
declarado, o atributo de lançamento não existe no enum, e a única API que aceitaria uma aresta
programática documenta que aquele parâmetro não é suportado. O item deve ser **fechado como não
portável** em `docs/estudo-ninfer.md:205` e em `docs/backlog-noite.md:433`, com este documento
como evidência. A linha de `docs/estudo-ninfer.md:205` pedia exatamente esta confirmação
(`grep -rn "ProgrammaticLaunch\|griddepcontrol" /opt/rocm/include`) e a resposta está em §2.1 e
§2.2.

**Uma correção de premissa, porque muda o que fazer depois.** O PDL sobrepõe o *tail* de um
kernel com o *head* do seguinte — ele ataca latência de rampa na GPU. O nosso custo medido é
**CPU-side**: 2,253 µs por kernel vazio **enfileirado** e 916 lançamentos = 2,06 ms por chunk
(`docs/estudo-prefill-c-nosso.md:141,446`), contra ~4,0 ms/token para as 1 940 partidas
(`README.md:258`). Um PDL hipotético não tiraria esses 4 ms; quem ataca enfileiramento é replay
de grafo, e ele **já foi medido**: 1,06× = 1,38 ms/token no passe de 497 matvecs
(`docs/rocm-estudo.md:291`) e 1,06× = 1,4 ms na sequência inteira (`README.md:259-260`). O
`docs/estudo-prefill-c-nosso.md:450-455` mede o mesmo por outro caminho e conclui **0,81
ms/token** de "launch tax" recuperável, com o veredito de que *"o despacho é ruído (<2,5 %)"* —
o número canônico lá é **0,129 ms/token = 1,58 %** (`:629`).

**O número "0,76 ms/token" que motivou esta nota não está no repositório.** Os únicos `0,76` do
nosso `docs/` são razões de ILP na autotuning do matvec (`docs/autotuning-gfx1201.md:146,229`,
replicadas em `docs/backlog-noite.md:210`, todas `0,76x`). Os números de replay de grafo que
existem são **1,38 ms/token** (`docs/rocm-estudo.md:291`) e **0,81 ms/token**
(`docs/estudo-prefill-c-nosso.md:450`). Qualquer decisão que pendure no `0,76` deveria usar um
dos dois medidos.

**O mais próximo que existe, e o que compra:**

1. **Replay de grafo HIP** — ataca a mesma conta (custo por lançamento), está medido em 1,06×, e
   o motor ainda não usa: `grep -rn "hipGraph" src include kernels` = 0 hits, só
   `tests/bench_matvec_shapes_gpu.hip`. Compra ~1,4 ms/token se o ganho do bench sobreviver
   dentro do motor — o `README.md:259-260` avisa que a maior parte dos lançamentos já se
   sobrepõe a kernels reais. É o único item aqui com número medido.
2. **Topologia do grafo** — as arestas que hoje existem por hábito e não por dependência de
   dados são sobreposição grátis, sem atributo nenhum. É a parte *portável* do PDL e a única que
   não exige hardware. O passo barato é o que `docs/estudo-ninfer.md:900` já propõe: comparar,
   por kernel, os buffers escritos pelo produtor com os lidos pelo consumidor.
3. **Sinalizador + spin** — a emulação à mão. Só vale depois de (1) e (2), porque o teto dela é
   o mesmo dos dois e o risco de deadlock por grade cheia é novo.

**O que não fazer:** manter o item aberto. Não há caminho de protótipo no ROCm 7.2.4 — não
existe API para chamar, nem atributo para setar, nem instrução para emitir.
