# Build reprodutível — versões pinadas, build limpo e armadilhas

> **O que este documento é.** Uma receita de build verificada *nesta máquina* (RX 9070 XT
> `gfx1201`, CachyOS), com as versões exatas que funcionam, os comandos exatos do build
> limpo que foi executado (com tempo, saída e exit code), o que o build realmente exige do
> disco, o que instalar por família de distro, e cada armadilha de build conhecida com
> sintoma → causa → correção.
>
> **O que este documento não é.** Nada aqui foi obtido rodando o motor, testes ou
> benchmarks: não houve execução de GPU (a fila da GPU é de outro agente, `docs/gpu-queue.md`).
> Também **nada foi instalado nem modificado** na máquina — só leitura (`--version`,
> `pacman -Q`, `rocminfo`), aritmética e *builds* de CPU em `/tmp`.
>
> Data da coleta: 2026-09-13 · branch `feat/build-repro` · commit base `aa15eea` (M8).

---

## 0. Resumo

| pergunta | resposta |
|---|---|
| Versão verificada | **ROCm 7.2.4**, `amdclang++` **22.0.0git** (`rocm-llvm f58b06d`), CMake **4.4.3**, kernel **7.2.4-3-cachyos** |
| Build limpo funcionou? | **Sim** — `cmake -S . -B /tmp/build-clean -DCMAKE_BUILD_TYPE=Release && cmake --build /tmp/build-clean -j6`, **exit 0**, 27 alvos, **4 min 59,9 s** de parede (`-j6`), 433 avisos |
| O `build/` existente é necessário? | **Não.** Nenhum caminho do CMake lê de `build/`, e o diretório é *gitignored* (`.gitignore:1`). O build limpo em `/tmp` prova isso. Nesta worktree o `build/` nem existe |
| Dependências externas ao repositório | `/opt/rocm` (obrigatória, com erro claro) e um checkout+build do **llama.cpp** (obrigatório **para configurar**, mas não para rodar o motor) |
| Versão mínima de ROCm/LLVM | Piso por recurso: **gfx1201 no backend AMDGPU** (ROCm ≥ 6.4.1, externo) + o builtin `__builtin_amdgcn_s_prefetch_data`, que exige a feature **`gfx12-insts`**. Só 7.2.4/amdclang 22 foi verificado aqui |
| O build precisa de `hipcc`? | **Não** — o CMake **rejeita** o wrapper `hipcc` (mensagem literal abaixo); o compilador é o `amdclang++` |

---

## 1. Versões exatas que funcionam (medidas nesta máquina)

### 1.1 Toolchain

| item | valor | comando que capturou |
|---|---|---|
| ROCm (meta) | `7.2.4` | `cat /opt/rocm/.info/version` |
| HIP runtime | `7.2.53211-3d9ef42` | `hipcc --version` / `hipconfig --version` |
| Compilador de device | `AMD clang version 22.0.0git (/srcdest/rocm-llvm f58b06dce1f9c15707c5f808fd002e18c2accf7e)` · `InstalledDir: /opt/rocm/lib/llvm/bin` | `amdclang++ --version` |
| Compilador de host (CXX) | `GNU 16.2.1` (`/usr/bin/c++`) | saída do configure (`-- The CXX compiler identification is GNU 16.2.1`) + `g++ --version` |
| CMake | `4.4.3` | `cmake --version` |
| Make | `GNU Make 4.4.1` | `make --version` |
| LLD do ROCm (é o que linka) | `AMD LLD 22.0.0 (/srcdest/rocm-llvm f58b06d…)`, `/opt/rocm/lib/llvm/bin/ld.lld -> lld` | `/opt/rocm/lib/llvm/bin/ld.lld --version` |
| LLD do sistema (não é o usado) | `LLD 22.1.8` | `ld.lld --version` |
| Kernel | `7.2.4-3-cachyos` (CachyOS, base Arch) | `uname -r` |
| glibc | `2.44` | `ldd --version` |

**Cuidado com a confusão de LLVMs:** a máquina tem **duas** toolchains LLVM — a do sistema
(`clang`/`llvm`/`ld.lld` 22.1.8, pacotes `clang 22.1.8-2`, `llvm 22.1.8-2`) e a **do ROCm**
(`rocm-llvm 2:7.2.4-2`, versão `22.0.0git`, em `/opt/rocm/lib/llvm`). O build usa **a do ROCm**:
`/opt/rocm/bin/amdclang++` é symlink para `/opt/rocm/lib/llvm/bin/amdclang++` (`ls -l`), e o
`CMakeLists.txt:5-7` fixa isso. Reportar `ld.lld --version` (22.1.8) como "o linker do build"
é um erro: o linker efetivo é o **AMD LLD 22.0.0** do ROCm.

### 1.2 Pacotes instalados (Arch/CachyOS)

`pacman -Q` (filtrado ao que importa):

```
rocm-core 7.2.4-1.1          hip-runtime-amd 7.2.4-1.1     rocm-language-runtime 7.2.4-1
rocm-llvm 2:7.2.4-2          rocm-device-libs 2:7.2.4-2    rocminfo 7.2.4-1.1
hipblas 7.2.4-1.1 (não usado)  clang 22.1.8-2 / llvm 22.1.8-2 (do sistema, não usados)
```

`pacman -Qi hip-runtime-amd` mostra as dependências que o ROCm arrasta sozinho:
`rocm-core, comgr, rocminfo, rocm-llvm, libelf, rocprofiler-register, numactl, mesa`.

### 1.3 Agente / ISA (`rocminfo`, primeiras linhas do agente de GPU)

```
Agent 2
  Name:                 gfx1201            Marketing Name:  AMD Radeon RX 9070 XT
  Cache Info:  L1: 32 KB   L2: 8192 KB (8 MiB)
  Compute Unit: 64     Wavefront Size: 32     Workgroup Max Size: 1024
  Cacheline: 256 B     Chip ID: 0x7550        Uuid: GPU-a4cc531a20b04853
HSA: Runtime Version 1.18 (Ext 1.15), ROCk module is loaded, XNACK NO, VMM YES
```

`hipconfig --full` confirma o resto: `HIP_COMPILER clang`, `HIP_PLATFORM amd`,
`HIP_RUNTIME rocclr`, `HIP_CLANG_PATH /opt/rocm/lib/llvm/bin`, `Host CPU znver3`.

---

## 2. Qual é a versão **mínima** — e a evidência no código

A pergunta "ROCm mínimo" se decompõe em *pisos de recurso*. Cada um foi checado contra o
código deste repositório e contra a tabela de builtins do clang instalado
(`/opt/rocm/lib/llvm/include/clang/Basic/BuiltinsAMDGPU.def`).

| recurso usado | onde | exigência | sensível à versão? |
|---|---|---|---|
| `--offload-arch=gfx1201` | `CMakeLists.txt:32,33` (+ todas as flags de teste) | o backend AMDGPU precisa **conhecer gfx1201** | **sim** — é o piso duro: sem o alvo, o build nem começa |
| `__builtin_amdgcn_s_prefetch_data` | `include/rdna4/matvec.cuh:157` | builtin com gate **`gfx12-insts`** (`BuiltinsAMDGPU.def:561`) | **sim, é o piso mais apertado e o menos óbvio** — é o único builtin *gfx12-only* do projeto |
| `__builtin_amdgcn_sudot4` | `include/rdna4/vecdotq.cuh:88` | gate **`dot8-insts`** (`BuiltinsAMDGPU.def:306`) | sim (feature do alvo) |
| `__builtin_amdgcn_perm` (24 chamadas) | `vecdotq.cuh:140-149,899-1123` | gate `gfx8-insts` (`def:264`) | não (antigo, sempre presente) |
| `__shfl_xor_sync(0xffffffffull, …)` (5 sítios) | `attn.cuh:130,328,386`, `kv.h:271`, `matvec.cuh` | HIP 7.2 declara o `mask` como **inteiro de 64 bits** | **sim** — ver 2.2 |
| `__half2float(__ushort_as_half(bits))` | `include/rdna4/fp16.h:18`, `kv.h:90,145` | header `amd_hip_fp16.h` | não (estável há muitas versões) |
| shared memory dinâmica | `attn.cuh:189,438` | máx. **33 024 B** com `WPB=32` | **não** — ver 2.3 |
| `hipGetDeviceProperties` | `device.h:22-27` | só `gcnArchName` e `totalGlobalMem` (em testes: `name`) | não (`gcnArchName` existe desde HIP 4.x) |
| `hipMemcpy`/`hipFree` com retorno ignorado | 422 sítios, quase todos em `tests/` | `[[nodiscard]]` (`hip_runtime_api.h:293`, `__HIP_NODISCARD`) | **sim, mas só como ruído** — ver 2.4 |

### 2.1 O piso: `gfx1201` no backend AMDGPU

Os alvos GFX1200/GFX1201 (RDNA4) entraram no backend AMDGPU do LLVM no ciclo do LLVM 18
([Phoronix, nov/2023](https://www.phoronix.com/news/AMD-LLVM-RDNA4-GFX1200)), e o primeiro
ROCm com suporte formal a RDNA4 foi o **6.4.1** ([Phoronix, mai/2025](https://www.phoronix.com/forums/forum/linux-graphics-x-org-drivers/open-source-amd-linux/1547935-amd-releases-rocm-6-4-1-with-rdna4-gpu-support)).
Isso é **evidência externa**, não medida nesta máquina: aqui só existe o 7.2.4, e não há
como verificar toolchains antigas sem instalar (proibido nesta tarefa). Trate
"ROCm ≥ 6.4.1" como *piso por disponibilidade de alvo*, e "ROCm 7.2.4 + amdclang 22" como
**a única combinação realmente verificada**.

O que **foi** verificado localmente, e é acionável:

```
$ /opt/rocm/bin/amdclang++ -O3 --offload-arch=gfx1201 -S -x hip dot.hip -o dot.s
$ grep -oE 'v_dot4_i32_iu8|s_prefetch_data|v_perm_b32' dot.s | sort | uniq -c
      1 s_prefetch_data
      1 v_dot4_i32_iu8
```

Ou seja: com o toolchain instalado, os dois builtins críticos **passam pelo gate de feature**
e chegam à ISA certa (`v_dot4_i32_iu8` = o `dp4a` do M2; `s_prefetch_data` = o único prefetch
real em gfx1201). O sintoma quando *não* passam é literal e imediato — ver §6.1.

### 2.2 `__shfl_xor_sync`: a máscara **precisa** ser de 64 bits no HIP 7.2

`/opt/rocm/include/hip/amd_detail/amd_warp_sync_functions.h:306`:

```cpp
template <typename MaskT, typename T>
__device__ inline T __shfl_xor_sync(MaskT mask, T var, int laneMask, int width = warpSize) {
  static_assert(__hip_internal::is_integral<MaskT>::value && sizeof(MaskT) == 8,
                "The mask must be a 64-bit integer. "
                "Implicitly promoting a smaller integer is almost always an error.");
  __hip_adjust_mask_for_wave32(mask);
  ...
```

Duas consequências práticas: (a) o `0xffffffffull` do código é **obrigatório**, não estilo —
trocar por `0xffffffffu` ("wave32 usa só 32 bits") **não compila**; (b) a máscara é ajustada
para wave32 por `__hip_adjust_mask_for_wave32`, então não há o que "otimizar" aqui.

### 2.3 Shared memory dinâmica: **não** é sensível à versão

O teto de LDS nesta placa é 128 KB/WGP e **64 KB por workgroup**
(`docs/rocm-estudo.md` §B.1, medido + ISA). A atenção pede, no máximo
`WPB · (2 + head_dim) · 4 B` com `WPB=32`, `head_dim=256` (`attn.cuh:189` e `:438`):

```
32 · (2 + 256) · 4 = 33 024 B = 32,25 KiB
```

como *shared memory dinâmica* na chamada (`attn_kernel<<<…, threads, smem, stream>>>`). Isso
está **abaixo** do limite por workgroup e abaixo até do limite "clássico" de 48 KB que exige
opt-in (`hipFuncSetAttribute`): **não há** `hipFuncSetAttribute`/`cudaFuncAttribute*` em
nenhum lugar do repositório (grep vazio), e nenhum kernel depende de LDS grande. Portanto
este item **não** é um piso de versão — ao contrário do que a intuição sugere.

### 2.4 `[[nodiscard]]`: 422 dos 433 avisos são "novidade" do header do ROCm 7.2

O HIP 7.2 marca o retorno de `hipMemcpy`/`hipFree`/`hipMemGetInfo` como `[[nodiscard]]`
(`hip_runtime_api.h:293`). O código (majoritariamente `tests/*.hip`) ignora esses retornos em
422 lugares → 422 avisos `-Wunused-value`. Em ROCm antigo (sem o atributo) os mesmos 422
avisos **não existiriam**. É ruído, não defeito — mas é a explicação de "por que o build
deste repo é barulhento só nesta versão".

---

## 3. O build limpo que foi executado (limpo, fora da árvore)

Comandos exatos (diretório novo em `/tmp`, **nunca** o `build/` do repo):

```bash
source scripts/rocm-env.sh
cmake -S . -B /tmp/build-clean -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/build-clean -j6
```

Cabeça da saída (configure):

```
-- The CXX compiler identification is GNU 16.2.1
-- The HIP compiler identification is Clang 22.0.0
-- Check for working CXX compiler: /usr/bin/c++ - skipped
-- Check for working HIP compiler: /opt/rocm/bin/amdclang++ - skipped
-- Configuring done (4.1s)
-- Generating done (0.1s)
-- Build files have been written to: /tmp/build-clean
```

Rabo da saída (build):

```
[ 95%] Built target bench-matvec-shapes-gpu
[ 98%] Linking HIP executable check-mtp-gpu
[ 98%] Built target check-mtp-gpu
[ 99%] Linking HIP executable check-batch-gpu
[ 99%] Built target check-batch-gpu
[100%] Linking HIP executable rdna4-infer
[100%] Built target rdna4-infer
```

Resultado:

| | |
|---|---|
| exit code | **0** |
| parede (`time`) | **real 4m59,890s** · user 10m7,536s · sys 0m11,763s (`-j6`) |
| alvos construídos | **29** (`Built target` distintos): 27 executáveis — o motor `rdna4-infer`, 10 `check-*` de CPU, 11 `check-*`/`bench-*` de GPU e 5 oráculos `oracle-*` — mais `librdna4_serve.so` e `librdna4_tokenizer.a` |
| compilações | 93 passos de compilação de TU (CXX e HIP) |
| avisos | **433** — 422 `-Wunused-value` (nodiscard, §2.4), 5 `IQ3S_N_SCALE redefinido`, 4 escape hexadecimal (g++), 2 `-Wformat` |

Flags efetivas de uma TU HIP (de `/tmp/build-clean/CMakeFiles/rdna4-infer.dir/flags.make`):

```
HIP_FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1201 --offload-arch=gfx1201
CXX_FLAGS = -O3 -DNDEBUG -std=gnu++17
```

Linha de link real (`CMakeFiles/rdna4-infer.dir/link.txt`, abreviada):

```
/opt/rocm/bin/amdclang++ -O3 -DNDEBUG --offload-arch=gfx1201 --offload-arch=gfx1201 \
  --hip-link ... src/main.hip.o ... -o rdna4-infer \
  -Wl,-rpath,/tmp/build-clean /opt/rocm/lib/libamdhip64.so librdna4_serve.so \
  /opt/rocm/lib/libamdhip64.so.7.2.53211-3d9ef42 -lgcc -latomic_asneeded -lgcc
```

Três fatos que essa linha prova de uma vez: (a) o **motor não linka llama.cpp nem ggml**;
(b) `--offload-arch=gfx1201` aparece no **link** também; (c) o binário depende de
`librdna4_serve.so` (mesma pasta via `rpath`) — copiar só o `rdna4-infer` para outro lugar
quebra (`ldd`: `librdna4_serve.so => /tmp/build-clean/librdna4_serve.so`).

### 3.1 Avisos que são de versão (o que olhar num build de outra máquina)

| aviso | quantidade | origem | é problema? |
|---|---|---|---|
| `ignoring return value of type 'hipError_t' declared with 'nodiscard'` | 422 | ROCm 7.2 (atributo no header) + código de teste que ignora retornos | não |
| `'IQ3S_N_SCALE' redefinido` (`ggml-common.h:414` vs `quants.h:15`) | 5 | 1 fixture (`tests/dequant_cpu_oracle.cpp`) que inclui **nossos** headers e os do llama.cpp; ambos definem 4 (`QK_K/64` com `QK_K=256`) | não (mesmo valor), mas muda se o llama.cpp mudar o macro |
| `sequência de escape hexa fora de alcance` (`src/backend/tokenizer.cpp:140`) | 4 | **único aviso em código que roda** — ver §6.6 | ⚠️ **sim, e é um achado** |
| `format specifies type 'size_t' … but the argument has type 'int'` (`tests/check_kvctx_gpu.hip:289`) | 2 | `%zu` com `int` na mensagem do gate | não (só a mensagem) |

---

## 4. O que o build exige do disco (e como ele falha quando falta)

### 4.1 `build/` **não** é usado por nada

- `.gitignore:1` ignora `build/`; nesta worktree o diretório **não existe** (`ls -ld build` →
  "Arquivo ou diretório inexistente") e o build limpo passou assim mesmo.
- `grep -rn 'build/' CMakeLists.txt` não retorna **nenhum** caminho de leitura (as únicas
  ocorrências são o `-B build` dos exemplos no `README.md:45-46` e o `build/bin` da árvore do
  **llama.cpp**, que é outra coisa).
- **Sem dependência de artefato velho**: configure e build funcionam num diretório novo e
  vazio. (Foi exatamente o que o `/tmp/build-clean` fez.)

### 4.2 Caminhos absolutos no `CMakeLists.txt`

| caminho | linhas | existe aqui | o que acontece se faltar |
|---|---|---|---|
| `/opt/rocm/bin/amdclang++` | 5-7 (`if(EXISTS …)`) | sim | cai no default do CMake; `PATH` **não** é necessário (§5.2) |
| `/opt/rocm/include` | 36-38, 77-79, 90, 100, 139, 149, 157, 189-191, 200-202, 243-245, 260-262, 283-285 | sim | **silencioso** — diretório de include inexistente não é erro nem no CMake nem no clang; a falha aparece só depois, como erro de compilação de header do HIP |
| `/opt/rocm/lib` (`find_library(AMDHIP64 … REQUIRED)`) | 40 | sim | **erro claro** (único gate do ROCm): `CMake Error at CMakeLists.txt:40 (find_library): Could not find AMDHIP64 using the following names: amdhip64` |
| `LLAMA_CPP_SRC` (default `/home/marcelo/Projetos/llama.cpp`) | 56 (CACHE PATH) | sim | `…/ggml/src`, `…/ggml/include`, `…/include`, `…/common` entram como `-I` **sem validação**; a falha vem na compilação: `fatal error: ggml-quants.h: Arquivo ou diretório inexistente` |
| `LLAMA_CPP_BUILD` (default `…/llama.cpp/build`) | 58 (CACHE PATH) | sim | `libggml-base.so` é **validado por `FATAL_ERROR`** (64-67); `libllama.so`/`libllama-common.so`/`libggml.so` entram como **caminho completo** em `target_link_libraries` (114, 126-127, 167, 177, 272) e **não** são validados: a falha é do Make, em tempo de build |

**Testes feitos (todos de CPU, nada instalado):**

```bash
# A) llama.cpp ausente  ->  CMake aborta no configure
cmake -S . -B /tmp/cfg-nollama -DLLAMA_CPP_SRC=/nonexistent -DLLAMA_CPP_BUILD=/nonexistent
# CMake Error at CMakeLists.txt:66 (message):
#   libggml-base.so not found under /nonexistent/bin — build llama.cpp first
# exit 1  (Configuring incomplete)

# B) só um libggml-base.so vazio presente  ->  configure PASSA
mkdir -p /tmp/fakellama/bin && : > /tmp/fakellama/bin/libggml-base.so
cmake -S . -B /tmp/cfg-partial -DLLAMA_CPP_SRC=/tmp/fakellama-src -DLLAMA_CPP_BUILD=/tmp/fakellama
# exit 0  — nenhum dos -I nem libllama.so é conferido

# C) headers reais + libllama.so ausente  ->  falha no BUILD (não no configure)
cmake --build /tmp/cfg-link --target oracle-tokenize -j6
# make[3]: *** Sem regra para processar o alvo '/tmp/fakellama/bin/libllama.so',
#              necessário por 'oracle-tokenize'.  Pare.
```

### 4.3 As dependências de llama.cpp são opcionais?

- **Um único `message(FATAL_ERROR)` existe no `CMakeLists.txt`** (linha 66) — a checagem do
  `libggml-base.so`. Não há outro. Mas ele roda no **configure**, então *não existe* caminho
  "buildar só o motor sem llama.cpp" com o CMakeLists como está: o configure aborta antes.
  (Contorno sem editar nada: apontar `LLAMA_CPP_BUILD` para um diretório onde exista
  qualquer arquivo chamado `libggml-base.so` — o teste B mostra que um arquivo de 0 byte
  passa — e então `cmake --build … --target rdna4-infer`.)
- **11 dos 27 alvos** linkam llama.cpp/ggml (`check-dequant`, `check-dequant-gpu`,
  `check-matvec-gpu`, `check-nn-gpu`, `check-graph-gpu`, `check-kvctx-gpu`, `oracle-tokenize`,
  `oracle-chat`, `oracle-next-token`, `oracle-eog`, `oracle-mtp`).
- **`cmake --build` sem `--target` constrói todos** → com llama.cpp quebrado, o build padrão
  **falha**. Só o motor é independente: `cmake --build <dir> --target rdna4-infer` funciona.
- Em tempo de execução, **nada** de llama.cpp é preciso: o `link.txt` do `rdna4-infer` lista
  só `libamdhip64` e `librdna4_serve.so` (confere com `README.md:121-123`).

---

## 5. O que o usuário precisa instalar, o env do repo e os arquivos de modelo

### 5.1 Uma linha por família de distro

Verificado nesta máquina (base Arch):

```bash
# Arch / CachyOS / Manjaro  — o pacote existe nos repos: `extra/rocm-hip-sdk 7.2.4-1`
sudo pacman -S --needed cmake gcc make rocm-hip-sdk
# mínimo equivalente (o hip-runtime-amd já arrasta rocm-llvm/rocminfo/comgr):
sudo pacman -S --needed cmake gcc make hip-runtime-amd rocm-device-libs
```

Não verificados aqui (evidência externa, ver links): Ubuntu/Debian
`sudo apt install rocm-hip-runtime-dev rocm-llvm cmake g++` e RHEL/Fedora
`sudo dnf install rocm-hip-runtime-devel rocm-llvm cmake gcc-c++` — a nomenclatura
`…-runtime-dev`/`…-runtime-devel` é a atual segundo a
[página de instalação da AMD](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/install-methods/package-manager/package-manager-ubuntu.html)
(a documentação antiga usava `rocm-hip-sdk`).

Não é preciso instalar: `hipcc` (o CMake o rejeita), `rocblas`/`hipblas`,
`rocprofiler`/`omniperf` (não instalados aqui de propósito, `docs/gpu-queue.md:11-14`),
nem a toolchain LLVM do sistema.

### 5.2 `scripts/rocm-env.sh` — lido, e continua correto

```bash
# scripts/rocm-env.sh (4 linhas, íntegro)
export PATH="/opt/rocm/bin:${PATH}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
```

O que ele cobre e o que ele **não** precisa cobrir (com a evidência):

| pergunta | resposta |
|---|---|
| O build precisa dele? | **Não.** Configurei com `PATH` sem `/opt/rocm/bin` e o CMake achou o compilador sozinho (`CMakeLists.txt:5-7` fixa o caminho absoluto): `-- Check for working HIP compiler: /opt/rocm/bin/amdclang++ - skipped`, exit 0 |
| Precisa pôr `/opt/rocm/llvm/bin` no PATH? | **Não** — `/opt/rocm/bin/amdclang++` é symlink para `/opt/rocm/lib/llvm/bin/amdclang++` |
| Precisa de `LD_LIBRARY_PATH`? | **Não** — o pacote instala `/etc/ld.so.conf.d/rocm.conf` com `/opt/rocm/lib` (o `ldd` do binário resolve `libamdhip64.so.7` e `libhsa-runtime64.so.1` de lá) |
| Precisa exportar `ROCM_PATH`? | Já vem de `/etc/profile.d/rocm.sh` (`export ROCM_PATH=/opt/rocm` + `append_path /opt/rocm/bin`) em shell de login; o script só garante `HIP_PATH` |
| Então para que serve? | Para `hipcc`, `rocminfo`, `rocm-smi` e afins em shell **não**-login (o caso do CI/agente). Está correto e é suficiente; um `export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"` seria paridade inofensiva |

### 5.3 Arquivos de modelo (caminhos, tamanhos, onde são esperados)

O motor **não** procura modelo por padrão: o caminho vem sempre de `-m <arquivo.gguf>`
(não há variável de ambiente de modelo). Os scripts de gate usam um default fixo (ex.:
`scripts/check_golden_run.sh`, `scripts/compare_ppl.sh` → `MODEL=${MODEL:-/mnt/raid0/GGUF/…}`).

| arquivo | bytes | decimal | o que é |
|---|---|---|---|
| `/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf` | 12 040 883 104 | 12,04 GB | primário (866 tensores; soma dos tensores = 12 029 886 464 B = 12,030 GB) |
| `…/Qwen3.8-27B-UD-IQ4_XS.gguf` | 14 252 845 984 | 14,25 GB | secundário |
| `…/mmproj-F16.gguf`, `…/mmproj-q8_0.gguf` | 927,6 MB / 625,7 MB | — | **projeção de visão: não usada por este motor** |

Hparams relevantes lidos do header do IQ3_S (parser próprio, só o cabeçalho): `qwen35`,
`block_count 65`, `nextn_predict_layers 1` → 64 camadas de tronco, das quais 16 de atenção
plena (`qwen35.full_attention_interval 4`) e 48 recorrentes (GDN); `head_count 24`,
`head_count_kv 4`, `key_length = value_length = 256`, `rope.dimension_count 64`,
`freq_base 1e7`, `embedding_length 5120`, `feed_forward_length 17408`,
`context_length 262144` (o `--ctx-size` default com que o CLI roda é 4096).

Também é preciso (para os gates, não para rodar): `tests/golden/*` está **commitado**
(7 arquivos: tokenizer, eog, chat, golden run), mas `reference/` — onde vivem os dumps de
oráculo do llama.cpp (`oracle_capital_cpu.txt` etc.) e `reference/data/wikitext-2-raw` — é
**gitignored** (`.gitignore:12-13`) e é gerado por `scripts/capture_oracle.sh`, cujo default é
`LLAMA_DIR=/home/marcelo/Projetos/llama.cpp` (`scripts/capture_oracle.sh:29`). Sem esses
dumps, `check-graph-gpu` não tem contra o que comparar.

---

## 6. Armadilhas de build (sintoma → causa → correção)

### 6.1 Uma TU HIP **precisa** da extensão `.hip` (ou de `LANGUAGE HIP`)

**Sintoma** (verificado compilando o mesmo arquivo com as duas extensões):

```
$ amdclang++ -O3 -DNDEBUG --offload-arch=gfx1201 -c t.cpp
clang++: warning: argument unused during compilation: '--offload-arch=gfx1201'
t.cpp:1:1: error: unknown type name '__global__'
t.cpp:2:23: error: expected expression        # k<<<1,1>>>(p)
```

**Causa:** sem a extensão `.hip`, o `amdclang++` compila em **modo host puro** — o
`--offload-arch` é aceito e **ignorado**, `__global__` não existe e `<<<>>>` não é sintaxe.
O CMake decide a linguagem pela extensão, então um kernel num `.cpp` silenciosamente vira
código de host (e o erro só aparece quando o corpo usa algo de device).

**Correção:** manter `tests/*.hip` e `src/main.hip` como `.hip` (foi o "bug de build
resolvido" do M2, `PLAN.md:66`). Para um arquivo que precisa de outra extensão, use
`set_source_files_properties(x.cpp PROPERTIES LANGUAGE HIP)`.

### 6.2 A flag de otimização do HIP: o estado atual e o que **não** funciona

**Histórico (M2, `PLAN.md:80`):** *"Bug de build crítico encontrado: o código HIP estava sendo
compilado em `-O0`"* — o combo CMake/ROCm não define flag de otimização para HIP, e as flags
eram só `--offload-arch=gfx1201`. Efeito medido: **4,4 GB/s → 268 GB/s (62×)**. A correção foi
`CMAKE_HIP_FLAGS_RELEASE = "-O3 -DNDEBUG"`.

**Estado atual (verificado, `CMakeLists.txt:15-20`):**

```cmake
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)                  # default: Release se ninguém disser nada
endif()
set(CMAKE_HIP_FLAGS_RELEASE "-O3 -DNDEBUG")      # <- a correção do M2, no lugar
set(CMAKE_HIP_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_HIP_FLAGS_DEBUG "-O0 -g")
```

`flags.make` do build limpo confirma `HIP_FLAGS = -O3 -DNDEBUG …`. **A flag está no lugar.**
Sem fast-math em nenhum lugar (grep por `ffast-math|Ofast|fast_math|ffp-contract` → vazio), o
que é requisito de projeto (exatidão bit a bit).

**Armadilha nova (testada):** como o `set()` é de variável **normal**, ele **sombreia** a
variável de cache. Passar a flag na linha de comando é **silenciosamente ignorado**:

```
$ cmake -S . -B /tmp/cfg-flags -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_FLAGS_RELEASE="-O0 -g"
$ grep -m1 '^HIP_FLAGS' /tmp/cfg-flags/CMakeFiles/rdna4-infer.dir/flags.make
HIP_FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1201 --offload-arch=gfx1201   # -O0 perdido
```

(`CMAKE_HIP_FLAGS_RELEASE:STRING=-O0 -g` está no `CMakeCache.txt`, e mesmo assim não vale.)

**Correção:** o jeito suportado de mexer na otimização é `CMAKE_BUILD_TYPE`
(`Release` → `-O3 -DNDEBUG`; `RelWithDebInfo` → `-O2 -g -DNDEBUG`; `Debug` → `-O0 -g`) ou
editar o `CMakeLists.txt`. `-DCMAKE_HIP_FLAGS=…` também não ajuda: as flags do usuário entram
**antes** das de config, então o `-O3` continua ganhando.

### 6.3 `--offload-arch=gfx1201` no **compile e no link** — e o arch autodetectado

O projeto já faz certo: `target_compile_options(... $<$<COMPILE_LANGUAGE:HIP>:--offload-arch=gfx1201>)`
**e** `target_link_options(... --offload-arch=gfx1201)` (`CMakeLists.txt:32-33`, repetido em
todos os alvos de GPU). Linkar sem o `--offload-arch` produz um binário sem código de device
(o `--hip-link` não sabe qual fatbin embutir).

**Armadilha nova e séria:** o CMake **também** injeta o arch que ele detecta na máquina. No
`CMakeCache.txt` do build limpo aparece, sem docstring e sem ter sido passado por ninguém:

```
CMAKE_HIP_ARCHITECTURES:STRING=gfx1201
```

e é por isso que as flags mostram `--offload-arch=gfx1201` **duas vezes** (uma do CMake, uma
do `target_compile_options`). Consequência: **em máquina sem GPU, ou com outra GPU AMD, o
build compila para outra arquitetura além de gfx1201** — e aí ele quebra, porque o builtin de
prefetch é `gfx12`-only. Reproduzido simulando a detecção:

```
$ cmake -S . -B /tmp/cfg-arch2 -DCMAKE_BUILD_TYPE=Release -DCMAKE_HIP_ARCHITECTURES=gfx1030
$ grep -m1 '^HIP_FLAGS' /tmp/cfg-arch2/CMakeFiles/check-matmul-gpu.dir/flags.make
HIP_FLAGS = -O3 -DNDEBUG -std=gnu++17 --offload-arch=gfx1030 --offload-arch=gfx1201
$ cmake --build /tmp/cfg-arch2 --target check-matmul-gpu -j6
include/rdna4/matvec.cuh:157:3: error: '__builtin_amdgcn_s_prefetch_data'
                                    needs target feature gfx12-insts
```

**Correção:** fixe o alvo explicitamente em máquina que não é a de referência
(`cmake -S . -B build -DCMAKE_HIP_ARCHITECTURES=gfx1201 …`), ou confira o
`CMAKE_HIP_ARCHITECTURES` no cache antes de culpar o código.

### 6.4 `rdna4_serve` é uma biblioteca compartilhada por causa de `duplicate symbol` — **ainda verdade**

O comentário em `CMakeLists.txt:222-228` diz que `graph.cuh` define os `__global__` **sem**
`static`/`inline`, então uma segunda TU que o inclua colide com a cópia do `main.hip`.
**Verifiquei em duas frentes:**

1. No código: os kernels dos headers incluídos por `graph.cuh` são declarados assim —
   `__global__ void rope_kernel(...)` (`attn.cuh:29`), `attn_merge_kernel` (`:401`),
   `mul_kernel`/`add_kernel`/`rms_norm_kernel`/`unary_kernel` (`nn.cuh`), `conv1d_state_kernel`
   e `delta_rule_kernel` (`gdn.cuh:25,57`). Nenhum tem `static` nem `inline`.
2. Linkando de verdade uma segunda TU (`#include "rdna4/graph.cuh"`) contra o
   `main.hip.o` do build limpo:

```
ld.lld: error: duplicate symbol: rdna4::rope_kernel(float*, int, int, int, int, float, int const*)
ld.lld: error: duplicate symbol: rdna4::attn_merge_kernel(float const*, float*, int, int, int)
ld.lld: error: duplicate symbol: rdna4::add_kernel(float const*, float const*, float*, long)
ld.lld: error: duplicate symbol: rdna4::mul_kernel(float const*, float const*, float*, long)
ld.lld: error: duplicate symbol: rdna4::rms_norm_kernel(...)   ...
# 20 símbolos duplicados no total (cada kernel + o __device_stub__ correspondente)
```

**Conclusão:** a afirmação continua verdadeira e o desenho (servidor como `.so`, `CMakeLists.txt:229-234`)
continua necessário. A alternativa "marcar os kernels como internos" mexeria em 6 headers de
outro workstream — não fazer isso neste lote. Nota de operação: como é um `.so`, os símbolos
do kernel ficam **duplicados entre o executável e a biblioteca** (53 `T` exportados no `.so`,
55 no executável) e o linker dinâmico resolve pelo executável; não há erro, mas o tamanho
do build paga por isso.

### 6.5 `build/` é gitignored (e o oráculo também)

`.gitignore:1` = `build/`; `.gitignore:12-13` = `reference/` e `reference`. Consequências
práticas: (a) ninguém deve commitar artefato de build — o build limpo em `/tmp` mantém a
árvore limpa por construção; (b) os dumps de oráculo **não** viajam no clone, então
`check-graph-gpu`/`compare_ppl.sh` precisam de `scripts/capture_oracle.sh` (que exige o
checkout do llama.cpp).

### 6.6 O único aviso em código que roda: um `\x` que come o `e` de "end"

`src/backend/tokenizer.cpp:140` contém, byte a byte (é a terceira entrada da lista):

```cpp
"<turn|>", "<|tool_response>", "<\xEF\xBD\x9Cend\xE2\x96\x81of\xE2\x96\x81sentence\xEF\xBD\x9C>",
```

O escape `\x9C` é seguido de `e`, que é dígito hexadecimal: o compilador lê **`\x9Ce`** =
0x9CE, que não cabe num byte.

- **GCC 16.2.1 avisa e trunca**: `hexgcc` reproduz isso — a string vira 26 bytes
  (`3C EF BD CE 6E 64 …`, ou seja o `e` de "end" **desaparece**) e **não** é igual ao texto
  pretendido `<\uFF5Cend\u2581of\u2581sentence\uFF5C>`.
- **`amdclang++` 22 trata como ERRO**: `error: hex escape sequence out of range`. Ou seja,
  mover essa TU para o lado HIP/clang **quebra o build**.
- **Impacto real neste modelo: nenhum.** Varri os 248 320 tokens do IQ3_S: **zero** contêm
  U+FF5C, e esse texto não existe no vocab, então o `token_to_id_.find(txt)` simplesmente não
  acha (os FIM 248063/248064/248065 vêm da outra lista, `kFimTexts`, e o `check-eog` passa
  5/5). É uma bomba-relógio de manutenção, não um bug ativo.
- **Correção:** quebrar o literal (`"<\xEF\xBD\x9C" "end" …`) ou usar `"\uFF5C end…"` com
  `u8`/UTF-8 direto no arquivo. Vale corrigir junto com qualquer mexida no tokenizer.

### 6.7 `hypcc`/`hipcc` **não** é aceito pelo CMake (o comentário do arquivo confere)

```
$ cmake -S . -B /tmp/cfg-hipcc -DCMAKE_HIP_COMPILER=/opt/rocm/bin/hipcc
CMake Error at /usr/share/cmake/Modules/CMakeDetermineHIPCompiler.cmake:75 (message):
  CMAKE_HIP_COMPILER is set to the hipcc wrapper:
   /opt/rocm/bin/hipcc
  This is not supported.  Use Clang directly, or let CMake pick a default.
```

**Correção:** não passe `-DCMAKE_HIP_COMPILER`; o `CMakeLists.txt:5-7` já aponta para
`/opt/rocm/bin/amdclang++`.

---

## 7. Reproduzir tudo o que está neste documento

```bash
# versões (§1)
cat /opt/rocm/.info/version; hipcc --version; hipconfig --full
amdclang++ --version; cmake --version; make --version; uname -r; ldd --version | head -1
rocminfo | sed -n '1,60p'
pacman -Q | grep -E 'rocm|hip|clang|llvm'          # Arch/CachyOS

# build limpo cronometrado (§3)
source scripts/rocm-env.sh
rm -rf /tmp/build-clean
time ( cmake -S . -B /tmp/build-clean -DCMAKE_BUILD_TYPE=Release \
       && cmake --build /tmp/build-clean -j6 )      # exit 0, ~5 min com -j6
grep -m1 '^HIP_FLAGS' /tmp/build-clean/CMakeFiles/rdna4-infer.dir/flags.make

# dependências externas (§4)
cmake -S . -B /tmp/x1 -DLLAMA_CPP_SRC=/nonexistent -DLLAMA_CPP_BUILD=/nonexistent   # FATAL_ERROR
grep -n 'FATAL_ERROR\|LLAMA_CPP\|/opt/rocm' CMakeLists.txt
grep -l 'llama\|ggml' /tmp/build-clean/CMakeFiles/*/link.txt                        # 11 alvos

# arch autodetectado (§6.3)
grep CMAKE_HIP_ARCHITECTURES /tmp/build-clean/CMakeCache.txt
```
