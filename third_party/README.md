# third_party/

Módulos vendored do llama.cpp (licença MIT — manter cabeçalho de copyright e
anotar aqui a origem de cada arquivo):

| arquivo destino | origem em `.ref/llama.cpp` | para o milestone |
|---|---|---|
| (a preencher em M1) `gguf.*` | `ggml/include/gguf.h`, `ggml/src/gguf.cpp` | M1 loader |
| (a preencher em M2) `vecdotq.cuh`, `mmvq.cu`, `mmq*.cuh` | `ggml/src/ggml-cuda/` | M2 kernels |

Regra: copiar o mínimo necessário, nunca o backend inteiro. Revisão de origem:
`git -C .ref/llama.cpp rev-parse --short HEAD` na hora de vendorizar.
