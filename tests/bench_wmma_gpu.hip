// Front D do estudo de prefill (docs/estudo-prefill-d-wmma.md):
// quanto vale o caminho de matrix cores (wmma i32_16x16x16_iu8) neste gfx1201,
// contra o dp4a que o motor usa hoje. Esqueleto criado pelo coordenador para o
// agente da frente D preencher -- o alvo de build ja' existe no CMakeLists.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  std::printf("bench-wmma-gpu: stub (frente D ainda nao preencheu)\n");
  return 0;
}
