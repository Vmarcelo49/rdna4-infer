// Frente F do estudo de prefill (docs/plano-prefill.md, degrau D2):
// o GEMM tilejado com os BLOCOS REAIS do modelo (iq3_s e companhia) e staging em f16 na LDS.
// Esqueleto criado pelo coordenador; o alvo de build ja' existe no CMakeLists.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  std::printf("bench-gemm-quant-gpu: stub (frente F ainda nao preencheu)\n");
  return 0;
}
