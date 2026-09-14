// Frente D2 do plano do prefill (docs/plano-prefill.md): o GEMM tilejado int8 dentro do
// vocabulario do motor, medido contra o GEMV em lote que embarca hoje.
// Esqueleto do coordenador -- o alvo de build ja' existe no CMakeLists.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  std::printf("bench-gemm-engine-gpu: stub (frente D2 ainda nao preencheu)\n");
  return 0;
}
