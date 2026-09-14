// Frente H do estudo de prefill (docs/plano-prefill.md, degrau D4):
// MMQ int8 com WMMA 16x16x16 sobre os BLOCOS REAIS do modelo, bit-exato contra o dp4a do motor.
// Receita do operando: docs/estudo-prefill-d-wmma.md (b)/(c). Esqueleto do coordenador.
#include <hip/hip_runtime.h>
#include <cstdio>
int main() {
  std::printf("bench-mmq-wmma-gpu: stub (frente H ainda nao preencheu)\n");
  return 0;
}
