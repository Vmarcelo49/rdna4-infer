# tests/

- `golden/` — prompt fixo + saída esperada do teste de fumaça (M4). Arquivos `.gguf`
  nunca entram aqui (ficam em `/mnt/raid0/GGUF/...`, fora do git).
- Unitários de M2/M3 (GPU-vs-CPU) vivem junto ao código que testam ou aqui,
  a definir em cada milestone.
