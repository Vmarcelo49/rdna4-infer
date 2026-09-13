# scripts/

Helpers (Python/shell) — nunca fazem parte do binário.

- `rocm-env.sh` — `source` para expor o toolchain ROCm (`hipcc`, `rocm-smi`).
- `inspect_gguf.py` — despeja metadados de um `.gguf` (KVs, histograma de tipos,
  layouts por camada). Usado para validar o loader de M1 contra os arquivos UD.
  Futuros: `bench.py` (tok/s e perplexidade por quant), `smoke.py` (golden test de M4).
