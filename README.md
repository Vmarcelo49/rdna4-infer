# rdna4-infer

A from-scratch inference engine for **AMD RDNA4 (RX 9070 XT, `gfx1201`)**, focused on a
single workload: **Qwen3.8 27B dense in GGUF**, quantized to fit 16 GB of VRAM.

Instead of being another general-purpose runner, `rdna4-infer` trades breadth for depth:
one GPU, one architecture, one model family — so every layer of the stack (loader, kernels,
graph, sampler) can be tuned for exactly that target.

## Objectives

- Run `Qwen3.8-27B` (`UD-IQ3_S` 12 GB primary, `UD-IQ4_XS` 14 GB secondary) fully in VRAM
  on a stock RX 9070 XT via a `run` CLI with streaming output.
- HIP-only backend compiled for `gfx1201`, reusing battle-tested llama.cpp modules
  (GGUF reader, RDNA4-tuned MMVQ/MMQ, Q/I-quant dequant, sampler) under MIT — no full fork.
- Beat-the-reference mindset: llama.cpp baselines are recorded in `docs/` and every
  tuning decision is measured against them (see `PLAN.md` M5).

## Non-goals (v1)

Other models and MoE variants, servers/APIs, quantization tooling, other GPUs or OSes,
offload and multi-GPU. Full scope in `SPEC.md`.

## Layout

```
├── CMakeLists.txt        # HIP-only, --offload-arch=gfx1201
├── src/                  # engine (main.hip; backend/: loader, graph, sampler)
├── include/rdna4/        # public headers
├── kernels/              # HIP kernels (vendored + own)
├── third_party/          # llama.cpp vendoring policy (MIT)
├── scripts/              # rocm-env.sh, inspect_gguf.py (+ bench.py, smoke.py)
├── tests/                # golden/ smoke test (no .gguf in git)
├── docs/                 # research: AMD docs, upstream mentions, kernels, GGUF/quant, baselines
└── .ref/                 # reference clones (vLLM, SGLang, hipfire, llama.cpp)
```

## Build

```bash
source scripts/rocm-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/rdna4-infer   # verifies gfx1201 device
```

Requires Linux + ROCm with `amdclang++` (tested with ROCm 7.2).

## Docs

- `SPEC.md` — what v1 will and will not have.
- `PLAN.md` — M0–M5 milestones with acceptance criteria and code references.
- `docs/rdna4-gfx1201-referencias-amd.md` — official AMD documentation index.
- `docs/referencias-upstream-gfx1201-qwen35.md` — gfx1201/Qwen3.5 mentions across
  vLLM, SGLang, hipfire and llama.cpp.
- `docs/kernels-ia-gfx1201.md` — kernel programming guide (AMD + community reports).
- `docs/gguf-qwen-quantizacao-llamacpp.md` — GGUF layout for Qwen + quantization.
- `docs/baseline-vulkan-iq3s.md` — llama.cpp baselines (Vulkan, IQ3_S, 32K ctx) to beat.
