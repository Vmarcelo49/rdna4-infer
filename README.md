# rdna4-infer

A from-scratch inference engine for **AMD RDNA4 (RX 9070 XT, `gfx1201`)**, focused on a
single workload: **Qwen3.8 27B dense in GGUF**, quantized to fit 16 GB of VRAM.

Instead of being another general-purpose runner, `rdna4-infer` trades breadth for depth:
one GPU, one architecture, one model family — so every layer of the stack (loader, kernels,
graph, sampler) can be tuned for exactly that target.

## Objectives

- Run `Qwen3.8-27B` (`UD-IQ3_S` 12 GB primary, `UD-IQ4_XS` 14 GB secondary) fully in VRAM
  on a stock RX 9070 XT via a `run` CLI with streaming output, or behind an
  **OpenAI-compatible HTTP server** (`serve`) that common harnesses can talk to.
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
├── src/main.hip          # CLI: info | run | bench | ppl | tokenize
├── src/backend/          # gguf, loader, model, tokenizer, unicode, sampler, chat
├── include/rdna4/        # headers + kernels (graph, matvec, attn, gdn, kv, dequant)
├── tests/                # 16 checks (CPU + GPU) and the oracle tools
├── scripts/              # build/run helpers and the acceptance gates
├── docs/                 # research notes + docs/medicoes-m5.md (measurements)
└── reference/            # oracle captures (gitignored)
```

Everything is built into one binary plus one test binary per check; there is no
library to install and no runtime besides ROCm.

## Build

```bash
source scripts/rocm-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/rdna4-infer   # verifies gfx1201 device
```

Requires Linux + ROCm with `amdclang++` (tested with ROCm 7.2).

## Usage

```bash
# device + VRAM budget check (no model needed)
./build/rdna4-infer

# generate text (stdout is only the generated text; stats go to stderr)
./build/rdna4-infer run -m model.gguf -p "The capital of France is" -n 64 --greedy
./build/rdna4-infer run -m model.gguf --chat -p "What is 2+2?" -n 128 --temp 0.7 --seed 42

# benchmark (decode/prefill tok/s, VRAM, effective bandwidth)
./build/rdna4-infer bench -m model.gguf -n 64 --reps 3
./build/rdna4-infer bench -m model.gguf --ctx-size 65536 --cache-type-k q4_0                           --cache-type-v q4_0 --fill-cache      # long-context decode

# OpenAI-compatible server (chat completions with SSE streaming, /v1/models, /health)
./build/rdna4-infer serve -m model.gguf --port 8080 --ctx-size 8192
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'

# perplexity on a corpus (llama-perplexity-compatible tiling)
./build/rdna4-infer ppl -m model.gguf -f corpus.txt --ctx-size 512 --stride 512 --chunks 10
```

`run` flags: `-m -p -n --chat --system --no-thinking --reasoning-effort --temp
--greedy --top-k --top-p --min-p --repeat-penalty --repeat-last-n --seed
--ctx-size --cache-type-k/-v -v --no-stats`. Sampler defaults come from the GGUF's
`general.sampling.*` metadata (this model ships `top_k 20`, `top_p 0.95`,
`temp 1.0`); any flag given on the command line wins.

KV cache types: `f16` (default, matches llama.cpp), `q8_0`, `q4_0`, `f32` — chosen
independently for K and V. `q4_0` is what makes a 64K+ context fit in 16 GB.

Exit codes: `0` success, `1` usage/IO/model error, `2` no `gfx1201` device,
`3` insufficient VRAM.

## Measured numbers

`docs/medicoes-m5.md` has the full table set (both GGUF files, KV types, context
lengths) with the exact command for every number. Headline, IQ3_S on the RX 9070 XT:

| | this engine | llama.cpp (Vulkan) on the same machine |
|---|---|---|
| decode (IQ3_S, 4K, KV f16) | 26.8 tok/s | 39.7 tok/s |
| decode (IQ4_XS) | 27.0 tok/s | — |
| prefill (per-token path) | 28.8 tok/s | 440 tok/s (batched) |
| effective weight bandwidth | 336 GB/s (IQ3_S) / 364 GB/s (IQ4_XS) | — |
| perplexity (wikitext-2, 10×512 tokens, per position) | within **0.25 %** (IQ3_S) / **0.15 %** (IQ4_XS) | reference |
| decode 64K f16 / 131K q4_0 (IQ3_S) | 18.9 / 13.0 tok/s | — |
| MTP (NextN) draft acceptance | 86.7 % on natural text (llama.cpp's own driver: 87.5 %) | — |

## Validation

Every number above is backed by a gate in `tests/` or `scripts/`. The ones worth
running after a change:

```bash
./build/check-graph-gpu  <model.gguf> <oracle_dump> -        # per-node vs llama.cpp
./build/check-kvctx-gpu  <model.gguf> 65536 q4_0             # long context
./build/check-tokenizer  <model.gguf> tests/golden/tokenizer_ids.txt tests/tokenizer_corpus.txt
./build/check-chat       tests/golden/chat_renders.txt
./build/check-eog        <model.gguf> tests/golden/eog_ids.txt
./build/check-stream     && ./build/check-sampler && ./build/check-dequant-gpu <model.gguf>
./scripts/check_golden_run.sh                                # generation is reproducible
./scripts/compare_llama_greedy.sh 32                         # greedy == llama.cpp, token by token
./scripts/compare_ppl.sh <model.gguf> 10                     # perplexity vs llama.cpp, per position
./scripts/check_attn_split.sh                                # split-KV == unsplit on real text
```

The oracles (`build/oracle-*`) are the only binaries that link llama.cpp; they
produce the committed references under `tests/golden/` and are never used at
runtime by the engine.

## Known limitations (v1)

- **Prefill is not batched**: the prompt is processed one token at a time, so a long
  prompt costs `n_tokens × decode_time` (27.9 tok/s) instead of llama.cpp's 440 tok/s.
  Decode is at ~70 % of the reference; prefill is the gap that matters for long
  prompts. Measured numbers and the rejected alternatives are in
  `docs/medicoes-m5.md`.
- **Long context was attention-bound; that is now fixed** (M7): decode is 23.7 tok/s
  at 4K, 24.1 at 16K, 19.0 at 64K (f16 KV) and 13.2 at 131K (`q4_0`), against
  22.1/13.7/4.2/2.3 before. Vectorized cache loads, 32 warps/CTA and splitting the
  key range across CTAs (with an online-softmax merge) bought 1.1-5.8× depending on
  context; `docs/medicoes-m7.md` has the breakdown, and the split path is validated
  against the unsplit one by `scripts/check_attn_split.sh`.
- Use `f16` KV wherever it fits: it is 38 % faster than `q4_0` at the same context.
- The CLI is single-turn (`--chat` renders one system+user turn); multi-turn
  rendering is implemented and tested in `chat_render`, the CLI just does not offer
  a conversation file yet.
- `output.weight` is required (no tied-embedding fallback) and only `gfx1201`
  builds/runs.
- **MTP (block 64) works but does not pay yet**: `--mtp` drafts with the NextN head
  at 86.7 % acceptance and reproduces plain greedy output exactly, but every mode
  still runs one trunk forward per committed token, so it is 9-10 % *slower* than
  plain greedy. With batched verification it would be ~2.5× (measured draft step
  2.24 ms vs 35-36 ms trunk step). See `docs/mtp.md`.
- The server is single-request and has no keep-alive or prefix-cache reuse; see
  `docs/servidor-openai.md` for the full list of what it does not implement.

## Docs

- `SPEC.md` — what v1 will and will not have.
- `PLAN.md` — M0–M5 milestones with acceptance criteria and code references.
- `docs/rdna4-gfx1201-referencias-amd.md` — official AMD documentation index.
- `docs/referencias-upstream-gfx1201-qwen35.md` — gfx1201/Qwen3.5 mentions across
  vLLM, SGLang, hipfire and llama.cpp.
- `docs/kernels-ia-gfx1201.md` — kernel programming guide (AMD + community reports).
- `docs/gguf-qwen-quantizacao-llamacpp.md` — GGUF layout for Qwen + quantization.
- `docs/baseline-vulkan-iq3s.md` — llama.cpp baselines (Vulkan, IQ3_S, 32K ctx) to beat.
- `docs/medicoes-m5.md` — every measured number: reference baseline, per-token cost
  decomposition, VRAM/context/KV tables for both GGUF files, perplexity per chunk.
