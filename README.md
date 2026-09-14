# rdna4-infer

A from-scratch inference engine for **AMD RDNA4 (RX 9070 XT, `gfx1201`)**, focused on a
single workload: **Qwen3.8 27B dense in GGUF**, quantized to fit 16 GB of VRAM.

One GPU, one architecture, one model family — so every layer of the stack (loader, quant
kernels, attention, graph, sampler) is written for exactly that target and measured against
real numbers. **llama.cpp is not a dependency of the runtime**: it is used two ways only,
both offline — as the *oracle* that produces the golden references the tests compare against
(`build/oracle-*`, the only binaries that link it), and as the MIT-licensed reference for
block layouts (`include/rdna4/quants.h`) and batched-kernel organization, always with the
source line cited in the comment.

## Objectives

- Run `Qwen3.8-27B` (`UD-IQ3_S` 12 GB primary, `UD-IQ4_XS` 14 GB secondary) fully in VRAM
  on a stock RX 9070 XT, via a `run` CLI with streaming output or behind an
  **OpenAI-compatible HTTP server** (`serve`).
- HIP-only, `gfx1201` only, C++17, no inline assembly, no library to install.
- **Bit-exactness is a safety property**, not a slogan: the batched prefill path, the
  batched matvec and `proj_qq` must be bit-identical to the per-token path, and there is a
  gate for each (`check-batch-gpu`, `check-matmul-gpu`). Where an optimization cannot be
  exact (split-KV attention), it gets a numeric gate on real text instead
  (`scripts/check_attn_split.sh`), and the divergence is documented in percent.
- Every tuning decision is measured on this card and recorded with the command that
  produced it; hypotheses that failed are recorded too, with the number that killed them.

## Non-goals (v1)

Other models or MoE variants, quantization tooling, other GPUs/OSes, offload, multi-GPU,
and speculative decoding (measured, declined — see *Known limitations*). No chat UI: the
server speaks the OpenAI API so existing harnesses can be pointed at it. Full scope in
`SPEC.md`.

## Layout

```
├── CMakeLists.txt        # HIP-only, --offload-arch=gfx1201 (compile AND link)
├── src/main.hip          # CLI: info | run | bench | ppl | tokenize | serve
├── src/backend/          # gguf, loader, model, tokenizer, unicode, sampler, chat
├── src/server/           # http, json, serve (the OpenAI-compatible surface)
├── include/rdna4/        # headers + kernels (graph, matvec, attn, gdn, kv, dequant, nn)
├── tests/                # check/bench binaries (CPU + GPU) and the oracle tools
├── scripts/              # build/run helpers, the GPU lock and the acceptance gates
├── docs/                 # research notes + one measurement doc per milestone
└── reference/            # oracle captures (gitignored)
```

Everything builds into one CLI binary, a `librdna4_serve.so` and one binary per check.
There is no runtime dependency beyond ROCm.

## Build

```bash
source scripts/rocm-env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
./build/rdna4-infer            # device + VRAM check, no model needed
```

`docs/build-repro.md` pins the whole thing (ROCm 7.2.4, `amdclang++` 22.0.0git, CachyOS) and
records what a clean out-of-tree build costs (~5 min, ~2 GB). The pitfalls that bite on a
fresh machine, all verified there:

- `--offload-arch=gfx1201` is needed on **compile and link**; CMake's HIP support does not
  add it for you.
- `CMAKE_HIP_FLAGS_RELEASE` must be set **in `CMakeLists.txt`** — passing
  `-DCMAKE_HIP_FLAGS_RELEASE=` on the command line is silently ignored, and without it the
  device code is built at `-O0` (100× slower kernels, no error).
- HIP translation units must be `.hip`; the quant kernels are compiled **without**
  fast-math on purpose (bit-exactness against the CPU oracle).
- ROCm 7.2 headers mark `hipError_t` `[[nodiscard]]`: 422 of the 433 default-build warnings
  in this tree come from test code ignoring it, not from the engine.

## Usage

```bash
# generate text (stdout is only the generated text; stats go to stderr)
./build/rdna4-infer run -m model.gguf -p "The capital of France is" -n 64 --greedy
./build/rdna4-infer run -m model.gguf --chat -p "What is 2+2?" -n 128 --temp 0.7 --seed 42

# benchmark (decode/prefill tok/s, VRAM, effective bandwidth)
./build/rdna4-infer bench -m model.gguf -n 64 --reps 3
./build/rdna4-infer bench -m model.gguf --ctx-size 131072 --start-pos 131000 \
    --cache-type-k q4_0 --cache-type-v q4_0 --fill-cache     # long-context decode

# OpenAI-compatible server (chat completions with SSE streaming, /v1/models, /health)
./build/rdna4-infer serve -m model.gguf --port 8080 --ctx-size 8192
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"hi"}],"stream":true}'

# perplexity on a corpus (llama-perplexity-compatible tiling)
./build/rdna4-infer ppl -m model.gguf -f corpus.txt --ctx-size 512 --stride 512 --chunks 10

# tokenize / inspect
./build/rdna4-infer tokenize -m model.gguf -p "hello"
./build/rdna4-infer info -m model.gguf --ctx-size 65536 --cache-type-k q4_0 --cache-type-v f16
```

`run` flags: `-m -p -n --chat --system --no-thinking --reasoning-effort --temp --greedy
--top-k --top-p --min-p --repeat-penalty --repeat-last-n --seed --ctx-size
--cache-type-k/-v --mtp -v --no-stats`. Sampler defaults come from the GGUF's
`general.sampling.*` metadata (this model ships `top_k 20`, `top_p 0.95`, `temp 1.0`); any
flag wins over the metadata.

KV cache types: `f16` (default, matches llama.cpp), `q8_0`, `q4_0`, `f32`, chosen
independently for K and V — the VRAM check counts both (`info` prints the breakdown).
`q4_0` is what makes 128K context fit in 16 GB.

Exit codes: `0` success, `1` usage/IO/model error, `2` no `gfx1201` device, `3` insufficient
VRAM.

## Quantization coverage

The engine implements **15 dtypes** — `f32` and the 14 quantized types that the two UD files
actually contain (`docs/quants-inventario.md` has the byte-level inventory, generated from the
two files' headers). Each quantized type has a fused GEMV, a batched variant and a dequant
kernel, all gated against llama.cpp's CPU dequant (`check-dequant-gpu`) and its CPU
`vec_dot` (`check-matvec-gpu`).

Per-token weight traffic — what the decode actually reads, i.e. excluding `token_embd.weight`
(one row is read, not the matrix) and the whole `blk.64.*` MTP block:

| | `UD-IQ3_S` | `UD-IQ4_XS` |
|---|---|---|
| file / tensors | 12.030 GB, 866 | 14.242 GB, 866 |
| **read per token** | **11.122 GB** | **13.334 GB** |
| largest shares | `iq3_s` 33.4 %, `iq4_xs` 23.3 %, `iq3_xxs` 16.8 %, `q5_k` 8.9 % | `iq4_xs` 54.1 %, `q5_k` 15.7 %, `iq3_s` 12.1 %, `q4_k` 9.5 % |
| dtypes present | 15 | 13 (no `iq2_xxs`, no `iq1_s`) |
| fits | 128K ctx with `q4_0` KV (13.86 GiB) | 32K with `f16` KV (15.66 GiB); 128K does not fit |

Correction of record: the per-type "share of model" column in `docs/medicoes-m6.md` was
computed over `total_bytes()` (the whole file) instead of the per-token stream, which
overstated `q3_k` (7.9 % vs the real 3.6 %) and `q6_k` (2.9 % vs 0.04 %) and understated the
`iq*` types. `docs/quants-inventario.md` §3.3 reproduces the wrong numbers by arithmetic to
prove where they came from, and gives the corrected table.

## Kernels: what runs on the hardware

The short version (`docs/qwen-kernels.md` is the audit, per kernel, with the ISA):

- **There is no matrix-core instruction to fall back from.** `gfx1201` has **no MFMA**: the
  intrinsic does not compile (`needs target feature mai-insts` — a build error, not a silent
  downgrade). `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` *does* compile, to
  `v_wmma_i32_16x16x16_iu8`.
- The only hardware arithmetic intrinsic in this engine is
  `__builtin_amdgcn_sudot4` → `v_dot4_i32_iu8` (96 sites, the `vec_dot_*_q8_1` bodies), plus
  `__builtin_amdgcn_perm` → `v_perm_b32` (24) and `s_prefetch_data`. Zero inline asm.
- **The decode path is not falling back**: `v_dot4_i32_iu8` is the right instruction at full
  rate, and the ISA of every hot kernel contains no WMMA/MFMA. What the ISA *does* show is
  where the real gaps are: the batched/prefill matvec runs 512 `dp4a` and **0 WMMA** per
  kernel (llama.cpp's MMQ uses WMMA int8 for the same tensors), and the attention kernel uses
  28 scalar `fmac_f32` and **no packed f16** even with an `f16` KV cache — `v_pk_fma_f16` and
  `v_dot2_f32_f16` run at 1.5× the fp32 FMA rate on this card.
- WMMA cannot help decode at all: it is `16×16×16`, and decode is a GEMV with M=1. It only
  helps at `N ≥ 8-16`, which is the prefill path — the milestone-sized item below.
- Dequantizing weights to `f16` on load is not an option on this card: `iq3_s` at 3.44 bpw
  would become 16 bpw, so the 11.1 GiB trunk would need ~52 GiB.

## Memory and the KV cache

The cache is contiguous per layer, addressed directly, with no paging and no indirection
(`docs/kv-memoria-desenho.md`, which also prices the alternatives). Per decode step:

| context (IQ3_S) | weights | KV read (DRAM) | KV read (logical/L2) | GDN state | total DRAM |
|---|---|---|---|---|---|
| 4K, `f16` | 11.133 GB | 0.268 GB | 1.611 GB | 0.302 GB | 11.71 GB |
| 64K, `f16` | 11.133 GB | 4.295 GB | 25.770 GB | 0.302 GB | 15.74 GB |
| 128K, `q4_0` | 11.133 GB | 2.416 GB | 14.496 GB | 0.302 GB | 13.87 GB |

Two facts drive the design: the GQA ratio is 6:1, so with one CTA per query head each KV row
is read **6 times** (the difference between the two KV columns), and the GDN recurrent state
is 302 MB per token of pure round-trip traffic. The split-KV attention trades exactness for
parallelism and is gated numerically; the split partials and the logits copy are rounding
errors in this budget (`docs/kv-memoria-desenho.md` §4).

## Measured numbers

Every number was measured on this machine and has its exact command in the docs named below
(the llama.cpp column is `llama-bench -p 512 -n 128 -r 3`, hot model, from
`docs/baseline-vulkan-iq3s.md`). IQ3_S unless stated.

| | this engine | llama.cpp Vulkan, same machine |
|---|---|---|
| decode, 4K `f16` | 29.3 tok/s (start of 4K) / 26.8 (end of 4K) | 39.7 tok/s |
| decode, 16K / 64K `f16` | 24.3 / 18.9 tok/s | — |
| decode, 128K `q4_0` | 13.0 tok/s | — |
| prefill, batched (N≤16) | 70.2 tok/s (512-token prompt: 69.5) | 440 tok/s (batched) |
| weight bandwidth (11.122 GB/token ÷ ms per token) | ~326 GB/s | ≥442 GB/s (derived) |
| 64K `q4_0` decode | 17.9 tok/s (was 4.2 before M7) | — |
| IQ4_XS decode, 4K `f16` | 27.0 tok/s | — |
| perplexity (wikitext-2, 10×512 tokens, per position) | within **0.25 %** (IQ3_S) / **0.15 %** (IQ4_XS) | reference |
| MTP (NextN) draft acceptance | 86.7 % on natural text (output identical) | 87.5 % (its own driver) |

Sources: `docs/medicoes-m5.md` (both files, KV types, contexts, perplexity),
`medicoes-m7.md` (long context), `medicoes-m8.md` (batched prefill, MTP projection),
`medicoes-banda-e-gargalos.md` (per-phase time and the traffic budget).

## How it compares to llama.cpp's Vulkan backend

`docs/vulkan-vs-hip.md` is a code-to-code comparison (shader by shader, dispatch by
dispatch). The five findings that matter:

1. **For 77.7 % of this model's bytes, Vulkan uses no integer dot either.** The `*_q8_1`
   (MMQ int8) shader is generated only for legacy quants, k-quants and `mxfp4`; `iq2_*`,
   `iq3_*` and `iq4_xs` have no int8 path in GEMV or GEMM — their decode GEMV is fp32 FMA
   over fp32 activations. So this engine's `dp4a` is the *better* instruction and still
   loses on wall clock, which moves the investigation to the LUT, latency and dispatch gaps.
2. **The IQ lookup table lives in LDS on the Vulkan side** and is charged to the GEMM's
   shared-memory budget; here it is a `__device__` global and each `vec_dot` issues 8
   `global_load_b32`. Steal candidate, bit-exact.
3. **Vulkan's flash attention solves the 6× GQA redundancy at the workgroup level** (one K/V
   row loaded once for the 6 query heads of the group, with `split_k = 32`). This engine
   re-reads per query head. That is why the relative gap grows with context.
4. **Dispatch count**: ~1 940 launches per token here (497 matvec + 257 activation
   quantizations), against ~1 300-1 400 for the same graph there; the difference is a list of
   named fusions (MUL_MAT+ADD, SIGMOID_MUL/SILU_MUL, RMS_NORM_MUL, SSM_CONV_SILU,
   ROPE+VIEW+SET_ROWS). At the measured 3.5 µs per-launch floor that is ~6.8 ms/token, 20 %
   of a 4K token — the highest-value, lowest-risk item on the list.
5. **Derived effective bandwidth**: ≥ 442 GB/s (≈480 GB/s discounting the LM head and glue)
   against this engine's measured 402-421 GB/s.

Capability line measured on this machine (`llama-bench`, RADV):
`fp16: dot2 | int dot: 1 | matrix cores: KHR_coopmat` — coopmat1 *is* enabled on RADV
gfx1201, which closes the open question in `docs/vulkan-vs-hip.md` §6; it does not change
finding 1, because there is no coopmat shader for the types that carry 77.7 % of the bytes.

## Validation

Every number above has a gate. `scripts/check_all.sh` runs the whole set with the GPU lock
taken **once** (`--quick` skips the long-context gates); the individual commands:

```bash
# correctness against llama.cpp's own CPU/GPU oracles
GRAPH_LAST_TOKEN=1 ./build/check-graph-gpu <model.gguf> reference/oracle_prompt6_ub1_tok7_cpu.txt -
./build/check-dequant-gpu <model.gguf>      # 14 quantized types vs the CPU dequant oracle
./build/check-matvec-gpu  <model.gguf>      # fused GEMV vs the CPU vec_dot oracle
./build/check-nn-gpu      <model.gguf>      # device primitives (norms, rope, softmax, GDN)
./build/check-kvctx-gpu   <model.gguf> 65536 q4_0
# bit-exactness of the fast paths
./build/check-matmul-gpu  <model.gguf>      # batched matvec == N separate GEMVs
./build/check-batch-gpu   <model.gguf>      # batched prefill == per-token path
# tokenizer, chat template, EOG set, sampler, streaming, HTTP
./build/check-tokenizer <model.gguf> tests/golden/tokenizer_ids.txt tests/tokenizer_corpus.txt
./build/check-chat tests/golden/chat_renders.txt ; ./build/check-eog <model.gguf> tests/golden/eog_ids.txt
./build/check-sampler ; ./build/check-stream ; ./build/check-server-http
# end-to-end equivalence with the reference implementation
./scripts/compare_llama_greedy.sh 32        # greedy generation identical, token by token
./scripts/compare_ppl.sh <model.gguf> 10    # perplexity vs llama.cpp, per position
./scripts/check_golden_run.sh               # generation is reproducible run to run
./scripts/check_attn_split.sh               # split-KV == unsplit on real text (PPL)
./scripts/check_server.sh                   # the OpenAI surface, including 12 invalid-parameter cases
# loader hardening (CPU-only): forged GGUF headers must be rejected cleanly
./scripts/check_hardening.sh [pre_fix_worktree]
```

## Code quality and hardening

`docs/auditoria-qualidade.md` is a full audit (error handling on every HIP call site, memory,
style, dead code, build hygiene): 4 CRITICAL, 19 IMPORTANT, 11 cosmetic. The four CRITICAL
findings and three IMPORTANT ones are **fixed in this tree**, each with before/after evidence:

| finding | what it was | proof it is fixed |
|---|---|---|
| E1 | the DeepSeek EOG literal `"\x9Cend…"` was a single 0x9CE character (hex-escape maximal munch) | adjacent literals + `static_assert` on the 27-byte length |
| M7 | `doff + offset + bytes` wrapped, so a forged offset passed both geometry guards and the `fseek` read GGUF header bytes **as weights** | old loader: accepted, returned ASCII `qwen35` as `token_embd.weight` (`f32[0]=1.78e28`); now rejected with `offset/size sum overflows 64 bits` |
| M8 | a file-controlled string length went straight into `resize` | old loader: `exit 134`, `std::bad_alloc`; now `truncated or corrupt header` |
| H1 | `1e400` in a request became `temperature = inf` (uniform softmax) and `seed = inf` (UB cast) | all numeric fields need `isfinite` + bounds; 12 cases in `check_server.py` |
| M1 | `Graph::release()` freed 18 pointers without nulling them (double free on a second call) | members are nulled; `release()` is idempotent |
| M4 | the `serve` VRAM budget ignored `--cache-type-v` | `kv_cache_bytes(ctx, k, v)`, one source of truth; mixed K/V now counted (2.56 GiB where the old formula said 1.13 GiB) |
| M5 | `--ctx-size` was cast from `uint64_t` to `int` unvalidated | capped at `1..2^24` with a message, before any HIP call |

`scripts/check_hardening.sh` reproduces the loader proofs (it compiles the probe against a
pre-fix checkout, so the "before" column is a measurement, not a claim). Warning hygiene is
tracked per translation unit: the engine compiles with **0** warnings under `-Wall -Wextra`
except one line repeated in `attn.cuh` (98 × `-Wsign-compare`) and three unused parameters —
those are queued with the attention changes rather than silenced with a flag.

## Known limitations (v1)

- **Prefill is batched up to N=16** (bit-identical to the per-token path): ~70 tok/s against
  llama.cpp's batched numbers. The ceiling is not batching — it is the `vec_dot` inner loop
  (issue-bound per weight byte, measured in M6): even with an infinitely fast matvec the
  per-token scaffolding caps prefill near 145 tok/s. Reaching llama.cpp's range needs a tiled
  MMQ-style kernel with weights staged in LDS and wide int8 dots, i.e. giving up
  bit-exactness for a numerically-gated kernel. Milestone-sized, not done.
- **Long context**: decode used to be attention-bound (4.2 tok/s at 64K with `q4_0` KV); see
  `docs/medicoes-m7.md` for the fix (vectorized cache loads, more warps per CTA, split-KV)
  and the numbers. The GQA re-read is still per-query-head, which is the next known win.
- Prefer `f16` KV where it fits: it is measurably faster than `q4_0` at the same context,
  and `q4_0` exists to make 128K fit, not to be fast.
- **MTP (block 64) works but does not pay**: `--mtp` drafts with the NextN head at 86.7 %
  acceptance and reproduces plain greedy output exactly, but every mode still runs one trunk
  forward per committed token, so it is 9-10 % *slower* than plain greedy. With batched
  verification the measured projection is only 1.3-1.5×, so it was deliberately not built.
  `docs/mtp.md`, `docs/medicoes-m8.md`.
- The CLI is single-turn (`--chat` renders one system+user turn); multi-turn rendering exists
  and is tested (`chat_render`) but there is no conversation-file flag yet.
- `output.weight` is required (no tied-embedding fallback) and only `gfx1201` builds/runs.
- The server is single-request, with no keep-alive and no prefix-cache reuse; it validates
  request parameters but does not implement `logprobs`, `n>1`, tools or vision. Full list in
  `docs/servidor-openai.md`.

## Docs

- `SPEC.md` — what v1 will and will not have. `PLAN.md` — milestones with acceptance
  criteria, the measured outcome of each, and the pieces that were re-scoped by measurement.
- `docs/auditoria-qualidade.md` — quality audit and the fixes applied from it.
- `docs/rdna4-gfx1201-hardware-brief.md`, `docs/rdna4-gfx1201-referencias-amd.md`,
  `docs/kernels-ia-gfx1201.md`, `docs/referencias-upstream-gfx1201-qwen35.md` — the hardware
  and the upstream landscape, with sources.
- `docs/gguf-qwen-quantizacao-llamacpp.md`, `docs/quants-inventario.md`,
  `docs/qwen-kernels.md` — GGUF/quantization layout, the byte inventory of both UD files, and
  the per-kernel ISA audit.
- `docs/kv-memoria-desenho.md` — KV layout, round-trips and the per-token traffic budget.
- `docs/vulkan-vs-hip.md` — Vulkan vs HIP, code to code, with the steal list.
- `docs/rocm-estudo.md` — where the time goes, the launch floor, and the optimization
  ranking. `docs/autotuning-gfx1201.md` — the tuning space, what shipped, what was rejected.
- `docs/build-repro.md` — pinned versions, a clean build, and the traps.
- `docs/medicoes-m5.md`, `medicoes-m6.md`, `medicoes-m7.md`, `medicoes-m8.md`,
  `medicoes-banda-e-gargalos.md` — every measured number, per milestone.
- `docs/mtp.md`, `docs/servidor-openai.md`, `docs/gpu-queue.md`, `docs/agentes-paralelos.md`
  — the MTP study, the server surface, and the rules for running several agents on one GPU.
