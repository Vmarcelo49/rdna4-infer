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
./build/rdna4-infer bench -m model.gguf --prefill 512 --prefill-reps 3   # prefill only, best of 3
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
| 64K, `f16` | 11.133 GB | 4.295 GB | 25.770 GB | 0.302 GB | 15.74 GB — **does not fit: see below** |
| 128K, `q4_0` | 11.133 GB | 2.416 GB | 14.496 GB | 0.302 GB | 13.87 GB |

Two facts drive the design: the GQA ratio is 6:1, so with one CTA per query head each KV row
is read **6 times** (the difference between the two KV columns), and the GDN recurrent state
is 302 MB per token of pure round-trip traffic. The split-KV attention trades exactness for
parallelism and is gated numerically; the split partials and the logits copy are rounding
errors in this budget (`docs/kv-memoria-desenho.md` §4).

**The KV budget is not a table entry, it is a cliff** (measured, `docs/medicoes-banda-e-gargalos.md`
§2.4). IQ3_S + `f16` KV fits up to **48K** (1.30 GiB free); at **64K it is over the limit** —
0.30 GiB free, ~2.1 GB spill to GTT (`gtt_used` 2557 MB against 466-472 MB when it fits) and
decode collapses from ~18 to **2.16 and 7.97 tok/s** in two identical runs, with the lock held
and `/dev/kfd` empty. Phases that read a lot fail together (attention 13.3×, matvecs 6-23×)
while the cheap ones do not move: the signature of traffic over the system bus, not of
contention. Above ~56K the supported configuration is **KV `q8_0`**, which is also *faster*
than `q4_0` at 64K (18.28 vs 18.10-18.17 tok/s in the measurement front's window; attention
19.87 vs 21.03 ms) and leaves 2.18 GiB free. (The **speed** argument was weaker than it looked: `q8_0` measured 18.28 against
18.10-18.17 for `q4_0` in two *separate* runs, i.e. inside the 2-3 % session-to-session
variance, not in an interleaved A/B. What justifies the recommendation is the VRAM headroom and
the doubled precision, not a speed win — the attention there is issue-bound in the dequant, so
the extra precision is close to free, which is a weaker and truer claim.)

## Measured numbers

Final table, measured on the merged tree in one quiet window (2026-09-14, nothing else on
the GPU, `scripts/gpu-lock.sh` held), IQ3_S, `f16` KV unless stated; the exact command of
every row is right below it. The llama.cpp column is `llama-bench -p 64 -n 64 -r 3`
(Vulkan/RADV, hot model), the same reference `docs/medicoes-m5.md` uses, so the prompt length
is comparable (64 tokens).

| | this engine | llama.cpp Vulkan, same machine |
|---|---|---|
| decode, short context (positions 5-37) | **30.4 tok/s** (best of 3; 29.5 mean) | 40.0 ± 0.02 tok/s (tg64; 39.7 in M5) |
| decode, end of 4K `f16` | **29.3 tok/s** (34.2 ms/token, 325 GB/s of weights) | 37-38 tok/s at 32K (`llama-cli`) |
| decode, 16K `f16` | **26.7 tok/s** (321 GB/s of weights) | — |
| decode, 64K `f16` | **does not fit**: 2.1 GB spill to GTT, 2.16-7.97 tok/s | — |
| decode, 64K `q8_0` | **19.3 tok/s** (13.75 GiB in use; the measurement front's window saw 18.28 on the same configuration) | — |
| decode, 128K `q4_0` | **14.0 tok/s** (13.88 GiB in use) | — |
| prefill, batched (N≤16) | **123.9 tok/s** on a 512-token prompt (73.05 at the start of the night; +69.7 % = ~+43 % from batching the per-token scaffolding and +18.6 % from the `float4` state load in `delta_rule`, both bit-exact) | 575 ± 65 tok/s (pp64, re-measured; 440 ± 77 recorded in M5 — pp64 is noisy), 1143 ± 30 (pp512) |
| weight bandwidth, end to end | **325 GB/s at 4K = 51 %** of the measured 633 GB/s DRAM roofline | ≥442 GB/s (derived) |
| weight bandwidth, matvec alone | **436 GB/s = 69 %**; the LM head reaches 620 GB/s = 98 % | — |
| IQ4_XS decode, 4K `f16` | 27.0 tok/s | — |
| perplexity (wikitext-2, 10×512 tokens, per position) | within **0.25 %** (IQ3_S) / **0.15 %** (IQ4_XS) | reference |
| MTP (NextN) draft acceptance | 86.7 % on natural text (output identical) | 87.5 % (its own driver) |

```bash
# the engine rows, in order (one command each):
./build/rdna4-infer bench -m IQ3_S.gguf -n 32 --reps 3
./build/rdna4-infer bench -m IQ3_S.gguf -n 32 --reps 3 --ctx-size 4096  --start-pos 4000  --fill-cache
./build/rdna4-infer bench -m IQ3_S.gguf -n 32 --reps 3 --ctx-size 16384 --start-pos 16256 --fill-cache
./build/rdna4-infer bench -m IQ3_S.gguf -n 32 --reps 3 --ctx-size 65536 --start-pos 65408 --fill-cache \
    --cache-type-k q8_0 --cache-type-v q8_0
./build/rdna4-infer bench -m IQ3_S.gguf -n 16 --reps 2 --ctx-size 131072 --start-pos 131000 --fill-cache \
    --cache-type-k q4_0 --cache-type-v q4_0
./build/rdna4-infer bench -m IQ3_S.gguf -n 32 --reps 3 --prefill 512
./build/rdna4-infer info -m IQ3_S.gguf --ctx-size 65536 --cache-type-k f16 --cache-type-v f16   # insufficient VRAM
```

Two notes that keep the table honest. **The window matters**: these numbers are 8-9 % *better*
than the M7/M8 records at the same positions (4K: 26.8 → 29.3), while the controlled A/B of the
tuning changes that landed since measured +1.2 % there — the rest is machine state, because the
earlier records were taken while other jobs shared the GPU. **The prefill row deserves a second
look**: the gap is real (73 vs 440-1143) and it *widens* with prompt length, since a long prompt
amortizes per-call overhead while our per-token scaffolding does not. Sources: `docs/medicoes-m5.md` (both files, KV types,
contexts, perplexity), `medicoes-m7.md` (long context), `medicoes-m8.md` (batched prefill,
MTP projection), `medicoes-banda-e-gargalos.md` (per-phase time and the traffic budget).

## Where a token goes

Measured with HIP events inside the real graph (`bench-phases-gpu`), context 4K, `f16` KV,
best pass on a quiet window, minus the measured cost of each event mark (1479 marks × 4.03 µs
= 5.95 ms, i.e. level-2 instrumentation inflates a token by 16 % if you do not subtract it).
Full detail in `docs/medicoes-banda-e-gargalos.md` §1.

| phase | ms/token | share | lever measured |
|---|---|---|---|
| weight matvec, trunk (496 launches) | 23.51 | 65.3 % | issue-bound at 436 GB/s: 5.9 ms recoverable in the `vec_dot` bodies |
| GDN recurrence (`delta_rule` 3.97 + conv + scalars + norms) | 5.63 | 15.6 % | 48 CTAs = 9 % occupancy; a 4-threads-per-row rewrite is worth up to ~3 ms |
| norms / rope / elementwise | 2.96 | 8.2 % | fusing `rms_norm`+`quantize`, the GDN scalars and `kv_write` 8→1 ≈ 1.1 ms |
| LM head (1 launch) | 1.43 | 4.0 % | at 98 % of the read roofline (874 MB/token, unavoidable) |
| attention (16 layers, split) | 1.38 | 3.8 % | memory-latency bound here; at 64K it becomes issue-bound in the KV dequant |
| activation quantization (257 launches) | 0.88 | 2.4 % | the q8_1 activation is also the entire numerical error of the matvec (see below) |
| logits copy + sampler | 0.65 | 1.8 % | 993 KB per token; the greedy path now takes the argmax on the device |
| **total** | **36.4** | | clean measurement at the same position: 35.65 ms; the head appears once |

The 1 940 launches per token cost ~4.0 ms (11 %) at the measured 2.2-3.5 µs dispatch floor —
but measure before believing it: replaying the same launch sequence as a HIP graph bought only
1.06× (1.4 ms), because most of those launches overlap with real kernels. The irreducible part
is the ~1 440 *small* kernels, which are at the floor (2.77-3.16 µs against 2.20 µs for an
empty kernel).

## Numerical accuracy of the quantization

`docs/quants-precisao.md` measures each format against an fp64 dot over dequantized weights,
and separates the two error sources. The result is counter-intuitive and useful:

- **The matvec arithmetic is not the error.** For all 14 formats the total error of one
  projection is the error of the `q8_1` activation quantization: **3.6e-3 to 4.0e-3**. The
  kernel arithmetic itself is 6.4e-8 to 1.0e-7 for the nine exact formats, and the worst
  truncating format is `iq1_s` at 4.4e-4.
- **Between formats, weight distance** (294 tensors, IQ3_S vs IQ4_XS): `iq3_s`↔`iq4_xs`
  1.19e-1, `iq4_xs`↔`q5_k` 6.5e-2, `q4_k`↔`q5_k` 6.4e-2 — i.e. IQ4_XS represents these
  tensors with about half the error of IQ3_S.
- **End to end** (perplexity on wikitext-2, per chunk, against llama.cpp): IQ3_S
  0.185/0.105/0.008/0.018 %, IQ4_XS 0.053/0.015/0.099/0.116 %. "IQ4_XS is consistently
  better" does **not** hold: it wins on two chunks and loses on two, and its worst absolute
  NLL delta (0.873) is larger than IQ3_S's (0.458). What the extra 1.2 GiB of IQ4_XS buys is
  a smaller weight error that the activation error then hides.
- Measured recommendation: **IQ3_S + `f16` KV up to 48K**, **IQ3_S + `q8_0` KV above ~56K**.
  IQ4_XS does not even initialize at 48K with `q4_0` KV or 32K with `f16` (`hipMalloc`
  measured), so it is not the file to use for long context on this card.

## The 131K target, as measured tonight

The night's target was **131K of context with the KV cache quantized to `q5_0` (K) / `q4_1`
(V) and MTP delivering a real speedup**. State of each part, with the number that supports it:

- **KV `q5_0`/`q4_1` implemented and measured at 131K**: 13.73 tok/s, 14.25 GiB of VRAM in
  use, 1.68 GiB free, no GTT spill. The two new formats were validated byte-for-byte against
  llama.cpp's reference quantizers (0 differing bytes on 8 patterns) and their load paths match
  the reference dequantization exactly (`max|gpu−cpu| = 0`).
- **At 131K the KV format does not buy speed** (13.73 to 14.34 tok/s across a 2× range of
  cache sizes): the decode is limited by attention latency/occupancy (165-172 GB/s effective
  on the weights, 27-29 % of peak). So the choice is quality and headroom, and `q8_0`/`q4_1`
  (best measured mean KLD, 0.93 GiB free) is the documented alternative for non-MTP use.
- **`f16`/`f16` at 131K does not allocate** (`hipMalloc failed`): 8 GiB of KV do not fit.
- **Prefill, which is what makes a long context usable at all**, went from 73.05 to
  **123.9 tok/s** (+69.7 %, bit-exact) by batching the per-token scaffolding; a real 131K
  prompt therefore costs ~18 minutes before the first token. Everything about the 131K decode
  numbers above is a *synthetic seeded cache* (`bench --start-pos --fill-cache`): the cost of
  decoding at that position, not the quality of a real 131K conversation.
- **Quality at long context**: the RoPE path is verified against the reference up to position
  262 143 (see limitation 5), and per-chunk perplexity against llama.cpp is ≤ 0.25 % at
  512-token windows. Quality with **real text** was probed to 16-24K tonight; a real 131K
  quality measurement does not fit in a night at 123.9 tok/s of prefill.

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
   against this engine's 436 GB/s on the trunk matvec (measured) and 325 GB/s end to end; the
   "402-421 GB/s" that used to be quoted here came from a *replay estimate*
   (`docs/rocm-estudo.md` §A.1), not from a measurement.

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
pre-fix checkout, so the "before" column is a measurement, not a claim). Warning hygiene: `-Wall -Wextra` is **on** for both shipping targets (`rdna4-infer` and
`rdna4_serve`) and the build is clean — the 98 × `-Wsign-compare` and three unused parameters
that used to be the reason for leaving it off are fixed (the split kernel's tail write is now a
strided loop, and the parameters are `[[maybe_unused]]`). The test binaries are deliberately
left out: ROCm 7.2 marks `hipError_t` `[[nodiscard]]` and their 422 ignored returns are test
noise, not engine risk.

## Known limitations (v1)

Ordered by how much they cost the user, with the number that justifies each. Nothing here is
"should be fine" — each line is a measurement or a code fact with a pointer.

1. **Prefill is still the weak number: 123.9 tok/s at 512 tokens against llama.cpp's 1143 (a
   9× gap)**, so a 4K prompt costs ~33 s before the first token (was 56 s at the start of the
   night). That +69.7 % decomposes as ~+43 % from batching the per-token scaffolding and
   +18.6 % from the `delta_rule` state load — the review caught the two docs attributing the
   whole of it to different changes, which would have made whoever inherited the lever
   overestimate the second one by 3.7×. The matvec is *not* the whole
   story: inside `forward_batch` the weight pass is shared across the 16-token chunk, but the
   attention, the GDN recurrence, the norms and the elementwise chains still run **per token**
   (~11 ms/token of scaffolding, measured in `docs/medicoes-banda-e-gargalos.md` §1), which is
   ~80 % of prefill time. Even with an infinitely fast matvec that scaffolding caps prefill
   near 145 tok/s — a bound the night already beat by batching, which moved the ceiling too.
   What is left is the batched matvec itself: it re-runs the dequantization/LUT/sign assembly
   once per token (`matvec.cuh` `matvec_kernel_batch`), reading 12 GB per 16-token chunk at
   **109 GB/s** against 446 GB/s for the same weights on the per-token path. Hoisting the
   block dequantization out of the token loop is bit-exact and worth an estimated 2-2.5×
   (≈200 tok/s); past that the route is a tiled MMQ-style int8 kernel (weights in LDS, WMMA
   int8 — possible on this card, gives up bit-exactness, needs numeric gating).
2. **The KV cache has a cliff, not a curve — and at 131K the format buys quality, not
   speed.** IQ3_S + `f16` KV fits to 48K; at 64K it spills ~2.1 GB into GTT and decode
   collapses from ~18 to **2.16-7.97 tok/s** with no error message, and `f16` at 131K does
   not even allocate (`hipMalloc failed` — 8 GiB of KV). Measured at 131K (16 decode tokens,
   clean window):

   | K / V | decode | VRAM in use | free | GTT |
   |---|---|---|---|---|
   | `q5_0`/`q4_1` (**default**) | 13.73 tok/s | 14.25 GiB | 1.68 GiB | 30 MB |
   | `q8_0`/`q4_1` (best quality) | 14.32 tok/s | 15.00 GiB | 0.93 GiB | 30 MB |
   | `q8_0`/`q8_0` | 14.34 tok/s | 15.87 GiB | **0.05 GiB** | 75 MB |
   | `q4_0`/`q4_0` | 14.08 tok/s | 13.87 GiB | 2.05 GiB | 75 MB |
   | `f16`/`f16` | — | — | — | does not allocate |

   The spread from the smallest to the largest cache that runs is **1.9 %**: at 131K the
   decode is limited by attention latency/occupancy (165-172 GB/s effective on the weights,
   27-29 % of peak), not by KV bandwidth, so the format choice is quality plus headroom.
   Default is `q5_0`/`q4_1` because it leaves 1.68 GiB for the MTP state planes. The quality
   question was then measured instead of assumed — mean KL over the full vocabulary at 4096
   tokens of real text (256 probes, `docs/journal-kv.md` §7.1):

   | K / V | mean KL (nats/pos) | PPL | greedy token changed |
   |---|---|---|---|
   | `f16`/`f16` (reference) | 0 | 5.91711 | 0/256 |
   | `q8_0`/`q8_0` | **0.000492** | 5.92405 | 3/256 |
   | `q8_0`/`q4_1` | 0.001443 | 5.93386 | 5/256 |
   | **`q5_0`/`q4_1`** (default) | 0.001715 | 5.95800 | 4/256 |
   | `q5_0`/`q4_0` | 0.002118 | 5.93234 | 7/256 |
   | `q4_0`/`q4_0` | 0.003208 | 5.93976 | 8/256 |

   Isolating each axis: **K `q8_0` is 16 % lower KL than K `q5_0`** (0.001443 vs 0.001715, with
   V fixed) and **V `q4_1` is 19 % lower KL than V `q4_0`** (0.001715 vs 0.002118, with K
   fixed) — the same directions llama.cpp's PR #21038 measured, now reproduced here. So the
   night's premise holds: `q4_1` for V pays, and `q4_0` must not be used for K (worst of all,
   0.003208). `q8_0`/`q4_1` is the better-quality option (16 % less KL for 0.75 GiB more) for
   anyone not spending that headroom on MTP. Note that **perplexity does not order these
   formats**: `q5_0`/`q4_0` beats `q5_0`/`q4_1` on PPL (5.932 vs 5.958) while having 23 % more
   KL and nearly double the greedy divergence — PPL is blind to this, KL is not.
3. **MTP (block 64) works and is exact, but whether it pays depends on the *batched* path
   being fast — and on the final tree it does not, yet.** Batched verification exists and is
   correct: `--mtp` output is **byte-identical to plain greedy** (`md5` of stdout, every mode,
   every prompt tested), the recurrent-state rollback is bit-exact (`max|d| = 0`) at 0.54 ms
   per snapshot+restore pair, and a verified row costs 8.0 ms against 34.4 ms for a whole
   per-token step. Two sets of measurements disagree about the payoff, and both are in the repo:

   | measured on | greedy | `--mtp --draft 2` | `--draft 3` | note |
   |---|---|---|---|---|
   | the MTP branch (before the kernel merge) | 29.37 | **35.69 = 1.22×** | 30.11 = 1.03× | 3996-token prompt, 68 % acceptance |
   | the final merged tree (this table's tree) | 32.27 | **31.03 = 0.96×** | 25.69 = 0.80× | 3683-token prompt, same md5 output |

   They are both real, and the difference has a cause: the kernel work landed **after** the MTP
   measurement and made the **per-token** path 17 % faster without making the **batched** path
   faster — the batched matvec still reads 12 GB per 16-token chunk at 109 GB/s against 446 GB/s
   for the same weights per token. MTP's verify is a batched forward, so its advantage shrank
   from 1.22× to 0.96× while the machine got faster. That also explains the one case where the
   mechanism clearly wins today: on code-like text the draft acceptance is 95 % and the same
   code measures **2.03×** (61.7 vs 30.1 tok/s); on prose the acceptance is 68 % and every
   rejected round pays a second trunk pass.
   **Conclusion for the next session:** MTP's multiplier is gated on the batched matvec, not on
   the MTP machinery — fix `matvec_kernel_batch` (the open item below) and the 1.2-2.0× becomes
   visible on prose too. `docs/mtp.md`, `docs/journal-mtp.md` §9.
4. **Long context works but the GQA re-read is still per query head**: with a 6:1 ratio each
   K/V row is read **6 times** per token (25.77 GB of logical KV traffic at 64K against
   4.295 GB of unique bytes). The attention kernel saturates ~1.35 TB/s of L2, so the L2 hides
   most of it — but a Vulkan-style grouped attention (one row loaded once for the 6 query
   heads) is a known, unclaimed win. Decode was attention-bound before M7 (4.2 tok/s at 64K);
   `docs/medicoes-m7.md` has the fix and the curve. The split path is not bit-exact and its
   deviation is now measured on real text as a ladder: **5.1989** unsplit, **5.2054** at 4
   splits (+0.125 %), **5.2114** at 16 splits with the wide CTA (+0.240 %) — all inside the
   0.5 % gate, but the wide end costs twice the deviation of the narrow one, which is the
   price of the +5 % kernel speed it buys at 131K.
5. **"Runs at 131K" is not "is good at 131K", and we say exactly which half is measured.**
   (a) *Implementation*: our RoPE was diffed against the real `ggml_rope_multi` up to position
   262 143 (`check-rope-long-gpu`) — relative L2 ≤ 1e-3 at the far end, explained by one ulp of
   theta, against a sensitivity control showing that a misconfigured YaRN would be 24-42 %
   off. No rope scaling is applied by the GGUFs or by llama.cpp for this model, and ours
   matches. (b) *Cost*: the 131K numbers above use a **synthetic seeded cache**
   (`bench --start-pos --fill-cache`), so they measure the cost of decoding at that position.
   (c) *Quality*: measured with real text only up to **16-24K**, because a real 131K prefill
   at the current 74.9 tok/s takes ~29 minutes per run. Per-chunk perplexity against llama.cpp
   is ≤ 0.25 % at 512-token windows. A 131K quality claim beyond that is not supported by
   tonight's data, and the report says so.
6. **The attention kernel's *unsplit* variant ignores its `WPB` template parameter**
   (`include/rdna4/attn.cuh`), so any warp-per-CTA sweep done through it measured redundant
   warps. It affects diagnostics, not the shipped split path (which uses the parameter), and
   the shipped constant is 8.
7. **`--mtp` and `--chat` are CLI-only**: the OpenAI-compatible server does not expose
   speculative decoding, and it is single-request with no keep-alive and no prefix-cache
   reuse. It validates request parameters (including non-finite ones) but does not implement
   `logprobs`, `n>1`, tools or vision — full list in `docs/servidor-openai.md`.
8. **Smaller, documented, deliberately not fixed**: the CLI is single-turn (`--chat` renders
   one system+user turn; multi-turn rendering exists and is tested in `chat_render`);
   `output.weight` is required (no tied-embedding fallback); only `gfx1201` builds and runs;
   `--ctx-size` is capped at 2^24 (the graph takes an `int`); 47 buffer sizes are computed in
   `int` and widened afterwards (safe for this model, latent for another — audit finding M10);
   and the test binaries still ignore 422 `[[nodiscard]]` HIP returns (test noise, not engine
   risk — the engine's own 19 `hipMalloc` sites and 41 kernel launches are all checked).

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
  `medicoes-banda-e-gargalos.md` — every measured number, per milestone, plus the per-phase
  token budget, the measured DRAM roofline and the GTT cliff.
- `docs/quants-precisao.md` — per-format accuracy, activation error vs kernel error, and
  format-to-format weight distance.
- `docs/mtp.md`, `docs/servidor-openai.md`, `docs/gpu-queue.md`, `docs/agentes-paralelos.md`
  — the MTP study, the server surface, and the rules for running several agents on one GPU.
