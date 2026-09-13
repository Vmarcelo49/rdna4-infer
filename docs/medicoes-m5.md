# rdna4-infer — measurement log (M5)

Everything in this file was produced on the reference machine (RX 9070 XT,
`gfx1201`, ROCm 7.2, 12-core host) with **no other GPU or CPU load** — this GPU
drops to a deep DPM state between kernels, so a contended run measures the clock
ramp instead of the kernel (see `PLAN.md` M2 "Passo 5").

Reproduce with:

```bash
./build/rdna4-infer bench -m <model.gguf> -n 64 --reps 3      # decode/VRAM/bandwidth
./build/rdna4-infer ppl   -m <model.gguf> -f <corpus> --chunks 10
./scripts/compare_ppl.sh  <model.gguf> 10                     # vs llama.cpp, per token
./scripts/check_golden_run.sh                                 # regeneration gate
```

## 1. Reference (llama.cpp, Vulkan/RADV, same machine and model)

`llama-bench -m Qwen3.8-27B-UD-IQ3_S.gguf -ngl 99 -p 64 -n 64 -r 3`:

| test | t/s |
|---|---|
| pp64 (batched prefill) | 440.49 ± 76.70 |
| tg64 (decode) | 39.73 ± 0.45 |

## 2. This engine

`rdna4-infer bench -m Qwen3.8-27B-UD-IQ3_S.gguf -n 32 --reps 3` (greedy, RNG-free,
8-token warm-up excluded):

| layers executed | tok/s | per-token |
|---|---|---|
| 0 (embedding + final norm + LM head + logits copy + sampler) | 510.7 | 1.96 ms |
| 1 | 359.9 | 2.78 ms |
| 64 (full model) | 27.9 | 35.83 ms |

Derived from that decomposition:

| component | cost | share |
|---|---|---|
| LM head + glue + sampler | 1.96 ms | 5.5 % |
| 64 trunk layers | 33.87 ms (0.529 ms/layer) | 94.5 % |
| — of which activation quantization (upper bound, measured by skipping every `quantize_q8_1` launch) | 1.5 ms | 4.1 % |
| effective weight bandwidth (11.20 GiB read per token) | 335 GB/s | — |

The matvec itself was tuned in M2 against a measured read ceiling of **619 GB/s**
(same block traversal, all bytes); the aggregate matvec-only estimate is 12.02 GB
in ~28.9 ms = **416 GB/s**. So of the 35.8 ms per token, ~28.9 ms is the matvec at
its measured speed and ~6.9 ms is everything else (64×3 norms, GDN conv/delta,
attention, ~450 activation-quantization launches, residual adds) — i.e. many small
kernels, not bandwidth.

Two hypotheses were measured and **rejected** rather than assumed:

- *HIP graphs would close the gap.* The host launch path on this machine costs
  **1.04 µs/launch** (1.39 µs with `hipGetLastError`, measured with a null kernel
  launched 2000×). ~1300 launches/token ≈ 1.4 ms; the launch loop's wall time is
  the device back-pressuring a full command queue, not host cost.
- *Fusing the activation quantization would help.* Skipping **all** `quantize_q8_1`
  launches (450/token) changes decode from 27.73 → 28.82 tok/s, i.e. **4.1 %**.

## 3. Perplexity (quality gate)

`scripts/compare_ppl.sh` drives both engines over **identical 768-token windows**
of `wikitext-2-raw/wiki.test.raw` (ctx 512, stride 512, the tiling
`llama-perplexity --ppl-stride` uses) and compares the **per-position** NLL.
This engine's tokenizer produced id streams identical to llama.cpp's over the whole
corpus (297 193 tokens), verified by the script before scoring.

| chunk | IQ3_S ours | IQ3_S ref | rel | IQ4_XS ours | IQ4_XS ref | rel |
|---|---|---|---|---|---|---|
| 0 | 3.8231 | 3.8160 | 0.185 % | 3.7992 | 3.8012 | 0.053 % |
| 1 | 8.7500 | 8.7591 | 0.105 % | 8.7351 | 8.7338 | 0.015 % |
| 2 | 7.2332 | 7.2326 | 0.008 % | 7.2500 | 7.2429 | 0.099 % |
| 3 | 6.3395 | 6.3407 | 0.018 % | 6.2930 | 6.2857 | 0.116 % |
| 4 | 7.2493 | 7.2422 | 0.097 % | 7.2015 | 7.2016 | 0.001 % |
| 5 | 7.7695 | 7.7698 | 0.004 % | 7.7487 | 7.7388 | 0.128 % |
| 6 | 7.2097 | 7.1966 | 0.182 % | 7.1737 | 7.1693 | 0.062 % |
| 7 | 10.8278 | 10.8368 | 0.083 % | 10.6307 | 10.6369 | 0.058 % |
| 8 | 10.8037 | 10.8308 | 0.250 % | 10.5937 | 10.6032 | 0.090 % |
| 9 | 9.2028 | 9.1933 | 0.103 % | 9.1063 | 9.1200 | 0.150 % |
| **worst** | | | **0.250 %** | | | **0.150 %** |

Worst single-position difference over 5120 scored tokens: **0.458 nats** (IQ3_S) and
**0.873 nats** (IQ4_XS, on a token with NLL ≈ 9). Per chunk the deviation is
0.004–0.25 %, i.e. what a different (but correct) matmul reduction order produces on
a 248 320-way softmax; the M3 node-level comparison measured the same ~0.1–0.2
logit deviation.

The by-product is a useful sanity check on the quantization: IQ4_XS is consistently
slightly *better* than IQ3_S (3.799 vs 3.823 on chunk 0), as a larger quant should be.


Why not compare against `llama-perplexity`'s own per-chunk number: its strided mode
does not score a fresh window. For chunk 2 of this corpus the tool prints PPL
**5.8045**, while driving the *same* model over the same 768-token window (fresh
state, identical positions) gives **8.7591** — the tool's number carries context
from earlier chunks. Comparing per-position against the reference model removes the
ambiguity and localises any divergence to a single token.

## 4. VRAM and context (both files)

`rdna4-infer bench` (32 decode tokens, 2 reps) and `info` (the pre-check budget).
"fill" = the KV cache and recurrent state are pre-seeded (`--fill-cache`), so the
decode step runs against a *synthetic* full cache without waiting for a long prefill
(same trick as the M3 long-context test).

| model | ctx | kv K/V | decode | VRAM in use | free |
|---|---|---|---|---|---|
| IQ3_S | 4 096 | f16 / f16 | 27.57 tok/s | 11.87 GiB | 4.05 GiB |
| IQ3_S | 16 384 | q8_0 / q8_0 | 28.60 tok/s | 12.15 GiB | 3.77 GiB |
| IQ3_S | 24 576 | f16 / f16 | 28.55 tok/s | 13.11 GiB | 2.81 GiB |
| IQ3_S | 32 768 | f16 / f16 | 28.21 tok/s | 13.61 GiB | 2.31 GiB |
| IQ3_S | 65 536 | q4_0 / q4_0 (fill) | 28.74 tok/s | 12.73 GiB | 3.19 GiB |
| IQ3_S | 131 072 | q4_0 / q4_0 (fill) | 28.59 tok/s | 13.86 GiB | 2.06 GiB |
| IQ4_XS | 4 096 | f16 / f16 | 27.02 tok/s | 13.91 GiB | 2.01 GiB |
| IQ4_XS | 16 384 | q8_0 / q8_0 | 26.63 tok/s | 14.19 GiB | 1.73 GiB |
| IQ4_XS | 24 576 | f16 / f16 | 25.03 tok/s | 15.16 GiB | 0.76 GiB |
| IQ4_XS | 32 768 | f16 / f16 | 25.38 tok/s | 15.66 GiB | 0.26 GiB |
| IQ4_XS | 65 536 | q4_0 / q4_0 (fill) | 25.83 tok/s | 14.79 GiB | 1.13 GiB |
| IQ4_XS | 131 072 | q4_0 / q4_0 (fill) | **fails** (`hipMalloc failed`) | — | — |

Two things worth reading off that table:

- **Decode speed is almost independent of context and KV type** (25.0–28.7 tok/s
  across a 32× context range): the weights (11.2 / 13.3 GiB) dominate the memory
  traffic of a token, so the KV cache only matters for whether the model fits.
- **Effective bandwidth is higher on IQ4_XS than on IQ3_S** (363.8 vs 336.2 GB/s,
  computed as bytes-per-token ÷ time), which is why a file 18 % larger decodes at
  nearly the same speed: IQ4_XS is mostly `q4_K`/`iq4_xs`, while IQ3_S is dominated by
  `iq3_s`, the type M2 showed to be issue-bound (its byte-op emulation costs ~2.2×).

Prefill (per-token path, 512-token prompt):

| model | prefill | decode |
|---|---|---|
| IQ3_S | 28.84 tok/s (17.75 s) | 27.57 tok/s |
| IQ4_XS | 27.43 tok/s (18.67 s) | 27.02 tok/s |

`info`'s budget check is deliberately conservative: for IQ4_XS at 32K f16 it refuses
(need 16.27 GiB > 15.92 GiB) while the real run fits with 15.66 GiB in use — it
assumes 1.00 GiB of overhead where the engine actually uses ~0.40 GiB. It is a
pre-flight guard, not a measurement.


## 5. Known gaps

1. **Batched prefill** — the only large gap: 28.8 tok/s per-token vs 440 tok/s
   batched in llama.cpp (**16×**) for a 512-token prompt. Decode is at 70 % of the
   reference. Batching means a GEMV→GEMM change in all 14 quant kernels (the weight
   traffic per prompt drops from `n_tokens × weights` to `weights`), which is
   milestone-sized work, not a low-risk tune: it is the first item of M6.
2. Attention is not tiled: at 64K a decode step costs ~262 ms (M3 measurement).
3. `IQ4_XS` + 131K context does not fit; 64K `q4_0` is the practical long-context
   configuration for that file (and anything ≥ 24K f16 is within 1 GiB of the limit).
4. The CLI is single-turn; `chat_render` supports multi-turn but no CLI flag exposes it.

