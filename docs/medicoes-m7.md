# rdna4-infer — M7 measurement: the attention kernel at long context

M6's corrected `bench --start-pos` numbers showed long context is **attention-bound**,
not weight-bound: a decode step costs 45 ms at 4K, 174 ms at 64K (f16 KV) and 441 ms
at 131K, while the weight stream is a constant ~36 ms. This is what that time is
spent on, what was fixed, and what is left.

## Diagnosis (measured, `tests/bench_attn_gpu.hip`)

The shipping kernel (`attn_kernel`, one CTA per query head, 8 warps splitting keys,
one 5-shuffle reduction + 2 `expf` per key per warp) is **memory-latency-bound**, not
softmax-bound and not bandwidth-bound. Three measurements pin that down:

| observation | number | what it rules out |
|---|---|---|
| kernel time vs cache size | 7.0-9.4 ms per layer for 36 MiB (q4_0), 128 MiB (f16) and 256 MiB (f32) at the same `t` | bandwidth: the time does not follow the bytes |
| achieved cache bandwidth at 64K | 27 GB/s f16 of ~600 GB/s measured DRAM | bandwidth saturation |
| scaling with warps per CTA (same work) | 8 → 16 → 32 warps: f16 64K 9.42 → 5.85 → **3.69 ms** | "the algorithm needs a rewrite" |
| 24 query-head CTAs on 64 CUs | 40 CUs idle | concurrency |

The fix was therefore not a new algorithm but **more requests in flight**:

1. **Vectorized cache loads.** `kv_load8<CT>()` (in `kv.h`) gives each lane its 8
   consecutive dims in ONE access (16 B for f16, 8 B for q4_0/q8_0, 32 B for f32)
   instead of 8 scalar 1-2 byte loads whose warp-wide addresses straddle cache lines.
   The values and their order are identical to 8 `kv_load()` calls.
2. **32 warps per CTA** (`kAttnWarpsPerBlock`, the 1024-thread maximum, up from 8).

## Result

Kernel, one layer, one query token (ms):

| t | f16 before | f16 after | q4_0 before | q4_0 after |
|---|---|---|---|---|
| 512 | 0.213 | **0.059** (3.6×) | — | — |
| 4 096 | 0.347 | **0.254** (1.4×) | — | — |
| 16 384 | 1.371 | **0.525** (2.6×) | 1.406 | 0.764 (1.8×) |
| 65 536 | 9.420 | **3.685** (2.6×) | 13.010 | 4.095 (3.2×) |

End to end (`bench --start-pos`, IQ3_S, decode at the end of the context):

| ctx / KV | before | after | gain |
|---|---|---|---|
| 4 096 f16 | 22.13 tok/s | 24.11 tok/s | 1.09× |
| 16 384 f16 | 13.72 tok/s | 14.82 tok/s | 1.08× |
| 65 536 f16 | 5.76 tok/s | 6.64 tok/s | 1.15× |
| 65 536 q4_0 | 4.18 tok/s | 6.40 tok/s | **1.53×** |
| 131 072 q4_0 | 2.27 tok/s | 3.66 tok/s | **1.61×** |

The engine gains less than the isolated kernel because the attention now competes with
the weight stream for DRAM (37 GB/s achieved in-engine vs 69 GB/s in the bench), and
because 16 launches per token start colder than a warmed single kernel.

**Correctness:** unchanged and verified — `check-graph-gpu` (per-node oracle) PASS,
`check-kvctx-gpu` OK, `check-rope-gpu` OK, `scripts/check_golden_run.sh` OK,
argmax PASS. The batched/`GQA` prototype in the bench is *bit-identical* to the
shipping kernel (`rel-L2 0.0e+00`), which is also how the `Q4_0` nibble mapping in
`kv_load8` was caught (it was wrong on the first attempt; the scalar path was the
reference).

## Rejected: sharing K/V rows across the GQA group

The 6 query heads of a KV group re-read the same rows (`n_head/n_head_kv` = 6×
redundant traffic), so a prototype kernel kept all 6 heads' queries and accumulators
in registers per warp and computed 6 dots per loaded row. It is bit-exact, and it is
**8-12× slower**: the grid collapses from 24 CTAs to `n_head_kv` = 4, and the register
pressure destroys occupancy. Sharing across heads only pays if one warp handles
several heads *and* the grid is widened by splitting the position range — i.e. after
the split-KV work below, not before it. Kept in the bench as a cross-check.

## Split-KV across CTAs (step 2, done)

The key range is now also split across CTAs: grid = `n_head × n_splits`
(`attn_split_kernel`), each CTA walking keys `j = w + 32·s, step 32·S`, writing one
partial `(m, l, acc[head_dim])` per split; `attn_merge_kernel` combines the splits
with the same online-softmax arithmetic. Splits are chosen by context
(`keys / 2048`, capped at 16, i.e. 384 CTAs at 64K) and `RD_ATTN_SPLITS` forces a
count for testing.

Kernel, one layer, one query token (ms), and the difference against the single-CTA
path over the same inputs:

| t | unsplit | split | speedup | split vs unsplit |
|---|---|---|---|---|
| 4 096 | 0.586 | 0.295 | 2.0× | rel-L2 3.5e-07 |
| 16 384 | 1.356 | 0.248 | 5.5× | rel-L2 6.5e-07 |
| 65 536 | 7.909 | 1.187 | **6.7×** | rel-L2 1.2e-06 |

End to end (`bench --start-pos`), decode at the END of the context:

| model | ctx / KV | M5 baseline | 32 warps | + split-KV | total |
|---|---|---|---|---|---|
| IQ3_S | 4 096 f16 | 22.13 | 24.11 | 23.73 | 1.07× |
| IQ3_S | 16 384 f16 | 13.72 | 14.82 | **24.08** | **1.76×** |
| IQ3_S | 65 536 f16 | — | — | **18.98** | — |
| IQ3_S | 65 536 q4_0 | 4.18 | 6.40 | **17.81** | **4.3×** |
| IQ3_S | 131 072 q4_0 | 2.27 | 3.66 | **13.19** | **5.8×** |
| IQ4_XS | 32 768 f16 | 9.09 | — | **20.79** | **2.3×** |
| IQ4_XS | 65 536 q4_0 | 4.18 | — | **17.11** | **4.1×** |

**Decode is now nearly flat in context** (23.7 tok/s at 4K, 24.1 at 16K, 19.0 at 64K,
13.2 at 131K): the attention is no longer what long context costs. It also flips the
KV-type advice back — at 64K f16 (18.98) now beats q4_0 (17.81) — and long *prefill*
benefits from the same kernel (27.7 tok/s for a 2048-token prompt, versus 28.8 at 512:
almost no degradation, where the M6-era attention made long prompts progressively
slower).

### The bug this found

Forcing more splits than keys (2 splits at position 0) produced **NaN logits**:
an empty split leaves every warp slice at `m = -INFINITY`, and the CTA merge computed
`expf(-inf - -inf)`. Empty splits now contribute zeros (guarded in both the CTA merge
and the final merge, plus the unsplit kernel's merge for the same latent hazard). The
auto policy never produced empty splits, which is exactly why the *forced* comparison
exists.

### Correctness of the split path

It is not bit-identical (the summation order across keys changes), so the gate is
numerical equivalence **on real text**: `scripts/check_attn_split.sh` compares
perplexity over wiki text with `RD_ATTN_SPLITS=1` vs `4` — 5.1989 vs 5.1917, i.e.
**0.14 %**, inside the 0.004-0.25 % inter-chunk spread the M5 gate measured. The
synthetic-cache comparison in `check-kvctx-gpu` is kept but labelled informative: with
random keys the attention output is a near-cancelling average, so a 1e-7 reordering
difference is amplified to percent level there (measured rel-L2 1.7e-02 on the logits
with the argmax preserved and `l_out#7` — a GDN layer on the same residual —
bit-identical). All other gates stay green: graph oracle PASS, argmax PASS, rope OK,
kvctx OK, golden run OK.

## What is left

Attention in-engine is still ~215 GB/s of ~600 GB/s (1.19 ms per layer at 64K × 16
layers ≈ 19 ms/token against a 36 ms/token weight stream), so the next levers are
smaller: GQA row sharing now that the grid is wide (24 × 16 = 384 CTAs, so sharing
rows across the 6 query heads no longer costs parallelism), and a per-(type,N) re-tune
of the batched matvec. Neither is needed for usability anymore.
