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

## What is left

Attention is still at 69 GB/s (bench) / 37 GB/s (in-engine) of ~600 GB/s, so the
remaining lever is concurrency: **split the key range across CTAs** (grid =
n_head × splits, e.g. 24 × 8 = 192 CTAs) with a second online-softmax merge kernel
(the merge of the warp partials already exists in-kernel). Expected: another 2-4×
at 64K, which would put long-context decode in the 15-25 tok/s range instead of 6.6.
That is the next step; it needs a scratch buffer in the graph and its own gate.
