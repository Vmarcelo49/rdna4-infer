# Chunk-cap scaling 2026-09-16 — Qwen3.8-27B UD-IQ3_S on RX 9070 XT (gfx1201)

**STATUS: FINAL** — all protocol items measured, MEASURE-ONLY (no source touched).
Model (read-only): `/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf`
Binary: `./build/rdna4-infer` as-is (no build, no `rocm-env.sh`).
Every GPU command ran as `timeout 1200 scripts/gpu-lock.sh …`, ONE hold per batch, never concurrent.
`bench` times prefill `--prefill-reps N` internally (per-pass lines on stderr, best on stdout);
decode section is warmup 8 + `--reps` (best AND mean as printed). All windows clean.

## 1. Exact commands

```bash
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
# Batch A — one gpu-lock hold, 9 sequential runs:
timeout 1200 scripts/gpu-lock.sh bash -c '
M=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf
for cap in 128 256 512; do for i in 1 2 3; do
  RD_PREFILL_CHUNK=$cap ./build/rdna4-infer bench -m $M --prefill 512 --prefill-reps 3
done; done'
# Batch B — winner cap x long prefill (one hold, 2 runs):
timeout 1200 scripts/gpu-lock.sh bash -c '
M=...; for i in 1 2; do
  RD_PREFILL_CHUNK=512 ./build/rdna4-infer bench -m $M --prefill 4096 --prefill-reps 2
done'
# Batch C — winner cap x decode short (one hold, 3 runs):
timeout 1200 scripts/gpu-lock.sh bash -c '
M=...; for i in 1 2 3; do
  RD_PREFILL_CHUNK=512 ./build/rdna4-infer bench -m $M -p "The capital of France is" -n 16 --reps 3
done'
```

## 2. Results table (IQ3_S, f16 KV, ctx 4096)

| cap | workload | tok/s best (per invocation) | tok/s mean of reps (per invocation) | VRAM in-use | status |
|---|---|---|---|---|---|
| 128 | prefill 512 | 368.13 / 366.58 / 366.15 | 345.89 / 345.46 / 345.31 | 11.98 GiB ×3 | OK |
| 256 | prefill 512 | 401.56 / 401.34 / 401.73 | 377.71 / 377.90 / 377.79 | 12.08 GiB ×3 | OK |
| 512 | prefill 512 | 411.22 / 410.89 / 410.70 | 386.80 / 386.58 / 386.45 | 12.28 GiB ×3 | OK, nothing breaks |
| 512 | prefill 4096 | 353.39 / 351.24 | 350.32 / 347.79 | 12.29 GiB ×2 | OK |
| 512 | decode short (pos 5..21) | 37.91 / 37.94 / 37.94 | 37.09 / 37.10 / 37.07 | 12.28 GiB ×3 | OK, unaffected |

Raw passes (best is always a warm pass; rep 1 is first-touch cold, ~17–18% slower at every cap):
- 128: (301.63/367.90/368.13) (303.45/366.58/366.36) (303.67/366.15/366.11)
- 256: (330.48/401.10/401.56) (331.08/401.34/401.28) (330.33/401.32/401.73)
- 512: (338.43/411.22/410.74) (338.49/410.89/410.36) (338.15/410.70/410.50)
- 4096@512: (347.25/353.39) (344.34/351.24)
- Cross-check baselines reproduced at cap 128: 368.13 ≈ prior 366.27 (≤0.5%, noise).

Gains (best-of-best): 128→256 **+9.1%** (401.73/368.13); 128→512 **+11.7%** (411.22/368.13);
256→512 only **+2.4%**. Long prefill 4096@512: 353.39 vs prior 310.77 @128 = **+13.7%**.
Decode @512 (37.94 best / ~37.1 mean) == baseline @128 (37.88 / 36.29): unaffected, as expected
(chunk cap never touches the GEMV decode path). Bystander decode sections agree:
37.6–37.7 @128, 37.58–37.59 @256, 37.55–37.58 @512 (64-token, longer ctx than batch C).

## 3. Binding-limit analysis — which limit binds first at each cap

`graph_buffer_bytes()` for the IQ3_S shape (recomputed by hand from `device.h:105-143`,
matches the `check_kvtype.hip:210` pin 269 822 436 B @128):

| max_batch (= cap) | buffers | Δ vs 128 | measured VRAM Δ |
|---|---|---|---|
| 16 | 164.28 MiB | −93.0 MiB | — |
| 128 | 257.32 MiB | — | 11.98 GiB (ref) |
| 256 | 363.66 MiB | +106.3 MiB | +0.10 GiB ≈ +102 MiB ✓ |
| 512 | 576.33 MiB | +319.0 MiB | +0.30 GiB ≈ +307 MiB ✓ |

Marginal cost ≈ **871 108 B/token** (~0.87 MB/tok, as the `device.h:205-207` comment says;
dominant term is the split-KV partials `batch·24·16·258·4` = 396 288 B/tok, ~45%).

- **Workspace**: grows linearly, no cliff. Largest term at 512 is still only 0.56 GiB.
- **KV**: independent of cap. f16 @ctx4096 = 268 MiB (bench ctx); @131K f16 = 8.59 GiB —
  that config exceeds 15.92 GiB at *every* cap (weights 10.877 GiB trunk alone + KV),
  so the **KV type binds there, never the chunk cap**.
- **Weights**: 10.877 GiB trunk, constant. Margin `kOverheadBytes` 448 MiB, constant.
- **GDN staging** (`d_qkb_`/`d_vcb_`, `graph.cuh:465-468`): sized by `batch_max_`,
  inside the per-token batch block — linear, no cliff.
- **Attention splits**: `attn_splits_for(pos)` depends on key count, capped at
  `kAttnMaxSplits` = 16; the chunk is cut into equal-split runs (`graph.cuh:1467-1476`),
  so cap changes *batching*, not split decisions. No break.
- **Hard ceiling**: `kMaxChunkHost` = 512 (`graph.cuh:200`; stack `pos_host[512]`,
  `d_emb_ids_` fixed 512·4 B = 2 KiB, embed path rejects n > 512). Cap 512 == ceiling,
  which is why "what breaks at 512" = nothing: it is the designed max, and the run proves it.
- **Headroom @512** (measured 12.28–12.29 GiB of 15.92): 3.6 GiB free at ctx 4K.
  Projected 131K q5_0/q4_1 @512: 10.877 + 2.625 + 0.563 + 0.4375 ≈ **14.5 GiB — fits**
  (~1.4 GiB headroom). Chunk cap is never the first binding limit in any tested
  or projected config; weights + KV dominate by >10:1.

Caveat (not measured here, gates must confirm before flipping the default): chunks
with n > 16 use the tiled GEMM, which is NOT bit-exact vs the GEMV path (declared
tolerance `kTolChunkRelL2`/`kTolChunkMaxAbs` in `check_batch_gpu.hip`). Bigger cap =
more GEMM rows per prompt; PPL/golden gates should be re-run at 512. Diminishing
returns also apply: 256 captures ~78% of the 512 gain (9.1 of 11.7 pts).

## VERDICT

**SHIP: default 128→512, +11.7% prefill-512 (368.13→411.22), +13.7% prefill-4096
(310.77→353.39), decode unaffected (37.94 vs 37.88), nothing breaks at the 512
ceiling (VRAM 12.28/15.92 GiB); 256 (+9.1%) is the fallback if gates want margin —
pending PPL/golden re-run at 512 for the GEMM-tolerance caveat. Do not commit.**
