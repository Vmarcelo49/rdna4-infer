#!/usr/bin/env bash
# M7 gate: the split-KV attention must not change the model's behaviour.
#
#   ./scripts/check_attn_split.sh [MODEL] [SPLITS]
#
# The split path sums the keys in a different order, so it is not bit-identical to
# the single-CTA path -- it has to be *numerically* equivalent, and that has to be
# checked on REAL text: the synthetic-cache test (check-kvctx-gpu) runs the
# attention against random keys, where the output is a near-cancelling average and
# a 1e-7 reordering difference is amplified by the cancellation to percent level
# (see docs/medicoes-m7.md). Perplexity over real wiki text is the honest measure.
#
# RD_ATTN_SPLITS forces the split count (1 = the pre-M7 single-CTA path) so both
# paths can be compared at the same context.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# Serialise GPU access (docs/gpu-queue.md rule 1). A caller that already holds the
# lock exports GPU_LOCK_HELD=1: flock is not reentrant across processes, so
# re-locking from inside a locked run would block until the waiter times out and
# the gate would fail for the wrong reason.
if [ "${GPU_LOCK_HELD:-0}" != 1 ]; then
  export GPU_LOCK_HELD=1
  exec "$ROOT/scripts/gpu-lock.sh" "$0" "$@"
fi
BIN="${BIN:-$ROOT/build/rdna4-infer}"
# Strip an optional --quick flag (iteration mode: fewer chunks, no wide case).
# check_all.sh --quick skips this gate entirely; this flag is for direct runs.
QUICK="${QUICK:-0}"
_cli_model=""; _cli_splits=""
for _a in "$@"; do
  if [ "$_a" = "--quick" ]; then QUICK=1; else
    if [ -z "$_cli_model" ]; then _cli_model="$_a"; else _cli_splits="$_a"; fi
  fi
done
MODEL="${_cli_model:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
SPLITS="${_cli_splits:-4}"
CORPUS="${CORPUS:-$ROOT/reference/data/wikitext-2-raw/wiki.test.raw}"
CTX="${CTX:-1024}"
STRIDE="${STRIDE:-1024}"
CHUNKS="${CHUNKS:-2}"
if [ "$QUICK" = 1 ]; then
  # 3 loads -> 2 (CHUNKS 2->1 halves the cheapest gate; WIDE=0 drops the 3rd
  # run). Real-text split equivalence still covered by the base-vs-split pair.
  CHUNKS=1
  WIDE="${WIDE:-0}"
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -f "$CORPUS" ] || { echo "no corpus at $CORPUS (see scripts/get-wikitext-2.sh)" >&2; exit 2; }

run() {  # splits -> writes ppl to stdout
  RD_ATTN_SPLITS="$1" "$BIN" ppl -m "$MODEL" -f "$CORPUS" --ctx-size "$CTX" --stride "$STRIDE" \
    --chunks "$CHUNKS" --no-stats 2>/dev/null | sed -n 's/^Final estimate: PPL = \([0-9.]*\).*/\1/p'
}

base="$(run 1)"
split="$(run "$SPLITS")"
echo "PPL unsplit (1 CTA/head): $base"
echo "PPL split   ($SPLITS CTAs/head): $split"
if [ -z "$base" ] || [ -z "$split" ]; then
  echo "check-attn-split: FAILED (no PPL: dataset too small for ctx $CTX / $CHUNKS chunks?)"
  exit 1
fi
python3 - "$base" "$split" <<'PY' || exit 1
import sys
a, b = float(sys.argv[1]), float(sys.argv[2])
rel = abs(a - b) / a
print(f"relative difference: {rel*100:.3f}% (limit 0.5%)")
if rel > 0.005:
    print("check-attn-split: FAILED")
    sys.exit(1)
print("check-attn-split: OK")
PY

# ---------------------------------------------------------------------------
# The wide-CTA end of the two-ended WPB rule (>=16 splits -> 16 warps) had NO gate
# on real text: the only test that touched it ran against a synthetic cache with a
# 5e-2 tolerance, so a wrong merge order there would have gone unnoticed (review
# finding F5, docs/adversarial-noite.md). RD_ATTN_SPLITS forces the split count, and
# 16 splits is exactly where the rule switches the CTA to 16 warps -- so this case
# exercises the shipped wide combination against the unsplit reference on real text.
# It costs two short runs at CTX (the same as the case above), not a 8K context.
if [ "${WIDE:-1}" = 1 ]; then
  wide="$(run 16)"
  echo "PPL wide    (16 CTAs/head -> 16 warps/CTA, the shipped long-context rule): $wide"
  if [ -z "$wide" ]; then
    echo "check-attn-split: FAILED (no PPL for the wide case)"
    exit 1
  fi
  python3 - "$base" "$wide" <<'PY' || exit 1
import sys
a, b = float(sys.argv[1]), float(sys.argv[2])
rel = abs(a - b) / a
print(f"wide vs unsplit: {rel*100:.3f}% (limit 0.5%)")
if rel > 0.005:
    print("check-attn-split: FAILED (the wide-CTA rule is not numerically equivalent")
    print("                  to the unsplit path on real text)")
    sys.exit(1)
print("check-attn-split: OK (wide rule included)")
PY
fi
