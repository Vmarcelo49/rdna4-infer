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
MODEL="${1:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
SPLITS="${2:-4}"
CORPUS="${CORPUS:-$ROOT/reference/data/wikitext-2-raw/wiki.test.raw}"
CTX="${CTX:-1024}"
STRIDE="${STRIDE:-1024}"
CHUNKS="${CHUNKS:-2}"
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
