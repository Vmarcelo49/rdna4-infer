#!/usr/bin/env bash
# M5 quality gate: perplexity of this engine vs llama.cpp's model on the *same*
# sequences.
#
#   ./scripts/compare_ppl.sh [MODEL] [CHUNKS] [CORPUS] [--quick]
#
# QUICK=1 in the env (or a --quick argument anywhere) runs 3 windows instead
# of 10: a fast smoke version of the same gate (same windows 0..2, same
# comparison table). An explicit CHUNKS argument still wins over --quick.
#
# Why not compare against `llama-perplexity`'s own number: its strided mode resets
# the cache per chunk but its per-chunk value does not correspond to a fresh
# window (measured: for chunk 2 of wikitext-2 the tool reports PPL 5.80, while
# driving the *same* model over the same 768-token window gives 8.76 — i.e. the
# tool has extra context). Comparing per-token NLLs against the reference model
# over identical windows removes that ambiguity and is strictly more informative:
# it localises any divergence to a position.
#
# Method
#   - the corpus is tokenized by this engine's tokenizer and by llama.cpp; the id
#     streams must be identical (they are verified here, not assumed);
#   - window = ctx + stride/2 tokens, scored positions [window - stride - 1, window - 1),
#     exactly the tiling `llama-perplexity --ppl-stride` uses;
#   - this engine scores every window in one process (`ppl --nll-out`);
#   - the reference is driven over the same windows with
#     tests/oracle_next_token.cpp's ORACLE_NLL_OUT mode (one token at a time,
#     logits at every step), loading the 12GB model ONCE via its --nll-batch
#     mode (one process scores every window file in-process); binaries that
#     predate the flag fall back to one run per window;
#   - per chunk: mean NLL and PPL on both sides, plus the largest per-position
#     difference.
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
ORACLE="${ORACLE:-$ROOT/build/oracle-next-token}"
TOKORACLE="${TOKORACLE:-$ROOT/build/oracle-tokenize}"
MODEL=""
CHUNKS=""
CORPUS=""
QUICK="${QUICK:-0}"
for a in "$@"; do
  case "$a" in
    --quick) QUICK=1 ;;
    *) if [ -z "$MODEL" ]; then MODEL="$a";
       elif [ -z "$CHUNKS" ]; then CHUNKS="$a";
       elif [ -z "$CORPUS" ]; then CORPUS="$a";
       else echo "unexpected argument: $a" >&2; exit 2; fi ;;
  esac
done
[ -n "$MODEL" ] || MODEL="/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf"
if [ -z "$CHUNKS" ]; then
  if [ "$QUICK" = 1 ]; then CHUNKS=3; else CHUNKS=10; fi
fi
[ -n "$CORPUS" ] || CORPUS="$ROOT/reference/data/wikitext-2-raw/wiki.test.raw"
CTX="${CTX:-512}"
STRIDE="${STRIDE:-512}"
# KV cache types for THIS engine (frente KV). The llama.cpp reference below runs
# its own default f16 cache, so a run with KV_K/KV_V quantized measures "our
# quantized cache vs a clean f16 reference", which is exactly the delta the night
# wants; the f16 run of the same command is the engine's own floor on top of it.
KV_K="${KV_K:-f16}"
KV_V="${KV_V:-f16}"
ORACLE_NGL="${ORACLE_NGL:-99}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

[ -f "$CORPUS" ] || { echo "no corpus at $CORPUS" >&2; exit 2; }

echo "== tokenizing the corpus with both tokenizers"
"$BIN" tokenize -m "$MODEL" -f "$CORPUS" > "$TMP/ours.ids" 2>/dev/null || exit 1
"$TOKORACLE" "$MODEL" --raw "$CORPUS" > "$TMP/ref.ids" 2>/dev/null || exit 1
python3 - "$TMP/ours.ids" "$TMP/ref.ids" <<'PY' || exit 1
import sys
a = [int(x) for x in open(sys.argv[1])]
b = [int(x) for x in open(sys.argv[2])]
# llama.cpp's `-f` drops a trailing newline, which changes only the very last
# token; the chunks compared here are far from the end.
n = min(len(a), len(b))
first = next((i for i in range(n) if a[i] != b[i]), None)
print(f"   tokens: ours {len(a)}, llama.cpp {len(b)}, first difference at {first}")
if first is not None and first < n - 4:
    print("   FAIL: the id streams differ well before the end")
    sys.exit(1)
PY

if [ "$QUICK" = 1 ]; then echo "== quick mode: 3 chunks"; fi
echo "== this engine: $CHUNKS chunks of ctx $CTX / stride $STRIDE, KV k=$KV_K v=$KV_V"
"$BIN" ppl -m "$MODEL" -f "$CORPUS" --ctx-size "$CTX" --stride "$STRIDE" --chunks "$CHUNKS" \
       --cache-type-k "$KV_K" --cache-type-v "$KV_V" \
       --nll-out "$TMP/ours_nll.txt" 2>"$TMP/ours.log" | tail -2
grep -E "corpus|scored" "$TMP/ours.log" | sed 's/^/   /'

# Build the window token lists once (they are shared by both sides).
python3 - "$TMP/ours.ids" "$TMP" "$CHUNKS" "$CTX" "$STRIDE" <<'PY'
import sys
ids = [int(x) for x in open(sys.argv[1])]
tmp, chunks, ctx, stride = sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
window = ctx + stride // 2
for c in range(chunks):
    start = c * stride
    seg = ids[start:start + window]
    if len(seg) < window:
        break
    open(f"{tmp}/win{c}.ids", "w").write(" ".join(str(t) for t in seg))
print(f"   windows written: {min(chunks, (len(ids) - window)//stride + 1)} of {window} tokens")
PY

echo "== reference model (llama.cpp, ngl $ORACLE_NGL): scoring windows with one model load"
# Batch mode (--nll-batch: one 12GB load, every window scored in-process) when
# the binary supports it. The probe needs no model or GPU (the binary prints
# batch usage before any backend init); older binaries fall back to one run
# per window. Either way each chunk ends with "$TMP/ref_$c.log" whose last
# line is the historical "nll written to ..." line.
: > "$TMP/winlist.txt"
for c in $(seq 0 $((CHUNKS - 1))); do
  [ -f "$TMP/win$c.ids" ] || break
  echo "$TMP/win$c.ids" >> "$TMP/winlist.txt"
done
NWINS="$(wc -l < "$TMP/winlist.txt" | tr -d ' ')"
if [ "$NWINS" -gt 0 ] && "$ORACLE" x --nll-batch 2>&1 | grep -qi "nll-batch"; then
  ORACLE_NGL="$ORACLE_NGL" "$ORACLE" "$MODEL" --nll-batch "$TMP/winlist.txt" "$TMP" \
    > "$TMP/ref_batch.log" 2>&1 || {
      echo "   oracle batch run failed (see $TMP/ref_batch.log)"; exit 1; }
  c=0
  while [ "$c" -lt "$NWINS" ]; do
    grep -F "ref_nll_${c}.txt" "$TMP/ref_batch.log" 2>/dev/null | tail -1 > "$TMP/ref_${c}.log" || true
    [ -s "$TMP/ref_${c}.log" ] || echo "oracle batch: no log line for chunk $c" > "$TMP/ref_${c}.log"
    [ -f "$TMP/ref_nll_${c}.txt" ] || {
      echo "   oracle failed on chunk $c (see $TMP/ref_batch.log)"; exit 1; }
    printf "   chunk %s: %s\n" "$c" "$(tail -1 "$TMP/ref_$c.log")"
    c=$((c + 1))
  done
else
  echo "   (oracle predates --nll-batch: one run per window)"
  for c in $(seq 0 $((CHUNKS - 1))); do
    [ -f "$TMP/win$c.ids" ] || break
    # shellcheck disable=SC2046
    ORACLE_NGL="$ORACLE_NGL" ORACLE_NLL_OUT="$TMP/ref_nll_$c.txt" \
      "$ORACLE" "$MODEL" $(cat "$TMP/win$c.ids") > "$TMP/ref_$c.log" 2>&1 || {
        echo "   oracle failed on chunk $c (see $TMP/ref_$c.log)"; exit 1; }
    printf "   chunk %s: %s\n" "$c" "$(tail -1 "$TMP/ref_$c.log")"
  done
fi

python3 - "$TMP" "$CHUNKS" "$CTX" "$STRIDE" <<'PY'
import math, sys, os
tmp, chunks, ctx, stride = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
window = ctx + stride // 2
first_scored = window - stride - 1

ours = {}
for line in open(f"{tmp}/ours_nll.txt"):
    i, nll, tok = line.split()
    ours[int(i)] = (float(nll), int(tok))

print(f"{'chunk':>5} {'ours PPL':>10} {'ref PPL':>10} {'rel diff':>9} {'max |dNLL|':>11} {'n':>5}")
worst_rel = 0.0
worst_max = 0.0
for c in range(chunks):
    p = f"{tmp}/ref_nll_{c}.txt"
    if not os.path.exists(p):
        break
    ref = {}
    for line in open(p):
        i, nll, tok = line.split()
        ref[int(i)] = (float(nll), int(tok))
    start = c * stride
    pairs = []
    for t in range(first_scored, window - 1):
        o = ours.get(start + t + 1)
        r = ref.get(t + 1)
        if o is None or r is None:
            continue
        if o[1] != r[1]:
            print(f"  FAIL chunk {c} position {t}: token mismatch {o[1]} vs {r[1]}")
            sys.exit(1)
        pairs.append((o[0], r[0]))
    if not pairs:
        continue
    mo = sum(x[0] for x in pairs) / len(pairs)
    mr = sum(x[1] for x in pairs) / len(pairs)
    rel = abs(math.exp(mo) - math.exp(mr)) / math.exp(mr)
    maxd = max(abs(x[0] - x[1]) for x in pairs)
    worst_rel = max(worst_rel, rel)
    worst_max = max(worst_max, maxd)
    print(f"{c:>5} {math.exp(mo):>10.4f} {math.exp(mr):>10.4f} {rel*100:>8.3f}% {maxd:>11.4f} {len(pairs):>5}")

print(f"\nworst chunk deviation: {worst_rel*100:.3f}%, worst single-position |dNLL|: {worst_max:.4f}")
if worst_rel > 0.01:
    print("compare-ppl: FAILED (a chunk deviates by more than 1%)")
    sys.exit(1)
print("compare-ppl: OK")
PY
