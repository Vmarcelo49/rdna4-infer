#!/usr/bin/env bash
# M4 acceptance (cross-engine): greedy decoding of a fixed prompt, this engine vs
# llama.cpp on the same model and the same GPU.
#
#   ./scripts/compare_llama_greedy.sh [N]
#
# Greedy is the only directly comparable mode: it needs no RNG, so both engines
# are deterministic. The engine's own tokenizer output feeds llama.cpp (it was
# verified bit-exact against llama.cpp over 3000 fuzz strings in M4 step 1), so a
# divergence can only come from the graph.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/rdna4-infer}"
ORACLE="${ORACLE:-$ROOT/build/oracle-next-token}"
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
PROMPT="${PROMPT:-The capital of France is}"
N="${1:-${N:-16}}"
ORACLE_NGL="${ORACLE_NGL:-99}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 1. this engine, greedy, verbose (stderr carries the prompt ids)
"$BIN" run -m "$MODEL" -p "$PROMPT" -n "$N" --greedy -v >"$TMP/rdna4.txt" 2>"$TMP/rdna4.err" ||
  { echo "rdna4 run failed" >&2; exit 1; }
grep '^prompt ids:' "$TMP/rdna4.err" | sed 's/^prompt ids://' >"$TMP/prompt.ids"
grep '^generated ids:' "$TMP/rdna4.err" | sed 's/^generated ids://' >"$TMP/rdna4.ids"
echo "prompt ids:$(cat "$TMP/prompt.ids")"

# 2. llama.cpp, same prompt, per-token decode (the engine's per-token path)
# shellcheck disable=SC2086
ORACLE_NGL="$ORACLE_NGL" ORACLE_STEP=1 ORACLE_GREEDY="$N" \
  ORACLE_IDS_OUT="$TMP/llama.ids" ORACLE_TEXT_OUT="$TMP/llama.txt" \
  "$ORACLE" "$MODEL" $(cat "$TMP/prompt.ids") >"$TMP/oracle.log" 2>&1 ||
  { echo "oracle failed (see $TMP/oracle.log)"; tail -5 "$TMP/oracle.log"; exit 1; }

# 3. compare token by token (an id-by-id diff localises the first divergence)
python3 - "$TMP/rdna4.ids" "$TMP/llama.ids" "$TMP/rdna4.txt" "$TMP/llama.txt" "$N" <<'PY'
import sys
a = [int(x) for x in open(sys.argv[1]).read().split()]
b = [int(x) for x in open(sys.argv[2]).read().split()]
want = int(sys.argv[5])
n = min(len(a), len(b))
first = next((i for i in range(n) if a[i] != b[i]), None)
print(f"rdna4: {len(a)} tokens, llama.cpp: {len(b)} tokens (asked for {want})")
# An empty comparison is not evidence: two engines that both emit nothing match
# vacuously, so require at least one token on both sides (review M4).
if not a or not b:
    print("VACUOUS: at least one engine produced no tokens — nothing was compared")
    sys.exit(1)
if first is None and len(a) == len(b):
    print(f"IDS MATCH: all {len(a)} greedy tokens identical")
elif first is None:
    print(f"IDS MATCH on the common prefix of {n}; lengths differ")
else:
    print(f"FIRST DIVERGENCE at token {first}: rdna4 {a[first]} vs llama.cpp {b[first]}")
    print("  rdna4 ids:", a[:first + 4])
    print("  llama ids:", b[:first + 4])
ta, tb = open(sys.argv[3], 'rb').read(), open(sys.argv[4], 'rb').read()
print(f"text: rdna4 {len(ta)} bytes, llama.cpp {len(tb)} bytes, "
      f"{'IDENTICAL' if ta == tb else 'DIFFERENT'}")
sys.stdout.write("rdna4 text: " + repr(ta) + "\n")
sys.stdout.write("llama text: " + repr(tb) + "\n")
# a run that stopped early is still a comparison, but a run shorter than asked
# for means the engines disagreed about stopping (or the prompt decode failed)
if len(a) < want or len(b) < want:
    print(f"NOTE: asked for {want} tokens, got {len(a)}/{len(b)} (early EOG?)")
sys.exit(0 if (ta == tb and a == b) else 1)
PY
