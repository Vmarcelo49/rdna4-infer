#!/usr/bin/env bash
# M4 acceptance (determinism): the same prompt + the same seed must produce
# exactly the same bytes, and the committed golden output must not change.
#
#   ./scripts/check_golden_run.sh            # verify against tests/golden/
#   UPDATE=1 ./scripts/check_golden_run.sh   # rewrite the golden files
#
# The golden files are the engine's own output for a fixed prompt/seed (there is
# no independent oracle for a full sampled continuation — llama.cpp's RNG stream
# is not ours, see PLAN.md M4); the independent cross-check of *greedy* decoding
# against llama.cpp is scripts/compare_llama_greedy.sh.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/rdna4-infer}"
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
GOLD="${GOLD:-$ROOT/tests/golden}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-32}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

if [ ! -x "$BIN" ]; then
  echo "no binary at $BIN (build first)" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "no model at $MODEL (set MODEL=...)" >&2
  exit 2
fi

# run_case <label> <out> <err> <extra args...>
run_case() {
  local label="$1" out="$2" err="$3"; shift 3
  if ! "$BIN" run -m "$MODEL" -p "$PROMPT" -n "$N" -v "$@" >"$out" 2>"$err"; then
    bad "$label: engine exited non-zero"
    return 1
  fi
  return 0
}

# ---------------------------------------------------------------- greedy ----
# temp 0 => RNG-independent, so this is a pure model/engine regression gate.
greedy_ok=1
run_case "greedy" "$TMP/g.out" "$TMP/g.err" --greedy || greedy_ok=0
grep '^generated ids:' "$TMP/g.err" > "$TMP/g.ids" || { bad "greedy: no 'generated ids:' line"; greedy_ok=0; }

# Never overwrite a committed golden with the output of a run that failed: a
# broken build would otherwise destroy the reference it is supposed to be
# checked against (review M4).
if [ "$greedy_ok" = "1" ] && [ "${UPDATE:-0}" = "1" ]; then
  cp "$TMP/g.out" "$GOLD/run_greedy_IQ3_S.txt"
  cp "$TMP/g.ids" "$GOLD/run_greedy_ids_IQ3_S.txt"
  note "updated $GOLD/run_greedy_IQ3_S.txt and run_greedy_ids_IQ3_S.txt"
else
  for pair in "g.out:run_greedy_IQ3_S.txt" "g.ids:run_greedy_ids_IQ3_S.txt"; do
    got="$TMP/${pair%%:*}"; want="$GOLD/${pair##*:}"
    if [ ! -f "$want" ]; then
      bad "missing golden $want (run with UPDATE=1 to create it)"
    elif ! cmp -s "$got" "$want"; then
      bad "greedy mismatch vs ${pair##*:}"
      diff -u "$want" "$got" | head -30
    fi
  done
fi

# -------------------------------------------------------------- sampling ----
# temp 1 + a fixed seed: exercises the RNG path, and the same seed twice must be
# bit-identical (the M4 acceptance criterion for the sampler).
run_case "sample-a" "$TMP/s1.out" "$TMP/s1.err" --temp 1.0 --seed 42
run_case "sample-b" "$TMP/s2.out" "$TMP/s2.err" --temp 1.0 --seed 42
if ! cmp -s "$TMP/s1.out" "$TMP/s2.out"; then
  bad "same seed produced different text"
  diff -u "$TMP/s1.out" "$TMP/s2.out" | head -20
fi
sample_ok=1
grep '^generated ids:' "$TMP/s1.err" > "$TMP/s1.ids" || { bad "sample: no 'generated ids:' line"; sample_ok=0; }
if [ "$sample_ok" != "1" ]; then UPDATE=0; fi

if [ "$sample_ok" = "1" ] && [ "${UPDATE:-0}" = "1" ]; then
  cp "$TMP/s1.out" "$GOLD/run_sample_IQ3_S.txt"
  cp "$TMP/s1.ids" "$GOLD/run_sample_ids_IQ3_S.txt"
  note "updated $GOLD/run_sample_IQ3_S.txt and run_sample_ids_IQ3_S.txt"
else
  for pair in "s1.out:run_sample_IQ3_S.txt" "s1.ids:run_sample_ids_IQ3_S.txt"; do
    got="$TMP/${pair%%:*}"; want="$GOLD/${pair##*:}"
    if [ ! -f "$want" ]; then
      bad "missing golden $want (run with UPDATE=1 to create it)"
    elif ! cmp -s "$got" "$want"; then
      bad "sampled mismatch vs ${pair##*:}"
      diff -u "$want" "$got" | head -30
    fi
  done
fi

# ------------------------------------------------------- stdout purity -----
# stdout must carry generated text only: running with and without -v must give
# byte-identical stdout (every diagnostic belongs on stderr).
run_case "no-verbose" "$TMP/p.out" "$TMP/p.err" --greedy
if ! cmp -s "$TMP/p.out" "$TMP/g.out"; then
  bad "-v changed stdout (a diagnostic leaked into the generated text)"
  diff <(cat "$TMP/g.out") <(cat "$TMP/p.out") | head -10
fi
if [ -s "$TMP/p.err" ]; then
  note "no-verbose stderr: $(wc -l < "$TMP/p.err") line(s)"
else
  bad "stderr is empty without --no-stats (stats should still be printed)"
fi

# ------------------------------------------------------------ seed control --
# A different seed must produce different text: without this, a sampler that
# ignored --seed (constant RNG state) would still pass the "same seed twice"
# check above and look deterministic for the wrong reason.
run_case "sample-c" "$TMP/s3.out" "$TMP/s3.err" --temp 1.0 --seed 43
if cmp -s "$TMP/s1.out" "$TMP/s3.out"; then
  bad "seed 43 produced the same text as seed 42 (the seed is ignored)"
fi

# --------------------------------------------------------------- q4_0 kv ----
# The quantized-cache path must stay coherent too (it is the 64K+ configuration).
# The logits do shift, so there is no full-text golden; the greedy first token
# (" Paris", a large logit gap in f16) must survive the q4_0 cache.
run_case "kv-q4_0" "$TMP/q.out" "$TMP/q.err" --greedy --cache-type-k q4_0 --cache-type-v q4_0
if [ -s "$TMP/q.out" ]; then
  note "kv q4_0: $(wc -c < "$TMP/q.out") bytes"
else
  bad "kv q4_0: empty output"
fi
if ! head -c 8 "$TMP/q.out" | grep -q 'Paris'; then
  bad "kv q4_0: greedy continuation does not start with ' Paris' (got '$(head -c 20 "$TMP/q.out")')"
fi

# mixed cache types (the graph supports k/v independently; only f16/f16 and
# q4_0/q4_0 had been exercised)
for combo in "f16 q8_0" "q8_0 f16" "f32 q4_0"; do
  set -- $combo
  if run_case "kv-$1-$2" "$TMP/m.out" "$TMP/m.err" --greedy --cache-type-k "$1" --cache-type-v "$2"; then
    if head -c 8 "$TMP/m.out" | grep -q 'Paris'; then
      note "kv $1/$2: ok"
    else
      bad "kv $1/$2: continuation does not start with ' Paris' ('$(head -c 20 "$TMP/m.out")')"
    fi
  fi
done

if [ "$fail" = "0" ]; then
  note "check-golden-run: OK"
else
  note "check-golden-run: FAILED"
fi
exit "$fail"
