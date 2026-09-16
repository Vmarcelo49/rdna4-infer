#!/usr/bin/env bash
# M4 acceptance (determinism): the same prompt + the same seed must produce
# exactly the same bytes, and the committed golden output must not change.
#
#   ./scripts/check_golden_run.sh            # verify against tests/golden/
#   UPDATE=1 ./scripts/check_golden_run.sh   # rewrite the golden files
#   QUICK=1 ./scripts/check_golden_run.sh    # fast gate (N=16, skips nightly cases)
#   ./scripts/check_golden_run.sh --quick    # same as QUICK=1
#
# QUICK mode (QUICK=1 env or --quick flag):
#   (a) N defaults to 16 instead of 32; an explicit N env still wins.
#   (b) G7-G9 mixed-KV combos are skipped with a "SKIP (nightly)" note;
#       G6 (kv-q4_0, the shipped config) still runs.
#   (c) the argmax-vs-host loop runs 1 prompt x 2 arms instead of 3 prompts x 2.
#   QUICK=1 refuses UPDATE=1 (goldens are full-mode): rewrite without --quick.
#
# Reuse hook for scripts/compare_llama_greedy.sh (that script is NOT edited here):
#   G1's greedy stdout/full-stderr ($TMP/g.out, $TMP/g.err) are copied on success
#   to the well-known path $ROOT/.tmp-golden-g1.out / $ROOT/.tmp-golden-g1.err so
#   the compare script can reuse them without a second 12GB engine load (it needs
#   the full stderr: prompt ids AND generated ids). $TMP/g.ids (generated only)
#   goes to $ROOT/.tmp-golden-g1.ids for completeness.
#   If GOLDEN_REUSE is set to an existing directory, copies are also placed there
#   as $GOLDEN_REUSE/g.out, g.err and g.ids.
# The golden files are the engine's own output for a fixed prompt/seed (there is
# no independent oracle for a full sampled continuation — llama.cpp's RNG stream
# is not ours, see PLAN.md M4); the independent cross-check of *greedy* decoding
# against llama.cpp is scripts/compare_llama_greedy.sh.
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
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
GOLD="${GOLD:-$ROOT/tests/golden}"
PROMPT="${PROMPT:-The capital of France is}"
QUICK="${QUICK:-0}"
for _a in ${1+"$@"}; do
  if [ "$_a" = "--quick" ]; then QUICK=1; fi
done
unset _a
# QUICK default N=16; an explicit non-empty N env still wins.
if [ -z "${N:-}" ]; then
  if [ "$QUICK" = "1" ]; then N=16; else N=32; fi
fi
if [ "$QUICK" = "1" ] && [ "${UPDATE:-0}" = "1" ]; then
  echo "QUICK=1 refuses UPDATE=1: goldens are full-mode (rewrite without --quick)" >&2
  exit 2
fi
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

fail=0
note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*"; fail=1; }

if [ "$QUICK" = "1" ]; then
  note "quick mode: N=$N, mixed-KV combos skipped, 1 argmax prompt"
fi

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

# Reuse hook (see header): publish G1's greedy outputs so
# scripts/compare_llama_greedy.sh can reuse them without reloading the engine.
if [ "$greedy_ok" = "1" ] && [ -f "$TMP/g.out" ] && [ -f "$TMP/g.err" ]; then
  cp "$TMP/g.out" "$ROOT/.tmp-golden-g1.out"
  cp "$TMP/g.err" "$ROOT/.tmp-golden-g1.err"
  cp "$TMP/g.ids" "$ROOT/.tmp-golden-g1.ids"
  if [ -n "${GOLDEN_REUSE:-}" ] && [ -d "$GOLDEN_REUSE" ]; then
    cp "$TMP/g.out" "$GOLDEN_REUSE/g.out"
    cp "$TMP/g.err" "$GOLDEN_REUSE/g.err"
    cp "$TMP/g.ids" "$GOLDEN_REUSE/g.ids"
  fi
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
# q4_0/q4_0 had been exercised). Nightly-only: skipped in QUICK mode.
if [ "$QUICK" = "1" ]; then
  note "SKIP (nightly): mixed KV combos f16/q8_0, q8_0/f16, f32/q4_0"
else
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
fi

# ---------------------------------------------------------------------------
# The device argmax (greedy fast path) and the host sampler must return the SAME id
# for every prompt shape. This is not decoration: `prefill_ids` ends in a BATCH
# chunk for most prompt lengths (7 = 4+3, 12 = 8+4), and `forward_batch` used to
# leave the device argmax stale, so `run --greedy` emitted token 0 ("!") as its
# first generated token. The stored golden uses a 5-token prompt (4+1) -- the one
# shape that does not trigger it -- which is why the suite stayed green (review
# finding R9). RD_NO_ARGMAX=1 forces the host path, so this is a differential test
# with no stored fixture to go stale.
argmax_prompts=("The capital of France is Paris"
  "Explain in one sentence what a KV cache is"
  "one two three four five six seven eight nine ten eleven twelve")
if [ "$QUICK" = "1" ]; then
  argmax_prompts=("The capital of France is Paris")
fi
for p in "${argmax_prompts[@]}"; do
  a="$("$BIN" run -m "$MODEL" -p "$p" -n 8 --greedy --no-stats 2>/dev/null)"
  b="$(RD_NO_ARGMAX=1 "$BIN" run -m "$MODEL" -p "$p" -n 8 --greedy --no-stats 2>/dev/null)"
  if [ -z "$a" ] || [ -z "$b" ]; then
    bad "argmax vs host sampler: one arm produced nothing ('$p')"
  elif [ "$a" != "$b" ]; then
    bad "argmax vs host sampler differ for '$p': fast='$(printf '%s' "$a" | head -c 30)' host='$(printf '%s' "$b" | head -c 30)'"
  else
    note "argmax == host sampler: $(printf '%s' "$p" | cut -c1-28)…"
  fi
done

if [ "$fail" = "0" ]; then
  note "check-golden-run: OK"
else
  note "check-golden-run: FAILED"
fi
exit "$fail"
