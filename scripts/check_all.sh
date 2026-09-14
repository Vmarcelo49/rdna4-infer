#!/usr/bin/env bash
# One GPU session, every gate.
#
#   scripts/check_all.sh [model.gguf] [--quick]
#
# Three phases, so the GPU lock is taken exactly once and never nested:
#   A. CPU gates, no lock (they allocate no VRAM).
#   B. every GPU gate that does NOT lock by itself, in one re-exec under
#      scripts/gpu-lock.sh — the model is loaded a handful of times instead of a
#      dozen, and no second process can slip in between the gates.
#   C. the gates that take the lock themselves (check_server.sh execs gpu-lock),
#      run last so they cannot deadlock against B.
#
# --quick skips the long-context gates (64K/128K decode, check-kvctx-gpu, ppl):
# what a code change needs while iterating, while the full set is what a
# milestone close needs. Gate names and arguments match README.md, so a failure
# here is reproducible by hand.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="$ROOT/build"
MODEL="${1:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
QUICK=0
for a in "$@"; do [ "$a" = "--quick" ] && QUICK=1; done
PHASE="${CHECK_ALL_PHASE:-all}"   # all | gpu (the re-exec that holds the lock)
fail=0
step() {  # step <label> <cmd...>
  local label="$1"; shift
  local t0=$SECONDS
  if "$@" > /tmp/check_all_step.log 2>&1; then
    printf '  OK    %-32s %4ds\n' "$label" "$((SECONDS - t0))"
  else
    printf '  FAIL  %-32s %4ds  (tail of /tmp/check_all_step.log)\n' "$label" "$((SECONDS - t0))"
    tail -6 /tmp/check_all_step.log | sed 's/^/        /'
    fail=1
  fi
}
quick_arg() { [ "$QUICK" = 1 ] && printf '%s' --quick; }

if [ "$PHASE" = all ]; then
  echo "== A. CPU gates (no GPU lock) =="
  step "check-dtype"       "$B/check-dtype" "$MODEL"
  step "check-tokenizer"   "$B/check-tokenizer" "$MODEL" "$ROOT/tests/golden/tokenizer_ids.txt" "$ROOT/tests/tokenizer_corpus.txt"
  step "check-eog"         "$B/check-eog" "$MODEL" "$ROOT/tests/golden/eog_ids.txt"
  step "check-chat"        "$B/check-chat" "$ROOT/tests/golden/chat_renders.txt"
  step "check-sampler"     "$B/check-sampler"
  step "check-stream"      "$B/check-stream"
  step "check-server-http" "$B/check-server-http"
  step "check-loader"      "$B/check-loader" "$MODEL"
  step "check-hardening"   "$ROOT/scripts/check_hardening.sh"
  [ -x "$B/check-tuning" ] && step "check-tuning" "$B/check-tuning"

  if [ "${CHECK_ALL_CPU_ONLY:-0}" = 1 ]; then
    if [ "$fail" -eq 0 ]; then echo "check_all (CPU half): PASS"; else echo "check_all (CPU half): FAIL"; fi
    exit "$fail"
  fi

  echo "== B. GPU gates under a single lock =="
  if ! "$ROOT/scripts/gpu-lock.sh" env GPU_LOCK_HELD=1 CHECK_ALL_PHASE=gpu \
         "$0" "$MODEL" $(quick_arg); then
    fail=1
  fi

  echo "== C. gates that lock by themselves =="
  step "serve (HTTP)" "$ROOT/scripts/check_server.sh"
  if [ "$fail" -eq 0 ]; then echo "check_all: PASS"; else echo "check_all: FAIL"; fi
  exit "$fail"
fi

# ---- phase B: the whole block runs with the lock already held ----------------
step "check-graph-gpu"    env GRAPH_LAST_TOKEN=1 "$B/check-graph-gpu" "$MODEL" \
                               "$ROOT/reference/oracle_prompt6_ub1_tok7_cpu.txt" -
step "check-dequant-gpu"  "$B/check-dequant-gpu" "$MODEL"
step "check-matvec-gpu"   "$B/check-matvec-gpu" "$MODEL"
step "check-nn-gpu"       "$B/check-nn-gpu" "$MODEL" \
                               "$ROOT/reference/oracle_hello_cpu.txt" 9419
step "check-rope-gpu"     "$B/check-rope-gpu"
step "check-matmul-gpu"   "$B/check-matmul-gpu" "$MODEL"
step "check-batch-gpu"    "$B/check-batch-gpu" "$MODEL"
step "check-mtp-gpu"      "$B/check-mtp-gpu" "$MODEL" 64 3
step "golden run"         "$ROOT/scripts/check_golden_run.sh"
step "greedy vs llama"    "$ROOT/scripts/compare_llama_greedy.sh" 32
if [ "$QUICK" -eq 0 ]; then
  step "attn split vs text" "$ROOT/scripts/check_attn_split.sh"
  step "kvctx 64K q4_0"     "$B/check-kvctx-gpu" "$MODEL" 65536 q4_0
  step "ppl vs llama"       "$ROOT/scripts/compare_ppl.sh" "$MODEL" 10
fi
if [ -x "$B/check-regression-gpu" ]; then
  step "regression suite"   "$B/check-regression-gpu" "$MODEL" \
                            "$ROOT/tests/golden/regression_greedy_f16.txt"
fi

if [ "$fail" -eq 0 ]; then echo "check_all(B): PASS"; else echo "check_all(B): FAIL"; fi
exit "$fail"
