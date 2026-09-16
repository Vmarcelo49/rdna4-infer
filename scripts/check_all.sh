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
missing=""   # gate binaries that should exist but do not (see F3)
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
  echo "== A. CPU gates (no GPU lock, parallel) =="
  # The 10 CPU steps share nothing but the read-only model file and allocate no
  # VRAM, so they fan out with `&` and rejoin with a single `wait` instead of
  # running serially behind one shared /tmp/check_all_step.log. Each step gets
  # its own log (/tmp/check_all_<label>.log); its OK/FAIL line goes to a sibling
  # .report file because a background subshell cannot set the parent's fail
  # variable — aggregation (fail=1 plus the missing-binary guard) happens in
  # label order after `wait`, so the per-label lines read exactly as serially.
  A_TAGS=""
  astep() {  # astep <label> <cmd...>
    local label="$1"; shift
    local log="/tmp/check_all_${label}.log"
    rm -f "/tmp/check_all_${label}.report" "/tmp/check_all_${label}.fail"
    A_TAGS="$A_TAGS $label"
    ( _t0=$SECONDS
      if "$@" >"$log" 2>&1; then
        printf '  OK    %-32s %4ds\n' "$label" "$((SECONDS - _t0))" \
          >"/tmp/check_all_${label}.report"
      else
        { printf '  FAIL  %-32s %4ds  (tail of %s)\n' "$label" "$((SECONDS - _t0))" "$log"
          tail -6 "$log" | sed 's/^/        /'
        } >"/tmp/check_all_${label}.report"
        : >"/tmp/check_all_${label}.fail"
      fi
    ) &
  }
  astep "check-dtype"       "$B/check-dtype" "$MODEL"
  astep "check-tokenizer"   "$B/check-tokenizer" "$MODEL" "$ROOT/tests/golden/tokenizer_ids.txt" "$ROOT/tests/tokenizer_corpus.txt"
  astep "check-eog"         "$B/check-eog" "$MODEL" "$ROOT/tests/golden/eog_ids.txt"
  astep "check-chat"        "$B/check-chat" "$ROOT/tests/golden/chat_renders.txt"
  astep "check-sampler"     "$B/check-sampler"
  astep "check-stream"      "$B/check-stream"
  astep "check-server-http" "$B/check-server-http"
  astep "check-loader"      "$B/check-loader" "$MODEL"
  astep "check-hardening"   "$ROOT/scripts/check_hardening.sh"
  if [ -x "$B/check-tuning" ]; then
    astep "check-tuning" "$B/check-tuning"
  else
    # A missing gate binary must FAIL, not quietly shrink the battery: the review
    # front proved this construct printed PASS with 9 of 10 gates (finding F3).
    # Reported through the same per-label file so it keeps its serial position.
    A_TAGS="$A_TAGS check-tuning"
    printf '  FAIL  %-32s binary not built (%s)\n' "check-tuning" "$B/check-tuning" \
      > /tmp/check_all_check-tuning.report
    : > /tmp/check_all_check-tuning.fail
    missing="$missing check-tuning"
    fail=1
  fi
  wait
  for _tag in $A_TAGS; do
    cat "/tmp/check_all_${_tag}.report"
    [ -f "/tmp/check_all_${_tag}.fail" ] && fail=1
  done

  if [ "${CHECK_ALL_CPU_ONLY:-0}" = 1 ]; then
    if [ "$fail" -eq 0 ]; then echo "check_all (CPU half): PASS"; else echo "check_all (CPU half): FAIL"; fi
    exit "$fail"
  fi

  echo "== B. GPU gates under a single lock =="
  # gpu-lock.sh itself exports GPU_LOCK_HELD=1 to its children (see that script):
  # the gates that self-lock must not re-lock, or they wait for their own parent.
  if ! "$ROOT/scripts/gpu-lock.sh" env CHECK_ALL_PHASE=gpu "$0" "$MODEL" $(quick_arg); then
    fail=1
  fi

  echo "== C. gates that lock by themselves =="
  step "serve (HTTP)" "$ROOT/scripts/check_server.sh" $(quick_arg)
  if [ -n "$missing" ]; then
    echo "check_all: FAIL — gate binaries missing:$missing (build them: cmake --build build)"
  fi
  if [ "$fail" -eq 0 ]; then echo "check_all: PASS"; else echo "check_all: FAIL"; fi
  exit "$fail"
fi

# ---- phase B: the whole block runs with the lock already held ----------------
if [ "$QUICK" = 1 ]; then
  # Iteration knobs for the GPU binaries that support them (all self-consistent:
  # internal bit-exact/numerics checks, no committed goldens). Deliberately NOT
  # exported: REGRESSION_QUICK (a quick run needs its own --record golden) and
  # anything affecting check_golden_run.sh (its outputs are compared against the
  # committed full-mode goldens, so QUICK there would always FAIL).
  export BATCH_QUICK=1 MTP_QUICK=1 KVCTX_QUICK=1 MATMUL_QUICK=1 ROPE_QUICK=1 \
         KVQUANT_QUICK=1
fi
# Stale guard for the golden->llama reuse below: the .tmp files must come from
# THIS run's golden step, never from a previous checkout state.
rm -f "$ROOT/.tmp-golden-g1.out" "$ROOT/.tmp-golden-g1.err" "$ROOT/.tmp-golden-g1.ids"
step "check-graph-gpu"    env GRAPH_LAST_TOKEN=1 "$B/check-graph-gpu" "$MODEL" \
                               "$ROOT/reference/oracle_prompt6_ub1_tok7_cpu.txt" -
step "check-dequant-gpu"  "$B/check-dequant-gpu" "$MODEL"
step "check-matvec-gpu"   "$B/check-matvec-gpu" "$MODEL"
step "check-nn-gpu"       "$B/check-nn-gpu" "$MODEL" \
                               "$ROOT/reference/oracle_hello_cpu.txt" 9419
step "check-rope-gpu"     "$B/check-rope-gpu"
step "check-matmul-gpu"   "$B/check-matmul-gpu" "$MODEL"
step "check-batch-gpu"    "$B/check-batch-gpu" "$MODEL"
# Pares de KV no caminho em lote: o gate acima crava o default (f16/f16) e foi por
# isso que o bug de stride de V sobreviveu (K=q5_0/V=q4_1 = page fault no prefill).
step "check-kvbatch"      "$ROOT/scripts/check_kvbatch.sh" "$MODEL" $(quick_arg)
step "check-mtp-gpu"      "$B/check-mtp-gpu" "$MODEL" 64 3
step "golden run"         "$ROOT/scripts/check_golden_run.sh"
# Same prompt/flags/N as golden G1 (defaults): reuse its outputs when this run's
# golden step just published them (stale files were removed above, and the
# compare script itself requires GOLDEN_REUSE_N/PROMPT to match). Saves 1 load.
step "greedy vs llama"    env GOLDEN_REUSE_TXT="$ROOT/.tmp-golden-g1.out" \
                              GOLDEN_REUSE_ERR="$ROOT/.tmp-golden-g1.err" \
                              GOLDEN_REUSE_N=32 \
                              GOLDEN_REUSE_PROMPT="The capital of France is" \
                              "$ROOT/scripts/compare_llama_greedy.sh" 32
if [ "$QUICK" -eq 0 ]; then
  step "attn split vs text" "$ROOT/scripts/check_attn_split.sh"
  step "kvctx 64K q4_0"     "$B/check-kvctx-gpu" "$MODEL" 65536 q4_0
  step "ppl vs llama"       "$ROOT/scripts/compare_ppl.sh" "$MODEL" 10
fi
if [ -x "$B/check-regression-gpu" ]; then
  step "regression suite"   "$B/check-regression-gpu" "$MODEL" \
                            "$ROOT/tests/golden/regression_greedy_f16.txt"
else
  printf '  FAIL  %-32s binary not built (%s)\n' "regression suite" "$B/check-regression-gpu"
  missing="$missing check-regression-gpu"
  fail=1
fi

if [ -n "$missing" ]; then
  echo "check_all(B): FAIL — gate binaries missing:$missing (build them: cmake --build build)"
fi
if [ "$fail" -eq 0 ]; then echo "check_all(B): PASS"; else echo "check_all(B): FAIL"; fi
exit "$fail"
