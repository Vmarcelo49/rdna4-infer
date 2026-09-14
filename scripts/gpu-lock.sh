#!/usr/bin/env bash
# Serialise GPU access between concurrent agents.
#
# A single engine run maps 11.9-15.7 GiB of the card's 15.9 GiB, so two runs at
# the same time fail with "hipMalloc failed" / OOM. Every command that touches the
# GPU (any test, bench, ppl, oracle tool) must go through this wrapper:
#
#   scripts/gpu-lock.sh ./build/check-graph-gpu model.gguf dump.txt -
#   scripts/gpu-lock.sh ./build/rdna4-infer bench -m model.gguf -n 32
#
# It waits (default 1 h) for the lock and then runs the command unchanged, so the
# exit status is the command's.
set -u
LOCK="${GPU_LOCK:-/tmp/rdna4-gpu.lock}"
WAIT="${GPU_LOCK_WAIT:-3600}"
# Reentrancy: GPU_LOCK_HELD=1 means an ANCESTOR of this process already holds the
# lock (this script sets it for every command it runs), so taking it again would
# wait for our own parent until the timeout. That is not hypothetical: a matrix
# script that took one lock and then called this wrapper per configuration spent
# 2 h of the night with the GPU idle and every configuration dying on RC=124.
# Fixing it here makes nesting safe for every caller, including ad-hoc scripts
# that nobody told about the convention.
if [ "${GPU_LOCK_HELD:-0}" = 1 ]; then
  exec "$@"
fi

if ! command -v flock >/dev/null; then
  echo "gpu-lock: flock not found; running without serialisation" >&2
  exec "$@"
fi
# GPU_LOCK_HELD=1 tells the gates that already self-lock (scripts/check_golden_run.sh,
# compare_llama_greedy.sh, check_attn_split.sh, compare_ppl.sh, check_server.sh,
# check_regression.sh) not to take the lock again: flock is NOT reentrant between
# processes, so a gate that locks inside a locked run waits for its own parent until
# the timeout and then fails. Setting it here makes that impossible instead of
# relying on every caller to export it (measured the hard way: an ad-hoc wrapper
# without it hung check_golden_run.sh for 18 minutes).
exec flock -w "$WAIT" "$LOCK" env GPU_LOCK_HELD=1 "$@"
