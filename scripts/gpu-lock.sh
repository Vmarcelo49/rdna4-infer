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
# The lock serialises *runs*, but a process that just died (SIGTERM, timeout, a kill)
# keeps its VRAM for a few seconds while the driver drains it -- and a model load
# that starts inside that window dies with "hipMalloc failed for blk.N...", which
# looks like a broken engine. Measured four times tonight (twice as false-red gates).
# So: after taking the lock, wait for the card to go quiet before handing over.
exec flock -w "$WAIT" "$LOCK" env GPU_LOCK_HELD=1 bash -c '
  # 1 GiB: the desktop and a fresh context sit far below it; a draining model is
  # several GiB above it.
  f=$(ls /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1)
  if [ -n "$f" ]; then
    t0=$SECONDS
    while [ $((SECONDS - t0)) -lt "${GPU_LOCK_DRAIN_WAIT:-90}" ]; do
      used=$(cat "$f" 2>/dev/null || echo 0)
      [ "${used:-0}" -lt $((1 << 30)) ] && break
      sleep 2
    done
    used=$(cat "$f" 2>/dev/null || echo 0)
    if [ "${used:-0}" -ge $((1 << 30)) ]; then
      echo "gpu-lock: aviso: VRAM ainda em $((used / 1048576)) MiB apos a espera; seguindo" >&2
    fi
  fi
  exec "$@"
' -- "$@"
