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
if ! command -v flock >/dev/null; then
  echo "gpu-lock: flock not found; running without serialisation" >&2
  exec "$@"
fi
exec flock -w "$WAIT" "$LOCK" "$@"
