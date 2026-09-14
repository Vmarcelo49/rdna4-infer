#!/usr/bin/env bash
# Acceptance test for the OpenAI-compatible server (`rdna4-infer serve`).
#
# The server maps the whole model on the GPU (12-16 GiB of the card's 15.9 GiB),
# so the client runs under scripts/gpu-lock.sh like every other GPU command; with
# --greedy-cli-check the test also runs `run --chat --greedy` *after* stopping the
# server, so the two model loads are sequential and never overlap.
#
#   ./scripts/check_server.sh
#   MODEL=/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ4_XS.gguf \
#     ./scripts/check_server.sh --max-tokens 24
#
# Build first:  cmake --build build -j6 --target rdna4-infer
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/rdna4-infer}"
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
PORT="${PORT:-8099}"
CTX="${CTX:-4096}"
GREEDY_CHECK="${GREEDY_CHECK:-1}"

if [ ! -x "$BIN" ]; then
  echo "no binary at $BIN (build first: cmake --build build -j6 --target rdna4-infer)" >&2
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  echo "no model at $MODEL (set MODEL=...)" >&2
  exit 2
fi

# The CPU-only half of the server has its own fast, GPU-free check.
if [ -x "$ROOT/build/check-server-http" ]; then
  "$ROOT/build/check-server-http" || exit 1
fi

args=(python3 "$ROOT/tests/check_server.py" --binary "$BIN" --model "$MODEL"
      --port "$PORT" --ctx-size "$CTX")
if [ "$GREEDY_CHECK" = "1" ]; then
  args+=(--greedy-cli-check)
fi
args+=("$@")

echo "== check_server.sh: $(basename "$MODEL"), port $PORT, ctx $CTX =="
exec "$ROOT/scripts/gpu-lock.sh" "${args[@]}"
