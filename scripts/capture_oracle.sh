#!/usr/bin/env bash
# Captures the llama.cpp per-node activation dump used as the M3 oracle.
#
# `llama-eval-callback` prints every named node of the qwen35 graph (the names
# come from the cb() calls in llama.cpp's src/models/qwen35.cpp), with its first
# and last three values plus the element sum. check-nn-gpu (and the M3 layer
# tests) diff our kernels against those numbers.
#
# Usage: scripts/capture_oracle.sh [prompt] [n_predict] [ngl]
#   default: "Hello", 1 token, -ngl 0 (CPU: deterministic, no GPU scheduling
#            differences; pass 99 to capture the Vulkan path instead)
#
# Output: reference/oracle_<prompt>_<backend>.txt  (gitignored: it is ~4 MB)
set -euo pipefail

LLAMA_DIR="${LLAMA_DIR:-/home/marcelo/Projetos/llama.cpp}"
EVAL_CB="${EVAL_CB:-$LLAMA_DIR/build/bin/llama-eval-callback}"
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"

PROMPT="${1:-Hello}"
NPRED="${2:-1}"
NGL="${3:-0}"

if [[ ! -x "$EVAL_CB" ]]; then
  echo "error: $EVAL_CB not found (build llama.cpp first, or set EVAL_CB)" >&2
  exit 1
fi
if [[ ! -f "$MODEL" ]]; then
  echo "error: model not found: $MODEL (set MODEL=)" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/reference"

BACKEND=$([[ "$NGL" == "0" ]] && echo cpu || echo gpu)
OUT="$ROOT/reference/oracle_${PROMPT}_${BACKEND}.txt"

echo "capturing: prompt='$PROMPT' n_predict=$NPRED ngl=$NGL"
echo "  model : $MODEL"
echo "  output: $OUT"

# --seed/--temp make the sampling deterministic; with n_predict=1 the graph is
# the prefill of a single token, which is what the layer tests compare against.
"$EVAL_CB" -m "$MODEL" -p "$PROMPT" -n "$NPRED" -ngl "$NGL" --seed 1 --temp 0 > "$OUT" 2>&1

echo "  nodes : $(grep -c 'common_debug_cb_eval' "$OUT")"
echo "  token : $(grep -oE 'number of input tokens = [0-9]+' "$OUT" | head -1) / $(grep -A1 'number of input tokens' "$OUT" | tail -1 | tr -d ' ')"
