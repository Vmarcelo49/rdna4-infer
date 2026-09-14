#!/usr/bin/env bash
# Suite de regressao numerica (task 7) — o gate que roda depois de QUALQUER
# mudanca no motor. Ver docs/autotuning-gfx1201.md (§gate de pre-merge).
#
#   ./scripts/check_regression.sh              # verifica contra o golden commitado
#   UPDATE=1 ./scripts/check_regression.sh     # regrava o golden (so depois de
#                                              # validar a mudanca com os outros gates)
#
# O que ele compara (detalhes em tests/check_regression_gpu.hip):
#   - 7 prompts fixos (curto, prosa longa, codigo, multilingue/CJK, chat com o
#     template do modelo, prefill real de 4096 tokens, e 64K com cache semeado);
#   - ids gerados em greedy puro: BIT-EXATOS;
#   - logits do ultimo passo: rel-L2 <= 1e-5 na subamostra e |d||logits|||/||logits||
#     <= 1e-5 (a tolerancia existe para reordenamento legitimo da atencao).
#
# Falha alto se a referencia sumir (o braco do teste devolve 1 e este script
# propaga o codigo). Custo medido: ~2 min (o alvo do gate e <= 5 min).
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BIN:-$ROOT/build/check-regression-gpu}"
MODEL="${MODEL:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
GOLD="${GOLD:-$ROOT/tests/golden/regression_greedy_f16.txt}"
LOCK="$ROOT/scripts/gpu-lock.sh"

note() { printf '%s\n' "$*"; }
bad() { printf 'FAIL: %s\n' "$*" >&2; }

if [ ! -x "$BIN" ]; then
  bad "nao ha binario em $BIN (rode: cmake --build build --target check-regression-gpu)"
  exit 2
fi
if [ ! -f "$MODEL" ]; then
  bad "nao ha modelo em $MODEL (defina MODEL=...)"
  exit 2
fi
if [ "${UPDATE:-0}" != "1" ] && [ ! -f "$GOLD" ]; then
  # Referencia ausente NAO pode virar PASS: sem baseline nao ha o que comparar.
  bad "referencia de regressao AUSENTE: $GOLD (grave com UPDATE=1 apos validar a mudanca)"
  exit 2
fi
if [ ! -x "$LOCK" ]; then
  bad "sem $LOCK (toda execucao de GPU passa pelo lock)"
  exit 2
fi

args=("$MODEL" "$GOLD")
if [ "${UPDATE:-0}" = "1" ]; then
  args+=(--record)
fi

# Toda execucao que toca a GPU passa pelo scripts/gpu-lock.sh (docs/gpu-queue.md).
# Dois cuidados que este lote aprendeu na marra:
#  1. o script se trava SOZINHO quando ninguem tem o lock (senao quem chamar
#     `check_regression.sh` direto fura a fila);
#  2. com GPU_LOCK_HELD=1 (quem ja tem o lock, ex. scripts/check_all.sh da main) ele
#     NAO trava de novo -- flock no mesmo arquivo por outro processo trava para sempre.
if [ "${GPU_LOCK_HELD:-0}" != "1" ]; then
  export GPU_LOCK_HELD=1
  exec "$LOCK" "$0" "$@"
fi

start=$(date +%s)
out="$(mktemp)"
trap 'rm -f "$out"' EXIT
"$BIN" "${args[@]}" 2>&1 | tee "$out"
rc=${PIPESTATUS[0]}
end=$(date +%s)
note "check-regression: $(basename "$BIN") rc=$rc, parede $(($end - $start)) s ($GOLD)"

if [ "$rc" != "0" ]; then
  bad "suite de regressao FALHOU (rc=$rc) — saida acima"
  exit "$rc"
fi
# Sanidade do relatorio: um PASS sem as linhas de comparacao seria um gate vazio.
if ! grep -q 'rel-L2' "$out"; then
  bad "saida sem linha de comparacao de logits (gate vazio)"
  exit 1
fi
note "check-regression: OK"
exit 0
