#!/usr/bin/env bash
# Aceite do alvo da rodada noturna: 131K + KV q5_0(K)/q4_1(V) + MTP com ganho real.
#
#   scripts/check_target.sh [model.gguf] [--quick]
#
# Cada bloco diz o que mede e imprime PASS/FAIL; recursos que ainda não existem no
# build são marcados SKIP (a rodada noturna implementa q5_0/q4_1 e a verificação em
# batch do MTP em frentes separadas). Tudo passa por scripts/gpu-lock.sh com timeout.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
B="$ROOT/build"
MODEL="${1:-/mnt/raid0/GGUF/unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_S.gguf}"
QUICK=0
for a in "$@"; do [ "$a" = "--quick" ] && QUICK=1; done
fail=0
say() { printf '%s\n' "$*"; }
ok() { printf '  OK    %s\n' "$*"; }
bad() { printf '  FAIL  %s\n' "$*"; fail=1; }
skip() { printf '  SKIP  %s\n' "$*"; }
# uma sessão de GPU por bloco, com timeout duro
# timeout INSIDE the lock: the other order burns the budget waiting in the queue and
# fails with RC=124 without ever running (docs/noite-regras.md 1.2, review finding R6).
gpu() { "$ROOT/scripts/gpu-lock.sh" timeout 900 "$@"; }
vram_mib() { cat /sys/class/drm/card*/device/mem_info_vram_used 2>/dev/null | head -1 | awk '{printf "%d", $1/1048576}'; }

say "== alvo: 131K de contexto, KV q5_0/q4_1, MTP com ganho =="
say "modelo: $MODEL"
say "VRAM antes: $(vram_mib) MiB"

# ---------------------------------------------------------------- 1. formatos KV
say
say "1. os formatos de KV do alvo existem?"
if gpu "$B/rdna4-infer" info -m "$MODEL" --ctx-size 131072 --cache-type-k q5_0 \
      --cache-type-v q4_1 > /tmp/target-kv.log 2>&1; then
  ok "q5_0/q4_1 aceitos; $(grep -o 'kv(ctx=[^)]*)' /tmp/target-kv.log | head -1) $(tail -1 /tmp/target-kv.log)"
  grep -q "budget OK" /tmp/target-kv.log && ok "orçamento de VRAM aprovado em 131K" \
                                         || bad "orçamento reprovado em 131K: $(tail -1 /tmp/target-kv.log)"
else
  bad "info com q5_0/q4_1 falhou: $(tail -2 /tmp/target-kv.log | tr '\n' ' ')"
fi

# ------------------------------------------------------------- 2. decode a 131K
say
say "2. decode a 131K com o KV do alvo (mede tok/s e VRAM; GTT = falha silenciosa)"
if gpu "$B/rdna4-infer" bench -m "$MODEL" -n 16 --reps 2 --ctx-size 131072 \
     --start-pos 131000 --fill-cache --cache-type-k q5_0 --cache-type-v q4_1 \
     > /tmp/target-131k.log 2>&1; then
  tps=$(grep -o 'best [0-9.]* s ([0-9.]* tok/s)' /tmp/target-131k.log | grep -o '([0-9.]*' | tr -d '(')
  used=$(grep -o 'vram in use: [0-9.]* GiB' /tmp/target-131k.log | head -1)
  say "  $used ; decode ${tps:-?} tok/s"
  awk -v t="${tps:-0}" 'BEGIN{exit !(t >= 10)}' \
    && ok "131K acima de 10 tok/s" || bad "131K abaixo de 10 tok/s (${tps:-?}) — suspeita de transbordo para GTT"
else
  bad "bench em 131K com q5_0/q4_1 falhou: $(tail -2 /tmp/target-131k.log | tr '\n' ' ')"
fi
gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1 | awk '{printf "%d", $1/1048576}')
say "  gtt_used depois: ${gtt:-?} MiB (acima de ~600 MiB indica transbordo)"

# ------------------------------------------------------------------------- 3. MTP
say
say "3. MTP entrega ganho sobre greedy puro? (mesmo prompt, mesmo greedy)"
PROMPT="Explain in three sentences why memory bandwidth limits token generation."
if gpu "$B/rdna4-infer" run -m "$MODEL" -p "$PROMPT" -n 64 --greedy --no-stats \
     > /tmp/target-greedy.txt 2>/tmp/target-greedy.err; then
  g_tps=$(grep -o '[0-9.]* tok/s' /tmp/target-greedy.err | tail -1)
  say "  greedy: ${g_tps:-?}"
  if gpu "$B/rdna4-infer" run -m "$MODEL" -p "$PROMPT" -n 64 --greedy --mtp --no-stats \
       > /tmp/target-mtp.txt 2>/tmp/target-mtp.err; then
    m_tps=$(grep -o '[0-9.]* tok/s' /tmp/target-mtp.err | tail -1)
    say "  mtp   : ${m_tps:-?}"
    if cmp -s /tmp/target-greedy.txt /tmp/target-mtp.txt; then
      ok "a saída com --mtp é idêntica à do greedy puro"
    else
      bad "--mtp mudou a saída (não é mais exato)"
      diff <(head -c 400 /tmp/target-greedy.txt) <(head -c 400 /tmp/target-mtp.txt) | head -4
    fi
    awk -v g="${g_tps% tok/s}" -v m="${m_tps% tok/s}" 'BEGIN{exit !(m > g*1.15)}' \
      && ok "MTP ≥ 1,15× o greedy puro" \
      || bad "MTP sem ganho relevante (${g_tps:-?} -> ${m_tps:-?}) — 1,15× é o piso do alvo"
  else
    bad "--mtp falhou: $(tail -2 /tmp/target-mtp.err | tr '\n' ' ')"
  fi
else
  bad "greedy falhou: $(tail -2 /tmp/target-greedy.err | tr '\n' ' ')"
fi

# ---------------------------------------------------------------------- 4. gates
say
say "4. correção (não se negocia)"
n=0
run_gate() { local l="$1"; shift; n=$((n+1));
  if timeout 900 "$@" > /tmp/target-gate.log 2>&1; then ok "$l"; else bad "$l: $(tail -3 /tmp/target-gate.log | tr '\n' ' ')"; fi; }
run_gate "check_golden_run.sh" "$ROOT/scripts/check_golden_run.sh"
run_gate "check_regression.sh" "$ROOT/scripts/check_regression.sh"
run_gate "greedy vs llama.cpp (32)" "$ROOT/scripts/compare_llama_greedy.sh" 32
if [ "$QUICK" -eq 0 ]; then
  run_gate "check_attn_split.sh" "$ROOT/scripts/check_attn_split.sh"
  run_gate "ppl vs llama.cpp" "$ROOT/scripts/compare_ppl.sh" "$MODEL" 10
  run_gate "check-kvctx-gpu 131072 q4_0" gpu "$B/check-kvctx-gpu" "$MODEL" 131072 q4_0
fi

say
say "VRAM depois: $(vram_mib) MiB"
if [ "$fail" -eq 0 ]; then say "check_target: PASS ($n gates de correção)"; else say "check_target: FAIL"; fi
exit "$fail"
