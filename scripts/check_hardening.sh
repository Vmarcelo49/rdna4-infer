#!/usr/bin/env bash
# Hardening check for the GGUF loader (audit findings M7/M8).
#
# Forges four headers out of the valid fake model (tests/gen_forged_gguf.py),
# each with a single file-controlled field that the old code trusted, and
# asserts the hardened loader rejects all of them cleanly: an error message, no
# signal (an uncaught bad_alloc/length_error shows up as 134/SIGABRT), and no
# tensor read at the forged offset.
#
#   scripts/check_hardening.sh                       # hardened loader only
#   scripts/check_hardening.sh <pre_fix_worktree>    # + before/after comparison
#
# The optional argument is the *root of a pre-fix checkout* (any worktree
# branched before this fix): the probe is compiled against those sources, which
# is what turns the table from "we reject it" into "we used to accept it and
# read KV bytes as weights".
#
# Pure CPU: tests/check_forge.cpp only parses the header and reads one small
# tensor (it makes no HIP call at all), so this does NOT take the GPU lock.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROBE="${PROBE:-$ROOT/build/check-forge}"
OLD_SRC="${1:-}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 "$ROOT/tests/gen_forged_gguf.py" "$TMP" >/dev/null || exit 1

fail=0
# probe <binary> <file> <tensor> -> "rejected: ...", "accepted: ...", "signal N"
probe() {
  local out code
  out="$("$1" "$2" "$3" 2>&1)"; code=$?
  if [ "$code" -ge 128 ]; then echo "signal $((code - 128)) ($out)"; return; fi
  if [ "$code" -eq 2 ]; then echo "usage error ($out)"; return; fi
  printf '%s\n' "$(printf '%s' "$out" | head -1)"
}

TEN_LAST=blk.1.nextn.shared_head_norm.weight
TEN_FIRST=token_embd.weight
echo "== check_hardening: hardened loader ($PROBE) =="
if [ ! -x "$PROBE" ]; then
  echo "  FAIL  $PROBE not built (make check-forge)"; exit 1
fi
check() {  # check <file> <tensor> <expect: accept|reject> <must-match>
  local v; v="$(probe "$PROBE" "$TMP/$1" "$2")"
  if [ "$3" = accept ]; then
    case "$v" in accepted*) echo "  OK    $1: $v" ;; *) echo "  FAIL  $1: $v"; fail=1 ;; esac
  else
    case "$v" in
      rejected:*"$4"*) echo "  OK    $1: $v" ;;
      *) echo "  FAIL  $1: $v (wanted a rejection mentioning '$4')"; fail=1 ;;
    esac
  fi
}
check ok.gguf "$TEN_FIRST" accept
check forge_offset_wrap.gguf "$TEN_FIRST" reject "offset/size sum overflows 64 bits"
check forge_dim_huge.gguf "$TEN_LAST" reject "exceeds the"
check forge_str_len.gguf "$TEN_FIRST" reject "truncated or corrupt header"
check forge_arr_len.gguf "$TEN_FIRST" reject "truncated or corrupt header"

if [ -n "$OLD_SRC" ]; then
  echo
  echo "== before/after against $OLD_SRC =="
  oldprobe="$TMP/check-forge-old"
  if ! g++ -O2 -std=c++17 -I"$OLD_SRC/include" "$ROOT/tests/check_forge.cpp" \
       "$OLD_SRC/src/backend/gguf.cpp" "$OLD_SRC/src/backend/loader.cpp" \
       -o "$oldprobe" 2>"$TMP/gcc.log"; then
    echo "  FAIL  old probe did not compile"; sed -n '1,5p' "$TMP/gcc.log"; exit 1
  fi
  for spec in \
    "ok.gguf|$TEN_FIRST" \
    "forge_offset_wrap.gguf|$TEN_FIRST" \
    "forge_dim_huge.gguf|$TEN_LAST" \
    "forge_str_len.gguf|$TEN_FIRST" \
    "forge_arr_len.gguf|$TEN_FIRST" ; do
    file="${spec%%|*}"; ten="${spec##*|}"
    printf '  %-24s old[%s]\n  %-24s new[%s]\n' "$file" \
      "$(probe "$oldprobe" "$TMP/$file" "$ten")" "" "$(probe "$PROBE" "$TMP/$file" "$ten")"
  done
  # The offset-wrap case must show the old loader handing back KV bytes
  # ("qwen35" in hex) as the weights of token_embd.weight.
  o="$(probe "$oldprobe" "$TMP/forge_offset_wrap.gguf" "$TEN_FIRST")"
  if printf '%s' "$o" | grep -q "71 77 65 6e 33 35"; then
    echo "  OK    old loader returned KV bytes (ascii 'qwen35') as weights; new rejects it"
  else
    echo "  NOTE  old loader did not reproduce the garbage read: $o"; fail=1
  fi
fi

if [ "$fail" -eq 0 ]; then echo "check_hardening: PASS"; else echo "check_hardening: FAIL"; fi
exit "$fail"
