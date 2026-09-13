#!/usr/bin/env python3
"""Extract one token's node dump from a `-ub 1` eval-callback capture.

With `-ub 1` (ubatch size 1) llama.cpp evaluates one token per graph, so the
dump contains one full set of node entries per token, in order — i.e. the same
per-token computation our engine does. Every set starts at `model.input_embed`,
which makes the split unambiguous.

usage: scripts/extract_dump_block.py <dump.txt> <index, 1-based> [out.txt]

With no output path the extracted block goes to stdout. The chosen token's id
(as printed in the eval-callback log header) is reported on stderr.
"""
import re
import sys


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    src, index = sys.argv[1], int(sys.argv[2])
    dst = sys.argv[3] if len(sys.argv) > 3 else None

    with open(src, "r", errors="replace") as fh:
        lines = fh.readlines()

    starts = [
        i
        for i, line in enumerate(lines)
        if "common_debug_cb_eval:" in line and "model.input_embed = (f32)" in line
    ]
    if not starts:
        print("no `model.input_embed` node found: not a -ub 1 dump?", file=sys.stderr)
        return 1
    if index < 1 or index > len(starts):
        print(f"index {index} out of range (1..{len(starts)})", file=sys.stderr)
        return 1

    begin = starts[index - 1]
    end = starts[index] if index < len(starts) else len(lines)
    block = lines[begin:end]

    # token id of this block: the last "I    <id>" line before the block start
    tok = None
    for line in reversed(lines[:begin]):
        m = re.match(r"^[0-9.]+ [A-Z] +(\d+)\s*$", line)
        if m:
            tok = int(m.group(1))
            break
    print(f"block {index}/{len(starts)}: lines {begin + 1}..{end}, token={tok}", file=sys.stderr)

    text = "".join(block)
    if dst:
        with open(dst, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
