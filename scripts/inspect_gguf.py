#!/usr/bin/env python3
"""Dump GGUF metadata relevant to rdna4-infer (arch, KVs, dtype histogram, per-layer layouts).

Usage: python3 scripts/inspect_gguf.py /path/model.gguf
Uses gguf-py from .ref/llama.cpp if installed package is absent.
"""
import re
import sys
from collections import Counter
from pathlib import Path

try:
    import gguf
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / ".ref" / "llama.cpp" / "gguf-py"))
    import gguf

SKIP_ARRAYS = ("tokenizer.ggml.tokens", "tokenizer.ggml.scores",
               "tokenizer.ggml.merges", "tokenizer.ggml.token_type")


def main(path: str) -> None:
    r = gguf.GGUFReader(path)
    print(f"file: {path}")
    for k, field in r.fields.items():
        if k in SKIP_ARRAYS:
            print(f"  {k} = <array len {len(field.contents())}>")
            continue
        s = str(field.contents())
        print(f"  {k} = {s[:200]}{'...' if len(s) > 200 else ''}")
    print(f"n_tensors: {len(r.tensors)}")
    print("dtypes:", dict(Counter(t.tensor_type.name for t in r.tensors)))
    layouts: dict[tuple, list] = {}
    for t in r.tensors:
        m = re.match(r"blk\.(\d+)\.(.*)", t.name)
        if m:
            layouts.setdefault(int(m.group(1)), []).append(m.group(2))
    by_layout: dict[tuple, int] = Counter(tuple(sorted(v)) for v in layouts.values())
    for lay, cnt in by_layout.items():
        ex = next(i for i, v in sorted(layouts.items()) if tuple(sorted(v)) == lay)
        print(f"  x{cnt} e.g. blk.{ex}: {list(lay)}")


if __name__ == "__main__":
    main(sys.argv[1])
