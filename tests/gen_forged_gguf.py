#!/usr/bin/env python3
"""Forge corrupt GGUF headers to test the loader hardening (audit findings M7/M8).

Every file below is the *valid* fake model from tests/gen_fake_qwen35.py with one
field rewritten, so the only reason a load can fail is the field under test:

  forge_offset_wrap.gguf  one tensor's offset = 2^64 - data_offset + 64, so the
                          old `doff + offset + bytes` wrapped to a small aligned
                          value inside the file: both geometry guards passed and
                          the later fseek read GGUF header bytes as weights
                          (loader.cpp, audit M7).
  forge_dim_huge.gguf     one tensor's ne[0] = INT64_MAX, so tensor_bytes() wraps
                          to 2^64-4: geometry could not see it and the following
                          vector::resize threw length_error (audit M7).
  forge_str_len.gguf      a KV string value length = 2^40: string::resize tried a
                          1 TiB allocation -> bad_alloc (audit M8).
  forge_arr_len.gguf      an array count = 2^40: vector<Value>::resize reserved
                          ~80 TB before reading the first element (audit M8).

The three M8/M7 size cases used to abort the process (uncaught bad_alloc /
length_error); the guards in gguf.cpp and loader.cpp turn them into a clean `err`.

Usage: tests/gen_forged_gguf.py <outdir>
"""
import importlib.util
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
U64 = (1 << 64) - 1
KV0 = 24  # first KV entry: magic(4) + version(4) + n_tensors(8) + n_kv(8)


def load_generator(outdir):
    """Import tests/gen_fake_qwen35.py so its KV/tensor tables can be reused.

    Importing it generates the normal (valid) variants into `outdir`.
    """
    sys.argv = [os.path.join(HERE, "gen_fake_qwen35.py"), outdir]
    spec = importlib.util.spec_from_file_location(
        "gen_fake_qwen35", os.path.join(HERE, "gen_fake_qwen35.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def find(buf, needle, what):
    i = buf.find(needle)
    if i < 0:
        raise SystemExit(f"gen_forged_gguf: {what}: pattern {needle!r} not found")
    return i


def patch(path, off, raw):
    with open(path, "r+b") as f:
        f.seek(off)
        f.write(raw)


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: gen_forged_gguf.py <outdir>")
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    g = load_generator(out)
    with open(os.path.join(out, "ok.gguf"), "rb") as f:
        valid = f.read()

    # Exact header layout of build(): 4+4+8+8, the KV block, then the tensor
    # table (len-prefixed name, ndims, dims, dtype, offset), padded to 32.
    tl = [("token_embd.weight", (16, 8)), ("output_norm.weight", (16,)),
          ("output.weight", (16, 8))]
    tl += [("blk.0." + n, d) for n, d in g.GDN]
    tl += [("blk.1." + n, d) for n, d in g.FULL + g.NEXTN]
    kv_bytes = sum(len(k) + len(v) for k, v in g.KV)
    tbl_bytes = sum(8 + len(n) + 4 + 8 * len(d) + 4 + 8 for n, d in tl)
    hdr = 24 + kv_bytes + tbl_bytes  # magic(4) version(4) n_tensors(8) n_kv(8)
    doff = (hdr + 31) // 32 * 32

    # 1) offset wrap. The patched tensor has to be the FIRST one: the bound of
    #    tensor i-1 is doff + offset[i], so a wrapped offset anywhere else makes
    #    the *predecessor's* guard fire first (measured: mid-table and last-tensor
    #    variants are rejected by the old code too, for the wrong reason). The
    #    first tensor has no predecessor, so with doff + offset wrapping to 64 its
    #    own checks pass - end = 64 + 512 = 576 is aligned and below the next
    #    tensor's start - and the read lands on file offset 64, inside the KV
    #    section, returning header bytes as weights.
    off_wrap = (U64 + 1) - doff + 64
    name = b"token_embd.weight"
    p = find(valid, name, "tensor token_embd.weight")
    off_field = p + len(name) + 4 + 8 * 2 + 4  # after name: ndims(4), dims(8x2), dtype(4)
    f1 = os.path.join(out, "forge_offset_wrap.gguf")
    with open(f1, "wb") as f:
        f.write(valid)
    patch(f1, off_field, struct.pack("<Q", off_wrap))

    # 2) ne[0] = INT64_MAX: tensor_bytes(F32) = (2^63-1)*4 mod 2^64 = 2^64-4
    name = b"output_norm.weight"
    p = find(valid, name, "tensor output_norm.weight")
    dim_field = p + len(name) + 4  # ndims(4), then the first dim
    f2 = os.path.join(out, "forge_dim_huge.gguf")
    with open(f2, "wb") as f:
        f.write(valid)
    patch(f2, dim_field, struct.pack("<q", (1 << 63) - 1))

    # 3) KV string length: the value-length field of KV[0] (a STR value).
    #    KV entries are (encoded key = len(8) + bytes, encoded value).
    k0, v0 = g.KV[0]
    assert valid[KV0:KV0 + 8] == struct.pack("<Q", len(k0) - 8), "KV[0] layout changed"
    str_len_field = KV0 + len(k0) + 4  # key, value type tag, then value length
    assert valid[str_len_field:str_len_field + 8] == struct.pack("<Q", len(v0) - 12), \
        "KV[0] value is not the STR we expect"
    f3 = os.path.join(out, "forge_str_len.gguf")
    with open(f3, "wb") as f:
        f.write(valid)
    patch(f3, str_len_field, struct.pack("<Q", 1 << 40))

    # 4) KV array count: the only ARR value is qwen35.rope.dimension_sections
    kv_off = KV0
    arr_count = None
    for k, v in g.KV:
        if v[0:4] == struct.pack("<I", 9):  # value type tag 9 = ARRAY
            arr_count = kv_off + len(k) + 4 + 4  # key, type tag, element type
            break
        kv_off += len(k) + len(v)
    assert arr_count is not None, "no array KV in the fake model"
    f4 = os.path.join(out, "forge_arr_len.gguf")
    with open(f4, "wb") as f:
        f.write(valid)
    patch(f4, arr_count, struct.pack("<Q", 1 << 40))

    print(f"gen_forged_gguf: doff={doff} offset_wrap={off_wrap}")
    print("  " + " ".join(os.path.basename(p) for p in (f1, f2, f3, f4)))


if __name__ == "__main__":
    main()
