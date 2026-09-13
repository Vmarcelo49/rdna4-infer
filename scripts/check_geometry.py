#!/usr/bin/env python3
"""Empirical validation of the DType block-size table against a real UD GGUF.

For every tensor: size = ceil(n_elem / block_elems) * block_bytes must
(a) fit in the span until the next tensor offset (file end for the last one),
(b) leave (offset + size) aligned to general.alignment,
(c) for the last tensor, end exactly at the end of the file.
"""
import struct
import sys

# (ggml_type id, block_elems, block_bytes)
TABLE = {
    0:  (1, 4),     # F32
    8:  (32, 34),   # Q8_0
    10: (256, 84),  # Q2_K
    11: (256, 110), # Q3_K
    12: (256, 144), # Q4_K
    13: (256, 176), # Q5_K
    14: (256, 210), # Q6_K
    16: (256, 66),  # IQ2_XXS
    17: (256, 74),  # IQ2_XS
    18: (256, 98),  # IQ3_XXS
    19: (256, 50),  # IQ1_S
    20: (32, 18),   # IQ4_NL
    21: (256, 110), # IQ3_S
    22: (256, 82),  # IQ2_S
    23: (256, 136), # IQ4_XS
}


VALUE_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read_str(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return f.read(n).decode()


def skip_value(f, t):
    if t in VALUE_SIZE:
        f.seek(VALUE_SIZE[t], 1)
    elif t == 8:
        f.seek(struct.unpack("<Q", f.read(8))[0], 1)
    elif t == 9:
        at = struct.unpack("<i", f.read(4))[0]
        n = struct.unpack("<Q", f.read(8))[0]
        for _ in range(n):
            skip_value(f, at)
    else:
        raise RuntimeError(f"unknown KV value type {t}")


def main():
    path = sys.argv[1]
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF"
        version, n_tensors, n_kv = struct.unpack("<IQQ", f.read(20))
        assert version == 3
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            t = struct.unpack("<i", f.read(4))[0]
            if key == "general.alignment":
                v = struct.unpack("<I", f.read(4))[0]
                alignment = v
            else:
                skip_value(f, t)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}q", f.read(8 * nd))
            dtype = struct.unpack("<I", f.read(4))[0]
            offset = struct.unpack("<Q", f.read(8))[0]
            tensors.append((name, dims, dtype, offset))
        # tensor offsets are relative to the data section start, which is the
        # header end padded to alignment (ggml gguf reader: GGML_PAD(tell, align))
        data_offset = (f.tell() + alignment - 1) // alignment * alignment
        file_size = f.seek(0, 2)
    errors = 0
    total = 0
    for i, (name, dims, dtype, off) in enumerate(tensors):
        n = 1
        for d in dims:
            n *= d
        be, bb = TABLE.get(dtype, (None, None))
        if be is None:
            print(f"UNMAPPED dtype {dtype}: {name}")
            errors += 1
            continue
        size = -((-n // be)) * bb  # ceil
        total += size
        aoff = data_offset + off  # file-absolute
        end = aoff + size
        nxt = (data_offset + tensors[i + 1][3]) if i + 1 < len(tensors) else file_size
        pad = nxt - end
        if end > nxt:
            print(f"OVERFLOW {name}: size {size} exceeds span (pad {pad})")
            errors += 1
        if end % alignment != 0:
            print(f"UNALIGNED {name}: offset+size {end} % {alignment} != 0")
            errors += 1
        if i + 1 == len(tensors) and end != file_size:
            print(f"LAST-MISMATCH: end {end} != file_size {file_size} (diff {file_size - end})")
            errors += 1
        if pad > 1024:
            print(f"BIG-PAD {name}: pad {pad}")
            errors += 1
    print(f"alignment={alignment} tensors={len(tensors)} total_data={total} "
          f"file={file_size} header+pad={file_size - total}")
    print("OK" if errors == 0 else f"{errors} ERRORS")


if __name__ == "__main__":
    main()
