import struct

def sval(b):
    return struct.pack("<Q", len(b)) + b
def strv(b):
    return struct.pack("<I", 8) + struct.pack("<Q", len(b)) + b
def i32(x):  return struct.pack("<I", 4) + struct.pack("<I", x)
def f32(x):  return struct.pack("<I", 6) + struct.pack("<f", x)
def arr_i32(vals):
    return (struct.pack("<I", 9) + struct.pack("<i", 5) +
            struct.pack("<Q", len(vals)) + b"".join(struct.pack("<i", v) for v in vals))

KV = [
    (sval(b"general.architecture"), strv(b"qwen35")),
    (sval(b"qwen35.block_count"), i32(2)),
    (sval(b"qwen35.full_attention_interval"), i32(4)),
    (sval(b"qwen35.nextn_predict_layers"), i32(1)),
    (sval(b"qwen35.context_length"), i32(1024)),
    (sval(b"qwen35.embedding_length"), i32(16)),
    (sval(b"qwen35.feed_forward_length"), i32(32)),
    (sval(b"qwen35.attention.head_count"), i32(2)),
    (sval(b"qwen35.attention.head_count_kv"), i32(1)),
    (sval(b"qwen35.attention.key_length"), i32(8)),
    (sval(b"qwen35.attention.value_length"), i32(8)),
    (sval(b"qwen35.ssm.conv_kernel"), i32(4)),
    (sval(b"qwen35.ssm.state_size"), i32(8)),
    (sval(b"qwen35.ssm.group_count"), i32(1)),
    (sval(b"qwen35.ssm.time_step_rank"), i32(8)),
    (sval(b"qwen35.ssm.inner_size"), i32(16)),
    (sval(b"qwen35.rope.dimension_count"), i32(16)),
    (sval(b"qwen35.rope.dimension_sections"), arr_i32([2,2,2,2])),
    (sval(b"qwen35.attention.layer_norm_rms_epsilon"), f32(1e-6)),
    (sval(b"qwen35.rope.freq_base"), f32(1e7)),
    (sval(b"tokenizer.ggml.bos_token_id"), i32(1)),
    (sval(b"tokenizer.ggml.eos_token_id"), i32(2)),
    (sval(b"tokenizer.ggml.padding_token_id"), i32(0)),
]

GDN = [("attn_gate.weight",(16,16)),("attn_norm.weight",(16,)),("attn_qkv.weight",(16,40)),
       ("ffn_down.weight",(32,16)),("ffn_gate.weight",(16,32)),("ffn_up.weight",(16,32)),
       ("post_attention_norm.weight",(16,)),("ssm_a",(8,)),("ssm_alpha.weight",(16,8)),
       ("ssm_beta.weight",(16,8)),("ssm_conv1d.weight",(4,40)),("ssm_dt.bias",(8,)),
       ("ssm_norm.weight",(8,)),("ssm_out.weight",(16,16))]
FULL = [("attn_k.weight",(16,8)),("attn_k_norm.weight",(8,)),("attn_norm.weight",(16,)),
        ("attn_output.weight",(16,16)),("attn_q.weight",(16,32)),("attn_q_norm.weight",(8,)),
        ("attn_v.weight",(16,8)),("ffn_down.weight",(32,16)),("ffn_gate.weight",(16,32)),
        ("ffn_up.weight",(16,32)),("post_attention_norm.weight",(16,))]
# eh_proj maps concat(hidden, embed) -> hidden: [2*emb, emb] = [32,16]
NEXTN = [("nextn.eh_proj.weight",(32,16)),("nextn.enorm.weight",(16,)),
         ("nextn.hnorm.weight",(16,)),("nextn.shared_head_norm.weight",(16,))]

def build(path, drop=(), bad_dim=None, extra=()):
    tens = [("token_embd.weight",(16,8)),("output_norm.weight",(16,)),("output.weight",(16,8))]
    for n,d in GDN:
        if n in drop: continue
        tens.append(("blk.0."+n, d))
    for n,d in FULL + NEXTN:
        tens.append(("blk.1."+n, d))
    for n,d in extra:
        tens.append((n, d))
    if bad_dim:
        tens = [(n, bad_dim.get(n, d)) for n,d in tens]
    off = 0; offs = []
    for n,d in tens:
        ne = 1
        for x in d: ne *= x
        offs.append((off, ne))
        off = ((off + ne*4) + 31)//32*32   # pad each tensor to 32
    body = bytearray()
    for o,ne in offs:
        body += struct.pack("<%df" % ne, *range(ne))
        body += b"\x00"*((32 - (ne*4)%32)%32)
    tbl = bytearray()
    for (n,d),(o,ne) in zip(tens, offs):
        e = struct.pack("<Q", len(n)) + n.encode()
        e += struct.pack("<I", len(d))
        for x in d: e += struct.pack("<q", x)
        e += struct.pack("<I", 0) + struct.pack("<Q", o)
        tbl += e
    hdr = struct.pack("<4s", b"GGUF") + struct.pack("<I", 3) + struct.pack("<QQ", len(tens), len(KV))
    hdr += b"".join(k+v for k,v in KV)
    hdr += tbl
    hdr += b"\x00"*((32 - len(hdr)%32)%32)
    open(path,"wb").write(hdr + body)

import sys
out = sys.argv[1] if len(sys.argv) > 1 else "."
import os
os.makedirs(out, exist_ok=True)
def p(name): return os.path.join(out, name)
build(p("ok.gguf"))
build(p("missing.gguf"), drop=("ssm_a",))
build(p("baddim.gguf"), bad_dim={"blk.0.ssm_out.weight": (8,16)})
build(p("extra.gguf"), extra=[("blk.0.foo.weight",(16,))])
print("generated in", out)
