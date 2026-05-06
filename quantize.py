#!/usr/bin/env python3
"""
Output layout:
  - exp_lut       (512  * 4 bytes) exp(-i/64) LUT for softmax
  - silu_lut      (1024 * 4 bytes) silu(x) LUT for FFN gating
  - weight tensors (int32 Q16.16 or int8 + global int32 scale)
  - tokenizer     (per token: int32 score, null-terminated UTF-8 string)
"""

import os, struct
import numpy as np

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
SRC_MODEL  = os.path.join(MODELS_DIR, "stories260K.bin")
SRC_TOK    = os.path.join(MODELS_DIR, "tok512.bin")
DST        = os.path.join(MODELS_DIR, "stories260K_int.bin")

SCALE = 65536

SKIP_QUANT = {
    "token_embedding_table", "rms_att_weight", "rms_ffn_weight",
    "rms_final_weight", "freq_cis",
}

def pad_to_paragraph(fout):
    rem = fout.tell() % 16
    if rem:
        fout.write(b"\x00" * (16 - rem))

def load_w(f, n):
    return np.frombuffer(f.read(n * 4), dtype=np.float32).copy()

def make_exp_lut(n=512):
    return np.round(np.exp(-np.arange(n) / 64.0) * SCALE).astype(np.int32)

def make_silu_lut(n=1024):
    x = np.arange(n) / 64.0 - 8.0
    return np.round(x / (1.0 + np.exp(-x)) * SCALE).astype(np.int32)

def write_fixed(fout, w, name):
    flat  = w.flatten().astype(np.float64)
    fixed = np.round(flat * SCALE).astype(np.int32)
    offset = fout.tell()
    fout.write(fixed.tobytes())
    pad_to_paragraph(fout)
    print(f"  {name:<30}  i32  {fixed.size:>8}  "
          f"range=[{flat.min():.4f}, {flat.max():.4f}]  "
          f"@ 0x{0x20000 + offset:X}")

def write_q8(fout, w, name):
    flat = w.flatten().astype(np.float64)
    amax = max(np.max(np.abs(flat)), 1e-9)  # avoid div-by-zero
    q    = np.round(flat / amax * 127.0).clip(-127, 127).astype(np.int8)
    scale_q = np.round(amax / 127.0 * SCALE).astype(np.int32)
    offset = fout.tell()
    fout.write(q.tobytes())
    fout.write(struct.pack("<i", int(scale_q)))
    pad_to_paragraph(fout)
    err = float(np.max(np.abs(q.astype(np.float64) * amax / 127.0 - flat)))
    print(f"  {name:<30}  Q8   {flat.size:>8}  err={err:.5f}  "
          f"@ 0x{0x20000 + offset:X}")

def write_tensor(fout, w, name):
    (write_fixed if name in SKIP_QUANT else write_q8)(fout, w, name)

def load_tokenizer(path):
    tokens = []
    with open(path, "rb") as f:
        f.read(4)
        while chunk := f.read(8):
            if len(chunk) < 8:
                break
            score, slen = struct.unpack("<fi", chunk)
            tokens.append((score, f.read(slen).decode("utf-8", errors="replace")))
    return tokens

def write_tokenizer(fout, tokens):
    for score, s in tokens:
        fout.write(struct.pack("<i", int(round(score * SCALE))))
        fout.write(s.encode("utf-8") + b"\0")
    print(f"  tokenizer: {len(tokens)} tokens")

def main():
    with open(SRC_MODEL, "rb") as f:
        dim, hidden, nl, nh, nkv, vs, seq = struct.unpack("<7i", f.read(28))
    vocab = abs(vs)
    hd, kvd = dim // nh, nkv * (dim // nh)

    # Detect weight tying: if file is small enough, output projection = embedding
    weights_floats = (vocab*dim + nl*dim + nl*dim*(nh*hd) + nl*dim*kvd*2
                      + nl*(nh*hd)*dim + nl*dim + nl*hidden*dim*2
                      + nl*dim*hidden + dim + seq*(hd//2)*2)
    shared = os.path.getsize(SRC_MODEL) <= 28 + weights_floats * 4

    print(f"Model: dim={dim} hidden={hidden} layers={nl} heads={nh} "
          f"kv={nkv} vocab={vocab} seq={seq} shared={shared}")

    tokens = load_tokenizer(SRC_TOK)
    assert len(tokens) == vocab, f"Tokenizer size mismatch: {len(tokens)} != {vocab}"

    with open(SRC_MODEL, "rb") as fin, open(DST, "wb") as fout:
        fin.read(28)  # skip header

        print("Lookup tables:")
        for lut, label in [(make_exp_lut(512), "exp_lut"), (make_silu_lut(1024), "silu_lut")]:
            fout.write(lut.tobytes())
            pad_to_paragraph(fout)
            print(f"  {label}: {len(lut)} entries, {len(lut)*4} bytes")

        print("Tensors:")
        def blk(name, n): write_tensor(fout, load_w(fin, n), name)

        blk("token_embedding_table", vocab * dim)
        blk("rms_att_weight", nl * dim)

        # Fuse Q, K, V projections into one matrix for a single matmul call
        wq = load_w(fin, nl * dim * dim).reshape(nl, dim, dim)
        wk = load_w(fin, nl * dim * kvd).reshape(nl, kvd, dim)
        wv = load_w(fin, nl * dim * kvd).reshape(nl, kvd, dim)
        write_tensor(fout, np.concatenate([wq, wk, wv], axis=1).flatten(), "wqkv")

        blk("wo", nl * (nh * hd) * dim)
        blk("rms_ffn_weight", nl * dim)

        # Fuse gate and up projections for the FFN (SwiGLU)
        w1 = load_w(fin, nl * hidden * dim).reshape(nl, hidden, dim)
        blk("w2", nl * dim * hidden)
        w3 = load_w(fin, nl * hidden * dim).reshape(nl, hidden, dim)
        write_tensor(fout, np.concatenate([w1, w3], axis=1).flatten(), "w13")

        blk("rms_final_weight", dim)

        # Interleave cos/sin for RoPE so apply_rope can use sequential loads
        cos = load_w(fin, seq * (hd // 2))
        sin = load_w(fin, seq * (hd // 2))
        freq = np.empty(len(cos) + len(sin), dtype=np.float32)
        freq[0::2], freq[1::2] = cos, sin
        write_tensor(fout, freq, "freq_cis")

        if not shared:
            blk("wcls", vocab * dim)

        print("Tokenizer:")
        write_tokenizer(fout, tokens)

    size, orig = os.path.getsize(DST), os.path.getsize(SRC_MODEL)
    print(f"\nOriginal: {orig//1024}KB  Output: {size//1024}KB  "
          f"({100*size//orig}% of original)")

if __name__ == "__main__":
    main()