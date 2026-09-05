"""Exports the tagger to a compact binary artifact, ready for a JS port.

File contents:
  - JSON header (architecture, reference tokenizer, list of tensors)
  - BPE merges as pairs of uint16 integers (what a BPE encoder needs)
  - weights: each matrix quantized per row over `bits` bits (symmetric),
    bits packed tight, with one fp16 scale per row; vectors (biases, norms) in fp16

This is not a simulation: the bytes written are the ones the JS will read.
The quantization is the same as quantize.py (measured lossless at int4 and int3).

    python train/export.py runs/mot_d32_c2.pt --bits 4
    -> runs/mot_d32_c2.int4.bin
"""
import argparse
import io
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def merges_uint16(tokenizer_json):
    """The BPE merges, encoded as pairs of token ids (uint16 each).

    Also returns what a byte-level BPE encoder needs to know without the
    tokenizer.json: the id of each of the 256 bytes (`byte_ids`, in order
    0..255) and the id produced by merge k (`base` + k, checked here).
    """
    tok = json.load(io.open(tokenizer_json, encoding="utf-8"))
    vocab = tok["model"]["vocab"]
    base = len(tok.get("added_tokens", [])) + 256
    pairs = []
    for k, m in enumerate(tok["model"]["merges"]):
        a, b = m if isinstance(m, list) else m.split(" ", 1)
        assert vocab[a + b] == base + k, "merge %d: id %d expected %d" % (k, vocab[a + b], base + k)
        pairs.append((vocab[a], vocab[b]))
    # byte -> GPT-2 character table (bytes_to_unicode), then -> id
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for byte_val in range(256):
        if byte_val not in bs:
            bs.append(byte_val)
            cs.append(256 + n)
            n += 1
    b2u = dict(zip(bs, cs))
    byte_ids = [vocab[chr(b2u[o])] for o in range(256)]
    return np.array(pairs, dtype=np.uint16), byte_ids, base


def pack_bits(q, bits):
    """Signed integers over `bits` bits -> packed bytes (shifted to unsigned)."""
    u = (q + (1 << (bits - 1))).astype(np.uint32).flatten()
    n = u.size
    if bits == 8:
        return u.astype(np.uint8).tobytes()
    # generic packing as a bit stream
    total = np.zeros((n * bits + 7) // 8, dtype=np.uint8)
    pos = 0
    for v in u:
        for k in range(bits):
            if (v >> k) & 1:
                total[(pos + k) >> 3] |= 1 << ((pos + k) & 7)
        pos += bits
    return total.tobytes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--bits", type=int, default=4)
    p.add_argument("--tokenizer", default=os.path.join(HERE, "tokenizer_2048.json"))
    args = p.parse_args()

    state = torch.load(args.weights, map_location="cpu")
    meta = args.weights[:-3] + ".json"
    j = json.load(open(meta)) if os.path.exists(meta) else {}
    merges, byte_ids, base = merges_uint16(args.tokenizer)
    qmax = 2 ** (args.bits - 1) - 1

    tensors, blobs = [], []
    for name, t in state.items():
        if name.startswith("length_head") or name == "table_pos":
            continue    # no length head; the sinusoidal table is recomputed (0 bytes)
        a = t.numpy()
        if args.bits == 32:
            # reference precision: everything in float32, nothing quantized
            blob = a.astype(np.float32).tobytes()
            tensors.append({"name": name, "shape": list(a.shape), "type": "f32", "bytes": len(blob)})
        elif a.ndim >= 2 and a.size >= 64:
            sc = np.maximum(np.abs(a).max(axis=1, keepdims=True), 1e-8) / qmax
            q = np.clip(np.round(a / sc), -qmax, qmax).astype(np.int32)
            blob = pack_bits(q, args.bits) + sc.astype(np.float16).tobytes()
            tensors.append({"name": name, "shape": list(a.shape), "type": "q%d" % args.bits, "bytes": len(blob)})
        else:
            blob = a.astype(np.float16).tobytes()
            tensors.append({"name": name, "shape": list(a.shape), "type": "f16", "bytes": len(blob)})
        blobs.append(blob)

    # `format`: file format version. A reader branches on it; a model in
    # format 1 will stay readable by every future runtime. Then the file's
    # identity (a file must describe itself), the architecture, the tokenizer.
    header = {"format": 1, "name": j.get("name_public", "aparte-titler"), "version": j.get("version", "0.0.0"),
              "model": j.get("model", os.path.basename(args.weights)[:-3]), "languages": j.get("languages", []),
              "precision": "fp32" if args.bits == 32 else "int%d" % args.bits,
              "license": j.get("license", "MIT"), "url": j.get("url", "https://apartejs.dev/models/titler/"),
              # `heads` was hard-coded to 4 until 2026-09-04: weight shapes do not
              # depend on it, so a model trained with another head count exported
              # silently wrong. `decoding` documents the runtime's default decoder
              # (1.1); a file without it decodes with the same values. `seed` is
              # the training seed, 0 when it was not fixed.
              "dim": j.get("dim", 32), "layers": j.get("layers", 2), "heads": j.get("heads", 4),
              "seed": j.get("seed", 0),
              # `decoding.threshold` belongs to the MODEL, never a fixed 0.5.
              # Measured 2026-09-05 across 19 scopes: a fixed 0.5 cost up to 15
              # points on models calibrated low (Czech, Croatian, Lithuanian,
              # Hungarian) because the filter rejected every word and the
              # three-word floor turned the hybrid into a disguised budget of 3.
              # The rule is min(calibrated, 0.5): inside a top 6 the decoder only
              # has to drop clear negatives, never to be stricter than a half.
              # `pure_threshold` keeps the calibrated value for the plain
              # threshold decoder, which does rule on every word.
              "decoding": {"default": "hybrid",
                           "threshold": round(min(float(j.get("threshold", 0.5)), 0.5), 4),
                           "pure_threshold": round(float(j.get("threshold", 0.5)), 4),
                           "min_words": 3, "max_words": 6, "keep_all_up_to": 4},
              "positions": j.get("positions", "learned"), "dim_emb": j.get("dim_emb", 0),
              "vocab": base + int(merges.shape[0]), "n_merges": int(merges.shape[0]),
              "merges_base": base, "byte_ids": byte_ids, "pad": 0, "max_chars": 1200, "max_pos": 512,
              "bits": args.bits, "tensors": tensors}
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    output = args.weights[:-3] + (".fp32.bin" if args.bits == 32 else ".int%d.bin" % args.bits)
    with open(output, "wb") as fh:
        fh.write(b"LMAP")
        fh.write(struct.pack("<I", len(header_bytes)))
        fh.write(header_bytes)
        fh.write(struct.pack("<I", merges.size * 2))
        fh.write(merges.tobytes())
        for b in blobs:
            fh.write(b)

    size = os.path.getsize(output)
    weights_q = sum(t["bytes"] for t in tensors if t["type"].startswith("q"))
    weights_f = sum(t["bytes"] for t in tensors if t["type"] == "f16")
    print("%s : %s bytes (%.1f KB)" % (os.path.relpath(output, ROOT), format(size, ","), size / 1024))
    print("   header %d | merges %d (%d pairs) | weights q%d %d | f16 vectors %d"
          % (len(header_bytes) + 8, merges.size * 2 + 4, merges.shape[0], args.bits, weights_q, weights_f))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
