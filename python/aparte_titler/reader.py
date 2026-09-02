"""Reference implementation of the aparte-titler model format, in numpy.

Everything a runtime has to do, without torch or the `tokenizers` library —
the executable specification that the JavaScript runtime is tested against:

  1. read the `.bin`: JSON header, BPE merges (uint16 pairs), tensors
     (integers packed on `bits` bits + one fp16 scale per row; fp16 vectors)
  2. tokenize with byte-level BPE: GPT-2 pre-tokenization (regex), bytes -> ids
     (the `byte_ids` table of the header), merges by rank, merge k gives id base + k
  3. the transformer in float32, one decision per word (mean of its token logits)
  4. decode: the 6 best-scored words, recomposed in message order

    python reader.py titler-v1-latin-int3.bin "Can you explain how photosynthesis works?"
    python reader.py titler-v1-latin-int3.bin --check gold.jsonl        # tokens vs HF tokenizers
    python reader.py titler-v1-latin-int3.bin --references a.jsonl b.jsonl out.jsonl
"""
import io
import json
import math
import os
import re
import struct
import sys

import numpy as np

PUNCT = ".,;:!?\"'()[]{}<>*`“”‘’«»…-–—"   # stripped at the edges of a word only
WORD = re.compile(r"[^\W_]+")
# GPT-2's split (HF ByteLevel, use_regex=True):
#   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
# without the `regex` module: \p{L} -> [^\W\d_], \p{N} -> \d, and the rest
# (punctuation, symbols, underscore) -> any non-blank that is neither
SPLIT = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?[^\W\d_]+| ?\d+| ?(?:(?![^\W\d_]|\d)\S)+|\s+(?!\S)|\s+")


# --------------------------------------------------------------- reading

def read_model(path):
    """(header, merges as an (n, 2) uint16 array, {tensor name: float32 array})."""
    with open(path, "rb") as fh:
        assert fh.read(4) == b"LMAP", "not an aparte-titler model file"
        n = struct.unpack("<I", fh.read(4))[0]
        header = json.loads(fh.read(n).decode("utf-8"))
        assert header.get("format", 1) == 1, "format %s is not supported" % header.get("format")
        n = struct.unpack("<I", fh.read(4))[0]
        merges = np.frombuffer(fh.read(n), dtype=np.uint16).reshape(-1, 2)
        weights = {}
        for t in header["tensors"]:
            blob = fh.read(t["bytes"])
            shape = tuple(t["shape"])
            if t["type"] == "f32":
                weights[t["name"]] = np.frombuffer(blob, dtype=np.float32).reshape(shape)
            elif t["type"] == "f16":
                weights[t["name"]] = np.frombuffer(blob, dtype=np.float16).astype(np.float32).reshape(shape)
            else:
                bits = int(t["type"][1:])
                rows = shape[0]
                count = int(np.prod(shape))
                packed = (count * bits + 7) // 8
                q = unpack_bits(np.frombuffer(blob[:packed], dtype=np.uint8), count, bits)
                scale = np.frombuffer(blob[packed:], dtype=np.float16).astype(np.float32).reshape(rows, 1)
                weights[t["name"]] = (q.reshape(shape) * scale).astype(np.float32)
    return header, merges, weights


def unpack_bits(data, n, bits):
    """Bit stream, least significant bit first: n signed integers of `bits` bits."""
    if bits == 8:
        return data[:n].astype(np.int32) - 128
    b = np.unpackbits(data, bitorder="little")[:n * bits].reshape(n, bits)
    u = (b.astype(np.int32) << np.arange(bits)).sum(axis=1)
    return u - (1 << (bits - 1))


# ------------------------------------------------------------- tokenizer

class Tokenizer:
    def __init__(self, header, merges):
        self.byte_ids = header["byte_ids"]         # token id of each of the 256 bytes
        self.base = header["merges_base"]         # merge k produces token id base + k
        self.rank = {(int(a), int(b)): k for k, (a, b) in enumerate(merges)}

    def encode(self, text):
        """Token ids and (start, end) character offsets, as HF with trim_offsets=False."""
        ids, offsets = [], []
        for m in SPLIT.finditer(text):
            start = m.start()
            char_of_byte, tokens = [], []
            for i, ch in enumerate(m.group()):
                for b in ch.encode("utf-8"):
                    tokens.append((self.byte_ids[b], 1))
                    char_of_byte.append(i)
            while len(tokens) > 1:                 # merge the lowest-ranked pair first
                best, at = None, -1
                for j in range(len(tokens) - 1):
                    r = self.rank.get((tokens[j][0], tokens[j + 1][0]))
                    if r is not None and (best is None or r < best):
                        best, at = r, j
                if best is None:
                    break
                tokens[at:at + 2] = [(self.base + best, tokens[at][1] + tokens[at + 1][1])]
            pos = 0
            for tid, n in tokens:
                ids.append(tid)
                offsets.append((start + char_of_byte[pos], start + char_of_byte[pos + n - 1] + 1))
                pos += n
        return ids, offsets


# ----------------------------------------------------------------- model

def layer_norm(x, g, b, eps=1e-5):
    m = x.mean(axis=-1, keepdims=True)
    v = ((x - m) ** 2).mean(axis=-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


class Model:
    def __init__(self, header, weights):
        self.h, self.w = header, weights
        self.D, self.H = header["dim"], header["heads"]

    def logits(self, ids):
        """One logit per token: positive = keep."""
        w, D, H = self.w, self.D, self.H
        L = len(ids)
        e = w["emb_tok.weight"][ids]                                    # (L, dim_emb) or (L, D)
        if self.h["dim_emb"]:
            e = e @ w["proj_emb.weight"].T                              # factorized embeddings
        pos = np.minimum(np.arange(L), self.h["max_pos"] - 1)
        if self.h["positions"] == "sinus":
            freq = np.exp(np.arange(0, D, 2) * (-math.log(10000.0) / D))
            table = np.zeros((L, D), dtype=np.float32)
            table[:, 0::2] = np.sin(pos[:, None] * freq)
            table[:, 1::2] = np.cos(pos[:, None] * freq)
            x = e + table
        else:
            x = e + w["emb_pos.weight"][pos]
        Dh = D // H
        for c in range(self.h["layers"]):
            k = "blocks.%d." % c
            y = layer_norm(x, w[k + "n1.weight"], w[k + "n1.bias"])
            qkv = y @ w[k + "att.qkv.weight"].T + w[k + "att.qkv.bias"]    # (L, 3D)
            q, kk, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
            out = np.zeros((L, D), dtype=np.float32)
            for head in range(H):                                       # every token sees every token
                sl = slice(head * Dh, (head + 1) * Dh)
                s = (q[:, sl] @ kk[:, sl].T) / math.sqrt(Dh)
                out[:, sl] = softmax(s) @ v[:, sl]
            x = x + out @ w[k + "att.output.weight"].T + w[k + "att.output.bias"]
            y = layer_norm(x, w[k + "n2.weight"], w[k + "n2.bias"])
            f = gelu(y @ w[k + "ff.net.0.weight"].T + w[k + "ff.net.0.bias"])
            x = x + f @ w[k + "ff.net.2.weight"].T + w[k + "ff.net.2.bias"]
        xn = layer_norm(x, w["norm.weight"], w["norm.bias"])
        return (xn @ w["head.weight"].T + w["head.bias"]).reshape(-1)  # (L,)


# -------------------------------------------------------------- decoding

JOINERS = {"-", "'", "’"}


def word_ids(text, offsets):
    """A new word starts on a token whose first character is a space or not a
    letter/digit — except a hyphen or an apostrophe glued between two letters
    ("Pourrais-tu", "l'école"), which keeps the word whole: a title must never
    start with a fragment like "-tu"."""
    out, num, prev_end = [], -1, -1
    for i, (a, b) in enumerate(offsets):
        first = text[a:b][:1]
        starts = i == 0 or first.isspace() or not first.isalnum()
        if starts and i > 0 and first in JOINERS and a == prev_end and a > 0 and text[a - 1].isalnum()                 and a + 1 < len(text) and text[a + 1].isalnum():
            starts = False
        if starts:
            num += 1
        out.append(num)
        prev_end = b
    return out


def recompose(text, offsets, keep):
    """Kept tokens, merged into character intervals, sliced from the text."""
    spans = []
    for (a, b), y in zip(offsets, keep):
        if not y or b <= a:
            continue
        if spans and a <= spans[-1][1]:
            spans[-1][1] = max(spans[-1][1], b)
        else:
            spans.append([a, b])
    return " ".join(m for m in (text[a:b].strip().strip(PUNCT) for a, b in spans) if m)


class Titler:
    def __init__(self, path):
        self.header, merges, weights = read_model(path)
        self.tokenizer = Tokenizer(self.header, merges)
        self.model = Model(self.header, weights)

    def title(self, message, budget=6):
        """The `budget` best-scored words, in message order."""
        text = message.strip()[:self.header["max_chars"]]
        ids, offsets = self.tokenizer.encode(text)
        if not ids:
            return ""
        z = self.model.logits(ids)
        groups = {}
        for i, wid in enumerate(word_ids(text, offsets)):
            groups.setdefault(wid, []).append(i)
        score = {wid: 1.0 / (1.0 + math.exp(-float(np.mean(z[g])))) for wid, g in groups.items()}
        candidates = [(wid, g) for wid, g in groups.items() if WORD.search(text[offsets[g[0]][0]:offsets[g[-1]][1]])]
        top = sorted(candidates, key=lambda c: -score[c[0]])[:budget]   # stable: ties keep text order
        # one span per kept word, in text order: punctuation between two kept words is never carried over
        spans = sorted((offsets[g[0]][0], offsets[g[-1]][1]) for _, g in top)
        return " ".join(m for m in (text[a:b].strip().strip(PUNCT) for a, b in spans) if m)


# ------------------------------------------------------------ utilities

def check(path, gold_file, tokenizer_json=None):
    """Tokens and offsets against the HF `tokenizers` library (needs the tokenizer.json)."""
    from tokenizers import Tokenizer as HFTokenizer
    t = Titler(path)
    hf = HFTokenizer.from_file(tokenizer_json)
    n = same_ids = same_offsets = 0
    for line in io.open(gold_file, encoding="utf-8"):
        text = json.loads(line)["text"].strip()[:t.header["max_chars"]]
        ids, offsets = t.tokenizer.encode(text)
        enc = hf.encode(text)
        n += 1
        same_ids += ids == enc.ids
        same_offsets += [tuple(o) for o in offsets] == [tuple(o) for o in enc.offsets]
    print("%d messages: identical tokens %d, identical offsets %d" % (n, same_ids, same_offsets))


def references(path, inputs, output):
    """Reference titles for the JavaScript equality test: {text, title} per line."""
    t = Titler(path)
    n = 0
    with io.open(output, "w", encoding="utf-8") as out:
        for f in inputs:
            for line in io.open(f, encoding="utf-8"):
                text = json.loads(line)["text"]
                out.write(json.dumps({"text": text, "title": t.title(text)}, ensure_ascii=False) + "\n")
                n += 1
    print("%d references -> %s" % (n, output))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if len(sys.argv) > 2 and sys.argv[2] == "--check":
        check(sys.argv[1], sys.argv[3], sys.argv[4] if len(sys.argv) > 4 else None)
    elif len(sys.argv) > 2 and sys.argv[2] == "--references":
        references(sys.argv[1], sys.argv[3:-1], sys.argv[-1])
    else:
        print(Titler(sys.argv[1]).title(sys.argv[2]))
