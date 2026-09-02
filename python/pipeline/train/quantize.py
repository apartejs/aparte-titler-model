"""Simulated post-training quantization: each weight is rounded onto an int8
or int4 grid (scale per output row, symmetric), then cast back to float.
This is exactly what a JS port would do if it stored the weights as integers
and dequantized them on the fly. Embeddings are handled the same way (scale
per row = per token).

    python train/quantize.py runs/mot_d32_c2.pt --bits 8 4
"""
import argparse
import io
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval"))


def quantize_tensor(w, bits):
    """Symmetric rounding onto 2^bits levels, one scale per row."""
    if w.dim() < 2 or w.numel() < 64:          # biases, norms: kept in float
        return w.clone(), 0
    qmax = 2 ** (bits - 1) - 1
    scale = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(w / scale).clamp(-qmax, qmax)
    return q * scale, w.numel()


def quantize_state(state, bits):
    out, n_q, n_tot = {}, 0, 0
    for k, v in state.items():
        if v.dtype.is_floating_point:
            out[k], nq = quantize_tensor(v, bits)
            n_q += nq
        else:
            out[k] = v.clone()
        n_tot += v.numel()
    return out, n_q, n_tot


def size_kb(state, bits, n_q):
    """Bytes: weights quantized to `bits`, the rest in fp16, plus one fp16 scale per row."""
    rows = sum(v.shape[0] for v in state.values() if v.dim() >= 2 and v.numel() >= 64)
    rest = sum(v.numel() for v in state.values()) - n_q
    return (n_q * bits / 8 + rest * 2 + rows * 2) / 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--bits", type=int, nargs="+", default=[8, 4])
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--layers", type=int, default=2)
    args = p.parse_args()

    import score as S
    from labels import word_ids_per_token
    from model import Tagger
    from tokenizers import Tokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(os.environ.get("TITLER_TOKENIZER", os.path.join(HERE, "tokenizer_2048.json")))
    gold = S.load_gold(tok)
    cons = S.load_consensus()
    state = torch.load(args.weights, map_location="cpu")
    meta = args.weights[:-3] + ".json"
    threshold = json.load(open(meta))["threshold"] if os.path.exists(meta) else 0.5

    opts = {}
    if os.path.exists(meta):
        j = json.load(open(meta))
        opts = {k: j[k] for k in ("positions", "dim_emb") if k in j}

    def model_from(state_q):
        m = Tagger(tok.get_vocab_size(), args.dim, 4, args.layers, **opts).to(dev)
        m.load_state_dict(state_q, strict=False)   # weights with no length head: fine
        m.eval()
        return m

    def predictor(m, budget=None):
        def f(g):
            with torch.no_grad():
                X = torch.tensor([g["ids"]], device=dev)
                W = torch.tensor([word_ids_per_token(g["text"], g["offsets"])], device=dev)
                pr = torch.sigmoid(m(X, torch.ones_like(X, dtype=torch.bool), W))[0].cpu().tolist()
            if budget:
                words = [gr for gr in S.word_groups(g["text"], g["offsets"])
                        if any(c.isalnum() for c in g["text"][g["offsets"][gr[0]][0]:g["offsets"][gr[-1]][1]])]
                top = sorted(words, key=lambda w: pr[w[0]], reverse=True)[:budget]
                kept = {i for w in top for i in w}
                return [1 if i in kept else 0 for i in range(len(g["ids"]))]
            return [1 if x >= threshold else 0 for x in pr]
        return f

    def measure(name, m):
        for label, budget in (("threshold", None), ("budget 6", 6)):
            f = predictor(m, budget)
            empty = sum(1 for g in gold if sum(f(g)) == 0)
            fm = S.evaluate_multi("  %-6s %-9s" % (name, label), f, gold, cons)
            print("           empty titles: %d" % empty)

    n_tot = sum(v.numel() for v in state.values())
    print("%s : %s parameters | fp32 %.0f KB | fp16 %.0f KB" % (os.path.basename(args.weights), format(n_tot, ","), n_tot * 4 / 1024, n_tot * 2 / 1024))
    measure("fp32", model_from(state))
    for bits in args.bits:
        state_q, n_q, _ = quantize_state(state, bits)
        print("int%d : %s weights quantized out of %s | ~%.0f KB with scales" % (bits, format(n_q, ","), format(n_tot, ","), size_kb(state, bits, n_q)))
        measure("int%d" % bits, model_from(state_q))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
