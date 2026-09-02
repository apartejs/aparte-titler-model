"""Inference from a trained checkpoint (torch): one message -> one title.

The model decides PER WORD (the trained mean of a word's token logits) and saw
short messages during training: no word-snapping, no "short message" rule.
What remains is a decoding, as for any model:
  - threshold: keep the words whose probability exceeds `threshold` (the model alone)
  - budget:    keep the `budget` best-scored words (a guaranteed length; the
               retained decoding is a budget of 6 — the length of a title is not
               predictable from the message)

    python train/titler.py --weights runs/fr_fr_2048.pt --tokenizer tokenizer_fr_2048.json "Peux-tu m'expliquer la photosynthèse ?"
"""
import argparse
import io
import json
import os
import re
import sys

import torch
from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval"))
from labels import word_ids_per_token, recompose, PUNCT  # noqa: E402
from model import Tagger  # noqa: E402
from score import word_groups  # noqa: E402

WORD = re.compile(r"[^\W_]+", re.UNICODE)   # letters and digits, every script
MAX_CHARS = 1200


class Titler:
    def __init__(self, tokenizer, weights, dim=32, heads=4, layers=2, threshold=None, device=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = Tokenizer.from_file(tokenizer)
        # the architecture (dim, layers, positions, factorized embeddings) is in
        # the json written next to the weights by the trainer
        meta_path = weights[:-3] + ".json"
        meta = json.load(io.open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
        self.model = Tagger(self.tok.get_vocab_size(), meta.get("dim", dim), heads, meta.get("layers", layers),
                            positions=meta.get("positions", "learned"), dim_emb=meta.get("dim_emb", 0)).to(self.dev)
        state = torch.load(weights, map_location=self.dev)
        # a length head is only usable if it was TRAINED (head_k > 0 in the json)
        self.has_length_head = float(meta.get("head_k", 0)) > 0
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.threshold = threshold if threshold is not None else meta.get("threshold", 0.5)

    def title(self, message, budget=None):
        """(title, decoding used). `budget`: None or "k" = the retained decoding
        (a learned length if the head was trained, else 6), an int = fixed budget,
        "threshold" = probability >= threshold."""
        text = message.strip()[:MAX_CHARS]
        enc = self.tok.encode(text)
        ids, offsets = enc.ids, enc.offsets
        if not ids:
            return "", "empty"
        with torch.no_grad():
            X = torch.tensor([ids], device=self.dev)
            W = torch.tensor([word_ids_per_token(text, offsets)], device=self.dev)
            z, k_logits = self.model(X, torch.ones_like(X, dtype=torch.bool), W, return_k=True)
            pr = torch.sigmoid(z)[0].cpu().tolist()
            k_pred = int(k_logits[0].argmax().item()) + 1
        if budget == "threshold":
            keep = [1 if x >= self.threshold else 0 for x in pr]
            mode = "threshold"
        else:
            if budget in (None, "k") and not self.has_length_head:
                budget = 6
            k = k_pred if budget in (None, "k") else int(budget)
            groups = [g for g in word_groups(text, offsets) if WORD.search(text[offsets[g[0]][0]:offsets[g[-1]][1]])]
            top = sorted(groups, key=lambda g: pr[g[0]], reverse=True)[:k]
            mode = "learned k = %d" % k if budget in (None, "k") else "budget %d" % k
            # one span per kept word, in text order: punctuation between two kept words is never carried over
            spans = sorted((offsets[g[0]][0], offsets[g[-1]][1]) for g in top)
            return " ".join(m for m in (text[a:b].strip().strip(PUNCT) for a, b in spans) if m), mode
        return recompose(text, offsets, keep), mode


def main():
    p = argparse.ArgumentParser()
    p.add_argument("message", nargs="?", default="")
    p.add_argument("--weights", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--decoding", default="k", help="k = the retained decoding (default); threshold; or an integer (fixed budget)")
    args = p.parse_args()
    t = Titler(tokenizer=args.tokenizer, weights=args.weights)
    budget = args.decoding if args.decoding in ("k", "threshold") else int(args.decoding)
    if args.message:
        title, mode = t.title(args.message, budget=budget)
        print("%s   (%s)" % (title, mode))
    else:
        for line in sys.stdin:
            print(t.title(line, budget=budget)[0])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
