"""Filters titles by agreement between two titlings of the same message.

The teacher does not emit a confidence score. Disagreement between two
titlings (two prompts on the same model, or two models) stands in for one —
measured on the gold set: where v1 and v2 agree (Jaccard >= 0.40), the label
scores 0.733; where they diverge (< 0.25), 0.526. Almost as discriminating as
a second model (v2/26B: 0.741 vs. 0.400), and free.

    python train/filter_agreement.py --threshold 0.4
    -> data/canonical/train_titles_v2_accord.jsonl  (v2 titles of the retained pairs)
"""
import argparse
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CANON = os.path.join(ROOT, "data", "canonical")
PUNCT = ".,;:!?\"'()[]{}<>*`“”‘’«»…"


def words(t):
    return {w.lower().strip(PUNCT) for w in (t or "").split() if w.strip(PUNCT)}


def jaccard(a, b):
    a, b = words(a), words(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def load(path):
    out = {}
    for l in io.open(path, encoding="utf-8"):
        r = json.loads(l)
        if r.get("title") and not r.get("refused"):
            out[r["id"]] = r
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a", default=os.path.join(CANON, "train_titles.jsonl"))
    p.add_argument("--b", default=os.path.join(CANON, "train_titles_v2.jsonl"))
    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--output", default=os.path.join(CANON, "train_titles_v2_accord.jsonl"))
    args = p.parse_args()

    A, B = load(args.a), load(args.b)
    common = [i for i in B if i in A]
    scores = {i: jaccard(A[i]["title"], B[i]["title"]) for i in common}
    kept = [i for i in common if scores[i] >= args.threshold]
    with io.open(args.output, "w", encoding="utf-8") as fh:
        for i in kept:
            r = dict(B[i])
            r["agreement"] = round(scores[i], 3)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    bands = [(0, 0.25), (0.25, 0.4), (0.4, 0.6), (0.6, 1.01)]
    print("common pairs: %s" % format(len(common), ","))
    for lo, hi in bands:
        n = sum(1 for i in common if lo <= scores[i] < hi)
        print("  agreement [%.2f, %.2f): %6s  (%.0f%%)" % (lo, hi, format(n, ","), 100 * n / len(common)))
    print("retained (>= %.2f): %s -> %s" % (args.threshold, format(len(kept), ","),
                                             os.path.relpath(args.output, ROOT)))


if __name__ == "__main__":
    main()
