"""Loading, the train/validation split, and batches grouped by length.

Discipline: the gold set (`eval/gold.jsonl`) is the final exam. It is NEVER
used to choose a threshold, a model size, or a number of epochs. Everything
that gets tuned gets tuned on `val`, split off here from the training data.

Batches are grouped by neighboring length: median 46 tokens, p95 421. Padding
everything to 512 would spend ~90% of the compute on padding.
"""
import io
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LABELED = os.environ.get("TITLER_LABELED", os.path.join(ROOT, "data", "canonical", "train_labeled.jsonl"))
PAD = 0
SEED = 20260902


def load(path=LABELED, val_fraction=0.05, limit=0):
    """Examples (ids, labels, words). `words` = word number per token, so that
    the model can decide per word; absent from older files, in which case None."""
    examples = []
    for l in io.open(path, encoding="utf-8"):
        r = json.loads(l)
        if not r["ids"] or sum(r["labels"]) == 0:
            continue
        examples.append((r["ids"], r["labels"], r.get("words")))
        if limit and len(examples) >= limit:
            break
    rng = random.Random(SEED)
    rng.shuffle(examples)
    n_val = int(len(examples) * val_fraction)
    return examples[n_val:], examples[:n_val]


def batches(examples, token_budget=16384, shuffle=True, seed=0):
    """Batches at a constant token budget, grouped by neighboring length.

    We sort by length, cut into batches whose cost (size x max length) fits
    within the budget, then shuffle the order of the batches. The residual
    padding is small because the examples in a given batch have a similar
    length.
    """
    by_length = sorted(examples, key=lambda e: len(e[0]))
    out, current = [], []
    for ex in by_length:
        n = len(current) + 1
        lmax = max(len(current[0][0]) if current else 0, len(ex[0]))
        if current and n * lmax > token_budget:
            out.append(current)
            current = []
        current.append(ex)
    if current:
        out.append(current)
    if shuffle:
        random.Random(seed).shuffle(out)
    return out


def pad_batch(batch):
    """Rectangular (ids, labels, mask, words). The mask is worth 0 on padding;
    `words` is worth -1 on padding, and -1 everywhere if the file did not carry
    word numbers (the model then falls back to the token)."""
    # length rounded up to a multiple of 32: otherwise every batch length (1..512)
    # produces attention tensors of a unique size, which the GPU allocator caches
    # one by one — 57 GB of RAM measured on 142,000 examples. With 16 possible
    # sizes, the cache stays bounded.
    lmax = max(len(ex[0]) for ex in batch)
    lmax = ((lmax + 31) // 32) * 32
    X, Y, M, W = [], [], [], []
    for ex in batch:
        ids, labels = ex[0], ex[1]
        words = ex[2] if len(ex) > 2 and ex[2] is not None else [-1] * len(ids)
        k = lmax - len(ids)
        X.append(list(ids) + [PAD] * k)
        Y.append(list(labels) + [0] * k)
        M.append([1] * len(ids) + [0] * k)
        W.append(list(words) + [-1] * k)
    return X, Y, M, W


def stats(name, examples):
    L = sorted(len(ex[0]) for ex in examples)
    pos = sum(sum(ex[1]) for ex in examples)
    tot = sum(L)
    print("  %-5s %s examples | tokens median %d p95 %d | %.1f%% to keep"
          % (name, format(len(examples), ","), L[len(L) // 2],
             L[min(len(L) - 1, int(0.95 * len(L)))], 100 * pos / tot))


if __name__ == "__main__":
    tr, va = load()
    print("split:")
    stats("train", tr)
    stats("val", va)
    b = batches(tr)
    padded = sum(len(l) * max(len(ex[0]) for ex in l) for l in b)
    useful = sum(len(ex[0]) for l in b for ex in l)
    print("  %s batches | padding: %.0f%% of the compute is useful"
          % (format(len(b), ","), 100 * useful / padded))
    b512 = [tr[i:i + 32] for i in range(0, len(tr), 32)]
    useful512 = sum(len(ex[0]) for l in b512 for ex in l)
    print("  (for comparison, batches of 32 padded to 512: %.0f%% useful)"
          % (100 * useful512 / (len(tr) * 512)))
