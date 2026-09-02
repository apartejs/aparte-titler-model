"""The project's reference score: token-level precision / recall / F1, on the gold set.

Why not accuracy: only 8% of tokens are to be kept. A model that answers
"discard" everywhere gets 92% accuracy and is useless. So we measure only
the "keep" class.

Two levels, deliberately:
  - TOKEN : does the model keep the right tokens? this is what the loss optimizes.
  - WORD  : after recomposition, do the words of the produced title cover
            those of the reference title? this is what a human perceives.

A `predictor` is a function text -> list of 0/1 labels, one per token.

    python eval/score.py            # evaluates the baselines
"""
import io
import json
import math
import os
import re
import sys
from collections import Counter

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))
from labels import align, label_tokens, recompose, title_words  # noqa: E402

TOKENIZER = os.environ.get("TITLER_TOKENIZER", os.path.join(ROOT, "train", "tokenizer.json"))
GOLD = os.environ.get("TITLER_GOLD", os.path.join(HERE, "gold.jsonl"))
MESSAGES = os.path.join(ROOT, "data", "canonical", "train_messages.jsonl")
MAX_TOKENS = 512


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def load_gold(tok):
    """Each gold entry, with its reference per-token labels."""
    out = []
    for l in io.open(GOLD, encoding="utf-8"):
        g = json.loads(l)
        if g.get("impossible"):
            continue
        spans = [tuple(sp) for sp in g["spans"]]
        ids, labels, offsets = label_tokens(tok, g["text"], spans, MAX_TOKENS)
        if sum(labels) == 0:
            continue
        out.append({"k": g["k"], "text": g["text"], "title": g["title"],
                    "ids": ids, "labels": labels, "offsets": offsets,
                    "bucket": g["bucket"]})
    return out


def evaluate(name, predictor, gold, detail=False):
    tp = fp = fn = 0
    mtp = mfp = mfn = 0
    by_bucket = {}
    examples = []
    for g in gold:
        pred = predictor(g)
        for y, p in zip(g["labels"], pred):
            if p and y:
                tp += 1
            elif p and not y:
                fp += 1
            elif y and not p:
                fn += 1
        # word level
        expected = Counter(m.lower() for m in title_words(g["title"]))
        obtained = Counter(m.lower() for m in title_words(recompose(g["text"], g["offsets"], pred)))
        inter = sum((expected & obtained).values())
        mtp += inter
        mfp += sum(obtained.values()) - inter
        mfn += sum(expected.values()) - inter
        b = by_bucket.setdefault(g["bucket"], [0, 0, 0])
        for y, p in zip(g["labels"], pred):
            if p and y:
                b[0] += 1
            elif p and not y:
                b[1] += 1
            elif y and not p:
                b[2] += 1
        if detail and len(examples) < 6:
            examples.append((g["title"], recompose(g["text"], g["offsets"], pred)))

    p, r, f = prf(tp, fp, fn)
    mp, mr, mf = prf(mtp, mfp, mfn)
    print("%-34s token P %.2f R %.2f F1 %.2f  |  word P %.2f R %.2f F1 %.2f"
          % (name, p, r, f, mp, mr, mf))
    if detail:
        for tr, (a, b_, c) in sorted(by_bucket.items()):
            print("      %-10s F1 %.2f" % (tr, prf(a, b_, c)[2]))
        for ref, obt in examples:
            print("      gold `%s`  ->  `%s`" % (ref, obt))
    return f


# ------------------------------------------ several references (annotators)

# The consensus labels are PER TOKEN, so they are tied to the tokenizer that
# produced them. A different tokenizer needs its own file (consensus.py --output).
CONSENSUS = os.environ.get("TITLER_CONSENSUS", os.path.join(HERE, "gold_consensus.jsonl"))


def load_consensus():
    """By k: per-annotator labels + consensus (>= 2 of 5)."""
    if not os.path.exists(CONSENSUS):
        return None
    return {r["k"]: r for r in map(json.loads, io.open(CONSENSUS, encoding="utf-8"))}


def evaluate_multi(name, predictor, gold, cons=None):
    """F1 against the best reference per message (max over the annotators),
    and against the consensus.

    A single reference caps everyone around 0.70: the task admits several
    good titles. Against the best of 5 references, the annotators score
    0.84-0.87 and the model's real gap becomes visible again.
    """
    cons = cons or load_consensus()
    if cons is None:
        return None
    c_max, c_cons = [], []
    for g in gold:
        r = cons.get(g["k"])
        if r is None:
            continue
        pred = predictor(g)
        best = None
        for lab in r["annotators"].values():
            tp = fp = fn = 0
            for y, p in zip(lab, pred):
                tp += p and y
                fp += p and not y
                fn += y and not p
            f = prf(tp, fp, fn)[2]
            if best is None or f > best[0]:
                best = (f, (tp, fp, fn))
        c_max.append(best[1])
        tp = fp = fn = 0
        for y, p in zip(r["labels"], pred):
            tp += p and y
            fp += p and not y
            fn += y and not p
        c_cons.append((tp, fp, fn))
    agg = lambda c: prf(sum(x[0] for x in c), sum(x[1] for x in c), sum(x[2] for x in c))[2]
    fm, fc = agg(c_max), agg(c_cons)
    n_votes = len(next(iter(cons.values()))["annotators"])
    print("%-34s max over %d annotators F1 %.3f  |  consensus F1 %.3f" % (name, n_votes, fm, fc))
    return fm


# ------------------------------------------------- reusable building blocks

STOPWORDS_TEXT = (
    "the a an of to in is are was for and or i you it me my we this that with "
    "can could would should do does did how what why when where please write "
    "give make create tell about on at be have has as from your")


def stopword_tokens(tok):
    """Ids of the frequent stopwords. Removing them from the start is worth +9
    points of F1: they are always there and carry no meaning."""
    out = set()
    for w in STOPWORDS_TEXT.split():
        for form in (" " + w, w, " " + w.capitalize(), w.capitalize()):
            ids = tok.encode(form).ids
            if len(ids) == 1:
                out.add(ids[0])
    return out


def word_groups(text, offsets):
    """Token indices grouped by word of the text."""
    groups, current = [], []
    for i, (a, b) in enumerate(offsets):
        raw = text[a:b]
        if current and (raw[:1].isspace() or not raw[:1].isalnum()):
            groups.append(current)
            current = []
        current.append(i)
    if current:
        groups.append(current)
    return groups


def snap_to_words(text, offsets, pred, threshold=0.34):
    """A word is kept whole or not at all.

    Without this, a word split across several tokens comes out mangled:
    `uzbekistan` becomes `uz kist`. Worth +8 points of word-level F1, for
    three lines. Should also be applied to the model's own output.
    """
    pred = list(pred)
    for grp in word_groups(text, offsets):
        v = 1 if sum(pred[i] for i in grp) / len(grp) >= threshold else 0
        for i in grp:
            pred[i] = v
    return pred


def baseline_reference(tok):
    """The floor to beat: token F1 0.57, word F1 0.57 on the gold set."""
    stop = stopword_tokens(tok)

    def f(g):
        out = [0] * len(g["ids"])
        for i in range(min(20, len(out))):
            if g["ids"][i] not in stop:
                out[i] = 1
        return snap_to_words(g["text"], g["offsets"], out)
    return f


# ---------------------------------------------------------------- baselines

def idf_from_corpus(tok, n=20000):
    """Document frequency of each token, over the training corpus."""
    df = Counter()
    total = 0
    for l in io.open(MESSAGES, encoding="utf-8"):
        m = json.loads(l)
        df.update(set(tok.encode(m["text"][:1200]).ids))
        total += 1
        if total >= n:
            break
    return {t: math.log(total / (1 + c)) for t, c in df.items()}, math.log(total)


def make_baselines(tok):
    idf, idf_max = idf_from_corpus(tok)

    def discard_all(g):
        return [0] * len(g["ids"])

    def first_n(k):
        def f(g):
            out = [0] * len(g["ids"])
            for i in range(min(k, len(out))):
                out[i] = 1
            return out
        return f

    def rarest(k):
        """The k rarest tokens in the corpus: rare words carry the meaning."""
        def f(g):
            scores = [(idf.get(t, idf_max), -i) for i, t in enumerate(g["ids"])]
            kept = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
            out = [0] * len(g["ids"])
            for i in kept:
                out[i] = 1
            return out
        return f

    return [("discard all", discard_all),
            ("the first 12 tokens", first_n(12)),
            ("the 12 rarest tokens", rarest(12)),
            ("REFERENCE: first 20, no stopwords, snapped to words",
             baseline_reference(tok))]


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    gold = load_gold(tok)
    density = sum(sum(g["labels"]) for g in gold) / sum(len(g["labels"]) for g in gold)
    print("gold: %d messages, %.1f%% of tokens are to be kept\n" % (len(gold), 100 * density))
    best = ("", 0)
    for name, f in make_baselines(tok):
        s = evaluate(name, f, gold, detail=name.startswith("REFERENCE"))
        if s > best[1]:
            best = (name, s)
    print("\nfloor to beat: %s, token F1 %.2f" % best)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
