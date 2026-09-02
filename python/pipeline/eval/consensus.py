"""Builds the multi-voice reference from N annotators.

A single reference caps everyone around 0.70: the task admits several good
titles. So we keep ALL the voices, and two readings:
  - max        : F1 against the best reference (excluding self for an annotator)
  - consensus  : tokens kept by at least `--mini` annotators

Inputs: title files, indexed either by "k" (blind annotations) or by "id"
(teacher_run.py output). The title is aligned onto the gold set's text with
the same rules as labels.py; an unalignable title counts as "nothing kept".

    python eval/consensus.py --mini 2 \
        annotator_a=eval/gold_labels.jsonl annotator_b=eval/gold_labels_b.jsonl 
        teacher=data/bakeoff/results/teacher-gold__extractive.jsonl

Outputs: eval/gold_consensus.jsonl (per-annotator labels + consensus),
         eval/gold_review_multi.md (every voice per message, for review).
"""
import argparse
import io
import json
import os
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "train"))
from labels import align, label_tokens  # noqa: E402

TODO = os.environ.get("TITLER_TODO", os.path.join(HERE, "gold_todo.jsonl"))
GOLD = os.environ.get("TITLER_GOLD", os.path.join(HERE, "gold.jsonl"))
OUTPUT = os.path.join(HERE, "gold_consensus.jsonl")
REVIEW = os.path.join(HERE, "gold_review_multi.md")
TOKENIZER = os.environ.get("TITLER_TOKENIZER", os.path.join(ROOT, "train", "tokenizer.json"))
MAX_TOKENS = 512


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def read_titles(spec, ids_by_k):
    """'name=f1+f2' -> {k: title}. Files indexed by k or by id."""
    k_by_id = {v: k for k, v in ids_by_k.items()}
    out = {}
    for path in spec.split("+"):
        for l in io.open(os.path.join(ROOT, path) if not os.path.isabs(path) else path, encoding="utf-8"):
            r = json.loads(l)
            if not r.get("title"):
                continue
            k = r["k"] if "k" in r else k_by_id.get(r.get("id"))
            if k is not None:
                out[k] = r["title"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("annotators", nargs="+", help="name=file[+file]")
    p.add_argument("--mini", type=int, default=2, help="minimum votes for the consensus")
    p.add_argument("--output", default=OUTPUT, help="the labels depend on the tokenizer: one file per tokenizer")
    p.add_argument("--review", default=REVIEW)
    args = p.parse_args()
    output, review = args.output, args.review

    tok = Tokenizer.from_file(TOKENIZER)
    todo = {json.loads(l)["k"]: json.loads(l) for l in io.open(TODO, encoding="utf-8")}
    ids_by_k = {k: m["id"] for k, m in todo.items()}
    impossible = set()
    if os.path.exists(GOLD):
        impossible = {json.loads(l)["k"] for l in io.open(GOLD, encoding="utf-8") if json.loads(l).get("impossible")}
    ks = [k for k in sorted(todo) if k not in impossible]

    # text truncated the same way everywhere, tokens and offsets computed once
    base = {}
    for k in ks:
        text = todo[k]["text"][:1200]
        enc = tok.encode(text)
        base[k] = (text, enc.ids[:MAX_TOKENS], enc.offsets[:MAX_TOKENS])

    names, titles, labels, unaligned = [], {}, {}, {}
    for spec in args.annotators:
        name, _, files = spec.partition("=")
        names.append(name)
        titles[name] = read_titles(files, ids_by_k)
        labels[name], unaligned[name] = {}, 0
        for k in ks:
            text, ids, _ = base[k]
            t = titles[name].get(k, "")
            spans, _, _ = align(t, text) if t else (None, None, None)
            if spans is None:
                unaligned[name] += 1
                labels[name][k] = [0] * len(ids)
            else:
                _, lab, _ = label_tokens(tok, text, spans, MAX_TOKENS)
                labels[name][k] = (lab + [0] * len(ids))[:len(ids)]

    def f1_between(a, b):
        tp = fp = fn = 0
        for k in ks:
            for y, q in zip(labels[a][k], labels[b][k]):
                tp += q and y
                fp += q and not y
                fn += y and not q
        return prf(tp, fp, fn)[2]

    def leave_one_out(a):
        c = []
        for k in ks:
            best = None
            for b in names:
                if b == a:
                    continue
                tp = fp = fn = 0
                for y, q in zip(labels[b][k], labels[a][k]):
                    tp += q and y
                    fp += q and not y
                    fn += y and not q
                f = prf(tp, fp, fn)[2]
                if best is None or f > best[0]:
                    best = (f, (tp, fp, fn))
            c.append(best[1])
        return prf(sum(x[0] for x in c), sum(x[1] for x in c), sum(x[2] for x in c))[2]

    print("%d annotators, %d messages (impossible ones excluded)\n" % (len(names), len(ks)))
    print("pairwise agreement (token F1):")
    print("%-10s" % "" + "".join("%8s" % n[:8] for n in names))
    for a in names:
        print("%-10s" % a[:10] + "".join("%8s" % ("--" if a == b else "%.2f" % f1_between(a, b)) for b in names))
    print("\n%-10s %10s %12s %8s" % ("", "leave-1-out", "avg agreement", "unalign."))
    for a in names:
        avg = sum(f1_between(a, b) for b in names if b != a) / (len(names) - 1)
        print("%-10s %10.3f %12.3f %8d" % (a, leave_one_out(a), avg, unaligned[a]))

    with io.open(output, "w", encoding="utf-8") as fh, io.open(review, "w", encoding="utf-8") as rv:
        rv.write("# %d-voice reference — review\n\n" % len(names))
        dens_n, dens_d = 0, 0
        for k in ks:
            text, ids, _ = base[k]
            votes = [sum(labels[n][k][i] for n in names) for i in range(len(ids))]
            cons = [1 if v >= args.mini else 0 for v in votes]
            dens_n += sum(cons)
            dens_d += len(cons)
            fh.write(json.dumps({"k": k, "id": ids_by_k[k], "labels": cons, "votes": votes,
                                 "annotators": {n: labels[n][k] for n in names}}, ensure_ascii=False) + "\n")
            rv.write("**%d.** %s\n" % (k, " ".join(text[:200].split())))
            for n in names:
                rv.write("- %-8s `%s`\n" % (n, titles[n].get(k, "(none)")))
            rv.write("\n")
    print("\n-> %s (consensus >= %d votes, density %.1f%%)" % (os.path.relpath(output, ROOT), args.mini, 100 * dens_n / dens_d))
    print("-> %s" % os.path.relpath(review, ROOT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
