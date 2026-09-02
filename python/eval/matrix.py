"""The score matrix: every shipped model (int3 `.bin`) on every gold set, plus
the teacher LLM and a no-model floor, all scored the same way — word-level F1
against a single reference (the human-quality titles of the gold set).

A single reference keeps everyone comparable: the teacher is also a voice of
the gold consensus, so a multi-reference score would favour it against itself.
Models are run through the reference reader on the `.bin` files: we score
what we ship.

    python matrix.py --models ../../packages --gold path/to/gold --teacher path/to/teacher --out scores.json

Gold files: `<lang>.jsonl` with {"id", "text", "title", "impossible"} per line
(the aparte-titler-gold dataset). Teacher files: `<lang>.jsonl` with {"id", "title"}.
"""
import argparse
import glob
import io
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "aparte_titler"))
from reader import Titler, PUNCT  # noqa: E402

LANGUAGES = "en fr es de pt it nl pl sv da fi cs ro no hu hr lt".split()


def words(title):
    return [w.strip(PUNCT).lower() for w in title.split() if w.strip(PUNCT)]


def f1(a, b):
    a, b = Counter(words(a)), Counter(words(b))
    if not a or not b:
        return 0.0
    inter = sum((a & b).values())
    return 2 * inter / (sum(a.values()) + sum(b.values())) if inter else 0.0


def floor(text):
    """The first 5 words that contain a letter or a digit: what 'no model' gives."""
    out = []
    for w in text.split():
        if any(ch.isalnum() for ch in w):
            out.append(w)
        if len(out) == 5:
            break
    return " ".join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", required=True, help="folder searched recursively for titler-*-int3.bin")
    p.add_argument("--gold", required=True, help="folder with <lang>.jsonl gold files")
    p.add_argument("--teacher", default="", help="folder with <lang>.jsonl teacher titles (optional)")
    p.add_argument("--out", default="scores.json")
    args = p.parse_args()

    golds = {}
    for lang in LANGUAGES:
        f = os.path.join(args.gold, lang + ".jsonl")
        if os.path.exists(f):
            golds[lang] = [g for g in map(json.loads, io.open(f, encoding="utf-8")) if not g.get("impossible")]
    scores = {"metric": "word-level F1 against the gold title (single reference), budget 6, impossibles excluded",
              "n_messages": {c: len(g) for c, g in golds.items()}, "models": {}, "teacher": {}, "floor": {}}
    for lang, gold in golds.items():
        scores["floor"][lang] = round(sum(f1(floor(g["text"]), g["title"]) for g in gold) / len(gold), 4)
        tf = os.path.join(args.teacher, lang + ".jsonl") if args.teacher else ""
        if tf and os.path.exists(tf):
            teacher = {r["id"]: r.get("title") or "" for r in map(json.loads, io.open(tf, encoding="utf-8"))}
            scores["teacher"][lang] = round(sum(f1(teacher.get(g["id"], ""), g["title"]) for g in gold) / len(gold), 4)

    t0 = time.perf_counter()
    for path in sorted(glob.glob(os.path.join(args.models, "**", "titler-*-int3.bin"), recursive=True)):
        name = os.path.basename(path)[len("titler-v1-"):-len("-int3.bin")]
        t = Titler(path)
        row = {lang: round(sum(f1(t.title(g["text"]), g["title"]) for g in gold) / len(gold), 4) for lang, gold in golds.items()}
        scores["models"][name] = {"scores": row, "bytes_int3": os.path.getsize(path), "params": None,
                                  "vocab": t.header["vocab"]}
        print("%-11s %s  (%.0fs)" % (name, " ".join("%s %.2f" % (c, row[c]) for c in golds), time.perf_counter() - t0), flush=True)
    with io.open(args.out, "w", encoding="utf-8") as fh:
        json.dump(scores, fh, indent=1, ensure_ascii=False)
    print("-> " + args.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
