"""Gold evaluation set: 300 messages drawn from eval_pool, labeled by hand.

Format of a label (eval/gold_labels.jsonl, one line per message):
    {"id": ..., "title": "words copied as-is in order", "impossible": false}

Rules checked by `check`:
  - every word of the title appears in the message (truncated to 1200
    characters, same as what the teacher sees), same case, in order;
  - 3 to 5 words (except `impossible`, where we keep a best effort).
`check` computes the [start, end) position of each retained word: that is
the tagger's label.

    python eval/build_gold.py select          # writes eval/gold_todo.jsonl (300)
    python eval/build_gold.py check           # checks gold_labels.jsonl, writes gold.jsonl
    python eval/build_gold.py show 0 50       # shows messages 0..49 to label
"""
import io
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
POOL = os.environ.get("TITLER_POOL", os.path.join(HERE, "..", "data", "canonical", "eval_pool.jsonl"))
TODO = os.environ.get("TITLER_TODO", os.path.join(HERE, "gold_todo.jsonl"))
LABELS = os.environ.get("TITLER_LABELS", os.path.join(HERE, "gold_labels.jsonl"))
GOLD = os.environ.get("TITLER_GOLD", os.path.join(HERE, "gold.jsonl"))
REVIEW = os.environ.get("TITLER_REVIEW", os.path.join(HERE, "gold_review.md"))
MAX_CHARS = 1200
N = 300
# same distribution as train_messages: short 38%, medium 33%, long 18%, very long 11%
BUCKETS = [("short", 20, 100, 114), ("medium", 100, 400, 99),
           ("long", 400, 1500, 54), ("very_long", 1500, 8001, 33)]
TOKEN = re.compile(r"\S+")


def select():
    pool = [json.loads(l) for l in io.open(POOL, encoding="utf-8")]
    rng = random.Random(20260902)
    chosen = []
    if len(pool) <= N:
        # small reserve (partial languages): take everything
        for m in pool:
            name = next(t[0] for t in BUCKETS if t[1] <= m["n_chars"] < t[2])
            chosen.append(dict(m, bucket=name, text=m["text"][:MAX_CHARS]))
        print("reserve of %d messages: all retained" % len(pool))
    for name, lo, hi, n in ([] if len(pool) <= N else BUCKETS):
        cand = [m for m in pool if lo <= m["n_chars"] < hi]
        rng.shuffle(cand)
        for m in cand[:n]:
            chosen.append(dict(m, bucket=name, text=m["text"][:MAX_CHARS]))
        print("%-10s available=%4d retained=%d" % (name, len(cand), min(n, len(cand))))
    rng.shuffle(chosen)
    with io.open(TODO, "w", encoding="utf-8") as fh:
        for i, m in enumerate(chosen):
            fh.write(json.dumps(dict(m, k=i), ensure_ascii=False) + "\n")
    print("%d messages -> %s" % (len(chosen), os.path.relpath(TODO)))


def spans(title, text):
    """[start, end) spans of each title word in the text, in order, exact case.

    Cursor-based search over the raw text: a word must appear whole (not
    stuck to a letter or a digit), but surrounding punctuation is free. So
    `partition(self,` yields `partition`, `ending....manager` yields `ending`
    then `manager`, and `c++,` yields `c++`.
    """
    out, cursor = [], 0
    for word in title.split():
        pattern = r"(?<![^\W\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])" + re.escape(word) + r"(?![^\W\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])"
        m = re.compile(pattern).search(text, cursor)
        if m is None:
            return None, word
        out.append([m.start(), m.end()])
        cursor = m.end()
    return out, None


def check():
    # the labels are indexed by k (the number displayed by `show`)
    todo = {json.loads(l)["k"]: json.loads(l) for l in io.open(TODO, encoding="utf-8")}
    labels = [json.loads(l) for l in io.open(LABELS, encoding="utf-8")]
    ok, errors, gold = 0, [], []
    seen = set()
    for lab in labels:
        m = todo.get(lab["k"])
        if m is None:
            errors.append((str(lab["k"]), "unknown k")); continue
        if lab["k"] in seen:
            errors.append((str(lab["k"]), "duplicate")); continue
        seen.add(lab["k"])
        title = lab["title"].strip()
        n = len(title.split())
        if not lab.get("impossible") and not 3 <= n <= 5:
            errors.append((str(lab["k"]), "%d words: %r" % (n, title))); continue
        sp, missing = spans(title, m["text"])
        if sp is None:
            errors.append((str(lab["k"]), "word missing or out of order: %r in %r" % (missing, title))); continue
        ok += 1
        gold.append({"id": m["id"], "k": m["k"], "bucket": m["bucket"], "text": m["text"],
                     "title": title, "spans": sp, "impossible": bool(lab.get("impossible"))})
    missing_ks = [k for k in todo if k not in seen]
    print("%d valid labels, %d errors, %d messages without a label" % (ok, len(errors), len(missing_ks)))
    for i, e in errors:
        print("  ERROR k=%s : %s" % (i, e))
    if missing_ks:
        ks = sorted(todo[k]["k"] for k in missing_ks)
        print("  still to do: k =", ks[:20], "..." if len(ks) > 20 else "")
    gold.sort(key=lambda g: g["k"])
    with io.open(GOLD, "w", encoding="utf-8") as fh:
        for g in gold:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    with io.open(REVIEW, "w", encoding="utf-8") as fh:
        fh.write("# Gold evaluation set — review\n\nOne title per message. The words are copied as-is from the message.\n"
                 "Mentally check: is this a good title for finding this conversation again?\n\n")
        for g in gold:
            excerpt = " ".join(g["text"][:220].split())
            fh.write("**%d.** `%s`%s\n> %s\n\n" % (g["k"], g["title"], "  *(impossible)*" if g["impossible"] else "", excerpt))
    imp = sum(1 for g in gold if g["impossible"])
    print("-> %s (%d entries, %d impossible), review: %s" % (os.path.relpath(GOLD), len(gold), imp, os.path.relpath(REVIEW)))


def show(a, b):
    todo = [json.loads(l) for l in io.open(TODO, encoding="utf-8")]
    for m in todo[a:b]:
        print("=== k=%d  id=%s  (%s, %d c)" % (m["k"], m["id"][:8], m["bucket"], m["n_chars"]))
        print(m["text"])
        print()


if __name__ == "__main__":
    # Windows console in cp1252: the messages contain unicode
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "select"
    if cmd == "select":
        select()
    elif cmd == "check":
        check()
    elif cmd == "show":
        show(int(sys.argv[2]), int(sys.argv[3]))
