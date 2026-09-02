"""The teacher against the gold evaluation set.

Three questions, in order:
  1. The STRICT validator (every word of the title is in the message, case
     ignored) against a TOLERANT validator (stuck-on punctuation, inflections,
     a 1-character typo): how many more titles does the tolerant one recover?
  2. Are those recovered titles genuinely extractive titles? They are listed
     one by one for human review: that is where the tolerant validator can be wrong.
  3. Does the teacher say the same thing as the human? Word overlap between
     the teacher's title and the reference title (precision, recall, F1).

    python eval/compare_teacher.py data/bakeoff/results/teacher-gold__extractive.jsonl

Output: eval/teacher_vs_gold.md (the cases to review) + a console summary.
"""
import io
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "gold.jsonl")
OUTPUT = os.path.join(HERE, "teacher_vs_gold.md")
PUNCT = ".,;:!?\"'()[]{}<>*`“”‘’«»…"
SUFFIXES = ("ing", "ed", "es", "s", "ly", "er")


def words(text):
    """Lowercased words, outer punctuation stripped, inner punctuation kept:
    c++, node.js, usb-A, don't stay whole."""
    out = []
    for t in text.split():
        t = t.strip(PUNCT).lower()
        if t:
            out.append(t)
    return out


def stem(m):
    for s in SUFFIXES:
        if len(m) > len(s) + 3 and m.endswith(s):
            return m[:-len(s)]
    return m


def distance1(a, b):
    """True if a and b differ by at most one edit (insertion, deletion,
    substitution). Reserved for words of at least 5 letters."""
    if a == b:
        return True
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def present(word, lower_text):
    """STRICT: the word appears as-is in the message, bounded by
    non-alphanumerics. This is the gold checker's own rule. It accepts
    `tolkien` inside `Tolkien's`, `xml` inside `log4j2.xml`, `redefining`
    inside `1_Redefining`, `dorigo` inside `对Dorigo` — all cases where
    splitting on whitespace got it wrong."""
    pattern = r"(?<![^\W_\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])" + re.escape(word) + r"(?![^\W_\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])"
    return re.search(pattern, lower_text) is not None


def transposition(a, b):
    """Two adjacent letters swapped: serach/search, difussion/diffusion."""
    if len(a) != len(b) or len(a) < 5:
        return False
    d = [i for i in range(len(a)) if a[i] != b[i]]
    return len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]]


def match_word(word, lower_text, vocab):
    """How a title word connects back to the message: exact, stem, or typo.
    Returns (message word, rule) or (None, None)."""
    if present(word, lower_text):
        return word, "exact"
    r = stem(word)
    for v in vocab:
        if stem(v) == r:
            return v, "stem"
    for v in vocab:
        if distance1(word, v) or transposition(word, v):
            return v, "typo"
    return None, None


def f1(a, b):
    a, b = set(a), set(b)
    if not a or not b:
        return 0.0, 0.0, 0.0
    inter = len(a & b)
    p, r = inter / len(a), inter / len(b)
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def main(path):
    gold = {g["id"]: g for g in (json.loads(l) for l in io.open(GOLD, encoding="utf-8"))}
    teacher = [json.loads(l) for l in io.open(path, encoding="utf-8")]
    teacher = [t for t in teacher if t.get("title") and t["id"] in gold]

    strict_ok, tolerant_ok = 0, 0
    recovered, failures, scores = [], [], []
    for t in teacher:
        g = gold[t["id"]]
        lower_text = g["text"].lower()
        # vocabulary for the tolerant layer: words bounded by whitespace AND
        # alphanumeric runs, so that `search` can connect back to `serach`
        # even when it's stuck to punctuation
        vocab = set(words(g["text"])) | set(re.findall(r"[a-z0-9]+", lower_text))
        title = words(t["title"])
        rules = [match_word(m, lower_text, vocab) for m in title]
        strict = all(present(m, lower_text) for m in title) and bool(title)
        tolerant = all(r[0] is not None for r in rules) and bool(title)
        strict_ok += strict
        tolerant_ok += tolerant
        p, r, f = f1(title, words(g["title"]))
        scores.append((f, t, g))
        if tolerant and not strict:
            subst = ["%s->%s (%s)" % (m, r[0], r[1]) for m, r in zip(title, rules) if r[1] != "exact"]
            recovered.append((t, g, subst))
        elif not tolerant:
            missing = [m for m, r in zip(title, rules) if r[0] is None]
            failures.append((t, g, missing))

    n = len(teacher)
    f_avg = sum(s[0] for s in scores) / n
    f_good = sum(1 for s in scores if s[0] >= 0.5)
    exact_matches = sum(1 for s in scores if set(words(s[1]["title"])) == set(words(s[2]["title"])))
    impossible = sum(1 for g in gold.values() if g.get("impossible"))

    print("teacher on the %d gold messages" % n)
    print("  strict validator    : %d/%d (%.0f%%)" % (strict_ok, n, 100 * strict_ok / n))
    print("  tolerant validator  : %d/%d (%.0f%%)  -> %d titles recovered, to review"
          % (tolerant_ok, n, 100 * tolerant_ok / n, len(recovered)))
    print("  true failures       : %d (word not found even with tolerance)" % len(failures))
    print("  agreement with human: avg F1 %.2f | F1>=0.5 : %d/%d (%.0f%%) | same words: %d"
          % (f_avg, f_good, n, 100 * f_good / n, exact_matches))
    print("  gold 'impossible'   : %d" % impossible)

    with io.open(OUTPUT, "w", encoding="utf-8") as fh:
        fh.write("# Teacher against the gold evaluation set\n\n")
        fh.write("strict %d/%d · tolerant %d/%d · avg F1 %.2f\n\n" % (strict_ok, n, tolerant_ok, n, f_avg))
        fh.write("## Recovered by the tolerant validator — TO REVIEW (%d)\n\n"
                 "Is the tolerant validator wrongly accepting these? Each line shows the substitution.\n\n" % len(recovered))
        for t, g, subst in recovered:
            fh.write("- **k=%d** teacher `%s` — %s\n  - gold `%s`\n" % (g["k"], t["title"], "; ".join(subst), g["title"]))
        fh.write("\n## True teacher failures (%d)\n\n" % len(failures))
        for t, g, missing in failures:
            fh.write("- **k=%d** teacher `%s` — missing: %s\n  - gold `%s`\n" % (g["k"], t["title"], ", ".join(missing), g["title"]))
        fh.write("\n## Largest disagreements with the human (lowest F1)\n\n")
        for f, t, g in sorted(scores, key=lambda s: s[0])[:25]:
            excerpt = " ".join(g["text"][:120].split())
            fh.write("- **k=%d** F1 %.2f · teacher `%s` · gold `%s`%s\n  > %s\n"
                     % (g["k"], f, t["title"], g["title"], " *(impossible)*" if g["impossible"] else "", excerpt))
    print("-> %s" % os.path.relpath(OUTPUT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(HERE, "..", "data", "bakeoff", "results", "teacher-gold__extractive.jsonl"))
