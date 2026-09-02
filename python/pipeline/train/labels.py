"""From (message, title) to a per-token keep/discard label.

Three stages, in this order:

  1. ALIGNMENT   each word of the title -> a character span of the message.
     Rules, in order of preference and validated on the gold eval set:
       exact   the word appears as-is, bounded by non-alphanumerics.
               `tolkien` is found inside `Tolkien's`, `xml` inside `log4j2.xml`.
       stem    inflection: `explores` -> `explore`.
       typo    one edit or one transposition: `search` -> `serach`.
     We first try a subsequence alignment (a cursor that only moves forward).
     If the teacher reordered the words, we fall back to a free search and
     then sort the spans: the teacher's word order is discarded either way,
     since a tagger can only ever restitute the message's own order.

  2. TOKENS      a token is worth 1 if it overlaps a kept span. Whitespace at
     the head of a token (byte-level BPE) is ignored in the overlap test,
     otherwise the token ` the` would overlap the previous word's span.

  3. CHECK       the kept tokens are re-decoded and checked against the title's
     own words. A pair that fails to recompose is rejected.

Output format (train_labeled.jsonl, one line per message): "id", "title",
"ids" (token ids), "labels" (0/1 per token), and "words" — the word number of
each token (see `word_ids_per_token`), used so the model can score a whole
word as a single decision instead of one token at a time.

    python train/labels.py            # derives everything + checks against the gold set
"""
import io
import json
import os
import re
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TOKENIZER = os.environ.get("TITLER_TOKENIZER", os.path.join(HERE, "tokenizer.json"))
MESSAGES = os.environ.get("TITLER_MESSAGES", os.path.join(ROOT, "data", "canonical", "train_messages.jsonl"))
# TITLER_TITLES / TITLER_LABELED make it possible to derive a second set
# (e.g. v2 titles) without overwriting the first.
TITLES = os.environ.get("TITLER_TITLES", os.path.join(ROOT, "data", "canonical", "train_titles.jsonl"))
GOLD = os.environ.get("TITLER_GOLD", os.path.join(ROOT, "eval", "gold.jsonl"))
OUTPUT = os.environ.get("TITLER_LABELED", os.path.join(ROOT, "data", "canonical", "train_labeled.jsonl"))
MAX_CHARS = 1200
MAX_TOKENS = 512

JOINERS = {"-", "'", "’"}
PUNCT = ".,;:!?\"'()[]{}<>*`“”‘’«»…-–—"   # stripped at the edges of a word only
SUFFIXES = ("ing", "ed", "es", "s", "ly", "er")


def title_words(title):
    out = []
    for t in title.split():
        t = t.strip(PUNCT)
        if t:
            out.append(t)
    return out


def stem(m):
    for s in SUFFIXES:
        if len(m) > len(s) + 3 and m.endswith(s):
            return m[:-len(s)]
    return m


def _one_edit(a, b):
    if min(len(a), len(b)) < 5 or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        if len(d) == 1:
            return True
        return (len(d) == 2 and d[1] == d[0] + 1
                and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    short, long_ = (a, b) if len(a) < len(b) else (b, a)
    i = 0
    while i < len(short) and short[i] == long_[i]:
        i += 1
    return short[i:] == long_[i + 1:]


def occurrences(word, lower_text, msg_words):
    """Candidate spans for a title word, from most to least reliable."""
    exact = [(m.start(), m.end()) for m in
             re.finditer(r"(?<![^\W_\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])" + re.escape(word) + r"(?![^\W_\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u3400-\u4dbf])", lower_text)]
    if exact:
        return exact, "exact"
    r = stem(word)
    inflected = [sp for w, sp in msg_words if stem(w) == r]
    if inflected:
        return inflected, "stem"
    typo = [sp for w, sp in msg_words if _one_edit(word, w)]
    if typo:
        return typo, "typo"
    return [], None


def align(title, text):
    """Sorted spans to keep, or (None, offending word)."""
    lower_text = text.lower()
    msg_words = [(m.group().lower(), (m.start(), m.end()))
                 for m in re.finditer(r"[^\W_]+", text)]
    candidates, rules = [], []
    for word in title_words(title):
        occ, rule = occurrences(word.lower(), lower_text, msg_words)
        if not occ:
            return None, word, None
        candidates.append(occ)
        rules.append(rule)

    # subsequence: each word after the previous one. Otherwise, first occurrence.
    spans, cursor, ordered = [], 0, True
    for occ in candidates:
        next_span = next((sp for sp in occ if sp[0] >= cursor), None)
        if next_span is None:
            ordered = False
            next_span = occ[0]
        spans.append(next_span)
        cursor = next_span[1]
    if not ordered:
        spans = sorted(set(spans))
    return spans, None, ("exact" if all(r == "exact" for r in rules) else "tolerant")


def label_tokens(tok, text, spans, max_tokens=MAX_TOKENS):
    """ids + 0/1 labels per token. A token is worth 1 if it overlaps a span."""
    enc = tok.encode(text)
    ids, offsets = enc.ids[:max_tokens], enc.offsets[:max_tokens]
    labels = []
    for (a, b) in offsets:
        # ignore leading whitespace in the token, otherwise ` the` bites into the previous word
        while a < b and text[a].isspace():
            a += 1
        labels.append(1 if any(a < end and b > start for start, end in spans) and a < b else 0)
    return ids, labels, offsets


def word_ids_per_token(text, offsets):
    """Word number for each token. A word starts when the token opens on a
    space or on a non-alphanumeric character.

    Used by the model so that a word is a single decision: the scores of the
    tokens belonging to the same word are averaged BEFORE the decision, and
    that average is trained. Without this, the tagger cut a word into pieces
    in 193 titles out of 298.
    """
    out, num, prev_end = [], -1, -1
    for i, (a, b) in enumerate(offsets):
        raw = text[a:b]
        first = raw[:1]
        starts = i == 0 or first.isspace() or not first.isalnum()
        # a hyphen or an apostrophe glued between two letters does not split the
        # word ("Pourrais-tu", "l'ecole"): a title must never start with "-tu"
        if starts and i > 0 and first in JOINERS and a == prev_end and a > 0 and text[a - 1].isalnum()                 and a + 1 < len(text) and text[a + 1].isalnum():
            starts = False
        if starts:
            num += 1
        out.append(num)
        prev_end = b
    return out


def recompose(text, offsets, labels):
    """The kept tokens, glued back into words.

    We first merge the kept CHARACTER intervals, then extract each interval
    exactly once. Gluing token by token would duplicate multi-byte characters:
    in byte-level BPE, `ç` or `汉` are several byte-tokens pointing at the same
    character (`ça` -> `çça`).
    """
    intervals = []
    for (a, b), y in zip(offsets, labels):
        if not y or b <= a:
            continue
        if intervals and a <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], b)
        else:
            intervals.append([a, b])
    # punctuation stuck to the edge of a word is removed (`"Politics` -> `Politics`), interior punctuation stays (`c++`, `don't`)
    pieces = [text[a:b].strip().strip(PUNCT) for a, b in intervals]
    return " ".join(m for m in pieces if m)


def verify_on_gold(tok):
    """Does alignment recover the spans the human laid down?"""
    gold = [json.loads(l) for l in io.open(GOLD, encoding="utf-8")]
    exact, shifted, failures = 0, 0, []
    for g in gold:
        spans, offender, _ = align(g["title"], g["text"])
        if spans is None:
            failures.append((g["k"], "word not found: " + offender))
            continue
        expected = [tuple(sp) for sp in g["spans"]]
        obtained = [tuple(sp) for sp in spans]
        if obtained == sorted(expected):
            exact += 1
        else:
            shifted += 1
            if len(failures) < 8:
                failures.append((g["k"], "different spans: %s vs %s" % (obtained[:3], sorted(expected)[:3])))
    print("checking alignment against the 300 hand-placed spans:")
    print("  identical spans   : %d/%d (%.0f%%)" % (exact, len(gold), 100 * exact / len(gold)))
    print("  different spans   : %d   |   failures: %d"
          % (shifted, sum(1 for r in failures if "not found" in r[1])))
    for k, m in failures[:8]:
        print("    k=%-4d %s" % (k, m))
    return exact, len(gold)


def main():
    tok = Tokenizer.from_file(TOKENIZER)
    if os.path.exists(GOLD):
        verify_on_gold(tok)
    else:
        print("no gold (%s): skipping the alignment check" % os.path.relpath(GOLD, ROOT))

    texts = {}
    for l in io.open(MESSAGES, encoding="utf-8"):
        m = json.loads(l)
        texts[m["id"]] = m["text"][:MAX_CHARS]

    stats = {"read": 0, "no_title": 0, "refused": 0, "aligned_exact": 0,
             "aligned_tolerant": 0, "not_aligned": 0, "recompose_failed": 0, "written": 0}
    kept_total, tok_total, truncated = 0, 0, 0
    with io.open(OUTPUT, "w", encoding="utf-8") as fh:
        for l in io.open(TITLES, encoding="utf-8"):
            r = json.loads(l)
            stats["read"] += 1
            if not r.get("title"):
                stats["no_title"] += 1
                continue
            if r.get("refused"):
                stats["refused"] += 1
                continue
            text = texts.get(r["id"])
            if text is None:
                continue
            spans, offender, rule = align(r["title"], text)
            if spans is None:
                stats["not_aligned"] += 1
                continue
            stats["aligned_" + rule] += 1
            ids, labels, offsets = label_tokens(tok, text, spans)
            if sum(labels) == 0:
                stats["recompose_failed"] += 1
                continue
            recomposed = recompose(text, offsets, labels)
            if len(title_words(recomposed)) < len(title_words(r["title"])):
                # the title runs past the 512-token truncation
                truncated += 1
            kept_total += sum(labels)
            tok_total += len(labels)
            stats["written"] += 1
            fh.write(json.dumps({"id": r["id"], "title": r["title"], "ids": ids,
                                 "labels": labels,
                                 "words": word_ids_per_token(text, offsets)}, ensure_ascii=False) + "\n")

    print("\nlabel derivation:")
    for k, v in stats.items():
        print("  %-16s %s" % (k, format(v, ",")))
    if stats["written"]:
        print("  %-16s %.1f%% of tokens are kept on average"
              % ("density", 100 * kept_total / tok_total))
        print("  %-16s %s pairs where the title runs past 512 tokens"
              % ("truncated", format(truncated, ",")))
    print("-> %s" % os.path.relpath(OUTPUT, ROOT))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
