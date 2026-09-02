"""The message corpora, one language at a time: EXACTLY 12,000 training
messages and up to 800 for the gold set when the real data allows it,
otherwise whatever there is, marked "partial" (for the group models).

REAL sources, in priority order (never translated):
  1. WildChat-4.8M   real first user messages (3.2M conversations)
  2. OASST2          root prompts written by volunteers (13,000, 28 languages)
  3. Aya             human prompts (195,000, 65 languages), question/task style
  4. MFAQ            questions from FAQ web pages (CC0, 21 languages), at most
                     MFAQ_CAP_PER_SITE questions per site: uncapped, five
                     travel sites would make up half the questions
Each source is taken in full, in order; only the one that overflows the need
is sampled down to the shortfall: all the real chat is kept. A single pass
over WildChat's 86 parquet files keeps the candidates for every language
wanted; then, per language, the same steps as `build_corpus.py` (exact
duplicates, MinHash near-duplicates, uniform sampling over time).

Outputs, per language code:
  data/canonical/<code>_messages.jsonl      12,000 (or partial), `source` field
  data/canonical/<code>_eval_pool.jsonl      250 to 800, reserved for the gold set
  data/canonical/<code>_build_stats.json     the counts at each step
  data/canonical/languages_summary.md        the final table

    python -u data/languages.py                 # every language in LANGUAGES
    python -u data/languages.py es de            # only some of them
    python -u data/languages.py --candidates     # only redo the reading pass
    python -u data/languages.py --skip-read      # reuse the already-extracted candidates
"""
import argparse
import glob
import hashlib
import io
import json
import os
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasketch import MinHash, MinHashLSH

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_corpus import normalize, minhash, NUM_PERM, THRESHOLD, MIN_CHARS, MAX_CHARS  # noqa: E402

SOURCE = os.environ.get("TITLER_WILDCHAT", os.path.join(HERE, "wildchat48"))
OASST2 = os.path.join(HERE, "oasst2", "data")
AYA = os.path.join(HERE, "aya", "data")
MFAQ = os.path.join(HERE, "mfaq", "data")
OUT = os.path.join(HERE, "canonical")
CANDIDATES = os.path.join(OUT, "candidates")
# the gold set takes 300 messages from the reserve; 250 is enough if a
# language is a bit short (German: 12,282 after deduplication, gold set of 282)
N_TRAIN, N_EVAL, N_EVAL_MIN = 12_000, 800, 250
SEED = 20260902

# code -> language name in WildChat. Latin alphabet, official EU languages
# (excluding Irish and Maltese: no messages), plus English and French already
# done with WildChat-1M, kept here so they can be redone.
LANGUAGES = {
    "en": "English", "fr": "French",
    # tier 1: EFIGS+P
    "es": "Spanish", "de": "German", "pt": "Portuguese", "it": "Italian",
    # tier 2: Latin-1
    # WildChat splits Norwegian into "Bokmal" and "Nynorsk": we take Bokmal
    "nl": "Dutch", "da": "Danish", "sv": "Swedish", "no": "Bokmal", "fi": "Finnish", "is": "Icelandic",
    # tier 3: Latin-2 and Baltic
    "pl": "Polish", "cs": "Czech", "sk": "Slovak", "hu": "Hungarian", "ro": "Romanian",
    "hr": "Croatian", "sl": "Slovene", "lt": "Lithuanian", "lv": "Latvian", "et": "Estonian",
}
# codes for the same languages in Aya (ISO 639-3); OASST2 uses the two-letter
# code, sometimes with a suffix (pt-BR)
AYA_CODES = {"en": "eng", "fr": "fra", "es": "spa", "de": "deu", "pt": "por", "it": "ita", "nl": "nld",
             "da": "dan", "sv": "swe", "no": "nor", "fi": "fin", "is": "isl", "pl": "pol", "cs": "ces",
             "sk": "slk", "hu": "hun", "ro": "ron", "hr": "hrv", "sl": "slv", "lt": "lit", "lv": "lav", "et": "est"}


def pass_candidates(codes):
    """One pass over every WildChat parquet file: one candidates file per language."""
    wanted = {LANGUAGES[c]: c for c in codes}
    os.makedirs(CANDIDATES, exist_ok=True)
    outputs = {c: io.open(os.path.join(CANDIDATES, c + ".jsonl"), "w", encoding="utf-8") for c in codes}
    counts = Counter()
    languages_seen = Counter()
    files = sorted(glob.glob(os.path.join(SOURCE, "data", "*.parquet")))
    if not files:
        raise SystemExit("no parquet files in " + SOURCE)
    columns = ["conversation_hash", "timestamp", "conversation", "language", "toxic"]
    read, t0 = 0, time.perf_counter()
    for f in files:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=columns):
            i_ts = batch.schema.get_field_index("timestamp")
            batch = batch.set_column(i_ts, "timestamp", pc.cast(batch.column(i_ts), pa.int64()))
            # vectorized filter on language and toxicity before dropping into Python
            mask = pc.and_(pc.is_in(batch.column("language"), value_set=pa.array(list(wanted))),
                           pc.invert(pc.fill_null(batch.column("toxic"), True)))
            read += batch.num_rows
            languages_seen.update(batch.column("language").to_pylist())
            for row in batch.filter(mask).to_pylist():
                code = wanted[row["language"]]
                counts[code + "_non_toxic"] += 1
                conv = row["conversation"] or []
                if not conv or conv[0].get("role") != "user":
                    continue
                text = (conv[0].get("content") or "").strip()
                if not MIN_CHARS <= len(text) <= MAX_CHARS:
                    continue
                counts[code + "_length_ok"] += 1
                outputs[code].write(json.dumps({
                    "id": row["conversation_hash"], "text": text, "n_chars": len(text),
                    "timestamp": datetime.fromtimestamp(row["timestamp"] / 1e6, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"),
                    "source": "wildchat",
                }, ensure_ascii=False) + "\n")
        print("  %s : %s read, %.0fs" % (os.path.basename(f), format(read, ","), time.perf_counter() - t0), flush=True)
    for fh in outputs.values():
        fh.close()
    with io.open(os.path.join(CANDIDATES, "languages_wildchat.json"), "w", encoding="utf-8") as fh:
        json.dump({"read": read, "by_language": dict(languages_seen.most_common()), "candidates": dict(counts)}, fh, indent=2, ensure_ascii=False)
    return counts


def pass_short(codes):
    """The SHORT messages (1 to 19 characters, below the corpus threshold),
    distinct, per language: they go into training labeled "keep everything",
    so the model doesn't need a "short message" rule."""
    wanted = {LANGUAGES[c]: c for c in codes}
    seen = {c: set() for c in codes}
    outputs = {c: io.open(os.path.join(OUT, c + "_short_texts.jsonl"), "w", encoding="utf-8") for c in codes}
    counts = Counter()
    columns = ["conversation_hash", "conversation", "language", "toxic"]
    t0 = time.perf_counter()
    for f in sorted(glob.glob(os.path.join(SOURCE, "data", "*.parquet"))):
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=columns):
            mask = pc.and_(pc.is_in(batch.column("language"), value_set=pa.array(list(wanted))),
                           pc.invert(pc.fill_null(batch.column("toxic"), True)))
            for row in batch.filter(mask).to_pylist():
                conv = row["conversation"] or []
                if not conv or conv[0].get("role") != "user":
                    continue
                text = (conv[0].get("content") or "").strip()
                if not 1 <= len(text) < MIN_CHARS:
                    continue
                code = wanted[row["language"]]
                key = normalize(text)
                if key in seen[code]:
                    continue
                seen[code].add(key)
                counts[code] += 1
                outputs[code].write(json.dumps({"id": row["conversation_hash"], "text": text}, ensure_ascii=False) + "\n")
    for fh in outputs.values():
        fh.close()
    print("distinct short messages (%.0fs) : %s" % (time.perf_counter() - t0, ", ".join("%s %d" % (c, counts[c]) for c in codes)), flush=True)
    return counts


def read_oasst2(code):
    """First messages (prompter with no parent) of the language, in OASST2."""
    out = []
    for f in sorted(glob.glob(os.path.join(OASST2, "*.parquet"))):
        t = pq.read_table(f, columns=["message_id", "role", "parent_id", "lang", "text", "created_date"])
        for mid, role, parent, lang, text, date in zip(*[t.column(c).to_pylist() for c in
                                                          ("message_id", "role", "parent_id", "lang", "text", "created_date")]):
            if role != "prompter" or parent is not None or not (lang == code or lang.startswith(code + "-")):
                continue
            text = (text or "").strip()
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                out.append({"id": "oasst2_" + mid, "text": text, "n_chars": len(text),
                            "timestamp": str(date)[:19].replace(" ", "T"), "source": "oasst2"})
    return out


def read_aya(code):
    """Human prompts of the language in Aya (no timestamp: they go at the end
    of the pool, and the regular-step draw samples them proportionally)."""
    out = []
    for f in sorted(glob.glob(os.path.join(AYA, "*.parquet"))):
        t = pq.read_table(f, columns=["language_code", "inputs"])
        for lang, text in zip(t.column("language_code").to_pylist(), t.column("inputs").to_pylist()):
            if lang != AYA_CODES[code]:
                continue
            text = (text or "").strip()
            if MIN_CHARS <= len(text) <= MAX_CHARS:
                out.append({"id": "aya_" + hashlib.sha1(text.encode("utf-8")).hexdigest()[:16], "text": text,
                            "n_chars": len(text), "timestamp": "9999-12-31T00:00:00", "source": "aya"})
    return out


class Deduplicator:
    """Exact duplicates then MinHash near-duplicates, memory shared across sources."""

    def __init__(self):
        self.seen = set()
        self.lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
        self.exact = 0
        self.near = 0

    def filter(self, messages):
        kept = []
        for m in messages:
            key = normalize(m["text"])
            if key in self.seen:
                self.exact += 1
                continue
            self.seen.add(key)
            mh = minhash(m["text"])
            if self.lsh.query(mh):
                self.near += 1
                continue
            self.lsh.insert(m["id"], mh)
            kept.append(m)
        return kept


MFAQ_CAP_PER_SITE = 100   # Paul's decision: less bias rather than 12,000 from everywhere


def read_mfaq(code, cap=MFAQ_CAP_PER_SITE):
    """Questions from FAQ web pages (MFAQ, CC0): real questions, not chat
    messages. At least three words, and AT MOST `cap` questions per site:
    uncapped, five travel sites (momondo, tripadvisor, hotels.com...) make up
    50 to 78% of the questions, with templates where only the city changes."""
    by_site = {}
    for f in sorted(glob.glob(os.path.join(MFAQ, code, "*.jsonl"))):
        for l in io.open(f, encoding="utf-8"):
            page = json.loads(l)
            site = page.get("domain") or "?"
            for qa in page.get("qa_pairs") or [page]:
                q = (qa.get("question") or "").strip()
                if MIN_CHARS <= len(q) <= MAX_CHARS and len(q.split()) >= 3:
                    by_site.setdefault(site, []).append(q)
    rng = random.Random(SEED)
    out = []
    for site in sorted(by_site):
        qs = by_site[site]
        if len(qs) > cap:
            qs = rng.sample(qs, cap)
        for q in qs:
            out.append({"id": "mfaq_" + hashlib.sha1(q.encode("utf-8")).hexdigest()[:16], "text": q,
                        "n_chars": len(q), "timestamp": "9999-12-31T00:00:00", "source": "mfaq", "site": site})
    return out


def sample(messages, n):
    """n messages at a regular step in time order: coverage of the whole range."""
    messages = sorted(messages, key=lambda m: m["timestamp"])
    step = len(messages) / n
    return [messages[int(i * step)] for i in range(n)]


def build(code):
    """One language: WildChat, completed if needed by OASST2, Aya then MFAQ -> 12,000 + gold.

    The sources are taken in full, in priority order; only the one that
    overflows the need is sampled down, to the shortfall. This way all the
    real chat is kept, and MFAQ only fills the gap.
    """
    t0 = time.perf_counter()
    dd = Deduplicator()
    needed = N_TRAIN + N_EVAL
    stats = {"language": LANGUAGES[code], "sources": {}}
    pool = []
    sources = [("wildchat", lambda: [dict(json.loads(l), source="wildchat")
                                     for l in io.open(os.path.join(CANDIDATES, code + ".jsonl"), encoding="utf-8")]),
               ("oasst2", lambda: read_oasst2(code)), ("aya", lambda: read_aya(code)), ("mfaq", lambda: read_mfaq(code))]
    for name, read in sources:
        if len(pool) >= needed:
            break
        candidates = read()
        kept = dd.filter(candidates)
        taken = kept if len(pool) + len(kept) <= needed else sample(kept, needed - len(pool))
        pool.extend(taken)
        stats["sources"][name] = {"candidates": len(candidates), "kept": len(kept), "taken": len(taken)}
    stats.update({"candidates": sum(s["candidates"] for s in stats["sources"].values()),
                  "after_deduplication": sum(s["kept"] for s in stats["sources"].values()),
                  "exact_duplicates": dd.exact, "near_duplicates": dd.near})

    n_eval = min(N_EVAL, len(pool) - N_TRAIN)
    if n_eval < N_EVAL_MIN:
        # not enough for 12,000: write out what there is anyway, marked
        # "partial", for the group models; a small reserve for a gold set
        n_eval = min(300, len(pool) // 10)
        stats["shortfall"] = N_TRAIN + N_EVAL_MIN - len(pool)
    if True:
        chosen = list(pool)
        random.Random(SEED).shuffle(chosen)
        eval_pool, train = chosen[:n_eval], chosen[n_eval:]
        assert len(train) == (N_TRAIN if "shortfall" not in stats else len(pool) - n_eval) and len(eval_pool) == n_eval
        for name, items in ((code + "_messages.jsonl", train), (code + "_eval_pool.jsonl", eval_pool)):
            with io.open(os.path.join(OUT, name), "w", encoding="utf-8") as fh:
                for m in items:
                    fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        buckets = [("short", 20, 100), ("medium", 100, 400), ("long", 400, 1500), ("very_long", 1500, 8001)]
        stats.update({"train": len(train), "eval_pool": len(eval_pool),
                      "train_by_source": dict(Counter(m["source"] for m in train)),
                      "train_by_bucket": {n: sum(1 for m in train if lo <= m["n_chars"] < hi) for n, lo, hi in buckets}})
    stats["duration_s"] = round(time.perf_counter() - t0)
    with io.open(os.path.join(OUT, code + "_build_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("codes", nargs="*", help="language codes (default: all)")
    ap.add_argument("--candidates", action="store_true", help="only redo the reading pass")
    ap.add_argument("--skip-read", action="store_true", help="reuse the already-extracted candidates")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing corpus")
    ap.add_argument("--short", action="store_true", help="only extract the short messages (< 20 characters)")
    args = ap.parse_args()
    codes = args.codes or list(LANGUAGES)
    unknown = [c for c in codes if c not in LANGUAGES]
    if unknown:
        raise SystemExit("unknown codes: %s (known: %s)" % (unknown, " ".join(LANGUAGES)))
    os.makedirs(OUT, exist_ok=True)
    if args.short:
        pass_short(codes)
        return
    # a corpus already titled by the teacher must never change under its titles
    existing = [c for c in codes if os.path.exists(os.path.join(OUT, c + "_messages.jsonl"))]
    if existing and not args.overwrite and not args.candidates:
        raise SystemExit("corpora already present: %s — rerun without these codes, or with --overwrite" % " ".join(existing))

    if not args.skip_read:
        print("1. one pass over %s for %d languages" % (SOURCE, len(codes)), flush=True)
        pass_candidates(codes)
    if args.candidates:
        return

    print("2. per language: WildChat then OASST2 then Aya if needed, deduplication, drawing %d + %d" % (N_TRAIN, N_EVAL), flush=True)
    rows = []
    for code in codes:
        s = build(code)
        status = "OK" if "shortfall" not in s else "partial (short %s)" % format(s["shortfall"], ",")
        src = " ".join("%s %s" % (k, format(v["kept"], ",")) for k, v in s["sources"].items())
        print("  %-3s %-11s %-45s -> %s (%ds)" % (code, s["language"], src, status, s["duration_s"]), flush=True)
        rows.append((code, s, status))

    with io.open(os.path.join(OUT, "languages_summary.md"), "w", encoding="utf-8") as fh:
        fh.write("# Corpus by language\n\nWildChat-4.8M first, completed by OASST2, Aya then MFAQ only if there's a shortfall. "
                 "Per source: available after deduplication -> taken.\n\n"
                 "| code | language | WildChat | OASST2 | Aya | MFAQ | train | gold | status |\n|---|---|---|---|---|---|---|---|---|\n")
        for code, s, status in rows:
            def g(k):
                if k not in s["sources"]:
                    return "—"
                v = s["sources"][k]
                return format(v["kept"], ",") if v["taken"] == v["kept"] else "%s -> %s" % (format(v["kept"], ","), format(v["taken"], ","))
            fh.write("| %s | %s | %s | %s | %s | %s | %s | %s | %s |\n" % (
                code, s["language"], g("wildchat"), g("oasst2"), g("aya"), g("mfaq"),
                format(s["train"], ","), format(s["eval_pool"], ","), status))
    print("-> %s" % os.path.relpath(os.path.join(OUT, "languages_summary.md")))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
