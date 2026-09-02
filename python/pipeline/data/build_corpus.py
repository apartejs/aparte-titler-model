"""Builds the message corpus from a locally downloaded WildChat-1M.

Steps, in order:
  1. filter     : English, non-toxic, first user turn, 20-8000 characters
  2. exact      : strict duplicates (case and whitespace normalized)
  3. minhash    : near-duplicates (copy-pasted templates like "prompt generator
                  for Midjourney", 12% of the bake-off sample)
  4. exclusion  : the 50 bake-off messages
  5. sampling   : uniform over the whole time range (the dataset is sorted by
                  date, taking the first ones = a snapshot of April 2023)

Outputs, disjoint:
  data/canonical/train_messages.jsonl   50,000 messages for title generation
  data/canonical/eval_pool.jsonl         1,000 messages reserved for the gold eval set
  data/canonical/build_stats.json        the counts at each step

    python -u data/build_corpus.py
"""
import glob
import io
import json
import os
import random
import re
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datetime import datetime, timezone
from datasketch import MinHash, MinHashLSH

HERE = os.path.dirname(os.path.abspath(__file__))
# TITLER_WILDCHAT: folder of the parquet files (WildChat-1M by default, WildChat-4.8M for the languages)
PARQUET = sorted(glob.glob(os.path.join(os.environ.get("TITLER_WILDCHAT", os.path.join(HERE, "wildchat")), "data", "*.parquet")))
BAKEOFF = os.path.join(HERE, "bakeoff", "messages.jsonl")
OUT = os.path.join(HERE, "canonical")

N_TRAIN = 50_000
N_EVAL_POOL = 1_000
LANGUAGE = "English"
MIN_CHARS, MAX_CHARS = 20, 8_000
SEED = 20260901

# MinHash: 3-word shingles, threshold 0.8. Wide enough to catch a template
# copied with one variable changed, narrow enough to keep two genuinely
# different questions on the same subject.
NUM_PERM = 128
THRESHOLD = 0.8
# letters and digits from every script: in ASCII, `é` split French words and
# the shingles only ever saw fragments
WORD = re.compile(r"[^\W_]+")


def normalize(text):
    return " ".join(text.lower().split())


def shingles(text):
    words = WORD.findall(text.lower())
    if len(words) < 3:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + 3]) for i in range(len(words) - 2)}


def minhash(text):
    m = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        m.update(s.encode("utf-8"))
    return m


def read_messages(language=LANGUAGE):
    """One dict per retained conversation, without ever loading a whole parquet file."""
    columns = ["conversation_hash", "timestamp", "conversation", "language", "toxic"]
    stats = {"read": 0, "english_non_toxic": 0, "first_user_turn": 0, "length_ok": 0}
    for f in PARQUET:
        pf = pq.ParquetFile(f)
        for batch in pf.iter_batches(batch_size=4096, columns=columns):
            # timestamp with a timezone: converting it in Python needs a
            # timezone database that's absent on Windows. The epoch in
            # microseconds doesn't need one.
            i_ts = batch.schema.get_field_index("timestamp")
            epoch = pc.cast(batch.column(i_ts), pa.int64())
            batch = batch.set_column(i_ts, "timestamp", epoch)
            for row in batch.to_pylist():
                stats["read"] += 1
                if row["language"] != language or row["toxic"]:
                    continue
                stats["english_non_toxic"] += 1
                conv = row["conversation"] or []
                if not conv or conv[0].get("role") != "user":
                    continue
                stats["first_user_turn"] += 1
                text = (conv[0].get("content") or "").strip()
                if not MIN_CHARS <= len(text) <= MAX_CHARS:
                    continue
                stats["length_ok"] += 1
                yield {"id": row["conversation_hash"], "text": text,
                       "n_chars": len(text),
                       "timestamp": datetime.fromtimestamp(
                           row["timestamp"] / 1e6, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")}
        print("  %s : %s read" % (os.path.basename(f), format(stats["read"], ",")), flush=True)
    read_messages.stats = stats


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="English")
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-eval", type=int, default=N_EVAL_POOL)
    ap.add_argument("--prefix", default="train", help="output names: <prefix>_messages.jsonl, <prefix>_eval_pool.jsonl")
    args = ap.parse_args()
    t0 = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)

    print("1. reading + filters")
    messages = list(read_messages(args.language))
    stats = dict(read_messages.stats)
    print("   -> %s candidate messages (%.0fs)" % (format(len(messages), ","), time.perf_counter() - t0), flush=True)

    print("2. exact duplicates")
    seen, unique = set(), []
    for m in messages:
        key = normalize(m["text"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(m)
    stats["after_exact"] = len(unique)
    print("   -> %s (%s removed)" % (format(len(unique), ","), format(len(messages) - len(unique), ",")), flush=True)

    print("3. MinHash near-duplicates (threshold %.1f, %d permutations)" % (THRESHOLD, NUM_PERM))
    t1 = time.perf_counter()
    lsh = MinHashLSH(threshold=THRESHOLD, num_perm=NUM_PERM)
    kept, near_duplicates = [], 0
    for i, m in enumerate(unique):
        mh = minhash(m["text"])
        if lsh.query(mh):
            near_duplicates += 1
            continue
        lsh.insert(m["id"], mh)
        kept.append(m)
        if i and i % 50_000 == 0:
            print("   ... %s processed, %s kept (%.0fs)" % (format(i, ","), format(len(kept), ","), time.perf_counter() - t1), flush=True)
    stats["after_minhash"] = len(kept)
    stats["near_duplicates_removed"] = near_duplicates
    print("   -> %s (%s near-duplicates removed, %.0fs)" % (format(len(kept), ","), format(near_duplicates, ","), time.perf_counter() - t1), flush=True)

    print("4. excluding the 50 bake-off messages")
    bakeoff_ids = {json.loads(l)["id"] for l in io.open(BAKEOFF, encoding="utf-8")}
    bakeoff_texts = {normalize(json.loads(l)["text"]) for l in io.open(BAKEOFF, encoding="utf-8")}
    pool = [m for m in kept if m["id"] not in bakeoff_ids and normalize(m["text"]) not in bakeoff_texts]
    stats["after_bakeoff_exclusion"] = len(pool)
    print("   -> %s" % format(len(pool), ","), flush=True)

    print("5. uniform sampling over the whole time range")
    pool.sort(key=lambda m: m["timestamp"])
    rng = random.Random(SEED)
    needed = args.n_train + args.n_eval
    if len(pool) < needed:
        raise SystemExit("pool too small: %d < %d" % (len(pool), needed))
    # a regular-step draw without replacement preserves the time coverage,
    # the final shuffle then breaks the chronological order in the file
    step = len(pool) / needed
    chosen = [pool[int(i * step)] for i in range(needed)]
    rng.shuffle(chosen)
    eval_pool, train = chosen[:args.n_eval], chosen[args.n_eval:]

    def write(name, items):
        path = os.path.join(OUT, name)
        with io.open(path, "w", encoding="utf-8") as fh:
            for m in items:
                fh.write(json.dumps(m, ensure_ascii=False) + "\n")
        return path

    write(args.prefix + "_messages.jsonl", train)
    write(args.prefix + "_eval_pool.jsonl", eval_pool)

    buckets = [("short", 20, 100), ("medium", 100, 400), ("long", 400, 1500), ("very_long", 1500, 8001)]
    stats["train_by_bucket"] = {name: sum(1 for m in train if lo <= m["n_chars"] < hi) for name, lo, hi in buckets}
    stats["train_first_timestamp"] = min(m["timestamp"] for m in train)
    stats["train_last_timestamp"] = max(m["timestamp"] for m in train)
    stats["train"] = len(train)
    stats["eval_pool"] = len(eval_pool)
    stats["duration_s"] = round(time.perf_counter() - t0)
    with io.open(os.path.join(OUT, args.prefix + "_build_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    print("\nsummary:")
    for k, v in stats.items():
        print("  %-28s %s" % (k, v))


if __name__ == "__main__":
    main()
