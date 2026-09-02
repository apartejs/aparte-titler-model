"""Trains the tagger's BPE tokenizer on the message corpus.

Algorithm: the `tokenizers` library (Rust). Vocabulary: our own, learned on
our 50,000 messages. A generic vocabulary (GPT-2: 50,257 entries) would cost
6.4M parameters in the embedding table alone, more than the entire model.

Choices, and their reasons:
  - ByteLevel: never an unknown. Chinese, Cyrillic, and emoji (1% of the
    corpus) pass through as bytes instead of producing a useless <unk> token.
  - ByteLevel pre_tokenizer: pre-splits on whitespace and punctuation BEFORE
    the merges, so no token ever straddles two words. Essential here: we must
    be able to recompose a title from the kept tokens.
  - no lowercasing: we restitute positions in the original text, and
    `US` != `us`.

    python train/train_tokenizer.py --vocab 4096
"""
import argparse
import io
import json
import os
import time

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MESSAGES = os.path.join(ROOT, "data", "canonical", "train_messages.jsonl")
GOLD = os.path.join(ROOT, "eval", "gold.jsonl")
MAX_CHARS = 1200

SPECIAL_TOKENS = ["<pad>", "<unk>"]


def messages(path, limit=0):
    out = []
    for line in io.open(path, encoding="utf-8"):
        m = json.loads(line)
        out.append(m["text"][:MAX_CHARS])
        if limit and len(out) >= limit:
            break
    return out


def train(texts, vocab):
    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    trainer = trainers.BpeTrainer(
        vocab_size=vocab, special_tokens=SPECIAL_TOKENS, show_progress=False,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return tok


def measure(tok, texts, name):
    n_tokens, n_chars, lengths = 0, 0, []
    for t in texts:
        ids = tok.encode(t).ids
        n_tokens += len(ids)
        n_chars += len(t)
        lengths.append(len(ids))
    lengths.sort()
    q = lambda p: lengths[min(len(lengths) - 1, int(p * len(lengths)))]
    print("  %-12s %.2f characters/token | tokens per message: median %d, p95 %d, max %d"
          % (name, n_chars / n_tokens, q(0.5), q(0.95), lengths[-1]))
    return q(0.95), lengths[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vocab", type=int, default=4096)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--messages", nargs="+", default=[MESSAGES], help="one or more jsonl files of messages (e.g. EN + FR)")
    p.add_argument("--out", default=os.environ.get("TITLER_TOKENIZER", os.path.join(HERE, "tokenizer.json")))
    args = p.parse_args()

    texts = []
    for path in args.messages:
        texts += messages(path, args.limit)
    print("corpus: %s messages" % format(len(texts), ","))
    t0 = time.perf_counter()
    tok = train(texts, args.vocab)
    print("BPE vocab=%d trained in %.1fs" % (args.vocab, time.perf_counter() - t0))

    print("\ncompression:")
    measure(tok, texts[:5000], "train")
    if os.path.exists(GOLD):
        measure(tok, [json.loads(l)["text"] for l in io.open(GOLD, encoding="utf-8")], "gold")

    print("\nsplits (the rare word gets cut, never unknown):")
    for word in ["the", "python", "sacagewea", "spawn_the_ball", "c++", "log4j2.xml",
                "buisness", "该研究", "Midjourney"]:
        pieces = tok.encode(word).tokens
        print("  %-16s -> %s" % (word, " | ".join(pieces)))

    # exact round-trip: essential, we restitute the original text
    ok = all(tok.decode(tok.encode(t).ids) == t for t in texts[:2000])
    print("\nexact round-trip on 2000 messages: %s" % ("yes" if ok else "NO"))

    tok.save(args.out)
    print("-> %s (%.0f KB)" % (os.path.relpath(args.out, ROOT), os.path.getsize(args.out) / 1024))


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
