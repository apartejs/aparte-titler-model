"""Short messages (< 20 characters) labeled "keep everything".

A three-word message is its own title. Rather than a rule at decoding time,
these examples go into training: every token that overlaps a word is worth 1,
punctuation alone is worth 0. The format is the same as labels.py (ids,
labels, words), for the same tokenizer.

    TITLER_TOKENIZER=... python train/short_messages.py <texts.jsonl> <output.jsonl>
"""
import io
import json
import os
import re
import sys

from tokenizers import Tokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from labels import label_tokens, word_ids_per_token  # noqa: E402

WORD = re.compile(r"[^\W_]+")


def main(input_path, output_path):
    tok = Tokenizer.from_file(os.environ["TITLER_TOKENIZER"])
    n, read_n = 0, 0
    with io.open(output_path, "w", encoding="utf-8") as fh:
        for l in io.open(input_path, encoding="utf-8"):
            r = json.loads(l)
            read_n += 1
            text = r["text"].strip()
            spans = [(m.start(), m.end()) for m in WORD.finditer(text)]
            if not spans:
                continue
            ids, labels, offsets = label_tokens(tok, text, spans)
            if sum(labels) == 0:
                continue
            fh.write(json.dumps({"id": r["id"], "title": text, "ids": ids, "labels": labels,
                                 "words": word_ids_per_token(text, offsets)}, ensure_ascii=False) + "\n")
            n += 1
    print("%d/%d short messages labeled -> %s" % (n, read_n, os.path.relpath(output_path)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main(sys.argv[1], sys.argv[2])
