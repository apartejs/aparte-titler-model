"""Rewrite the results tables of the README and of the model card from scores.json,
so that no number is ever typed by hand.

    python tables.py --scores ../../../hf-model/scores.json --readme ../../README.md --card ../../../hf-model/README.md
"""
import argparse
import io
import json
import re

LANGUAGES = "en fr es de pt it nl pl sv da fi cs ro no hu hr lt".split()
NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese", "it": "Italian",
         "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "cs": "Czech",
         "ro": "Romanian", "no": "Norwegian", "hu": "Hungarian", "hr": "Croatian", "lt": "Lithuanian"}
EFIGSP = ["en", "fr", "es", "de", "pt", "it"]


def kb(n):
    return "%d KB" % round(n / 1000)


def readme_table(s):
    m = s["models"]
    lines = ["| language | one model (%s) | latin (%s) | LLM (2.4 GB) | no model |" % (kb(m["fr"]["bytes_int3"]), kb(m["latin"]["bytes_int3"])),
             "|---|---|---|---|---|"]
    for c in LANGUAGES:
        lines.append("| %s | %.3f | %.3f | %.3f | %.3f |" % (NAMES[c], m[c]["scores"][c], m["latin"]["scores"][c], s["teacher"][c], s["floor"][c]))
    return "\n".join(lines)


def card_table(s):
    m = s["models"]
    lines = ["| language | one model (%s) | efigsp (%s) | latin-mini (%s) | latin (%s) | LLM (2.4 GB) | no model |"
             % (kb(m["fr"]["bytes_int3"]), kb(m["efigsp"]["bytes_int3"]), kb(m["latin-mini"]["bytes_int3"]), kb(m["latin"]["bytes_int3"])),
             "|---|---|---|---|---|---|---|"]
    for c in LANGUAGES:
        lines.append("| %s | %.3f | %s | %.3f | %.3f | %.3f | %.3f |" % (
            NAMES[c], m[c]["scores"][c], "%.3f" % m["efigsp"]["scores"][c] if c in EFIGSP else "—",
            m["latin-mini"]["scores"][c], m["latin"]["scores"][c], s["teacher"][c], s["floor"][c]))
    return "\n".join(lines)


def replace_table(text, header_start, table):
    """Replace the Markdown table whose header starts with `header_start`."""
    pattern = re.compile(r"^\| language \| one model.*?(?=\n\n)", re.S | re.M)
    m = pattern.search(text)
    assert m, "table not found"
    return text[:m.start()] + table + text[m.end():]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--readme", default="")
    p.add_argument("--card", default="")
    args = p.parse_args()
    s = json.load(io.open(args.scores, encoding="utf-8"))
    for path, table in ((args.readme, readme_table(s)), (args.card, card_table(s))):
        if not path:
            continue
        text = io.open(path, encoding="utf-8").read()
        io.open(path, "w", encoding="utf-8").write(replace_table(text, "| language | one model", table))
        print("table rewritten in", path)


if __name__ == "__main__":
    main()
