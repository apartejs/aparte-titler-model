"""Rewrite the results tables of the README and of the model card from scores.json,
so that no number is ever typed by hand.

    python tables.py --scores ../../../hf-model/scores.json --readme ../../README.md --card ../../../hf-model/README.md

Three tables, each found in the Markdown by its header row:

  | language | one model ...    scores per language and scope, current version
  | scope | 1.0 | 1.1 | 1.2    the same scope across versions, same benchmark

scores.json comes from matrix.py over the published int3 files. The current
version is keyed by its bare scope ("latin"), older ones as "<scope>@<version>"
("latin@1.1").
"""
import argparse
import io
import json
import re
import statistics as st

LANGUAGES = "en fr es de pt it nl pl sv da fi cs ro no hu hr lt".split()
NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese", "it": "Italian",
         "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "cs": "Czech",
         "ro": "Romanian", "no": "Norwegian", "hu": "Hungarian", "hr": "Croatian", "lt": "Lithuanian"}
EFIGSP = ["en", "fr", "es", "de", "pt", "it"]
GROUPS = {"efigsp": EFIGSP, "latin-mini": LANGUAGES, "latin": LANGUAGES}
OLDER = ["1", "1.1"]


def kb(n):
    return "%d KB" % round(n / 1000)


def own(m, key, scope):
    """A model on its own language(s): one language, or the mean over a group's."""
    e = m.get(key)
    if not e:
        return None
    langs = GROUPS.get(scope, [scope])
    return st.mean(e["scores"][c] for c in langs if c in e["scores"])


def readme_table(s):
    m = s["models"]
    lines = ["| language | one model (%s) | latin (%s) | no model |" % (kb(m["fr"]["bytes_int3"]), kb(m["latin"]["bytes_int3"])),
             "|---|---|---|---|"]
    for c in LANGUAGES:
        lines.append("| %s | %.3f | %.3f | %.3f |" % (NAMES[c], m[c]["scores"][c], m["latin"]["scores"][c], s["floor"][c]))
    return "\n".join(lines)


def card_table(s):
    m = s["models"]
    lines = ["| language | one model (%s) | efigsp (%s) | latin-mini (%s) | latin (%s) | no model |"
             % (kb(m["fr"]["bytes_int3"]), kb(m["efigsp"]["bytes_int3"]), kb(m["latin-mini"]["bytes_int3"]), kb(m["latin"]["bytes_int3"])),
             "|---|---|---|---|---|---|"]
    for c in LANGUAGES:
        lines.append("| %s | %.3f | %s | %.3f | %.3f | %.3f |" % (
            NAMES[c], m[c]["scores"][c], "%.3f" % m["efigsp"]["scores"][c] if c in EFIGSP else "—",
            m["latin-mini"]["scores"][c], m["latin"]["scores"][c], s["floor"][c]))
    return "\n".join(lines)


def versions_table(s):
    """Every scope at 1.0, 1.1 and the current version, all on the same benchmark."""
    m = s["models"]
    lines = ["| scope | 1.0 | 1.1 | 1.2 | 1.2 − 1.1 |", "|---|---|---|---|---|"]
    gaps = []
    for scope in LANGUAGES + list(GROUPS):
        v = [own(m, "%s@%s" % (scope, o), scope) for o in OLDER] + [own(m, scope, scope)]
        cells = ["%.4f" % x if x is not None else "—" for x in v]
        gap = v[2] - v[1] if None not in (v[1], v[2]) else None
        if gap is not None:
            gaps.append(gap)
        name = NAMES.get(scope, scope)
        lines.append("| %s | %s | %s | %s | %s |" % (name, cells[0], cells[1], cells[2],
                                                   "%+.4f" % gap if gap is not None else "—"))
    lines.append("| **mean** | | | | **%+.4f** |" % st.mean(gaps))
    return "\n".join(lines)


def replace_table(text, header_start, table):
    """Replace the Markdown table whose header row starts with `header_start`."""
    pattern = re.compile(r"^" + re.escape(header_start) + r".*?(?=\n\n|\Z)", re.S | re.M)
    m = pattern.search(text)
    assert m, "table not found: " + header_start
    return text[:m.start()] + table + text[m.end():]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", required=True)
    p.add_argument("--readme", default="")
    p.add_argument("--card", default="")
    args = p.parse_args()
    s = json.load(io.open(args.scores, encoding="utf-8"))
    if args.readme:
        text = io.open(args.readme, encoding="utf-8").read()
        io.open(args.readme, "w", encoding="utf-8").write(replace_table(text, "| language | one model", readme_table(s)))
        print("tables rewritten in", args.readme)
    if args.card:
        text = io.open(args.card, encoding="utf-8").read()
        text = replace_table(text, "| language | one model", card_table(s))
        text = replace_table(text, "| scope | 1.0 | 1.1 | 1.2", versions_table(s))
        io.open(args.card, "w", encoding="utf-8").write(text)
        print("tables rewritten in", args.card)


if __name__ == "__main__":
    main()
