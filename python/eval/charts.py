"""The charts of the model card, generated from scores.json (see matrix.py).

  1. score_vs_size.png   score against size (log), the shipped variants vs an LLM
  2. per_language.png    bars per language: every variant, the LLM, the floor
  3. matrix.png          heat map: every shipped model on every language
  4. table.md            the same numbers as a Markdown table

    python charts.py --scores scores.json --out charts/
"""
import argparse
import io
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LANGUAGES = "en fr es de pt it nl pl sv da fi cs ro no hu hr lt".split()
NAMES = {"en": "English", "fr": "French", "es": "Spanish", "de": "German", "pt": "Portuguese", "it": "Italian",
         "nl": "Dutch", "pl": "Polish", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "cs": "Czech",
         "ro": "Romanian", "no": "Norwegian", "hu": "Hungarian", "hr": "Croatian", "lt": "Lithuanian"}
CHAT = ["en", "fr", "es", "de", "pt", "it", "nl", "pl"]     # real-chat gold sets (the others are mostly FAQ questions)
LLM_BYTES = 2.4e9                                            # gemma 4 e2b, Q4_K_M
COLORS = {"mono": "#4C78A8", "efigsp": "#F58518", "latin-mini": "#54A24B", "latin": "#B279A2",
          "teacher": "#E45756", "floor": "#9D9D9D"}
GROUPS = ["efigsp", "latin-mini", "latin"]


def covered(s, name, lang):
    langs = {"efigsp": ["en", "fr", "es", "de", "pt", "it"]}.get(name, LANGUAGES)
    return lang in langs


def score_vs_size(s, out):
    def mean(f, langs):
        v = [f(c) for c in langs]
        return sum(v) / len(v), min(v), max(v)
    points = [("one model per language", 40222, COLORS["mono"], mean(lambda c: s["models"][c]["scores"][c], CHAT))]
    for g in GROUPS:
        langs = [c for c in CHAT if covered(s, g, c)]
        points.append(("%s (%d languages)" % (g, 6 if g == "efigsp" else 17), s["models"][g]["bytes_int3"], COLORS[g],
                       mean(lambda c: s["models"][g]["scores"][c], langs)))
    if s["teacher"]:
        points.append(("a small LLM (gemma 4 e2b)", LLM_BYTES, COLORS["teacher"], mean(lambda c: s["teacher"][c], CHAT)))
    fl = mean(lambda c: s["floor"][c], CHAT)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axhspan(fl[1], fl[2], color=COLORS["floor"], alpha=0.15)
    ax.axhline(fl[0], color=COLORS["floor"], ls="--", lw=1)
    ax.text(2.5e4, fl[0] + 0.006, "first 5 words, no model", color=COLORS["floor"], fontsize=9)
    for name, size, color, (m, lo, hi) in points:
        label = "%s — %s, F1 %.2f" % (name, "%d KB" % round(size / 1000) if size < 1e6 else "2.4 GB", m)
        ax.errorbar([size], [m], yerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=9, capsize=4, lw=1.5, label=label)
        if size > 1e6:
            ax.annotate("an LLM call\n2.4 GB, seconds per title", (size, m), textcoords="offset points", xytext=(-12, 10), ha="right", fontsize=9, color=color)
    ax.annotate("in the browser\n40-133 KB, 6-7 ms per title", (1.3e5, 0.64), fontsize=9, color="#555555")
    ax.legend(loc="center", fontsize=9, title="aparte-titler v1 (int3 files) vs the LLM it replaces", title_fontsize=9)
    ax.set_xscale("log")
    ax.set_xlim(2e4, 3e10)
    ax.set_ylim(0.2, 0.8)
    ticks = [(1e4, "10 KB"), (1e5, "100 KB"), (1e6, "1 MB"), (1e7, "10 MB"), (1e8, "100 MB"), (1e9, "1 GB"), (1e10, "10 GB")]
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([t[1] for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    for x, name in [(4e4, "an app icon"), (3e6, "a photo"), (7e8, "a movie in HD")]:
        ax.axvline(x, color="#BBBBBB", lw=0.8, ls=":")
        ax.text(x, 0.215, " " + name, color="#777777", fontsize=8, rotation=90, va="bottom", ha="left")
    ax.set_xlabel("model weight (log scale)")
    ax.set_ylabel("word F1 vs the gold title, mean over the 8 real-chat languages (bar = min-max)")
    ax.set_title("aparte-titler v1 — conversation titles in the browser: the weight of an icon, 92% of an LLM's score, no API call")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "score_vs_size.png"), dpi=130)
    plt.close(fig)


def per_language(s, out):
    series = [("mono", "one model per language (40 KB)", lambda c: s["models"][c]["scores"][c]),
              ("efigsp", "efigsp, 6 languages (77 KB)", lambda c: s["models"]["efigsp"]["scores"][c] if covered(s, "efigsp", c) else None),
              ("latin-mini", "latin-mini, 17 languages (96 KB)", lambda c: s["models"]["latin-mini"]["scores"][c]),
              ("latin", "latin, 17 languages (133 KB)", lambda c: s["models"]["latin"]["scores"][c]),
              ("teacher", "a small LLM, gemma 4 e2b (2.4 GB)", lambda c: s["teacher"].get(c)),
              ("floor", "first 5 words (no model)", lambda c: s["floor"][c])]
    fig, ax = plt.subplots(figsize=(17, 5))
    x = np.arange(len(LANGUAGES))
    w = 0.13
    for i, (key, label, f) in enumerate(series):
        vals = [f(c) for c in LANGUAGES]
        ax.bar(x + (i - 2.5) * w, [v if v is not None else 0 for v in vals], w, label=label, color=COLORS[key])
    ax.set_xticks(x)
    ax.set_xticklabels(["%s\n%s" % (NAMES[c], c) for c in LANGUAGES], fontsize=8)
    ax.set_ylabel("word F1 vs the gold title (budget 6)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("aparte-titler v1 — per language, every shipped variant, against an LLM and against no model at all")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_language.png"), dpi=130)
    plt.close(fig)


def matrix(s, out):
    names = [n for n in LANGUAGES if n in s["models"]] + [g for g in GROUPS if g in s["models"]]
    M = np.array([[s["models"][n]["scores"][c] for c in LANGUAGES] for n in names])
    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(M, cmap="viridis", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(LANGUAGES)))
    ax.set_xticklabels(LANGUAGES)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(["%s (%d KB)" % (n, round(s["models"][n]["bytes_int3"] / 1000)) for n in names], fontsize=8)
    for i, n in enumerate(names):
        for j, c in enumerate(LANGUAGES):
            trained = (n == c) or (n in GROUPS and covered(s, n, c))
            ax.text(j, i, "%.2f" % M[i, j], ha="center", va="center", fontsize=6.5,
                    color="white" if M[i, j] < 0.62 else "black", fontweight="bold" if trained else "normal")
    ax.set_xlabel("gold set (language of the test messages)")
    ax.set_ylabel("shipped model (int3)")
    ax.set_title("aparte-titler v1 — every model on every language (bold = trained on that language)")
    fig.colorbar(im, ax=ax, label="word F1 vs the gold title")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "matrix.png"), dpi=130)
    plt.close(fig)


def table(s, out):
    lines = ["| language | one model (40 KB) | efigsp (77 KB) | latin-mini (96 KB) | latin (133 KB) | gemma 4 e2b (2.4 GB) | first 5 words |",
             "|---|---|---|---|---|---|---|"]
    for c in LANGUAGES:
        lines.append("| %s | %.3f | %s | %.3f | %.3f | %s | %.3f |" % (
            NAMES[c], s["models"][c]["scores"][c], "%.3f" % s["models"]["efigsp"]["scores"][c] if covered(s, "efigsp", c) else "—",
            s["models"]["latin-mini"]["scores"][c], s["models"]["latin"]["scores"][c],
            "%.3f" % s["teacher"][c] if c in s["teacher"] else "—", s["floor"][c]))
    with io.open(os.path.join(out, "table.md"), "w", encoding="utf-8") as fh:
        fh.write("Word-level F1 against the gold title (single reference), budget 6, impossibles excluded. "
                 "Sizes are the int3 files.\n\n" + "\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="scores.json")
    p.add_argument("--out", default="charts")
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)
    s = json.load(io.open(args.scores, encoding="utf-8"))
    score_vs_size(s, args.out)
    per_language(s, args.out)
    matrix(s, args.out)
    table(s, args.out)
    print("-> %s: score_vs_size.png, per_language.png, matrix.png, table.md" % args.out)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
