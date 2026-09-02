"""The pilot for one language: the same steps, in the same order, for each one.

It invents nothing: it calls the existing scripts with the right files.
Prerequisites (done separately): the corpus (`data/languages.py`), the
teacher's titles (`data/generate_titles.py`), the tokenizer
(`train_tokenizer.py`), the gold set (`eval/build_gold.py check` on the annotator's
titles), and the teacher on the gold set (`eval/run_teacher.py`).

Steps, per language code and per tokenizer (by default its own, `<code>_2048`):
  labels     titles -> per-token labels (+ short messages "keep everything")
             -> data/canonical/<code>_train_<tok>.jsonl
  consensus  annotator + teacher on the gold set, per tokenizer
             -> eval/<code>_gold_consensus_<tok>.jsonl
  train      the chosen tagger (2 layers, dim 32, embeddings 8), 10 epochs
             -> runs/<code>_<tok>.pt + .json + .log
  eval       the model on the language's gold set, one and several references

    python train/pipeline_language.py es                        # everything
    python train/pipeline_language.py es --steps labels         # one step
    python train/pipeline_language.py es --tokenizer efigsp_6144 --steps labels consensus
"""
import argparse
import io
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable   # the interpreter running this script, on every platform
CANON = os.path.join(ROOT, "data", "canonical")
EVAL = os.path.join(ROOT, "eval")
RUNS = os.path.join(ROOT, "runs")


def run(cmd, env=None, log=None):
    print("$ " + " ".join(os.path.relpath(c, ROOT) if os.path.isabs(c) and os.path.exists(c) else c for c in cmd), flush=True)
    if log:
        with io.open(log, "w", encoding="utf-8") as fh:
            r = subprocess.run(cmd, env=env, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    else:
        r = subprocess.run(cmd, env=env, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit("failed (%d): %s" % (r.returncode, cmd[1] if len(cmd) > 1 else cmd))


def paths(code, tok):
    c = {
        "tokenizer": os.path.join(HERE, "tokenizer_%s.json" % tok),
        "messages": os.path.join(CANON, "%s_messages.jsonl" % code),
        "titles": os.path.join(CANON, "%s_titles.jsonl" % code),
        "short_texts": os.path.join(CANON, "%s_short_texts.jsonl" % code),
        "labeled": os.path.join(CANON, "%s_labeled_%s.jsonl" % (code, tok)),
        "short_labeled": os.path.join(CANON, "%s_short_labeled_%s.jsonl" % (code, tok)),
        "train": os.path.join(CANON, "%s_train_%s.jsonl" % (code, tok)),
        "todo": os.path.join(EVAL, "%s_gold_todo.jsonl" % code),
        "gold": os.path.join(EVAL, "%s_gold.jsonl" % code),
        "annotator": os.path.join(EVAL, "%s_gold_labels.jsonl" % code),
        # eval/run_teacher.py <gguf> <code>gold-v2 --messages eval/<code>_gold_todo.jsonl --prompt data/prompt_v2.txt
        "teacher_gold": os.path.join(ROOT, "data", "bakeoff", "results", "%sgold-v2__extractive.jsonl" % code),
        "consensus": os.path.join(EVAL, "%s_gold_consensus_%s.jsonl" % (code, tok)),
        "review": os.path.join(EVAL, "%s_gold_review_multi_%s.md" % (code, tok)),
        "name": "%s_%s" % (code, tok),
    }
    c["weights"] = os.path.join(RUNS, c["name"] + ".pt")
    c["log"] = os.path.join(RUNS, c["name"] + ".log")
    return c


def step_labels(c):
    if not os.path.exists(c["tokenizer"]):
        # a missing monolingual tokenizer is trained right here (2,048 merges on the corpus);
        # a group tokenizer must already exist (train_tokenizer.py --messages ...)
        vocab = os.path.basename(c["tokenizer"])[:-5].rsplit("_", 1)[-1]
        if not vocab.isdigit() or "efigsp" in c["tokenizer"]:
            raise SystemExit("missing tokenizer: %s" % os.path.relpath(c["tokenizer"], ROOT))
        run([PY, os.path.join(HERE, "train_tokenizer.py"), "--vocab", vocab, "--messages", c["messages"],
             "--out", c["tokenizer"]])
    env = dict(os.environ, TITLER_TOKENIZER=c["tokenizer"], TITLER_MESSAGES=c["messages"], TITLER_TITLES=c["titles"],
               TITLER_LABELED=c["labeled"], TITLER_GOLD=c["gold"])
    run([PY, os.path.join(HERE, "labels.py")], env)
    parts = [c["labeled"]]
    if os.path.exists(c["short_texts"]):
        run([PY, os.path.join(HERE, "short_messages.py"), c["short_texts"], c["short_labeled"]], dict(os.environ, TITLER_TOKENIZER=c["tokenizer"]))
        parts.append(c["short_labeled"])
    n = 0
    with io.open(c["train"], "w", encoding="utf-8") as out:
        for p in parts:
            for l in io.open(p, encoding="utf-8"):
                out.write(l)
                n += 1
    print("-> %s (%d examples)" % (os.path.relpath(c["train"], ROOT), n), flush=True)


def step_consensus(c):
    for k in ("todo", "gold", "annotator", "teacher_gold"):
        if not os.path.exists(c[k]):
            raise SystemExit("consensus: missing %s" % os.path.relpath(c[k], ROOT))
    env = dict(os.environ, TITLER_TOKENIZER=c["tokenizer"], TITLER_TODO=c["todo"], TITLER_GOLD=c["gold"])
    run([PY, os.path.join(EVAL, "consensus.py"), "annotator=" + c["annotator"], "e2b=" + c["teacher_gold"],
         "--output", c["consensus"], "--review", c["review"]], env)


def step_train(c, epochs):
    gold = c["gold"]
    if not os.path.exists(gold):
        # no gold set for this language: the English gold set stands in as a
        # sanity check at the end (the "gold" numbers in the log are then
        # meaningless; the checkpoint is chosen on validation, not on the gold set)
        gold = os.path.join(EVAL, "gold.jsonl")
        print("  (no gold set %s: final eval on the English gold set, ignore it)" % os.path.basename(c["gold"]), flush=True)
    env = dict(os.environ, TITLER_TOKENIZER=c["tokenizer"], TITLER_LABELED=c["train"], TITLER_GOLD=gold, TITLER_CONSENSUS=c["consensus"])
    run([PY, os.path.join(HERE, "train.py"), "--name", c["name"], "--layers", "2", "--dim", "32", "--dim-emb", "8",
         "--epochs", str(epochs)], env, log=c["log"])
    for l in io.open(c["log"], encoding="utf-8"):
        if "annotators" in l or "budget 6 words" in l or "parameters" in l:
            print("  " + l.rstrip())


def step_eval(c):
    run([PY, os.path.join(EVAL, "evaluate_run.py"), c["weights"], "--tokenizer", c["tokenizer"], "--gold", c["gold"],
         "--consensus", c["consensus"]])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("code")
    p.add_argument("--tokenizer", default="", help="tokenizer name (default: <code>_2048)")
    p.add_argument("--steps", nargs="+", default=["labels", "consensus", "train", "eval"])
    p.add_argument("--epochs", type=int, default=10)
    args = p.parse_args()
    c = paths(args.code, args.tokenizer or args.code + "_2048")
    for e in args.steps:
        print("\n=== %s : %s (tokenizer %s)" % (args.code, e, os.path.basename(c["tokenizer"])), flush=True)
        {"labels": lambda: step_labels(c), "consensus": lambda: step_consensus(c),
         "train": lambda: step_train(c, args.epochs), "eval": lambda: step_eval(c)}[e]()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
