"""A group model: several languages, one shared tokenizer, a single tagger.

For each language: labels (+ short messages) and consensus with the group
tokenizer, via the pilot. Then the union of the sets (each language capped at
CAP examples so that English doesn't drown out the others), training, and the
score on each language's gold set. The tokenizer must already exist
(train_tokenizer.py --vocab N --messages <one corpus per language>).

    python -u train/group.py latin_12288 en fr es de pt it nl pl sv da fi cs ro no hu hr lt
"""
import io
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable   # the interpreter running this script, on every platform
CANON = os.path.join(ROOT, "data", "canonical")
EVAL = os.path.join(ROOT, "eval")
RUNS = os.path.join(ROOT, "runs")
CAP = 12_000


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M"), msg), flush=True)


def main():
    tok_name, codes = sys.argv[1], sys.argv[2:]
    tok = os.path.join(HERE, "tokenizer_%s.json" % tok_name)
    if not os.path.exists(tok):
        raise SystemExit("missing tokenizer: " + tok)
    epochs = os.environ.get("TITLER_EPOCHS", "8")

    for code in codes:
        if os.path.exists(os.path.join(CANON, "%s_train_%s.jsonl" % (code, tok_name))) and not os.environ.get("TITLER_REDO"):
            log("%s: labels already there, keeping them" % code)
            continue
        steps = ["labels"] + (["consensus"] if os.path.exists(os.path.join(EVAL, "%s_gold.jsonl" % code)) else [])
        log("%s : %s" % (code, " ".join(steps)))
        r = subprocess.run([PY, os.path.join(HERE, "pipeline_language.py"), code, "--tokenizer", tok_name, "--steps"] + steps,
                           cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log("FAILED %s (%d)" % (code, r.returncode))

    train = os.path.join(CANON, "%s_train.jsonl" % tok_name)
    total = 0
    with io.open(train, "w", encoding="utf-8") as out:
        for code in codes:
            f = os.path.join(CANON, "%s_train_%s.jsonl" % (code, tok_name))
            if not os.path.exists(f):
                log("no labels for %s: language missing" % code)
                continue
            n = 0
            for l in io.open(f, encoding="utf-8"):
                if n >= CAP:
                    break
                out.write(l)
                n += 1
            total += n
            log("%s : %d examples" % (code, n))
    log("%d examples in total" % total)

    name = "%s_emb8" % tok_name
    env = dict(os.environ, TITLER_TOKENIZER=tok, TITLER_LABELED=train, TITLER_GOLD=os.path.join(EVAL, "gold.jsonl"),
               TITLER_CONSENSUS=os.path.join(EVAL, "en_gold_consensus_%s.jsonl" % tok_name))
    log("training %s (%s epochs)" % (name, epochs))
    with io.open(os.path.join(RUNS, name + ".log"), "w", encoding="utf-8") as fh:
        subprocess.run([PY, os.path.join(HERE, "train.py"), "--name", name, "--layers", "2", "--dim", "32",
                        "--dim-emb", "8", "--epochs", epochs], env=env, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    for l in io.open(os.path.join(RUNS, name + ".log"), encoding="utf-8"):
        if "parameters" in l:
            log(l.strip())

    log("scores per gold set (budget 6, max over the annotators):")
    for code in codes:
        gold = os.path.join(EVAL, "%s_gold.jsonl" % code)
        if not os.path.exists(gold):
            continue
        r = subprocess.run([PY, os.path.join(EVAL, "evaluate_run.py"), os.path.join(RUNS, name + ".pt"), "--tokenizer", tok,
                            "--gold", gold, "--consensus", os.path.join(EVAL, "%s_gold_consensus_%s.jsonl" % (code, tok_name))],
                           cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
        line = [l for l in r.stdout.splitlines() if "annotators" in l]
        print("  %s : %s" % (code, line[-1].split("annotators")[-1].strip() if line else "?"), flush=True)
    subprocess.run([PY, os.path.join(HERE, "export.py"), os.path.join(RUNS, name + ".pt"), "--bits", "3", "--tokenizer", tok], cwd=ROOT)
    log("group %s done" % tok_name)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
