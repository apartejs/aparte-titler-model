"""The batch: all languages, one after another, without spending a single API token.

For each language: waits for its titles to finish (the teacher's queue),
builds the gold set if the annotator delivered one (`build_gold.py check`), then
drives it with its own tokenizer (labels, consensus if there's a gold set,
training, eval) and, for languages in the EFIGS+P group, the labels with the
group tokenizer as well. At the end, the EFIGS+P group model (English capped
at 12,000 examples) is evaluated against every available gold set.

    python -u train/batch_languages.py > runs/batch.log 2>&1
"""
import glob
import io
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable   # the interpreter running this script, on every platform
CANON = os.path.join(ROOT, "data", "canonical")
EVAL = os.path.join(ROOT, "eval")
RUNS = os.path.join(ROOT, "runs")
RESULTS = os.path.join(ROOT, "data", "bakeoff", "results")
LANGUAGES = "es de pt it nl pl sv da fi cs ro no hu hr lt".split()
GROUP = ["en", "fr", "es", "de", "pt", "it"]
GROUP_TOKENIZER = "efigsp_6144"
CAP_EN = 12_000


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M"), msg), flush=True)


def run(cmd, env=None):
    log("$ " + " ".join(os.path.basename(x) if os.sep in x else x for x in cmd))
    r = subprocess.run(cmd, env=env, cwd=ROOT)
    if r.returncode != 0:
        log("FAILED (%d), continuing" % r.returncode)
    return r.returncode == 0


def prepare_en():
    """English was done before the pilot existed, under different names: copy them over."""
    copies = [(os.path.join(CANON, "train_messages.jsonl"), os.path.join(CANON, "en_messages.jsonl")),
              (os.path.join(CANON, "train_titles_v2_accord.jsonl"), os.path.join(CANON, "en_titles.jsonl")),
              (os.path.join(CANON, "short_texts.jsonl"), os.path.join(CANON, "en_short_texts.jsonl")),
              (os.path.join(EVAL, "gold_todo.jsonl"), os.path.join(EVAL, "en_gold_todo.jsonl")),
              (os.path.join(EVAL, "gold.jsonl"), os.path.join(EVAL, "en_gold.jsonl")),
              (os.path.join(EVAL, "gold_labels.jsonl"), os.path.join(EVAL, "en_gold_labels.jsonl")),
              (os.path.join(RESULTS, "teacher-gold-v2__extractive.jsonl"), os.path.join(RESULTS, "engold-v2__extractive.jsonl")),
              (os.path.join(RESULTS, "teacher-frgold-v2__extractive.jsonl"), os.path.join(RESULTS, "frgold-v2__extractive.jsonl"))]
    for src, dst in copies:
        if os.path.exists(src) and not os.path.exists(dst):
            shutil.copyfile(src, dst)


def wait_for_titles(code, max_h=8):
    log_path = os.path.join(CANON, "generate_%s.log" % code)
    t0 = time.time()
    while time.time() - t0 < max_h * 3600:
        if os.path.exists(log_path) and any(l.startswith("done") for l in io.open(log_path, encoding="utf-8", errors="replace")):
            return True
        time.sleep(60)
    log("titles for %s still missing after %d h: language skipped" % (code, max_h))
    return False


def build_gold(code):
    """True if a usable gold set exists (the annotator's labels exist + the check passes, >= 100 entries)."""
    labels = os.path.join(EVAL, "%s_gold_labels.jsonl" % code)
    gold = os.path.join(EVAL, "%s_gold.jsonl" % code)
    if not os.path.exists(labels):
        return os.path.exists(gold) and sum(1 for _ in io.open(gold, encoding="utf-8")) >= 100
    env = dict(os.environ, TITLER_TODO=os.path.join(EVAL, "%s_gold_todo.jsonl" % code), TITLER_LABELS=labels,
               TITLER_GOLD=gold, TITLER_REVIEW=os.path.join(EVAL, "%s_gold_review.md" % code))
    run([PY, os.path.join(EVAL, "build_gold.py"), "check"], env)
    n = sum(1 for _ in io.open(gold, encoding="utf-8")) if os.path.exists(gold) else 0
    log("gold %s : %d entries" % (code, n))
    return n >= 100


def run_pipeline(code, tok, steps):
    cmd = [PY, os.path.join(HERE, "pipeline_language.py"), code, "--steps"] + steps
    if tok:
        cmd += ["--tokenizer", tok]
    return run(cmd)


def group_model():
    log("=== EFIGS+P group (%s)" % GROUP_TOKENIZER)
    train = os.path.join(CANON, "efigsp_train_%s.jsonl" % GROUP_TOKENIZER.split("_")[-1])
    n_total = 0
    with io.open(train, "w", encoding="utf-8") as out:
        for code in GROUP:
            f = os.path.join(CANON, "%s_train_%s.jsonl" % (code, GROUP_TOKENIZER))
            if not os.path.exists(f):
                log("group: no labels for %s, language missing from the group" % code)
                continue
            n = 0
            for l in io.open(f, encoding="utf-8"):
                if code == "en" and n >= CAP_EN:
                    break
                out.write(l)
                n += 1
            log("group: %s %d examples" % (code, n))
            n_total += n
    log("group: %d examples in total" % n_total)
    name = "%s_emb8" % GROUP_TOKENIZER
    tok = os.path.join(HERE, "tokenizer_%s.json" % GROUP_TOKENIZER)
    env = dict(os.environ, TITLER_TOKENIZER=tok, TITLER_LABELED=train, TITLER_GOLD=os.path.join(EVAL, "gold.jsonl"),
               TITLER_CONSENSUS=os.path.join(EVAL, "en_gold_consensus_%s.jsonl" % GROUP_TOKENIZER))
    with io.open(os.path.join(RUNS, name + ".log"), "w", encoding="utf-8") as fh:
        subprocess.run([PY, os.path.join(HERE, "train.py"), "--name", name, "--layers", "2", "--dim", "32",
                        "--dim-emb", "8", "--epochs", "8"], env=env, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    for code in GROUP + [c for c in LANGUAGES if c not in GROUP]:
        gold = os.path.join(EVAL, "%s_gold.jsonl" % code)
        if not os.path.exists(gold):
            continue
        cons = os.path.join(EVAL, "%s_gold_consensus_%s.jsonl" % (code, GROUP_TOKENIZER))
        log("group on the %s gold set:" % code)
        run([PY, os.path.join(EVAL, "evaluate_run.py"), os.path.join(RUNS, name + ".pt"), "--tokenizer", tok,
             "--gold", gold, "--consensus", cons])


def main():
    prepare_en()
    for code in LANGUAGES:
        log("=== %s: waiting for titles" % code)
        if not wait_for_titles(code):
            continue
        gold = build_gold(code)
        steps = ["labels", "consensus", "train", "eval"] if gold else ["labels", "train"]
        log("=== %s: driving %s (gold set: %s)" % (code, " ".join(steps), "yes" if gold else "no"))
        run_pipeline(code, None, steps)
        if code in GROUP:
            run_pipeline(code, GROUP_TOKENIZER, ["labels"] + (["consensus"] if gold else []))
    for code in ("en", "fr"):
        run_pipeline(code, GROUP_TOKENIZER, ["labels", "consensus"])
    group_model()
    log("=== batch done")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
