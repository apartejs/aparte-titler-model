"""The set of deliverable files: each model in each precision, named
according to the convention (`titler-v1-<scope>-<precision>.bin`), with a manifest.

    python train/distribute.py            # -> dist/v1/
"""
import io
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PY = sys.executable   # the interpreter running this script, on every platform
RUNS = os.path.join(ROOT, "runs")
DIST = os.path.join(ROOT, "dist", "v1")
VERSION = "v1"

LANGUAGES = "es de pt it nl pl sv da fi cs ro no hu hr lt".split()
# scope -> (run, tokenizer, covered languages)
MODELS = {"en": ("red_emb8", "2048", ["en"]), "fr": ("fr_emb8", "fr_2048", ["fr"])}
for c in LANGUAGES:
    MODELS[c] = ("%s_%s_2048" % (c, c), "%s_2048" % c, [c])
MODELS["efigsp"] = ("efigsp_6144_emb8", "efigsp_6144", ["en", "fr", "es", "de", "pt", "it"])
MODELS["latin"] = ("latin_12288_emb8", "latin_12288", ["en", "fr"] + LANGUAGES)
MODELS["latin-mini"] = ("latin_8192_emb8", "latin_8192", ["en", "fr"] + LANGUAGES)
PRECISIONS = [("fp32", 32), ("int8", 8), ("int4", 4), ("int3", 3)]


def main():
    os.makedirs(DIST, exist_ok=True)
    manifest = []
    for scope, (run, tok, languages) in MODELS.items():
        pt = os.path.join(RUNS, run + ".pt")
        meta = json.load(io.open(os.path.join(RUNS, run + ".json"), encoding="utf-8"))
        for precision_name, bits in PRECISIONS:
            r = subprocess.run([PY, os.path.join(HERE, "export.py"), pt, "--bits", str(bits),
                                "--tokenizer", os.path.join(HERE, "tokenizer_%s.json" % tok)],
                               capture_output=True, text=True, cwd=ROOT)
            if r.returncode != 0:
                print("FAILED %s %s: %s" % (scope, precision_name, r.stderr.strip()[-200:]), flush=True)
                continue
            src = pt[:-3] + (".fp32.bin" if bits == 32 else ".int%d.bin" % bits)
            name = "titler-%s-%s-%s.bin" % (VERSION, scope, precision_name)
            dst = os.path.join(DIST, name)
            shutil.move(src, dst)
            manifest.append({"file": name, "scope": scope, "precision": precision_name, "bytes": os.path.getsize(dst),
                              "parameters": meta["parameters"], "vocab": meta["vocab"], "languages": languages,
                              "run": run, "tokenizer": tok})
        print("%-10s %s" % (scope, "  ".join("%s %s" % (m["precision"], format(m["bytes"], ",")) for m in manifest if m["scope"] == scope)), flush=True)
    with io.open(os.path.join(DIST, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print("%d files -> %s" % (len(manifest), os.path.relpath(DIST, ROOT)))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
