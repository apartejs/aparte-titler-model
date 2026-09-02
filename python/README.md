# Python

Three things live here. Nothing depends on a deep-learning framework except the training pipeline itself.

## `aparte_titler/reader.py` — the reference implementation

The executable specification of the model file format, in numpy: reads a `.bin`, tokenizes (byte-level BPE with GPT-2's split), runs the transformer, decodes the 6-word title. The JavaScript runtime is tested for equality against it.

```bash
python aparte_titler/reader.py ../packages/titler-latin/model/titler-v1-latin-int3.bin "Can you explain how photosynthesis works?"
python aparte_titler/reader.py model.bin --check gold.jsonl tokenizer.json   # tokens vs the `tokenizers` library
python aparte_titler/reader.py model.bin --references a.jsonl b.jsonl out.jsonl   # reference titles for the JS test
```

## `eval/` — scores and charts

```bash
python eval/matrix.py --models ../packages --gold path/to/aparte-titler-gold/gold --out scores.json
python eval/charts.py --scores scores.json --out charts/
```

`matrix.py` scores every `titler-*-int3.bin` it finds on every gold set (word-level F1 against the gold title, single reference), plus a no-model floor and, if given, the teacher's titles. `charts.py` draws the three figures of the model card and writes the table.

## `pipeline/` — training from scratch

The whole pipeline, in order: corpus (WildChat, OASST2, Aya, MFAQ), teacher titles (gemma 4 e2b on llama.cpp), tokenizer, labels and short messages, gold set, training, quantization, export. See `pipeline/README.md`.

## Requirements

```
numpy               # the reference reader needs only this
matplotlib          # charts
torch, tokenizers, pyarrow, datasketch   # the training pipeline
```

Python 3.11 or later.
