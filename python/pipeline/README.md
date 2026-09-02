# Titler training pipeline

English-language port of the training pipeline for the extractive title tagger. Every script below is a faithful translation of the original (French) lab code: same defaults, same regexes, same numeric constants, same output formats.

## Scripts

- `data/build_corpus.py` — builds the English message corpus from a local WildChat-1M download (filter, dedupe, MinHash near-dedupe, exclude the bake-off sample, sample uniformly over time).
- `data/languages.py` — the same corpus construction, per language, blending WildChat-4.8M with OASST2, Aya, and MFAQ to reach 12,000 training messages.
- `data/generate_titles.py` — generates extractive titles for the corpus by running the teacher (gemma-4-e2b under llama.cpp) over every message.
- `data/teacher_run.py` — the low-level teacher runner: dispatches messages to one or more loaded model instances and records their titles.
- `data/prompt_v2.txt` — the English teacher prompt used by `generate_titles.py`.
- `eval/run_teacher.py` — starts llama-server instances directly (no LM Studio) and runs `teacher_run.py` against them, typically for gold-set scoring.
- `eval/build_gold.py` — selects the gold evaluation sample, and turns hand-written labels into checked word spans (`select` / `check` / `show`).
- `eval/consensus.py` — builds a multi-annotator consensus reference (per-token votes) from several title files.
- `eval/compare_teacher.py` — compares the teacher's titles against the gold set with a strict and a tolerant word-matching validator.
- `eval/score.py` — the project's reference scorer: token- and word-level precision/recall/F1 against the gold set, plus the reusable baselines.
- `eval/evaluate_run.py` — scores one trained model checkpoint against any gold set (not just the one it was trained on).
- `train/train_tokenizer.py` — trains the tagger's byte-level BPE tokenizer on the message corpus.
- `train/labels.py` — aligns each (message, title) pair into per-token keep/discard labels.
- `train/short_messages.py` — labels short messages (< 20 characters) as "keep everything".
- `train/filter_agreement.py` — filters teacher titles by agreement between two independent titlings of the same message.
- `train/data.py` — loads labeled examples, splits train/validation, and builds length-grouped batches.
- `train/model.py` — the tagger model: a small encoder transformer with an optional learned length head.
- `train/train.py` — trains the tagger and reports token- and word-level scores against the gold set.
- `train/pipeline_language.py` — drives the full labels -> consensus -> train -> eval sequence for one language.
- `train/batch_languages.py` — runs the per-language pipeline for every language, then trains the EFIGS+P group model.
- `train/group.py` — trains one shared-tokenizer group model across an arbitrary set of languages.
- `train/quantize.py` — simulates post-training int8/int4 quantization and measures the resulting score.
- `train/export.py` — exports a trained checkpoint to the compact binary format a JS runtime reads.
- `train/distribute.py` — exports every model at every precision into a versioned, manifested release folder.

## Pipeline order

1. **Corpus** — `data/build_corpus.py` (English) or `data/languages.py` (per language)
2. **Teacher titles** — `data/generate_titles.py` (uses `data/teacher_run.py` and `data/prompt_v2.txt`)
3. **Tokenizer** — `train/train_tokenizer.py`
4. **Labels + short messages** — `train/labels.py`, `train/short_messages.py` (optionally filtered with `train/filter_agreement.py`)
5. **Gold set** — `eval/build_gold.py`, `eval/run_teacher.py`, `eval/consensus.py`, `eval/compare_teacher.py`, `eval/score.py`
6. **Train** — `train/train.py`, orchestrated per language by `train/pipeline_language.py`, `train/batch_languages.py`, and `train/group.py`; scored with `eval/evaluate_run.py`
7. **Quantize / export** — `train/quantize.py`, `train/export.py`, `train/distribute.py`

## Environment variables

Path overrides (all optional; each script falls back to a sensible default under `data/canonical/`, `eval/`, or `train/`):

- `TITLER_WILDCHAT` — folder of the source WildChat parquet files
- `TITLER_TOKENIZER` — tokenizer JSON file
- `TITLER_MESSAGES` — training messages JSONL
- `TITLER_TITLES` — teacher titles JSONL
- `TITLER_LABELED` — derived per-token labels JSONL
- `TITLER_GOLD` — gold evaluation set JSONL
- `TITLER_CONSENSUS` — multi-annotator consensus JSONL
- `TITLER_TODO` — gold-set selection to label JSONL
- `TITLER_LABELS` — raw hand-written gold labels JSONL
- `TITLER_REVIEW` — gold-set review Markdown output
- `TITLER_POOL` — eval pool JSONL that the gold set is drawn from
- `TITLER_EPOCHS` — number of training epochs (used by `train/group.py`)
- `TITLER_REDO` — set (to any value) to force re-labeling of a language already labeled (used by `train/group.py`)

Teacher server locations (required by `data/generate_titles.py` and `eval/run_teacher.py`; each raises a clear error if unset):

- `TITLER_LLAMA_SERVER` — path to `llama-server.exe`
- `TITLER_GGUF` — path to the teacher's GGUF weights (`data/generate_titles.py` only)
- `TITLER_LLAMA_VENDOR_BIN` — folder of the vendor DLLs llama-server needs on its PATH

Requires Python 3.11+, torch, tokenizers, numpy, pyarrow, datasketch; the teacher runs on llama.cpp's llama-server.
