# aparte-titler

A title for a conversation, from its first message, in the browser — the weight of an icon, no API call.

aparte-titler is a tiny transformer — 58k parameters for one language, 141k for seventeen — that picks 3 to 6 words of the user's first message and returns them as the conversation's title. It covers **17 European languages** in a single 133 KB file, runs in a few milliseconds on one CPU core (2–4 ms in Python, 6–7 ms in the pure-JS runtime, for a typical 300-character message), and never sends the message anywhere.

On the benchmark, over the eight languages with real chat data, it scores **about twice the naive baseline** (the first five words of the message) and **~92 % of a 2.4 GB LLM**, a model 18,000× heavier (94 % over all 17 languages); on four languages (Danish, Finnish, Hungarian, Czech) it is on par with that LLM — above it on two, within a point on the other two. And because it only ever copies words from the message, **it is immune to prompt injection by construction**: an instruction hidden in a message can at worst produce a poor title — it can never be executed.

- Presentation and story: **[apartejs.dev/models/titler](https://apartejs.dev/models/titler/)** · live demo: **[apartejs.dev/models/titler/#demo](https://apartejs.dev/models/titler/#demo)**
- Models, scores and charts: **[huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler)**
- Training data: [apartejs/aparte-titler-data](https://huggingface.co/datasets/apartejs/aparte-titler-data) · Benchmark: [apartejs/aparte-titler-gold](https://huggingface.co/datasets/apartejs/aparte-titler-gold)

## Quick start

```bash
npm install @aparte/titler-latin
```

```js
import { loadTitler } from "@aparte/titler-latin";

const titler = await loadTitler();
titler.title("Write me a cover letter for a junior data analyst position at a bank");
// -> "cover letter junior analyst position bank"
```

Python, with the reference implementation (numpy only):

```bash
python python/aparte_titler/reader.py titler-v1-latin-int3.bin "Peux-tu m'expliquer la photosynthèse chez les plantes ?"
```

Give the model **only the user's first message** — never a system prompt or the history: it titles whatever it is given.

## Packages

| package | contents | size |
|---|---|---|
| [`@aparte/titler`](packages/titler) | the runtime alone, one ESM file; bring any `.bin` from Hugging Face | 6 KB minified |
| [`@aparte/titler-latin`](packages/titler-latin) | runtime + the default model: en, fr, es, de, pt, it, nl, pl, sv, da, fi, cs, ro, no, hu, hr, lt | 133 KB |
| [`@aparte/titler-latin-mini`](packages/titler-latin-mini) | same 17 languages, smaller vocabulary (−0.5 point on average) | 96 KB |
| [`@aparte/titler-efigsp`](packages/titler-efigsp) | en, fr, es, de, pt, it | 77 KB |

One-language models (40 KB each) and every model in `fp32`, `int8`, `int4` and `int3` are on [Hugging Face](https://huggingface.co/apartejs/aparte-titler); the runtime loads any of them. Every model also exists as an ONNX graph (fp32 and int8) for ONNX Runtime users, in the [`onnx/`](https://huggingface.co/apartejs/aparte-titler/tree/main/onnx) folder of the same repository.

Each model package bundles the runtime, so it has **no dependency at all**: installing it installs one package. `@aparte/titler` is the runtime on its own, for bringing your own model file.

## Results

Word-level F1 against the gold title, one reference, 300 messages per language, the shipped `int3` files. "LLM" is gemma 4 e2b (2.4 GB), the model that taught it; "no model" is the first five words of the message.

| language | one model (40 KB) | latin (133 KB) | LLM (2.4 GB) | no model |
|---|---|---|---|---|
| English | 0.626 | 0.610 | 0.672 | 0.315 |
| French | 0.625 | 0.624 | 0.681 | 0.328 |
| Spanish | 0.611 | 0.607 | 0.681 | 0.300 |
| German | 0.614 | 0.622 | 0.657 | 0.283 |
| Portuguese | 0.587 | 0.593 | 0.628 | 0.329 |
| Italian | 0.586 | 0.595 | 0.682 | 0.293 |
| Dutch | 0.630 | 0.630 | 0.710 | 0.266 |
| Polish | 0.609 | 0.602 | 0.618 | 0.384 |
| Swedish | 0.758 | 0.755 | 0.809 | 0.431 |
| Danish | 0.752 | 0.756 | 0.758 | 0.484 |
| Finnish | 0.789 | 0.786 | 0.773 | 0.674 |
| Czech | 0.766 | 0.769 | 0.775 | 0.510 |
| Romanian | 0.673 | 0.761 | 0.794 | 0.471 |
| Norwegian | 0.747 | 0.768 | 0.815 | 0.438 |
| Hungarian | 0.705 | 0.734 | 0.706 | 0.509 |
| Croatian | 0.602 | 0.754 | 0.811 | 0.492 |
| Lithuanian | 0.499 | 0.564 | 0.657 | 0.343 |

Three things the table shows. The 17-language model is as good as a dedicated one-language model on the big languages and **better on the small ones** (Romanian, Croatian, Lithuanian): the transformer is shared, the small languages borrow what the big ones taught it. **On Danish, Finnish, Hungarian and Czech the 133 KB model matches or beats the 2.4 GB LLM** — those sets are mostly short FAQ-style questions, an easier task, but the same task for both. And every system roughly doubles the no-model floor.

One word on the metric: it uses a single reference title per message. Several different titles are often valid, so a single reference under-counts every system in the same way, the LLM included — it lowers every number without biasing the comparison. Two careful annotators agree at only 0.66–0.74 on it.

Full matrix (every model on every language), charts and per-precision sizes: [model card](https://huggingface.co/apartejs/aparte-titler).

## How it works

- **Extractive.** The model scores every word of the message; the title is the six best-scored words in message order. Words are always copied verbatim: it never rewrites, never invents, never fixes a typo — and it cannot execute an instruction found in a message.
- **Tiny.** A 2-layer transformer encoder, width 32, 4 heads, factorized embeddings (8 dimensions), byte-level BPE. 58k parameters for one language, 141k for 17. 82 % of the weight is in the vocabulary tables, not in the transformer.
- **Distilled.** Trained from scratch on titles written by gemma 4 e2b for 12,000 real first messages per language (WildChat, OASST2, Aya, MFAQ), with the words of each title aligned back to the message.
- **Quantized.** `int3` (3 bits per weight, one fp16 scale per row) keeps the score of `fp32` on every benchmark — it changes the exact title in about a third of the messages, by swapping near-tied words.
- **Measured.** A gold set of 300 messages per language, titled blind by a frontier model, with the teacher LLM as a second voice. Everything in the card is produced by `python/eval/matrix.py` and `python/eval/charts.py`.

Known limits: long messages that start with a preamble ("act as an expert in…") get titled on the preamble; chat-speak stop words ("pk", "ya") are not always dropped; typos in the message are copied as they are; Lithuanian is the weakest language (0.563 with the 17-language model, 0.500 alone) because it has 1,134 training messages, ten times fewer than the others.

## Repository layout

```
packages/titler/             the runtime (src/titler.js + titler.d.ts) and its equality test
packages/titler-latin/       runtime + default model         packages/titler-latin-mini/, packages/titler-efigsp/
python/aparte_titler/        reader.py: the reference implementation of the file format (numpy only)
python/eval/                 matrix.py (scores), charts.py (figures)
python/pipeline/             the full training pipeline: corpus, teacher, tokenizer, labels, training, quantization, export
tests/references/            1,496 reference titles the JS runtime must reproduce exactly
```

```bash
pnpm install && pnpm test          # the runtime against the reference titles (1,496 messages, 5 languages)
pnpm bench                         # latency: ms per title, titles/s, µs per character, on your machine
```

The test suite is not a benchmark (its harness dominates, tens of seconds for 1,496 messages); `pnpm bench` loads the model once and loops, which is what the latency figures come from.

## Model file format

A `.bin` file is: the magic `LMAP`, a JSON header (`format: 1`, then the file's identity — `name`, `version`, `model`, `languages`, `precision`, `license`, `url` — the architecture, the BPE merge base, the byte-to-id table and the tensor list), the BPE merges as `uint16` pairs, then the tensors — quantized rows packed on `bits` bits with an fp16 scale per row, vectors in fp16 (or everything in fp32). `python/aparte_titler/reader.py` is the executable specification; the JS runtime is tested against it.

Why 133 KB for 141k parameters? The 3-bit weights are 53 KB; the file adds one fp16 scale per matrix row (26 KB, mostly for the 12,288 embedding rows), the BPE merge table (12,030 pairs, 48 KB), the JSON header with the byte-to-id table and the file's identity (4 KB) and the fp16 bias and normalization vectors (2 KB).

Versioning: `1.x` releases keep the file format; a `2.0` changes it, and the runtime keeps reading format 1.

## License

Code and weights: [MIT](LICENSE) © 2026 Paul Richez. The training data and the benchmark carry the licenses of their sources — see their cards.
