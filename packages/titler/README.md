# @aparte/titler

The runtime for [aparte-titler](https://apartejs.dev/models/titler/) models: a title of 3 to 6 words for a conversation, copied verbatim from its first message, in the browser or in Node, with no API call. One file, no dependency, ~6 KB minified, 6–7 ms per title on one CPU core.

This package is the runtime only. It reads the `.bin` model files published at [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler). If you just want it to work, install [`@aparte/titler-latin`](https://www.npmjs.com/package/@aparte/titler-latin) instead: same runtime, default model bundled (17 languages, 133 KB).

## Install

```bash
npm install @aparte/titler
```

## Use

```js
import { Titler } from "@aparte/titler";

// any model from https://huggingface.co/apartejs/aparte-titler
const url = "https://huggingface.co/apartejs/aparte-titler/resolve/main/titler-v1-latin-int3.bin";
const titler = new Titler(await fetch(url).then((r) => r.arrayBuffer()));

titler.title("Can you explain how photosynthesis works in plants?");
// -> "explain photosynthesis works plants"

titler.title("Peux-tu m'expliquer la photosynthèse chez les plantes ?", 4);
// -> a 4-word title

titler.words("Can you explain how photosynthesis works in plants?");
// -> { words: [{ word: "explain", score: 0.91, start: 8, end: 15, ... }, ...] }
```

In Node, read the file instead of fetching it:

```js
import { readFile } from "node:fs/promises";
const buf = await readFile("titler-v1-fr-int3.bin");
const titler = new Titler(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength));
```

Give the model **only the user's first message** — never a system prompt or the conversation history: it picks the most salient words of whatever it is given.

## Which model file?

| file | covers | int3 size |
|---|---|---|
| `titler-v1-latin-int3.bin` | 17 European languages (en, fr, es, de, pt, it, nl, pl, sv, da, fi, cs, ro, no, hu, hr, lt) | 133 KB |
| `titler-v1-latin-mini-int3.bin` | the same 17, smaller vocabulary, −0.5 point on average | 96 KB |
| `titler-v1-efigsp-int3.bin` | en, fr, es, de, pt, it | 77 KB |
| `titler-v1-<lang>-int3.bin` | one language | 40 KB |

Every model also exists in `fp32`, `int8` and `int4`. `int3` is the recommended precision: same score as `fp32` on every benchmark. Scores, sizes and charts: [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler).

## What it does, exactly

A 2-layer transformer encoder (58k parameters for one language, 141k for seventeen) scores every word of the message; the title is the best-scored words, kept in message order. The words are always copied from the message — it never rewrites, never invents, never corrects a typo. It also cannot execute anything: the worst an injected instruction can do is a poor title.

## Format and versions

The runtime reads model files of `format: 1`. A future format 2 (new architecture) will ship with a new major version of this package that still reads format 1 files. Models and runtime follow the same versioning: `1.x` keeps the file format, `2.0` changes it.

## License

MIT © 2026 Paul Richez. The models have their own cards and licenses on Hugging Face.
