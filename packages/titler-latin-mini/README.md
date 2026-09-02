# @aparte/titler-latin-mini

[aparte-titler](https://apartejs.dev/models/titler/) with its default model bundled: a title of 3 to 6 words for a conversation, copied from its first message, in the **same 17 European languages as `@aparte/titler-latin`, with a smaller vocabulary** (−0.5 point on average). 96 KB installed, no configuration, no API call, 6–7 ms per title.

## Install

```bash
npm install @aparte/titler-latin-mini
```

## Use

```js
import { loadTitler } from "@aparte/titler-latin-mini";

const titler = await loadTitler();   // loads the bundled model once (fetch in the browser, fs in Node)

titler.title("Peux-tu m'expliquer la photosynthèse chez les plantes ?");
// -> "expliquer photosynthèse plantes"
```

`loadTitler()` resolves the model file relative to the package (`new URL("../model/…", import.meta.url)`), which works with modern bundlers, `<script type="module">` and Node ≥ 18. If your bundler does not carry the `.bin` file over, import `modelUrl` and serve the file yourself, or fetch it from [Hugging Face](https://huggingface.co/apartejs/aparte-titler) and use `@aparte/titler` directly.

The package depends on `@aparte/titler` and re-exports `Titler`, so importing everything from `@aparte/titler-latin-mini` needs no other dependency. If you import `@aparte/titler` directly, declare it in your own `package.json` too (pnpm does not hoist it).

Give the model **only the user's first message**, never a system prompt or the history.

## Sisters

- [`@aparte/titler-latin`](https://www.npmjs.com/package/@aparte/titler-latin) — the default model, same 17 languages, 133 KB, +0.5 point on average.
- [`@aparte/titler-efigsp`](https://www.npmjs.com/package/@aparte/titler-efigsp) — English, French, Spanish, German, Portuguese, Italian, 77 KB.
- [`@aparte/titler`](https://www.npmjs.com/package/@aparte/titler) — the runtime alone, bring your own model (one language = 40 KB).

Scores, sizes and charts: [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler).

## License

MIT © 2026 Paul Richez.
