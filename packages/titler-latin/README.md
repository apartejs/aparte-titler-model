# @aparte/titler-latin

[aparte-titler](https://apartejs.dev/models/titler/) with its default model bundled: a title of 3 to 6 words for a conversation, copied from its first message, in **17 European languages** — English, French, Spanish, German, Portuguese, Italian, Dutch, Polish, Swedish, Danish, Finnish, Czech, Romanian, Norwegian, Hungarian, Croatian, Lithuanian. 133 KB installed, no configuration, no API call, 6–7 ms per title.

## Install

```bash
npm install @aparte/titler-latin
```

## Use

```js
import { loadTitler } from "@aparte/titler-latin";

const titler = await loadTitler();   // loads the bundled model once (fetch in the browser, fs in Node)

titler.title("Salut ! Tu peux me donner une recette de pain sans gluten facile pour ce week-end ?");
// -> "recette pain sans gluten facile week-end"
```

`loadTitler()` resolves the model file relative to the package (`new URL("../model/…", import.meta.url)`), which works with modern bundlers, `<script type="module">` and Node ≥ 18. If your bundler does not carry the `.bin` file over, import `modelUrl` and serve the file yourself, or fetch it from [Hugging Face](https://huggingface.co/apartejs/aparte-titler) and use `@aparte/titler` directly.

The runtime is bundled in, so **this package has no dependency**: one line in your `package.json`, one package installed. It also exports `Titler`, if you want to read another model file yourself.

Give the model **only the user's first message**, never a system prompt or the history.

## Sisters

- [`@aparte/titler-latin-mini`](https://www.npmjs.com/package/@aparte/titler-latin-mini) — same 17 languages, 96 KB, −0.5 point on average.
- [`@aparte/titler-efigsp`](https://www.npmjs.com/package/@aparte/titler-efigsp) — English, French, Spanish, German, Portuguese, Italian, 77 KB.
- [`@aparte/titler`](https://www.npmjs.com/package/@aparte/titler) — the runtime alone, bring your own model (one language = 40 KB).

Scores, sizes and charts: [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler).

## License

MIT © 2026 Paul Richez.
