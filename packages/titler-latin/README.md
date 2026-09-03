# @aparte/titler-latin

[aparte-titler](https://apartejs.dev/models/titler/) with its default model bundled: a title of 3 to 6 words for a conversation, copied from its first message, in **17 European languages** — English, French, Spanish, German, Portuguese, Italian, Dutch, Polish, Swedish, Danish, Finnish, Czech, Romanian, Norwegian, Hungarian, Croatian, Lithuanian. 133 KB installed, no configuration, no API call, 6–7 ms per title.

## Install

```bash
npm install @aparte/titler-latin
```

## Use

```js
import { loadTitler } from "@aparte/titler-latin";

const titler = await loadTitler();   // the bundled model, loaded once

titler.title("Salut ! Tu peux me donner une recette de pain sans gluten facile pour ce week-end ?");
// -> "recette pain sans gluten facile week-end"
```

### In a bundled app

With no argument, `loadTitler()` uses the model that ships inside the package: read from disk in Node, fetched from `modelUrl` in a browser that loads the package as plain ES modules.

A bundler rewrites neither `import.meta.url` nor the model's path, so once your app is built that URL points next to your bundle rather than inside the package. Serve the `.bin` and hand it over:

```js
// Vite: let the bundler emit the file and give you its URL
import modelUrl from "@aparte/titler-latin/model?url";

const titler = await loadTitler(modelUrl);
```

```js
// anywhere else: copy node_modules/@aparte/titler-latin/model/titler-v1-latin-int3.bin
// into the directory you serve, then
const titler = await loadTitler("/models/titler-v1-latin-int3.bin");
```

`loadTitler()` also accepts an `ArrayBuffer`, a typed array or a `Response`, so the model can come from a cache, a service worker or your own asset pipeline.


The runtime is bundled in, so **this package has no dependency**: one line in your `package.json`, one package installed. It also exports `Titler`, if you want to read another model file yourself.

Give the model **only the user's first message**, never a system prompt or the history.

## Sisters

- [`@aparte/titler-latin-mini`](https://www.npmjs.com/package/@aparte/titler-latin-mini) — same 17 languages, 96 KB, −0.5 point on average.
- [`@aparte/titler-efigsp`](https://www.npmjs.com/package/@aparte/titler-efigsp) — English, French, Spanish, German, Portuguese, Italian, 77 KB.
- [`@aparte/titler`](https://www.npmjs.com/package/@aparte/titler) — the runtime alone, bring your own model (one language = 40 KB).

Scores, sizes and charts: [huggingface.co/apartejs/aparte-titler](https://huggingface.co/apartejs/aparte-titler).

## License

MIT © 2026 Paul Richez.
