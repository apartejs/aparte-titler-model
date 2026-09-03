// @aparte/titler-efigsp — the runtime plus the six-language model (int3, ~77 KB), ready to use with no configuration.
//
//   import { loadTitler } from "@aparte/titler-efigsp";
//   const titler = await loadTitler();
//   titler.title("Salut ! Tu peux me donner une recette de pain sans gluten facile pour ce week-end ?");
//
// This is the portable entry: it fetches the model, and imports nothing that a
// browser bundler cannot resolve. Node picks index.node.js instead (the "node"
// condition), which reads the bundled file straight from disk.
//
// A bundler rewrites neither `import.meta.url` nor the model's path, so once
// your app is bundled the default URL no longer points at the package: serve
// the `.bin` yourself and pass it in — `loadTitler(url)`, or with Vite
// `import modelUrl from "@aparte/titler-efigsp/model?url"`. See the README.
export { Titler, modelUrl, languages } from "./common.js";
import { modelUrl, fetchBytes, loader } from "./common.js";

/** Load a model once and return a ready Titler. No argument = the bundled model. */
export const loadTitler = loader(() => fetchBytes(modelUrl));
