---
"@aparte/titler": patch
"@aparte/titler-latin": patch
"@aparte/titler-latin-mini": patch
"@aparte/titler-efigsp": patch
---

A model package now has no dependency at all. `@aparte/titler-latin`, `-latin-mini` and `-efigsp` bundle their own copy of the runtime next to the `.bin`, instead of depending on `@aparte/titler`: installing one of them installs one package, and importing `Titler` from it needs nothing else declared — which strict package managers such as pnpm required before. `@aparte/titler` is unchanged and stays the runtime on its own, for bringing your own model file.

The copies are generated from `packages/titler/src` by `pnpm sync-runtime` and checked byte for byte by the test suite, so there is still a single runtime. One consequence of bundling: the `Titler` exported by a model package is no longer the same class object as the one exported by `@aparte/titler`, so an `instanceof` check across the two packages is now false. The API, the titles and the model files are identical.
