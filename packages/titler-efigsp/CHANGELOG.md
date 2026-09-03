# @aparte/titler-efigsp

## 1.0.3

### Patch Changes

- c67ccda: A model package now has no dependency at all. `@aparte/titler-latin`, `-latin-mini` and `-efigsp` bundle their own copy of the runtime next to the `.bin`, instead of depending on `@aparte/titler`: installing one of them installs one package, and importing `Titler` from it needs nothing else declared — which strict package managers such as pnpm required before. `@aparte/titler` is unchanged and stays the runtime on its own, for bringing your own model file.

  The copies are generated from `packages/titler/src` by `pnpm sync-runtime` and checked byte for byte by the test suite, so there is still a single runtime. One consequence of bundling: the `Titler` exported by a model package is no longer the same class object as the one exported by `@aparte/titler`, so an `instanceof` check across the two packages is now false. The API, the titles and the model files are identical.

## 1.0.2

### Patch Changes

- 995ebf0: A title never contains a bare fragment (issue #1). A hyphen or an apostrophe glued between two letters no longer splits a word ("Pourrais-tu", "l'école" are scored whole), and a dash stuck to the front or the back of a word is trimmed like other punctuation ("-Don't" → "Don't"). Scores on the benchmark move by ±1 point; the reference titles of the equality test were regenerated accordingly.
- Updated dependencies [995ebf0]
  - @aparte/titler@1.0.2

## 1.0.1

### Patch Changes

- 64591b2: Accurate package descriptions and keywords for each model package; one size figure everywhere (133 KB for the 17-language model); a `pnpm bench` latency benchmark in the repository.
- Updated dependencies [64591b2]
  - @aparte/titler@1.0.1
