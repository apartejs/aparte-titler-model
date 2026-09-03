# @aparte/titler-efigsp

## 1.0.4

### Patch Changes

- 688096b: A model package builds for the browser again, and can be pointed at the model file your app actually serves (issue #2).

  The file-system read moved behind the `node` export condition, so nothing outside Node has to resolve `node:fs/promises` any more: an esbuild or Vite browser build used to fail on it even though the branch was guarded at runtime and never reached. The portable entry is now the default, so a tool that knows nothing about export conditions gets the one that builds everywhere.

  `loadTitler()` now also takes the model to load: a URL, a string, an `ArrayBuffer`, a typed array or a `Response`. With no argument it still uses the model bundled in the package, which is what Node and plain ES modules want. This is what a bundled app needs, because a bundler rewrites neither `import.meta.url` nor the package-relative path the default builds from — so the old no-argument call resolved next to the bundle and 404'd. The READMEs said this "works with modern bundlers"; it did not, and they now show the Vite and copy-the-file recipes instead. The test suite builds each package with esbuild for the browser so the regression cannot come back.

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
