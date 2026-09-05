# @aparte/titler

## 1.1.0

### Minor Changes

- Version 1.1 models, and a decoder fix that also improves the 1.0 files.

  **Decoder.** The hybrid decoder compared word probabilities to a hard-coded
  0.5 instead of the threshold each model was calibrated to. On models calibrated
  low — Czech at 0.20, Croatian at 0.10 — the filter rejected every word, the
  three-word floor took over, and the hybrid silently became a fixed budget of 3.
  Measured across 19 scopes: up to 15 points lost, and every model above 0.5
  gained while every model below it lost, without exception. Each `.bin` now
  carries its own threshold and the runtime reads it.

  **Weights.** Retrained on a corpus rebuilt without LMSYS-Chat-1M, whose licence
  forbids redistribution, so every published line can now be redistributed. The
  Hungarian corpus doubled: half of it had been lost to a dead titling server and
  recorded as teacher failures. +0.015 on average across the 20 scopes, same
  runtime, 12 of 20 improving.

  **Scores are now reported by message length.** A single figure mixed a short
  message (0.84) with a long one (0.45) in proportions set by the gold rather
  than by the model. The card also publishes agreement between human references
  as the practical ceiling, and says plainly which languages are not measured
  beyond 20 words.

  The 1.0 files stay online and stay listed in the manifest.

## 1.0.4

## 1.0.3

### Patch Changes

- c67ccda: A model package now has no dependency at all. `@aparte/titler-latin`, `-latin-mini` and `-efigsp` bundle their own copy of the runtime next to the `.bin`, instead of depending on `@aparte/titler`: installing one of them installs one package, and importing `Titler` from it needs nothing else declared — which strict package managers such as pnpm required before. `@aparte/titler` is unchanged and stays the runtime on its own, for bringing your own model file.

  The copies are generated from `packages/titler/src` by `pnpm sync-runtime` and checked byte for byte by the test suite, so there is still a single runtime. One consequence of bundling: the `Titler` exported by a model package is no longer the same class object as the one exported by `@aparte/titler`, so an `instanceof` check across the two packages is now false. The API, the titles and the model files are identical.

## 1.0.2

### Patch Changes

- 995ebf0: A title never contains a bare fragment (issue #1). A hyphen or an apostrophe glued between two letters no longer splits a word ("Pourrais-tu", "l'école" are scored whole), and a dash stuck to the front or the back of a word is trimmed like other punctuation ("-Don't" → "Don't"). Scores on the benchmark move by ±1 point; the reference titles of the equality test were regenerated accordingly.

## 1.0.1

### Patch Changes

- 64591b2: Accurate package descriptions and keywords for each model package; one size figure everywhere (133 KB for the 17-language model); a `pnpm bench` latency benchmark in the repository.
