# @aparte/titler-efigsp

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
