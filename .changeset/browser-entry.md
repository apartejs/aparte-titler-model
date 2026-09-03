---
"@aparte/titler-latin": patch
"@aparte/titler-latin-mini": patch
"@aparte/titler-efigsp": patch
---

A model package builds for the browser again, and can be pointed at the model file your app actually serves (issue #2).

The file-system read moved behind the `node` export condition, so nothing outside Node has to resolve `node:fs/promises` any more: an esbuild or Vite browser build used to fail on it even though the branch was guarded at runtime and never reached. The portable entry is now the default, so a tool that knows nothing about export conditions gets the one that builds everywhere.

`loadTitler()` now also takes the model to load: a URL, a string, an `ArrayBuffer`, a typed array or a `Response`. With no argument it still uses the model bundled in the package, which is what Node and plain ES modules want. This is what a bundled app needs, because a bundler rewrites neither `import.meta.url` nor the package-relative path the default builds from — so the old no-argument call resolved next to the bundle and 404'd. The READMEs said this "works with modern bundlers"; it did not, and they now show the Vite and copy-the-file recipes instead. The test suite builds each package with esbuild for the browser so the regression cannot come back.
