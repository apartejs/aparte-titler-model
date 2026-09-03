// A model package must build for the browser (issue #2) and keep working in
// Node. The file-system read lives behind the "node" export condition; if it
// ever leaks into the portable entry, a browser bundle fails to resolve
// node:fs/promises and this test fails instead of the user's build.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

import { PACKAGES } from "../../../scripts/sync-runtime.mjs";

const MESSAGE = "Write me a cover letter for a junior data analyst position at a bank";

for (const pkg of PACKAGES) {
  test(`@aparte/${pkg} builds for the browser`, async () => {
    const result = await build({
      stdin: { contents: `export { loadTitler, Titler, modelUrl, languages } from "@aparte/${pkg}";`, resolveDir: process.cwd(), loader: "js" },
      bundle: true, platform: "browser", format: "esm", write: false, logLevel: "silent",
    });
    assert.ok(!/["']node:/.test(result.outputFiles[0].text),
      `the browser bundle of @aparte/${pkg} still imports a Node built-in`);
  });

  test(`@aparte/${pkg} loads its bundled model in Node, and any model handed to it`, async () => {
    const { loadTitler, modelUrl, languages } = await import(`@aparte/${pkg}`);
    const bundled = await loadTitler();
    assert.ok(languages.length >= 6);
    assert.equal(await loadTitler(), bundled, "the bundled model is loaded once and cached");

    const bytes = readFileSync(fileURLToPath(modelUrl));
    for (const source of [bytes, bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)]) {
      const passed = await loadTitler(source);
      assert.notEqual(passed, bundled, "a model handed in is not the cached one");
      assert.equal(passed.title(MESSAGE), bundled.title(MESSAGE));
    }
    await assert.rejects(() => loadTitler(42), TypeError);
  });
}
