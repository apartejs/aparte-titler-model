// The model packages bundle their own copy of the runtime so that they have no
// dependency. This checks that the copies are exactly the runtime — a forgotten
// `pnpm sync-runtime` must fail here, not ship two different runtimes.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

import { PACKAGES, FILES, copyPath, expected } from "../../../scripts/sync-runtime.mjs";

for (const pkg of PACKAGES) {
  for (const file of FILES) {
    test(`@aparte/${pkg} bundles the current ${file}`, () => {
      assert.equal(readFileSync(copyPath(pkg, file), "utf8"), expected(file),
        `packages/${pkg}/src/${file} is out of date — run \`pnpm sync-runtime\``);
    });
  }
}
