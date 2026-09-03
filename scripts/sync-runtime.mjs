// Copies the runtime into every model package, so that a model package depends
// on nothing at all: the JavaScript file and the `.bin` travel together.
//
//   pnpm sync-runtime
//
// `packages/titler/src` holds the only editable copy. The generated ones are
// checked byte for byte by packages/titler/test/copies.test.mjs, so a forgotten
// sync fails the test suite instead of shipping two different runtimes.
import { readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

/** The packages that bundle a model, and therefore a copy of the runtime. */
export const PACKAGES = ["titler-latin", "titler-latin-mini", "titler-efigsp"];

/** The runtime files copied into each of them. */
export const FILES = ["titler.js", "titler.d.ts"];

/** Prepended to a copy, so nobody edits the wrong file. */
export const banner = (file) =>
  `// GENERATED FILE — a verbatim copy of packages/titler/src/${file}, bundled here\n` +
  `// so that this package has no dependency. Edit the original, then run\n` +
  `// \`pnpm sync-runtime\` from the repository root.\n\n`;

export const sourcePath = (file) => join(ROOT, "packages", "titler", "src", file);
export const copyPath = (pkg, file) => join(ROOT, "packages", pkg, "src", file);

/** The exact expected content of a copy. */
export function expected(file) {
  return banner(file) + readFileSync(sourcePath(file), "utf8");
}

function main() {
  for (const pkg of PACKAGES) {
    for (const file of FILES) {
      writeFileSync(copyPath(pkg, file), expected(file));
      console.log("wrote packages/%s/src/%s", pkg, file);
    }
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) main();
