// @aparte/titler-efigsp — the runtime plus the default model (6 languages,
// int3, 77 KB), ready to use with no configuration.
//
//   import { loadTitler } from "@aparte/titler-efigsp";
//   const titler = await loadTitler();
//   titler.title("Salut ! Tu peux me donner une recette de pain sans gluten facile pour ce week-end ?");
//
// The model file ships inside the package; it is loaded relative to this
// module (fetch in the browser, the file system in Node).
import { Titler } from "@aparte/titler";

export { Titler } from "@aparte/titler";

/** URL of the bundled model file, for loaders that want to fetch it themselves. */
export const modelUrl = new URL("../model/titler-v1-efigsp-int3.bin", import.meta.url);

/** Languages covered by this model (ISO 639-1). */
export const languages = ["en", "fr", "es", "de", "pt", "it"];

async function readModelBytes() {
  if (typeof process !== "undefined" && process.versions && process.versions.node && modelUrl.protocol === "file:") {
    const { readFile } = await import("node:fs/promises");
    const { fileURLToPath } = await import("node:url");
    const buf = await readFile(fileURLToPath(modelUrl));
    return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  }
  const response = await fetch(modelUrl);
  if (!response.ok) throw new Error("could not load the titler model from " + modelUrl + " (" + response.status + ")");
  return response.arrayBuffer();
}

let cached;

/** Load the bundled model once and return a ready Titler. */
export async function loadTitler() {
  cached ??= readModelBytes().then((bytes) => new Titler(bytes));
  return cached;
}
