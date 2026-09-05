// Latency benchmark: loads the model once, titles the 1,496 reference messages
// in a loop, and prints ms per title, titles per second and µs per character.
// The test suite is not a benchmark (its harness dominates); this is.
//
//   pnpm bench                      # default model, 3 passes
//   node packages/titler/bench/bench.mjs path/to/model.bin [passes]
import { readFileSync } from "node:fs";
import { Titler } from "../src/titler.js";

const here = new URL(".", import.meta.url);
const modelPath = process.argv[2] ?? new URL("../../titler-latin/model/titler-v1.1-latin-int3.bin", here);
const passes = Number(process.argv[3] ?? 3);
const file = readFileSync(modelPath);
const titler = new Titler(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
const texts = readFileSync(new URL("../../../tests/references/latin-int3.jsonl", here), "utf-8")
  .split("\n").filter(Boolean).map((l) => JSON.parse(l).text);
const chars = texts.reduce((n, t) => n + Math.min(t.trim().length, titler.header.max_chars), 0);

console.log(`${titler.header.name} ${titler.header.version} — ${titler.header.model} ${titler.header.precision}, ${titler.header.languages.length} languages, ${file.byteLength} bytes`);
console.log(`${texts.length} messages, ${Math.round(chars / texts.length)} characters on average, one CPU core, Node ${process.versions.node}`);
for (const t of texts.slice(0, 50)) titler.title(t);          // warm-up (JIT)
for (let pass = 1; pass <= passes; pass++) {
  const t0 = performance.now();
  for (const t of texts) titler.title(t);
  const ms = performance.now() - t0;
  console.log(`pass ${pass}: ${(ms / texts.length).toFixed(2)} ms per title, ${Math.round(1000 * texts.length / ms)} titles/s, ${(1000 * ms / chars).toFixed(1)} µs per character`);
}
