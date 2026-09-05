// Equality test: the JS runtime must give exactly the titles of the Python
// reference implementation, message for message, on 1,496 messages in five
// languages (French, English, Spanish, Polish, Finnish) at the same precision.
//
//   node --test packages/titler/test/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Titler } from "../src/titler.js";

const here = new URL(".", import.meta.url);
const modelFile = readFileSync(new URL("../../titler-latin/model/titler-v1.1-latin-int3.bin", here));
const references = readFileSync(new URL("../../../tests/references/latin-int3.jsonl", here), "utf-8")
  .split("\n").filter(Boolean).map((line) => JSON.parse(line));

test("reads the model header", () => {
  const titler = new Titler(modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength));
  assert.equal(titler.header.format, 1);
  assert.equal(titler.header.vocab, 12288);
});

test("gives the reference titles on 1,496 messages", () => {
  const titler = new Titler(modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength));
  const mismatches = [];
  for (const { text, title } of references) {
    const mine = titler.title(text);
    if (mine !== title) mismatches.push({ expected: title, got: mine, text: text.slice(0, 80) });
  }
  assert.deepEqual(mismatches, [], `${mismatches.length} of ${references.length} titles differ`);
});

test("an empty or punctuation-only message gives an empty title", () => {
  const titler = new Titler(modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength));
  assert.equal(titler.title(""), "");
  assert.equal(titler.title("   ...  !!  "), "");
});
