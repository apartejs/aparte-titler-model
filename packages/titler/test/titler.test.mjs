// Equality test: the JS runtime must give exactly the titles of the Python
// reference implementation, message for message, on 1,496 messages in five
// languages (French, English, Spanish, Polish, Finnish) at the same precision.
//
//   node --test packages/titler/test/
import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { Titler } from "../src/titler.js";

const here = new URL(".", import.meta.url);
const modelFile = readFileSync(new URL("../../titler-latin/model/titler-v1.2-latin-int3.bin", here));
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

// The same equality, at the three other precisions. They take four separate
// paths through the loader -- int8 has its own branch in unpackBits, int4 goes
// through the generic bit reader, fp32 is not packed at all -- and until now
// only int3 was ever checked. The npm packages ship int3 alone, so these files
// live in the Hugging Face repo next door: the test runs where they are and is
// reported as skipped where they are not (in CI, for instance).
for (const precision of ["fp32", "int8", "int4"]) {
  const bin = new URL(`../../../../hf-model/v1.2/titler-v1.2-latin-${precision}.bin`, here);
  const refs = new URL(`../../../tests/references/latin-${precision}.jsonl`, here);
  const present = existsSync(bin) && existsSync(refs);
  test(`gives the reference titles at ${precision}`, { skip: present ? false : `titler-v1.2-latin-${precision}.bin not found` }, () => {
    const file = readFileSync(bin);
    const titler = new Titler(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
    const expected = readFileSync(refs, "utf-8").split("\n").filter(Boolean).map((line) => JSON.parse(line));
    const mismatches = expected.filter(({ text, title }) => titler.title(text) !== title).length;
    assert.equal(mismatches, 0, `${mismatches} of ${expected.length} titles differ at ${precision}`);
  });
}

test("an empty or punctuation-only message gives an empty title", () => {
  const titler = new Titler(modelFile.buffer.slice(modelFile.byteOffset, modelFile.byteOffset + modelFile.byteLength));
  assert.equal(titler.title(""), "");
  assert.equal(titler.title("   ...  !!  "), "");
});
