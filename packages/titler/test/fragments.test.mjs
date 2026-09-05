// Regression test for issue #1: a hyphenated or elided French form must never
// be split so that a bare fragment ("-tu", "'école") lands in the title.
//
//   node --test packages/titler/test/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Titler } from "../src/titler.js";

const here = new URL(".", import.meta.url);
const file = readFileSync(new URL("../../titler-efigsp/model/titler-v1.1-efigsp-int3.bin", here));
const titler = new Titler(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));

const probes = [
  "Peux-tu me résumer ce document juridique ?",
  "Pourrais-tu vérifier les calculs de la facture ?",
  "Est-ce que tu peux lire ce fichier Excel ?",
  "Dis-moi comment installer Docker sur Windows",
  "Rappelle-moi de relancer le client vendredi",
  "Va-t-il pleuvoir demain à Lille ?",
  "Explique-moi le rendez-vous de mardi",
  "Où en est le sous-traitant sur le chantier ?",
  "Peux-tu m'expliquer comment fonctionne la photosynthèse chez les plantes ?",
  "L'école d'ingénieurs qu'elle a choisie est à Lyon",
];

test("no title word is a bare fragment (issue #1)", () => {
  for (const message of probes) {
    const title = titler.title(message);
    for (const word of title.split(" ")) {
      assert.ok(!/^[-'’]/.test(word), `"${message}" -> "${title}": "${word}" starts with a joiner`);
    }
  }
});

test("a hyphenated form is scored as one word", () => {
  const { words } = titler.words("Pourrais-tu vérifier les calculs de la facture ?");
  assert.deepEqual(words.map((w) => w.word), ["Pourrais-tu", "vérifier", "les", "calculs", "de", "la", "facture"]);
});
