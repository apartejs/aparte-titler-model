---
"@aparte/titler": patch
"@aparte/titler-latin": patch
"@aparte/titler-latin-mini": patch
"@aparte/titler-efigsp": patch
---

A title never contains a bare fragment (issue #1). A hyphen or an apostrophe glued between two letters no longer splits a word ("Pourrais-tu", "l'école" are scored whole), and a dash stuck to the front or the back of a word is trimmed like other punctuation ("-Don't" → "Don't"). Scores on the benchmark move by ±1 point; the reference titles of the equality test were regenerated accordingly.
