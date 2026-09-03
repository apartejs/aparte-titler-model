"""Is this message really written in the language its corpus claims?

WildChat labels a *conversation*, not a message, so a corpus built from it
carries messages in another language: an English system prompt, a pasted Russian
exercise, a Chinese request. They are noise in training and, worse, unanswerable
items in an evaluation set.

Two traps make an off-the-shelf identifier the wrong tool here:

  - a message is often the user's own words plus a large pasted block of code or
    of English text, and the identifier answers about the block, not about the
    request. "erkläre was hier mit dem java code passiert" plus fifty lines of
    Java reads as English to any identifier, and it is a perfectly good German
    message.
  - close languages swap freely. Danish reads as Norwegian, Dutch as Afrikaans,
    Spanish as Catalan, and the expected language then gets no weight at all.

So we do not identify the language of the message. We ask a narrower question
with a clear answer: do the *function words* of the expected language appear at
a rate comparable to the best-scoring language? Function words are what a user
writes and what pasted material does not carry over, they are frequent enough to
show up in one short sentence, and close languages share them, which is exactly
the tolerance we want.

A message is called off-language only when the answer is not arguable:
  - another writing system dominates its letters, or
  - the expected language scores far below the best-scoring one.

Everything else is kept, including anything too short or too code-heavy to
judge: the cost of dropping a good message is higher than that of keeping a
doubtful one.

    python data/language_check.py --gold ../../hf-gold/gold      # report
    python data/language_check.py --gold ../../hf-gold/gold --apply
"""
import argparse
import io
import json
import os
import re
import sys
import unicodedata

# ~40 function words per language: articles, pronouns, prepositions, auxiliaries.
WORDS = {
    "en": "the be to of and a in that have i it for not on with he as you do at this but his by from they we say her she or an will my one all would there their what is are was can how please",
    "fr": "le la les de des du un une et est que qui pour dans sur avec pas plus je tu il elle nous vous ils ce cette au aux en ne se sont être avoir fait comment pourquoi quel quelle mon ma mes ton votre par sans sous chez très bien tout tous toute peux peut",
    "es": "el la los las de del un una y es que en por para con no se su sus al lo como más pero le ya o este esta si porque qué cuando muy sin sobre también me hasta hay donde quien desde todo nos quiero puedes cómo",
    "de": "der die das den dem des ein eine einen und ist zu mit auf für von nicht sich im dass er es sie ich du wir ihr als auch an aus bei nach wie was wenn oder aber wird kann bitte mir mich hier mein",
    "pt": "o a os as de do da dos das um uma e é que em por para com não se seu sua ao como mais mas ou este esta porque quando muito sem sobre também me até há onde quem desde tudo nos quero você pode",
    "it": "il lo la i gli le di del della un una e è che in per con non si su come più ma o questo questa se perché quando molto senza anche mi dove chi tutto ci voglio puoi sono ho mia mio",
    "nl": "de het een van en is in dat op te met voor niet zijn er aan ook als maar om door over ze hij ik je we deze dit naar kan hoe wat wil heb mijn bij uit worden",
    "pl": "i w na z do nie to że się jest a o jak ale po za co tak dla od przez tylko już czy gdy mnie mi ja ty my wy są być mam może proszę",
    "sv": "och i att det som en på är av för med till den har de inte om ett men var jag du vi han hon kan ska från eller när hur vad min mig så",
    "da": "og i at det en den til er af for med på de har ikke der som men om jeg du vi han hun kan skal fra eller når hvordan hvad min mig så hvor",
    "fi": "ja on ei että se oli hän ne mutta kuin jos kun niin mitä myös vain voi olen olet tämä tässä siitä sekä jotta koska miten minä sinä me te",
    "cs": "a v se na je že o s z do to ale jak který pro tak po jsem jsi jsme byl být mi mě co když nebo už kde prosím",
    "ro": "și de la în cu un o este să nu pe care ce din pentru mai dar ca se am ai are fi dacă cum când unde eu tu noi foarte vreau poți",
    "no": "og i at det en den til er av for med på de har ikke der som men om jeg du vi han hun kan skal fra eller når hvordan hva min meg så hvor",
    "hu": "a az és hogy nem is egy de van ez meg már még csak mint mert vagy ha itt ott én te mi ti ők kell lehet hogyan mit kérlek",
    "hr": "i je se na u da za s su ne li kao ali od do po što koji kada gdje ja ti mi vi oni može molim sam si biti",
    "lt": "ir yra kad ne į su iš ar bet kaip tai jis ji aš tu mes jūs jie būti gali prašau kur kada ką dėl per jei",
}
WORDS = {k: set(v.split()) for k, v in WORDS.items()}

TOKEN = re.compile(r"[^\W\d_]+", re.UNICODE)
MIN_TOKENS = 12        # below this, a message carries too few function words to judge
MIN_LATIN = 0.5        # below this share of latin letters, another writing system rules
MIN_BEST = 0.20        # the winner must be a strong match, else nobody wins and we keep
TOLERANCE = 0.5        # the expected language may score half of the winner and still pass


def latin_share(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    return sum(1 for c in letters if "LATIN" in unicodedata.name(c, "")) / len(letters)


def scores(text):
    """Share of the message's word tokens that are function words, per language."""
    tokens = [t.lower() for t in TOKEN.findall(text)]
    if len(tokens) < MIN_TOKENS:
        return None
    return {lang: sum(1 for t in tokens if t in words) / len(tokens) for lang, words in WORDS.items()}


def check(text, expected):
    """None if the message passes, else why it does not."""
    share = latin_share(text)
    if share < MIN_LATIN:
        return "another writing system (%.0f%% latin letters)" % (100 * share)
    s = scores(text)
    if not s:
        return None
    best = max(s, key=s.get)
    # Finnish, Lithuanian and Czech carry far fewer function words per token than
    # English does, so a weak winner means "unjudgeable", not "wrong language".
    if s[best] < MIN_BEST:
        return None
    if s[expected] < TOLERANCE * s[best]:
        return "%s %.0f%% of function words, %s %.0f%%" % (expected, 100 * s[expected], best, 100 * s[best])
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gold", required=True, help="folder of <lang>.jsonl gold files")
    p.add_argument("--apply", action="store_true", help="write back an off_language flag on the messages that fail")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    total = kept = 0
    for lang in WORDS:
        path = os.path.join(args.gold, lang + ".jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in io.open(path, encoding="utf-8")]
        flagged = 0
        for r in rows:
            why = check(r["text"], lang)
            if why:
                flagged += 1
                if not args.quiet:
                    print("  %-3s %-46s %s" % (lang, why, r["text"].replace("\n", " ")[:64]))
            if args.apply:
                if why:
                    r["off_language"] = True
                else:
                    r.pop("off_language", None)
        total += flagged
        kept += len(rows) - flagged
        print("%-3s %3d off-language of %3d" % (lang, flagged, len(rows)))
        if args.apply:
            with io.open(path, "w", encoding="utf-8") as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d off-language, %d kept%s" % (total, kept, " (written)" if args.apply else ""))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
