"""A trained model, scored on ANY gold set.

The trainer only scores against the gold set of its own language. Here we
cross them: the EN+FR model on the English gold set AND on the French gold
set, the FR-only model on the French gold set, the chosen English model on
the English gold set — the same decoding as the trainer (budget of 6 words,
average token score per word).

    python eval/evaluate_run.py runs/enfr_emb8.pt --tokenizer train/tokenizer_enfr_3072.json \
        --gold eval/fr_gold.jsonl --consensus eval/fr_gold_consensus_enfr.jsonl
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("weights")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--gold", default=os.path.join(HERE, "gold.jsonl"))
    p.add_argument("--consensus", default="")
    p.add_argument("--budget", type=int, default=6)
    args = p.parse_args()

    # score.py reads its paths from the environment at import time
    os.environ["TITLER_TOKENIZER"] = args.tokenizer
    os.environ["TITLER_GOLD"] = args.gold
    os.environ["TITLER_CONSENSUS"] = args.consensus or os.path.join(HERE, "does_not_exist.jsonl")
    sys.path.insert(0, HERE)
    sys.path.insert(0, os.path.join(ROOT, "train"))
    import torch
    import score as S
    from labels import word_ids_per_token
    from titler import Titler

    t = Titler(tokenizer=args.tokenizer, weights=args.weights)
    gold = S.load_gold(t.tok)

    def pred(g):
        with torch.no_grad():
            X = torch.tensor([g["ids"]], device=t.dev)
            W = torch.tensor([word_ids_per_token(g["text"], g["offsets"])], device=t.dev)
            z = t.model(X, torch.ones_like(X, dtype=torch.bool), W)
            pr = torch.sigmoid(z)[0].cpu().tolist()
        words = [gr for gr in S.word_groups(g["text"], g["offsets"])
                if any(c.isalnum() for c in g["text"][g["offsets"][gr[0]][0]:g["offsets"][gr[-1]][1]])]
        top = sorted(words, key=lambda w: sum(pr[i] for i in w) / len(w), reverse=True)[:args.budget]
        kept = {i for w in top for i in w}
        return [1 if i in kept else 0 for i in range(len(g["ids"]))]

    name = "%s on %s" % (os.path.basename(args.weights)[:-3], os.path.basename(args.gold))
    print("gold: %d messages" % len(gold))
    S.evaluate(name + ", budget %d" % args.budget, pred, gold, detail=True)
    if args.consensus and os.path.exists(args.consensus):
        S.evaluate_multi(name, pred, gold)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
