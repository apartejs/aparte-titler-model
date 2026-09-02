"""Training the tagger. Used for steps 2, 3 and for the ablations.

    python train/train.py --layers 1 --name step2
    python train/train.py --layers 4 --name step3

Rules fixed by earlier measurements:
  - weighted loss: only 8% positives, otherwise the model learns to discard everything
  - batches grouped by length: 99% useful compute instead of 21%
  - threshold AND budget chosen on `val`, never on the gold set
"""
import argparse
import json
import os
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "eval"))
import data  # noqa: E402
import score as S  # noqa: E402
from model import Tagger  # noqa: E402
from labels import word_ids_per_token  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

TOKENIZER = os.environ.get("TITLER_TOKENIZER", os.path.join(HERE, "tokenizer.json"))
RUNS = os.path.join(ROOT, "runs")


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def probs(model, examples, dev, budget_tokens=32768):
    """Per-token probabilities, in batches grouped by length."""
    model.eval()
    out = [None] * len(examples)
    indices = sorted(range(len(examples)), key=lambda i: len(examples[i][0]))
    i = 0
    with torch.no_grad():
        while i < len(indices):
            # the chunk grows as long as (size x longest length) fits within
            # the budget. The old version sized the chunk on the SHORTEST
            # message: 16,000 examples padded to 512 in a single batch, 57 GB
            # of RAM.
            chunk = [indices[i]]
            i += 1
            while i < len(indices) and (len(chunk) + 1) * len(examples[indices[i]][0]) <= budget_tokens:
                chunk.append(indices[i])
                i += 1
            X, _, M, W = data.pad_batch([examples[j] for j in chunk])
            X = torch.tensor(X, device=dev)
            M = torch.tensor(M, dtype=torch.bool, device=dev)
            W = torch.tensor(W, device=dev)
            p = torch.sigmoid(model(X, M, W)).cpu().tolist()
            for j, row in zip(chunk, p):
                out[j] = row[:len(examples[j][0])]
    return out


def evaluate_val(pv, val):
    """Best threshold and best budget, measured on val."""
    def f_threshold(s):
        tp = fp = fn = 0
        for pr, ex in zip(pv, val):
            y = ex[1]
            for x, yy in zip(pr, y):
                p = x >= s
                tp += p and yy
                fp += p and not yy
                fn += yy and not p
        return prf(tp, fp, fn)
    best = max(((s, f_threshold(s)) for s in [i / 100 for i in range(5, 96, 5)]),
               key=lambda t: t[1][2])
    return best[0], best[1]


def on_gold(name, model, dev, tok, threshold, budget=None, detail=True):
    gold = S.load_gold(tok)

    def predictor(g):
        with torch.no_grad():
            X = torch.tensor([g["ids"]], device=dev)
            M = torch.ones_like(X, dtype=torch.bool)
            W = torch.tensor([word_ids_per_token(g["text"], g["offsets"])], device=dev)
            z, k_logits = model(X, M, W, return_k=True)
            pr = torch.sigmoid(z)[0].cpu().tolist()
            k_pred = int(k_logits[0].argmax().item()) + 1
        if budget == "k":
            budget_eff = k_pred
        else:
            budget_eff = budget
        if budget_eff:
            words = []
            for gr in S.word_groups(g["text"], g["offsets"]):
                a, b = g["offsets"][gr[0]][0], g["offsets"][gr[-1]][1]
                if any(c.isalnum() for c in g["text"][a:b]):
                    words.append(gr)
            top = sorted(words, key=lambda w: sum(pr[i] for i in w) / len(w),
                         reverse=True)[:budget_eff]
            kept = {i for w in top for i in w}
            return [1 if i in kept else 0 for i in range(len(g["ids"]))]
        # no snapping to words: the logits are already constant per word
        return [1 if x >= threshold else 0 for x in pr]
    return S.evaluate(name, predictor, gold, detail=detail)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--budget-tokens", type=int, default=16384)
    p.add_argument("--name", default="run")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--pos-weight", type=float, default=0.0,
                   help="weight of the positives; 0 = (negatives/positives) as before")
    p.add_argument("--length", type=float, default=0.0,
                   help="weight of the length penalty: sum of per-word probabilities outside [3, 6]")
    p.add_argument("--positions", default="learned", choices=["learned", "sinus"])
    p.add_argument("--dim-emb", type=int, default=0, help="factorized embeddings (0 = off)")
    p.add_argument("--head-k", type=float, default=0.0,
                   help="weight of the length head (0 = disabled)")
    p.add_argument("--fixed-threshold", type=float, default=0.0,
                   help="decide at this threshold on the gold set (0 = best threshold on val)")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = Tokenizer.from_file(TOKENIZER)
    train, val = data.load(limit=args.limit)
    model = Tagger(tok.get_vocab_size(), args.dim, args.heads, args.layers,
                   dropout=args.dropout, positions=args.positions, dim_emb=args.dim_emb).to(dev)
    total, emb = model.num_parameters()
    print("%s : %d layer(s), D=%d, H=%d | %s parameters (of which %s are embeddings, %.0f%%)"
          % (args.name, args.layers, args.dim, args.heads,
             format(total, ","), format(emb, ","), 100 * emb / total))
    print("train %s | val %s | device %s"
          % (format(len(train), ","), format(len(val), ","), dev))

    pos = sum(sum(ex[1]) for ex in train)
    tot = sum(len(ex[1]) for ex in train)
    pw = torch.tensor([args.pos_weight if args.pos_weight > 0 else (tot - pos) / pos], device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    batches_per_epoch = len(data.batches(train, args.budget_tokens))
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * batches_per_epoch, pct_start=0.15)

    best = {"f1": -1.0, "threshold": 0.5, "epoch": 0}   # the 1st epoch always gets saved
    global_step = 0
    for ep in range(args.epochs):
        model.train()
        t0, loss_sum, n = time.perf_counter(), 0.0, 0
        for batch in data.batches(train, args.budget_tokens, seed=ep):
            X, Y, M, W = data.pad_batch(batch)
            X = torch.tensor(X, device=dev)
            Y = torch.tensor(Y, dtype=torch.float32, device=dev)
            Mb = torch.tensor(M, dtype=torch.bool, device=dev)
            W = torch.tensor(W, device=dev)
            Mf = Mb.float()
            logits, k_logits = model(X, Mb, W, return_k=True)
            loss = (loss_fn(logits, Y) * Mf).sum() / Mf.sum()
            if args.head_k > 0:
                # target: number of words kept in the label (1..K_MAX)
                word_start = torch.ones_like(W, dtype=torch.bool)
                word_start[:, 1:] = W[:, 1:] != W[:, :-1]
                k_target = ((Y > 0.5) & word_start & (W >= 0)).sum(dim=1).clamp(min=1, max=model.K_MAX) - 1
                loss = loss + args.head_k * torch.nn.functional.cross_entropy(k_logits, k_target)
            if args.length > 0:
                # Learned length, on what we actually decode: among the message's
                # words ranked by logit, the 3rd must clear the threshold
                # (logit >= +margin) and the 7th must stay below it
                # (logit <= -margin). Constraining the sum of the probabilities
                # isn't enough: 40 words at 0.09 sum to 3.6 without any of them
                # clearing 0.5.
                word_start = torch.ones_like(W, dtype=torch.bool)
                word_start[:, 1:] = W[:, 1:] != W[:, :-1]
                word_start &= W >= 0
                z_words = logits.masked_fill(~word_start, float("-inf"))
                sorted_z = z_words.sort(dim=1, descending=True).values      # (B, L)
                n_words = word_start.sum(dim=1)                              # (B,)
                margin = 1.0
                # 3rd word (or the last one if the message has fewer than 3)
                i3 = (n_words - 1).clamp(max=2)
                z3 = sorted_z.gather(1, i3[:, None]).squeeze(1)
                low = torch.relu(margin - z3)
                # 7th word, only if the message has at least 7
                a7 = n_words >= 7
                z7 = sorted_z[:, 6].masked_fill(~a7, float("-inf"))
                high = torch.relu(z7 + margin).masked_fill(~a7, 0.0)
                warmup = min(1.0, global_step / max(1, batches_per_epoch))
                loss = loss + args.length * warmup * (low + high).mean()
            global_step += 1
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * Mf.sum().item()
            n += Mf.sum().item()
            if global_step % 200 == 0 and dev == "cuda":
                # returns to the system the blocks the allocator keeps cached
                # (integrated GPU: its memory is the machine's RAM)
                torch.cuda.empty_cache()
        pv = probs(model, val, dev)
        threshold, (val_p, val_r, val_f) = evaluate_val(pv, val)
        marker = ""
        if val_f > best["f1"]:
            best = {"f1": val_f, "threshold": threshold, "epoch": ep + 1}
            os.makedirs(RUNS, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(RUNS, args.name + ".pt"))
            marker = "  <- best, saved"
        print("  epoch %d: loss %.4f | val P %.2f R %.2f F1 %.2f (threshold %.2f) | %.0fs%s"
              % (ep + 1, loss_sum / n, val_p, val_r, val_f, threshold, time.perf_counter() - t0, marker))

    model.load_state_dict(torch.load(os.path.join(RUNS, args.name + ".pt")))
    model.eval()
    if args.fixed_threshold > 0:
        best["threshold"] = args.fixed_threshold
    print("\n--- final exam on the gold set (never seen, epoch %d) ---" % best["epoch"])
    S.evaluate("heuristic baseline", S.baseline_reference(tok), S.load_gold(tok))
    f_threshold = on_gold("%s, threshold %.2f" % (args.name, best["threshold"]),
                          model, dev, tok, best["threshold"])
    f_budget = {k: on_gold("%s, budget %d words" % (args.name, k), model, dev, tok,
                           best["threshold"], budget=k, detail=(k == 5))
                for k in (4, 5, 6)}
    if args.head_k > 0:
        f_budget["k"] = on_gold("%s, k LEARNED" % args.name, model, dev, tok, best["threshold"], budget="k")
    # the project's reference metric: against the best of the 5 references
    cons = S.load_consensus()
    if cons is not None:
        gold = S.load_gold(tok)
        for k in (None, 6) + (("k",) if args.head_k > 0 else ()):
            def pred(g, k=k):
                with torch.no_grad():
                    X = torch.tensor([g["ids"]], device=dev)
                    W = torch.tensor([word_ids_per_token(g["text"], g["offsets"])], device=dev)
                    z, kl = model(X, torch.ones_like(X, dtype=torch.bool), W, return_k=True)
                    pr = torch.sigmoid(z)[0].cpu().tolist()
                if k == "k":
                    k = int(kl[0].argmax().item()) + 1
                if k:
                    words = [gr for gr in S.word_groups(g["text"], g["offsets"])
                            if any(c.isalnum() for c in g["text"][g["offsets"][gr[0]][0]:g["offsets"][gr[-1]][1]])]
                    top = sorted(words, key=lambda w: sum(pr[i] for i in w) / len(w), reverse=True)[:k]
                    kept = {i for w in top for i in w}
                    return [1 if i in kept else 0 for i in range(len(g["ids"]))]
                return [1 if x >= best["threshold"] else 0 for x in pr]
            S.evaluate_multi("%s, %s" % (args.name, {None: "threshold", 6: "budget 6", "k": "k learned"}[k]), pred, gold, cons)
    json.dump({"name": args.name, "layers": args.layers, "dim": args.dim,
               "positions": args.positions, "dim_emb": args.dim_emb, "vocab": tok.get_vocab_size(),
               "head_k": args.head_k,
               "parameters": total, "f1_val": best["f1"], "threshold": best["threshold"],
               "f1_gold_threshold": f_threshold,
               "f1_gold_budget": {str(k): v for k, v in f_budget.items()}},
              open(os.path.join(RUNS, args.name + ".json"), "w"), indent=2)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
