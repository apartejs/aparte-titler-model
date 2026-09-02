"""The tagger: an encoder transformer that says keep/discard for each token.

THIS FILE IS MEANT TO BE READ. Every line of computation carries the shape of
its tensors in a comment. Notation:

    B = batch size            L = sequence length (tokens)
    D = model width            H = number of heads          Dh = D / H

The difference from a GPT comes down to one line, flagged further below: a GPT
masks the future (each token only sees what's to its left), an encoder masks
nothing (each token sees the whole message). That's the only structural
difference. The rest of the block is identical.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class Attention(nn.Module):
    """Each token looks at every other token and brings back a mix of them.

    Three roles per token:
      query (Q)  what I'm looking for
      key   (K)  what I offer to the others
      value (V)  what I hand over if I'm chosen
    """

    def __init__(self, D, H):
        super().__init__()
        assert D % H == 0
        self.H, self.Dh = H, D // H
        self.qkv = nn.Linear(D, 3 * D)   # the three projections in one shot
        self.output = nn.Linear(D, D)

    def forward(self, x, mask):
        B, L, D = x.shape
        # a single multiplication produces Q, K, V             (B, L, 3D)
        q, k, v = self.qkv(x).split(D, dim=2)                # 3 x (B, L, D)
        # split the heads: each one works on Dh dimensions
        reshape = lambda t: t.view(B, L, self.H, self.Dh).transpose(1, 2)
        q, k, v = reshape(q), reshape(k), reshape(v)         # 3 x (B, H, L, Dh)

        # dot product of every query with every key           (B, H, L, L)
        # dividing by sqrt(Dh) keeps the scores in a range where softmax
        # stays sensitive: without it, large D saturates the softmax.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.Dh)

        # padding tokens (<pad>) are never allowed to be attended to
        scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

        # ---- THE LINE THAT SETS THIS APART FROM A GPT ----
        # a decoder would add a triangular mask here to forbid looking at the
        # future. We don't: this is an encoder, every token is allowed to see
        # the whole message, both before and after itself.

        weights = scores.softmax(dim=-1)                     # (B, H, L, L)
        mix = weights @ v                                    # (B, H, L, Dh)
        mix = mix.transpose(1, 2).reshape(B, L, D)            # (B, L, D)
        return self.output(mix)


class FeedForward(nn.Module):
    """Widens then narrows. This is where the model "thinks" token by token,
    with no exchange between them: the exchange is attention's job."""

    def __init__(self, D, factor=4):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(D, factor * D), nn.GELU(),
                                 nn.Linear(factor * D, D))

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Normalization, then attention or feed-forward, then addition to the residual.

    The `x + ...` (residual connection) is what makes it possible to stack
    layers: each block ADDS to the representation instead of replacing it, so
    the gradient always finds a direct path back to the input.
    """

    def __init__(self, D, H):
        super().__init__()
        self.n1, self.att = nn.LayerNorm(D), Attention(D, H)
        self.n2, self.ff = nn.LayerNorm(D), FeedForward(D)

    def forward(self, x, mask):
        x = x + self.att(self.n1(x), mask)                   # (B, L, D)
        x = x + self.ff(self.n2(x))                           # (B, L, D)
        return x


class Tagger(nn.Module):
    """ids -> one logit per token. Positive = keep."""
    K_MAX = 6

    def __init__(self, vocab, D=128, H=4, layers=1, max_pos=512, dropout=0.0,
                 positions="learned", dim_emb=0):
        """positions: "learned" (table max_pos x D) or "sinus" (fixed, 0 parameters).
        dim_emb   : 0 = table vocab x D; otherwise table vocab x dim_emb then a
                    projection dim_emb -> D (factorized embeddings, ALBERT)."""
        super().__init__()
        self.max_pos, self.positions, self.dim_emb = max_pos, positions, dim_emb
        if dim_emb:
            self.emb_tok = nn.Embedding(vocab, dim_emb)
            self.proj_emb = nn.Linear(dim_emb, D, bias=False)
        else:
            self.emb_tok = nn.Embedding(vocab, D)
        if positions == "sinus":
            pos = torch.arange(max_pos)[:, None].float()
            freq = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
            table = torch.zeros(max_pos, D)
            table[:, 0::2] = torch.sin(pos * freq)
            table[:, 1::2] = torch.cos(pos * freq)
            self.register_buffer("table_pos", table)
        else:
            self.emb_pos = nn.Embedding(max_pos, D)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(D, H) for _ in range(layers)])
        self.norm = nn.LayerNorm(D)
        self.head = nn.Linear(D, 1)
        # Length head: how many words to keep (1..K_MAX), predicted from the
        # message average. Decoding takes the k best-scored words, with k
        # PREDICTED — no more hand-set fixed budget. A separate head doesn't
        # disturb the ranking, unlike a penalty added to the loss
        # (measured: two length penalties dropped F1 from 0.77 to 0.60-0.66).
        self.length_head = nn.Linear(D, self.K_MAX)
        self.apply(self._init)
        # The head starts at the base rate (~9% of tokens kept): sigmoid(-2.3).
        # Without this, every word starts at 0.5 and a 40-word message "promises"
        # 20 kept words; a length penalty then explodes and the model collapses
        # to "keep nothing" as early as the first epoch (measured: loss 26 instead of 0.8).
        with torch.no_grad():
            self.head.bias.fill_(-2.3)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, ids, mask, words=None, return_k=False):
        """words: (B, L) word number per token, -1 on padding.

        When it is given, the decision is made PER WORD: each token's logit is
        replaced by the average of its word's logits. A word is therefore kept
        or discarded as one block, by construction — and since the loss sees
        these averaged logits, the model learns to score words, not fragments.
        Without `words`, this falls back to a per-token decision.
        """
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device).clamp(max=self.max_pos - 1)
        e = self.emb_tok(ids)                                    # (B, L, D or dim_emb)
        if self.dim_emb:
            e = self.proj_emb(e)                                 # (B, L, D)
        p = self.table_pos[pos] if self.positions == "sinus" else self.emb_pos(pos)
        x = e + p[None, :, :]                                    # (B, L, D)
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, mask)                                  # (B, L, D)
        xn = self.norm(x)
        z = self.head(xn).squeeze(-1)                           # (B, L)
        if words is not None:
            z = self.average_per_word(z, words)
        if not return_k:
            return z
        # length: average of the message's valid tokens -> logits over 1..K_MAX
        m = mask.float().unsqueeze(-1)                           # (B, L, 1)
        avg = (xn * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)    # (B, D)
        return z, self.length_head(avg)                           # (B, K_MAX)

    @staticmethod
    def average_per_word(z, words):
        """Replaces each logit with the average of its word's logits."""
        B, L = z.shape
        valid = words >= 0                                      # (B, L)
        n_words = int(words.max().item()) + 1 if valid.any() else 1
        # a global id per (example, word) so we can sum in a single operation
        key = (torch.arange(B, device=z.device)[:, None] * n_words + words.clamp(min=0))
        key = key.masked_fill(~valid, B * n_words)              # junk bucket
        sums = torch.zeros(B * n_words + 1, device=z.device).index_add_(0, key.flatten(), (z * valid).flatten())
        counts = torch.zeros(B * n_words + 1, device=z.device).index_add_(0, key.flatten(), valid.flatten().float())
        averages = sums / counts.clamp(min=1)
        return torch.where(valid, averages[key], z)

    def num_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        emb = self.emb_tok.weight.numel() + (self.emb_pos.weight.numel() if self.positions != "sinus" else 0)
        return total, emb
