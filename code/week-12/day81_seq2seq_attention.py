"""Day 81: seq2seq (GRU encoder-decoder) on sequence reversal, without and with dot-product attention; accuracy vs length."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
V, H, SOS = 12, 64, 10   # tokens 0-9, 10 = start symbol
def batch(n, g, lo=4, hi=16):
    L = torch.randint(lo, hi + 1, (1,), generator=g).item(); x = torch.randint(0, 10, (n, L), generator=g); return x, x.flip(1)
class Seq2Seq(nn.Module):
    def __init__(s, attn): super().__init__(); s.attn = attn; s.emb = nn.Embedding(V, H); s.enc = nn.GRU(H, H, batch_first=True); s.dec = nn.GRUCell(H + (H if attn else 0), H); s.out = nn.Linear(H * (2 if attn else 1), V)
    def forward(s, x, y):
        E, h = s.enc(s.emb(x)); h = h[0]; inp = torch.full((len(x),), SOS); logits = []; s.last_w = []
        for t in range(y.shape[1]):
            e = s.emb(inp)
            if s.attn:
                w = torch.softmax((E @ h[:, :, None]).squeeze(2) / H ** 0.5, 1); ctx = (w[:, :, None] * E).sum(1); s.last_w.append(w)
                h = s.dec(torch.cat([e, ctx], 1)); o = s.out(torch.cat([h, ctx], 1))
            else: h = s.dec(e, h); o = s.out(h)
            logits.append(o); inp = y[:, t] if s.training else o.argmax(1)    # teacher forcing in training, greedy at test
        return torch.stack(logits, 1)
for attn in (False, True):
    torch.manual_seed(0); m = Seq2Seq(attn); o = torch.optim.Adam(m.parameters(), 2e-3); g = torch.Generator().manual_seed(0); t = time.perf_counter()
    for step in range(1500):
        x, y = batch(64, g); l = nn.functional.cross_entropy(m(x, y).reshape(-1, V), y.reshape(-1)); o.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
    m.eval(); res = []
    with torch.no_grad():
        for L in (8, 16, 24):
            x, y = batch(500, torch.Generator().manual_seed(L), L, L); p = m(x, y).argmax(2); res.append(f"len {L}: token acc {(p == y).float().mean():.3f}, exact {(p == y).all(1).float().mean():.3f}")
    print(f"{'with' if attn else 'without':7s} attention ({sum(q.numel() for q in m.parameters()):,} params, {time.perf_counter()-t:.0f} s):", " | ".join(res))
    if attn:
        x, y = batch(1, torch.Generator().manual_seed(5), 8, 8); m(x, y); W = torch.stack(m.last_w, 1)[0]
        print("attention argmax per output step (expect 7,6,...,0 for reversal):", W.argmax(1).tolist())
