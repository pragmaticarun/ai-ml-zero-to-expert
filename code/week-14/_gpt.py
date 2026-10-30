"""Shared tiny GPT (Days 94-119). Pre-LN decoder-only transformer, CPU-scale. Optional KV cache for Day 111."""
import math, torch, torch.nn as nn, torch.nn.functional as F
class Attention(nn.Module):
    def __init__(s, d, h, drop=0.0):
        super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d, bias=False); s.o = nn.Linear(d, d, bias=False); s.drop = drop
    def forward(s, x, cache=None):
        B, T, D = x.shape; q, k, v = s.qkv(x).view(B, T, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        if cache is not None:                       # cache = dict with 'k','v' of shape (B,h,T_past,dh)
            if "k" in cache: k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v
        Tk = k.shape[2]; S = q @ k.transpose(-1, -2) / math.sqrt(D // s.h)
        mask = torch.ones(T, Tk, dtype=torch.bool).tril(Tk - T); S = S.masked_fill(~mask, float("-inf"))
        W = F.dropout(S.softmax(-1), s.drop, s.training); return s.o((W @ v).transpose(1, 2).reshape(B, T, D))
class Block(nn.Module):
    def __init__(s, d, h, drop=0.0):
        super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); s.attn = Attention(d, h, drop)
        s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(drop))
    def forward(s, x, cache=None): x = x + s.attn(s.ln1(x), cache); return x + s.mlp(s.ln2(x))
class GPT(nn.Module):
    def __init__(s, vocab, ctx=128, d=128, h=4, L=4, drop=0.0):
        super().__init__(); s.ctx = ctx; s.tok = nn.Embedding(vocab, d); s.pos = nn.Embedding(ctx, d)
        s.blocks = nn.ModuleList(Block(d, h, drop) for _ in range(L)); s.ln = nn.LayerNorm(d); s.head = nn.Linear(d, vocab, bias=False)
        s.head.weight = s.tok.weight; s.apply(s._init)
    def _init(s, m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, 0, 0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, idx, targets=None, caches=None, start=0):
        T = idx.shape[1]; x = s.tok(idx) + s.pos(torch.arange(start, start + T))
        for i, b in enumerate(s.blocks): x = b(x, None if caches is None else caches[i])
        logits = s.head(s.ln(x))
        return logits if targets is None else (logits, F.cross_entropy(logits.flatten(0, 1), targets.flatten()))
    @torch.no_grad()
    def generate(s, idx, n, temperature=1.0, top_k=None, use_cache=True, greedy=False, gen=None):
        s.eval(); caches = [dict() for _ in s.blocks] if use_cache else None; start = 0; inp = idx
        for _ in range(n):
            if use_cache and start + inp.shape[1] > s.ctx:   # context full: drop the cache and re-prefill the last ctx/2 tokens
                caches = [dict() for _ in s.blocks]; inp = idx[:, -(s.ctx // 2):]; start = 0
            if use_cache: logits = s(inp, caches=caches, start=start)[:, -1]; start += inp.shape[1]
            else: logits = s(idx[:, -s.ctx:])[:, -1]
            if greedy: nxt = logits.argmax(-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k: v, _ = logits.topk(top_k); logits[logits < v[:, [-1]]] = float("-inf")
                nxt = torch.multinomial(logits.softmax(-1), 1, generator=gen)
            idx = torch.cat([idx, nxt], 1); inp = nxt
        return idx
def nparams(m): return sum(p.numel() for p in m.parameters())
def batches(data, ctx, bs, g):
    i = torch.randint(len(data) - ctx - 1, (bs,), generator=g); return torch.stack([data[j:j+ctx] for j in i]), torch.stack([data[j+1:j+ctx+1] for j in i])
@torch.no_grad()
def eval_loss(m, data, ctx, bs=32, n=20, seed=123):
    m.eval(); g = torch.Generator().manual_seed(seed); return sum(m(*batches(data, ctx, bs, g))[1].item() for _ in range(n)) / n
def train(m, tr, va, ctx, steps, bs=32, lr=3e-3, seed=0, log_every=None, wd=0.1):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(o, lr, total_steps=steps, pct_start=0.1); hist = []
    for st in range(1, steps + 1):
        m.train(); x, y = batches(tr, ctx, bs, g); _, l = m(x, y); o.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step(); sch.step()
        if log_every and (st % log_every == 0 or st == steps): v = eval_loss(m, va, ctx); hist.append((st, l.item(), v)); print(f"  step {st:5d} train {l.item():.3f} val {v:.3f}", flush=True)
    return hist
def char_data():
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from common import load_shakespeare; text = load_shakespeare(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data)); return text, chars, data[:n], data[n:]
