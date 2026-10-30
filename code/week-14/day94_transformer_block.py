"""Day 94: the transformer block (pre-LN): residual stream, layer norm, MLP. Parameter accounting and pre-LN vs post-LN gradients at depth."""
import sys, os, torch, torch.nn as nn; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gpt import Block, GPT, nparams, Attention
torch.manual_seed(0); d, h = 128, 4; b = Block(d, h)
print(f"one block d={d}: attention {nparams(b.attn):,} (4d^2={4*d*d:,}), MLP {nparams(b.mlp):,} (8d^2+5d={8*d*d+5*d:,}), 2 LayerNorms {nparams(b.ln1)+nparams(b.ln2)}, total {nparams(b):,}")
m = GPT(65, ctx=128, d=d, h=h, L=4); print(f"GPT vocab 65 ctx 128 d 128 L 4: {nparams(m):,} params (embeddings shared with output head: {m.head.weight is m.tok.weight})")
class PostLN(nn.Module):
    def __init__(s, d, h): super().__init__(); s.ln1, s.ln2 = nn.LayerNorm(d), nn.LayerNorm(d); s.attn = Attention(d, h); s.mlp = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
    def forward(s, x): x = s.ln1(x + s.attn(x)); return s.ln2(x + s.mlp(x))
for name, B in (("pre-LN", Block), ("post-LN", PostLN)):
    for L in (2, 12, 24):
        torch.manual_seed(0); net = nn.Sequential(*[B(64, 4) for _ in range(L)]); x = torch.randn(8, 32, 64)
        r = torch.randn(8, 32, 64, generator=torch.Generator().manual_seed(1)); (net(x) * r).mean().backward(); g = [blk.attn.qkv.weight.grad.norm().item() for blk in net]
        print(f"{name:7s} L={L:2d}: grad norm of first-block qkv {g[0]:.2e}, last-block {g[-1]:.2e}, ratio first/last {g[0]/g[-1]:.2f}")
torch.manual_seed(0); net = nn.Sequential(*[Block(64, 4) for _ in range(12)]); x = torch.randn(8, 32, 64); norms = []
for blk in net: x = blk(x); norms.append(x.norm(dim=-1).mean().item())
print("pre-LN residual stream norm after blocks 1,4,8,12:", [round(norms[i], 2) for i in (0, 3, 7, 11)])
# Training stability: 8 blocks, d=64, lr 3e-3 with NO warmup, 150 steps on Tiny Shakespeare characters.
from _gpt import char_data, batches
_, chars, tr, va = char_data()
class TinyLM(nn.Module):
    def __init__(s, B, L=8, d=64): super().__init__(); s.e = nn.Embedding(len(chars), d); s.p = nn.Embedding(64, d); s.b = nn.Sequential(*[B(d, 4) for _ in range(L)]); s.ln = nn.LayerNorm(d); s.out = nn.Linear(d, len(chars))
    def forward(s, x): return s.out(s.ln(s.b(s.e(x) + s.p(torch.arange(x.shape[1])))))
for name, B in (("pre-LN", Block), ("post-LN", PostLN)):
    for lr in (3e-3, 1e-2):
        torch.manual_seed(0); m = TinyLM(B); o = torch.optim.AdamW(m.parameters(), lr=lr); g = torch.Generator().manual_seed(0); losses = []
        for st in range(150):
            x, y = batches(tr, 64, 32, g); l = nn.functional.cross_entropy(m(x).flatten(0, 1), y.flatten()); o.zero_grad(); l.backward(); o.step(); losses.append(l.item())
        print(f"{name:7s} lr {lr:.0e}, no warmup: loss at step 1 {losses[0]:.3f}, mean of last 20 steps {sum(losses[-20:])/20:.3f}")
