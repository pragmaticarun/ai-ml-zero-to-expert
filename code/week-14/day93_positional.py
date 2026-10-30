"""Day 93: positional encodings. Attention alone is permutation-equivariant; sinusoidal and RoPE fix that. Checks RoPE's relative-position property."""
import math, torch
torch.manual_seed(0)
def attn(x): S = x @ x.T / math.sqrt(x.shape[1]); return S.softmax(-1) @ x
x = torch.randn(5, 8); p = torch.tensor([3, 1, 4, 0, 2])
print("no positions: permuting inputs just permutes outputs:", torch.allclose(attn(x)[p], attn(x[p]), atol=1e-6))
def sinusoidal(T, d):
    pos = torch.arange(T)[:, None]; i = torch.arange(0, d, 2); ang = pos / 10000 ** (i / d)
    pe = torch.zeros(T, d); pe[:, 0::2], pe[:, 1::2] = torch.sin(ang), torch.cos(ang); return pe
pe = sinusoidal(64, 16); print("sinusoidal: shape", tuple(pe.shape), "| values in [-1,1]:", bool(pe.abs().max() <= 1))
pe8 = sinusoidal(5, 8); xp = x + pe8; print("with sinusoidal added: permutation-equivariant?", torch.allclose(attn(xp)[p], attn(x[p] + pe8), atol=1e-6))
d = pe @ pe.T; print("PE dot products from position 10:", [round(d[10, 10 + k].item(), 2) for k in (0, 1, 2, 4, 8, 16)])
print("same offset, different start (pos 10->14 vs 40->44):", round(d[10, 14].item(), 4), round(d[40, 44].item(), 4))
def rope(x, pos, base=10000.0):
    d = x.shape[-1]; th = pos[:, None] / base ** (torch.arange(0, d, 2) / d); c, s = th.cos(), th.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]; out = torch.empty_like(x); out[..., 0::2] = x1 * c - x2 * s; out[..., 1::2] = x1 * s + x2 * c; return out
q, k = torch.randn(1, 16), torch.randn(1, 16)
dots = [(rope(q, torch.tensor([m + 0.0])) @ rope(k, torch.tensor([m - 3.0])).T).item() for m in (3, 10, 100, 1000)]
print("RoPE q.k with offset 3 at positions 3,10,100,1000:", [round(v, 5) for v in dots])
print("RoPE preserves vector norms:", torch.allclose(rope(q, torch.tensor([7.0])).norm(), q.norm()))
qq = torch.ones(1, 16); kk = torch.ones(1, 16)
print("RoPE q.k for identical all-ones q,k by distance:", [round((rope(qq, torch.tensor([0.0])) @ rope(kk, torch.tensor([float(t)])).T).item(), 2) for t in (0, 1, 4, 16, 64, 256)])
