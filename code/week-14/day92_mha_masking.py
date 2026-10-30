"""Day 92: self-attention and multi-head attention in PyTorch, with causal and padding masks, checked against torch."""
import math, time, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); B, T, D, h = 2, 6, 16, 4
def mha(x, Wqkv, Wo, h, mask=None):
    B, T, D = x.shape; q, k, v = (x @ Wqkv.T).view(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)
    S = q @ k.transpose(-1, -2) / math.sqrt(D // h)
    if mask is not None: S = S.masked_fill(~mask, float("-inf"))
    W = S.softmax(-1); return ((W @ v).transpose(1, 2).reshape(B, T, D)) @ Wo.T, W
ref = nn.MultiheadAttention(D, h, bias=False, batch_first=True); x = torch.randn(B, T, D)
Wqkv, Wo = ref.in_proj_weight.detach(), ref.out_proj.weight.detach()
causal = torch.ones(T, T, dtype=torch.bool).tril()
mine, W = mha(x, Wqkv, Wo, h, causal); theirs = ref(x, x, x, attn_mask=~causal, need_weights=False)[0]
print(f"causal MHA vs nn.MultiheadAttention: max diff {(mine - theirs).abs().max():.1e}")
print("causal: weights above diagonal all zero:", bool((W.triu(1) == 0).all()))
lengths = torch.tensor([6, 4]); pad = torch.arange(T)[None] < lengths[:, None]          # True = real token
m2 = (causal[None] & pad[:, None, :])[:, None]                                         # (B,1,T,T): no query may look at padding keys
out, W2 = mha(x, Wqkv, Wo, h, m2)
print("sequence 2 (length 4): weight on padded keys:", W2[1, :, :, 4:].abs().max().item())
alone, _ = mha(x[1:, :4], Wqkv, Wo, h, causal[:4, :4]); print(f"padded batch vs the short sequence alone (real positions): max diff {(out[1, :4] - alone[0]).abs().max():.1e}")
print("a fully masked row gives NaN:", torch.isnan(torch.tensor([float('-inf')] * 3).softmax(0)).all().item())
print(f"F.scaled_dot_product_attention (is_causal) vs mine: max diff {(F.scaled_dot_product_attention(*[t for t in (x @ Wqkv.T).view(B, T, 3, h, D // h).permute(2, 0, 3, 1, 4)], is_causal=True).transpose(1, 2).reshape(B, T, D) @ Wo.T - mine).abs().max():.1e}")
print(f"params: 4*D^2 = {4 * D * D}, nn.MultiheadAttention(bias=False) = {sum(p.numel() for p in ref.parameters())}")
for T2 in (256, 512, 1024, 2048):
    q = torch.randn(1, 4, T2, 64); t = time.perf_counter()
    for _ in range(3): S = (q @ q.transpose(-1, -2)).softmax(-1) @ q
    print(f"T={T2:5d}: naive attention {1000 * (time.perf_counter() - t) / 3:7.1f} ms, score matrix {4 * T2 * T2 * 4 / 2**20:6.1f} MiB (fp32, 4 heads)")
