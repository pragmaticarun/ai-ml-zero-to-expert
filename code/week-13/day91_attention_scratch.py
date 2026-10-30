"""Day 91 project: scaled dot-product attention from scratch (NumPy), causal mask, multi-head, checked against PyTorch."""
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0)
def softmax(z): z = z - z.max(-1, keepdims=True); e = np.exp(z); return e / e.sum(-1, keepdims=True)
def attention(Q, K, V, mask=None):
    S = Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None: S = np.where(mask, S, -np.inf)
    W = softmax(S); return W @ V, W
T, d = 5, 8; X = rng.normal(size=(T, d)); Wq, Wk, Wv = (rng.normal(size=(d, d)) / np.sqrt(d) for _ in range(3))
out, W = attention(X @ Wq, X @ Wk, X @ Wv); print("attention weights rows sum to 1:", np.allclose(W.sum(1), 1), "| shape", W.shape)
causal = np.tril(np.ones((T, T), bool)); _, Wc = attention(X @ Wq, X @ Wk, X @ Wv, causal); print("causal weights, upper triangle all zero:", bool((np.triu(Wc, 1) == 0).all())); print(np.round(Wc, 3))
ref = nn.functional.scaled_dot_product_attention(*(torch.tensor(a)[None] for a in (X @ Wq, X @ Wk, X @ Wv)), is_causal=True)[0].numpy()
print(f"max diff vs torch scaled_dot_product_attention (causal): {np.abs(ref - attention(X @ Wq, X @ Wk, X @ Wv, causal)[0]).max():.1e}")
for dk in (4, 64, 1024):
    q, k = rng.normal(size=(1000, dk)), rng.normal(size=(1000, dk)); s = (q * k).sum(1); print(f"d_k={dk:4d}: std of q.k {s.std():.2f}, after /sqrt(d_k) {(s/np.sqrt(dk)).std():.2f}")
print("perm-equivariance: permuting the input rows permutes outputs:", np.allclose(attention(*(X[[2, 0, 1, 4, 3]] @ M for M in (Wq, Wk, Wv)))[0], out[[2, 0, 1, 4, 3]]))
def mha(X, Wq, Wk, Wv, Wo, h):
    T, D = X.shape; sp = lambda A: A.reshape(T, h, D // h).transpose(1, 0, 2)
    o, _ = attention(sp(X @ Wq), sp(X @ Wk), sp(X @ Wv)); return o.transpose(1, 0, 2).reshape(T, D) @ Wo
torch.manual_seed(0); D, h = 16, 4; ref = nn.MultiheadAttention(D, h, bias=False, batch_first=True); Xm = rng.normal(size=(6, D))
Wi = ref.in_proj_weight.detach().numpy(); Wo = ref.out_proj.weight.detach().numpy()
mine = mha(Xm, Wi[:D].T, Wi[D:2*D].T, Wi[2*D:].T, Wo.T, h); theirs = ref(*(torch.tensor(Xm, dtype=torch.float)[None],) * 3)[0][0].detach().numpy()
print(f"my multi-head attention vs nn.MultiheadAttention: max diff {np.abs(mine - theirs).max():.1e}; params {sum(p.numel() for p in ref.parameters())} = 4*D^2 = {4*D*D}")
