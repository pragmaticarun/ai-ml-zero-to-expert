"""Day 64: tensors, autograd (checked against Day 63's hand-derived NumPy gradients), nn.Module and a training loop on digits."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn, time
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
torch.manual_seed(0)
# 1. tensors and broadcasting
a = torch.arange(6.).reshape(2, 3); b = torch.tensor([10., 20., 30.])
print("a+b =", (a + b).tolist(), "| dtype", a.dtype, "| shares memory with numpy:", np.shares_memory(a.numpy(), a.numpy()))
# 2. autograd on a scalar function: f = x^3 + 2xy, df/dx = 3x^2+2y, df/dy = 2x
x = torch.tensor(2., requires_grad=True); y = torch.tensor(-1., requires_grad=True); f = x**3 + 2*x*y; f.backward()
print(f"f={f.item():.1f} df/dx={x.grad.item():.1f} (expect 10.0) df/dy={y.grad.item():.1f} (expect 4.0)")
# 3. gradients accumulate unless zeroed
f2 = x**3 + 2*x*y; f2.backward(); print("after a second backward without zeroing, x.grad =", x.grad.item(), "(doubled)")
# 4. autograd vs hand-derived backprop (Day 63 network)
X, yy = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.25, random_state=0, stratify=yy)
g = np.random.default_rng(1); W1 = g.normal(size=(64, 16)) * np.sqrt(2/64); W2 = g.normal(size=(16, 10)) * np.sqrt(1/16); b1 = np.zeros(16); b2 = np.zeros(10)
Xs, ys = Xtr[:20], ytr[:20]; m = 20
Z1 = Xs @ W1 + b1; A1 = np.maximum(0, Z1); Z2 = A1 @ W2 + b2; P = np.exp(Z2 - Z2.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
dZ2 = P.copy(); dZ2[np.arange(m), ys] -= 1; dZ2 /= m; dW2 = A1.T @ dZ2; dW1 = Xs.T @ ((dZ2 @ W2.T) * (Z1 > 0))
t = {k: torch.tensor(v, requires_grad=True) for k, v in dict(W1=W1, b1=b1, W2=W2, b2=b2).items()}
logits = torch.relu(torch.tensor(Xs) @ t["W1"] + t["b1"]) @ t["W2"] + t["b2"]
nn.functional.cross_entropy(logits, torch.tensor(ys)).backward()
for k, ref in (("W1", dW1), ("W2", dW2)): print(f"autograd vs NumPy {k}: max abs diff {np.abs(t[k].grad.numpy() - ref).max():.2e}")
# 5. nn.Module + training loop
class MLP(nn.Module):
    def __init__(s, h=64): super().__init__(); s.net = nn.Sequential(nn.Linear(64, h), nn.ReLU(), nn.Linear(h, 10))
    def forward(s, x): return s.net(x)
model = MLP(); print("parameters:", sum(p.numel() for p in model.parameters()))
opt = torch.optim.SGD(model.parameters(), lr=0.2); Xt, yt = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr); Xv, yv = torch.tensor(Xte, dtype=torch.float32), torch.tensor(yte)
t0 = time.perf_counter()
for ep in range(60):
    model.train(); perm = torch.randperm(len(yt))
    for s in range(0, len(perm), 32):
        i = perm[s:s+32]; loss = nn.functional.cross_entropy(model(Xt[i]), yt[i]); opt.zero_grad(); loss.backward(); opt.step()
    if ep + 1 in (1, 5, 20, 60):
        model.eval()
        with torch.no_grad(): print(f"epoch {ep+1:2d} train loss {nn.functional.cross_entropy(model(Xt), yt).item():.4f} test acc {(model(Xv).argmax(1) == yv).float().mean().item():.4f}")
print(f"60 epochs took {time.perf_counter()-t0:.1f} s")
