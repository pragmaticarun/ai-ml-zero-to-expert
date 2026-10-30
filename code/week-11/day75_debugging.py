"""Day 75: debugging deep nets: initial-loss check, overfit one batch, shuffled-label sanity check, and injected bugs (Fashion-MNIST)."""
import sys, os, math; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import torch, torch.nn as nn
from _mnist import tensors, evaluate
Xtr, ytr, Xte, yte = tensors(fashion=True, n_train=5000); Xtr, Xte = Xtr[:, None], Xte[:, None]
def cnn(): torch.manual_seed(0); return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32*49, 10))
m = cnn()
with torch.no_grad(): l0 = nn.functional.cross_entropy(m(Xtr[:1000]), ytr[:1000]).item()
print(f"1. initial loss {l0:.4f} vs ln(10) = {math.log(10):.4f}")
xb, yb = Xtr[:32], ytr[:32]; o = torch.optim.Adam(m.parameters(), 1e-3)
for it in range(1, 201):
    l = nn.functional.cross_entropy(m(xb), yb); o.zero_grad(); l.backward(); o.step()
    if it in (1, 50, 100, 200): print(f"2. overfit one batch of 32: step {it:3d} loss {l.item():.4f}")
def run(bug=None, labels=None, epochs=3):
    m = cnn(); o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9); y = ytr if labels is None else labels; g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        p = torch.randperm(5000, generator=g)
        for s in range(0, 5000, 64):
            i = p[s:s+64]; out = m(Xtr[i])
            if bug == "softmax_then_CE": out = out.softmax(1)
            l = nn.functional.cross_entropy(out, y[i])
            if bug != "no_zero_grad": o.zero_grad()
            l.backward(); o.step()
    return evaluate(m, Xtr, y)[1], evaluate(m, Xte, yte)[1]
for name, kw in [("correct", {}), ("labels shuffled (sanity check)", dict(labels=ytr[torch.randperm(5000, generator=torch.Generator().manual_seed(1))])),
                 ("bug: softmax before cross_entropy", dict(bug="softmax_then_CE")), ("bug: forgot zero_grad", dict(bug="no_zero_grad"))]:
    tr, te = run(**kw); print(f"3. {name:34s}: train acc {tr:.4f} test acc {te:.4f}")
x = torch.arange(2*2*2*3).reshape(2, 2, 2, 3); good = x.permute(0, 3, 1, 2); bad = x.view(2, 3, 2, 2)
print("4. permute vs view on NHWC->NCHW: equal?", torch.equal(good, bad), "| first channel, permute:", good[0, 0].flatten().tolist(), "view:", bad[0, 0].flatten().tolist())
m = cnn(); l = nn.functional.cross_entropy(m(Xtr[:256]), ytr[:256]); l.backward()
print("5. grad norms per parameter:", {n: round(p.grad.norm().item(), 4) for n, p in m.named_parameters() if n.endswith("weight")})
