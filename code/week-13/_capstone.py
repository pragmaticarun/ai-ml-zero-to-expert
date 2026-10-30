"""Month-3 capstone shared code: a small ResNet for CIFAR-10-small (32x32) and a configurable trainer. CPU scale."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-11"))
import torch, torch.nn as nn
from _cifar import splits, flip_crop
class Block(nn.Module):
    def __init__(s, cin, cout, stride, bn=True, residual=True):
        super().__init__(); N = (lambda c: nn.BatchNorm2d(c)) if bn else (lambda c: nn.Identity()); s.residual = residual
        s.f = nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=not bn), N(cout), nn.ReLU(), nn.Conv2d(cout, cout, 3, 1, 1, bias=not bn), N(cout))
        s.skip = nn.Identity() if (stride == 1 and cin == cout) else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=not bn), N(cout))
    def forward(s, x): y = s.f(x); return torch.relu(y + s.skip(x) if s.residual else y)
def make_model(width=32, bn=True, residual=True, seed=0):
    torch.manual_seed(seed); w = width
    return nn.Sequential(nn.Conv2d(3, w, 3, 1, 1, bias=not bn), nn.BatchNorm2d(w) if bn else nn.Identity(), nn.ReLU(),
                         Block(w, w, 1, bn, residual), Block(w, 2 * w, 2, bn, residual), Block(2 * w, 4 * w, 2, bn, residual),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(4 * w, 10))
@torch.no_grad()
def predict(m, X, tta=False):
    m.eval(); out = []
    for i in range(0, len(X), 500):
        p = m(X[i:i+500]).softmax(1)
        if tta: p = (p + m(X[i:i+500].flip(3)).softmax(1)) / 2
        out.append(p)
    return torch.cat(out)
def fit(width=32, bn=True, residual=True, aug=True, wd=5e-4, ls=0.0, epochs=10, lr=0.05, frac=1.0, seed=0, verbose=False, data=None):
    (Xtr, ytr), (Xva, yva), te, names, _ = data or splits()
    n = int(len(ytr) * frac); Xtr, ytr = Xtr[:n], ytr[:n]; m = make_model(width, bn, residual, seed)
    o = torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9, weight_decay=wd, nesterov=True); steps = epochs * ((n + 127) // 128)
    sch = torch.optim.lr_scheduler.OneCycleLR(o, lr, total_steps=steps); g = torch.Generator().manual_seed(seed); hist = []; t = time.perf_counter()
    torch.manual_seed(seed)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, generator=g); tl = 0
        for i in range(0, n, 128):
            b = perm[i:i+128]; xb = flip_crop(Xtr[b]) if aug else Xtr[b]
            l = nn.functional.cross_entropy(m(xb), ytr[b], label_smoothing=ls); o.zero_grad(); l.backward(); o.step(); sch.step(); tl += l.item() * len(b)
        va = (predict(m, Xva).argmax(1) == yva).float().mean().item(); hist.append((ep + 1, tl / n, va))
        if verbose: print(f"  epoch {ep+1:2d}: train loss {tl/n:.4f} val acc {va:.4f} [{time.perf_counter()-t:.0f} s]", flush=True)
    return m, hist, time.perf_counter() - t
