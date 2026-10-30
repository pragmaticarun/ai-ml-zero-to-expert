"""Day 89: cumulative review drills for Months 1-3. Each drill recomputes a fact from earlier lessons and checks it."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0); ok = 0; drills = 0
def check(name, cond, detail):
    global ok, drills; drills += 1; ok += bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name}: {detail}")
X = rng.normal(size=(200, 3)); w = np.array([1.0, -2.0, 0.5]); y = X @ w + 0.01 * rng.normal(size=200)
wh = np.linalg.lstsq(X, y, rcond=None)[0]; check("M1 least squares recovers w", np.allclose(wh, w, atol=0.01), np.round(wh, 3))
p = np.array([0.7, 0.2, 0.1]); check("M1 cross-entropy of true class 0", abs(-np.log(p[0]) - 0.3567) < 1e-3, f"{-np.log(p[0]):.4f}")
from sklearn.model_selection import cross_val_score; from sklearn.linear_model import LogisticRegression; from sklearn.datasets import load_digits
Xd, yd = load_digits(return_X_y=True); s = cross_val_score(LogisticRegression(max_iter=2000), Xd / 16, yd, cv=5)
check("M2 5-fold CV on digits", s.mean() > 0.9, f"mean {s.mean():.4f} std {s.std():.4f}")
tp, fp, fn = 70, 30, 10; check("M2 precision/recall", (tp / (tp + fp), tp / (tp + fn)) == (0.7, 0.875), "P 0.700 R 0.875")
m = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)); n = sum(q.numel() for q in m.parameters())
check("M3 MLP param count 784-128-10", n == 784 * 128 + 128 + 128 * 10 + 10, f"{n:,}")
c = nn.Conv2d(3, 16, 3); check("M3 conv params 3->16, 3x3", sum(q.numel() for q in c.parameters()) == 448, "448")
check("M3 conv output size (32+2*1-3)/1+1", nn.Conv2d(3, 8, 3, padding=1)(torch.zeros(1, 3, 32, 32)).shape[-1] == 32, "32")
x = torch.tensor(3.0, requires_grad=True); (x ** 2).backward(); (x ** 2).backward(); check("M3 grads accumulate without zero_grad", x.grad.item() == 12.0, f"{x.grad.item()}")
torch.manual_seed(0); l = nn.functional.cross_entropy(nn.Linear(20, 10)(torch.randn(256, 20)) * 0.01, torch.randint(0, 10, (256,)))
check("M3 initial loss near ln(10)", abs(l.item() - np.log(10)) < 0.05, f"{l.item():.4f} vs {np.log(10):.4f}")
d = nn.Dropout(0.5); z = torch.ones(10000); d.train(); a = d(z); d.eval()
check("M3 dropout scales kept units by 1/(1-p) and is identity in eval", abs(a.mean().item() - 1) < 0.05 and torch.equal(d(z), z), f"train mean {a.mean():.3f}")
print(f"{ok}/{drills} drills passed"); sys.exit(0 if ok == drills else 1)
