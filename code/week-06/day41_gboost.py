"""Day 41: gradient boosting from scratch (squared loss, sklearn trees as weak learners) vs GradientBoostingRegressor vs HistGB."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import load_california, savefig
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
X, y = load_california(); X, y = X.to_numpy(), y.to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
class ScratchGB:
    def __init__(s, n=200, lr=0.1, depth=3): s.n, s.lr, s.depth = n, lr, depth
    def fit(s, X, y, Xv=None, yv=None):
        s.f0 = y.mean(); F = np.full(len(y), s.f0); s.trees = []; s.curve = []; Fv = np.full(len(yv), s.f0) if Xv is not None else None
        for m in range(s.n):
            r = y - F                                   # negative gradient of 0.5*(y-F)^2
            t = DecisionTreeRegressor(max_depth=s.depth, random_state=0).fit(X, r); s.trees.append(t)
            F += s.lr * t.predict(X)
            if Xv is not None: Fv += s.lr * t.predict(Xv); s.curve.append((((y - F) ** 2).mean(), ((yv - Fv) ** 2).mean()))
        return s
    def predict(s, X): return s.f0 + s.lr * sum(t.predict(X) for t in s.trees)
mse = lambda a, b: ((a - b) ** 2).mean()
t = time.perf_counter(); g = ScratchGB(200, 0.1, 3).fit(Xtr, ytr, Xte, yte); ts = time.perf_counter() - t
t = time.perf_counter(); sk = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=0).fit(Xtr, ytr); tk = time.perf_counter() - t
t = time.perf_counter(); h = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=0, early_stopping=False).fit(Xtr, ytr); th = time.perf_counter() - t
print(f"scratch GB       test MSE {mse(g.predict(Xte), yte):.4f}  fit {ts:.2f}s")
print(f"sklearn GB       test MSE {mse(sk.predict(Xte), yte):.4f}  fit {tk:.2f}s   pred diff vs scratch: mean {np.abs(sk.predict(Xte) - g.predict(Xte)).mean():.2e}, max {np.abs(sk.predict(Xte) - g.predict(Xte)).max():.2e}, identical on {np.mean(np.isclose(sk.predict(Xte), g.predict(Xte))):.3f} of rows")
print(f"HistGB (binned)  test MSE {mse(h.predict(Xte), yte):.4f}  fit {th:.2f}s")
c = np.array(g.curve); print("stage  train_MSE  test_MSE:", [(k, round(float(c[k-1, 0]), 4), round(float(c[k-1, 1]), 4)) for k in (1, 10, 50, 100, 200)])
for lr, n in [(1.0, 200), (0.1, 200), (0.01, 200), (0.01, 2000)]:
    e = GradientBoostingRegressor(n_estimators=n, learning_rate=lr, max_depth=3, random_state=0).fit(Xtr, ytr) if n <= 200 else HistGradientBoostingRegressor(max_iter=n, learning_rate=lr, max_depth=3, random_state=0, early_stopping=False).fit(Xtr, ytr)
    print(f"lr {lr:<5} trees {n:<5} test MSE {mse(e.predict(Xte), yte):.4f}" + ("  (HistGB, depth 3)" if n > 200 else ""))
plt.plot(c[:, 0], label="train"); plt.plot(c[:, 1], label="test"); plt.xlabel("boosting stage"); plt.ylabel("MSE"); plt.legend(); savefig(__file__, "day41_boost_curve.png")
