"""Day 23: batch gradient descent for univariate linear regression, with a learning curve."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
from housing import make_housing
X, y, _ = make_housing(200, seed=22); x = X[:, 0] / 1000
@np.errstate(over="ignore", invalid="ignore")  # we WANT to see divergence as nan, not warnings
def gd(x, y, lr, steps):
    w = b = 0.0; hist = []
    for _ in range(steps):
        err = w*x + b - y; w -= lr * (err*x).mean(); b -= lr * err.mean(); hist.append((err**2).mean()/2)
    return w, b, hist
for lr in [0.01, 0.1, 0.5, 0.9]:
    w, b, h = gd(x, y, lr, 3000); print(f"lr={lr}: w={w:.3f} b={b:.3f} final J={h[-1]:.4g} {'(diverged)' if not np.isfinite(h[-1]) or h[-1] > h[0] else ''}")
w, b, h = gd(x, y, 0.5, 3000); plt.plot(h); plt.yscale("log"); plt.xlabel("iteration"); plt.ylabel("J"); savefig(__file__, "day23_learning_curve.png")
print("closed form:", np.polyfit(x, y, 1).round(3))
