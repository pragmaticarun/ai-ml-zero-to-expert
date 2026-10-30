"""Day 25: why feature scaling matters: iterations to converge with raw vs scaled features."""
import numpy as np
from housing import make_housing
X, y, _ = make_housing(1000, seed=25)
def run(X, y, lr, tol=1e-6, max_it=200000):
    w = np.zeros(X.shape[1]); b = 0.0; prev = np.inf
    for it in range(max_it):
        err = X @ w + b - y; J = (err**2).mean()/2
        if not np.isfinite(J): return it, np.inf
        if abs(prev - J) < tol * max(J, 1): return it, J
        prev = J; w -= lr * X.T @ err / len(y); b -= lr * err.mean()
    return max_it, J
for lr in [1e-6, 1e-7, 2.5e-7]: print(f"RAW lr={lr:g}: iters, J =", run(X, y, lr))
mu, sd = X.mean(0), X.std(0); print(f"Z-SCORE lr=0.1: iters, J =", run((X-mu)/sd, y, 0.1))
mn, mx = X.min(0), X.max(0); print(f"MIN-MAX lr=0.5: iters, J =", run((X-mn)/(mx-mn), y, 0.5))
print("feature std devs (raw):", sd.round(2), "-> condition number raw X^TX: %.2e vs scaled: %.2e"
      % (np.linalg.cond(np.cov(X.T)), np.linalg.cond(np.cov(((X-mu)/sd).T))))
