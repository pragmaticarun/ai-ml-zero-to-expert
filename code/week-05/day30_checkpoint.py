"""Day 30 checkpoint: self-grading exercises. Fill the TODOs in a copy; run to check. Reference solutions run by default."""
import numpy as np
def ex1_vectorised_mse(X, w, b, y): return ((X @ w + b - y) ** 2).mean() / 2
def ex2_grad(X, w, b, y): e = X @ w + b - y; return X.T @ e / len(y), e.mean()
def ex3_sigmoid(z): return 1 / (1 + np.exp(-z))
def ex4_precision_recall(y, yh):
    tp = ((y == 1) & (yh == 1)).sum(); return tp / max((yh == 1).sum(), 1), tp / max((y == 1).sum(), 1)
def ex5_bayes(prior, like, false_pos): return like*prior / (like*prior + false_pos*(1-prior))
rng = np.random.default_rng(30); X = rng.normal(size=(20, 3)); w = rng.normal(size=3); y = rng.normal(size=20); b = 0.3
checks = []
J = lambda w_: ex1_vectorised_mse(X, w_, b, y); gw, gb = ex2_grad(X, w, b, y)
num = np.array([(J(w + e) - J(w - e)) / 2e-6 for e in np.eye(3) * 1e-6]); checks.append(("ex2 gradient matches numeric", np.allclose(gw, num, atol=1e-6)))
checks.append(("ex1 mse", np.isclose(ex1_vectorised_mse(np.eye(2), np.ones(2), 0, np.zeros(2)), 0.5)))
checks.append(("ex3 sigmoid(0)=0.5 & symmetry", np.isclose(ex3_sigmoid(0), .5) and np.isclose(ex3_sigmoid(2) + ex3_sigmoid(-2), 1)))
checks.append(("ex4 precision/recall", ex4_precision_recall(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0])) == (0.5, 0.5)))
checks.append(("ex5 bayes", np.isclose(ex5_bayes(.01, .95, .05), 0.0095/0.059)))
for n, ok in checks: print(("PASS " if ok else "FAIL ") + n)
print(f"{sum(ok for _, ok in checks)}/{len(checks)} passed")
