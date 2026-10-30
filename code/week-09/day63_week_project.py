"""Day 63 project: a 2-layer NumPy MLP on digits with hand-derived backprop, gradient check, mini-batch training."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
def init(n, h, k, seed=0):
    g = np.random.default_rng(seed); return {"W1": g.normal(size=(n, h)) * np.sqrt(2 / n), "b1": np.zeros(h), "W2": g.normal(size=(h, k)) * np.sqrt(1 / h), "b2": np.zeros(k)}
def forward(p, X):
    Z1 = X @ p["W1"] + p["b1"]; A1 = np.maximum(0, Z1); Z2 = A1 @ p["W2"] + p["b2"]
    P = np.exp(Z2 - Z2.max(1, keepdims=True)); P /= P.sum(1, keepdims=True); return Z1, A1, P
def loss_grad(p, X, y, lam=0.0):
    m = len(y); Z1, A1, P = forward(p, X); L = -np.log(P[np.arange(m), y] + 1e-12).mean() + lam / 2 * ((p["W1"]**2).sum() + (p["W2"]**2).sum())
    dZ2 = P.copy(); dZ2[np.arange(m), y] -= 1; dZ2 /= m
    g = {"W2": A1.T @ dZ2 + lam * p["W2"], "b2": dZ2.sum(0)}; dA1 = dZ2 @ p["W2"].T; dZ1 = dA1 * (Z1 > 0)
    g["W1"] = X.T @ dZ1 + lam * p["W1"]; g["b1"] = dZ1.sum(0); return L, g
p = init(64, 16, 10, seed=1); Xs, ys = Xtr[:20], ytr[:20]; _, g = loss_grad(p, Xs, ys, 1e-3); worst = 0
for k in p:
    num = np.zeros_like(p[k]); it = np.nditer(p[k], flags=["multi_index"])
    for _ in it:
        i = it.multi_index; o = p[k][i]; p[k][i] = o + 1e-5; a, _ = loss_grad(p, Xs, ys, 1e-3); p[k][i] = o - 1e-5; b, _ = loss_grad(p, Xs, ys, 1e-3); p[k][i] = o; num[i] = (a - b) / 2e-5
    rel = np.linalg.norm(num - g[k]) / (np.linalg.norm(num) + np.linalg.norm(g[k]) + 1e-12); worst = max(worst, rel); print(f"grad check {k}: relative error {rel:.2e}")
print("gradient check", "PASSED" if worst < 1e-6 else "FAILED")
p = init(64, 64, 10); rng = np.random.default_rng(0); lr, lam, hist = 0.2, 1e-4, []
for ep in range(60):
    o = rng.permutation(len(ytr))
    for s in range(0, len(o), 32):
        _, g = loss_grad(p, Xtr[o[s:s+32]], ytr[o[s:s+32]], lam)
        for k in p: p[k] -= lr * g[k]
    L, _ = loss_grad(p, Xtr, ytr, lam); acc = lambda X_, y_: (forward(p, X_)[2].argmax(1) == y_).mean(); hist.append((L, acc(Xtr, ytr), acc(Xte, yte)))
    if ep + 1 in (1, 5, 20, 60): print(f"epoch {ep+1:2d} train loss {L:.4f} train acc {hist[-1][1]:.4f} test acc {hist[-1][2]:.4f}")
print(f"baselines: logistic regression test acc {LogisticRegression(max_iter=3000).fit(Xtr, ytr).score(Xte, yte):.4f}; sklearn MLPClassifier(64) {MLPClassifier((64,), max_iter=500, random_state=0).fit(Xtr, ytr).score(Xte, yte):.4f}")
h = np.array(hist); plt.plot(h[:, 1], label="train acc"); plt.plot(h[:, 2], label="test acc"); plt.xlabel("epoch"); plt.legend(); savefig(__file__, "day63_mlp_curve.png")
