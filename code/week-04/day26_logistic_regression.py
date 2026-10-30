"""Day 26: logistic regression from scratch on a 2-D toy problem + breast-cancer dataset."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
sig = lambda z: 1 / (1 + np.exp(-z))
def fit_logreg(X, y, lr=0.1, steps=3000, lam=0.0):
    w = np.zeros(X.shape[1]); b = 0.0
    for _ in range(steps):
        p = sig(X @ w + b); w -= lr * (X.T @ (p - y) / len(y) + lam * w / len(y)); b -= lr * (p - y).mean()
    return w, b
def log_loss(p, y, eps=1e-12): p = np.clip(p, eps, 1-eps); return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()
rng = np.random.default_rng(26)
A = rng.normal([-1, -1], 1, (150, 2)); B = rng.normal([1.5, 1.5], 1, (150, 2)); X = np.vstack([A, B]); y = np.r_[np.zeros(150), np.ones(150)]
w, b = fit_logreg(X, y); acc = ((sig(X@w+b) > .5) == y).mean(); print(f"toy: w={w.round(3)}, b={b:.3f}, train acc={acc:.3f}, loss={log_loss(sig(X@w+b), y):.4f}")
plt.scatter(*X.T, c=y, s=8, cmap="bwr"); xs = np.linspace(-4, 4, 10); plt.plot(xs, -(w[0]*xs + b)/w[1], "k"); plt.title("decision boundary w·x+b=0"); savefig(__file__, "day26_boundary.png")
d = load_breast_cancer(); Xtr, Xte, ytr, yte = train_test_split(d.data, d.target, test_size=.25, random_state=0, stratify=d.target)
mu, sd = Xtr.mean(0), Xtr.std(0); Xtr_s, Xte_s = (Xtr-mu)/sd, (Xte-mu)/sd
w, b = fit_logreg(Xtr_s, ytr, lr=0.5, steps=5000, lam=1.0)
sk = LogisticRegression(C=1.0, max_iter=5000).fit(Xtr_s, ytr)
print(f"breast cancer test acc: scratch={((sig(Xte_s@w+b)>.5)==yte).mean():.4f}  sklearn={sk.score(Xte_s, yte):.4f}")
print("why not MSE? log-loss of a confident wrong prediction p=0.001 for y=1:", round(log_loss(np.array([.001]), np.array([1])), 3), "vs squared error", (1-.001)**2)
