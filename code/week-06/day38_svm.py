"""Day 38: linear SVM by subgradient descent on hinge loss vs LinearSVC; RBF kernel C/gamma grid on moons."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.datasets import load_breast_cancer, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC, SVC
X, y = load_breast_cancer(return_X_y=True); y = 2 * y - 1
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
C = 1.0; m, n = Xtr.shape; w = np.zeros(n); b = 0.0
for t in range(1, 3001):
    marg = ytr * (Xtr @ w + b); act = marg < 1
    gw = w - C * (ytr[act, None] * Xtr[act]).sum(0); gb = -C * ytr[act].sum()
    lr = 1.0 / (t + 100); w -= lr * gw; b -= lr * gb
obj = lambda w, b: 0.5 * w @ w + C * np.maximum(0, 1 - ytr * (Xtr @ w + b)).sum()
sk = LinearSVC(C=C, loss="hinge", dual=True, max_iter=200000, tol=1e-6).fit(Xtr, ytr)
print(f"scratch hinge-SVM test acc {(np.sign(Xte @ w + b) == yte).mean():.4f} objective {obj(w, b):.2f}")
print(f"LinearSVC        test acc {sk.score(Xte, yte):.4f} objective {obj(sk.coef_[0], sk.intercept_[0]):.2f}")
print("points on/inside margin (y*f<=1):", int((ytr * (Xtr @ w + b) <= 1 + 1e-9).sum()), "of", m)
Xm, ym = make_moons(400, noise=0.25, random_state=0); Xa, Xb, ya, yb = train_test_split(Xm, ym, test_size=0.5, random_state=0)
print("moons RBF SVC: C, gamma -> train acc / test acc / #support vectors")
fig, axs = plt.subplots(2, 3, figsize=(11, 7)); gx, gy = np.meshgrid(np.linspace(-2, 3, 200), np.linspace(-1.5, 2, 200))
for i, Cv in enumerate([0.1, 10]):
    for j, gm in enumerate([0.1, 1, 50]):
        s = SVC(C=Cv, gamma=gm).fit(Xa, ya); print(f"C={Cv:<4} gamma={gm:<4} {s.score(Xa, ya):.3f} / {s.score(Xb, yb):.3f} / {s.n_support_.sum()}")
        ax = axs[i, j]; ax.contourf(gx, gy, s.decision_function(np.c_[gx.ravel(), gy.ravel()]).reshape(gx.shape) > 0, alpha=.3); ax.scatter(*Xa.T, c=ya, s=8); ax.set_title(f"C={Cv}, γ={gm}")
savefig(__file__, "day38_rbf_grid.png")
print(f"linear SVC on moons test acc {SVC(kernel='linear').fit(Xa, ya).score(Xb, yb):.3f}")
