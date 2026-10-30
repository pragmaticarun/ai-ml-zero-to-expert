"""Day 60: Month-2 checkpoint: self-grading exercises. Copy, replace the reference bodies with your own, re-run."""
import numpy as np
from sklearn.datasets import load_breast_cancer, make_blobs
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
def ex1_ridge_closed_form(X, y, lam):            # X centred, y centred, no intercept
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
def ex2_kfold_indices(m, k):                     # list of (train_idx, val_idx), contiguous folds, no shuffle
    f = np.array_split(np.arange(m), k); return [(np.concatenate(f[:i] + f[i + 1:]), f[i]) for i in range(k)]
def ex3_gini(y): p = np.bincount(y) / len(y); return 1 - (p**2).sum()
def ex4_kmeans_step(X, C):                       # one Lloyd step: returns (labels, new centroids)
    lab = ((X[:, None] - C[None]) ** 2).sum(2).argmin(1); return lab, np.array([X[lab == j].mean(0) for j in range(len(C))])
def ex5_auc(y, s):                               # ROC-AUC via ranks (probability a positive outranks a negative)
    pos, neg = s[y == 1], s[y == 0]; return ((pos[:, None] > neg[None]).mean() + 0.5 * (pos[:, None] == neg[None]).mean())
def ex6_pca(X, k):                               # returns projected data (m x k), via SVD of centred X
    Xc = X - X.mean(0); _, _, Vt = np.linalg.svd(Xc, full_matrices=False); return Xc @ Vt[:k].T
rng = np.random.default_rng(60); X = rng.normal(size=(50, 4)); X -= X.mean(0); y = X @ [1., -2, 0, 3] + rng.normal(size=50); y -= y.mean()
checks = [("ex1 ridge == sklearn", np.allclose(ex1_ridge_closed_form(X, y, 2.0), Ridge(alpha=2.0, fit_intercept=False).fit(X, y).coef_)),
          ("ex2 kfold == sklearn KFold", all(np.array_equal(a, c) and np.array_equal(b, d) for (a, b), (c, d) in zip(ex2_kfold_indices(23, 5), KFold(5).split(np.zeros(23))))),
          ("ex3 gini([0,0,1,1]) = 0.5 and pure = 0", np.isclose(ex3_gini(np.array([0, 0, 1, 1])), .5) and ex3_gini(np.array([1, 1, 1])) == 0)]
Xb, _ = make_blobs(300, centers=3, random_state=0); C0 = Xb[:3].copy(); lab, C1 = ex4_kmeans_step(Xb, C0)
J = lambda C, l: ((Xb - C[l]) ** 2).sum(); checks.append(("ex4 k-means step does not increase inertia", J(C1, lab) <= J(C0, lab) + 1e-9))
Xc, yc = load_breast_cancer(return_X_y=True); s = Xc[:, 0] + rng.normal(size=len(yc)); checks.append(("ex5 auc == sklearn", np.isclose(ex5_auc(yc, s), roc_auc_score(yc, s))))
Z = ex6_pca(Xc, 2); checks.append(("ex6 PCA: uncorrelated components, variance descending", abs(np.corrcoef(Z.T)[0, 1]) < 1e-8 and Z[:, 0].var() >= Z[:, 1].var()))
for n, ok in checks: print(("PASS " if ok else "FAIL ") + n)
print(f"{sum(ok for _, ok in checks)}/{len(checks)} passed")
