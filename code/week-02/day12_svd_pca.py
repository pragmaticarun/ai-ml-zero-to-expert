"""Day 12: SVD intuition and PCA from scratch vs scikit-learn."""
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
X = load_digits().data; print("digits:", X.shape)
Xc = X - X.mean(0); U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
var = S**2 / (len(X) - 1); ratio = var / var.sum()
for k in [2, 10, 20, 40]: print(f"top {k:2d} components explain {ratio[:k].sum()*100:.1f}% variance")
Z = Xc @ Vt[:2].T; p = PCA(2).fit(X); Z2 = p.transform(X)
print("scratch vs sklearn (abs, sign may flip) max diff:", np.abs(np.abs(Z) - np.abs(Z2)).max().round(8))
for k in [5, 20]:
    Xk = (U[:, :k] * S[:k]) @ Vt[:k]; print(f"rank-{k} reconstruction RMSE: {np.sqrt(((Xk - Xc)**2).mean()):.3f}")
