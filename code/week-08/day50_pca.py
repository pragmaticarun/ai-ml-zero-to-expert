"""Day 50: PCA from scratch via SVD vs sklearn on digits; explained variance, reconstruction, PCA+logreg; t-SNE map."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
X, y = load_digits(return_X_y=True); mu = X.mean(0); Xc = X - mu
U, S, Vt = np.linalg.svd(Xc, full_matrices=False); evr = S**2 / (S**2).sum()
sk = PCA().fit(X); print("explained var ratio, first 5 (scratch):", evr[:5].round(4).tolist(), " sklearn same:", np.allclose(evr, sk.explained_variance_ratio_))
print("components agree up to sign:", np.allclose(np.abs(Vt[:10]), np.abs(sk.components_[:10]), atol=1e-8))
cum = np.cumsum(evr); print({p: int(np.searchsorted(cum, p) + 1) for p in (0.5, 0.8, 0.9, 0.95, 0.99)}, "components needed for that variance fraction (of 64)")
for k in [2, 10, 20, 40]:
    R = Xc @ Vt[:k].T @ Vt[:k] + mu; print(f"k={k:2d} reconstruction MSE per pixel {((X - R)**2).mean():.3f}  (sum of dropped eigenvalues / 64 = {(S[k:]**2).sum() / len(X) / 64:.3f})")
for k in [5, 10, 20, 64]:
    print(f"PCA({k}) + logreg 5-fold acc {cross_val_score(make_pipeline(StandardScaler(), PCA(k), LogisticRegression(max_iter=3000)), X, y, cv=5).mean():.4f}")
t = time.perf_counter(); E = TSNE(2, random_state=0, init="pca", perplexity=30).fit_transform(X); tt = time.perf_counter() - t
P2 = sk.transform(X)[:, :2]
print(f"t-SNE {tt:.1f}s; trustworthiness (k=10): t-SNE {trustworthiness(X, E, n_neighbors=10):.4f}  PCA-2D {trustworthiness(X, P2, n_neighbors=10):.4f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, Z, t_ in [(ax[0], P2, "PCA 2D"), (ax[1], E, "t-SNE")]: s = a.scatter(*Z.T, c=y, cmap="tab10", s=4); a.set_title(t_)
fig.colorbar(s, ax=ax); savefig(__file__, "day50_pca_tsne.png")
