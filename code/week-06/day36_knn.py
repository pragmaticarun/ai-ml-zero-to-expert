"""Day 36: k-NN from scratch (vectorised distances) vs sklearn; k sweep; scaling; curse of dimensionality."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.datasets import load_digits, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
def knn_predict(Xtr, ytr, Xq, k):
    D = (Xq**2).sum(1)[:, None] + (Xtr**2).sum(1)[None] - 2 * Xq @ Xtr.T      # Day 3 identity
    idx = np.argpartition(D, k, axis=1)[:, :k]
    return np.array([np.bincount(r, minlength=10).argmax() for r in ytr[idx]])
X, y = load_digits(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
t = time.perf_counter(); p = knn_predict(Xtr, ytr, Xte, 3); ts = time.perf_counter() - t
sk = KNeighborsClassifier(3).fit(Xtr, ytr); t = time.perf_counter(); ps = sk.predict(Xte); tk = time.perf_counter() - t
print(f"digits k=3 scratch acc {(p == yte).mean():.4f} ({ts*1000:.0f} ms)  sklearn acc {(ps == yte).mean():.4f} ({tk*1000:.0f} ms)  agree {(p == ps).mean():.4f}")
print("k sweep (5-fold CV acc on digits train):", {k: round(float(cross_val_score(KNeighborsClassifier(k), Xtr, ytr, cv=5).mean()), 4) for k in [1, 3, 5, 11, 25, 51]})
Xw, yw = load_wine(return_X_y=True)
print(f"wine k=5 CV acc unscaled {cross_val_score(KNeighborsClassifier(5), Xw, yw, cv=5).mean():.4f}  scaled {cross_val_score(make_pipeline(StandardScaler(), KNeighborsClassifier(5)), Xw, yw, cv=5).mean():.4f}")
print("wine feature ranges (max-min), first 5 and proline:", np.ptp(Xw, 0)[:5].round(2), np.ptp(Xw, 0)[12].round(0))
rng = np.random.default_rng(36)
print("dim  nearest/farthest distance ratio (1000 uniform points, 1 query)")
for d in [2, 10, 100, 1000]:
    P = rng.uniform(size=(1000, d)); q = rng.uniform(size=d); dist = np.linalg.norm(P - q, axis=1)
    print(f"{d:4d}  {dist.min() / dist.max():.3f}")
