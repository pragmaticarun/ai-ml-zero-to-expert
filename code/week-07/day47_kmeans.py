"""Day 47: k-means (Lloyd) from scratch with k-means++ init vs sklearn; elbow and silhouette; digits clusters vs labels."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.datasets import make_blobs, load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
def sqdist(X, C): return (X**2).sum(1)[:, None] + (C**2).sum(1)[None] - 2 * X @ C.T
def init_pp(X, k, rng):
    C = [X[rng.integers(len(X))]]
    for _ in range(k - 1): d = np.min(sqdist(X, np.array(C)), 1).clip(0); C.append(X[rng.choice(len(X), p=d / d.sum())])
    return np.array(C)
def kmeans(X, k, seed=0, pp=True, iters=300):
    rng = np.random.default_rng(seed); C = init_pp(X, k, rng) if pp else X[rng.choice(len(X), k, replace=False)]
    for it in range(iters):
        lab = sqdist(X, C).argmin(1); Cn = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
        if np.allclose(Cn, C): break
        C = Cn
    return C, lab, ((X - C[lab])**2).sum(), it + 1
X, yt = make_blobs(1500, centers=4, cluster_std=1.0, random_state=47)
C, lab, J, it = kmeans(X, 4); sk = KMeans(4, n_init=10, random_state=0).fit(X)
print(f"blobs: scratch inertia {J:.1f} in {it} iters; sklearn inertia {sk.inertia_:.1f}; ARI scratch vs truth {adjusted_rand_score(yt, lab):.4f}")
bad = [kmeans(X, 4, seed=s, pp=False)[2] for s in range(20)]; good = [kmeans(X, 4, seed=s)[2] for s in range(20)]
print(f"20 seeds  random init: inertia min {min(bad):.1f} max {max(bad):.1f}   k-means++: min {min(good):.1f} max {max(good):.1f}")
print("k  inertia  silhouette"); ks = range(2, 9); I = []; S = []
for k in ks:
    m = KMeans(k, n_init=10, random_state=0).fit(X); I.append(m.inertia_); S.append(silhouette_score(X, m.labels_)); print(f"{k}  {I[-1]:8.1f}  {S[-1]:.4f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.5)); ax[0].plot(ks, I, "o-"); ax[0].set_title("elbow (inertia)"); ax[1].plot(ks, S, "o-"); ax[1].set_title("silhouette"); savefig(__file__, "day47_choose_k.png")
Xd, yd = load_digits(return_X_y=True); t = time.perf_counter(); Cd, ld, _, itd = kmeans(Xd, 10); dt = time.perf_counter() - t
print(f"digits k=10: {itd} iters {dt*1000:.0f} ms; ARI vs true digit {adjusted_rand_score(yd, ld):.4f}")
maj = np.array([np.bincount(yd[ld == j]).argmax() for j in range(10)]); print("cluster->majority digit:", maj.tolist(), f" purity {(maj[ld] == yd).mean():.4f}")
