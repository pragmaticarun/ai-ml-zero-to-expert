"""Day 48: agglomerative vs DBSCAN vs k-means on moons; a GMM fitted by EM from scratch vs sklearn; BIC to choose components."""
import numpy as np
from sklearn.datasets import make_moons, make_blobs
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score
from scipy.stats import multivariate_normal as mvn
X, y = make_moons(600, noise=0.06, random_state=48)
for n, m in [("k-means k=2", KMeans(2, n_init=10, random_state=0)), ("agglomerative ward", AgglomerativeClustering(2)),
             ("agglomerative single", AgglomerativeClustering(2, linkage="single")), ("DBSCAN eps=0.2 min=5", DBSCAN(eps=0.2, min_samples=5))]:
    l = m.fit_predict(X); print(f"moons {n:24s} ARI {adjusted_rand_score(y, l):.4f}  clusters {len(set(l) - {-1})}  noise pts {(l == -1).sum()}")
for e in [0.05, 0.1, 0.3, 0.5]:
    l = DBSCAN(eps=e).fit_predict(X); print(f"  DBSCAN eps={e}: clusters {len(set(l) - {-1})}, noise {(l == -1).sum()}")
Xg, yg = make_blobs(900, centers=[[0, 0], [3, 3], [0, 4]], cluster_std=[0.5, 1.2, 0.8], random_state=48)
k = 3; rng = np.random.default_rng(0); mu = Xg[rng.choice(len(Xg), k, replace=False)]; S = np.array([np.eye(2)] * k); pi = np.ones(k) / k; prev = -np.inf
for it in range(500):
    dens = np.column_stack([pi[j] * mvn(mu[j], S[j]).pdf(Xg) for j in range(k)]); ll = np.log(dens.sum(1)).sum()
    R = dens / dens.sum(1, keepdims=True)                                     # E-step: responsibilities
    Nk = R.sum(0); pi = Nk / len(Xg); mu = (R.T @ Xg) / Nk[:, None]         # M-step
    S = np.array([((R[:, j, None] * (Xg - mu[j])).T @ (Xg - mu[j])) / Nk[j] + 1e-6 * np.eye(2) for j in range(k)])
    if it in (0, 4, 19): print(f"EM iter {it+1:3d} log-likelihood {ll:.2f}")
    if ll - prev < 1e-6: break
    prev = ll
gm = GaussianMixture(3, random_state=0).fit(Xg)
print(f"scratch EM converged at iter {it+1}, avg log-lik {ll/len(Xg):.4f}; sklearn avg log-lik {gm.score(Xg):.4f}")
o = np.argsort(mu[:, 0] + 10 * mu[:, 1]); print("scratch means:", mu[o].round(2).tolist(), " sklearn means:", gm.means_[np.argsort(gm.means_[:, 0] + 10 * gm.means_[:, 1])].round(2).tolist())
print("std per comp (scratch, sorted):", np.sqrt(np.array([np.diag(s) for s in S[o]])).round(2).tolist())
print("components  BIC"); [print(f"{c:10d}  {GaussianMixture(c, random_state=0, n_init=3).fit(Xg).bic(Xg):.1f}") for c in range(1, 7)]
print(f"soft assignment example: a point at (1.5,1.5) -> {gm.predict_proba([[1.5, 1.5]]).round(3).tolist()}")
