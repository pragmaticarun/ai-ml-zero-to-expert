"""Day 56 project: k-means kernels: NumPy loop vs NumPy matmul trick vs Rust (serial, rayon) vs sklearn. Same labels required."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
try: import rust_kmeans
except ImportError: rust_kmeans = None; print("rust_kmeans not built: cd code/week-08/rust_kmeans && maturin develop --release")
def best_of(f, r=3):
    ts = []; out = None
    for _ in range(r): t = time.perf_counter(); out = f(); ts.append(time.perf_counter() - t)
    return out, min(ts)
X, _ = make_blobs(200_000, n_features=16, centers=32, random_state=56); X = np.ascontiguousarray(X); rng = np.random.default_rng(0)
C = X[rng.choice(len(X), 32, replace=False)].copy(); print("X", X.shape, "k", len(C), "threads available", os.cpu_count())
def np_loop(X, C): return np.argmin(np.stack([((X - c) ** 2).sum(1) for c in C], 1), 1)
def np_mm(X, C): return np.argmin((C**2).sum(1)[None] - 2 * X @ C.T, 1)      # ||x||^2 is constant per row: drop it
res = {}
res["numpy per-centroid loop"] = best_of(lambda: np_loop(X, C))
res["numpy matmul trick"] = best_of(lambda: np_mm(X, C))
if rust_kmeans:
    res["rust serial"] = best_of(lambda: rust_kmeans.assign(X, C)); res["rust rayon"] = best_of(lambda: rust_kmeans.assign_par(X, C))
ref = res["numpy per-centroid loop"][0]
print("ASSIGN STEP (best of 3)          time      labels identical to loop")
for n, (lab, t) in res.items(): print(f"{n:30s} {t*1000:7.1f} ms  {(lab == ref).mean():.6f}")
def np_lloyd(X, C, iters):
    C = C.copy()
    for it in range(iters):
        lab = np_mm(X, C); Cn = np.array([X[lab == j].mean(0) for j in range(len(C))])
        if np.abs(Cn - C).sum() < 1e-12: break
        C = Cn
    return C, lab, it + 1
(Cn, ln, itn), tn = best_of(lambda: np_lloyd(X, C, 100), 1); print(f"\nFULL LLOYD from same init: numpy {tn:.2f}s ({itn} iters)", end="")
if rust_kmeans:
    (Cr, lr, inr, itr), tr = best_of(lambda: rust_kmeans.lloyd(X, C, 100), 1); print(f" | rust {tr:.2f}s ({itr} iters) | same labels {(lr == ln).mean():.6f} | max centroid diff {np.abs(Cr - Cn).max():.2e}")
sk, tk = best_of(lambda: KMeans(32, init=C, n_init=1, max_iter=100, algorithm="lloyd", tol=0).fit(X), 1)
print(f"sklearn KMeans(lloyd, same init): {tk:.2f}s ({sk.n_iter_} iters), labels agree with numpy {(sk.labels_ == ln).mean():.6f}")
