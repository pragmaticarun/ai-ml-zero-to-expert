"""Day 52: MovieLens-100k: global mean, biases, and matrix factorisation (SGD) from scratch; RMSE on u1.test."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import load_movielens
tr, te = load_movielens(1); nu, ni = 944, 1683
u, i, r = tr.user.to_numpy(), tr.item.to_numpy(), tr.rating.to_numpy(float); ut, it, rt = te.user.to_numpy(), te.item.to_numpy(), te.rating.to_numpy(float)
print(f"train {len(tr)} ratings, test {len(te)}; users {tr.user.nunique()}, items {tr.item.nunique()}; density {len(tr) / (943 * 1682):.4f}")
rmse = lambda p: np.sqrt(((p - rt) ** 2).mean()); mu = r.mean()
print(f"global mean {mu:.4f}: test RMSE {rmse(np.full(len(rt), mu)):.4f}")
@np.errstate(invalid="ignore", divide="ignore")
def biases(lam):
    bu = np.zeros(nu); bi = np.zeros(ni)
    for _ in range(10):
        bi = np.bincount(i, r - mu - bu[u], ni) / (np.bincount(i, minlength=ni) + lam)
        bu = np.bincount(u, r - mu - bi[i], nu) / (np.bincount(u, minlength=nu) + lam)
    return bu, bi
for lam in [0, 5, 25]:
    bu, bi = biases(lam)
    with np.errstate(invalid="ignore"): p = np.nan_to_num(mu + bu[ut] + bi[it], nan=mu)
    print(f"user+item biases lam={lam:2d}: test RMSE {rmse(np.clip(p, 1, 5)):.4f}")
def mf(k, lam, lr=0.01, epochs=30, seed=0):
    g = np.random.default_rng(seed); P = 0.1 * g.normal(size=(nu, k)); Q = 0.1 * g.normal(size=(ni, k)); bu = np.zeros(nu); bi = np.zeros(ni); hist = []
    for ep in range(epochs):
        for t in g.permutation(len(r)):
            a, b = u[t], i[t]; e = r[t] - (mu + bu[a] + bi[b] + P[a] @ Q[b])
            bu[a] += lr * (e - lam * bu[a]); bi[b] += lr * (e - lam * bi[b])
            P[a], Q[b] = P[a] + lr * (e * Q[b] - lam * P[a]), Q[b] + lr * (e * P[a] - lam * Q[b])
        if ep in (0, 9, 19, 29): hist.append((ep + 1, round(float(rmse(np.clip(mu + bu[ut] + bi[it] + (P[ut] * Q[it]).sum(1), 1, 5))), 4)))
    return hist, P, Q
for k, lam in [(20, 0.0), (20, 0.05)]:
    t0 = time.perf_counter(); h, P, Q = mf(k, lam); print(f"MF k={k} lam={lam}: (epoch, test RMSE) {h}  [{time.perf_counter() - t0:.0f}s, pure-Python SGD]")
