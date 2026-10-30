"""Day 145: (1) drift detection from scratch: PSI (decile bins from the reference) and the two-sample KS test (checked against scipy),
on production-like batches with injected drift; false-alarm rate on no-drift batches; does accuracy actually drop?
(2) label-error analysis loop: flip 5% of train labels, find them with out-of-fold predicted probabilities, measure precision/recall, retrain."""
import numpy as np, _mlops as M
from scipy.stats import ks_2samp
from sklearn.model_selection import StratifiedKFold
Xtr, Xte, ytr, yte, names = M.data(); m = M.train(Xtr, ytr)
def psi(ref, cur, bins=10):
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); e[0], e[-1] = -np.inf, np.inf
    r = np.histogram(ref, e)[0] / len(ref) + 1e-4; c = np.histogram(cur, e)[0] / len(cur) + 1e-4
    return float(((c - r) * np.log(c / r)).sum())
def ks(a, b):
    a, b = np.sort(a), np.sort(b); v = np.concatenate([a, b])
    return float(np.abs(np.searchsorted(a, v, "right") / len(a) - np.searchsorted(b, v, "right") / len(b)).max())
j = names.index("mean radius"); print(f"own KS statistic vs scipy on train vs test 'mean radius': {ks(Xtr[:, j], Xte[:, j]):.4f} vs {ks_2samp(Xtr[:, j], Xte[:, j]).statistic:.4f}")
rng = np.random.default_rng(0); pool = np.vstack([Xte]); ypool = yte
def batch(kind, n=200):
    i = rng.choice(len(pool), n); X, y = pool[i].copy(), ypool[i].copy()
    if kind == "shift texture +1 sd": X[:, names.index("mean texture")] += Xtr[:, names.index("mean texture")].std()
    if kind == "scale area x1.3 (unit/sensor change)": X[:, [names.index(c) for c in names if "area" in c]] *= 1.3
    if kind == "prior shift: 80% malignant":
        mi = np.where(ypool == 0)[0]; bi = np.where(ypool == 1)[0]; i = np.concatenate([rng.choice(mi, int(.8*n)), rng.choice(bi, n - int(.8*n))]); X, y = pool[i], ypool[i]
    if kind == "noise on all features (+0.3 sd)": X = X + rng.normal(size=X.shape) * Xtr.std(0) * 0.3
    return X.astype(np.float32), y
print("per batch of 200 (resampled from test): features flagged by PSI>0.2 and by KS p<0.01 (Bonferroni over 30 features: p<0.01/30), and accuracy")
for kind in ("no drift", "shift texture +1 sd", "scale area x1.3 (unit/sensor change)", "prior shift: 80% malignant", "noise on all features (+0.3 sd)"):
    X, y = batch(kind)
    P = [psi(Xtr[:, k], X[:, k]) for k in range(30)]; K = [ks_2samp(Xtr[:, k], X[:, k]).pvalue for k in range(30)]
    top = int(np.argmax(P)); print(f"  {kind:38s}: PSI>0.2 on {sum(p > 0.2 for p in P):2d}/30 (max {max(P):.2f} '{names[top]}'); KS p<0.01 on {sum(p < .01 for p in K):2d}, Bonferroni {sum(p < .01/30 for p in K):2d}; acc {M.acc(m, X, y):.3f}")
fa_psi = fa_ks = fa_b = 0; R = 50
for _ in range(R):
    X, _y = batch("no drift"); fa_psi += any(psi(Xtr[:, k], X[:, k]) > 0.2 for k in range(30)); K = [ks_2samp(Xtr[:, k], X[:, k]).pvalue for k in range(30)]; fa_ks += min(K) < .01; fa_b += min(K) < .01 / 30
print(f"false alarms over {R} no-drift batches (any of 30 features flagged): PSI>0.2 {fa_psi}/{R}; KS p<0.01 {fa_ks}/{R}; KS Bonferroni {fa_b}/{R}")
pt = M.proba(m, Xtr); X, _ = batch("prior shift: 80% malignant"); print(f"output drift (PSI of predicted probability) under prior shift: {psi(pt, M.proba(m, X)):.2f}; under no drift: {psi(pt, M.proba(m, batch('no drift')[0])):.2f}")
print("== (2) label-error loop: flip 5% of train labels")
yn = ytr.copy(); flip = rng.choice(len(ytr), int(.05 * len(ytr)), replace=False); yn[flip] = 1 - yn[flip]
oof = np.zeros(len(ytr))
for k, (a, b) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(Xtr, yn)): oof[b] = M.proba(M.train(Xtr[a], yn[a], seed=k), Xtr[b])
conf_wrong = np.where(np.where(yn == 1, oof, 1 - oof) < 0.2)[0]   # out-of-fold model gives the GIVEN label < 0.2 probability
tp = len(set(conf_wrong) & set(flip)); print(f"  flipped {len(flip)}; flagged {len(conf_wrong)}; true flips among flagged {tp} (precision {tp/max(1,len(conf_wrong)):.2f}, recall {tp/len(flip):.2f})")
mn = M.train(Xtr, yn); keep = np.setdiff1d(np.arange(len(ytr)), conf_wrong); mc = M.train(Xtr[keep], yn[keep])
print(f"  test acc: clean labels {M.acc(m, Xte, yte):.3f}; noisy labels {M.acc(mn, Xte, yte):.3f}; noisy minus flagged rows {M.acc(mc, Xte, yte):.3f}")
