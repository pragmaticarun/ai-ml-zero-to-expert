"""Day 141: scoping an ML project with numbers: baselines, a learning curve (is more data worth buying?), a cost-weighted metric, and slice-based error analysis."""
import time, numpy as np, _mlops as M
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
Xtr, Xte, ytr, yte, names = M.data()
print(f"data: {len(Xtr)} train / {len(Xte)} test rows, {Xtr.shape[1]} features, positive (benign) rate {ytr.mean():.3f}")
maj = int(np.bincount(ytr).argmax()); print(f"baseline majority class: acc {(yte == maj).mean():.3f}")
i = names.index("worst radius"); thr = np.median(Xtr[:, i]); print(f"baseline 1-feature rule (worst radius < train median -> benign): acc {((Xte[:, i] < thr) == yte).mean():.3f}")
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr); print(f"logistic regression: acc {lr.score(Xte, yte):.3f}")
t = time.perf_counter(); m = M.train(Xtr, ytr); print(f"MLP 30-32-32-2: acc {M.acc(m, Xte, yte):.3f}; train {time.perf_counter()-t:.1f} s")
print("learning curve (MLP, mean of 3 seeds of the subsample):")
for frac in (0.1, 0.25, 0.5, 1.0):
    a = []
    for s in range(3):
        idx = np.random.default_rng(s).choice(len(Xtr), int(frac * len(Xtr)), replace=False); a.append(M.acc(M.train(Xtr[idx], ytr[idx], seed=s), Xte, yte))
    print(f"  {frac:4.0%} of train ({int(frac*len(Xtr))} rows): test acc {np.mean(a):.3f} (min {min(a):.3f}, max {max(a):.3f})")
p = M.proba(m, Xte)
print("cost-weighted threshold choice (a missed malignant = 10x cost of a false alarm; malignant = class 0):")
for th in (0.5, 0.8, 0.95):
    pred = p > th; fn = int(((pred == 1) & (yte == 0)).sum()); fp = int(((pred == 0) & (yte == 1)).sum())
    print(f"  threshold {th}: missed malignant {fn}, false alarms {fp}, cost {10*fn+fp}")
print("error analysis by slice (tertiles of 'mean area' on test):")
q = np.quantile(Xte[:, names.index("mean area")], [1/3, 2/3]); sl = np.digitize(Xte[:, names.index("mean area")], q)
for k, n in enumerate(["small", "medium", "large"]):
    s = sl == k; print(f"  {n:6s}: n {s.sum():3d}, acc {((p[s] > 0.5) == yte[s]).mean():.3f}, errors {int(((p[s] > 0.5) != yte[s]).sum())}")
