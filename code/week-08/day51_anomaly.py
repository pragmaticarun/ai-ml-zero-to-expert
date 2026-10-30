"""Day 51: anomaly detection: per-feature Gaussian density with epsilon tuned on validation F1, vs IsolationForest."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score
from scipy.stats import norm, multivariate_normal
X, y = load_breast_cancer(return_X_y=True); rng = np.random.default_rng(51)
normal = X[y == 1]; anom = X[y == 0]; rng.shuffle(normal); rng.shuffle(anom)
anom = anom[:40]                                                       # make anomalies rare
Xtr = normal[:237]; Xva = np.r_[normal[237:297], anom[:20]]; yva = np.r_[np.zeros(60), np.ones(20)]
Xte = np.r_[normal[297:], anom[20:]]; yte = np.r_[np.zeros(len(normal) - 297), np.ones(20)]
print(f"train {len(Xtr)} normal only; val {len(Xva)} ({int(yva.sum())} anomalies); test {len(Xte)} ({int(yte.sum())} anomalies)")
tf = lambda Z: np.log1p(Z - X.min(0) + 1e-3)
sc = StandardScaler().fit(tf(Xtr)); Z = lambda A: sc.transform(tf(A))
logp_ind = lambda A: norm.logpdf(Z(A)).sum(1)                         # independent Gaussians: p(x) = prod_j p(x_j)
cov = np.cov(Z(Xtr).T) + 1e-3 * np.eye(X.shape[1]); mv = multivariate_normal(np.zeros(X.shape[1]), cov)
def tune(score_va):
    best = (0, None)
    for e in np.quantile(score_va, np.linspace(0.01, 0.6, 200)):
        f = f1_score(yva, score_va < e)
        if f > best[0]: best = (f, e)
    return best
for name, s in [("independent Gaussian", logp_ind), ("multivariate Gaussian", mv.logpdf)]:
    f, e = tune(s(Xva)); p = s(Xte) < e
    print(f"{name:22s} eps(log p)={e:.1f} val F1 {f:.3f} | test prec {precision_score(yte, p):.3f} rec {recall_score(yte, p):.3f} F1 {f1_score(yte, p):.3f} AP {average_precision_score(yte, -s(Xte)):.3f}")
iso = IsolationForest(n_estimators=300, random_state=0).fit(Z(Xtr)); sv = iso.score_samples(Z(Xva)); f, e = tune(sv); p = iso.score_samples(Z(Xte)) < e
print(f"{'IsolationForest':22s} thr={e:.3f}     val F1 {f:.3f} | test prec {precision_score(yte, p):.3f} rec {recall_score(yte, p):.3f} F1 {f1_score(yte, p):.3f} AP {average_precision_score(yte, -iso.score_samples(Z(Xte))):.3f}")
