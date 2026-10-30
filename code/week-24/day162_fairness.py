"""Day 162: fairness metrics on Adult (sex as the audited attribute): demographic parity, equalised odds, predictive parity, calibration by group."""
from _dep import *
Xtr, Xte, ytr, yte, gtr, gte = adult()
print(f"Adult: train {len(ytr)}, test {len(yte)}; P(>50K) overall {yte.mean():.3f}; test Female share {gte.mean():.3f}")
m = lr_pipe().fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]; yhat = (p >= 0.5).astype(int)
r, gap = show("logreg (with sex feature), t=0.5", yte, p, yhat, gte)
for k, nm in ((0, "Male"), (1, "Female")): print(f"  {nm:6s} n {r[k]['n']} base rate {r[k]['base']:.3f} acc {r[k]['acc']:.4f} ECE {r[k]['ece']:.4f}")
print(f"  gaps F-M: DP(sel) {gap['sel']:+.3f}, TPR {gap['tpr']:+.3f}, FPR {gap['fpr']:+.3f}, precision {gap['prec']:+.3f}, ECE {gap['ece']:+.4f}")
print("calibration by group (bin: mean p -> observed rate, count):")
for k, nm in ((0, "M"), (1, "F")):
    mm = gte == k; b = np.minimum((p[mm]*5).astype(int), 4)
    print(f"  {nm}: " + "; ".join(f"[{i/5:.1f},{(i+1)/5:.1f}) {p[mm][b==i].mean():.2f}->{yte[mm][b==i].mean():.2f} n{(b==i).sum()}" for i in range(5) if (b==i).any()))
m2 = lr_pipe(drop_sex=True).fit(Xtr, ytr); p2 = m2.predict_proba(Xte)[:, 1]
show("logreg WITHOUT sex feature, t=0.5", yte, p2, (p2 >= .5).astype(int), gte)
# proxy check: how well do the other features predict sex?
from sklearn.linear_model import LogisticRegression
Xs = m2[0].fit(Xtr).transform(Xtr); Xst = m2[0].transform(Xte); pr = LogisticRegression(max_iter=2000).fit(Xs, gtr)
from sklearn.metrics import roc_auc_score; print(f"proxy check: sex predicted from the other features, AUC {roc_auc_score(gte, pr.predict_proba(Xst)[:,1]):.3f}")
# impossibility illustration: different base rates => equal calibration and equal FPR/FNR cannot all hold (unless perfect)
print(f"base rates differ (M {r[0]['base']:.3f} vs F {r[1]['base']:.3f}), so an imperfect classifier cannot have equal precision AND equal FPR AND equal FNR across groups at once (Chouldechova 2017; Kleinberg et al. 2016 for calibration vs balance).")
