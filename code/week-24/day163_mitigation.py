"""Day 163: bias mitigation on Adult: reweighing (Kamiran & Calders), group-specific thresholds (post-processing), and the accuracy cost."""
from _dep import *
Xtr, Xte, ytr, yte, gtr, gte = adult()
def fitp(w=None, drop=False):
    m = lr_pipe(drop_sex=drop); m.fit(Xtr, ytr, logisticregression__sample_weight=w); return m.predict_proba(Xte)[:, 1], m.predict_proba(Xtr)[:, 1]
p, ptr = fitp(); show("baseline t=0.5", yte, p, (p>=.5).astype(int), gte)
# reweighing: w(g,y) = P(g)P(y)/P(g,y) makes label independent of group in the weighted training set
w = np.ones(len(ytr))
for g in (0,1):
    for y in (0,1):
        m = (gtr==g)&(ytr==y); w[m] = (gtr==g).mean()*(ytr==y).mean()/m.mean()
print("reweighing weights:", {f"g{g}y{y}": round(float(w[(gtr==g)&(ytr==y)][0]),3) for g in (0,1) for y in (0,1)})
pr, _ = fitp(w); show("reweighing (with sex), t=0.5", yte, pr, (pr>=.5).astype(int), gte)
pr2, _ = fitp(w, drop=True); show("reweighing (no sex), t=0.5", yte, pr2, (pr2>=.5).astype(int), gte)
# post-processing: thresholds chosen on a TRAIN-side validation split, not on test
from sklearn.model_selection import train_test_split
i_fit, i_val = train_test_split(np.arange(len(ytr)), test_size=0.3, random_state=1, stratify=ytr)
mv = lr_pipe().fit(Xtr.iloc[i_fit], ytr[i_fit]); pv = mv.predict_proba(Xtr.iloc[i_val])[:,1]; pt = mv.predict_proba(Xte)[:,1]; yv, gv = ytr[i_val], gtr[i_val]
ts = np.round(np.arange(0.05, 0.951, 0.01), 2)
def pick(target):     # for each group, threshold whose val metric is closest to target
    out = {}
    for g in (0,1):
        mm = gv==g; out[g] = min(ts, key=lambda t: abs(target(yv[mm], (pv[mm]>=t)) - target.goal))
    return out
def dp(y, h): return h.mean()
def tpr(y, h): return h[y==1].mean()
for name, f, goal in (("demographic parity (sel=0.18)", dp, 0.18), ("equal opportunity (TPR=0.60)", tpr, 0.60)):
    f.goal = goal; th = pick(f); yh = np.where(gte==1, pt>=th[1], pt>=th[0]).astype(int)
    print(f"group thresholds for {name}: M {th[0]:.2f} F {th[1]:.2f}"); show("  -> test", yte, pt, yh, gte)
print("accuracy/selection trade-off of a single threshold:")
for t in (0.3, 0.4, 0.5, 0.6): show(f"  single t={t}", yte, pt, (pt>=t).astype(int), gte)
