"""Day 28 project: end-to-end housing regression with error analysis (which houses do we get most wrong?)."""
import numpy as np
from housing import make_housing
from sklearn.model_selection import train_test_split
X, y, names = make_housing(2000, seed=28)
# twist: real data has non-linear effects; add a luxury premium for big houses the linear model can't see
y = y + np.where(X[:, 0] > 1800, 0.04*(X[:, 0]-1800), 0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=0)
mu, sd = Xtr.mean(0), Xtr.std(0)
def fit(A, t):
    Ad = np.column_stack([A, np.ones(len(A))]); return np.linalg.lstsq(Ad, t, rcond=None)[0]
def pred(th, A): return np.column_stack([A, np.ones(len(A))]) @ th
th = fit((Xtr-mu)/sd, ytr); r = yte - pred(th, (Xte-mu)/sd)
print(f"v1 linear: test MAE={np.abs(r).mean():.3f} RMSE={np.sqrt((r**2).mean()):.3f}")
bins = [350, 800, 1200, 1500, 1800, 2200, 4000]; cat = np.digitize(Xte[:, 0], bins)
print("error analysis by area bucket (mean residual, MAE, count):")
for k in np.unique(cat): s = cat == k; print(f"  {bins[k-1]:>5}-{bins[k] if k < len(bins) else '':>5}: {r[s].mean():+7.2f} {np.abs(r[s]).mean():6.2f} {s.sum():4d}")
def feats(A): return np.column_stack([A, np.maximum(A[:, 0] - 1800, 0)])   # hinge feature suggested by error analysis
Ftr, Fte = feats(Xtr), feats(Xte); mu2, sd2 = Ftr.mean(0), Ftr.std(0)
th2 = fit((Ftr-mu2)/sd2, ytr); r2 = yte - pred(th2, (Fte-mu2)/sd2)
print(f"v2 +hinge feature: test MAE={np.abs(r2).mean():.3f} RMSE={np.sqrt((r2**2).mean()):.3f}")
