"""Day 33: k-fold CV by hand vs cross_val_score, grid vs random search, nested CV (California housing)."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import load_california
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from scipy.stats import loguniform
X, y = load_california(); X, y = X.to_numpy(), y.to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
from sklearn.preprocessing import FunctionTransformer
HEAVY = [2, 3, 4, 5]   # AveRooms, AveBedrms, Population, AveOccup: long right tails
def tame(X): X = X.copy(); X[:, HEAVY] = np.log1p(X[:, HEAVY]); return X
raw = lambda a=1.0: make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=a))
pipe = lambda a=1.0: make_pipeline(FunctionTransformer(tame), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=a))
kf = KFold(5, shuffle=True, random_state=0)
r = cross_val_score(raw(), Xtr, ytr, cv=kf); print("RAW poly2 fold R2 (outliers squared!):", np.round(r, 4))
print("max AveOccup in train:", Xtr[:, 5].max().round(1), " median:", np.median(Xtr[:, 5]).round(2))
scores = []
for tr, va in kf.split(Xtr):
    m = pipe().fit(Xtr[tr], ytr[tr]); scores.append(m.score(Xtr[va], ytr[va]))
print("manual 5-fold R2:", np.round(scores, 4), f"mean {np.mean(scores):.4f} std {np.std(scores):.4f}")
sk = cross_val_score(pipe(), Xtr, ytr, cv=kf); print("cross_val_score  :", np.round(sk, 4), "identical:", np.allclose(sk, scores))
grid = {"ridge__alpha": np.logspace(-3, 3, 13)}
t = time.perf_counter(); gs = GridSearchCV(pipe(), grid, cv=kf).fit(Xtr, ytr); tg = time.perf_counter() - t
t = time.perf_counter(); rs = RandomizedSearchCV(pipe(), {"ridge__alpha": loguniform(1e-3, 1e3)}, n_iter=6, cv=kf, random_state=0).fit(Xtr, ytr); tr_ = time.perf_counter() - t
print(f"grid  (13 fits x5): best alpha {gs.best_params_['ridge__alpha']:.4g}  CV R2 {gs.best_score_:.4f}  {tg:.2f}s")
print(f"random( 6 fits x5): best alpha {rs.best_params_['ridge__alpha']:.4g}  CV R2 {rs.best_score_:.4f}  {tr_:.2f}s")
print("grid CV R2 by alpha:", {f"{a:g}": round(float(s), 4) for a, s in zip(grid["ridge__alpha"], gs.cv_results_["mean_test_score"])})
outer = KFold(5, shuffle=True, random_state=1)
nested = cross_val_score(GridSearchCV(pipe(), grid, cv=KFold(3, shuffle=True, random_state=0)), Xtr, ytr, cv=outer)
print(f"nested CV R2 {nested.mean():.4f} +- {nested.std():.4f}  (vs optimistic best_score_ {gs.best_score_:.4f})")
print(f"test R2 of refit grid winner (touched once): {gs.score(Xte, yte):.4f}")
