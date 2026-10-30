"""Day 43: impurity vs permutation importance (with a planted random column) and exact Shapley values by brute force."""
import sys, os, itertools, math; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_california
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
X, y = load_california(); rng = np.random.default_rng(43)
X = X.copy(); X["RANDOM_id"] = rng.permutation(len(X)).astype(float)   # pure noise, many unique values
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
rf = RandomForestRegressor(100, min_samples_leaf=1, n_jobs=-1, random_state=0).fit(Xtr, ytr)
pi = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=0, n_jobs=-1)
T = pd.DataFrame({"impurity (MDI)": rf.feature_importances_, "permutation on test (R2 drop)": pi.importances_mean, "perm std": pi.importances_std}, index=X.columns)
print(T.sort_values("permutation on test (R2 drop)", ascending=False).round(4).to_string())
print(f"RF test R2 {rf.score(Xte, yte):.4f}; RANDOM_id MDI rank {int((T['impurity (MDI)'] > T.loc['RANDOM_id', 'impurity (MDI)']).sum()) + 1} of {len(T)}")
# exact Shapley for a 4-feature model, baseline = training mean (interventional, independent features)
cols = ["MedInc", "HouseAge", "AveRooms", "Latitude"]; lm = LinearRegression().fit(Xtr[cols], ytr)
f = lambda Z: lm.predict(pd.DataFrame(Z, columns=cols)); base = Xtr[cols].mean().to_numpy(); x = Xte[cols].iloc[0].to_numpy()
def v(S): z = base.copy(); z[list(S)] = x[list(S)]; return f(z[None])[0]
n = len(cols); phi = np.zeros(n)
for j in range(n):
    for r in range(n):
        for S in itertools.combinations([k for k in range(n) if k != j], r):
            phi[j] += math.factorial(len(S)) * math.factorial(n - len(S) - 1) / math.factorial(n) * (v(S + (j,)) - v(S))
print("row 0 features:", dict(zip(cols, x.round(3).tolist())))
print("Shapley values:", dict(zip(cols, phi.round(4).tolist())))
print(f"sum of Shapley {phi.sum():.4f} = f(x) - f(baseline) = {v(tuple(range(n))) - v(()):.4f}")
print("linear check w_j*(x_j - mean_j):", dict(zip(cols, (lm.coef_ * (x - base)).round(4).tolist())))
