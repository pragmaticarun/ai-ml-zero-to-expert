"""Day 35 project: California housing with CV-based model selection; test set touched once."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_california
from sklearn.model_selection import KFold, cross_validate, train_test_split, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score
X, y = load_california()
def add_feats(D):
    D = D.copy(); D["RoomsPerHouse"] = D.AveRooms; D["BedrmRatio"] = D.AveBedrms / D.AveRooms
    D["LogPop"] = np.log1p(D.Population); D["LogOccup"] = np.log1p(D.AveOccup)
    D["AveRooms"] = np.log1p(D.AveRooms); D["AveBedrms"] = np.log1p(D.AveBedrms)
    return D.drop(columns=["Population", "AveOccup", "RoomsPerHouse"])
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
print("target capped at", y.max(), "; share of rows at cap:", round((y >= 5.0).mean(), 4))
kf = KFold(5, shuffle=True, random_state=0)
models = {
 "mean baseline": DummyRegressor(),
 "linear (raw)": make_pipeline(StandardScaler(), LinearRegression()),
 "linear + engineered": make_pipeline(FunctionTransformer(add_feats), StandardScaler(), LinearRegression()),
 "poly2 ridge (a=10)": make_pipeline(FunctionTransformer(add_feats), PolynomialFeatures(2), StandardScaler(), Ridge(alpha=10)),
 "HistGradientBoosting": HistGradientBoostingRegressor(random_state=0)}
rows = []
for name, m in models.items():
    t = time.perf_counter(); r = cross_validate(m, Xtr, ytr, cv=kf, scoring=("neg_mean_absolute_error", "r2")); dt = time.perf_counter() - t
    rows.append((name, -r["test_neg_mean_absolute_error"].mean(), r["test_neg_mean_absolute_error"].std(), r["test_r2"].mean(), dt))
print(pd.DataFrame(rows, columns=["model", "CV MAE", "MAE std", "CV R2", "seconds"]).round(4).to_string(index=False))
gs = GridSearchCV(HistGradientBoostingRegressor(random_state=0), {"learning_rate": [0.05, 0.1, 0.2], "max_leaf_nodes": [15, 31, 63]},
                  cv=kf, scoring="neg_mean_absolute_error").fit(Xtr, ytr)
print("tuned HGB:", gs.best_params_, f"CV MAE {-gs.best_score_:.4f}")
pred = gs.predict(Xte); mae = np.abs(pred - yte).mean()
print(f"FINAL test MAE {mae:.4f} (x$100k)  test R2 {r2_score(yte, pred):.4f}")
cap = yte >= 5.0; print(f"test MAE on capped rows {np.abs(pred-yte)[cap].mean():.4f} (n={cap.sum()}) vs others {np.abs(pred-yte)[~cap].mean():.4f}")
