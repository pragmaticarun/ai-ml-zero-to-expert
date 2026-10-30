"""Day 53: bike demand forecasting with lag features; shuffled KFold vs TimeSeriesSplit; naive baselines."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_bike
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_val_score
df = load_bike()
for c in ["workingday", "holiday"]: df[c] = (df[c].astype(str) == "True").astype(int)
for c in ["year", "month", "hour", "weekday"]: df[c] = df[c].astype(int)
y = df["count"].astype(float)
for L in [1, 2, 24, 168]: df[f"lag{L}"] = y.shift(L)
df["roll24"] = y.shift(1).rolling(24).mean(); df["leaky_roll"] = y.rolling(3, center=True).mean()   # centred window peeks at t+1!
df = df.iloc[168:-1].reset_index(drop=True); y = df.pop("count").astype(float)
base = [c for c in df.columns if not c.startswith(("lag", "roll", "leaky"))]; lags = base + ["lag1", "lag2", "lag24", "lag168", "roll24"]
m = lambda: HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=0)
mae = lambda cols, cv: -cross_val_score(m(), df[cols], y, cv=cv, scoring="neg_mean_absolute_error").mean()
ts = TimeSeriesSplit(5, test_size=1500); sh = KFold(5, shuffle=True, random_state=0)
print(f"naive 'same as 1 hour ago'  MAE {np.abs(y - df.lag1).mean():.1f}; 'same hour last week' MAE {np.abs(y - df.lag168).mean():.1f}")
for name, cols in [("calendar+weather only", base), ("+ lag features", lags), ("+ LEAKY centred rolling mean", lags + ["leaky_roll"])]:
    print(f"{name:30s} TimeSeriesSplit MAE {mae(cols, ts):6.1f}   shuffled KFold MAE {mae(cols, sh):6.1f}")
print("TimeSeriesSplit folds (train end -> test range):", [(int(a[-1]), int(b[0]), int(b[-1])) for a, b in ts.split(df)])
