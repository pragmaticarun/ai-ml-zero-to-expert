"""Day 7 project: EDA report on the synthetic housing table + baseline predictor."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np, pandas as pd
p = os.path.join(os.path.dirname(__file__), "out", "housing_synth.csv")
if not os.path.exists(p):
    import runpy; runpy.run_path(os.path.join(os.path.dirname(__file__), "day04_pandas_basics.py"))
df = pd.read_csv(p); rng = np.random.default_rng(7)
idx = rng.permutation(len(df)); tr, te = df.iloc[idx[:400]], df.iloc[idx[400:]]
base = tr.price_lakh.mean(); mae_base = (te.price_lakh - base).abs().mean()
print(f"Baseline 'predict the mean' test MAE: {mae_base:.2f} lakh")
city_mean = tr.groupby("city").price_lakh.mean(); mae_city = (te.price_lakh - te.city.map(city_mean)).abs().mean()
print(f"Baseline 'mean per city' test MAE: {mae_city:.2f} lakh")
w, b = np.polyfit(tr.area_sqft, tr.price_lakh, 1); mae_line = (te.price_lakh - (w*te.area_sqft + b)).abs().mean()
print(f"Line on area only test MAE: {mae_line:.2f} lakh")
print("correlations with price:\n", df.select_dtypes("number").corr()["price_lakh"].round(3))
df.boxplot(column="price_lakh", by="city"); savefig(__file__, "day07_price_by_city.png")
