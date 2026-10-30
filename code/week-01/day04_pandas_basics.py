"""Day 4: pandas for tabular ML data, using a synthetic housing table (no download)."""
import numpy as np, pandas as pd
rng = np.random.default_rng(4); n = 500
df = pd.DataFrame({
    "area_sqft": rng.normal(1200, 350, n).clip(350).round(),
    "bedrooms": rng.integers(1, 5, n),
    "age_years": rng.integers(0, 40, n),
    "city": rng.choice(["Chennai", "Bengaluru", "Pune"], n, p=[.4, .4, .2]),
})
city_premium = df.city.map({"Chennai": 0, "Bengaluru": 15, "Pune": 5})
df["price_lakh"] = (0.05*df.area_sqft + 4*df.bedrooms - 0.4*df.age_years + city_premium + rng.normal(0, 6, n)).round(1)
df.loc[rng.choice(n, 25, replace=False), "age_years"] = np.nan   # missing values, as in real life
print(df.head()); print(df.dtypes); print(df.describe().round(1))
print("missing per column:\n", df.isna().sum())
df["age_years"] = df["age_years"].fillna(df["age_years"].median())
print("mean price by city:\n", df.groupby("city")["price_lakh"].agg(["mean", "count"]).round(1))
df["price_per_sqft"] = df.price_lakh * 1e5 / df.area_sqft
print("top 3 most expensive per sqft:\n", df.nlargest(3, "price_per_sqft")[["city", "area_sqft", "price_lakh"]])
# price_per_sqft is derived from the TARGET -> using it as a feature is leakage, so drop it too
X = pd.get_dummies(df.drop(columns=["price_lakh", "price_per_sqft"]), columns=["city"], dtype=float)
print("model-ready columns:", list(X.columns)); print("as numpy:", X.to_numpy().shape)
import os; os.makedirs(os.path.join(os.path.dirname(__file__), "out"), exist_ok=True)
df.to_csv(os.path.join(os.path.dirname(__file__), "out", "housing_synth.csv"), index=False); print("wrote out/housing_synth.csv")
