"""Shared synthetic housing generator for Week 4 (deterministic, no download)."""
import numpy as np
def make_housing(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    area = rng.normal(1200, 350, n).clip(350)          # sq ft
    beds = rng.integers(1, 5, n).astype(float)
    age = rng.integers(0, 40, n).astype(float)
    dist = rng.uniform(1, 30, n)                         # km to city centre
    price = 0.05*area + 4*beds - 0.4*age - 0.8*dist + 20 + rng.normal(0, 6, n)
    X = np.column_stack([area, beds, age, dist]); return X, price, ["area_sqft", "bedrooms", "age_years", "dist_km"]
