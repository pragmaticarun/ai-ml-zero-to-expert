"""Day 6: train/validation/test splits, seeds, data leakage demo."""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
rng = np.random.default_rng(6)
X = rng.normal(size=(300, 1)); y = 3*X[:, 0] + rng.normal(0, 1, 300)
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=0)
print("sizes train/val/test:", len(X_tr), len(X_val), len(X_test))
m = LinearRegression().fit(X_tr, y_tr)
for name, A, b in [("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_test, y_test)]:
    print(f"{name:5s} MSE={np.mean((m.predict(A)-b)**2):.3f}")
# Same seed => same split; different seed => different numbers. Report seeds!
for s in [0, 0, 1]:
    a, *_ = train_test_split(X, y, test_size=.2, random_state=s); print("seed", s, "first train x:", a[0, 0].round(4))
# Leakage: pure-noise features + choosing features using ALL data looks great, then fails on new data
Xn = rng.normal(size=(100, 2000)); yn = rng.normal(size=100)
corr = np.abs([np.corrcoef(Xn[:, j], yn)[0, 1] for j in range(2000)]); top = np.argsort(corr)[-10:]
m2 = LinearRegression().fit(Xn[:80, top], yn[:80])
print("LEAKY: test R^2 on held-out 20 rows (features picked using them):", round(m2.score(Xn[80:, top], yn[80:]), 3))
corr_tr = np.abs([np.corrcoef(Xn[:80, j], yn[:80])[0, 1] for j in range(2000)]); top2 = np.argsort(corr_tr)[-10:]
m3 = LinearRegression().fit(Xn[:80, top2], yn[:80])
print("CLEAN: test R^2 (features picked on train only):", round(m3.score(Xn[80:, top2], yn[80:]), 3))
