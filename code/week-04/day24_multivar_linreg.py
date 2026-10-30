"""Day 24: multiple linear regression, vectorised; normal equation vs GD vs scikit-learn."""
import numpy as np
from sklearn.linear_model import LinearRegression
from housing import make_housing
X, y, names = make_housing(1000, seed=24)
Xs = (X - X.mean(0)) / X.std(0)                     # scaled (Day 25 explains why)
w = np.zeros(4); b = 0.0
for _ in range(2000):
    err = Xs @ w + b - y; w -= 0.1 * Xs.T @ err / len(y); b -= 0.1 * err.mean()
w_orig = w / X.std(0); b_orig = b - (w_orig * X.mean(0)).sum()
Xd = np.column_stack([X, np.ones(len(X))]); theta = np.linalg.lstsq(Xd, y, rcond=None)[0]
sk = LinearRegression().fit(X, y)
print(f"{'feature':10s} {'GD':>9s} {'lstsq':>9s} {'sklearn':>9s} {'true':>6s}")
for n_, a, c, d, t in zip(names, w_orig, theta, sk.coef_, [0.05, 4, -0.4, -0.8]): print(f"{n_:10s} {a:9.4f} {c:9.4f} {d:9.4f} {t:6.2f}")
print(f"intercept  {b_orig:9.3f} {theta[-1]:9.3f} {sk.intercept_:9.3f} {20:6.2f}")
print("R^2 train:", round(sk.score(X, y), 4))
