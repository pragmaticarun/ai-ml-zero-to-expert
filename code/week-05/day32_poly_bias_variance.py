"""Day 32: polynomial degree sweep and an empirical bias^2/variance decomposition."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
f = lambda x: np.sin(2 * np.pi * x); sigma = 0.3
rng = np.random.default_rng(32)
def sample(m): x = rng.uniform(0, 1, m); return x[:, None], f(x) + sigma * rng.normal(size=m)
Xtr, ytr = sample(30); Xva, yva = sample(1000)
mse = lambda m, X, y: ((m.predict(X) - y) ** 2).mean()
model = lambda d, a=0: make_pipeline(PolynomialFeatures(d), StandardScaler(), Ridge(alpha=a) if a else LinearRegression())
print("degree  train_MSE  val_MSE")
tr, va = [], []
for d in range(1, 16):
    m = model(d).fit(Xtr, ytr); tr.append(mse(m, Xtr, ytr)); va.append(mse(m, Xva, yva)); print(f"{d:6d}  {tr[-1]:9.4f}  {va[-1]:8.4f}")
print(f"best degree by val: {1 + int(np.argmin(va))}; noise floor sigma^2 = {sigma**2:.3f}")
plt.semilogy(range(1, 16), tr, "o-", label="train"); plt.semilogy(range(1, 16), va, "s-", label="validation"); plt.axhline(sigma**2, ls=":", c="k", label="noise σ²")
plt.xlabel("degree"); plt.ylabel("MSE"); plt.legend(); savefig(__file__, "day32_degree_sweep.png")
# bias-variance over 200 training sets
xg = np.linspace(0.05, 0.95, 50)[:, None]
print("model                bias^2   variance  bias^2+var+noise  measured_test_MSE")
for d, a in [(1, 0), (3, 0), (9, 0), (9, 1e-3)]:
    P = []; 
    for _ in range(200): X_, y_ = sample(30); P.append(model(d, a).fit(X_, y_).predict(xg))
    P = np.array(P); b2 = ((P.mean(0) - f(xg[:, 0])) ** 2).mean(); var = P.var(0).mean()
    yg = f(xg[:, 0]) + sigma * rng.normal(size=(200, 50)); meas = ((P - yg) ** 2).mean()
    print(f"deg {d} ridge {a:<7g}  {b2:7.4f}  {var:8.4f}  {b2 + var + sigma**2:16.4f}  {meas:17.4f}")
