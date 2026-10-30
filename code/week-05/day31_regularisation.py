"""Day 31: ridge, lasso, elastic net on the diabetes data. Closed-form ridge vs sklearn, coefficient paths, sparsity."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
X, y = load_diabetes(return_X_y=True); names = load_diabetes().feature_names
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
# 1. closed-form ridge (intercept unpenalised: centre y, X already centred) vs sklearn
lam = 10.0; ym = ytr.mean(); n = Xtr.shape[1]
w = np.linalg.solve(Xtr.T @ Xtr + lam * np.eye(n), Xtr.T @ (ytr - ym))
sk = Ridge(alpha=lam).fit(Xtr, ytr)
print(f"closed-form ridge vs sklearn: max|dw| = {np.abs(w - sk.coef_).max():.2e}, intercept {ym:.4f} vs {sk.intercept_:.4f}")
# 2. OLS vs ridge vs lasso vs elastic net test R2 and sparsity
for name, m in [("OLS", LinearRegression()), ("Ridge a=10", Ridge(alpha=10)), ("Lasso a=1", Lasso(alpha=1.0)),
                ("Lasso a=5", Lasso(alpha=5.0)), ("ElasticNet a=1 l1=0.5", ElasticNet(alpha=1.0, l1_ratio=0.5))]:
    m.fit(Xtr, ytr); nz = int((np.abs(m.coef_) > 1e-8).sum())
    print(f"{name:22s} test R2 {r2_score(yte, m.predict(Xte)):.4f}  nonzero coefs {nz}/{n}  ||w||2 {np.linalg.norm(m.coef_):.1f}")
# 3. paths
alphas = np.logspace(-2, 3, 40); rp = np.array([Ridge(alpha=a).fit(Xtr, ytr).coef_ for a in alphas])
la = np.logspace(-2, 1.7, 40); lp = np.array([Lasso(alpha=a, max_iter=10000).fit(Xtr, ytr).coef_ for a in la])
order = [names[j] for j in np.argsort([np.argmax(np.abs(lp[:, j]) < 1e-8) if (np.abs(lp[:, j]) < 1e-8).any() else 99 for j in range(n)])]
print("lasso drop-out order (first to zero -> last):", order)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].semilogx(alphas, rp); ax[0].set_title("Ridge path"); ax[0].set_xlabel("alpha")
ax[1].semilogx(la, lp); ax[1].set_title("Lasso path"); ax[1].set_xlabel("alpha"); ax[1].legend(names, fontsize=6)
savefig(__file__, "day31_paths.png")
# 4. correlated features: ridge shares, lasso picks one
rng = np.random.default_rng(31); z = rng.normal(size=500); Xc = np.c_[z + 0.01 * rng.normal(size=500), z + 0.01 * rng.normal(size=500)]; yc = 3 * z + rng.normal(size=500)
print("twin features  ridge coef:", Ridge(alpha=1).fit(Xc, yc).coef_.round(3), " lasso coef:", Lasso(alpha=0.1).fit(Xc, yc).coef_.round(3))
