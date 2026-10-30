"""Day 29 capstone: scratch NumPy models vs scikit-learn (accuracy + wall time), optional Rust/PyO3 kernel."""
import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-04"))
from housing import make_housing
from mlscratch import StandardScalerNP, LinearRegressionGD, LogisticRegressionGD
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
def timed(f, reps=3):
    best = 1e9
    for _ in range(reps): t = time.perf_counter(); out = f(); best = min(best, time.perf_counter() - t)
    return out, best
X, y, _ = make_housing(20000, seed=29); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.2, random_state=0)
sc = StandardScalerNP().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
print("== Linear regression (synthetic housing, 16000 train rows, 4 features)")
rows = []
m, t = timed(lambda: LinearRegressionGD(lr=0.1, steps=500, tol=0).fit(Xtr_s, ytr)); rows.append(("scratch NumPy GD (500 steps)", mean_absolute_error(yte, m.predict(Xte_s)), m.score(Xte_s, yte), t))
m, t = timed(lambda: LinearRegression().fit(Xtr_s, ytr)); rows.append(("sklearn LinearRegression (lstsq)", mean_absolute_error(yte, m.predict(Xte_s)), m.score(Xte_s, yte), t))
m, t = timed(lambda: SGDRegressor(max_iter=50, tol=None, random_state=0).fit(Xtr_s, ytr)); rows.append(("sklearn SGDRegressor (50 epochs)", mean_absolute_error(yte, m.predict(Xte_s)), m.score(Xte_s, yte), t))
try:
    import rust_gd
    Xc = np.ascontiguousarray(Xtr_s)
    (w, b), t = timed(lambda: rust_gd.linreg_gd(Xc, ytr, 0.1, 500)); p = Xte_s @ w + b
    rows.append(("Rust/PyO3 GD (500 steps)", mean_absolute_error(yte, p), 1 - ((yte-p)**2).sum()/((yte-yte.mean())**2).sum(), t))
except ImportError:
    print("(rust_gd not built; see week-05.md for maturin instructions)")
for name, mae, r2, t in rows: print(f"  {name:34s} MAE={mae:.4f}  R2={r2:.5f}  time={t*1e3:8.2f} ms")
print("== Logistic regression (breast cancer, standardised)")
d = load_breast_cancer(); y2 = 1 - d.target
Xtr, Xte, ytr, yte = train_test_split(d.data, y2, test_size=.25, random_state=0, stratify=y2)
sc = StandardScalerNP().fit(Xtr); Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
m, t = timed(lambda: LogisticRegressionGD(lr=0.5, steps=5000, l2=1.0).fit(Xtr_s, ytr))
print(f"  scratch NumPy GD           acc={m.score(Xte_s, yte):.4f} F1={f1_score(yte, m.predict(Xte_s)):.4f} time={t*1e3:8.2f} ms iters={len(m.history_)}")
m2, t = timed(lambda: LogisticRegression(max_iter=5000).fit(Xtr_s, ytr))
print(f"  sklearn LogisticRegression acc={m2.score(Xte_s, yte):.4f} F1={f1_score(yte, m2.predict(Xte_s)):.4f} time={t*1e3:8.2f} ms")
print("  max |coef difference| scratch vs sklearn:", np.abs(m.w_ - m2.coef_[0]).max().round(4), "(different regularisation scaling, expected)")
