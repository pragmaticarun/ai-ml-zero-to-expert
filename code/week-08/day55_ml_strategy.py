"""Day 55: diagnose before you fix: learning curves for a high-bias and a high-variance model; error analysis by slice."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_adult
from sklearn.model_selection import learning_curve, StratifiedKFold, train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
X, y = load_adult(); Xo = X.copy(); cat = X.select_dtypes("category").columns
Xo[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(X[cat]); cv = StratifiedKFold(5, shuffle=True, random_state=0)
sizes = [0.02, 0.1, 0.3, 0.6, 1.0]
for name, m, XX in [("logreg on 6 numeric cols (high bias?)", make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), X.select_dtypes("number")),
                    ("unpruned tree (high variance?)", DecisionTreeClassifier(random_state=0), Xo),
                    ("HistGB", HistGradientBoostingClassifier(random_state=0), Xo)]:
    n, tr, va = learning_curve(m, XX, y, train_sizes=sizes, cv=cv, n_jobs=-1)
    print(f"\n{name}\n  n_train  train_err  val_err  gap")
    for a, b, c in zip(n, 1 - tr.mean(1), 1 - va.mean(1)): print(f"  {a:7d}  {b:9.4f}  {c:7.4f}  {c - b:.4f}")
print("\nreference: majority-class error", round(float(y.mean()), 4))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
p = cross_val_predict(HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0), Xtr, ytr, cv=cv, method="predict_proba")[:, 1]
E = Xtr.assign(err=((p > .5) != ytr).astype(int), y=ytr)
for col in ["education", "sex", "marital-status"]:
    g = E.groupby(col, observed=True).agg(n=("err", "size"), err_rate=("err", "mean"), pos_rate=("y", "mean")).sort_values("n", ascending=False).head(5)
    g["share_of_all_errors"] = (g.n * g.err_rate / E.err.sum()); print(f"\nerror analysis by {col} (top 5 by size, CV predictions on train):"); print(g.round(4).to_string())
