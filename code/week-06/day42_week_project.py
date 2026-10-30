"""Day 42 project: adult income, leakage-safe tabular pipeline, logistic regression vs RF vs HistGB, test once."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_adult
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
X, y = load_adult()
print("missing values per column:", X.isna().sum()[X.isna().sum() > 0].to_dict())
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
ohe = ColumnTransformer([("num", StandardScaler(), sel(dtype_include="number")),
                         ("cat", make_pipeline(SimpleImputer(strategy="most_frequent"), OneHotEncoder(handle_unknown="ignore", min_frequency=20)), sel(dtype_include="category"))])
ordn = ColumnTransformer([("num", "passthrough", sel(dtype_include="number")),
                          ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), sel(dtype_include="category"))])
models = {"majority class": DummyClassifier(),
          "logreg + one-hot": make_pipeline(ohe, LogisticRegression(max_iter=2000)),
          "random forest": make_pipeline(ordn, RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=0)),
          "HistGB native categorical": HistGradientBoostingClassifier(categorical_features="from_dtype", random_state=0)}
cv = StratifiedKFold(5, shuffle=True, random_state=0); rows = []
for n, m in models.items():
    t = time.perf_counter(); r = cross_validate(m, Xtr, ytr, cv=cv, scoring=("accuracy", "roc_auc", "f1"), n_jobs=1)
    rows.append((n, r["test_accuracy"].mean(), r["test_roc_auc"].mean(), r["test_f1"].mean(), time.perf_counter() - t))
print(pd.DataFrame(rows, columns=["model", "CV acc", "CV ROC-AUC", "CV F1", "seconds"]).round(4).to_string(index=False))
best = models["HistGB native categorical"].fit(Xtr, ytr); p = best.predict_proba(Xte)[:, 1]
print(f"TEST (once) HistGB: acc {accuracy_score(yte, p > .5):.4f} ROC-AUC {roc_auc_score(yte, p):.4f} F1 {f1_score(yte, p > .5):.4f}")
