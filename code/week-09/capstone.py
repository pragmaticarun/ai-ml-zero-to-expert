"""Month-2 capstone library: data checks, candidate pipelines, CV selection, threshold choice, report helpers (adult income)."""
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as sel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.metrics import precision_recall_curve
EXPECTED = {"age": (17, 90), "hours-per-week": (1, 99), "education-num": (1, 16)}
def check_data(X, y):
    problems = []
    for c, (lo, hi) in EXPECTED.items():
        bad = (~X[c].between(lo, hi)).sum()
        if bad: problems.append(f"{c}: {bad} out of range [{lo},{hi}]")
    if set(np.unique(y)) - {0, 1}: problems.append("labels not binary")
    dup = X.duplicated().sum()
    return problems, {"rows": len(X), "duplicates": int(dup), "missing_cells": int(X.isna().sum().sum()), "pos_rate": round(float(np.mean(y)), 4)}
def candidates(seed=0):
    ohe = ColumnTransformer([("num", StandardScaler(), sel(dtype_include="number")),
                             ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"), OneHotEncoder(handle_unknown="ignore", min_frequency=20)), sel(dtype_include="category"))])
    ordn = ColumnTransformer([("num", "passthrough", sel(dtype_include="number")),
                              ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1), sel(dtype_include="category"))])
    return {"logreg": make_pipeline(ohe, LogisticRegression(C=0.1, max_iter=3000)),
            "random_forest": make_pipeline(ordn, RandomForestClassifier(300, min_samples_leaf=3, n_jobs=-1, random_state=seed)),
            "hist_gb": HistGradientBoostingClassifier(categorical_features="from_dtype", learning_rate=0.1, max_leaf_nodes=31, random_state=seed),
            "hist_gb_tuned": HistGradientBoostingClassifier(categorical_features="from_dtype", learning_rate=0.05, max_iter=500, max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, random_state=seed)}
def cv_table(models, X, y, seed=0):
    cv = StratifiedKFold(5, shuffle=True, random_state=seed); rows = []
    for n, m in models.items():
        r = cross_validate(m, X, y, cv=cv, scoring=("roc_auc", "average_precision", "neg_log_loss", "accuracy"))
        rows.append({"model": n, "roc_auc": r["test_roc_auc"].mean(), "roc_auc_std": r["test_roc_auc"].std(), "pr_auc": r["test_average_precision"].mean(),
                     "log_loss": -r["test_neg_log_loss"].mean(), "accuracy": r["test_accuracy"].mean(), "fit_s": r["fit_time"].mean()})
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
def pick_threshold(model, X, y, min_precision=0.7, seed=0):
    p = cross_val_predict(model, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    P, R, T = precision_recall_curve(y, p); ok = np.where(P[:-1] >= min_precision)[0]; i = ok[np.argmax(R[ok])]
    return float(T[i]), float(P[i]), float(R[i])
def slice_metrics(y, p, thr, groups):
    df = pd.DataFrame({"y": np.asarray(y), "yh": (p >= thr).astype(int), "g": np.asarray(groups)})
    f = lambda d: pd.Series({"n": len(d), "pos_rate": d.y.mean(), "pred_pos_rate": d.yh.mean(),
                             "recall": d.yh[d.y == 1].mean() if (d.y == 1).any() else np.nan, "precision": d.y[d.yh == 1].mean() if (d.yh == 1).any() else np.nan})
    return df.groupby("g", observed=True)[["y", "yh"]].apply(f)
