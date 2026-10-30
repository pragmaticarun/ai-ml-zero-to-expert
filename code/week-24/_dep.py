"""Week 24 shared helpers: Adult census data (OpenML 'adult' v2, cached), a logistic-regression model, group metrics, CIFAR CNN."""
import sys, os; H = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(H, ".."))
import numpy as np
from common import load_adult
OUT = os.path.join(H, "out"); os.makedirs(OUT, exist_ok=True)
def adult(seed=0):
    from sklearn.model_selection import train_test_split
    X, y = load_adult(); X = X.copy()
    for c in X.select_dtypes("category").columns: X[c] = X[c].cat.add_categories(["?"]).fillna("?")
    sex = (X["sex"].astype(str) == "Female").astype(int).values        # 1 = Female (protected attribute for the audit)
    return train_test_split(X, y.values, sex, test_size=0.3, random_state=seed, stratify=y.values)
def lr_pipe(drop_sex=False, C=1.0):
    from sklearn.compose import ColumnTransformer; from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler; from sklearn.linear_model import LogisticRegression
    cats = ["workclass","education","marital-status","occupation","relationship","race","native-country"] + ([] if drop_sex else ["sex"])
    nums = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]
    ct = ColumnTransformer([("c", OneHotEncoder(handle_unknown="ignore"), cats), ("n", StandardScaler(), nums)])
    return make_pipeline(ct, LogisticRegression(max_iter=2000, C=C))
def group_metrics(y, p, yhat, g):
    """Per-group selection rate, TPR, FPR, precision, ECE (10 bins); gaps = group1 - group0."""
    r = {}
    for k in (0, 1):
        m = g == k; yy, pp, hh = y[m], p[m], yhat[m]
        bins = np.minimum((pp * 10).astype(int), 9); ece = sum(abs(pp[bins == b].mean() - yy[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
        r[k] = dict(n=int(m.sum()), base=yy.mean(), sel=hh.mean(), tpr=hh[yy == 1].mean(), fpr=hh[yy == 0].mean(), prec=yy[hh == 1].mean() if hh.any() else np.nan, ece=ece, acc=(hh == yy).mean())
    gap = {k: r[1][k] - r[0][k] for k in ("sel", "tpr", "fpr", "prec", "ece", "acc")}
    gap["di_ratio"] = r[1]["sel"] / r[0]["sel"]; return r, gap
def show(name, y, p, yhat, g):
    r, gap = group_metrics(y, p, yhat, g)
    print(f"{name:34s} acc {(yhat==y).mean():.4f} | sel M {r[0]['sel']:.3f} F {r[1]['sel']:.3f} (DI {gap['di_ratio']:.2f}) | TPR M {r[0]['tpr']:.3f} F {r[1]['tpr']:.3f} | FPR M {r[0]['fpr']:.3f} F {r[1]['fpr']:.3f} | prec M {r[0]['prec']:.3f} F {r[1]['prec']:.3f}")
    return r, gap
def cifar_cnn():
    """Month 3 CNN (Day 73: CNN + BN + flip/crop, weights in code/week-11/out/day73_cnn.pt) and the fixed CIFAR-10-small split."""
    import torch, torch.nn as nn; sys.path.insert(0, os.path.join(H, "..", "week-11")); from _cifar import splits
    blk = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2)]
    m = nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128*16, 10))
    m.load_state_dict(torch.load(os.path.join(H, "..", "week-11", "out", "day73_cnn.pt"))); m.eval(); return m, splits()
