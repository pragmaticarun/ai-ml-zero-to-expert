"""Day 27: evaluation: confusion matrix, precision/recall/F1, thresholds, ROC-AUC, bias vs variance."""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score
d = load_breast_cancer(); y = 1 - d.target          # make 1 = malignant (the class we care about)
Xtr, Xte, ytr, yte = train_test_split(d.data, y, test_size=.25, random_state=0, stratify=y)
m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xtr, ytr); p = m.predict_proba(Xte)[:, 1]
for t in [0.5, 0.3, 0.1]:
    yh = (p >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yte, yh).ravel()
    print(f"thr={t}: TP={tp} FP={fp} FN={fn} TN={tn} precision={precision_score(yte, yh):.3f} recall={recall_score(yte, yh):.3f} F1={f1_score(yte, yh):.3f}")
print("ROC-AUC:", round(roc_auc_score(yte, p), 4), "| accuracy of always-'benign' baseline:", round((yte == 0).mean(), 4))
# Bias/variance: polynomial degree on a small noisy 1-D regression
rng = np.random.default_rng(27); x = rng.uniform(-3, 3, 40)[:, None]; t = np.sin(x[:, 0]) + rng.normal(0, .3, 40)
xv = rng.uniform(-3, 3, 200)[:, None]; tv = np.sin(xv[:, 0]) + rng.normal(0, .3, 200)
from sklearn.linear_model import LinearRegression, Ridge
for deg in [1, 3, 15]:
    for name, reg in [("OLS", LinearRegression()), ("Ridge a=1", Ridge(1.0))]:
        mdl = make_pipeline(PolynomialFeatures(deg), StandardScaler(), reg).fit(x, t)
        print(f"deg={deg:2d} {name:9s} train MSE={((mdl.predict(x)-t)**2).mean():.3f}  val MSE={((mdl.predict(xv)-tv)**2).mean():.3f}")
