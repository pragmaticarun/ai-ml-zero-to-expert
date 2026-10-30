"""Day 54: reliability diagrams, Brier score and log loss; calibrating NB/SVM; partial dependence for interpretability."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, matplotlib.pyplot as plt
from common import savefig, load_california
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.inspection import partial_dependence
X, y = make_classification(30000, 20, n_informative=4, n_redundant=10, random_state=54)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)
def prob(m): return m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else 1 / (1 + np.exp(-m.decision_function(Xte)))
models = {"logreg": LogisticRegression(max_iter=1000), "GaussianNB": GaussianNB(), "LinearSVC (sigmoid of margin)": LinearSVC(),
          "GaussianNB + isotonic": CalibratedClassifierCV(GaussianNB(), method="isotonic", cv=5), "LinearSVC + sigmoid (Platt)": CalibratedClassifierCV(LinearSVC(), method="sigmoid", cv=5)}
plt.plot([0, 1], [0, 1], "k:")
for n, m in models.items():
    p = np.clip(prob(m.fit(Xtr, ytr)), 1e-7, 1 - 1e-7); ft, mp = calibration_curve(yte, p, n_bins=10)
    print(f"{n:30s} Brier {brier_score_loss(yte, p):.4f}  log loss {log_loss(yte, p):.4f}  ROC-AUC {roc_auc_score(yte, p):.4f}")
    plt.plot(mp, ft, "o-", label=n)
plt.xlabel("mean predicted p"); plt.ylabel("observed frequency"); plt.legend(fontsize=7); savefig(__file__, "day54_reliability.png")
Xc, yc = load_california(); h = HistGradientBoostingRegressor(random_state=0).fit(Xc, yc)
pd_ = partial_dependence(h, Xc, ["MedInc"], grid_resolution=8); print("partial dependence of price on MedInc:", list(zip(pd_["grid_values"][0].round(2).tolist(), pd_["average"][0].round(3).tolist())))
