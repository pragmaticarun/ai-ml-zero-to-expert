"""Day 45: imbalanced classification: accuracy trap, class weights, oversampling, threshold tuning on validation."""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score, roc_auc_score, precision_recall_curve
X, y = make_classification(20000, 20, n_informative=6, weights=[0.98, 0.02], class_sep=0.8, flip_y=0.005, random_state=45)
Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=0); Xva, Xte, yva, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp, random_state=0)
print(f"positive rate train {ytr.mean():.4f}; n_pos test {yte.sum()}")
def rep(name, p, thr=0.5):
    yh = (p >= thr).astype(int)
    print(f"{name:34s} acc {accuracy_score(yte, yh):.4f}  prec {precision_score(yte, yh, zero_division=0):.3f}  rec {recall_score(yte, yh):.3f}  F1 {f1_score(yte, yh):.3f}  PR-AUC {average_precision_score(yte, p):.3f}  ROC-AUC {roc_auc_score(yte, p):.3f}")
rep("always negative", np.zeros(len(yte)))
plain = LogisticRegression(max_iter=1000).fit(Xtr, ytr); rep("logreg, thr 0.5", plain.predict_proba(Xte)[:, 1])
bal = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr); rep("logreg class_weight=balanced", bal.predict_proba(Xte)[:, 1])
rng = np.random.default_rng(45); pos = np.where(ytr == 1)[0]; extra = rng.choice(pos, (ytr == 0).sum() - len(pos))
ov = LogisticRegression(max_iter=1000).fit(np.r_[Xtr, Xtr[extra]], np.r_[ytr, ytr[extra]]); rep("logreg random oversampling", ov.predict_proba(Xte)[:, 1])
pv = plain.predict_proba(Xva)[:, 1]; P, R, T = precision_recall_curve(yva, pv); F = 2 * P * R / (P + R + 1e-12); t = T[np.argmax(F[:-1])]
print(f"threshold chosen on VALIDATION for best F1: {t:.4f}"); rep(f"logreg, thr {t:.3f} (tuned)", plain.predict_proba(Xte)[:, 1], t)
i = np.argmax(R[:-1] < 0.8) - 1; print(f"to get recall>=0.8 on validation: threshold {T[i]:.4f}, precision {P[i]:.3f}")
