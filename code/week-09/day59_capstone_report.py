"""Day 59: MONTH-2 CAPSTONE report: load the frozen model, evaluate ONCE on test, calibration, slices, write report.md."""
import sys, os, joblib, subprocess; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd
from common import load_adult, outdir
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, brier_score_loss, accuracy_score
from sklearn.calibration import calibration_curve
from capstone import slice_metrics
OUT = outdir(__file__); pth = os.path.join(OUT, "capstone_model.joblib")
if not os.path.exists(pth): subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "day58_capstone_build.py")], check=True)
B = joblib.load(pth); m, thr = B["model"], B["threshold"]
X, y = load_adult(); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
p = m.predict_proba(Xte)[:, 1]; yh = (p >= thr).astype(int)
res = {"model": B["name"], "threshold": round(thr, 4), "test_roc_auc": roc_auc_score(yte, p), "test_pr_auc": average_precision_score(yte, p),
       "precision@thr": precision_score(yte, yh), "recall@thr": recall_score(yte, yh), "accuracy@0.5": accuracy_score(yte, p >= .5), "brier": brier_score_loss(yte, p)}
for k, v in res.items(): print(f"{k:14s} {v if isinstance(v, str) else round(v, 4)}")
ft, mp = calibration_curve(yte, p, n_bins=10, strategy="quantile"); print("calibration (mean p -> observed):", list(zip(mp.round(3).tolist(), ft.round(3).tolist())))
S = {c: slice_metrics(yte, p, thr, Xte[c]) for c in ["sex", "race"]}
for c, s in S.items(): print(f"\nslices by {c}:"); print(s.round(3).to_string())
rng = np.random.default_rng(0); bs = [roc_auc_score(yte.iloc[i], p[i]) for i in (rng.integers(0, len(yte), len(yte)) for _ in range(200))]
print(f"\nbootstrap 95% interval for test ROC-AUC: [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]")
cv = pd.read_csv(os.path.join(OUT, "cv_table.csv"))
md = ["# Month-2 capstone report: adult income (>50K) classifier", "", "## CV comparison (train split only)", "```\n" + cv.round(4).to_string(index=False) + "\n```", "",
      "## Frozen model, test set (evaluated once)", *[f"- {k}: {v if isinstance(v, str) else round(v, 4)}" for k, v in res.items()],
      f"- bootstrap 95% ROC-AUC interval: [{np.percentile(bs, 2.5):.4f}, {np.percentile(bs, 97.5):.4f}]", "", "## Slices", *[f"### {c}\n```\n{s.round(3).to_string()}\n```" for c, s in S.items()]]
open(os.path.join(OUT, "report.md"), "w").write("\n".join(md)); print("wrote out/report.md")
