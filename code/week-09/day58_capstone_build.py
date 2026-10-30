"""Day 58: capstone build: data checks, CV comparison of 4 pipelines, threshold choice on CV, save the chosen model. No test set use."""
import sys, os, json, joblib; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from common import load_adult, outdir
from sklearn.model_selection import train_test_split
from capstone import check_data, candidates, cv_table, pick_threshold
OUT = outdir(__file__); X, y = load_adult()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
probs, summary = check_data(Xtr, ytr); print("data checks:", summary, "problems:", probs or "none")
T = cv_table(candidates(), Xtr, ytr); print(T.round(4).to_string(index=False)); T.to_csv(os.path.join(OUT, "cv_table.csv"), index=False)
best = T.iloc[0]["model"]; m = candidates()[best]
thr, P, R = pick_threshold(m, Xtr, ytr, min_precision=0.7); print(f"chosen: {best}; threshold for precision>=0.70 on CV preds: {thr:.4f} (CV precision {P:.4f}, recall {R:.4f})")
m.fit(Xtr, ytr); joblib.dump({"model": m, "threshold": thr, "name": best}, os.path.join(OUT, "capstone_model.joblib"))
json.dump({"chosen": best, "threshold": thr}, open(os.path.join(OUT, "capstone_choice.json"), "w")); print("saved out/capstone_model.joblib")
