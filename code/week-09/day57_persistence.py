"""Day 57: reproducible experiments: seeds, data fingerprint, config, joblib persistence, round-trip check, run log."""
import sys, os, json, hashlib, platform, time, joblib; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, pandas as pd, sklearn
from common import load_adult, outdir
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
OUT = outdir(__file__)
X, y = load_adult()
fp = hashlib.sha256(pd.util.hash_pandas_object(X.assign(y=y), index=True).values.tobytes()).hexdigest()[:16]
cfg = {"seed": 0, "test_size": 0.25, "model": "HistGradientBoostingClassifier", "params": {"learning_rate": 0.1, "max_leaf_nodes": 31, "random_state": 0}}
def run(cfg):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg["test_size"], random_state=cfg["seed"], stratify=y)
    m = HistGradientBoostingClassifier(categorical_features="from_dtype", **cfg["params"]).fit(Xtr, ytr)
    return m, Xte, yte, roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
m1, Xte, yte, a1 = run(cfg); m2, _, _, a2 = run(cfg)
print(f"data fingerprint {fp}; run1 AUC {a1:.6f} run2 AUC {a2:.6f} identical: {a1 == a2}")
aucs = []
for sd in range(5):
    c = json.loads(json.dumps(cfg)); c["seed"] = sd; c["params"]["random_state"] = sd; aucs.append(run(c)[3])
print("AUC over 5 seeds (split + model):", np.round(aucs, 4).tolist(), f"mean {np.mean(aucs):.4f} std {np.std(aucs):.4f}: report the spread, not one lucky run")
path = os.path.join(OUT, "model.joblib"); t = time.perf_counter(); joblib.dump(m1, path); m_load = joblib.load(path)
same = np.array_equal(m1.predict_proba(Xte), m_load.predict_proba(Xte))
print(f"joblib dump+load {1000*(time.perf_counter()-t):.0f} ms, file {os.path.getsize(path)/1024:.0f} KiB, predictions bit-identical after reload: {same}")
meta = {"data_sha256_16": fp, "config": cfg, "test_auc": round(a1, 6), "python": platform.python_version(), "sklearn": sklearn.__version__, "numpy": np.__version__, "pandas": pd.__version__}
json.dump(meta, open(os.path.join(OUT, "model_card.json"), "w"), indent=2); print("wrote model_card.json with keys", list(meta))
log = os.path.join(OUT, "runs.csv"); row = pd.DataFrame([{"time": time.strftime("%Y-%m-%d %H:%M"), "data": fp, "seed": cfg["seed"], **cfg["params"], "auc": round(a1, 6)}])
row.to_csv(log, mode="a", header=not os.path.exists(log), index=False); print("appended a row to runs.csv; rows now", len(pd.read_csv(log)))
