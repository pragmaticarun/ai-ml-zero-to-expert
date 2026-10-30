"""Day 143: experiment tracking with MLflow (local file store, no server) and with a 15-line own tracker; reproducibility checks."""
import os, json, time, shutil, subprocess, sys, numpy as np, torch, _mlops as M
import mlflow
Xtr, Xte, ytr, yte, _ = M.data()
from sklearn.model_selection import train_test_split
Xa, Xv, ya, yv = train_test_split(Xtr, ytr, test_size=0.25, random_state=1, stratify=ytr)  # validation split: choose on val, report test once
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"  # mlflow 3.17 file store is in maintenance mode; sqlite needs sqlalchemy (not in mlflow-skinny)
uri = os.path.join(M.OUT, "mlruns"); shutil.rmtree(uri, ignore_errors=True)
mlflow.set_tracking_uri("file:" + uri); mlflow.set_experiment("w21-mlp")
own = os.path.join(M.OUT, "own_runs.jsonl"); open(own, "w").close()
def own_log(**kw): open(own, "a").write(json.dumps(kw) + "\n")
print(f"mlflow {mlflow.__version__}; grid: hidden x lr, 3 seeds each")
t0 = time.perf_counter()
for h in (8, 32, 128):
    for lr in (1e-3, 1e-2):
        for s in range(3):
            m = M.train(Xa, ya, h=h, lr=lr, seed=s); va = M.acc(m, Xv, yv)
            with mlflow.start_run():
                mlflow.log_params({"h": h, "lr": lr, "seed": s}); mlflow.log_metric("val_acc", va); mlflow.set_tag("weights_sha", M.weights_hash(m))
            own_log(h=h, lr=lr, seed=s, val_acc=va, weights=M.weights_hash(m))
print(f"18 runs logged to both trackers in {time.perf_counter()-t0:.1f} s")
df = mlflow.search_runs(); g = df.groupby(["params.h", "params.lr"])["metrics.val_acc"].agg(["mean", "std"]).sort_values("mean", ascending=False)
print("mlflow search_runs, mean/std val acc over 3 seeds:"); print(g.round(3).to_string())
rows = [json.loads(l) for l in open(own)]; best = max(rows, key=lambda r: r["val_acc"])
print(f"own tracker: {len(rows)} rows; single best run h={best['h']} lr={best['lr']} seed={best['seed']} val {best['val_acc']:.3f} (picking one lucky seed = optimistic)")
bh, blr = g.index[0]; m = M.train(Xtr, ytr, h=int(bh), lr=float(blr), seed=0); print(f"chosen by mean: h={bh} lr={blr}; retrained on full train, TEST acc {M.acc(m, Xte, yte):.3f} (reported once)")
print("== reproducibility")
a, b = M.train(Xtr, ytr, seed=0), M.train(Xtr, ytr, seed=0); print(f"same seed twice, same process: weights {M.weights_hash(a)} vs {M.weights_hash(b)} equal {M.weights_hash(a)==M.weights_hash(b)}")
c = M.train(Xtr, ytr, seed=1); print(f"seed 1: weights {M.weights_hash(c)}; test acc seed0 {M.acc(a,Xte,yte):.3f} vs seed1 {M.acc(c,Xte,yte):.3f}")
code = "import sys;sys.path.insert(0,'.');import torch,_mlops as M;torch.set_num_threads(%d);a=M.data();print(M.weights_hash(M.train(a[0],a[2],seed=0)))"
for th in (1, 4):
    r = subprocess.run([sys.executable, "-c", code % th], capture_output=True, text=True); print(f"fresh process, {th} thread(s): {r.stdout.strip()}")
env = {"python": sys.version.split()[0], "torch": torch.__version__, "numpy": np.__version__, "mlflow": mlflow.__version__}
print("environment to record with every run:", json.dumps(env))
