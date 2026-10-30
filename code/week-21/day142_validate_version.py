"""Day 142: (1) data validation: a schema inferred from training data (types, null policy, ranges, label set) checked on new batches with injected faults;
(2) DVC-style content-addressed dataset versioning, built from scratch: sha256 of a canonical serialisation, an object store, and a small manifest/lineage file."""
import os, json, hashlib, shutil, numpy as np, _mlops as M
Xtr, Xte, ytr, yte, names = M.data()
def infer_schema(X, y):
    return {"n_features": X.shape[1], "features": {n: {"min": float(X[:, j].min()), "max": float(X[:, j].max())} for j, n in enumerate(names)}, "labels": sorted(set(int(v) for v in y))}
def validate(X, y, sc, tol=0.5):
    errs = []
    if X.shape[1] != sc["n_features"]: return [f"expected {sc['n_features']} columns, got {X.shape[1]}"]
    if np.isnan(X).any(): errs.append(f"{int(np.isnan(X).any(1).sum())} rows with NaN")
    for j, n in enumerate(names):
        lo, hi = sc["features"][n]["min"], sc["features"][n]["max"]; span = hi - lo; c = X[:, j][~np.isnan(X[:, j])]
        bad = ((c < lo - tol * span) | (c > hi + tol * span)).sum()
        if bad: errs.append(f"{n}: {int(bad)} values outside [{lo - tol*span:.3g}, {hi + tol*span:.3g}]")
    if (bad := set(int(v) for v in y) - set(sc["labels"])): errs.append(f"unknown labels {sorted(bad)}")
    return errs
sc = infer_schema(Xtr, ytr)
print("== (1) validation: schema inferred from train (ranges widened by 50% of span)")
rng = np.random.default_rng(0)
def fault(kind):
    X, y = Xte.copy(), yte.copy()
    if kind == "NaNs in 5 rows": X[rng.choice(len(X), 5, replace=False), 3] = np.nan
    if kind == "unit change: area in mm^2 (x100)": X[:, names.index("mean area")] *= 100
    if kind == "dropped column": X = X[:, 1:]
    if kind == "label 2 appears": y[:3] = 2
    if kind == "silent shift: radius +10%": X[:, names.index("mean radius")] *= 1.1
    return X, y
for k in ("clean test batch", "NaNs in 5 rows", "unit change: area in mm^2 (x100)", "dropped column", "label 2 appears", "silent shift: radius +10%"):
    e = validate(*fault(k), sc); print(f"  {k:34s}: {'PASS' if not e else 'FAIL'} {e[:2]}")
print("== (2) content-addressed versioning")
def canon(X, y, sort_rows=False):
    A = np.column_stack([X, y]).astype(np.float64)
    if sort_rows: A = A[np.lexsort(A.T[::-1])]
    return A.tobytes()
def h(b): return hashlib.sha256(b).hexdigest()
store = os.path.join(M.OUT, "dvc_store"); shutil.rmtree(store, ignore_errors=True); os.makedirs(store)
def commit(X, y, msg, parent=None):
    b = canon(X, y); d = h(b); p = os.path.join(store, d[:2], d[2:])
    new = not os.path.exists(p)
    if new: os.makedirs(os.path.dirname(p), exist_ok=True); open(p, "wb").write(b)
    ent = {"hash": d[:12], "rows": len(X), "msg": msg, "parent": parent}; open(os.path.join(store, "log.jsonl"), "a").write(json.dumps(ent) + "\n")
    return d[:12], new
v1, n1 = commit(Xtr, ytr, "initial train split"); print(f"  v1 {v1} stored {n1}")
v1b, n1b = commit(Xtr.copy(), ytr.copy(), "same data again", v1); print(f"  same data re-committed: {v1b} stored-new {n1b} (dedup)")
Xp = Xtr[np.random.default_rng(1).permutation(len(Xtr))]; yp = ytr[np.random.default_rng(1).permutation(len(Xtr))]
print(f"  row-shuffled copy: raw hash equal {h(canon(Xtr,ytr))==h(canon(Xp,yp))}; order-invariant hash equal {h(canon(Xtr,ytr,True))==h(canon(Xp,yp,True))}")
y2 = ytr.copy(); y2[7] = 1 - y2[7]; v2, _ = commit(Xtr, y2, "fixed 1 label (row 7)", v1); print(f"  one label flipped -> v2 {v2} (differs from v1: {v2 != v1})")
X3 = Xtr.copy(); X3[0, 0] += 1e-4; print(f"  one value +1e-4 -> hash differs: {h(canon(X3,ytr)) != h(canon(Xtr,ytr))} (hashes detect ANY change, even meaningless ones)")
sz = sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(store) for f in fs)
print(f"  store: {sum(1 for _ in open(os.path.join(store,'log.jsonl')))} log entries, {sz/1e3:.1f} kB on disk; lineage of v2: {v2} <- {v1}")
