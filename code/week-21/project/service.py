"""Week-21 project service: ONNX model behind FastAPI with input validation, version info, request logging and a drift endpoint."""
import os, sys, json, time, collections, numpy as np, onnxruntime as ort
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
HERE = os.path.dirname(os.path.abspath(__file__)); ART = os.environ.get("ARTIFACTS", os.path.join(HERE, "artifacts"))
META = json.load(open(os.path.join(ART, "meta.json"))); REF = np.load(os.path.join(ART, "reference.npy"))
so = ort.SessionOptions(); so.intra_op_num_threads = 1; SESS = ort.InferenceSession(os.path.join(ART, "model.onnx"), so)
def psi(ref, cur, bins=10):
    e = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1))); e[0], e[-1] = -np.inf, np.inf
    r = np.histogram(ref, e)[0] / len(ref) + 1e-4; c = np.histogram(cur, e)[0] / len(cur) + 1e-4; return float(((c - r) * np.log(c / r)).sum())
app = FastAPI(title="w21-breast-cancer-mlp"); WIN = collections.deque(maxlen=int(os.environ.get("WINDOW", 500))); STATS = {"requests": 0, "rows": 0, "rejected": 0, "lat_ms": collections.deque(maxlen=2000)}
class Req(BaseModel):
    rows: list[list[float]] = Field(min_length=1, max_length=512)
@app.get("/health")
def health(): return {"status": "ok", "model_version": META["weights_sha"], "data_version": META["data_hash"]}
@app.post("/predict")
def predict(r: Req):
    X = np.asarray(r.rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != META["n_features"]: STATS["rejected"] += 1; raise HTTPException(422, f"expected rows of {META['n_features']} features")
    if not np.isfinite(X).all(): STATS["rejected"] += 1; raise HTTPException(422, "non-finite values")
    t = time.perf_counter(); z = SESS.run(None, {"x": X})[0]; z = z - z.max(1, keepdims=True); p = np.exp(z[:, 1]) / np.exp(z).sum(1)
    STATS["lat_ms"].append((time.perf_counter() - t) * 1000); STATS["requests"] += 1; STATS["rows"] += len(X); WIN.extend(X)
    return {"p_benign": p.round(6).tolist(), "label": (p > META["threshold"]).astype(int).tolist(), "model_version": META["weights_sha"]}
@app.get("/metrics")
def metrics():
    out = {k: STATS[k] for k in ("requests", "rows", "rejected")}
    if STATS["lat_ms"]: out["model_p95_ms"] = float(np.percentile(STATS["lat_ms"], 95))
    if len(WIN) >= META["min_window"]:
        W = np.array(WIN); P = {META["features"][k]: psi(REF[:, k], W[:, k]) for k in range(REF.shape[1])}
        out["drift"] = {"window": len(W), "max_psi": max(P.values()), "feature": max(P, key=P.get), "alert": max(P.values()) > META["psi_alert"]}
    return out
