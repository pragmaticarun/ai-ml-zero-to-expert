"""FastAPI service for the Week-21 MLP. Backend chosen by env BACKEND=torch|onnx. Pydantic validates the request body."""
import os, time, json, collections, numpy as np, torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
import _mlops as M
HERE = os.path.dirname(os.path.abspath(__file__)); BACKEND = os.environ.get("BACKEND", "torch")
CARD = json.load(open(os.path.join(M.OUT, "model_meta.json")))
if BACKEND == "onnx":
    import onnxruntime as ort; so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(os.path.join(M.OUT, "model.onnx"), so); predict = lambda X: sess.run(None, {"x": X})[0]
else:
    model = M.MLP(); model.load_state_dict(torch.load(os.path.join(M.OUT, "model.pt"))); model.eval()
    def predict(X):
        with torch.no_grad(): return model(torch.from_numpy(X)).numpy()
app = FastAPI(); LOG = collections.deque(maxlen=5000)   # recent inputs + outputs, for monitoring (Day 145/147)
class Req(BaseModel):
    rows: list[list[float]] = Field(min_length=1, max_length=512)
@app.get("/health")
def health(): return {"status": "ok", "backend": BACKEND, "model_version": CARD["weights_sha"]}
@app.post("/predict")
def pred(r: Req):
    X = np.asarray(r.rows, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != 30: return {"error": f"expected rows of 30 features, got shape {list(X.shape)}"}
    t = time.perf_counter(); z = predict(X); p = (np.exp(z[:, 1]) / np.exp(z).sum(1)).tolist()
    for x, q in zip(X, p): LOG.append((x.tolist(), q))
    return {"p_benign": p, "model_version": CARD["weights_sha"], "infer_ms": (time.perf_counter() - t) * 1000}
