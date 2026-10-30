"""Week 23 shared helpers: a small MLP on scikit-learn's bundled digits dataset (offline, 1797 8x8 images),
cached weights, ONNX export, and a median-latency bench. Paths are relative to the repo's out/ folder."""
import os, time, statistics, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"); os.makedirs(OUT, exist_ok=True)
def P(name): return os.path.abspath(os.path.join(OUT, name))
def data():
    X, y = load_digits(return_X_y=True); X = (X / 16.0).astype(np.float32)
    return train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)  # Xtr, Xte, ytr, yte
def mlp(h=512, depth=2):
    layers, d = [], 64
    for _ in range(depth): layers += [nn.Linear(d, h), nn.ReLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, 10))
def train(model, Xtr, ytr, epochs=60, lr=1e-3, seed=0, teacher=None, T=4.0, alpha=0.7):
    torch.manual_seed(seed); opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, y = torch.from_numpy(Xtr), torch.from_numpy(ytr).long(); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(64):
            out = model(X[idx]); loss = nn.functional.cross_entropy(out, y[idx])
            if teacher is not None:  # Hinton-style distillation: KL between softened distributions, scaled by T^2
                with torch.no_grad(): tl = teacher(X[idx])
                kd = nn.functional.kl_div(nn.functional.log_softmax(out / T, 1), nn.functional.softmax(tl / T, 1), reduction="batchmean") * T * T
                loss = alpha * kd + (1 - alpha) * loss
            opt.zero_grad(); loss.backward(); opt.step()
    return model.eval()
def acc(model_or_fn, X, y):
    with torch.no_grad():
        out = model_or_fn(torch.from_numpy(X)) if isinstance(model_or_fn, nn.Module) else model_or_fn(X)
    out = out.numpy() if torch.is_tensor(out) else out
    return float((out.argmax(1) == y).mean())
def base_model():
    """Train once (seed 0) and cache; later days reuse the same fp32 weights."""
    Xtr, Xte, ytr, yte = data(); m = mlp(); f = P("w23_mlp.pt")
    if os.path.exists(f): m.load_state_dict(torch.load(f)); return m.eval()
    train(m, Xtr, ytr); torch.save(m.state_dict(), f); return m
def export_onnx(model, path, opset=17):
    torch.onnx.export(model, (torch.zeros(1, 64),), path, input_names=["x"], output_names=["logits"],
                      dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=opset, dynamo=False)
    return path
def bench(fn, reps=200, warm=20):
    for _ in range(warm): fn()
    ts = []
    for _ in range(reps): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e6  # microseconds
def rss_mb():
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith("VmRSS"): return int(l.split()[1]) / 1024
