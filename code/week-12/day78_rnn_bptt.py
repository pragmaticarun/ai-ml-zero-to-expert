"""Day 78: a vanilla RNN in NumPy with hand-written backprop through time (BPTT), gradient check, and the vanishing-gradient effect."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
rng = np.random.default_rng(0)
def init(nx, nh, ny, scale=0.5):
    return {"Wx": rng.normal(size=(nx, nh)) * scale / np.sqrt(nx), "Wh": rng.normal(size=(nh, nh)) * scale / np.sqrt(nh), "b": np.zeros(nh), "Wy": rng.normal(size=(nh, ny)) / np.sqrt(nh), "by": np.zeros(ny)}
def forward(p, xs):                     # xs (T, nx); predict a scalar from the last hidden state (regression, MSE)
    hs = [np.zeros(p["Wh"].shape[0])]
    for x in xs: hs.append(np.tanh(x @ p["Wx"] + hs[-1] @ p["Wh"] + p["b"]))
    return hs, hs[-1] @ p["Wy"] + p["by"]
def loss_grad(p, xs, y):
    hs, out = forward(p, xs); d = out - y; L = 0.5 * (d**2).sum()
    g = {k: np.zeros_like(v) for k, v in p.items()}; g["Wy"] = np.outer(hs[-1], d); g["by"] = d
    dh = p["Wy"] @ d; dh_norms = []
    for t in range(len(xs), 0, -1):     # backprop through time
        dz = dh * (1 - hs[t]**2); g["Wx"] += np.outer(xs[t-1], dz); g["Wh"] += np.outer(hs[t-1], dz); g["b"] += dz
        dh = p["Wh"] @ dz; dh_norms.append(np.linalg.norm(dh))
    return L, g, dh_norms[::-1]
p = init(3, 8, 1); xs = rng.normal(size=(6, 3)); y = np.array([0.7]); L, g, _ = loss_grad(p, xs, y)
worst = 0
for k in p:
    num = np.zeros_like(p[k]); it = np.nditer(p[k], flags=["multi_index"])
    for _ in it:
        i = it.multi_index; o = p[k][i]; p[k][i] = o + 1e-5; a = loss_grad(p, xs, y)[0]; p[k][i] = o - 1e-5; b = loss_grad(p, xs, y)[0]; p[k][i] = o; num[i] = (a - b) / 2e-5
    rel = np.linalg.norm(num - g[k]) / (np.linalg.norm(num) + np.linalg.norm(g[k]) + 1e-12); worst = max(worst, rel); print(f"BPTT grad check {k:2s}: relative error {rel:.2e}")
print("gradient check", "PASSED" if worst < 1e-6 else "FAILED")
# how much gradient reaches early time steps? |dL/dh_t| for T = 50 at three recurrent scales
for scale in (0.5, 1.0, 2.0):
    p = init(3, 32, 1, scale); xs = rng.normal(size=(50, 3)); _, _, n = loss_grad(p, xs, np.array([1.0]))
    print(f"Wh scale {scale}: |dL/dh| at t=49 {n[-1]:.2e}, t=25 {n[25]:.2e}, t=0 {n[0]:.2e}  (spectral radius of Wh {np.abs(np.linalg.eigvals(p['Wh'])).max():.2f})")
# train on a task needing memory: output = first input's feature 0, after T steps of noise
def task(T, n=200):
    X = rng.normal(size=(n, T, 3)); return X, X[:, 0, 0]
for T in (3, 20):
    p = init(3, 16, 1, 1.0); X, Y = task(T); lr = 0.02
    for ep in range(60):
        for i in range(len(Y)):
            _, g, _ = loss_grad(p, X[i], Y[i:i+1])
            for k in p: p[k] -= lr * np.clip(g[k], -1, 1)
    Xt, Yt = task(T, 200); mse = np.mean([(forward(p, Xt[i])[1][0] - Yt[i])**2 for i in range(200)])
    print(f"remember x_0 after T={T:2d} steps: test MSE {mse:.3f} (predicting 0 gives about {np.mean(Yt**2):.3f})")
