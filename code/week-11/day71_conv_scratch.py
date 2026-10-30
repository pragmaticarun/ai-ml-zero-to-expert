"""Day 71: convolution from scratch (loops and im2col), checked against torch; conv output-size arithmetic and parameter counts."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn
rng = np.random.default_rng(0)
def conv_loops(x, w, b, stride=1, pad=0):   # x (C,H,W), w (K,C,kh,kw)
    x = np.pad(x, ((0, 0), (pad, pad), (pad, pad))); K, C, kh, kw = w.shape; H = (x.shape[1] - kh) // stride + 1; W = (x.shape[2] - kw) // stride + 1
    out = np.zeros((K, H, W))
    for k in range(K):
        for i in range(H):
            for j in range(W): out[k, i, j] = (x[:, i*stride:i*stride+kh, j*stride:j*stride+kw] * w[k]).sum() + b[k]
    return out
def im2col(x, kh, kw, stride=1, pad=0):     # x (N,C,H,W) -> (N, C*kh*kw, L)
    x = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad))); N, C, H, W = x.shape; Ho = (H - kh) // stride + 1; Wo = (W - kw) // stride + 1
    cols = np.empty((N, C, kh, kw, Ho, Wo))
    for a in range(kh):
        for c in range(kw): cols[:, :, a, c] = x[:, :, a:a+stride*Ho:stride, c:c+stride*Wo:stride]
    return cols.reshape(N, C*kh*kw, Ho*Wo), Ho, Wo
def conv_im2col(x, w, b, stride=1, pad=0):
    cols, Ho, Wo = im2col(x, w.shape[2], w.shape[3], stride, pad); out = np.einsum("kf,nfl->nkl", w.reshape(w.shape[0], -1), cols) + b[None, :, None]
    return out.reshape(x.shape[0], w.shape[0], Ho, Wo)
# hand example: vertical-edge kernel on a step image
img = np.zeros((1, 5, 5)); img[:, :, 3:] = 1; k = np.array([[[[-1, 0, 1]] * 3]], dtype=float)
print("edge detector output (rows identical):", conv_loops(img, k, np.zeros(1))[0, 0].tolist())
x = rng.normal(size=(8, 3, 32, 32)); w = rng.normal(size=(16, 3, 3, 3)); b = rng.normal(size=16)
ref = nn.functional.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), stride=1, padding=1).numpy()
t = time.perf_counter(); lo = np.stack([conv_loops(xi, w, b, 1, 1) for xi in x]); tl = time.perf_counter() - t
t = time.perf_counter(); ic = conv_im2col(x, w, b, 1, 1); ti = time.perf_counter() - t
t = time.perf_counter(); [nn.functional.conv2d(torch.tensor(x), torch.tensor(w), torch.tensor(b), padding=1) for _ in range(10)]; tt = (time.perf_counter() - t) / 10
print(f"max |loops - torch| {np.abs(lo-ref).max():.1e}, max |im2col - torch| {np.abs(ic-ref).max():.1e}")
print(f"timing for 8x3x32x32 * 16x3x3x3: loops {tl*1000:.0f} ms, im2col {ti*1000:.1f} ms, torch {tt*1000:.2f} ms")
s2 = conv_im2col(x, w, b, stride=2, pad=1); print("stride 2, pad 1 output shape:", s2.shape, "| torch:", tuple(nn.functional.conv2d(torch.tensor(x), torch.tensor(w), stride=2, padding=1).shape))
out = lambda n, k, s, p: (n + 2*p - k) // s + 1
for n, k_, s, p in [(32, 3, 1, 0), (32, 3, 1, 1), (32, 5, 1, 2), (32, 3, 2, 1), (28, 5, 1, 0), (224, 7, 2, 3)]: print(f"out size n={n} k={k_} s={s} p={p}: {out(n, k_, s, p)}")
conv = nn.Conv2d(3, 16, 3); dense = nn.Linear(3*32*32, 16*30*30)
print("params Conv2d(3,16,3):", sum(q.numel() for q in conv.parameters()), "| dense layer with the same input/output sizes:", f"{sum(q.numel() for q in dense.parameters()):,}")
rf = 1; jump = 1
for k_, s in [(3, 1), (3, 1), (2, 2), (3, 1), (3, 1), (2, 2)]: rf += (k_ - 1) * jump; jump *= s
print("receptive field after conv3,conv3,pool2,conv3,conv3,pool2:", rf)
