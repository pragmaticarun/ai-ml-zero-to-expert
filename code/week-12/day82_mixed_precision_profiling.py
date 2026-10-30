"""Day 82: number formats, bf16 autocast on CPU, loss scaling, torch.profiler, and a Rust/PyO3 batch-collate kernel.
CPU-only box: speedups from fp16/bf16 seen here are NOT representative of GPUs with tensor cores."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import numpy as np, torch, torch.nn as nn
from torch.profiler import profile, ProfilerActivity
torch.manual_seed(0)
for dt in (torch.float32, torch.float16, torch.bfloat16):
    fi = torch.finfo(dt); print(f"{str(dt):15s} bits {fi.bits:2d} max {fi.max:.3e} smallest normal {fi.tiny:.3e} eps {fi.eps:.3e}")
print("fp16: 70000 ->", torch.tensor(70000.0).half().item(), "| bf16: 70000 ->", torch.tensor(70000.0).bfloat16().item(), "| 1+1e-3 in bf16 ->", (torch.tensor(1.0, dtype=torch.bfloat16) + 1e-3).item())
g = torch.tensor(1e-8); print(f"a gradient of 1e-8 in fp16: {g.half().item()} ; scaled by 65536 first: {(g * 65536).half().item():.4e} -> unscaled {(g * 65536).half().float().item() / 65536:.3e}")
def bench(f, n=20):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e3
a, b = torch.randn(1024, 1024), torch.randn(1024, 1024)
print(f"1024x1024 matmul: fp32 {bench(lambda: a @ b):.2f} ms, bf16 {bench(lambda: a.bfloat16() @ b.bfloat16()):.2f} ms (incl. casts)")
ref = a.double() @ b.double(); print(f"relative error vs float64: fp32 {((a @ b).double() - ref).norm() / ref.norm():.1e}, bf16 {((a.bfloat16() @ b.bfloat16()).double() - ref).norm() / ref.norm():.1e}")
from _mnist import tensors, evaluate
Xtr, ytr, Xte, yte = tensors(n_train=10000); Xtr, Xte = Xtr.view(-1, 1, 28, 28), Xte.view(-1, 1, 28, 28)
def cnn(): return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(32 * 49, 10))
for mode in ("fp32", "bf16 autocast"):
    torch.manual_seed(0); m = cnn(); o = torch.optim.Adam(m.parameters(), 1e-3); t = time.perf_counter()
    for ep in range(2):
        for i in range(0, len(ytr), 128):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mode != "fp32"):
                out = m(Xtr[i:i+128]); l = nn.functional.cross_entropy(out.float(), ytr[i:i+128])
            o.zero_grad(); l.backward(); o.step()
    m.eval()
    with torch.no_grad(): acc = (m(Xte).argmax(1) == yte).float().mean().item()
    print(f"{mode:14s}: 2 epochs {time.perf_counter()-t:.1f} s, test acc {acc:.4f}, weights dtype {next(m.parameters()).dtype}, logits dtype under autocast {out.dtype}")
m = cnn(); o = torch.optim.SGD(m.parameters(), 0.01); xb, yb = Xtr[:256], ytr[:256]
with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
    for _ in range(5): l = nn.functional.cross_entropy(m(xb), yb); o.zero_grad(); l.backward(); o.step()
print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=8, max_name_column_width=40))
os.makedirs("out", exist_ok=True); prof.export_chrome_trace("out/day82_trace.json"); print("trace written to out/day82_trace.json (open in chrome://tracing or Perfetto)")
from common import load_cifar_small
import rust_loader
X, _, _ = load_cifar_small(); X = np.ascontiguousarray(X.reshape(len(X), -1)); rng = np.random.default_rng(0); idx = rng.integers(0, len(X), 512).astype(np.int64); mean, std = 0.47, 0.25
def np_c(): return (X[idx].astype(np.float32) / 255 - mean) / std
Xt = torch.from_numpy(X)
def torch_c(): return (Xt[torch.from_numpy(idx)].float() / 255 - mean) / std
r1, r2 = np_c(), rust_loader.collate_par(X, idx, mean, std); print(f"Rust vs NumPy collate max diff {np.abs(r1 - r2).max():.1e}, shape {r2.shape}")
for name, f in [("NumPy", np_c), ("torch", torch_c), ("Rust serial", lambda: rust_loader.collate(X, idx, mean, std)), ("Rust rayon", lambda: rust_loader.collate_par(X, idx, mean, std))]:
    print(f"collate 512 CIFAR images with {name:11s}: {bench(f, 50):.2f} ms")
try: rust_loader.collate(X, np.array([len(X)], dtype=np.int64), mean, std)
except IndexError as e: print("bad index raises IndexError:", e)
