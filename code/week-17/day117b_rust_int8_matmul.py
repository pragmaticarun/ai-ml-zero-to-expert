"""Day 117b: a Rust/PyO3 int8 weight-only matmul kernel (rust_qmatmul/) vs torch fp32 and torch dequantise-then-matmul, on SmolLM2-135M's
down_proj shape (576 x 1536). Each timing runs in its own subprocess, because rayon's idle worker threads spin and slow down torch's threads
(a first version that timed everything in one process measured torch 400x slower than in isolation)."""
import subprocess, sys
SETUP = """
import time, numpy as np, torch, os; torch.set_num_threads(8); torch.manual_seed(0)
W = torch.randn(576, 1536) * 0.05; s = W.abs().amax(1) / 127; Wq = (W / s[:, None]).round().clamp(-128, 127).to(torch.int8); B = {B}
x = torch.randn(B, 1536); xn, wn, sn = x.numpy(), Wq.numpy(), s.numpy().astype(np.float32)
def bench(f, n=200):
    f(); t = time.perf_counter()
    for _ in range(n): f()
    return (time.perf_counter() - t) / n * 1e6
"""
CASES = {"torch fp32": "x @ W.T", "torch dequant+matmul": "x @ (Wq.float() * s[:, None]).T",
         "rust int8 1 thread": "rust_qmatmul.matmul_i8(xn, wn, sn, False)", "rust int8 rayon 8 threads": "rust_qmatmul.matmul_i8(xn, wn, sn, True)"}
chk = SETUP.format(B=4) + "import rust_qmatmul\ny = torch.from_numpy(rust_qmatmul.matmul_i8(xn, wn, sn, True)); print(f'max |rust - torch dequant| {(y - x @ (Wq.float()*s[:,None]).T).abs().max():.2e}; vs unquantised fp32 {(y - x @ W.T).abs().max():.2e}')"
print(subprocess.run([sys.executable, "-c", chk], capture_output=True, text=True, env={"RAYON_NUM_THREADS": "8", "PATH": ""}).stdout.strip())
for B in (1, 64):
    row = []
    for name, expr in CASES.items():
        code = SETUP.format(B=B) + ("import rust_qmatmul\n" if "rust" in name else "") + f"print(f'{{bench(lambda: {expr}):.1f}}')"
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"RAYON_NUM_THREADS": "8"}); row.append(f"{name} {r.stdout.strip() or r.stderr[-200:]} us")
    print(f"batch {B:2d}: " + " | ".join(row))
print(f"weight bytes: fp32 {576*1536*4:,} vs int8 {576*1536:,} + fp32 scales {576*4:,}")
