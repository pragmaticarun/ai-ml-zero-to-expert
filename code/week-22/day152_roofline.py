"""Day 152: memory, bandwidth and roofline thinking on this CPU. Measures: memory bandwidth (copy/sum of a 256 MB array),
matmul throughput vs size, dtype and threads (torch), arithmetic intensity of common ops, access-pattern/layout effects,
temporaries vs in-place ('fusion by hand'), and a Python loop vs vectorised code. Best-of-N timings, CPU only."""
import time, numpy as np, torch
def best(fn, n=5):
    fn(); ts = []
    for _ in range(n): t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return min(ts)
torch.set_num_threads(1)
N = 32 * 1024 * 1024; a = np.ones(N, np.float64); b = np.empty_like(a)   # 256 MB each
t = best(lambda: np.copyto(b, a)); bw_copy = 2 * a.nbytes / t / 1e9
t = best(lambda: a.sum()); bw_sum = a.nbytes / t / 1e9
t = best(lambda: np.add(a, a, out=b)); bw_add = 3 * a.nbytes / t / 1e9
print(f"(1) bandwidth, 256 MB float64, 1 thread: copy {bw_copy:.1f} GB/s (read+write), sum {bw_sum:.1f} GB/s, a+a->b {bw_add:.1f} GB/s ({N/t/1e9:.2f} GFLOP/s)")
print("(2) matmul (torch, square n x n), GFLOP/s = 2n^3/time:")
peak = {}
for dt in (torch.float64, torch.float32, torch.bfloat16):
    for th in (1, 8):
        torch.set_num_threads(th); row = []
        for n in (64, 256, 1024, 2048):
            x = torch.randn(n, n).to(dt); y = torch.randn(n, n).to(dt); t = best(lambda: x @ y, 3); row.append(2 * n ** 3 / t / 1e9)
        peak[(dt, th)] = max(row); print(f"    {str(dt).split('.')[1]:9s} {th} thr: " + ", ".join(f"n={n}: {g:7.1f}" for n, g in zip((64, 256, 1024, 2048), row)))
torch.set_num_threads(1); P = peak[(torch.float32, 1)]; B = bw_copy
print(f"(3) roofline, float32, 1 thread: measured peak {P:.0f} GFLOP/s, bandwidth {B:.1f} GB/s -> ridge point {P/B:.1f} FLOP/byte")
for name, ai in (("vector add (1 flop / 12 bytes)", 1 / 12), ("dot product (2 flops / 8 bytes)", 2 / 8), ("matvec n=4096 (~2 flops / 4 bytes)", 0.5),
                 ("matmul n=1024 (2n^3 / 12n^2 bytes)", 2 * 1024 / 12)):
    print(f"    {name:38s}: intensity {ai:7.2f} -> bound {'memory' if ai < P/B else 'compute'}; attainable <= {min(P, ai*B):6.1f} GFLOP/s")
x = torch.randn(4096, 4096); v = torch.randn(4096); t = best(lambda: x @ v)
print(f"    measured matvec n=4096 float32: {2*4096*4096/t/1e9:.1f} GFLOP/s, {x.numel()*4/t/1e9:.1f} GB/s of matrix read (above the DRAM roof: the 64 MB matrix fits in the 320 MB L3 cache)")
M = np.random.rand(4096, 4096)
tr, tc = best(lambda: M.sum(axis=1)), best(lambda: M.sum(axis=0))
tl = best(lambda: [M[i, :].sum() for i in range(0, 4096, 8)]); tk = best(lambda: [M[:, j].sum() for j in range(0, 4096, 8)])
print(f"(4) layout, 4096x4096 float64 C-order: sum(axis=1) {tr*1e3:.1f} ms, sum(axis=0) {tc*1e3:.1f} ms; "
      f"512 row slices {tl*1e3:.1f} ms vs 512 column slices {tk*1e3:.1f} ms ({tk/tl:.1f}x slower, strided)")
Mt = M.T; tcopy = best(lambda: np.ascontiguousarray(Mt), 3); print(f"    making the transpose contiguous costs {tcopy*1e3:.1f} ms ({2*M.nbytes/tcopy/1e9:.1f} GB/s)")
x = np.random.rand(N // 4); y = np.empty_like(x)
t1 = best(lambda: 2.0 * x * x + 3.0 * x + 1.0)
def inplace():
    np.multiply(x, 2.0, out=y); np.add(y, 3.0, out=y); np.multiply(y, x, out=y); np.add(y, 1.0, out=y)
t2 = best(inplace); print(f"(5) 2x^2+3x+1 on 8M float64: naive (temporaries) {t1*1e3:.1f} ms; in-place Horner with out= {t2*1e3:.1f} ms ({t1/t2:.1f}x); "
                           f"max diff {np.abs((2*x*x+3*x+1)-y).max():.1e}")
xs = x[:1_000_000].tolist(); ys = xs[::-1]
tp = best(lambda: sum(p * q for p, q in zip(xs, ys)), 3); xn, yn = np.array(xs), np.array(ys); tn = best(lambda: xn @ yn)
print(f"(6) dot product of 1M floats: Python loop {tp*1e3:.1f} ms, numpy {tn*1e3:.3f} ms ({tp/tn:.0f}x)")
