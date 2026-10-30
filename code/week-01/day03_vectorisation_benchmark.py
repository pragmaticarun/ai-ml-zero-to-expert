"""Day 3: vectorisation. Python loop vs NumPy for dot product and pairwise distances.
Timings are machine-dependent: run it yourself and record YOUR numbers."""
import time, numpy as np
rng = np.random.default_rng(1)
def timeit(f, reps=3):
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter(); r = f(); best = min(best, time.perf_counter() - t)
    return best, r
n = 1_000_000
x, y = rng.random(n), rng.random(n); xl, yl = x.tolist(), y.tolist()
t_loop, r1 = timeit(lambda: sum(a * b for a, b in zip(xl, yl)))
t_np, r2 = timeit(lambda: float(np.dot(x, y)))
print(f"dot n={n}: python loop {t_loop*1e3:.1f} ms | np.dot {t_np*1e3:.3f} ms | speedup x{t_loop/t_np:.0f} | same? {abs(r1-r2)<1e-6*n}")
# Pairwise squared distances: 3 ways
P = rng.random((400, 8)); Q = rng.random((300, 8))
def d_loop():
    return [[sum((p[k]-q[k])**2 for k in range(8)) for q in Ql] for p in Pl]
Pl, Ql = P.tolist(), Q.tolist()
def d_bcast(): return ((P[:, None, :] - Q[None, :, :]) ** 2).sum(-1)
def d_gemm(): return (P**2).sum(1)[:, None] + (Q**2).sum(1)[None, :] - 2 * P @ Q.T
for name, f in [("loops", d_loop), ("broadcast", d_bcast), ("matmul trick", d_gemm)]:
    t, r = timeit(f, reps=2); print(f"pairwise {name:12s}: {t*1e3:8.2f} ms")
print("broadcast vs matmul max abs diff:", np.abs(d_bcast() - d_gemm()).max())
# Memory layout matters (like C++ row-major): sum over rows vs columns of a C-ordered array
A = rng.random((3000, 3000))
t_r, _ = timeit(lambda: A.sum(axis=1)); t_c, _ = timeit(lambda: np.asfortranarray(A).sum(axis=1))
print(f"row-sum C-order {t_r*1e3:.2f} ms vs F-order (incl. conversion) {t_c*1e3:.2f} ms")
