"""Day 153: Rust extensions with PyO3 (crate rust_kernels/). Builds the crate twice with maturin (portable x86-64 default, then
RUSTFLAGS='-C target-cpu=native', which lets the compiler use AVX-512/FMA on this CPU), and benchmarks in a fresh process each time:
dot product (naive single accumulator vs 16-accumulator SIMD-friendly vs rayon) and row softmax (3-pass vs online, serial vs rayon)
against numpy. Correctness checked against numpy first."""
import subprocess, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); CR = os.path.join(HERE, "rust_kernels")
BENCH = r'''
import time, numpy as np, rust_kernels as R
def best(f, n=7):
    f(); ts = []
    for _ in range(n): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
rng = np.random.default_rng(0)
for n in (1 << 16, 1 << 20, 1 << 24):
    a, b = rng.standard_normal(n, dtype=np.float32), rng.standard_normal(n, dtype=np.float32); ref = float(a.astype(np.float64) @ b)
    r = {k: best(f) for k, f in (("numpy", lambda: a @ b), ("naive", lambda: R.dot_naive(a, b)), ("simd", lambda: R.dot_simd(a, b)), ("rayon", lambda: R.dot_par(a, b)))}
    err = {k: abs(f(a, b) - ref) / abs(ref) for k, f in (("naive", R.dot_naive), ("simd", R.dot_simd), ("numpy", lambda x, y: float(x @ y)))}
    print(f"  dot n={n:>9,} ({8*n/2**20:.1f} MB): " + ", ".join(f"{k} {v*1e6:8.1f} us ({2*n/v/1e9:5.1f} GFLOP/s)" for k, v in r.items())
          + f"; rel.err vs float64: naive {err['naive']:.1e}, simd {err['simd']:.1e}, numpy {err['numpy']:.1e}")
x = rng.standard_normal((4096, 1024), dtype=np.float32) * 4
def np_soft(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)
ref = np_soft(x.astype(np.float64))
for name, f in (("numpy", lambda: np_soft(x)), ("rust 3-pass", lambda: R.softmax_rows(x)), ("rust online", lambda: R.softmax_rows(x, online=True)),
                ("rust 3-pass rayon", lambda: R.softmax_rows(x, parallel=True)), ("rust online rayon", lambda: R.softmax_rows(x, online=True, parallel=True))):
    t = best(f, 5); print(f"  softmax 4096x1024 {name:18s}: {t*1e3:6.2f} ms ({x.nbytes*2/t/1e9:5.1f} GB/s effective), max err {np.abs(f()-ref).max():.1e}")
'''
for label, flags in (("portable x86-64 (default)", ""), ("target-cpu=native", "-C target-cpu=native")):
    env = dict(os.environ, CARGO_NET_OFFLINE="true", RUSTFLAGS=flags); t = time.perf_counter()
    b = subprocess.run(["maturin", "develop", "--release", "-q"], cwd=CR, env=env, capture_output=True, text=True)
    print(f"== build {label}: exit {b.returncode}, {time.perf_counter()-t:.0f} s", flush=True)
    if b.returncode: print(b.stderr[-2000:]); sys.exit(1)
    r = subprocess.run([sys.executable, "-c", BENCH], capture_output=True, text=True, env=dict(os.environ, RAYON_NUM_THREADS="8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    print(r.stdout + r.stderr[-1500:], flush=True)
print("numpy pinned to 1 BLAS thread (OPENBLAS_NUM_THREADS=1); rayon used 8 threads. Final installed build: target-cpu=native.")
