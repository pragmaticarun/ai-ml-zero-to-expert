"""Day 9: matrix multiplication 3 ways + timing; matmul as 'many dot products'."""
import time, numpy as np
rng = np.random.default_rng(9)
def matmul_naive(A, B):
    n, k = len(A), len(A[0]); m = len(B[0]); C = [[0.0]*m for _ in range(n)]
    for i in range(n):
        for p in range(k):
            a = A[i][p]; Bp = B[p]; Ci = C[i]
            for j in range(m): Ci[j] += a * Bp[j]     # i-p-j order: cache-friendly, like in C++
    return C
for n in [64, 128]:
    A, B = rng.random((n, n)), rng.random((n, n))
    t = time.perf_counter(); C1 = matmul_naive(A.tolist(), B.tolist()); t1 = time.perf_counter() - t
    t = time.perf_counter(); C2 = A @ B; t2 = time.perf_counter() - t
    flops = 2 * n**3
    print(f"n={n}: pure python {t1*1e3:.1f} ms ({flops/t1/1e6:.1f} MFLOP/s) | numpy {t2*1e3:.3f} ms | max diff {np.abs(np.array(C1)-C2).max():.2e}")
n = 1024; A, B = rng.random((n, n)), rng.random((n, n)); A @ B
t = time.perf_counter(); A @ B; t2 = time.perf_counter() - t
print(f"n=1024 numpy (BLAS): {t2*1e3:.1f} ms = {2*n**3/t2/1e9:.1f} GFLOP/s")
# Shapes: (batch, features) @ (features, outputs)
X = rng.random((5, 3)); W = rng.random((3, 2)); print("X@W shape:", (X @ W).shape)
print("non-commutative: AB==BA?", np.allclose(A[:3, :3] @ B[:3, :3], B[:3, :3] @ A[:3, :3]))
