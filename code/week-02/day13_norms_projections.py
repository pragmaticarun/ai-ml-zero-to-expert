"""Day 13: projections, orthogonality, Gram-Schmidt / QR, and fitting as projection."""
import numpy as np
rng = np.random.default_rng(13)
A = rng.normal(size=(6, 2)); y = rng.normal(size=6)
P = A @ np.linalg.inv(A.T @ A) @ A.T; yhat = P @ y; r = y - yhat
print("P symmetric:", np.allclose(P, P.T), " P@P==P:", np.allclose(P @ P, P), " residual ⟂ columns:", (A.T @ r).round(10))
def gram_schmidt(A):
    Q = np.zeros_like(A)
    for j in range(A.shape[1]):
        v = A[:, j] - Q[:, :j] @ (Q[:, :j].T @ A[:, j]); Q[:, j] = v / np.linalg.norm(v)
    return Q
Q = gram_schmidt(A); print("Q^T Q =\n", (Q.T @ Q).round(10)); Qn, R = np.linalg.qr(A)
print("same span as numpy QR:", np.allclose(Q @ Q.T, Qn @ Qn.T))
x = np.array([1., 2, 3]); print("distances: L2", np.linalg.norm(x), "L1", np.linalg.norm(x, 1), "Linf", np.linalg.norm(x, np.inf))
