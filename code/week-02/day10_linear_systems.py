"""Day 10: solving Ax=b, why not to invert, least squares = 'best fit' when no exact answer."""
import numpy as np
rng = np.random.default_rng(10)
A = np.array([[2., 1.], [1., 3.]]); b = np.array([5., 10.])
x = np.linalg.solve(A, b); print("solve:", x, "check A@x:", A @ x)
# Overdetermined: 7 houses, 2 unknowns (w, b) -> no exact solution, use least squares
sizes = np.array([6, 8, 10, 12, 15, 18, 20.]); prices = np.array([32, 41, 50, 58, 74, 88, 97.])
Xd = np.column_stack([sizes, np.ones_like(sizes)])
theta, res, rank, sv = np.linalg.lstsq(Xd, prices, rcond=None); print("lstsq [w,b]:", theta.round(4))
theta_ne = np.linalg.solve(Xd.T @ Xd, Xd.T @ prices); print("normal equation [w,b]:", theta_ne.round(4))
# Ill-conditioning: nearly-collinear columns
H = np.array([[1, 1], [1, 1 + 1e-10]]); print("cond(H)=%.2e" % np.linalg.cond(H))
n = 12; Hil = 1 / (np.arange(n)[:, None] + np.arange(n)[None, :] + 1); xt = np.ones(n); bh = Hil @ xt
print("Hilbert(12) cond=%.2e" % np.linalg.cond(Hil))
print("err solve:", np.abs(np.linalg.solve(Hil, bh) - xt).max(), " err inv@b:", np.abs(np.linalg.inv(Hil) @ bh - xt).max())
