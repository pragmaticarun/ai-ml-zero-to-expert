"""Day 11: eigenvectors as 'directions a matrix only stretches'; power iteration."""
import numpy as np
A = np.array([[2., 1.], [1., 2.]])
vals, vecs = np.linalg.eigh(A); print("eigenvalues:", vals, "\neigenvectors (columns):\n", vecs.round(4))
for i in range(2): print(f"A@v{i} =", (A @ vecs[:, i]).round(4), " lambda*v =", (vals[i]*vecs[:, i]).round(4))
v = np.array([1., 0.])
for k in range(20): v = A @ v; v /= np.linalg.norm(v)
print("power iteration ->", v.round(4), " Rayleigh quotient:", round(v @ A @ v, 6))
# Covariance eigenvectors = principal directions of a data cloud
rng = np.random.default_rng(11); X = rng.normal(size=(2000, 2)) @ np.array([[3, 0], [0, .5]]) @ np.array([[.8, .6], [-.6, .8]])
C = np.cov(X.T); ev, evec = np.linalg.eigh(C); print("cov eigenvalues:", ev.round(3), " top direction:", evec[:, -1].round(3))
# Repeated application: A^k blows up or shrinks depending on |lambda| (think RNN gradients later!)
M = np.array([[0.9, 0], [0, 1.1]]); print("||M^50 x||:", np.linalg.norm(np.linalg.matrix_power(M, 50) @ np.ones(2)).round(2))
