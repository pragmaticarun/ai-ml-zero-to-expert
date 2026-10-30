"""Day 16: partial derivatives, gradient vectors, gradient checking."""
import numpy as np
def f(v): x, y = v; return x**2 + 3*y**2 + x*y
def grad_f(v): x, y = v; return np.array([2*x + y, 6*y + x])
def num_grad(f, v, h=1e-6):
    g = np.zeros_like(v)
    for i in range(len(v)):
        e = np.zeros_like(v); e[i] = h; g[i] = (f(v+e) - f(v-e)) / (2*h)
    return g
v = np.array([1.0, -2.0]); ga, gn = grad_f(v), num_grad(f, v)
print("analytic", ga, "numeric", gn.round(6), "rel err", np.linalg.norm(ga-gn)/np.linalg.norm(ga+gn))
# Gradient of MSE for linear regression, vectorised: J = 1/(2m) ||Xw - y||^2  => grad = X^T(Xw - y)/m
rng = np.random.default_rng(16); X = rng.normal(size=(50, 3)); y = rng.normal(size=50); w = rng.normal(size=3)
J = lambda w: ((X@w - y)**2).sum() / (2*len(y)); g = X.T @ (X@w - y) / len(y)
print("MSE grad check rel err:", np.linalg.norm(g - num_grad(J, w)) / np.linalg.norm(g + num_grad(J, w)))
# gradient points uphill: a small step against it lowers f
print("f(v)=", f(v), " f(v - 0.1*grad)=", round(f(v - 0.1*ga), 4))
