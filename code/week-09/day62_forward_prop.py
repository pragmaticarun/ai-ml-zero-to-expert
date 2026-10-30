"""Day 62: forward propagation: shapes, activations, a hand-built 2-layer net that solves XOR, vectorised batch forward."""
import numpy as np
sig = lambda z: 1 / (1 + np.exp(-z)); relu = lambda z: np.maximum(0, z)
XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
W1 = np.array([[20., 20.], [20., 20.]]); b1 = np.array([-10., -30.])      # unit1 ~ OR, unit2 ~ AND   (W1: in x hidden)
W2 = np.array([[20.], [-20.]]); b2 = np.array([-10.])                    # output ~ OR AND NOT AND = XOR
A1 = sig(XOR @ W1 + b1); A2 = sig(A1 @ W2 + b2)
print("hidden activations (OR, AND):"); print(A1.round(4)); print("output:", A2.ravel().round(4).tolist())
rng = np.random.default_rng(62); m, n, h, k = 5, 64, 32, 10
X = rng.normal(size=(m, n)); W1 = rng.normal(size=(n, h)) * np.sqrt(2 / n); b1 = np.zeros(h); W2 = rng.normal(size=(h, k)) * np.sqrt(1 / h); b2 = np.zeros(k)
Z1 = X @ W1 + b1; A1 = relu(Z1); Z2 = A1 @ W2 + b2; P = np.exp(Z2 - Z2.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
print("shapes X", X.shape, "W1", W1.shape, "Z1", Z1.shape, "W2", W2.shape, "Z2", Z2.shape, "rows of P sum to 1:", np.allclose(P.sum(1), 1))
print("parameters:", W1.size + b1.size + W2.size + b2.size, "= 64*32+32+32*10+10")
print("share of ReLU units active:", round(float((Z1 > 0).mean()), 3))
loop = np.array([relu(x @ W1 + b1) @ W2 + b2 for x in X]); print("per-row loop == batched matmul:", np.allclose(loop, Z2))
for name, scale in [("std 1", 1.0), ("He sqrt(2/n)", np.sqrt(2 / n)), ("std 0.01", 0.01)]:
    a = rng.normal(size=(1000, n))
    for _ in range(10): a = relu(a @ (rng.normal(size=(n, n)) * scale))
    print(f"init {name:13s}: activation std after 10 ReLU layers {a.std():.3e}")
