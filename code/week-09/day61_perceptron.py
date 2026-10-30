"""Day 61: the perceptron rule, logistic neuron, and the XOR limit of a single linear unit."""
import numpy as np
from sklearn.linear_model import Perceptron, LogisticRegression
rng = np.random.default_rng(61)
X = rng.normal(size=(200, 2)); y = (X @ [2., -1] + 0.5 > 0).astype(int)
w = np.zeros(2); b = 0.0; mistakes = []
for ep in range(20):
    k = 0
    for xi, yi in zip(X, y):
        yh = int(xi @ w + b > 0)
        if yh != yi: w += (yi - yh) * xi; b += (yi - yh); k += 1
    mistakes.append(k)
    if k == 0: break
print("perceptron mistakes per epoch:", mistakes, f"-> converged in {len(mistakes)} epochs; w/|w| {np.round(w / np.linalg.norm(w), 3).tolist()} vs true {np.round(np.array([2, -1]) / np.sqrt(5), 3).tolist()}")
print(f"sklearn Perceptron train acc {Perceptron(random_state=0).fit(X, y).score(X, y):.3f}")
XOR = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float); yx = np.array([0, 1, 1, 0])
w = np.zeros(2); b = 0.0; errs = []
for ep in range(100):
    k = 0
    for xi, yi in zip(XOR, yx):
        yh = int(xi @ w + b > 0)
        if yh != yi: w += (yi - yh) * xi; b += (yi - yh); k += 1
    errs.append(k)
print("XOR perceptron mistakes, epochs 1-5 and 96-100:", errs[:5], errs[-5:], "(never reaches 0)")
print("XOR logistic regression accuracy:", LogisticRegression().fit(XOR, yx).score(XOR, yx))
F = np.c_[XOR, XOR[:, 0] * XOR[:, 1]]; print("XOR + product feature x1*x2, logistic acc:", LogisticRegression(C=100).fit(F, yx).score(F, yx))
z = np.linspace(-4, 4, 9); s = 1 / (1 + np.exp(-z)); print("sigmoid'(z)=s(1-s) at z=-4..4:", (s * (1 - s)).round(3).tolist(), "max 0.25 at z=0")
