"""Day 34: softmax regression from scratch on digits vs sklearn multinomial logistic regression."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
def softmax(Z): Z = Z - Z.max(1, keepdims=True); E = np.exp(Z); return E / E.sum(1, keepdims=True)
z = np.array([[1000., 1001., 1002.]])
with np.errstate(over="ignore", invalid="ignore"): naive = np.exp(z) / np.exp(z).sum()
print("naive softmax of [1000,1001,1002]:", naive, " stable:", softmax(z).round(4))
X, y = load_digits(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
m, n = Xtr.shape; K = 10; Y = np.eye(K)[ytr]; W = np.zeros((n, K)); b = np.zeros(K); lam = 1e-2; lr = 0.5
t = time.perf_counter()
for it in range(500):
    P = softmax(Xtr @ W + b); G = P - Y
    W -= lr * (Xtr.T @ G / m + lam * W); b -= lr * G.mean(0)
    if it in (0, 9, 99, 499): print(f"iter {it+1:3d} cross-entropy {-np.log(P[np.arange(m), ytr] + 1e-12).mean():.4f}")
ts = time.perf_counter() - t
acc = (np.argmax(Xte @ W + b, 1) == yte).mean()
t = time.perf_counter(); sk = LogisticRegression(C=1 / (lam * m), max_iter=2000).fit(Xtr, ytr); tk = time.perf_counter() - t
print(f"scratch softmax test acc {acc:.4f} ({ts*1000:.0f} ms)   sklearn {sk.score(Xte, yte):.4f} ({tk*1000:.0f} ms)")
print("row sums of probabilities all 1:", np.allclose(softmax(Xte @ W + b).sum(1), 1))
cm = confusion_matrix(yte, np.argmax(Xte @ W + b, 1)); off = cm - np.diag(np.diag(cm)); i, j = np.unravel_index(off.argmax(), off.shape)
print(f"most common confusion: true {i} predicted as {j} ({off[i, j]} times); total errors {off.sum()} of {len(yte)}")
ovr_acc = np.mean(np.argmax(np.column_stack([LogisticRegression(max_iter=2000).fit(Xtr, ytr == k).decision_function(Xte) for k in range(K)]), 1) == yte)
print(f"one-vs-rest (10 binary logregs) test acc {ovr_acc:.4f}")
