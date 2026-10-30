"""Day 39: a decision tree from scratch (Gini/entropy, recursive best split) vs sklearn; depth sweep."""
import numpy as np, time
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
def gini(y): p = np.bincount(y, minlength=2) / len(y); return 1 - (p**2).sum()
def entropy(y): p = np.bincount(y, minlength=2) / len(y); p = p[p > 0]; return -(p * np.log2(p)).sum()
print("worked example, parent [6 pos, 4 neg]: gini", round(gini(np.array([1]*6+[0]*4)), 4), "entropy", round(entropy(np.array([1]*6+[0]*4)), 4))
L, R = np.array([1]*5+[0]*1), np.array([1]*1+[0]*3)
print("split -> L[5,1] R[1,3]: weighted gini", round(0.6*gini(L)+0.4*gini(R), 4), " info gain", round(entropy(np.array([1]*6+[0]*4)) - 0.6*entropy(L) - 0.4*entropy(R), 4))
def best_split(X, y, crit):
    best = (None, None, crit(y)); m = len(y)
    for j in range(X.shape[1]):
        o = np.argsort(X[:, j]); xs, ys = X[o, j], y[o]
        for i in range(1, m):
            if xs[i] == xs[i - 1]: continue
            s = (i * crit(ys[:i]) + (m - i) * crit(ys[i:])) / m
            if s < best[2]: best = (j, (xs[i] + xs[i - 1]) / 2, s)
    return best
def build(X, y, depth, crit, min_leaf=1):
    if depth == 0 or len(np.unique(y)) == 1 or len(y) < 2 * min_leaf: return int(np.bincount(y).argmax())
    j, t, _ = best_split(X, y, crit)
    if j is None: return int(np.bincount(y).argmax())
    l = X[:, j] <= t; return (j, t, build(X[l], y[l], depth - 1, crit), build(X[~l], y[~l], depth - 1, crit))
def pred1(node, x):
    while isinstance(node, tuple): node = node[2] if x[node[0]] <= node[1] else node[3]
    return node
X, y = load_breast_cancer(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
for d in [1, 3]:
    t = time.perf_counter(); tree = build(Xtr, ytr, d, gini); ts = time.perf_counter() - t
    sk = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr)
    acc = np.mean([pred1(tree, x) == t_ for x, t_ in zip(Xte, yte)])
    print(f"depth {d}: scratch test acc {acc:.4f} ({ts:.2f}s)  sklearn {sk.score(Xte, yte):.4f}  root split scratch=(feat {tree[0]}, thr {tree[1]:.3f}) sklearn=(feat {sk.tree_.feature[0]}, thr {sk.tree_.threshold[0]:.3f})")
print(export_text(DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr), feature_names=list(load_breast_cancer().feature_names)))
print("depth  train_acc  test_acc  leaves")
for d in [1, 2, 3, 5, 8, None]:
    s = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xtr, ytr); print(f"{str(d):5s}  {s.score(Xtr, ytr):9.4f}  {s.score(Xte, yte):8.4f}  {s.get_n_leaves()}")
