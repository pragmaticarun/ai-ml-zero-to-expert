"""Day 40: bagging by hand, random forests, OOB score, number of trees, decorrelation (adult income)."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
from common import load_adult
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
X, y = load_adult(); cat = X.select_dtypes("category").columns
X = X.copy(); X[cat] = OrdinalEncoder(encoded_missing_value=-1).fit_transform(X[cat]); X = X.to_numpy(float); y = y.to_numpy()
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
print(f"adult: {X.shape}, positive rate {y.mean():.4f}")
rng = np.random.default_rng(40); one = DecisionTreeClassifier(random_state=0).fit(Xtr, ytr)
print(f"single full tree: train {one.score(Xtr, ytr):.4f} test {one.score(Xte, yte):.4f}")
votes = np.zeros(len(yte)); P = []
for b in range(25):
    i = rng.integers(0, len(ytr), len(ytr)); t = DecisionTreeClassifier(random_state=b).fit(Xtr[i], ytr[i]); p = t.predict(Xte); P.append(p); votes += p
    if b + 1 in (1, 5, 25): print(f"hand-bagged {b+1:2d} trees: test acc {((votes / (b + 1)) > 0.5).astype(int).__eq__(yte).mean():.4f}")
print("unique rows in one bootstrap sample:", round(len(np.unique(rng.integers(0, len(ytr), len(ytr)))) / len(ytr), 4), "(theory 1-1/e = 0.6321)")
P = np.array(P); print(f"mean pairwise agreement between bagged trees: {np.mean([(P[a] == P[b]).mean() for a in range(10) for b in range(a+1, 10)]):.4f}")
for mf in [None, "sqrt"]:
    t = time.perf_counter(); rf = RandomForestClassifier(200, max_features=mf, oob_score=True, n_jobs=-1, random_state=0).fit(Xtr, ytr); dt = time.perf_counter() - t
    Q = np.array([e.predict(Xte) for e in rf.estimators_[:10]]); agr = np.mean([(Q[a] == Q[b]).mean() for a in range(10) for b in range(a+1, 10)])
    print(f"RF max_features={mf}: OOB {rf.oob_score_:.4f} test {rf.score(Xte, yte):.4f} tree agreement {agr:.4f} ({dt:.1f}s)")
print("n_trees  test_acc")
for n in [1, 5, 20, 100, 300]:
    print(f"{n:7d}  {RandomForestClassifier(n, n_jobs=-1, random_state=0, min_samples_leaf=1).fit(Xtr, ytr).score(Xte, yte):.4f}")
