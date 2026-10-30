"""Day 37: Gaussian NB from scratch vs sklearn; multinomial NB on tiny text; generative vs discriminative learning curves."""
import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
class GNB:
    def fit(self, X, y):
        self.c = np.unique(y); self.mu = np.array([X[y == k].mean(0) for k in self.c])
        self.var = np.array([X[y == k].var(0) for k in self.c]) + 1e-9 * X.var(0).max(); self.lp = np.log([np.mean(y == k) for k in self.c]); return self
    def joint(self, X):
        return self.lp - 0.5 * (np.log(2 * np.pi * self.var)[None] + (X[:, None, :] - self.mu[None]) ** 2 / self.var[None]).sum(2)
    def predict(self, X): return self.c[self.joint(X).argmax(1)]
X, y = load_wine(return_X_y=True); Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
g = GNB().fit(Xtr, ytr); s = GaussianNB().fit(Xtr, ytr)
print(f"wine GaussianNB scratch acc {(g.predict(Xte) == yte).mean():.4f}  sklearn {s.score(Xte, yte):.4f}  same preds {(g.predict(Xte) == s.predict(Xte)).all()}")
J = g.joint(Xte[:1]); print("log joint for first test row:", J.round(2), "-> posterior", np.exp(J - np.logaddexp.reduce(J, axis=1)[:, None]).round(4))
docs = ["cheap pills buy now", "buy cheap watches now", "win money now", "meeting at noon tomorrow", "lunch tomorrow at noon?", "project meeting notes attached"]
lab = [1, 1, 1, 0, 0, 0]
cv = CountVectorizer(); Xt = cv.fit_transform(docs); nb = MultinomialNB(alpha=1.0).fit(Xt, lab)
for q in ["buy now", "meeting tomorrow", "cheap meeting"]:
    print(f"P(spam | '{q}') = {nb.predict_proba(cv.transform([q]))[0, 1]:.3f}")
Xb, yb = load_breast_cancer(return_X_y=True); Xa, Xh, ya, yh = train_test_split(Xb, yb, test_size=0.4, random_state=0, stratify=yb)
print("train_size  GaussianNB  LogReg   (mean test acc over 30 random draws)")
rng = np.random.default_rng(37)
for m in [10, 20, 40, 80, 160, 340]:
    a = []; b = []
    for _ in range(30):
        while True:
            i = rng.choice(len(ya), m, replace=False)
            if len(np.unique(ya[i])) == 2: break
        a.append(GaussianNB().fit(Xa[i], ya[i]).score(Xh, yh)); b.append(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(Xa[i], ya[i]).score(Xh, yh))
    print(f"{m:10d}  {np.mean(a):10.4f}  {np.mean(b):.4f}")
