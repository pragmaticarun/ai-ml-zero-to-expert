"""Month-1 capstone library: linear and logistic regression from scratch in NumPy (sklearn-like API)."""
import numpy as np

class StandardScalerNP:
    def fit(self, X): self.mu_, self.sd_ = X.mean(0), X.std(0); self.sd_[self.sd_ == 0] = 1; return self
    def transform(self, X): return (X - self.mu_) / self.sd_

class LinearRegressionGD:
    def __init__(self, lr=0.1, steps=2000, l2=0.0, tol=1e-10):
        self.lr, self.steps, self.l2, self.tol = lr, steps, l2, tol
    def fit(self, X, y):
        m, n = X.shape; self.w_, self.b_ = np.zeros(n), 0.0; self.history_ = []
        for _ in range(self.steps):
            err = X @ self.w_ + self.b_ - y
            J = (err @ err) / (2*m) + self.l2 * (self.w_ @ self.w_) / (2*m); self.history_.append(J)
            if not np.isfinite(J): raise FloatingPointError("diverged: lower the learning rate or scale features")
            self.w_ -= self.lr * (X.T @ err + self.l2 * self.w_) / m; self.b_ -= self.lr * err.mean()
            if len(self.history_) > 1 and abs(self.history_[-2] - J) < self.tol * max(J, 1): break
        return self
    def predict(self, X): return X @ self.w_ + self.b_
    def score(self, X, y): r = y - self.predict(X); return 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))

class LogisticRegressionGD:
    def __init__(self, lr=0.5, steps=5000, l2=1.0, tol=1e-10):
        self.lr, self.steps, self.l2, self.tol = lr, steps, l2, tol
    @staticmethod
    def _sig(z): return np.where(z >= 0, 1/(1+np.exp(-np.abs(z))), np.exp(-np.abs(z))/(1+np.exp(-np.abs(z))))  # stable
    def fit(self, X, y):
        m, n = X.shape; self.w_, self.b_ = np.zeros(n), 0.0; self.history_ = []
        for _ in range(self.steps):
            p = self._sig(X @ self.w_ + self.b_); pc = np.clip(p, 1e-12, 1-1e-12)
            J = -(y*np.log(pc) + (1-y)*np.log(1-pc)).mean() + self.l2*(self.w_ @ self.w_)/(2*m); self.history_.append(J)
            self.w_ -= self.lr * (X.T @ (p - y) + self.l2*self.w_) / m; self.b_ -= self.lr * (p - y).mean()
            if len(self.history_) > 1 and abs(self.history_[-2] - J) < self.tol: break
        return self
    def predict_proba(self, X): return self._sig(X @ self.w_ + self.b_)
    def predict(self, X, threshold=0.5): return (self.predict_proba(X) >= threshold).astype(int)
    def score(self, X, y): return (self.predict(X) == y).mean()
