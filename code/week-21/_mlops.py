"""Week 21 shared helpers: data (sklearn's bundled breast-cancer table, no download), a small torch MLP, training, metrics."""
import os, json, hashlib, numpy as np, torch, torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
torch.set_num_threads(1)
def data(seed=0):
    d = load_breast_cancer(); X, y = d.data.astype(np.float32), d.target.astype(np.int64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    return Xtr, Xte, ytr, yte, list(d.feature_names)
class MLP(nn.Module):
    def __init__(self, d_in=30, h=32, mu=None, sd=None):
        super().__init__()
        self.register_buffer("mu", torch.zeros(d_in) if mu is None else torch.tensor(mu)); self.register_buffer("sd", torch.ones(d_in) if sd is None else torch.tensor(sd))
        self.net = nn.Sequential(nn.Linear(d_in, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 2))
    def forward(self, x): return self.net((x - self.mu) / self.sd)   # normalisation lives INSIDE the model, so serving can't forget it
def train(Xtr, ytr, h=32, lr=1e-2, epochs=60, wd=1e-4, seed=0):
    torch.manual_seed(seed); m = MLP(Xtr.shape[1], h, Xtr.mean(0), Xtr.std(0) + 1e-6)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wd); X, y = torch.tensor(Xtr), torch.tensor(ytr); g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        for idx in torch.randperm(len(X), generator=g).split(32):
            opt.zero_grad(); nn.functional.cross_entropy(m(X[idx]), y[idx]).backward(); opt.step()
    return m.eval()
@torch.no_grad()
def proba(m, X): return torch.softmax(m(torch.tensor(X, dtype=torch.float32)), -1)[:, 1].numpy()
def acc(m, X, y): return float(((proba(m, X) > 0.5) == y).mean())
def weights_hash(m): return hashlib.sha256(b"".join(p.detach().numpy().tobytes() for p in m.state_dict().values())).hexdigest()[:16]
