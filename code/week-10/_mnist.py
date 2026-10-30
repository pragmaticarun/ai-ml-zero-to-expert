"""Shared MNIST/Fashion-MNIST tensors for Week 10 scripts (normalised with train statistics)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch
from common import load_mnist
def tensors(fashion=False, n_train=None, seed=0):
    Xtr, ytr, Xte, yte = load_mnist(fashion)
    Xtr = torch.from_numpy(Xtr).float() / 255; Xte = torch.from_numpy(Xte).float() / 255
    m, s = Xtr.mean(), Xtr.std(); Xtr, Xte = (Xtr - m) / s, (Xte - m) / s
    ytr, yte = torch.from_numpy(ytr).long(), torch.from_numpy(yte).long()
    if n_train:
        i = torch.randperm(len(ytr), generator=torch.Generator().manual_seed(seed))[:n_train]; Xtr, ytr = Xtr[i], ytr[i]
    return Xtr, ytr, Xte, yte
def evaluate(model, X, y, bs=2000):
    model.eval(); c = 0; L = 0.0
    with torch.no_grad():
        for s in range(0, len(y), bs):
            o = model(X[s:s+bs]); L += torch.nn.functional.cross_entropy(o, y[s:s+bs], reduction="sum").item(); c += (o.argmax(1) == y[s:s+bs]).sum().item()
    model.train(); return L / len(y), c / len(y)
def train(model, opt, X, y, epochs, bs=128, sched=None, seed=0, aug=None, per_epoch=None):
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        model.train(); p = torch.randperm(len(y), generator=g)
        for s in range(0, len(y), bs):
            xb = X[p[s:s+bs]]; xb = aug(xb) if aug else xb
            l = torch.nn.functional.cross_entropy(model(xb), y[p[s:s+bs]]); opt.zero_grad(); l.backward(); opt.step()
            if sched is not None and getattr(sched, "_per_step", False): sched.step()
        if sched is not None and not getattr(sched, "_per_step", False): sched.step()
        if per_epoch: per_epoch(ep)
