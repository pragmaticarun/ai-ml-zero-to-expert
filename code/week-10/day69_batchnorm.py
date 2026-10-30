"""Day 69: batch norm and layer norm: deeper MLP with and without normalisation, at two learning rates (MNIST subset)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
from _mnist import tensors, evaluate, train
Xtr, ytr, Xte, yte = tensors(n_train=10000)
def deep(norm, depth=10, w=128):
    torch.manual_seed(0); L = [nn.Flatten()]; d = 784
    for _ in range(depth):
        L.append(nn.Linear(d, w)); d = w
        if norm == "batch": L.append(nn.BatchNorm1d(w))
        if norm == "layer": L.append(nn.LayerNorm(w))
        L.append(nn.ReLU())
    return nn.Sequential(*L, nn.Linear(d, 10))
for lr in (0.01, 0.1, 0.5):
    for norm in ("none", "batch", "layer"):
        m = deep(norm); train(m, torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9), Xtr, ytr, epochs=5)
        l, a = evaluate(m, Xte, yte); print(f"depth 10, lr {lr:<4} norm {norm:5s}: test loss {l:.3f} acc {a:.4f}")
# what BN does to one batch
bn = nn.BatchNorm1d(3); x = torch.randn(256, 3) * torch.tensor([1., 5., 20.]) + torch.tensor([0., 10., -50.])
y = bn(x); print("BN train mode: batch means", [round(v, 3) for v in y.mean(0).tolist()], "stds", [round(v, 3) for v in y.std(0, unbiased=False).tolist()])
print("running_mean after one batch (momentum 0.1):", [round(v, 2) for v in bn.running_mean.tolist()])
bn.eval(); one = bn(x[:1]); bn.train()
try: bn(x[:1]); print("train mode on a batch of 1: ok")
except ValueError as e: print("train mode on a batch of 1 raises:", str(e)[:60])
print("eval mode on a batch of 1 works:", tuple(one.shape))
# parameter count of BN vs LN
print("BatchNorm1d(128) params:", sum(p.numel() for p in nn.BatchNorm1d(128).parameters()), "+ buffers", sum(b.numel() for b in nn.BatchNorm1d(128).buffers()), "| LayerNorm(128) params:", sum(p.numel() for p in nn.LayerNorm(128).parameters()))
