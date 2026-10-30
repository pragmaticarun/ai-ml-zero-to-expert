"""Day 72: LeNet-style, VGG-style and ResNet-style networks: shapes, parameter counts, and a plain-vs-residual depth experiment on Fashion-MNIST."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import torch, torch.nn as nn
from _mnist import tensors, evaluate, train
npar = lambda m: sum(p.numel() for p in m.parameters())
lenet = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Linear(16*5*5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
x = torch.zeros(1, 1, 28, 28); print("LeNet-style shapes:", end=" ")
for layer in lenet:
    x = layer(x)
    if isinstance(layer, (nn.Conv2d, nn.MaxPool2d, nn.Linear)): print(tuple(x.shape[1:]), end=" ")
print(f"\nLeNet-style params: {npar(lenet):,}")
def vgg_block(i, o): return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
vgg = nn.Sequential(vgg_block(3, 32), vgg_block(32, 64), vgg_block(64, 128), nn.Flatten(), nn.Linear(128*4*4, 10)); print(f"small VGG-style (CIFAR) params: {npar(vgg):,}")
class Block(nn.Module):
    def __init__(s, c, residual=True):
        super().__init__(); s.r = residual; s.f = nn.Sequential(nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1, bias=False), nn.BatchNorm2d(c))
    def forward(s, x): return torch.relu(s.f(x) + x) if s.r else torch.relu(s.f(x))
def net(n_blocks, residual, c=16):
    torch.manual_seed(0); return nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), *[Block(c, residual) for _ in range(n_blocks)], nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c, 10))
print(f"one residual block (16 ch) params: {npar(Block(16)):,}")
Xtr, ytr, Xte, yte = tensors(fashion=True, n_train=5000); Xtr, Xte = Xtr[:, None], Xte[:2000, None]; yte = yte[:2000]
for nb in (2, 10):
    for res in (False, True):
        m = net(nb, res); x = Xtr[:256]; l = nn.functional.cross_entropy(m(x), ytr[:256]); l.backward(); g0 = m[0].weight.grad.norm().item()
        t = time.perf_counter(); train(m, torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9), Xtr, ytr, epochs=5); el = time.perf_counter() - t
        print(f"{'residual' if res else 'plain':8s} {nb:2d} blocks ({2*nb+2} weight layers): first-layer grad norm at init {g0:.2e}, train loss {evaluate(m, Xtr, ytr)[0]:.3f}, test acc after 5 epochs {evaluate(m, Xte, yte)[1]:.4f} ({el:.0f} s)")
