"""Day 66: initialisation and vanishing/exploding gradients in a 10-layer MLP (MNIST subset)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, matplotlib.pyplot as plt
from _mnist import tensors, evaluate, train
from common import savefig
Xtr, ytr, Xte, yte = tensors(n_train=5000)
def deep(act, init, depth=10, w=128):
    torch.manual_seed(0); layers = []; d = 784
    for _ in range(depth): layers += [nn.Linear(d, w), act()]; d = w
    m = nn.Sequential(nn.Flatten(), *layers, nn.Linear(d, 10))
    for mod in m:
        if isinstance(mod, nn.Linear):
            nn.init.zeros_(mod.bias)
            if init == "std0.01": nn.init.normal_(mod.weight, 0, 0.01)
            elif init == "std1": nn.init.normal_(mod.weight, 0, 1.0)
            elif init == "xavier": nn.init.xavier_normal_(mod.weight)
            elif init == "he": nn.init.kaiming_normal_(mod.weight, nonlinearity="relu")
    return m
plt.figure(figsize=(6, 4))
for act, init in [(nn.Sigmoid, "xavier"), (nn.Tanh, "xavier"), (nn.ReLU, "std0.01"), (nn.ReLU, "std1"), (nn.ReLU, "he")]:
    m = deep(act, init); loss = nn.functional.cross_entropy(m(Xtr[:512]), ytr[:512]); loss.backward()
    norms = [mod.weight.grad.norm().item() for mod in m if isinstance(mod, nn.Linear)]
    name = f"{act.__name__}+{init}"; plt.semilogy(range(1, 12), norms, marker="o", label=name)
    opt = torch.optim.SGD(m.parameters(), lr=0.05); train(m, opt, Xtr, ytr, epochs=5)
    print(f"{name:15s} grad norm layer1 {norms[0]:.2e} layer11 {norms[-1]:.2e} ratio {norms[0]/max(norms[-1],1e-30):.1e} | initial loss {loss.item():.3f} | test acc after 5 epochs {evaluate(m, Xte, yte)[1]:.4f}")
plt.xlabel("layer"); plt.ylabel("weight-gradient norm at init"); plt.legend(fontsize=7); savefig(__file__, "day66_grad_norms.png")
# exploding gradients and clipping: plain RNN-like repeated multiplication
torch.manual_seed(0); W = torch.randn(64, 64) * 1.5 / 8; h = torch.randn(64)
for t in range(1, 51):
    h = W @ h
    if t in (10, 30, 50): print(f"repeated W (spectral radius {torch.linalg.eigvals(W).abs().max().item():.2f}): |h| after {t} steps = {h.norm().item():.2e}")
g = torch.tensor([30., 40.]); print("clip_grad_norm to 5: [30,40] ->", (g * min(1, 5 / g.norm())).tolist())
# He init is sensitive to the learning rate at depth 10: try smaller steps
for lr in (0.01, 0.02):
    m = deep(nn.ReLU, "he"); train(m, torch.optim.SGD(m.parameters(), lr=lr), Xtr, ytr, epochs=5); print(f"ReLU+he lr {lr}: test acc after 5 epochs {evaluate(m, Xte, yte)[1]:.4f}")
