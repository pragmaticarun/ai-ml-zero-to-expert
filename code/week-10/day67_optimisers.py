"""Day 67: SGD, momentum, RMSProp, Adam, AdamW; learning-rate schedules and an LR range test (MNIST subset)."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from _mnist import tensors, evaluate, train
from common import savefig
# 1. optimiser paths on an elongated quadratic f = 0.5*(x^2 + 25 y^2)
def path(make, steps=50):
    p = torch.tensor([-4.0, 1.0], requires_grad=True); o = make([p])
    for _ in range(steps):
        f = 0.5 * (p[0]**2 + 25 * p[1]**2); o.zero_grad(); f.backward(); o.step()
    return 0.5 * (p[0]**2 + 25 * p[1]**2).item()
for name, mk in [("SGD lr .03", lambda q: torch.optim.SGD(q, lr=0.03)), ("SGD lr .085", lambda q: torch.optim.SGD(q, lr=0.085)),
                 ("momentum .9 lr .03", lambda q: torch.optim.SGD(q, lr=0.03, momentum=0.9)), ("RMSProp lr .03", lambda q: torch.optim.RMSprop(q, lr=0.03)),
                 ("Adam lr .1", lambda q: torch.optim.Adam(q, lr=0.1))]:
    print(f"quadratic, 50 steps, {name:20s}: f = {path(mk):.2e}")
# 2. optimisers on MNIST (10k images, 784-256-10, 5 epochs)
Xtr, ytr, Xte, yte = tensors(n_train=10000)
def mlp(): torch.manual_seed(0); return nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10))
for name, mk in [("SGD 0.05", lambda q: torch.optim.SGD(q, lr=0.05)), ("SGD+momentum 0.05/0.9", lambda q: torch.optim.SGD(q, lr=0.05, momentum=0.9)),
                 ("RMSProp 1e-3", lambda q: torch.optim.RMSprop(q, lr=1e-3)), ("Adam 1e-3", lambda q: torch.optim.Adam(q, lr=1e-3)),
                 ("Adam 1e-1 (too high)", lambda q: torch.optim.Adam(q, lr=1e-1)), ("AdamW 1e-3 wd 0.01", lambda q: torch.optim.AdamW(q, lr=1e-3, weight_decay=0.01))]:
    m = mlp(); o = mk(m.parameters()); train(m, o, Xtr, ytr, epochs=5); print(f"MNIST-10k 5 epochs {name:24s}: test loss {evaluate(m, Xte, yte)[0]:.4f} acc {evaluate(m, Xte, yte)[1]:.4f}")
# 3. schedules (SGD+momentum, 10 epochs)
steps_per_ep = (10000 + 127) // 128
for name in ["constant", "step(5, 0.1)", "cosine", "onecycle"]:
    m = mlp(); o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9); s = None
    if name.startswith("step"): s = torch.optim.lr_scheduler.StepLR(o, 5, 0.1)
    elif name == "cosine": s = torch.optim.lr_scheduler.CosineAnnealingLR(o, 10)
    elif name == "onecycle": s = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=0.1, total_steps=10 * steps_per_ep); s._per_step = True
    train(m, o, Xtr, ytr, epochs=10, sched=s); print(f"schedule {name:13s} 10 epochs: test acc {evaluate(m, Xte, yte)[1]:.4f}")
# 4. LR range test: increase lr exponentially over 100 steps, record loss
m = mlp(); o = torch.optim.SGD(m.parameters(), lr=1e-5, momentum=0.9); lrs = np.logspace(-5, 1, 100); losses = []; g = torch.Generator().manual_seed(0)
for lr in lrs:
    for pg in o.param_groups: pg["lr"] = lr
    i = torch.randint(0, 10000, (128,), generator=g); l = nn.functional.cross_entropy(m(Xtr[i]), ytr[i]); o.zero_grad(); l.backward(); o.step(); losses.append(l.item())
sm = np.convolve(losses, np.ones(5) / 5, mode="same"); best = lrs[np.nanargmin(sm[2:-2]) + 2]
div = next((lr for lr, l in zip(lrs, losses) if not np.isfinite(l) or l > 4 * losses[0]), None)
print(f"LR range test: lowest smoothed loss at lr {best:.3g}; loss blew up (>4x start) at lr {div:.3g}" if div else f"LR range test: lowest smoothed loss at lr {best:.3g}")
plt.semilogx(lrs, losses); plt.ylim(0, 5); plt.xlabel("learning rate"); plt.ylabel("batch loss"); savefig(__file__, "day67_lr_range.png")
