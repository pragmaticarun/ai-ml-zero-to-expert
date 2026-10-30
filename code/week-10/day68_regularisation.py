"""Day 68: regularisation on a deliberately small training set (Fashion-MNIST, 2,000 images): L2, dropout, early stopping, augmentation."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, matplotlib.pyplot as plt
from _mnist import tensors, evaluate, train
from common import savefig
Xall, yall, Xte, yte = tensors(fashion=True, n_train=7000)
Xtr, ytr, Xva, yva = Xall[:2000], yall[:2000], Xall[2000:7000], yall[2000:7000]   # 2,000 train, 5,000 validation; test kept for the end
def mlp(p=0.0): torch.manual_seed(0); return nn.Sequential(nn.Flatten(), nn.Linear(784, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 512), nn.ReLU(), nn.Dropout(p), nn.Linear(512, 10))
def shift(xb):  # random translation by up to 2 pixels (pads with background value)
    bg = xb.min().item(); out = torch.full_like(xb, bg); dx, dy = torch.randint(-2, 3, (2,)).tolist()
    out[:, max(dy,0):28+min(dy,0), max(dx,0):28+min(dx,0)] = xb[:, max(-dy,0):28+min(-dy,0), max(-dx,0):28+min(-dx,0)]; return out
res = {}
for name, p, wd, aug in [("baseline", 0, 0, None), ("weight decay 5e-4", 0, 5e-4, None), ("dropout 0.5", 0.5, 0, None), ("shift augmentation", 0, 0, shift), ("dropout+wd+aug", 0.5, 5e-4, shift)]:
    torch.manual_seed(0); m = mlp(p); o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9, weight_decay=wd); hist = []
    train(m, o, Xtr, ytr, epochs=60, aug=aug, per_epoch=lambda ep: hist.append((evaluate(m, Xtr, ytr), evaluate(m, Xva, yva))))
    tr_acc = hist[-1][0][1]; va = [h[1] for h in hist]; best = min(range(60), key=lambda i: va[i][0])
    res[name] = va; print(f"{name:20s} final: train acc {tr_acc:.4f} val acc {va[-1][1]:.4f} val loss {va[-1][0]:.3f} | best val loss {va[best][0]:.3f} at epoch {best+1} (val acc there {va[best][1]:.4f})")
for k, v in res.items(): plt.plot([x[0] for x in v], label=k)
plt.xlabel("epoch"); plt.ylabel("validation loss"); plt.legend(fontsize=7); savefig(__file__, "day68_val_loss.png")
# early stopping with patience 5 on the baseline
torch.manual_seed(0); m = mlp(); o = torch.optim.SGD(m.parameters(), lr=0.05, momentum=0.9); best, bad, state = 9e9, 0, None
for ep in range(60):
    train(m, o, Xtr, ytr, epochs=1, seed=ep); vl, va = evaluate(m, Xva, yva)
    if vl < best: best, bad, state, bep = vl, 0, {k: v.clone() for k, v in m.state_dict().items()}, ep + 1
    else: bad += 1
    if bad == 5: break
m.load_state_dict(state); print(f"early stopping (patience 5): stopped after epoch {ep+1}, restored epoch {bep}, val loss {best:.3f}, val acc {evaluate(m, Xva, yva)[1]:.4f}")
# dropout: train() vs eval() mode
d = nn.Dropout(0.5); torch.manual_seed(0); x = torch.ones(8); print("dropout train mode:", d(x).tolist(), "| eval mode:", d.eval()(x).tolist())
