"""Day 77 project: Fashion-MNIST CNN classifier with validation-based choices, one test evaluation and an error analysis.
CPU budget: 20k training images, 5k validation, full 10k test."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from _mnist import tensors, evaluate, train
from common import savefig
from sklearn.linear_model import LogisticRegression
NAMES = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
Xall, yall, Xte, yte = tensors(fashion=True); Xall, Xte = Xall[:, None], Xte[:, None]
p = torch.randperm(60000, generator=torch.Generator().manual_seed(0)); Xtr, ytr, Xva, yva = Xall[p[:20000]], yall[p[:20000]], Xall[p[50000:55000]], yall[p[50000:55000]]
t = time.perf_counter(); base = LogisticRegression(max_iter=200).fit(Xtr.flatten(1).numpy(), ytr.numpy()); print(f"baseline logistic regression (same 20k train): val acc {base.score(Xva.flatten(1).numpy(), yva.numpy()):.4f} ({time.perf_counter()-t:.0f} s)")
def cnn(c):
    torch.manual_seed(0)
    return nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(), nn.MaxPool2d(2),
                         nn.Conv2d(c, 2*c, 3, padding=1), nn.BatchNorm2d(2*c), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(), nn.Dropout(0.3), nn.Linear(2*c*49, 10))
def hflip(xb): f = torch.rand(len(xb)) < 0.5; xb = xb.clone(); xb[f] = xb[f].flip(3); return xb
results = {}
for name, w, aug in [("cnn16", 16, None), ("cnn32", 32, None), ("cnn32+flip", 32, hflip)]:
    m = cnn(w); o = torch.optim.SGD(m.parameters(), lr=0.02, momentum=0.9, weight_decay=5e-4); s = torch.optim.lr_scheduler.CosineAnnealingLR(o, 6)
    t = time.perf_counter(); train(m, o, Xtr, ytr, 6, 128, sched=s, aug=aug); el = time.perf_counter() - t
    va = evaluate(m, Xva, yva); results[name] = (va[1], m); print(f"{name:11s}: train acc {evaluate(m, Xtr[:10000], ytr[:10000])[1]:.4f} val loss {va[0]:.4f} val acc {va[1]:.4f} ({el:.0f} s)")
best = max(results, key=lambda k: results[k][0]); m = results[best][1]; m.eval()
with torch.no_grad(): P = torch.cat([m(Xte[s:s+2000]).softmax(1) for s in range(0, 10000, 2000)])
pred = P.argmax(1); conf = P.max(1).values; ok = pred == yte; print(f"selected {best} on val -> TEST acc {ok.float().mean():.4f} ({(~ok).sum().item()} errors / 10000)")
print("per-class test acc:", {NAMES[c]: round(ok[yte == c].float().mean().item(), 3) for c in range(10)})
cm = np.zeros((10, 10), int)
for a, b in zip(yte.tolist(), pred.tolist()): cm[a, b] += 1
pairs = sorted(((int(cm[i, j] + cm[j, i]), NAMES[i], NAMES[j]) for i in range(10) for j in range(i + 1, 10)), reverse=True)[:4]
tot = (~ok).sum().item(); print("top confused pairs (both directions):", pairs, f"| share of all errors in top 4 pairs: {sum(n for n, _, _ in pairs)/tot:.2f}")
print(f"mean confidence: correct {conf[ok].mean():.3f}, wrong {conf[~ok].mean():.3f}; errors with confidence > 0.9: {((~ok) & (conf > 0.9)).sum().item()}")
ece = 0.0
for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
    b = (conf > lo) & (conf <= hi)
    if b.any(): ece += b.float().mean().item() * abs(ok[b].float().mean().item() - conf[b].mean().item())
print(f"expected calibration error (10 bins): {ece:.4f}")
idx = torch.nonzero(~ok).flatten(); idx = idx[conf[idx].argsort(descending=True)][:12]
fig, ax = plt.subplots(2, 6, figsize=(10, 4))
for a, i in zip(ax.flat, idx.tolist()): a.imshow(Xte[i, 0], cmap="gray"); a.set_title(f"T:{NAMES[yte[i]][:7]}\nP:{NAMES[pred[i]][:7]} {conf[i]:.2f}", fontsize=7); a.axis("off")
savefig(__file__, "day77_confident_errors.png")
