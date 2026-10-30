"""Day 73: train a small CNN on CIFAR-10-small (15k train) on CPU: MLP baseline vs CNN vs CNN+BN+augmentation."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import torch, torch.nn as nn, matplotlib.pyplot as plt
from _cifar import splits, flip_crop
from _mnist import evaluate, train
from common import savefig, outdir
(Xtr, ytr), (Xva, yva), (Xte, yte), names, _ = splits(); print("train/val/test:", len(ytr), len(yva), len(yte), "| class counts in train:", torch.bincount(ytr).tolist())
def cnn(bn):
    torch.manual_seed(0); B = (lambda c: nn.BatchNorm2d(c)) if bn else (lambda c: nn.Identity())
    blk = lambda i, o: [nn.Conv2d(i, o, 3, padding=1), B(o), nn.ReLU(), nn.Conv2d(o, o, 3, padding=1), B(o), nn.ReLU(), nn.MaxPool2d(2)]
    return nn.Sequential(*blk(3, 32), *blk(32, 64), *blk(64, 128), nn.Flatten(), nn.Dropout(0.3), nn.Linear(128*16, 10))
def mlp(): torch.manual_seed(0); return nn.Sequential(nn.Flatten(), nn.Linear(3072, 512), nn.ReLU(), nn.Linear(512, 10))
EPOCHS = 10; curves = {}
for name, m, aug in [("MLP 3072-512-10", mlp(), None), ("CNN (no BN, no aug)", cnn(False), None), ("CNN + BN", cnn(True), None), ("CNN + BN + flip/crop", cnn(True), flip_crop)]:
    o = torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4); s = torch.optim.lr_scheduler.CosineAnnealingLR(o, EPOCHS); h = []
    t = time.perf_counter(); train(m, o, Xtr, ytr, EPOCHS, bs=128, sched=s, aug=aug, per_epoch=lambda ep: h.append((evaluate(m, Xtr[:5000], ytr[:5000])[1], evaluate(m, Xva, yva)[1]))); el = time.perf_counter() - t
    curves[name] = h; print(f"{name:22s} params {sum(p.numel() for p in m.parameters()):>9,}: train acc {h[-1][0]:.4f} val acc {h[-1][1]:.4f} ({el:.0f} s, {el/EPOCHS:.1f} s/epoch)")
    if aug: torch.save(m.state_dict(), os.path.join(outdir(__file__), "day73_cnn.pt")); final = m
for k, v in curves.items(): plt.plot([b for a, b in v], label=k)
plt.xlabel("epoch"); plt.ylabel("val acc"); plt.legend(fontsize=7); savefig(__file__, "day73_val_acc.png")
final.eval()
with torch.no_grad(): p = final(Xva).argmax(1)
print("per-class val acc:", {names[c]: round((p[yva == c] == c).float().mean().item(), 3) for c in range(10)})
