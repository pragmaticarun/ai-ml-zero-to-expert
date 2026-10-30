"""Day 74: transfer learning with an ImageNet-pretrained ResNet-18 (torchvision weights, ~45 MB download) on 2,000 CIFAR-10-small images.
Compares: from scratch, frozen features + logistic regression, and fine-tuning the last stage. CPU-sized: 64x64 inputs."""
import sys, os, time; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, numpy as np
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression
from _cifar import splits
torch.manual_seed(0)
(Xtr, ytr), (Xva, yva), _, names, (mean, std) = splits()
IMN_M = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1); IMN_S = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
def prep(X):  # undo CIFAR normalisation, resize to 64, apply ImageNet normalisation
    X = X * std + mean; X = nn.functional.interpolate(X, size=64, mode="bilinear", align_corners=False); return (X - IMN_M) / IMN_S
N = 2000; Xs, ys = prep(Xtr[:N]), ytr[:N]; Xv = prep(Xva); print("train images:", N, "| val images:", len(yva), "| input", tuple(Xs.shape[1:]))
def acc(m, X, y):
    m.eval(); c = 0
    with torch.no_grad():
        for s in range(0, len(y), 500): c += (m(X[s:s+500]).argmax(1) == y[s:s+500]).sum().item()
    return c / len(y)
def fit(m, params, epochs, lr):
    o = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4); g = torch.Generator().manual_seed(0)
    for ep in range(epochs):
        m.train(); p = torch.randperm(N, generator=g)
        for s in range(0, N, 64):
            i = p[s:s+64]; xb = Xs[i]; f = torch.rand(len(i)) < 0.5; xb[f] = xb[f].flip(3)
            l = nn.functional.cross_entropy(m(xb), ys[i]); o.zero_grad(); l.backward(); o.step()
m = resnet18(num_classes=10); t = time.perf_counter(); fit(m, m.parameters(), 6, 0.05); print(f"ResNet-18 from scratch, 6 epochs: val acc {acc(m, Xv, yva):.4f} ({time.perf_counter()-t:.0f} s)")
w = ResNet18_Weights.IMAGENET1K_V1; m = resnet18(weights=w, progress=False); print("pretrained weights:", w.url.split("/")[-1], "| ImageNet top-1 reported in torchvision metadata:", w.meta["_metrics"]["ImageNet-1K"]["acc@1"])
m.fc = nn.Identity(); m.eval(); t = time.perf_counter()
with torch.no_grad(): Ftr = torch.cat([m(Xs[s:s+250]) for s in range(0, N, 250)]).numpy(); Fva = torch.cat([m(Xv[s:s+250]) for s in range(0, len(yva), 250)]).numpy()
lr = LogisticRegression(max_iter=3000, C=0.5).fit(Ftr, ys.numpy()); print(f"frozen ResNet-18 features ({Ftr.shape[1]}-d) + logistic regression: val acc {lr.score(Fva, yva.numpy()):.4f} ({time.perf_counter()-t:.0f} s incl. feature extraction)")
m = resnet18(weights=w, progress=False); m.fc = nn.Linear(512, 10)
for n_, p in m.named_parameters(): p.requires_grad = n_.startswith(("layer4", "fc"))
tr = [p for p in m.parameters() if p.requires_grad]; print(f"fine-tune: trainable params {sum(p.numel() for p in tr):,} of {sum(p.numel() for p in m.parameters()):,}")
t = time.perf_counter(); fit(m, tr, 4, 0.01); print(f"fine-tuned layer4+fc, 4 epochs: val acc {acc(m, Xv, yva):.4f} ({time.perf_counter()-t:.0f} s)")
