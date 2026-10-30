"""Day 65: Dataset, DataLoader, batching, transforms and devices, on MNIST."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn, time
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from common import load_mnist
torch.manual_seed(0)
Xtr, ytr, Xte, yte = load_mnist(); print("MNIST:", Xtr.shape, Xte.shape, "pixel range", Xtr.min(), Xtr.max())
mean, std = Xtr.mean() / 255, Xtr.std() / 255; print(f"train pixel mean {mean:.4f} std {std:.4f} (computed on TRAIN only)")
class MNISTDS(Dataset):
    def __init__(s, X, y): s.X, s.y = X, y
    def __len__(s): return len(s.y)
    def __getitem__(s, i): return (torch.from_numpy(s.X[i]).float().div(255) - mean) / std, int(s.y[i])
ds = MNISTDS(Xtr, ytr); x0, y0 = ds[0]; print("one item:", tuple(x0.shape), y0)
train_ds, val_ds = random_split(ds, [55000, 5000], generator=torch.Generator().manual_seed(0)); print("split:", len(train_ds), len(val_ds))
dl = DataLoader(train_ds, batch_size=128, shuffle=True); xb, yb = next(iter(dl)); print("batch:", tuple(xb.shape), tuple(yb.shape), "batches/epoch:", len(dl))
# timing: per-item Dataset vs pre-built tensor
def time_epoch(loader):
    t = time.perf_counter(); n = 0
    for xb, yb in loader: n += len(yb)
    return time.perf_counter() - t, n
Xall = ((torch.from_numpy(Xtr).float() / 255 - mean) / std); tds = TensorDataset(Xall, torch.from_numpy(ytr))
for name, loader in [("per-item Dataset, bs=128, workers=0", DataLoader(ds, 128, shuffle=True)),
                     ("per-item Dataset, bs=128, workers=2", DataLoader(ds, 128, shuffle=True, num_workers=2)),
                     ("TensorDataset, bs=128", DataLoader(tds, 128, shuffle=True)),
                     ("manual index slicing, bs=128", None)]:
    if loader is None:
        t = time.perf_counter(); p = torch.randperm(60000)
        for s in range(0, 60000, 128): xb = Xall[p[s:s+128]]
        print(f"{name}: {time.perf_counter()-t:.2f} s/epoch")
    else: s, n = time_epoch(loader); print(f"{name}: {s:.2f} s/epoch ({n} images)")
dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device:", dev, "| CUDA available:", torch.cuda.is_available(), "| CPU threads:", torch.get_num_threads())
# batch size vs throughput for one MLP training epoch (CPU)
for bs in (16, 128, 1024):
    torch.manual_seed(0); m = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 10)).to(dev); o = torch.optim.SGD(m.parameters(), lr=0.05 * bs / 128)
    t = time.perf_counter()
    for xb, yb in DataLoader(tds, bs, shuffle=True):
        l = nn.functional.cross_entropy(m(xb.to(dev)), yb.to(dev)); o.zero_grad(); l.backward(); o.step()
    el = time.perf_counter() - t
    with torch.no_grad(): acc = (m(((torch.from_numpy(Xte).float()/255-mean)/std).to(dev)).argmax(1).cpu().numpy() == yte).mean()
    print(f"batch {bs:5d}: 1 epoch {el:.2f} s, test acc after 1 epoch {acc:.4f} (lr scaled linearly)")
