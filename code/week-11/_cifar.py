"""Shared CIFAR-10-small split for Week 11-13 (OpenML 40926: 20,000 images). Fixed split: 15,000 train / 2,500 val / 2,500 test."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np
from common import load_cifar_small
def splits():
    X, y, names = load_cifar_small(); X = torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255; y = torch.from_numpy(y)
    p = torch.randperm(len(y), generator=torch.Generator().manual_seed(0)); tr, va, te = p[:15000], p[15000:17500], p[17500:]
    mean = X[tr].mean((0, 2, 3), keepdim=True); std = X[tr].std((0, 2, 3), keepdim=True); X = (X - mean) / std
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te]), names, (mean, std)
def flip_crop(xb):
    """Random horizontal flip + random crop with 4-pixel zero padding (per batch offsets, per-image flips)."""
    f = torch.rand(len(xb)) < 0.5; xb = xb.clone(); xb[f] = xb[f].flip(3)
    p = torch.nn.functional.pad(xb, (4, 4, 4, 4)); dx, dy = torch.randint(0, 9, (2,)).tolist(); return p[:, :, dy:dy+32, dx:dx+32]
