"""Day 76: signal pipelines for multimedia developers: spectrograms with torch.stft, SpecAugment-style masking, and a small CNN on a
SYNTHETIC audio task (no audio dataset is downloaded; clips are generated so the labels are known exactly)."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, numpy as np, matplotlib.pyplot as plt
from common import savefig
torch.manual_seed(0); SR = 8000; T = 1.0; t = torch.arange(int(SR * T)) / SR
def clip(kind, g):
    f0 = 300 + 900 * torch.rand(1, generator=g).item()
    if kind == 0: x = torch.sin(2 * np.pi * f0 * t)                                   # steady tone
    elif kind == 1: x = torch.sin(2 * np.pi * (f0 * t + 600 * t**2))                   # up-chirp
    elif kind == 2: x = torch.sin(2 * np.pi * f0 * t) * (torch.sin(2 * np.pi * 4 * t) > 0)   # 4 Hz on/off beeps
    else: x = torch.sin(2 * np.pi * f0 * t) + 0.6 * torch.sin(2 * np.pi * 2 * f0 * t)  # tone + harmonic
    return x + 2.0 * torch.randn(len(t), generator=g)                                  # heavy noise: noise power about 6-8x the signal power
KINDS = ["tone", "chirp", "beeps", "harmonic"]; g = torch.Generator().manual_seed(0)
X = torch.stack([clip(k % 4, g) for k in range(1600)]); y = torch.arange(1600) % 4
win = torch.hann_window(256)
def spec(x): S = torch.stft(x, n_fft=256, hop_length=128, window=win, return_complex=True).abs(); return torch.log1p(S)
S = spec(X); print("waveforms", tuple(X.shape), "-> log-magnitude spectrograms", tuple(S.shape), f"(freq bins {S.shape[1]} = 256/2+1, frames {S.shape[2]}, bin width {SR/256:.2f} Hz, hop {128/SR*1000:.0f} ms)")
fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))
for k in range(4): ax[k].imshow(S[k], origin="lower", aspect="auto"); ax[k].set_title(KINDS[k])
savefig(__file__, "day76_spectrograms.png")
def specaugment(s, g):
    s = s.clone(); B, F, Tt = s.shape
    for b in range(B):
        f = torch.randint(0, 12, (1,), generator=g).item(); f0 = torch.randint(0, F - f, (1,), generator=g).item(); s[b, f0:f0+f] = 0
        w = torch.randint(0, 10, (1,), generator=g).item(); t0 = torch.randint(0, Tt - w, (1,), generator=g).item(); s[b, :, t0:t0+w] = 0
    return s
mu, sd = S[:1200].mean(), S[:1200].std(); S = (S - mu) / sd
def net(): torch.manual_seed(0); return nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(512, 4))
for aug in (False, True):
    m = net(); o = torch.optim.Adam(m.parameters(), 1e-3); g2 = torch.Generator().manual_seed(0); tt = time.perf_counter()
    for ep in range(6):
        p = torch.randperm(1200, generator=g2)
        for s in range(0, 1200, 32):
            xb = S[p[s:s+32]]; xb = specaugment(xb, g2) if aug else xb
            l = nn.functional.cross_entropy(m(xb[:, None]), y[p[s:s+32]]); o.zero_grad(); l.backward(); o.step()
    m.eval()
    with torch.no_grad(): pr = m(S[1200:, None]).argmax(1)
    print(f"CNN on spectrograms, SpecAugment={aug}: held-out acc {(pr == y[1200:]).float().mean():.4f} on 400 clips ({time.perf_counter()-tt:.0f} s)")
# baseline: logistic regression on the time-averaged spectrum (throws away all timing information)
from sklearn.linear_model import LogisticRegression
F = S.mean(2).numpy(); lr = LogisticRegression(max_iter=2000).fit(F[:1200], y[:1200].numpy()); pr = lr.predict(F[1200:])
print(f"logistic regression on time-averaged spectrum: held-out acc {(pr == y[1200:].numpy()).mean():.4f}; per-class:", {KINDS[k]: round(float((pr[y[1200:].numpy() == k] == k).mean()), 3) for k in range(4)})
from torchvision.transforms import v2
tf = v2.Compose([v2.RandomResizedCrop(32, scale=(0.6, 1.0), antialias=True), v2.RandomHorizontalFlip(), v2.ColorJitter(0.3, 0.3)])
img = torch.rand(3, 48, 48); print("torchvision v2 pipeline output shape:", tuple(tf(img).shape))
