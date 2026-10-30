"""Day 97: a miniature scaling experiment. Five GPT sizes, same data and token budget; fit loss = a * N^(-alpha) + c on our own numbers."""
import sys, os, time, math, csv, numpy as np, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gpt import GPT, train, nparams, char_data
from scipy.optimize import curve_fit
_, chars, tr, va = char_data(); rows = []
for d, L in ((32, 1), (48, 2), (64, 2), (96, 3), (128, 4)):
    torch.manual_seed(0); m = GPT(len(chars), ctx=64, d=d, h=4, L=L); N = nparams(m) - m.tok.weight.numel() - m.pos.weight.numel()
    t = time.perf_counter(); h = train(m, tr, va, 64, steps=600, bs=32, lr=3e-3, log_every=600); rows.append((d, L, N, h[-1][2], time.perf_counter() - t))
    print(f"d={d:3d} L={L}: non-embedding params {N:8,d}, val loss {h[-1][2]:.3f} ({rows[-1][-1]:.0f} s)", flush=True)
N = np.array([r[2] for r in rows], float); Lv = np.array([r[3] for r in rows])
(a, al, c), _ = curve_fit(lambda n, a, al, c: a * n ** (-al) + c, N, Lv, p0=(10, 0.3, 1.5), maxfev=20000)
print(f"fit: loss = {a:.2f} * N^(-{al:.3f}) + {c:.3f}  (5 points, 600 steps x 32 x 64 = {600*32*64:,} tokens each; an illustration, not a law)")
print("extrapolated loss at N = 1e6 and 1e7:", [round(a * n ** (-al) + c, 3) for n in (1e6, 1e7)])
os.makedirs("out", exist_ok=True); w = csv.writer(open("out/day97_scaling.csv", "w", newline="")); w.writerow(["d", "L", "non_emb_params", "val_loss", "seconds"]); w.writerows(rows)
print("tokens per parameter for the largest model:", round(600 * 32 * 64 / N[-1], 2), "(Chinchilla's compute-optimal rule of thumb is about 20)")
