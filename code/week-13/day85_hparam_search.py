"""Day 85: hyperparameter search done honestly: grid vs random search on the same budget, a log file, and seed noise."""
import sys, os, time, math, csv; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-10"))
import torch, torch.nn as nn, numpy as np
from _mnist import tensors, evaluate, train
Xa, ya, _, _ = tensors(n_train=15000); Xa = Xa.flatten(1); Xtr, ytr, Xva, yva = Xa[:10000], ya[:10000], Xa[10000:], ya[10000:]   # test set untouched
def run(lr, wd, seed=0):
    torch.manual_seed(seed); m = nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 10))
    o = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd); train(m, o, Xtr, ytr, 3, 128, seed=seed); return evaluate(m, Xva, yva)[1]
os.makedirs("out", exist_ok=True); f = open("out/day85_trials.csv", "w", newline=""); w = csv.writer(f); w.writerow(["method", "trial", "lr", "weight_decay", "val_acc"])
grid = [(lr, wd) for lr in (1e-4, 1e-3, 1e-2) for wd in (1e-4, 1e-2, 1e0)]
rng = np.random.default_rng(0); rand = [(10 ** rng.uniform(-4.5, -1.5), 10 ** rng.uniform(-5, 0)) for _ in range(9)]
for name, trials in (("grid", grid), ("random", rand)):
    best = (0, None); t = time.perf_counter()
    for i, (lr, wd) in enumerate(trials):
        a = run(lr, wd); w.writerow([name, i, f"{lr:.2e}", f"{wd:.2e}", f"{a:.4f}"]); best = max(best, (a, (lr, wd)), key=lambda z: z[0])
    print(f"{name:6s} search, 9 trials: best val acc {best[0]:.4f} at lr {best[1][0]:.2e}, wd {best[1][1]:.2e}; distinct lr values tried {len(set(round(math.log10(l), 3) for l, _ in trials))} [{time.perf_counter()-t:.0f} s]")
f.close(); print("all trials logged to out/day85_trials.csv")
lr, wd = 1e-3, 1e-2; accs = [run(lr, wd, s) for s in range(5)]
print(f"same config (lr 1e-3, wd 1e-2), 5 seeds: val acc mean {np.mean(accs):.4f}, std {np.std(accs, ddof=1):.4f}, min {min(accs):.4f}, max {max(accs):.4f}")
print("=> differences between configs smaller than about 2 std of seed noise are not evidence of anything")
