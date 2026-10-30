"""Day 87 capstone ablations: change one thing at a time (8 epochs each, same seed and split), plus a learning-curve point."""
import sys, os, csv; sys.path.insert(0, os.path.dirname(__file__))
from _capstone import fit, splits
data = splits(); rows = []
configs = [("full model", {}), ("no augmentation", {"aug": False}), ("no residual connections", {"residual": False}), ("no batch norm (lr 0.01)", {"bn": False, "lr": 0.01}),
           ("half width (16)", {"width": 16}), ("no weight decay", {"wd": 0.0}), ("label smoothing 0.1", {"ls": 0.1}), ("50% of training data", {"frac": 0.5})]
for name, kw in configs:
    m, hist, el = fit(epochs=8, data=data, **kw); rows.append((name, hist[-1][1], hist[-1][2], el))
    print(f"{name:26s}: final train loss {hist[-1][1]:.4f} val acc {hist[-1][2]:.4f} ({el:.0f} s)", flush=True)
os.makedirs("out", exist_ok=True)
with open("out/day87_ablations.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["config", "train_loss", "val_acc", "seconds"]); w.writerows(rows)
print("logged to out/day87_ablations.csv; deltas vs full model:", {r[0]: round(r[2] - rows[0][2], 4) for r in rows[1:]})
