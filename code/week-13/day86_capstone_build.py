"""Day 86 capstone build: baselines and the main small-ResNet run on CIFAR-10-small; saves the checkpoint for Days 87-88."""
import sys, os, time, csv; sys.path.insert(0, os.path.dirname(__file__))
import torch, numpy as np
from _capstone import fit, splits, make_model
from sklearn.linear_model import LogisticRegression
data = splits(); (Xtr, ytr), (Xva, yva), _, names, _ = data
print(f"train {len(ytr)}, val {len(yva)}, test held out; class counts in train: {torch.bincount(ytr).tolist()}")
print(f"majority-class baseline val acc {(yva == torch.bincount(ytr).argmax()).float().mean():.4f}")
t = time.perf_counter(); lr = LogisticRegression(max_iter=300).fit(Xtr[:, :, ::2, ::2].flatten(1).numpy(), ytr.numpy())
print(f"logistic regression on 16x16 pixels val acc {lr.score(Xva[:, :, ::2, ::2].flatten(1).numpy(), yva.numpy()):.4f} ({time.perf_counter()-t:.0f} s)")
print(f"small ResNet params (width 32): {sum(p.numel() for p in make_model().parameters()):,}")
m, hist, el = fit(epochs=15, verbose=True, data=data)
os.makedirs("out", exist_ok=True); torch.save(m.state_dict(), "out/capstone.pt")
with open("out/day86_history.csv", "w", newline="") as f: w = csv.writer(f); w.writerow(["epoch", "train_loss", "val_acc"]); w.writerows(hist)
print(f"main run, 15 epochs: final val acc {hist[-1][2]:.4f}, best {max(h[2] for h in hist):.4f}, time {el:.0f} s; checkpoint out/capstone.pt")
