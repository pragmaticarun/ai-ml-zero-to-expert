"""Day 88 capstone error analysis: one test evaluation of the Day-86 model, then Ng-style slicing of the errors."""
import sys, os, csv; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, numpy as np, matplotlib.pyplot as plt
from _capstone import fit, splits, make_model, predict
from common import savefig
data = splits(); (Xtr, ytr), (Xva, yva), (Xte, yte), names, (mean, std) = data; m = make_model()
if os.path.exists("out/capstone.pt"): m.load_state_dict(torch.load("out/capstone.pt")); print("loaded out/capstone.pt")
else: print("no checkpoint; training the Day-86 model (15 epochs)"); m, _, _ = fit(epochs=15, data=data)
Ptr = predict(m, Xtr[:2500]); Pva = predict(m, Xva); P = predict(m, Xte); Pt = predict(m, Xte, tta=True)
acc = lambda P, y: (P.argmax(1) == y).float().mean().item()
print(f"train(first 2500) acc {acc(Ptr, ytr[:2500]):.4f} | val acc {acc(Pva, yva):.4f} | TEST acc {acc(P, yte):.4f} | test with flip-TTA {acc(Pt, yte):.4f}")
print(f"bias/variance view: train error {1-acc(Ptr, ytr[:2500]):.4f}, val-train gap {acc(Ptr, ytr[:2500])-acc(Pva, yva):.4f}")
pred = P.argmax(1); conf = P.max(1).values; ok = pred == yte; err = ~ok
print("per-class test acc:", {names[c]: round(ok[yte == c].float().mean().item(), 3) for c in range(10)})
cm = np.zeros((10, 10), int)
for a, b in zip(yte.tolist(), pred.tolist()): cm[a, b] += 1
pairs = sorted(((int(cm[i, j] + cm[j, i]), names[i], names[j]) for i in range(10) for j in range(i + 1, 10)), reverse=True)[:5]
print(f"errors {err.sum().item()} / {len(yte)}; top confused pairs: {pairs}; share in top 5: {sum(p[0] for p in pairs)/err.sum().item():.2f}")
animals = {2, 3, 4, 5, 6, 7}; cross = sum(1 for a, b in zip(yte[err].tolist(), pred[err].tolist()) if (a in animals) != (b in animals))
print(f"errors crossing animal/vehicle boundary: {cross} of {err.sum().item()}")
raw = Xte * std + mean; bright = raw.mean((1, 2, 3)); contrast = raw.std((1, 2, 3))
for nm, v in (("brightness", bright), ("contrast", contrast)):
    q = torch.quantile(v, torch.tensor([0.25, 0.75])); lo, hi = v <= q[0], v >= q[1]
    print(f"slice by {nm}: lowest quartile acc {ok[lo].float().mean():.4f}, highest quartile acc {ok[hi].float().mean():.4f}")
ece = 0.0
for l, h in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:]):
    b = (conf > l) & (conf <= h)
    if b.any(): ece += b.float().mean().item() * abs(ok[b].float().mean().item() - conf[b].mean().item())
print(f"confidence: correct {conf[ok].mean():.3f}, wrong {conf[err].mean():.3f}; ECE (10 bins) {ece:.4f}; wrong with conf > 0.9: {(err & (conf > 0.9)).sum().item()}")
for th in (0.5, 0.7, 0.9):
    k = conf >= th; print(f"abstain below confidence {th}: coverage {k.float().mean():.3f}, accuracy on kept {ok[k].float().mean():.4f}")
os.makedirs("out", exist_ok=True); idx = torch.nonzero(err).flatten()[:60]
with open("out/day88_review_sheet.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["test_index", "true", "pred", "confidence", "tag (fill in: ambiguous / label noise / small object / occluded / unusual view / model error)"])
    for i in idx.tolist(): w.writerow([i, names[yte[i]], names[pred[i]], f"{conf[i]:.3f}", ""])
print("wrote out/day88_review_sheet.csv (60 errors for manual tagging)")
top = torch.nonzero(err).flatten(); top = top[conf[top].argsort(descending=True)][:16]; fig, ax = plt.subplots(2, 8, figsize=(12, 3.6))
for a, i in zip(ax.flat, top.tolist()): a.imshow(raw[i].permute(1, 2, 0).clamp(0, 1).numpy()); a.set_title(f"T:{names[yte[i]][:6]}\nP:{names[pred[i]][:6]} {conf[i]:.2f}", fontsize=7); a.axis("off")
savefig(__file__, "day88_confident_errors.png")
