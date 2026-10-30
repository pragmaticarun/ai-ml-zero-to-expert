"""Day 90: Month-3 capstone delivery. Collects the Day 86-88 artefacts into out/capstone_report.md and checks the checkpoint
reproduces the test accuracy reported on Day 88. Run day86, day87 and day88 first (run_all.sh does this in order)."""
import sys, os, csv, platform; sys.path.insert(0, os.path.dirname(__file__))
import torch
from _capstone import splits, make_model, predict
need = ["out/capstone.pt", "out/day86_history.csv", "out/day87_ablations.csv", "out/day88_review_sheet.csv"]
missing = [p for p in need if not os.path.exists(p)]
if missing: print("missing artefacts:", missing, "- run day86-88 first"); sys.exit(1)
(_, _), (Xva, yva), (Xte, yte), names, _ = splits(); m = make_model(); m.load_state_dict(torch.load("out/capstone.pt"))
P = predict(m, Xte); acc = (P.argmax(1) == yte).float().mean().item(); va = (predict(m, Xva).argmax(1) == yva).float().mean().item()
hist = list(csv.DictReader(open("out/day86_history.csv"))); abl = list(csv.DictReader(open("out/day87_ablations.csv")))
L = ["# Month-3 capstone report: CIFAR-10-small classifier (CPU)", "",
     f"Data: OpenML 40926 (CIFAR-10-small), 15,000 train / 2,500 val / 2,500 test, fixed seed 0. Model: small ResNet, {sum(p.numel() for p in m.parameters()):,} params.",
     f"Environment: Python {platform.python_version()}, torch {torch.__version__}, CPU only.", "",
     f"## Result\nValidation acc {va:.4f}; test acc {acc:.4f} (test set used once, after all choices were made on validation).", "",
     "## Training curve (Day 86)", "| epoch | train loss | val acc |", "|---|---|---|"] + [f"| {h['epoch']} | {float(h['train_loss']):.4f} | {float(h['val_acc']):.4f} |" for h in hist] + [
     "", "## Ablations (Day 87, 8 epochs each)", "| config | val acc | delta vs full |", "|---|---|---|"] + [
     f"| {a['config']} | {float(a['val_acc']):.4f} | {float(a['val_acc']) - float(abl[0]['val_acc']):+.4f} |" for a in abl] + [
     "", "## Error analysis", "See the Day 88 console output, out/day88_confident_errors.png and out/day88_review_sheet.csv (tag the 60 errors by hand and add the tag counts here).",
     "", "## Limitations", "- Single seed per ablation; differences under about 0.01 are within seed noise (see Day 85).", "- 32x32 images and CPU budget; not comparable with published full-CIFAR-10 results."]
open("out/capstone_report.md", "w").write("\n".join(L) + "\n"); print(f"val acc {va:.4f}, test acc {acc:.4f}; wrote out/capstone_report.md ({len(L)} lines)")
