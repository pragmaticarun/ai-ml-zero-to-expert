"""Day 70 project: MNIST MLP in PyTorch with an experiment log. Select on validation (5,000 held out from train); test once."""
import sys, os, time, csv, json, platform; sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
from _mnist import tensors, evaluate, train
from common import outdir
Xall, yall, Xte, yte = tensors()
p = torch.randperm(60000, generator=torch.Generator().manual_seed(0)); Xtr, ytr, Xva, yva = Xall[p[:55000]], yall[p[:55000]], Xall[p[55000:]], yall[p[55000:]]
def build(c):
    torch.manual_seed(c["seed"]); L = [nn.Flatten()]; d = 784
    for h in c["hidden"]:
        L.append(nn.Linear(d, h)); d = h
        if c["bn"]: L.append(nn.BatchNorm1d(h))
        L += [nn.ReLU(), nn.Dropout(c["dropout"])]
    return nn.Sequential(*L, nn.Linear(d, 10))
CONFIGS = [dict(name="A_sgd_256", hidden=[256], bn=False, dropout=0.0, opt="sgd", lr=0.05, wd=0.0, sched="none"),
           dict(name="B_adam_512x2_dropout", hidden=[512, 512], bn=False, dropout=0.2, opt="adam", lr=1e-3, wd=0.0, sched="none"),
           dict(name="C_bn_512x2_onecycle", hidden=[512, 512], bn=True, dropout=0.2, opt="sgd", lr=0.1, wd=5e-4, sched="onecycle")]
log = os.path.join(outdir(__file__), "day70_runs.csv"); rows = []
for c in CONFIGS:
    c.update(seed=0, epochs=8, bs=128); m = build(c)
    o = torch.optim.Adam(m.parameters(), lr=c["lr"], weight_decay=c["wd"]) if c["opt"] == "adam" else torch.optim.SGD(m.parameters(), lr=c["lr"], momentum=0.9, weight_decay=c["wd"])
    s = None
    if c["sched"] == "onecycle": s = torch.optim.lr_scheduler.OneCycleLR(o, max_lr=c["lr"], total_steps=c["epochs"] * ((55000 + 127) // 128)); s._per_step = True
    t = time.perf_counter(); train(m, o, Xtr, ytr, c["epochs"], c["bs"], sched=s); el = time.perf_counter() - t
    trl, tra = evaluate(m, Xtr, ytr); val, vaa = evaluate(m, Xva, yva)
    row = dict(name=c["name"], params=sum(q.numel() for q in m.parameters()), train_acc=round(tra, 4), val_loss=round(val, 4), val_acc=round(vaa, 4), seconds=round(el, 1), config=json.dumps(c), torch=torch.__version__, device="cpu")
    rows.append((row, m)); print({k: v for k, v in row.items() if k not in ("config", "torch", "device")})
with open(log, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0][0])); w.writeheader(); [w.writerow(r) for r, _ in rows]
best, bm = max(rows, key=lambda r: r[0]["val_acc"]); tl, ta = evaluate(bm, Xte, yte)
print(f"selected on validation: {best['name']} (val acc {best['val_acc']}) -> TEST acc {ta:.4f}, test errors {round((1-ta)*10000)} / 10000")
with torch.no_grad(): pred = bm(Xte).argmax(1)
cm = torch.zeros(10, 10, dtype=torch.long)
for t_, p_ in zip(yte, pred): cm[t_, p_] += 1
off = [(cm[i, j].item(), i, j) for i in range(10) for j in range(10) if i != j]; off.sort(reverse=True)
print("most frequent confusions (count, true, predicted):", off[:4]); print("log written to", log)
