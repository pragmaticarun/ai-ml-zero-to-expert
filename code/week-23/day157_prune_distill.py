"""Day 157: pruning and distillation at toy scale.
(1) global unstructured magnitude pruning at several sparsities, with/without a short fine-tune; dense latency does NOT drop;
(2) structured pruning: drop whole hidden neurons by L2 norm, so the matrices really shrink (latency and size drop);
(3) knowledge distillation: a tiny student trained on labels only vs with the teacher's softened logits, 5 seeds each."""
import copy, statistics, numpy as np, torch, torch.nn as nn, torch.nn.utils.prune as prune, _dep as D
torch.set_num_threads(1)
Xtr, Xte, ytr, yte = D.data(); T = D.base_model(); x256 = torch.from_numpy(Xte[:256])
base_acc = D.acc(T, Xte, yte)
with torch.no_grad(): lat0 = D.bench(lambda: T(x256), reps=50)
print(f"teacher/base fp32 64-512-512-10: acc {base_acc:.4f}; b256 {lat0:.0f} us (1 thread)")
print("(1) global unstructured L1 pruning (all 3 Linear weights):")
for sp in [0.5, 0.8, 0.9, 0.95, 0.98]:
    m = copy.deepcopy(T); ps = [(l, "weight") for l in m if isinstance(l, nn.Linear)]
    prune.global_unstructured(ps, pruning_method=prune.L1Unstructured, amount=sp); a0 = D.acc(m, Xte, yte)
    D.train(m.train(), Xtr, ytr, epochs=5, lr=5e-4)  # masks stay applied during fine-tune
    for l, n in ps: prune.remove(l, n)
    z = sum((l.weight == 0).sum().item() for l, _ in ps) / sum(l.weight.numel() for l, _ in ps)
    with torch.no_grad(): lat = D.bench(lambda: m(x256), reps=50)
    print(f"    sparsity {sp:.2f}: zeros {z:.3f}; acc no-finetune {a0:.4f} -> after 5-epoch finetune {D.acc(m, Xte, yte):.4f}; dense b256 {lat:.0f} us")
print("(2) structured pruning: keep the top-k hidden neurons (L2 norm of incoming weights) in both hidden layers, then 5-epoch finetune:")
def shrink(model, k):
    l1, l2, l3 = [l for l in model if isinstance(l, nn.Linear)]
    i1 = l1.weight.norm(dim=1).topk(k).indices.sort().values; i2 = l2.weight[:, i1].norm(dim=1).topk(k).indices.sort().values
    n = D.mlp(h=k)
    with torch.no_grad():
        n[0].weight.copy_(l1.weight[i1]); n[0].bias.copy_(l1.bias[i1])
        n[2].weight.copy_(l2.weight[i2][:, i1]); n[2].bias.copy_(l2.bias[i2]); n[4].weight.copy_(l3.weight[:, i2]); n[4].bias.copy_(l3.bias)
    return n.eval()
for k in [256, 128, 64, 32, 16]:
    n = shrink(T, k); a0 = D.acc(n, Xte, yte); D.train(n.train(), Xtr, ytr, epochs=5, lr=5e-4)
    with torch.no_grad(): lat = D.bench(lambda: n(x256), reps=50)
    print(f"    hidden {k:3d}: params {sum(p.numel() for p in n.parameters()):6d}; acc before ft {a0:.4f} -> after {D.acc(n, Xte, yte):.4f}; b256 {lat:.0f} us ({lat0/lat:.1f}x faster)")
print("(3) distillation: student 64-h-10 (one hidden layer), T=4; teacher = the base model; 5 seeds each; same init and batch order for both arms:")
for h, ep, alpha in [(8, 30, 0.7), (8, 120, 0.7), (8, 120, 0.3), (16, 120, 0.3)]:
    plain, kd = [], []
    for s in range(5):
        torch.manual_seed(100 + s); a = D.mlp(h=h, depth=1); b = copy.deepcopy(a)
        plain.append(D.acc(D.train(a, Xtr, ytr, epochs=ep, seed=s), Xte, yte)); kd.append(D.acc(D.train(b, Xtr, ytr, epochs=ep, seed=s, teacher=T, alpha=alpha), Xte, yte))
    print(f"    h={h:2d} epochs={ep:3d} alpha={alpha}: labels-only mean {statistics.mean(plain):.4f} (sd {statistics.stdev(plain):.4f}) | distilled mean {statistics.mean(kd):.4f} (sd {statistics.stdev(kd):.4f}); per-seed diff {[round(k-p,4) for k,p in zip(kd,plain)]}")
