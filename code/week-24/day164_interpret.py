"""Day 164: interpretability: permutation importance; Shapley values by sampling (from scratch); LIME-style local surrogate; CNN gradient saliency + occlusion; linear probing of CNN layers."""
from _dep import *
import torch, time
Xtr, Xte, ytr, yte, gtr, gte = adult(); m = lr_pipe().fit(Xtr, ytr); rng = np.random.default_rng(0)
Xs = Xte.iloc[:3000].reset_index(drop=True); ys = yte[:3000]; base = (m.predict(Xs) == ys).mean()
imp = {}
for c in Xs.columns:
    d = []
    for r in range(3):
        Xp = Xs.copy(); Xp[c] = Xp[c].values[rng.permutation(len(Xp))]; d.append(base - (m.predict(Xp) == ys).mean())
    imp[c] = (np.mean(d), np.std(d))
print(f"(1) permutation importance (acc drop, 3 repeats, 3000 test rows; base acc {base:.4f}):")
for c, (a, s) in sorted(imp.items(), key=lambda kv: -kv[1][0]): print(f"    {c:16s} {a:+.4f} (sd {s:.4f})")
# (2) Shapley by permutation sampling: value = model logit; missing features filled from a random background row
f = lambda X: m.decision_function(X); bg = Xtr.sample(200, random_state=0).reset_index(drop=True)
x = Xs.iloc[[int(np.argmax(m.predict_proba(Xs)[:,1]))]]; cols = list(Xs.columns); P = 300; phi = dict.fromkeys(cols, 0.0); t = time.perf_counter()
for k in range(P):
    order = rng.permutation(cols); z = bg.iloc[[k % 200]].copy(); prev = f(z)[0]
    for c in order: z[c] = x[c].values; cur = f(z)[0]; phi[c] += (cur - prev) / P; prev = cur
fx = f(x)[0]; fb = f(bg).mean()
print(f"(2) sampled Shapley ({P} permutations, {time.perf_counter()-t:.1f} s) for the test row with highest P(>50K): f(x) {fx:.3f}, E[f(bg)] {fb:.3f}, sum phi {sum(phi.values()):.3f} (efficiency: should be ~ f(x)-E f = {fx-fb:.3f})")
for c, v in sorted(phi.items(), key=lambda kv: -abs(kv[1]))[:6]: print(f"    {c:16s} {v:+.3f}  (x = {x[c].values[0]})")
# (3) LIME-style: perturb by swapping features with background values, fit weighted ridge on binary 'kept' mask
from sklearn.linear_model import Ridge
Z = rng.integers(0, 2, (1000, len(cols))); rows = []
for i, zz in enumerate(Z):
    r = bg.iloc[[i % 200]].copy()
    for j, c in enumerate(cols):
        if zz[j]: r[c] = x[c].values
    rows.append(r)
import pandas as pd; R = pd.concat(rows, ignore_index=True); yz = f(R); wts = np.exp(-((len(cols) - Z.sum(1)) / len(cols))**2 / 0.25**2)
lime = Ridge(alpha=1.0).fit(Z, yz, sample_weight=wts); top = np.argsort(-abs(lime.coef_))[:6]
print("(3) LIME-style surrogate coefficients (same row; 1000 perturbations): " + ", ".join(f"{cols[j]} {lime.coef_[j]:+.3f}" for j in top))
sh_top = [c for c, _ in sorted(phi.items(), key=lambda kv: -abs(kv[1]))[:3]]; print(f"    top-3 agreement Shapley vs LIME: {len(set(sh_top) & set(cols[j] for j in top[:3]))}/3")
# (4) CNN saliency
cnn, ((_, _), (_, _), (Xc, yc), names, _) = cifar_cnn()
with torch.no_grad(): acc = (cnn(Xc).argmax(1) == yc).float().mean().item()
print(f"(4) Month 3 CNN test acc {acc:.4f} on {len(yc)} CIFAR-10-small test images")
xi = Xc[:64].clone().requires_grad_(True); out = cnn(xi); out.gather(1, out.argmax(1, keepdim=True)).sum().backward()
sal = xi.grad.abs().amax(1)                                  # 64 x 32 x 32
cen = sal[:, 8:24, 8:24].sum((1, 2)) / sal.sum((1, 2)); print(f"    gradient saliency: share of saliency in the central 16x16 (25% of pixels): mean {cen.mean():.3f}")
# occlusion check: does blanking the top-saliency 10% of pixels hurt more than random 10%?
def drop(mask):
    with torch.no_grad(): xo = Xc[:64] * (1 - mask[:, None]); o = cnn(xo).softmax(1)
    return o.gather(1, out.argmax(1, keepdim=True)).mean().item()
with torch.no_grad(): p0 = cnn(Xc[:64]).softmax(1).gather(1, out.argmax(1, keepdim=True)).mean().item()
k = int(0.1 * 1024); thr = sal.flatten(1).topk(k, 1).values[:, -1:, None]; msal = (sal >= thr).float()
mrand = torch.zeros(64, 1024); [mrand[i].index_fill_(0, torch.randperm(1024, generator=torch.Generator().manual_seed(i))[:k], 1.0) for i in range(64)]
print(f"    occlusion of 10% pixels (set to mean 0): conf {p0:.3f} -> top-saliency {drop(msal):.3f} vs random {drop(mrand.view(64,32,32)):.3f}")
# (5) probing: linear probe for animal-vs-vehicle at each block output
from sklearn.linear_model import LogisticRegression
(Xa, ya), (Xv, yv), _, _, _ = cifar_cnn()[1]; animal = torch.tensor([0,0,1,1,1,1,1,1,0,0])   # CIFAR order: plane, car, bird, cat, deer, dog, frog, horse, ship, truck
def feats(X, upto):
    with torch.no_grad(): return torch.cat([cnn[:upto](X[i:i+1000]).flatten(1) for i in range(0, len(X), 1000)]).numpy()
print("(5) linear probe (animal vs vehicle; train 3000 / test 2500 val images):")
for nm, upto in (("pixels", 0), ("block1", 7), ("block2", 14), ("block3", 21)):
    A = Xa[:3000].flatten(1).numpy() if upto == 0 else feats(Xa[:3000], upto); B = Xv.flatten(1).numpy() if upto == 0 else feats(Xv, upto)
    pr = LogisticRegression(max_iter=300, C=0.1).fit(A, animal[ya[:3000]].numpy()); print(f"    {nm:7s} dim {A.shape[1]:5d}: probe acc {pr.score(B, animal[yv].numpy()):.4f}")
print(f"    majority baseline {max(animal[yv].float().mean().item(), 1-animal[yv].float().mean().item()):.4f}")
