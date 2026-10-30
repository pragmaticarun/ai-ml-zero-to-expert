"""Day 166: adversarial robustness (FGSM, PGD L-inf) on the Month 3 CNN, corruption shift on CIFAR, covariate/subpopulation shift on Adult."""
from _dep import *
import torch, torch.nn.functional as F, time
torch.manual_seed(0); cnn, (_, _, (Xt, yt), names, (mean, std)) = cifar_cnn(); X, y = Xt[:500], yt[:500]
lo, hi = (0 - mean) / std, (1 - mean) / std                           # valid pixel range in normalised space
def acc(x):
    with torch.no_grad(): return (cnn(x).argmax(1) == y).float().mean().item()
def grad(x):
    x = x.clone().requires_grad_(True); F.cross_entropy(cnn(x), y).backward(); return x.grad
def fgsm(e): return torch.max(torch.min(X + (e / std) * grad(X).sign(), hi), lo)
def pgd(e, steps=10, a=None):
    a = a or e / 4; d = torch.empty_like(X).uniform_(-1, 1) * (e / std); x = torch.max(torch.min(X + d, hi), lo)
    for _ in range(steps):
        x = x + (a / std) * grad(x).sign(); x = torch.max(torch.min(x, X + e / std), X - e / std); x = torch.max(torch.min(x, hi), lo)
    return x
print(f"(1) Month 3 CNN on 500 CIFAR-10-small test images: clean acc {acc(X):.4f}")
for e255 in (1, 2, 4, 8):
    e = e255 / 255; t = time.perf_counter(); af = acc(fgsm(e)); tf = time.perf_counter() - t; t = time.perf_counter(); ap = acc(pgd(e)); tp = time.perf_counter() - t
    print(f"    eps {e255}/255 (L-inf, pixel scale): FGSM acc {af:.4f} ({tf:.1f} s) | PGD-10 acc {ap:.4f} ({tp:.1f} s)")
xr = torch.max(torch.min(X + (8/255/std) * torch.sign(torch.randn_like(X)), hi), lo); print(f"    random-sign noise of the same size (8/255): acc {acc(xr):.4f}")
# (2) corruption shift
print("(2) corruption shift (same 500 images):")
raw = X * std + mean
for nm, f in (("gaussian noise sd 0.05", lambda r: r + 0.05*torch.randn_like(r)), ("gaussian noise sd 0.1", lambda r: r + 0.1*torch.randn_like(r)), ("brightness +0.2", lambda r: r + 0.2), ("contrast x0.5", lambda r: (r - r.mean((1,2,3), keepdim=True))*0.5 + r.mean((1,2,3), keepdim=True)), ("blur 3x3 box", lambda r: F.avg_pool2d(F.pad(r, (1,1,1,1), mode="replicate"), 3, 1))):
    print(f"    {nm:24s} acc {acc((f(raw).clamp(0,1) - mean) / std):.4f}")
# (3) Adult: subpopulation shift + domain-classifier detection
from sklearn.metrics import roc_auc_score; from sklearn.linear_model import LogisticRegression
Xtr, Xte, ytr, yte, gtr, gte = adult()
young_tr = Xtr["age"].values < 40; young_te = Xte["age"].values < 40
m = lr_pipe().fit(Xtr[young_tr], ytr[young_tr]); mall = lr_pipe().fit(Xtr, ytr)
for nm, mask in (("age<40 (in-distribution)", young_te), ("age>=40 (shifted)", ~young_te)):
    print(f"(3) trained on age<40 only -> test {nm:26s}: acc {m.score(Xte[mask], yte[mask]):.4f} | model trained on all ages: {mall.score(Xte[mask], yte[mask]):.4f} | base rate {yte[mask].mean():.3f}")
ct = lr_pipe()[0].fit(Xtr); dom = np.r_[np.zeros(young_tr.sum()), np.ones((~young_te).sum())]
import pandas as pd
D = pd.concat([Xtr[young_tr], Xte[~young_te]]).copy(); D["age"] = 0                        # hide the shifting column itself
Zd = ct.transform(D); from sklearn.model_selection import cross_val_predict
pd_ = cross_val_predict(LogisticRegression(max_iter=2000), Zd, dom, cv=3, method="predict_proba")[:, 1]
print(f"    domain classifier (age<40 train vs age>=40 test, age column hidden): AUC {roc_auc_score(dom, pd_):.3f}  (0.5 = indistinguishable)")
r0 = Xte.sample(frac=0.5, random_state=1).index; A1 = Xte.loc[r0]; A2 = Xte.drop(r0); D2 = pd.concat([A1, A2]); dom2 = np.r_[np.zeros(len(A1)), np.ones(len(A2))]
p2 = cross_val_predict(LogisticRegression(max_iter=2000), ct.transform(D2), dom2, cv=3, method="predict_proba")[:, 1]; print(f"    same check on a random split of test (no shift): AUC {roc_auc_score(dom2, p2):.3f}")
