"""Day 165: privacy: Laplace mechanism, DP-SGD from scratch on Adult logistic regression (own RDP accountant, cross-checked with Opacus), membership-inference attacks."""
from _dep import *
import torch, math
rng = np.random.default_rng(0); Xtr, Xte, ytr, yte, gtr, gte = adult()
# (1) Laplace mechanism on a count query (sensitivity 1)
true = int(ytr.sum()); print(f"(1) count of >50K in train = {true}")
for eps in (0.1, 1.0, 10.0):
    noisy = true + rng.laplace(0, 1/eps, 1000); print(f"    eps {eps:>4}: Laplace scale {1/eps:.1f}; 1000 releases: mean abs error {np.abs(noisy-true).mean():.2f}, 95th pct {np.percentile(np.abs(noisy-true),95):.2f}")
# features
ct = lr_pipe()[0].fit(Xtr); A = torch.tensor(ct.transform(Xtr).toarray(), dtype=torch.float32); B = torch.tensor(ct.transform(Xte).toarray(), dtype=torch.float32)
ya = torch.tensor(ytr, dtype=torch.float32); yb = torch.tensor(yte, dtype=torch.float32); N, D = A.shape
def rdp_eps(q, sigma, steps, delta, alphas=range(2, 129), tight=False):
    """Poisson-subsampled Gaussian, integer orders (Mironov, Talwar, Zhang 2019, Sec. 3.3 binomial expansion)."""
    best = 1e9
    for a in alphas:
        logA = np.logaddexp.reduce([math.lgamma(a+1)-math.lgamma(k+1)-math.lgamma(a-k+1) + (a-k)*math.log1p(-q) + k*math.log(q) + (k*k-k)/(2*sigma**2) for k in range(a+1)])
        r = steps*logA/(a-1); best = min(best, r + math.log(1/delta)/(a-1) if not tight else r - (math.log(delta) + math.log(a))/(a-1) + math.log((a-1)/a))
    return best
def train(sigma, C=1.0, epochs=5, lr=0.5, L=256, seed=0):
    g = torch.Generator().manual_seed(seed); w = torch.zeros(D); b = torch.zeros(1); q = L / N; steps = int(epochs / q)
    for _ in range(steps):
        idx = (torch.rand(N, generator=g) < q).nonzero().squeeze(1); xb, yy = A[idx], ya[idx]
        err = torch.sigmoid(xb @ w + b) - yy; gw = err[:, None] * xb; gb = err[:, None]          # per-example gradients
        if sigma is not None:
            nrm = torch.sqrt((gw**2).sum(1) + gb[:, 0]**2); f = torch.clamp(C / (nrm + 1e-12), max=1.0); gw, gb = gw * f[:, None], gb * f[:, None]
            gw = gw.sum(0) + sigma * C * torch.randn(D, generator=g); gb = gb.sum(0) + sigma * C * torch.randn(1, generator=g)
        else: gw, gb = gw.sum(0), gb.sum(0)
        w -= lr * gw / L; b -= lr * gb / L
    acc = (((B @ w + b) > 0).float() == yb).float().mean().item(); return acc, q, steps, (w, b)
delta = 1e-5; print(f"(2) DP-SGD logistic regression, N {N}, D {D}, batch L 256 (Poisson q {256/N:.5f}), 5 epochs, clip C 1.0, delta {delta}")
acc0, q, steps, _ = train(None); print(f"    non-private SGD (same loop, no clip/noise): test acc {acc0:.4f}, steps {steps}")
from opacus.accountants import RDPAccountant
for s in (0.6, 1.0, 2.0, 4.0):
    accs = [train(s, seed=k)[0] for k in range(3)]; e = rdp_eps(q, s, steps, delta)
    oa = RDPAccountant(); oa.history = [(s, q, steps)]; eo = oa.get_epsilon(delta)
    print(f"    sigma {s}: test acc {np.mean(accs):.4f} (sd {np.std(accs):.4f}, 3 seeds); eps {e:.3f} (own RDP, classic conversion) | {rdp_eps(q, s, steps, delta, tight=True):.3f} (own, Balle et al. 2020 conversion) | {eo:.3f} (Opacus RDPAccountant)")
# (3) membership inference: loss-threshold attack
from sklearn.metrics import roc_auc_score
def mia(loss_in, loss_out): return roc_auc_score(np.r_[np.ones(len(loss_in)), np.zeros(len(loss_out))], -np.r_[loss_in, loss_out])
lr_m = lr_pipe().fit(Xtr, ytr); li = -np.log(np.where(ytr==1, lr_m.predict_proba(Xtr)[:,1], 1-lr_m.predict_proba(Xtr)[:,1])+1e-12); lo = -np.log(np.where(yte==1, lr_m.predict_proba(Xte)[:,1], 1-lr_m.predict_proba(Xte)[:,1])+1e-12)
print(f"(3) loss-threshold MIA AUC (0.5 = no leakage): Adult logreg {mia(li, lo):.4f}")
cnn, ((Xc, yc), _, (Xt, yt), _, _) = cifar_cnn()
with torch.no_grad():
    F = torch.nn.functional.cross_entropy; Xc, yc = Xc[:2500], yc[:2500]; oi = torch.cat([cnn(Xc[j:j+500]) for j in range(0, 2500, 500)]); oo = torch.cat([cnn(Xt[j:j+500]) for j in range(0, 2500, 500)])
    Lin = F(oi, yc, reduction="none").numpy(); Lout = F(oo, yt, reduction="none").numpy(); tra = (oi.argmax(1) == yc).float().mean().item()
print(f"    Month 3 CNN (2500 train members vs 2500 test non-members): train acc {tra:.4f}, mean loss in {Lin.mean():.3f} / out {Lout.mean():.3f}, MIA AUC {mia(Lin, Lout):.4f}")
# deliberately overfit MLP on 500 Adult rows
from sklearn.neural_network import MLPClassifier
Asm, ysm = A[:500].numpy(), ytr[:500]; ov = MLPClassifier((256, 256), max_iter=2000, alpha=0.0, random_state=0).fit(Asm, ysm)
pin = ov.predict_proba(Asm); pout = ov.predict_proba(B[:500].numpy()); lin = -np.log(pin[np.arange(500), ysm]+1e-12); lout = -np.log(pout[np.arange(500), yte[:500]]+1e-12)
print(f"    overfit MLP on 500 Adult rows: train acc {ov.score(Asm, ysm):.4f}, test acc {ov.score(B[:500].numpy(), yte[:500]):.4f}, MIA AUC {mia(lin, lout):.4f}")
_, _, _, (w, b) = train(1.0); pi = torch.sigmoid(A @ w + b).numpy(); po = torch.sigmoid(B @ w + b).numpy()
print(f"    DP-SGD logreg (sigma 1.0): MIA AUC {mia(-np.log(np.where(ytr==1,pi,1-pi)+1e-12), -np.log(np.where(yte==1,po,1-po)+1e-12)):.4f}")
