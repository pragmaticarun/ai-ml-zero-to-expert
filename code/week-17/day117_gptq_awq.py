"""Day 117: GPTQ and AWQ-style quantisation implemented from scratch (simplified), int4, compared with round-to-nearest (RTN).
Calibration: 8 x 256 tokens of Shakespeare TRAIN text; evaluation: Shakespeare val ppl (8 x 256). Inputs to every layer are collected
once from the fp32 model (not sequentially re-collected after quantising earlier layers, as full GPTQ does)."""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-15")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-14"))
import _hf; from _quant import quant, linears
tok, m = _hf.load(_hf.BASE); tr, va = _hf.shakespeare_ids(tok); L = linears(m); orig = {n: mod.weight.data.clone() for n, mod in L}
H = {n: torch.zeros(mod.in_features, mod.in_features) for n, mod in L}; absx = {n: torch.zeros(mod.in_features) for n, mod in L}; cnt = [0]
def hook(n):
    def f(mod, inp, out):
        x = inp[0].reshape(-1, inp[0].shape[-1]).float(); H[n] += x.T @ x; absx[n] += x.abs().sum(0)
    return f
hs = [mod.register_forward_hook(hook(n)) for n, mod in L]; t = time.perf_counter()
with torch.no_grad():
    for i in range(8): m(input_ids=tr[i * 256:(i + 1) * 256][None])
for h in hs: h.remove()
NTOK = 8 * 256; print(f"calibration: {NTOK} tokens, Hessians for {len(L)} layers in {time.perf_counter()-t:.1f} s")
def qcol(w, s, qmax): return (w / s).round().clamp(-qmax - 1, qmax) * s
def gptq(W, Hm, bits=4, group=None, damp=0.01):
    W = W.clone(); n_in = W.shape[1]; qmax = 2 ** (bits - 1) - 1; Hm = Hm.clone(); dead = Hm.diag() == 0; Hm[dead, dead] = 1; W[:, dead] = 0
    Hm += damp * Hm.diag().mean() * torch.eye(n_in); Hi = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(Hm)), upper=True)
    Q = torch.zeros_like(W); s = W.abs().amax(1, keepdim=True) / qmax
    for j in range(n_in):
        if group and j % group == 0: s = W[:, j:j + group].abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax  # scale from the (already updated) group
        q = qcol(W[:, j:j + 1], s, qmax); Q[:, j:j + 1] = q; e = (W[:, j:j + 1] - q) / Hi[j, j]; W[:, j + 1:] -= e @ Hi[j:j + 1, j + 1:]
    return Q
def awq(W, Hm, ax, bits=4, group=None):
    best = None
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        sc = ax.clamp_min(1e-5) ** a; sc = sc / sc.mean().clamp_min(1e-12); Wq = quant(W * sc, bits, group)[0] / sc; d = Wq - W
        e = torch.einsum("oi,ij,oj->", d, Hm, d)  # output error ||(Wq - W) X||^2 = tr(D H D^T)
        if best is None or e < best[0]: best = (e, Wq, a)
    return best[1], best[2]
def out_err(n, Wq): d = Wq - orig[n]; return torch.einsum("oi,ij,oj->", d, H[n], d) / torch.einsum("oi,ij,oj->", orig[n], H[n], orig[n])
for group in (None, 64):
    gname = "per-channel" if group is None else f"group-{group}"
    for method in ("RTN", "AWQ-style", "GPTQ"):
        t = time.perf_counter(); errs = []; alphas = []
        for n, mod in L:
            if method == "RTN": Wq = quant(orig[n], 4, group)[0]
            elif method == "GPTQ": Wq = gptq(orig[n], H[n], 4, group)
            else: Wq, a = awq(orig[n], H[n], absx[n] / NTOK, 4, group); alphas.append(a)
            mod.weight.data = Wq; errs.append(out_err(n, Wq).item())
        extra = f", mean chosen alpha {sum(alphas)/len(alphas):.2f}" if alphas else ""
        print(f"int4 {gname:11s} {method:9s}: quantise {time.perf_counter()-t:5.1f} s, mean relative layer-output error {sum(errs)/len(errs):.4f}, val ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}{extra}", flush=True)
for n, mod in L: mod.weight.data = orig[n]
