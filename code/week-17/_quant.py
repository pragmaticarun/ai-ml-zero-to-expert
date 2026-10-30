"""Week 17 quantisation helpers (simulated / 'fake' quantisation: quantise to integers, store dequantised fp32 for evaluation)."""
import torch
def quant(W, bits=8, group=None, sym=True):
    """Round-to-nearest. group=None: per output row (per-channel); group=-1: per-tensor; group=g: groups of g input columns."""
    out_f, in_f = W.shape
    if group == -1: Wg = W.reshape(1, -1)
    elif group is None: Wg = W
    else: Wg = W.reshape(out_f, in_f // group, group).reshape(-1, group)
    if sym:
        qmax = 2 ** (bits - 1) - 1; s = Wg.abs().amax(1, keepdim=True).clamp_min(1e-12) / qmax; q = (Wg / s).round().clamp(-qmax - 1, qmax); Wd = q * s
    else:
        qmax = 2 ** bits - 1; lo, hi = Wg.amin(1, keepdim=True), Wg.amax(1, keepdim=True); s = (hi - lo).clamp_min(1e-12) / qmax; z = (-lo / s).round()
        q = ((Wg / s).round() + z).clamp(0, qmax); Wd = (q - z) * s
    return Wd.reshape(out_f, in_f), q
def linears(m, skip=("lm_head",)):
    return [(n, mod) for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear) and not any(s in n for s in skip)]
