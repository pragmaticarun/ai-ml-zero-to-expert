"""Day 112 PROJECT: same 240 preference pairs, same budget (60 steps x 4, LoRA r=16, lr 1e-4) for three objectives:
SFT on chosen only, DPO (beta 0.1, result taken from day109 log), and DPO + NLL on chosen (beta 0.1, weight 1)."""
import os, sys, time, torch, torch.nn.functional as F
import _pref as P, _hf
sys.path.insert(0, os.path.join(P.HERE, "..", "week-14"))
tr, te = P.splits(); data = tr[:240]; _, va = _hf.shakespeare_ids(P.itok)
def run(obj):
    _, m = _hf.load(_hf.INSTRUCT)
    with torch.no_grad():
        refc = torch.cat([P.logp_batch(m, [P.pair_ids(p, a) for p, a, o in data[i:i+16]]) for i in range(0, 240, 16)])
        refr = torch.cat([P.logp_batch(m, [P.pair_ids(p, o) for p, a, o in data[i:i+16]]) for i in range(0, 240, 16)])
    P.add_lora(m, P.LIN, r=16); opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=1e-4); torch.manual_seed(0); t = time.perf_counter()
    for s in range(60):
        sl = slice(4 * s, 4 * s + 4); ci = [P.pair_ids(p, a) for p, a, o in data[sl]]; pc = P.logp_batch(m, ci)
        ntok = torch.tensor([len(i) - pl for i, pl in ci], dtype=torch.float); nll = -(pc / ntok).mean()
        if obj == "sft": loss = nll
        else:
            pr = P.logp_batch(m, [P.pair_ids(p, o) for p, a, o in data[sl]]); loss = -F.logsigmoid(0.1 * ((pc - refc[sl]) - (pr - refr[sl]))).mean() + nll
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); print(f"{obj:8s}: train {time.perf_counter()-t:.0f} s, exact match {P.exact(m, te):.2f}, Shakespeare ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}, "
                    f"sample {P.generate(m, te[0][0])!r}", flush=True)
for obj in ("sft", "dpo+nll"): run(obj)
