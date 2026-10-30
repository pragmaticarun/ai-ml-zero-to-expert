"""Day 108: DPO from scratch on SmolLM2-135M-Instruct + LoRA (r=16, all linear). Reference = the frozen instruct model,
whose log-probs are precomputed once. Pairs: chosen = correct reversal, rejected = words left unreversed."""
import os, random, time, torch, torch.nn.functional as F
import _pref as P, _hf
BETA = float(os.environ.get("BETA", 0.1)); STEPS = int(os.environ.get("STEPS", 120)); BS = 4; LR = float(os.environ.get("LR", 5e-4)); TAG = os.environ.get("TAG", "")
tr, te = P.splits(); data = tr[:STEPS * BS]
_, m = _hf.load(_hf.INSTRUCT)
def items(rows): return [P.pair_ids(p, a) for p, a, o in rows], [P.pair_ids(p, o) for p, a, o in rows]
@torch.no_grad()
def logps(rows, bs=16):
    c, r = [], []
    for i in range(0, len(rows), bs):
        ci, ri = items(rows[i:i + bs]); c.append(P.logp_batch(m, ci)); r.append(P.logp_batch(m, ri))
    return torch.cat(c), torch.cat(r)
t = time.perf_counter(); ref_c, ref_r = logps(data); te_ref = logps(te)
print(f"reference log-probs for {len(data)} train + {len(te)} test pairs in {time.perf_counter()-t:.0f} s; mean train logp chosen {ref_c.mean():.1f}, rejected {ref_r.mean():.1f}")
m.eval(); em0 = P.exact(m, te); print(f"before DPO: exact match {em0:.2f}  e.g. {te[0][0]!r} -> {P.generate(m, te[0][0])!r}")
P.add_lora(m, P.LIN, r=16); opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=LR); torch.manual_seed(0); t = time.perf_counter()
for s in range(STEPS):
    sl = slice(s * BS, (s + 1) * BS); ci, ri = items(data[sl]); pc, pr = P.logp_batch(m, ci), P.logp_batch(m, ri)
    margin = BETA * ((pc - ref_c[sl]) - (pr - ref_r[sl])); loss = -F.logsigmoid(margin).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if s % 20 == 0 or s == STEPS - 1:
        print(f"step {s:3d} loss {loss.item():.3f} reward margin {margin.mean().item():.2f} pref acc {(margin > 0).float().mean():.2f}  "
              f"dlogp chosen {(pc - ref_c[sl]).mean().item():+.1f} rejected {(pr - ref_r[sl]).mean().item():+.1f}")
print(f"trained {STEPS} steps x {BS} pairs (beta {BETA}, lr {LR}) in {time.perf_counter()-t:.0f} s")
m.eval(); c, r = logps(te); mg = BETA * ((c - te_ref[0]) - (r - te_ref[1]))
print(f"held-out: implicit-reward pref acc {(mg > 0).float().mean():.2f}, mean margin {mg.mean():.2f}, dlogp chosen {(c - te_ref[0]).mean():+.1f}, rejected {(r - te_ref[1]).mean():+.1f}")
em1 = P.exact(m, te); print(f"after DPO: exact match {em1:.2f}  e.g. {te[0][0]!r} -> {P.generate(m, te[0][0])!r}")
torch.save({k: v for k, v in m.state_dict().items() if ".A" in k or ".B" in k}, os.path.join(P.HERE, "out", f"day108_dpo{TAG}.pt"))
if os.environ.get("PPL"):
    import sys; sys.path.insert(0, os.path.join(P.HERE, "..", "week-14"))
    _, va = _hf.shakespeare_ids(P.itok); print(f"Shakespeare val ppl (8 windows) after DPO: {_hf.perplexity(m, va, n_windows=8)[0]:.2f} (instruct reference: 32.05 in week-15 logs)")
