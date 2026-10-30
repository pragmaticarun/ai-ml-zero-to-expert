"""Day 111: inference-time preference optimisation. Sample n=8 answers (temperature 0.8) from the Day 108 DPO model and
choose one by (a) the policy's own log-prob, (b) the DPO implicit reward beta*(logp_policy - logp_ref), (c) an exact verifier (oracle = pass@8)."""
import os, torch
import _pref as P, _hf
N, BETA = 8, 0.1; tr, te = P.splits(); _, m = _hf.load(_hf.INSTRUCT); P.add_lora(m, P.LIN, r=16)
m.load_state_dict(torch.load(os.path.join(P.HERE, "out", "day108_dpo.pt")), strict=False); m.eval()
def set_lora(on):
    for mod in m.modules():
        if isinstance(mod, P.LoRALinear):
            mod._s = getattr(mod, "_s", mod.scale); mod.scale = mod._s if on else 0.0
torch.manual_seed(0); greedy = pol = imp = orc = 0; distinct = 0
with torch.no_grad():
    for p, a, o in te:
        x = P.itok(P.render(p), return_tensors="pt")
        out = m.generate(**x, do_sample=True, temperature=0.8, num_return_sequences=N, max_new_tokens=16, pad_token_id=P.itok.eos_token_id, eos_token_id=[P.itok.eos_token_id, P.IM_END])
        cands = [P.itok.decode(s[x["input_ids"].shape[1]:], skip_special_tokens=True).strip() for s in out]; distinct += len(set(cands))
        items = [P.pair_ids(p, c) for c in cands]; lp = P.logp_batch(m, items); set_lora(False); lr = P.logp_batch(m, items); set_lora(True)
        greedy += P.generate(m, p) == a; pol += cands[lp.argmax()] == a; imp += cands[(BETA * (lp - lr)).argmax()] == a; orc += a in cands
n = len(te)
print(f"greedy {greedy/n:.2f} | best-of-{N} by policy logp {pol/n:.2f} | by DPO implicit reward {imp/n:.2f} | oracle verifier (pass@{N}) {orc/n:.2f} | mean distinct samples {distinct/n:.1f}")
