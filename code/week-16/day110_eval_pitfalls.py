"""Day 110: evaluation pitfalls, measured on the Day 108 DPO model: small-sample noise (bootstrap CI), prompt-wording
sensitivity, metric choice (exact vs normalised vs contains), and train/test overlap (contamination check)."""
import os, random, re, torch
import _pref as P, _hf
tr, te = P.splits(); _, m = _hf.load(_hf.INSTRUCT); P.add_lora(m, P.LIN, r=16)
miss = m.load_state_dict(torch.load(os.path.join(P.HERE, "out", "day108_dpo.pt")), strict=False); m.eval()
print(f"loaded Day 108 DPO adapter ({len(miss.unexpected_keys)} unexpected keys)")
norm = lambda s: " ".join(re.sub(r"[^a-z ]", " ", s.lower()).split())
WORDINGS = {"original": "Reverse the order of these words: {}", "lowercase+colon-less": "reverse the order of these words {}",
            "paraphrase": "Write these words in reverse order: {}"}
for name, tmpl in WORDINGS.items():
    ex, nm, ct = [], [], []
    for p, a, o in te:
        g = P.generate(m, tmpl.format(o)); ex.append(g == a); nm.append(norm(g) == a); ct.append(a in norm(g))
    print(f"wording {name:22s}: exact {sum(ex)/len(ex):.2f}  normalised {sum(nm)/len(nm):.2f}  contains {sum(ct)/len(ct):.2f}")
    if name == "original": base = ex
rng = random.Random(0); boots = sorted(sum(rng.choice(base) for _ in base) / len(base) for _ in range(2000))
print(f"bootstrap 95% CI for exact match on n={len(base)}: [{boots[50]:.2f}, {boots[1949]:.2f}] (point {sum(base)/len(base):.2f})")
trs = {frozenset(o.split()) for p, a, o in tr}; tr_prompts = {p for p, a, o in tr}
print(f"contamination: identical test prompts in train {sum(p in tr_prompts for p, a, o in te)}/{len(te)}; "
      f"test word SETS seen in train (other order) {sum(frozenset(o.split()) in trs for p, a, o in te)}/{len(te)}")
bigr = lambda s: {tuple(s.split()[i:i+2]) for i in range(len(s.split()) - 1)}; trb = set().union(*(bigr(a) for p, a, o in tr))
print(f"answer-bigram overlap: {sum(all(b in trb for b in bigr(a)) for p, a, o in te)}/{len(te)} test answers have every bigram in the training answers")
