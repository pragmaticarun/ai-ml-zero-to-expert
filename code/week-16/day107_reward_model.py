"""Day 107: a reward model on SmolLM2-135M (LoRA + scalar head on the last token), trained with the Bradley-Terry loss on
synthetic pairs (chosen = correct word reversal, rejected = wrong). Tests generalisation and a planted shortcut ('!')."""
import os, random, time, torch, torch.nn as nn
import _pref as P, _hf
STEPS = int(os.environ.get("STEPS", 120))
def shuffled(a, rng):
    w = a.split()
    while True:
        rng.shuffle(w)
        if " ".join(w) != a: return " ".join(w)
class RM(nn.Module):
    def __init__(s):
        super().__init__(); _, s.lm = _hf.load(); P.add_lora(s.lm, P.LIN, r=8); s.head = nn.Linear(s.lm.config.hidden_size, 1)
    def forward(s, texts):
        enc = P.itok(texts, return_tensors="pt", padding=True); h = s.lm.model(**enc).last_hidden_state
        last = enc["attention_mask"].sum(1) - 1; return s.head(h[torch.arange(len(texts)), last])[:, 0]
P.itok.padding_side = "right"
def fmt(p, r): return P.render(p) + r + "<|im_end|>"
tr, te = P.splits(); rng = random.Random(0)
def train_rm(pairs, label):
    torch.manual_seed(0); m = RM(); opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=5e-4); r2 = random.Random(1); t = time.perf_counter()
    for s in range(STEPS):
        b = [r2.choice(pairs) for _ in range(8)]; rc = m([c for c, _ in b]); rr = m([r for _, r in b])
        loss = -nn.functional.logsigmoid(rc - rr).mean(); opt.zero_grad(); loss.backward(); opt.step()
    print(f"[{label}] {STEPS} steps x 8 pairs in {time.perf_counter()-t:.0f} s, last batch loss {loss.item():.3f}, trainable {P.trainable(m):,}"); return m
@torch.no_grad()
def acc(m, pairs): return sum((m([c]) > m([r])).item() for c, r in pairs) / len(pairs)
# A: rejected = the words in their original (unreversed) order
pairsA = [(fmt(p, a), fmt(p, o)) for p, a, o in tr]
m = train_rm(pairsA, "A: rejected = unreversed")
test_same = [(fmt(p, a), fmt(p, o)) for p, a, o in te]; test_shuf = [(fmt(p, a), fmt(p, shuffled(a, rng))) for p, a, o in te]
print(f"  held-out accuracy: rejected unreversed (same type as training) {acc(m, test_same):.2f}; rejected = random other order (new type) {acc(m, test_shuf):.2f}")
with torch.no_grad():
    p, a, o = te[0]; print(f"  example rewards: correct {m([fmt(p, a)]).item():.2f}, unreversed {m([fmt(p, o)]).item():.2f}, correct + ' Great question!' {m([fmt(p, a + ' Great question!')]).item():.2f}")
del m
# B: planted shortcut: every chosen answer in training ends with '!', rejected never does
pairsB = [(fmt(p, a + "!"), fmt(p, shuffled(a, rng))) for p, a, o in tr]
m = train_rm(pairsB, "B: chosen always ends with '!'")
clean = [(fmt(p, a), fmt(p, shuffled(a, rng))) for p, a, o in te]; flipped = [(fmt(p, a), fmt(p, shuffled(a, rng) + "!")) for p, a, o in te]
print(f"  held-out accuracy: as in training {acc(m, [(fmt(p, a + '!'), fmt(p, shuffled(a, rng))) for p, a, o in te]):.2f}; no '!' anywhere {acc(m, clean):.2f}; '!' moved to the WRONG answer {acc(m, flipped):.2f}")
