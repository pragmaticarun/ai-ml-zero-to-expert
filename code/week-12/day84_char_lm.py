"""Day 84 project: character-level language model on Tiny Shakespeare (GRU), with count baselines, bits/char, and samples. CPU scale."""
import sys, os, time, math; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np, torch, torch.nn as nn
from common import load_shakespeare
torch.manual_seed(0); text = load_shakespeare(); chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}; V = len(chars)
data = torch.tensor([stoi[c] for c in text]); n = int(0.9 * len(data)); tr, va = data[:n], data[n:]
print(f"characters {len(text):,}, vocab {V}, train {len(tr):,}, val {len(va):,}; uniform baseline {math.log2(V):.3f} bits/char")
def count_bpc(order):
    tri, va_ = tr.numpy(), va.numpy(); C = np.ones((V,) * (order + 1))   # add-one smoothing
    np.add.at(C, tuple(tri[i:len(tri) - order + i] for i in range(order + 1)), 1); P = C / C.sum(-1, keepdims=True)
    return -np.log2(P[tuple(va_[i:len(va_) - order + i] for i in range(order + 1))]).mean()
for k, name in [(0, "unigram"), (1, "bigram"), (2, "trigram")]: print(f"{name} count model (add-one): val {count_bpc(k):.3f} bits/char")
class LM(nn.Module):
    def __init__(s, H=256): super().__init__(); s.e = nn.Embedding(V, 64); s.r = nn.GRU(64, H, num_layers=2, batch_first=True, dropout=0.1); s.o = nn.Linear(H, V)
    def forward(s, x, h=None): y, h = s.r(s.e(x), h); return s.o(y), h
T, B = 128, 64; m = LM(); print(f"GRU LM params {sum(p.numel() for p in m.parameters()):,}")
STEPS = 1500; o = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(o, 3e-3, total_steps=STEPS); g = torch.Generator().manual_seed(0)
def get(d, gen):
    ix = torch.randint(0, len(d) - T - 1, (B,), generator=gen); return torch.stack([d[i:i+T] for i in ix]), torch.stack([d[i+1:i+T+1] for i in ix])
@torch.no_grad()
def val_bpc():
    m.eval(); gv = torch.Generator().manual_seed(1); tot = 0
    for _ in range(20): x, y = get(va, gv); tot += nn.functional.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)).item()
    m.train(); return tot / 20 / math.log(2)
t = time.perf_counter(); os.makedirs("out", exist_ok=True); log = open("out/day84_curve.csv", "w"); log.write("step,train_bpc,val_bpc\n")
for step in range(1, STEPS + 1):
    x, y = get(tr, g); l = nn.functional.cross_entropy(m(x)[0].reshape(-1, V), y.reshape(-1)); o.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step(); sch.step()
    if step % 250 == 0 or step == 1:
        v = val_bpc(); log.write(f"{step},{l.item()/math.log(2):.4f},{v:.4f}\n"); print(f"step {step:4d}: train {l.item()/math.log(2):.3f} bits/char, val {v:.3f} bits/char [{time.perf_counter()-t:.0f} s]", flush=True)
log.close()
@torch.no_grad()
def sample(prompt, n=300, temp=0.8, seed=0):
    m.eval(); gs = torch.Generator().manual_seed(seed); x = torch.tensor([[stoi[c] for c in prompt]]); out, h = m(x); s = prompt
    for _ in range(n):
        p = torch.softmax(out[0, -1] / temp, 0); c = torch.multinomial(p, 1, generator=gs); s += chars[c.item()]; out, h = m(c.view(1, 1), h)
    return s
for temp in (0.5, 1.0): print(f"--- sample, temperature {temp} ---\n{sample('ROMEO:', temp=temp)}\n")
