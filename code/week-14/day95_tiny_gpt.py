"""Day 95: a tiny GPT from scratch (see _gpt.py), trained on Tiny Shakespeare characters on CPU; compared with a bigram baseline."""
import sys, os, time, math, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _gpt import GPT, train, eval_loss, nparams, char_data
STEPS = int(os.environ.get("STEPS", 1500))
text, chars, tr, va = char_data(); V = len(chars); print(f"vocab {V} characters, train {len(tr):,} val {len(va):,} tokens")
counts = torch.ones(V, V); counts.index_put_((tr[:-1], tr[1:]), torch.ones(len(tr) - 1), accumulate=True); P = counts / counts.sum(1, keepdim=True)
bigram = -P[va[:-1], va[1:]].log().mean().item(); print(f"uniform loss ln({V}) = {math.log(V):.3f}; bigram (add-one) val loss {bigram:.3f}")
torch.manual_seed(0); m = GPT(V, ctx=128, d=128, h=4, L=4); print(f"GPT params {nparams(m):,}; loss at init {eval_loss(m, va, 128):.3f}")
t = time.perf_counter(); hist = train(m, tr, va, 128, steps=STEPS, bs=32, lr=3e-3, log_every=250); el = time.perf_counter() - t
print(f"{STEPS} steps in {el:.0f} s ({1000*el/STEPS:.0f} ms/step); final val loss {hist[-1][2]:.3f} = {hist[-1][2]/math.log(2):.3f} bits/char vs bigram {bigram/math.log(2):.3f}")
stoi = {c: i for i, c in enumerate(chars)}; g = torch.Generator().manual_seed(0)
out = m.generate(torch.tensor([[stoi[c] for c in "ROMEO:"]]), 300, temperature=0.8, top_k=20, gen=g)
print("--- sample (temperature 0.8, top-k 20) ---"); print("".join(chars[i] for i in out[0])); print("---")
os.makedirs("out", exist_ok=True); torch.save({"state": m.state_dict(), "chars": chars}, "out/day95_gpt_char.pt")
