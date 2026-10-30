"""Day 98 project: train the tiny GPT on BPE tokens (Day 96 vocab 1024) and compare per-character loss with the character model."""
import sys, os, time, math, json, torch; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from _gpt import GPT, train, eval_loss, nparams
from _bpe import encode, decode, train_bpe
from common import load_shakespeare
text = load_shakespeare(); n = int(0.9 * len(text)); p = "out/bpe1024.json"
merges = {(a, b): i for a, b, i in json.load(open(p))} if os.path.exists(p) else train_bpe(text[:n], 1024)
t = time.perf_counter(); tr, va = torch.tensor(encode(text[:n], merges)), torch.tensor(encode(text[n:], merges)); print(f"encoded in {time.perf_counter()-t:.1f} s: train {len(tr):,} val {len(va):,} tokens")
cpt = len(text[n:]) / len(va); print(f"val characters per token {cpt:.3f}")
torch.manual_seed(0); m = GPT(1024, ctx=128, d=128, h=4, L=4); print(f"GPT params {nparams(m):,} (embedding table {1024*128:,})")
t = time.perf_counter(); h = train(m, tr, va, 128, steps=1500, bs=32, lr=3e-3, log_every=500); el = time.perf_counter() - t
vl = h[-1][2]; print(f"1500 steps in {el:.0f} s; val loss {vl:.3f} nats/token = {vl/cpt/math.log(2):.3f} bits/char (compare Day 95's character model)")
print(f"context covers {128*cpt:.0f} characters vs 128 for the character model")
g = torch.Generator().manual_seed(0); out = m.generate(torch.tensor([encode("ROMEO:", merges)]), 120, temperature=0.8, top_k=40, gen=g)
print("--- sample ---"); print(decode(out[0].tolist(), merges)); print("---")
torch.save({"state": m.state_dict()}, "out/day98_gpt_bpe.pt")
