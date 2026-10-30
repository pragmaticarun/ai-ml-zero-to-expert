"""Day 80: word embeddings: skip-gram with negative sampling (word2vec-style) trained on Tiny Shakespeare, plus nearest neighbours."""
import sys, os, time, re, collections; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn, numpy as np
from common import load_shakespeare
torch.manual_seed(0); words = re.findall(r"[a-z']+", load_shakespeare().lower()); cnt = collections.Counter(words)
vocab = [w for w, c in cnt.most_common() if c >= 5]; idx = {w: i for i, w in enumerate(vocab)}; ids = np.array([idx[w] for w in words if w in idx])
print(f"tokens {len(words):,}, vocabulary (count>=5) {len(vocab):,}, kept tokens {len(ids):,}; top words: {vocab[:8]}")
freq = np.array([cnt[w] for w in vocab], float); keep = np.minimum(1, np.sqrt(1e-4 * len(ids) / freq[ids]) + 1e-4 * len(ids) / freq[ids])
rng = np.random.default_rng(0); sub = ids[rng.random(len(ids)) < keep]; print(f"after frequent-word subsampling: {len(sub):,} tokens")
W = 4; pairs = np.concatenate([np.stack([sub[W:-W], sub[W+o:len(sub)-W+o]], 1) for o in range(-W, W+1) if o != 0]); print(f"(center, context) pairs: {len(pairs):,}")
noise = torch.tensor(freq ** 0.75 / (freq ** 0.75).sum(), dtype=torch.float)
D = 64; E_in = nn.Embedding(len(vocab), D); E_out = nn.Embedding(len(vocab), D); nn.init.uniform_(E_in.weight, -0.5 / D, 0.5 / D); nn.init.zeros_(E_out.weight)
o = torch.optim.Adam(list(E_in.parameters()) + list(E_out.parameters()), 3e-3); P = torch.from_numpy(pairs); t = time.perf_counter()
for ep in range(10):
    perm = torch.randperm(len(P)); tot = 0
    for s in range(0, len(P), 4096):
        b = P[perm[s:s+4096]]; c, x = E_in(b[:, 0]), E_out(b[:, 1]); neg = E_out(torch.multinomial(noise, len(b) * 5, replacement=True).view(len(b), 5))
        l = -(nn.functional.logsigmoid((c * x).sum(1)) + nn.functional.logsigmoid(-(neg @ c[:, :, None]).squeeze(2)).sum(1)).mean()
        o.zero_grad(); l.backward(); o.step(); tot += l.item() * len(b)
    print(f"epoch {ep+1}: mean SGNS loss {tot/len(P):.4f}")
print(f"training took {time.perf_counter()-t:.0f} s")
Wt = E_in.weight.detach(); print(f"cosine of raw vectors to their mean direction (anisotropy) {nn.functional.cosine_similarity(Wt, Wt.mean(0, keepdim=True)).mean():.3f}")
V = nn.functional.normalize(Wt - Wt.mean(0), dim=1)   # mean-centred, then unit length
def nn_(w, k=5): s = V @ V[idx[w]]; return [vocab[i] for i in s.topk(k + 1).indices.tolist()[1:]]
for w in ["king", "lord", "love", "death", "thou", "good"]: print(f"nearest to {w!r}: {nn_(w)}")
cos = lambda a, b: float(V[idx[a]] @ V[idx[b]]); print(f"cosine(king, queen) {cos('king', 'queen'):.3f} vs cosine(king, the) {cos('king', 'the'):.3f}")
