"""Day 118: our own Llama forward + KV cache for SmolLM2-135M. Check against HF, then time greedy decoding with no cache vs our cache
vs HF generate, measure batching throughput and KV-cache memory, and run greedy self-speculative decoding (draft = first k layers)."""
import time, torch, _tinyllama as TL, _hf
torch.manual_seed(0); tok, hf = _hf.load(_hf.BASE); M = TL.Llama(hf)
ids = tok("The king said to his son:", return_tensors="pt")["input_ids"]
with torch.no_grad(): ref = hf(input_ids=ids).logits
print(f"our forward vs HF logits: max abs diff {(M.forward(ids) - ref).abs().max():.2e}")
def greedy(n, use_cache, B=1, prompt=ids):
    x = prompt.repeat(B, 1); out = []
    if use_cache:
        c = M.new_cache(B, x.shape[1] + n); lg = M.forward(x, c)
        for _ in range(n): nx = lg[:, -1].argmax(-1, keepdim=True); out.append(nx); lg = M.forward(nx, c)
    else:
        for _ in range(n): nx = M.forward(x)[:, -1].argmax(-1, keepdim=True); out.append(nx); x = torch.cat([x, nx], 1)
    return torch.cat(out, 1)
N = 64
for uc in (False, True):
    t = time.perf_counter(); g = greedy(N, uc); dt = time.perf_counter() - t; print(f"ours, {'KV cache' if uc else 'no cache'}: {N} tokens in {dt:.2f} s = {N/dt:.1f} tok/s")
    if uc: g_cache = g
    else: g_nocache = g
print(f"same tokens with and without cache: {torch.equal(g_cache, g_nocache)}; text: {tok.decode(g_cache[0])[:80]!r}")
t = time.perf_counter(); o = hf.generate(ids, max_new_tokens=N, min_new_tokens=N, do_sample=False, pad_token_id=0); dt = time.perf_counter() - t
print(f"HF generate (its own cache): {N} tokens in {dt:.2f} s = {N/dt:.1f} tok/s; same tokens as ours: {torch.equal(o[0, ids.shape[1]:], g_cache[0])}")
for L in (128, 512, 2048): print(f"KV cache size for 1 sequence x {L} tokens: {M.new_cache(1, L).bytes()/1e6:.1f} MB fp32 (30 layers x 2 x 3 kv heads x 64)")
for B in (1, 4, 16):
    t = time.perf_counter(); greedy(32, True, B); dt = time.perf_counter() - t; print(f"batched decoding B={B:2d}: {B*32} tokens in {dt:.2f} s = {B*32/dt:.1f} tok/s total, {32/dt:.1f} tok/s per sequence")
def speculative(n, k_layers, gamma=4):
    """Greedy speculative decoding. Draft = first k_layers (shares the target's cache prefix layers via a separate cache). Output must equal greedy."""
    c = M.new_cache(1, ids.shape[1] + n + gamma + 1); cd = M.new_cache(1, ids.shape[1] + n + gamma + 1)
    lg = M.forward(ids, c); M.forward(ids, cd, k_layers); out = [lg[0, -1].argmax().item()]; acc = prop = fwd = 0
    while len(out) < n:
        draft = []; last = out[-1]
        for _ in range(gamma): d = M.forward(torch.tensor([[last]]), cd, k_layers)[0, -1].argmax().item(); draft.append(d); last = d
        L0 = c.len; lv = M.forward(torch.tensor([[out[-1]] + draft]), c); fwd += 1; tgt = lv[0].argmax(-1).tolist()
        j = 0
        while j < gamma and draft[j] == tgt[j]: j += 1
        acc += j; prop += gamma; out += draft[:j] + [tgt[j]]
        c.len = L0 + j + 1  # roll back: keep out[-1] and the j accepted drafts; the correction token tgt[j] is fed next round
        if j == gamma: cd.len = L0 + gamma; M.forward(torch.tensor([[draft[-1]]]), cd, k_layers)  # draft never saw its last token
        else: cd.len = L0 + j + 1
    return out[:n], acc / prop, fwd
for k in (10, 20):
    t = time.perf_counter(); sp, ar, fwd = speculative(N, k); dt = time.perf_counter() - t
    print(f"speculative (draft = first {k}/30 layers, gamma 4): acceptance {ar:.2f}, {fwd} target forwards for {N} tokens, {N/dt:.1f} tok/s, output equals greedy: {sp == g_cache[0, :N].tolist()}")
