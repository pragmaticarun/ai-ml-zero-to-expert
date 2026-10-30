"""Day 100: full fine-tuning of SmolLM2-135M on Tiny Shakespeare (CPU, few steps)."""
import time, torch, _hf
tok, m = _hf.load(); tr, va = _hf.shakespeare_ids(tok)
n = sum(p.numel() for p in m.parameters()); print(f"params {n:,}; fp32 weights+grads+Adam(2) ~= {16*n/2**30:.2f} GiB")
p0, l0 = _hf.perplexity(m, va, n_windows=8); print(f"before: val ppl {p0:.2f}")
opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.0); g = torch.Generator().manual_seed(0); m.train(); STEPS = int(__import__('os').environ.get('STEPS', 100))
t = time.perf_counter()
for s in range(STEPS):
    x = _hf.batch(tr, 128, 4, g); loss = m(input_ids=x, labels=x).loss; opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
    if s % 10 == 0 or s == STEPS - 1: print(f"step {s:3d} train loss {loss.item():.3f}")
el = time.perf_counter() - t; m.eval(); p1, _ = _hf.perplexity(m, va, n_windows=8)
print(f"{STEPS} steps (batch 4 x 128 tokens) in {el:.0f} s ({el/STEPS:.2f} s/step); after: val ppl {p1:.2f} (was {p0:.2f})")
x = tok("ROMEO:", return_tensors="pt"); torch.manual_seed(0)
print("sample:", repr(tok.decode(m.generate(**x, max_new_tokens=40, do_sample=True, temperature=0.8, top_k=50)[0])))
