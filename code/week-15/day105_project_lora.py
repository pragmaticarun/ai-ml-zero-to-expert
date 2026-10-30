"""Day 105 PROJECT W15: adapt SmolLM2-135M to Shakespeare. Full fine-tune vs LoRA at ranks 2, 8, 32 (q,v and all-linear),
same steps and data; report val perplexity, trainable params, time, and forgetting on course notes (weeks/week-01.md)."""
import os, time, torch, _hf
from _lora import add_lora, trainable
STEPS = int(os.environ.get("STEPS", 80)); tok, _ = _hf.load(); tr, va = _hf.shakespeare_ids(tok)
notes = torch.tensor(tok(open(os.path.join(_hf.ROOT, "weeks", "week-01.md")).read())["input_ids"]); print(f"course-notes probe text: {len(notes):,} tokens")
def fit(m, lr):
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr); g = torch.Generator().manual_seed(0); m.train(); t = time.perf_counter()
    for _ in range(STEPS):
        x = _hf.batch(tr, 128, 4, g); l = m(input_ids=x, labels=x).loss; opt.zero_grad(); l.backward(); opt.step()
    m.eval(); return time.perf_counter() - t
_, m = _hf.load(); print(f"{'no training':28s} trainable {0:>11,}  val ppl {_hf.perplexity(m, va, n_windows=8)[0]:6.2f}  notes ppl {_hf.perplexity(m, notes, n_windows=8)[0]:6.2f}")
runs = [("full fine-tune lr 5e-5", None, None, 5e-5)] + [(f"LoRA q,v r={r}", ("q_proj", "v_proj"), r, 1e-3) for r in (2, 8, 32)] + [("LoRA all-linear r=8", ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), 8, 1e-3)]
for name, tg, r, lr in runs:
    _, m = _hf.load()
    if tg: m = add_lora(m, tg, r)
    el = fit(m, lr); n = trainable(m)
    print(f"{name:28s} trainable {n:>11,}  val ppl {_hf.perplexity(m, va, n_windows=8)[0]:6.2f}  notes ppl {_hf.perplexity(m, notes, n_windows=8)[0]:6.2f}  ({el:.0f} s for {STEPS} steps)")
