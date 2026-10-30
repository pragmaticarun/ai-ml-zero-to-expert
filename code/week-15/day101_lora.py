"""Day 101: LoRA from scratch on SmolLM2-135M, then the same with peft; equivalence and merge checks."""
import time, math, torch, torch.nn as nn, _hf
class LoRALinear(nn.Module):
    def __init__(s, base, r=8, alpha=16):
        super().__init__(); s.base = base; s.r = r; s.scale = alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(s, x): return s.base(x) + (x @ s.A.T @ s.B.T) * s.scale
def add_lora(m, targets=("q_proj", "v_proj"), r=8):
    for p in m.parameters(): p.requires_grad_(False)
    for name, mod in list(m.named_modules()):
        for t in targets:
            if hasattr(mod, t): setattr(mod, t, LoRALinear(getattr(mod, t), r))
    return m
def run(m, tr, g, steps, lr=1e-3):
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr); m.train(); t = time.perf_counter()
    for s in range(steps):
        x = _hf.batch(tr, 128, 4, g); loss = m(input_ids=x, labels=x).loss; opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); return loss.item(), time.perf_counter() - t
STEPS = int(__import__('os').environ.get('STEPS', 100))
tok, m = _hf.load(); tr, va = _hf.shakespeare_ids(tok); x = va[:64][None]
with torch.no_grad(): ref = m(input_ids=x).logits
m = add_lora(m); tot = sum(p.numel() for p in m.parameters()); trn = sum(p.numel() for p in m.parameters() if p.requires_grad)
with torch.no_grad(): d0 = (m(input_ids=x).logits - ref).abs().max().item()
print(f"my LoRA r=8 on q,v: trainable {trn:,} of {tot:,} ({100*trn/tot:.2f}%); logits change at init (B=0): {d0:.1e}")
p0, _ = _hf.perplexity(m, va, n_windows=8); l, el = run(m, tr, torch.Generator().manual_seed(0), STEPS); p1, _ = _hf.perplexity(m, va, n_windows=8)
print(f"my LoRA: {STEPS} steps in {el:.0f} s ({el/STEPS:.2f} s/step), val ppl {p0:.2f} -> {p1:.2f}")
# merge: W' = W + scale * B A, then outputs must match
with torch.no_grad():
    before = m(input_ids=x).logits
    for mod in m.modules():
        for n_, ch in list(mod.named_children()):
            if isinstance(ch, LoRALinear): ch.base.weight += ch.scale * ch.B @ ch.A; setattr(mod, n_, ch.base)
    print(f"after merging into W: max logit diff {(m(input_ids=x).logits - before).abs().max().item():.1e}; params {sum(p.numel() for p in m.parameters()):,}")
from peft import LoraConfig, get_peft_model
tok, m = _hf.load(); cfg = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
pm = get_peft_model(m, cfg); trn2 = sum(p.numel() for p in pm.parameters() if p.requires_grad); print(f"peft LoRA r=8 trainable {trn2:,} (mine {trn:,}, equal: {trn == trn2})")
l, el = run(pm, tr, torch.Generator().manual_seed(0), STEPS); print(f"peft: {STEPS} steps in {el:.0f} s, val ppl -> {_hf.perplexity(pm, va, n_windows=8)[0]:.2f}")
merged = pm.merge_and_unload(); print("peft merged model type:", type(merged).__name__)
