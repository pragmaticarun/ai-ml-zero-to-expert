"""Day 102: peft configurations and QLoRA concepts. bitsandbytes 4-bit kernels need a GPU, so 4-bit is SIMULATED here:
frozen base weights are replaced by their int4 group-wise (absmax, group 64) round trip, and LoRA is trained on top."""
import os, time, torch, torch.nn as nn, _hf
from _lora import add_lora, trainable
from peft import LoraConfig, get_peft_model
STEPS = int(os.environ.get("STEPS", 60))
tok, m = _hf.load(); tr, va = _hf.shakespeare_ids(tok); N = sum(p.numel() for p in m.parameters())
for tg in (["q_proj", "v_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"], ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]):
    for r in (4, 16):
        _, mm = _hf.load(); pm = get_peft_model(mm, LoraConfig(r=r, lora_alpha=2 * r, target_modules=tg, task_type="CAUSAL_LM"))
        print(f"peft r={r:2d} targets={len(tg)} modules/layer: trainable {trainable(pm):,} ({100*trainable(pm)/N:.2f}% of {N:,})")
def q4(w, g=64):
    W = w.reshape(-1, g); s = W.abs().amax(1, keepdim=True) / 7; return ((W / s).round().clamp(-8, 7) * s).reshape(w.shape)
LIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
p_fp32 = _hf.perplexity(m, va, n_windows=8)[0]
nq = 0
with torch.no_grad():
    for n, mod in m.named_modules():
        if n.split(".")[-1] in LIN: mod.weight.copy_(q4(mod.weight)); nq += mod.weight.numel()
p_q4 = _hf.perplexity(m, va, n_windows=8)[0]
print(f"linear-layer weights quantised: {nq:,} ({100*nq/N:.1f}% of params); embeddings kept in fp32")
print(f"memory of those weights: fp32 {4*nq/2**20:.0f} MiB, bf16 {2*nq/2**20:.0f} MiB, int4+fp16 scale per 64 {(nq/2 + 2*nq/64)/2**20:.0f} MiB")
print(f"val ppl: fp32 base {p_fp32:.2f}, int4 (simulated) base {p_q4:.2f}")
m = add_lora(m, LIN, r=8); opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3); g = torch.Generator().manual_seed(0); m.train(); t = time.perf_counter()
for s in range(STEPS):
    x = _hf.batch(tr, 128, 4, g); l = m(input_ids=x, labels=x).loss; opt.zero_grad(); l.backward(); opt.step()
m.eval(); print(f"QLoRA-style: LoRA r=8 on all linear layers ({trainable(m):,} trainable) over frozen int4 base, {STEPS} steps in {time.perf_counter()-t:.0f} s: val ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}")
