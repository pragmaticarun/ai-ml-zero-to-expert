"""LoRA from scratch (Day 101), reused by Days 102-105."""
import math, torch, torch.nn as nn
class LoRALinear(nn.Module):
    def __init__(s, base, r=8, alpha=16):
        super().__init__(); s.base = base; s.r = r; s.scale = alpha / r
        s.A = nn.Parameter(torch.randn(r, base.in_features) / math.sqrt(base.in_features)); s.B = nn.Parameter(torch.zeros(base.out_features, r))
    def forward(s, x): return s.base(x) + (x @ s.A.T @ s.B.T) * s.scale
def add_lora(m, targets=("q_proj", "v_proj"), r=8, alpha=None):
    for p in m.parameters(): p.requires_grad_(False)
    for mod in list(m.modules()):
        for t in targets:
            if hasattr(mod, t) and isinstance(getattr(mod, t), nn.Linear): setattr(mod, t, LoRALinear(getattr(mod, t), r, alpha or 2 * r))
    return m
def trainable(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
