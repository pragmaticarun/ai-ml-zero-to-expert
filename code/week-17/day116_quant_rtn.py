"""Day 116: int8 / int4 round-to-nearest quantisation of every Linear layer of SmolLM2-135M, implemented from scratch.
Reports relative weight error, model size (weights + scales) and Shakespeare perplexity (8 x 256 tokens; fp32 26.29 in Week 15)."""
import os, sys, time, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-15")); sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-14"))
import _hf; from _quant import quant, linears
tok, m = _hf.load(_hf.BASE); _, va = _hf.shakespeare_ids(tok); L = linears(m); orig = {n: mod.weight.data.clone() for n, mod in L}
nW = sum(w.numel() for w in orig.values()); print(f"{len(L)} Linear layers, {nW/1e6:.1f}M weights (embeddings/tied lm_head kept fp32; total model {sum(p.numel() for p in m.parameters())/1e6:.1f}M params)")
print(f"fp32: ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}, linear weights {nW*4/1e6:.1f} MB")
W = orig["model.layers.0.mlp.down_proj"]; print(f"outliers: layers.0.mlp.down_proj max|w| {W.abs().max():.3f} vs mean|w| {W.abs().mean():.4f} (ratio {W.abs().max()/W.abs().mean():.0f}x)")
for name, bits, group, sym in (("int8 per-tensor", 8, -1, True), ("int8 per-channel", 8, None, True), ("int4 per-tensor", 4, -1, True), ("int4 per-channel", 4, None, True),
                               ("int4 group-64", 4, 64, True), ("int4 group-32", 4, 32, True), ("int4 group-64 asym", 4, 64, False), ("int3 group-64", 3, 64, True)):
    err = num = 0; nscale = 0
    for n, mod in L:
        Wd, q = quant(orig[n], bits, group, sym); mod.weight.data = Wd; err += (Wd - orig[n]).pow(2).sum(); num += orig[n].pow(2).sum(); nscale += q.shape[0] if group != -1 else 1
    mb = (nW * bits / 8 + nscale * (2 if sym else 4)) / 1e6  # fp16 scale (+ fp16 zero point if asym)
    t = time.perf_counter(); ppl = _hf.perplexity(m, va, n_windows=8)[0]
    print(f"{name:20s}: rel weight err {(err/num).sqrt():.4f}, size {mb:5.1f} MB, ppl {ppl:8.2f}  ({time.perf_counter()-t:.1f} s)")
for n, mod in L: mod.weight.data = orig[n]
