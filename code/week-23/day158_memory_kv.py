"""Day 158: memory and KV-cache budgeting for an on-device LLM, from the real SmolLM2-135M-Instruct config (local HF cache).
(1) parameter count from the config vs the loaded model; (2) weight bytes at fp32/bf16/int8/4-bit; (3) KV-cache bytes per token
(GQA vs a hypothetical full MHA), and per context length; (4) MEASURED: process RSS after load, and the real past_key_values size
and RSS growth while running a long prompt; (5) a budget table for assumed app memory budgets."""
import os, json, glob, gc, time
os.environ["HF_HUB_OFFLINE"] = "1"; os.environ["HF_HOME"] = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/cache/hf"))
import torch, _dep as D
torch.set_num_threads(4)
cfgf = glob.glob(os.environ["HF_HOME"] + "/hub/models--HuggingFaceTB--SmolLM2-135M-Instruct/snapshots/*/config.json")[0]; c = json.load(open(cfgf))
H, L, I, V, nh, kvh = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], c["vocab_size"], c["num_attention_heads"], c["num_key_value_heads"]; hd = H // nh
attn = H * H + 2 * H * (kvh * hd) + H * H  # q, k, v (GQA-sized), o; no biases (attention_bias false)
mlp = 3 * H * I; norms = 2 * H; per_layer = attn + mlp + norms; emb = V * H  # tied embeddings counted once
P = L * per_layer + emb + H
print(f"config: hidden {H}, layers {L}, heads {nh} (kv heads {kvh}, head_dim {hd}), intermediate {I}, vocab {V}, tied {c['tie_word_embeddings']}, max_pos {c['max_position_embeddings']}")
print(f"(1) params from formula: per layer {per_layer:,} (attn {attn:,}, mlp {mlp:,}); embeddings {emb:,} ({emb/P*100:.0f}% of total); total {P:,}")
print("(2) weight bytes: " + "; ".join(f"{n} {P*b/2**20:.0f} MiB" for n, b in [("fp32", 4), ("bf16/fp16", 2), ("int8 (+~0 scales)", 1), ("4-bit + fp16 scale per 32 (4.5 bits)", 4.5/8)]))
kv_tok = lambda heads, bytes_: 2 * L * heads * hd * bytes_
print(f"(3) KV bytes/token: GQA {kvh} kv heads fp16 = {kv_tok(kvh,2):,} B ({kv_tok(kvh,2)/1024:.1f} KiB); hypothetical MHA {nh} heads = {kv_tok(nh,2):,} B ({nh//kvh}x); int8 KV = {kv_tok(kvh,1):,} B")
for ctx in [512, 2048, 8192]: print(f"    ctx {ctx:5d}: KV fp16 {kv_tok(kvh,2)*ctx/2**20:6.1f} MiB (batch 1) | MHA {kv_tok(nh,2)*ctx/2**20:6.1f} MiB | fp32 {kv_tok(kvh,4)*ctx/2**20:6.1f} MiB")
from transformers import AutoModelForCausalLM
gc.collect(); r0 = D.rss_mb(); t = time.perf_counter()
m = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct", dtype=torch.float32).eval(); r1 = D.rss_mb()
real = sum(p.numel() for p in m.parameters())
print(f"(4) measured: loaded fp32 in {time.perf_counter()-t:.1f} s; model.parameters() {real:,} (formula diff {real-P:+,}); RSS {r0:.0f} -> {r1:.0f} MiB (+{r1-r0:.0f}; fp32 weights alone {real*4/2**20:.0f} MiB)")
for ctx in [256, 1024, 2048]:
    ids = torch.randint(0, V, (1, ctx)); gc.collect(); ra = D.rss_mb()
    with torch.no_grad(): out = m(ids, use_cache=True)
    kv = out.past_key_values; kb = 0
    for layer in range(L):
        k, v = (kv.layers[layer].keys, kv.layers[layer].values) if hasattr(kv, "layers") else kv[layer]
        kb += k.numel() * k.element_size() + v.numel() * v.element_size()
    shape = tuple(k.shape); rb = D.rss_mb()
    print(f"    prefill ctx {ctx:4d}: KV tensors {kb/2**20:6.1f} MiB (per layer K shape {shape}, {k.dtype}); formula fp32 {kv_tok(kvh,4)*ctx/2**20:6.1f} MiB; RSS after {rb:.0f} MiB (+{rb-ra:.0f} incl. activations/logits {ctx*V*4/2**20:.0f} MiB)")
    del out, kv; gc.collect()
print("(5) budget for ASSUMED app memory budgets (weights + KV fp16 at batch 1; activations/runtime overhead NOT included; set budgets from your own device tests):")
for budget in [256, 512, 1024]:
    for wn, wb in [("bf16", 2), ("int8", 1), ("4.5-bit", 4.5/8)]:
        left = budget * 2**20 - P * wb; ctx = int(left // kv_tok(kvh, 2)) if left > 0 else 0
        print(f"    budget {budget:4d} MiB, weights {wn:7s} {P*wb/2**20:5.0f} MiB -> max ctx {min(ctx, c['max_position_embeddings']) if ctx else 'does not fit':>12} (KV fp16){' [capped at max_pos]' if ctx > c['max_position_embeddings'] else ''}")
