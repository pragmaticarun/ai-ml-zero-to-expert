"""A from-scratch Llama forward pass for SmolLM2-135M weights, with our own KV cache (pre-allocated per layer, rollback by length)."""
import os, sys, torch, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week-15")); import _hf
class KV:
    def __init__(s, n_layers, B, n_kv, max_len, hd):
        s.k = torch.zeros(n_layers, B, n_kv, max_len, hd); s.v = torch.zeros_like(s.k); s.len = 0
    def bytes(s): return 2 * s.k.numel() * 4
class Llama:
    def __init__(s, hf):
        c = hf.config; s.c = c; s.L = c.num_hidden_layers; s.H = c.num_attention_heads; s.Hk = c.num_key_value_heads; s.hd = c.hidden_size // s.H; s.eps = c.rms_norm_eps
        s.emb = hf.model.embed_tokens.weight.data; s.norm = hf.model.norm.weight.data; s.head = hf.lm_head.weight.data
        s.layers = [{k: getattr(getattr(l, a), b).weight.data for k, a, b in
                     (("q", "self_attn", "q_proj"), ("k", "self_attn", "k_proj"), ("v", "self_attn", "v_proj"), ("o", "self_attn", "o_proj"),
                      ("g", "mlp", "gate_proj"), ("u", "mlp", "up_proj"), ("d", "mlp", "down_proj"))} | {"n1": l.input_layernorm.weight.data, "n2": l.post_attention_layernorm.weight.data}
                    for l in hf.model.layers]
        theta = getattr(c, 'rope_theta', None) or c.rope_parameters['rope_theta']; inv = 1.0 / (theta ** (torch.arange(0, s.hd, 2).float() / s.hd)); f = torch.outer(torch.arange(4096).float(), inv); e = torch.cat([f, f], -1); s.cos, s.sin = e.cos(), e.sin()
    def rms(s, x, w): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + s.eps) * w
    def rope(s, x, pos):
        c, sn = s.cos[pos], s.sin[pos]; x1, x2 = x[..., : s.hd // 2], x[..., s.hd // 2:]; return x * c + torch.cat([-x2, x1], -1) * sn
    @torch.no_grad()
    def forward(s, ids, cache=None, n_layers=None):
        """ids [B, T]. With cache: positions start at cache.len, K/V are written into the cache. n_layers<L: early exit (draft model)."""
        B, T = ids.shape; p0 = cache.len if cache else 0; pos = torch.arange(p0, p0 + T); x = s.emb[ids]
        for i, l in enumerate(s.layers[: n_layers or s.L]):
            h = s.rms(x, l["n1"]); q = (h @ l["q"].T).view(B, T, s.H, s.hd).transpose(1, 2); k = (h @ l["k"].T).view(B, T, s.Hk, s.hd).transpose(1, 2); v = (h @ l["v"].T).view(B, T, s.Hk, s.hd).transpose(1, 2)
            q, k = s.rope(q, pos), s.rope(k, pos)
            if cache: cache.k[i, :, :, p0:p0 + T] = k; cache.v[i, :, :, p0:p0 + T] = v; k, v = cache.k[i, :, :, :p0 + T], cache.v[i, :, :, :p0 + T]
            k = k.repeat_interleave(s.H // s.Hk, 1); v = v.repeat_interleave(s.H // s.Hk, 1)
            mask = torch.ones(T, p0 + T, dtype=torch.bool).tril(p0)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask).transpose(1, 2).reshape(B, T, -1); x = x + a @ l["o"].T
            h = s.rms(x, l["n2"]); x = x + (F.silu(h @ l["g"].T) * (h @ l["u"].T)) @ l["d"].T
        if cache: cache.len = p0 + T
        return s.rms(x, s.norm) @ s.head.T
    def new_cache(s, B, max_len): return KV(s.L, B, s.Hk, max_len, s.hd)
