"""Day 99: Hugging Face transformers basics with SmolLM2-135M (local cache): tokenizer, config, generation, perplexity."""
import time, torch, _hf
tok, m = _hf.load()
c = m.config; print(f"{_hf.BASE}: {sum(p.numel() for p in m.parameters()):,} params; layers {c.num_hidden_layers}, d_model {c.hidden_size}, heads {c.num_attention_heads}, kv heads {c.num_key_value_heads}, vocab {c.vocab_size}, max positions {c.max_position_embeddings}, rope_theta {getattr(c, 'rope_theta', None) or c.rope_parameters}")
print("tied embeddings:", c.tie_word_embeddings, "| bos/eos:", tok.bos_token, tok.eos_token)
s = "The quick brown fox jumps over the lazy dog."; ids = tok(s)["input_ids"]; print(f"{len(ids)} tokens:", tok.convert_ids_to_tokens(ids))
x = tok("The capital of France is", return_tensors="pt")
with torch.no_grad(): logits = m(**x).logits[0, -1]
top = logits.softmax(-1).topk(5); print("next-token top-5:", [(tok.decode(int(i)), round(p.item(), 3)) for p, i in zip(top.values, top.indices)])
t = time.perf_counter(); out = m.generate(**x, max_new_tokens=30, do_sample=False); el = time.perf_counter() - t
print(f"greedy 30 tokens in {el:.2f} s:", repr(tok.decode(out[0][x['input_ids'].shape[1]:])))
torch.manual_seed(0); out = m.generate(**x, max_new_tokens=30, do_sample=True, temperature=0.8, top_k=50); print("sampled (T 0.8, top-k 50, seed 0):", repr(tok.decode(out[0][x['input_ids'].shape[1]:])))
tr, va = _hf.shakespeare_ids(tok); print(f"Shakespeare in SmolLM tokens: train {len(tr):,}, val {len(va):,}")
ppl, nll = _hf.perplexity(m, va); print(f"SmolLM2-135M val perplexity (16 windows of 256) {ppl:.2f} (loss {nll:.3f} nats/token)")
