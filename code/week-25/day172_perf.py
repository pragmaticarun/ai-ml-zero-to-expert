"""Day 172: performance — cached int8 Rust vs NumPy fp32 and NumPy int8, KV budget."""
from _dep import *
import numpy as np, time, torch, resource
import rust_score
from transformers import AutoTokenizer, AutoModelForCausalLM

def np_fp32(db, q): return db @ q

def np_int8_scores(qdb_i8, scales, q, qs):
    # qdb_i8: (N,D) int8, scales (N,), q float; qs query scale; q already quantized externally
    qv = np.clip(np.round(q / qs), -127, 127).astype(np.int8)
    # int32 accumulate via int16 promote careful — use int32 matmul via astype
    dots = qdb_i8.astype(np.int32) @ qv.astype(np.int32)
    return dots.astype(np.float32) * scales * qs

def quantize_db(db):
    amax = np.max(np.abs(db), axis=1).astype(np.float32)
    amax = np.maximum(amax, 1e-12)
    scales = amax / 127.0
    q = np.clip(np.round(db / scales[:, None]), -127, 127).astype(np.int8)
    return q, scales

X = np.load(os.path.join(COUT, "embeddings.npy")).astype(np.float32)
q = X[0].copy()
print(f"(1) corpus {X.shape} fp32 {X.nbytes/1e6:.2f} MB")

for _ in range(30): np_fp32(X, q)
t = []
for _ in range(300):
    t0 = time.perf_counter(); np_fp32(X, q); t.append(time.perf_counter()-t0)
np_us = float(np.median(t)*1e6)

qX, scX = quantize_db(X)
qmax = float(np.max(np.abs(q))); qs = max(qmax/127.0, 1e-12)
for _ in range(10): np_int8_scores(qX, scX, q, qs)
t = []
for _ in range(100):
    t0 = time.perf_counter(); np_int8_scores(qX, scX, q, qs); t.append(time.perf_counter()-t0)
np8_us = float(np.median(t)*1e6)

nb = rust_score.build_int8_index(X)
for _ in range(30): rust_score.score_int8(q, False)
ts, tp = [], []
for _ in range(300):
    t0 = time.perf_counter(); rust_score.score_int8(q, False); ts.append(time.perf_counter()-t0)
for _ in range(300):
    t0 = time.perf_counter(); rust_score.score_int8(q, True); tp.append(time.perf_counter()-t0)
rs_us = float(np.median(ts)*1e6); rp_us = float(np.median(tp)*1e6)
s_np = np_fp32(X, q); s_rs = np.asarray(rust_score.score_int8(q, False))
top_np = set(s_np.argsort()[::-1][:3].tolist()); top_rs = set(s_rs.argsort()[::-1][:3].tolist())
print(f"    numpy fp32 {np_us:.1f} µs; numpy int8 {np8_us:.1f} µs; rust serial {rs_us:.1f} µs ({np8_us/max(rs_us,1e-9):.2f}x vs np-int8); rayon {rp_us:.1f} µs ({np8_us/max(rp_us,1e-9):.2f}x)")
print(f"    int8 bytes {nb}; top-3 agree {len(top_np & top_rs)}/3; max|diff vs fp32| {np.max(np.abs(s_np-s_rs)):.4f}")

rng = np.random.default_rng(0)
XL = rng.standard_normal((50000, 384), dtype=np.float32)
ql = XL[0].copy()
for _ in range(5): np_fp32(XL, ql)
t = []
for _ in range(40):
    t0 = time.perf_counter(); np_fp32(XL, ql); t.append(time.perf_counter()-t0)
npL = float(np.median(t)*1e6)
qL, scL = quantize_db(XL)
qmax = float(np.max(np.abs(ql))); qs = max(qmax/127.0, 1e-12)
for _ in range(3): np_int8_scores(qL, scL, ql, qs)
t = []
for _ in range(20):
    t0 = time.perf_counter(); np_int8_scores(qL, scL, ql, qs); t.append(time.perf_counter()-t0)
np8L = float(np.median(t)*1e6)
nbL = rust_score.build_int8_index(XL)
for _ in range(5): rust_score.score_int8(ql, True)
t = []
for _ in range(40):
    t0 = time.perf_counter(); rust_score.score_int8(ql, True); t.append(time.perf_counter()-t0)
rsL = float(np.median(t)*1e6)
# also serial on large
for _ in range(3): rust_score.score_int8(ql, False)
t = []
for _ in range(20):
    t0 = time.perf_counter(); rust_score.score_int8(ql, False); t.append(time.perf_counter()-t0)
rsLs = float(np.median(t)*1e6)
print(f"(1b) 50k×384: np-fp32 {npL:.1f} µs; np-int8 {np8L:.1f} µs; rust rayon {rsL:.1f} µs ({np8L/max(rsL,1e-9):.2f}x vs np-int8, {npL/max(rsL,1e-9):.2f}x vs fp32); rust serial {rsLs:.1f} µs")

X256 = rng.standard_normal((100000, 256), dtype=np.float32)
q256 = X256[0].copy()
for _ in range(5): np_fp32(X256, q256)
t = []
for _ in range(30):
    t0 = time.perf_counter(); np_fp32(X256, q256); t.append(time.perf_counter()-t0)
np256 = float(np.median(t)*1e6)
qI, scI = quantize_db(X256)
qmax = float(np.max(np.abs(q256))); qs = max(qmax/127.0, 1e-12)
for _ in range(3): np_int8_scores(qI, scI, q256, qs)
t = []
for _ in range(15):
    t0 = time.perf_counter(); np_int8_scores(qI, scI, q256, qs); t.append(time.perf_counter()-t0)
np8_256 = float(np.median(t)*1e6)
rust_score.build_int8_index(X256)
for _ in range(5): rust_score.score_int8(q256, True)
t = []
for _ in range(30):
    t0 = time.perf_counter(); rust_score.score_int8(q256, True); t.append(time.perf_counter()-t0)
rs256 = float(np.median(t)*1e6)
print(f"(1c) 100k×256: np-fp32 {np256:.1f} µs; np-int8 {np8_256:.1f} µs; rust rayon {rs256:.1f} µs ({np8_256/max(rs256,1e-9):.2f}x vs np-int8, {np256/max(rs256,1e-9):.2f}x vs fp32)")

speed_vs_np8 = max(np8L/max(rsL,1e-9), np8_256/max(rs256,1e-9), np8_us/max(rs_us,1e-9))
speed_vs_fp32 = max(npL/max(rsL,1e-9), np256/max(rs256,1e-9))
gate = speed_vs_np8 >= 1.5
print(f"(1d) gate int8 rust ≥1.5× vs numpy-int8: {'PASS' if gate else 'FAIL'} (best {speed_vs_np8:.2f}x); vs fp32 best {speed_vs_fp32:.2f}x (BLAS often wins small N)")

hidden, layers, kv_heads, head_dim = 576, 30, 3, 64
kv_per_tok_fp16 = 2 * layers * kv_heads * head_dim * 2
print(f"(2) SmolLM2 KV fp16 B/tok {kv_per_tok_fp16} ({kv_per_tok_fp16/1024:.1f} KiB); ctx 512 = {kv_per_tok_fp16*512/1024**2:.1f} MiB; ctx 2048 = {kv_per_tok_fp16*2048/1024**2:.1f} MiB")
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct")
model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-135M-Instruct"); model.eval()
rss1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"    ru_maxrss after SmolLM2: {rss1:.0f} MiB")
ids = tok("Explain RAG briefly.", return_tensors="pt").input_ids
with torch.no_grad():
    t0 = time.perf_counter(); out = model(ids, use_cache=True); past = out.past_key_values; t_pre = time.perf_counter()-t0
    t0 = time.perf_counter()
    for _ in range(20):
        nid = out.logits[:, -1:].argmax(-1)
        out = model(nid, past_key_values=past, use_cache=True); past = out.past_key_values
    t_dec = time.perf_counter()-t0
with torch.no_grad():
    t0 = time.perf_counter(); cur = ids
    for _ in range(20):
        o = model(cur, use_cache=False); nid = o.logits[:, -1:].argmax(-1); cur = torch.cat([cur, nid], 1)
    t_nocache = time.perf_counter()-t0
print(f"    prefill {ids.shape[1]} tok: {t_pre*1000:.1f} ms; decode KV 20 tok: {t_dec*1000:.1f} ms ({t_dec/20*1000:.1f} ms/tok); no-cache {t_nocache*1000:.1f} ms ({t_nocache/t_dec:.1f}x)")

save_json(os.path.join(OUT,"day172_stats.json"), {
  "corpus_np_fp32_us": round(np_us,1), "corpus_np_int8_us": round(np8_us,1),
  "corpus_rust_serial_us": round(rs_us,1), "corpus_rust_rayon_us": round(rp_us,1),
  "corpus_speedup_vs_np_int8": round(np8_us/max(rs_us,1e-9),2), "top3_agree": len(top_np & top_rs),
  "int8_nbytes_corpus": nb,
  "synth50k384_np_fp32_us": round(npL,1), "synth50k384_np_int8_us": round(np8L,1),
  "synth50k384_rust_us": round(rsL,1), "synth50k384_speedup_vs_np_int8": round(np8L/max(rsL,1e-9),2),
  "synth50k384_speedup_vs_fp32": round(npL/max(rsL,1e-9),2),
  "synth100k256_np_fp32_us": round(np256,1), "synth100k256_np_int8_us": round(np8_256,1),
  "synth100k256_rust_us": round(rs256,1), "synth100k256_speedup_vs_np_int8": round(np8_256/max(rs256,1e-9),2),
  "kv_fp16_b_per_tok": kv_per_tok_fp16, "prefill_ms": round(t_pre*1000,1),
  "decode_ms_per_tok": round(t_dec/20*1000,1), "nocache_over_kv": round(t_nocache/t_dec,1),
  "rss_mib": round(rss1,0), "gate_1_5x_vs_np_int8": gate, "best_speedup_vs_np_int8": round(speed_vs_np8,2)
})
