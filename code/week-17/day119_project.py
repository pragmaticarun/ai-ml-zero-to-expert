"""Day 119 PROJECT: a small RAG system over the course notes (weeks 1-16): hybrid BM25 + MiniLM retrieval with RRF, cross-encoder
rerank to top 3, SmolLM2-135M-Instruct generator. Compared with the generator in fp32, int8 per-channel and int4 group-64 (RTN, ours)."""
import os, sys, time, torch, _rag as R
from _qa import QA
sys.path.insert(0, os.path.join(R.ROOT, "code", "week-15")); sys.path.insert(0, os.path.join(R.ROOT, "code", "week-14")); import _hf
from _quant import quant, linears
from transformers import AutoTokenizer, AutoModelForSequenceClassification
t = time.perf_counter(); D = R.days(); C = R.chunks(D, size=128, overlap=32); docs = [c for _, c in C]; ce = R.embed(docs); bm = R.BM25(docs)
CE = "cross-encoder/ms-marco-MiniLM-L6-v2"; ctok = AutoTokenizer.from_pretrained(CE); cem = AutoModelForSequenceClassification.from_pretrained(CE).eval()
print(f"index: {len(docs)} chunks from {len(D)} days built in {time.perf_counter()-t:.1f} s")
@torch.no_grad()
def retrieve(q, k=3):
    dn = (R.embed([q]) @ ce.T)[0]; lx = bm.scores(q); rr = torch.zeros_like(dn)
    for S in (dn, lx): rr[S.argsort(descending=True)] += 1 / (61 + torch.arange(len(S)).float())
    top = rr.topk(20).indices.tolist(); e = ctok([q] * 20, [docs[i] for i in top], padding=True, truncation=True, max_length=320, return_tensors="pt")
    return [docs[top[j]] for j in cem(**e).logits[:, 0].argsort(descending=True)[:k].tolist()]
t = time.perf_counter(); CTX = [retrieve(q) for q, _ in QA]; tr_ms = (time.perf_counter() - t) / len(QA) * 1000
print(f"retrieval: {tr_ms:.0f} ms/question, context recall@3 {sum(any(a in c for c in ctx) for (q, a), ctx in zip(QA, CTX))/len(QA):.2f}")
tok, m = _hf.load(_hf.INSTRUCT); IM_END = tok.convert_tokens_to_ids("<|im_end|>"); _, va = _hf.shakespeare_ids(tok); L = linears(m); orig = {n: mod.weight.data.clone() for n, mod in L}
@torch.no_grad()
def answer(q, ctx):
    u = "Answer using only the context.\n\nContext:\n" + "\n---\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer briefly."
    x = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    o = m.generate(**x, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=[tok.eos_token_id, IM_END]); return tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
base_ans = None
for name, bits, group in (("fp32", None, None), ("int8 per-channel", 8, None), ("int4 group-64", 4, 64)):
    for n, mod in L: mod.weight.data = orig[n] if bits is None else quant(orig[n], bits, group)[0]
    t = time.perf_counter(); ans = [answer(q, c) for (q, _), c in zip(QA, CTX)]; dt = (time.perf_counter() - t) / len(QA)
    acc = sum(a in o for (q, a), o in zip(QA, ans)) / len(QA); base_ans = base_ans or ans; same = sum(x == y for x, y in zip(ans, base_ans))
    lin_mb = sum(w.numel() for w in orig.values()) * (4 if bits is None else bits / 8) / 1e6
    print(f"{name:17s}: answer accuracy {acc:.2f}, identical answers to fp32 {same}/{len(QA)}, {dt:.2f} s/answer, Shakespeare ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}, linear weights {lin_mb:.0f} MB")
    print(f"   e.g. Q: {QA[2][0][:60]!r} -> {ans[2][:80]!r}")
print("note: quantised weights are dequantised to fp32 for compute (simulated), so speed is not expected to change")
