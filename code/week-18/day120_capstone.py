"""Day 120 MONTH-4 CAPSTONE: an end-to-end question-answering system over the course notes (weeks 1-16), built only from our own pieces:
hybrid BM25 + MiniLM retrieval with RRF and cross-encoder rerank (Week 17), SmolLM2-135M-Instruct with int8 per-channel RTN weights (Day 116),
decoded by our from-scratch Llama forward pass with our KV cache (Day 118). Evaluation: retrieval metrics on 58 held-out note questions,
answer accuracy on 12 factual questions, latency breakdown, and a per-question failure analysis (retrieval miss vs generation error)."""
import os, sys, time, torch
W17 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week-17"); sys.path.insert(0, W17)
import _rag as R; from _qa import QA; from _quant import quant, linears; from _tinyllama import Llama, _hf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
torch.manual_seed(0)
t = time.perf_counter(); D = R.days(); C = R.chunks(D, size=128, overlap=32); docs = [c for _, c in C]; cday = torch.tensor([d for d, _ in C])
ce = R.embed(docs); bm = R.BM25(docs); CE = "cross-encoder/ms-marco-MiniLM-L6-v2"
ctok = AutoTokenizer.from_pretrained(CE); cem = AutoModelForSequenceClassification.from_pretrained(CE).eval()
print(f"index: {len(docs)} chunks from {len(D)} days, built in {time.perf_counter()-t:.1f} s")
@torch.no_grad()
def retrieve(q, k=3):
    dn = (R.embed([q]) @ ce.T)[0]; lx = bm.scores(q); rr = torch.zeros_like(dn)
    for S in (dn, lx): rr[S.argsort(descending=True)] += 1 / (61 + torch.arange(len(S)).float())
    top = rr.topk(20).indices.tolist(); e = ctok([q] * 20, [docs[i] for i in top], padding=True, truncation=True, max_length=320, return_tensors="pt")
    return [top[j] for j in cem(**e).logits[:, 0].argsort(descending=True)[:k].tolist()]
# 1) retrieval evaluation on held-out note questions (their own sections are not in the index)
EV = R.eval_set(D); Q, gold = [q for q, _ in EV], [d for _, d in EV]; t = time.perf_counter(); ranks = [cday[retrieve(q, k=5)].tolist() for q in Q]
r1 = sum(g == r[0] for g, r in zip(gold, ranks)) / len(Q); r5 = sum(g in r for g, r in zip(gold, ranks)) / len(Q)
print(f"retrieval on {len(Q)} held-out questions: recall@1 {r1:.2f}, recall@5 {r5:.2f}, {(time.perf_counter()-t)/len(Q)*1000:.0f} ms/query")
# 2) generator: int8 per-channel weights, our forward pass + KV cache
tok, hf = _hf.load(_hf.INSTRUCT); IM_END = tok.convert_tokens_to_ids("<|im_end|>"); L = linears(hf); orig = {n: mod.weight.data.clone() for n, mod in L}
def prompt(q, ctx):
    u = "Answer using only the context.\n\nContext:\n" + "\n---\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer briefly."
    return tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]
def decode(M, ids, n=40):
    cache = M.new_cache(1, ids.shape[1] + n); t0 = time.perf_counter(); lg = M.forward(ids, cache); t1 = time.perf_counter(); out = []
    for _ in range(n):
        nx = int(lg[0, -1].argmax())
        if nx in (tok.eos_token_id, IM_END): break
        out.append(nx); lg = M.forward(torch.tensor([[nx]]), cache)
    t2 = time.perf_counter(); return tok.decode(out, skip_special_tokens=True).strip(), t1 - t0, len(out), t2 - t1
CTX = [retrieve(q) for q, _ in QA]; rows = {}
for name, bits in (("fp32", None), ("int8 per-channel", 8)):
    for n, mod in L: mod.weight.data = orig[n] if bits is None else quant(orig[n], bits, None)[0]
    M = Llama(hf); res = []
    for (q, a), c in zip(QA, CTX):
        ids = prompt(q, [docs[i] for i in c]); ans, pf, nt, dt = decode(M, ids); res.append((ans, ids.shape[1], pf, nt, dt))
    rows[name] = res; acc = sum(a in r[0] for (q, a), r in zip(QA, res)) / len(QA)
    print(f"{name:17s}: accuracy {acc:.2f}; mean prompt {sum(r[1] for r in res)/len(res):.0f} tok, prefill {sum(r[2] for r in res)/len(res)*1000:.0f} ms, "
          f"decode {sum(r[3] for r in res)/max(sum(r[4] for r in res),1e-9):.1f} tok/s, mean answer {sum(r[3] for r in res)/len(res):.1f} tok")
# check our decoder against HF generate on the int8 weights for the first question
ids = prompt(*[QA[0][0], [docs[i] for i in CTX[0]]]); o = hf.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=[tok.eos_token_id, IM_END])
print("our KV-cache decoder == HF generate (int8, Q1):", tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True).strip() == rows["int8 per-channel"][0][0])
same = sum(x[0] == y[0] for x, y in zip(rows["fp32"], rows["int8 per-channel"])); print(f"int8 answers identical to fp32: {same}/{len(QA)}")
# 3) failure analysis (int8 system)
cats = {"correct": 0, "retrieval miss": 0, "generation error (answer was in context)": 0}
print("per-question analysis (int8):")
for i, ((q, a), c, r) in enumerate(zip(QA, CTX, rows["int8 per-channel"])):
    inctx = any(a in docs[j] for j in c); k = "correct" if a in r[0] else ("generation error (answer was in context)" if inctx else "retrieval miss"); cats[k] += 1
    print(f"  Q{i+1:02d} [{k}] gold={a!r} answer={r[0][:70]!r}")
print("failure categories:", cats)
