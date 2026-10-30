"""Day 115: retrieval-augmented generation with SmolLM2-135M-Instruct over the course notes: closed-book vs RAG (hybrid + rerank, top 3)
vs oracle passage. Metrics: answer string in output (accuracy), answer string in retrieved context (context recall)."""
import os, sys, time, torch, _rag as R
from _qa import QA
sys.path.insert(0, os.path.join(R.ROOT, "code", "week-15")); import _hf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
D = R.days(); C = R.chunks(D, size=128, overlap=32); docs = [c for _, c in C]; notes = open(os.path.join(R.ROOT, "weeks", "week-15.md")).read() + open(os.path.join(R.ROOT, "weeks", "week-16.md")).read()
missing = [a for _, a in QA if a not in notes]; print(f"{len(QA)} questions; answers missing from notes: {missing}")
ce = R.embed(docs); bm = R.BM25(docs); CE = "cross-encoder/ms-marco-MiniLM-L6-v2"; ctok = AutoTokenizer.from_pretrained(CE); cem = AutoModelForSequenceClassification.from_pretrained(CE).eval()
@torch.no_grad()
def retrieve(q, k=3):
    dn = (R.embed([q]) @ ce.T)[0]; lx = bm.scores(q); rr = torch.zeros_like(dn)
    for S in (dn, lx): rr[S.argsort(descending=True)] += 1 / (61 + torch.arange(len(S)).float())
    top = rr.topk(20).indices.tolist(); e = ctok([q] * 20, [docs[i] for i in top], padding=True, truncation=True, max_length=320, return_tensors="pt")
    return [docs[top[j]] for j in cem(**e).logits[:, 0].argsort(descending=True)[:k].tolist()]
tok, m = _hf.load(_hf.INSTRUCT); IM_END = tok.convert_tokens_to_ids("<|im_end|>")
@torch.no_grad()
def ask(q, ctx=None):
    u = q if ctx is None else "Answer using only the context.\n\nContext:\n" + "\n---\n".join(ctx) + f"\n\nQuestion: {q}\nAnswer briefly."
    x = tok.apply_chat_template([{"role": "user", "content": u}], add_generation_prompt=True, return_tensors="pt", return_dict=True)
    o = m.generate(**x, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id, eos_token_id=[tok.eos_token_id, IM_END])
    return tok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip(), x["input_ids"].shape[1]
res = {"closed": 0, "rag": 0, "oracle": 0}; ctx_hit = 0; t = time.perf_counter(); plen = []
for q, a in QA:
    ctx = retrieve(q); hit = any(a in c for c in ctx); ctx_hit += hit
    oracle = [next(c for c in docs if a in c)]
    for k, c in (("closed", None), ("rag", ctx), ("oracle", oracle)):
        out, L = ask(q, c); res[k] += a in out
        if k == "rag": plen.append(L); print(f"[ctx {'hit ' if hit else 'miss'}] {a!r:14s} RAG -> {out[:90]!r}")
n = len(QA); print(f"context recall@3 {ctx_hit/n:.2f} | accuracy closed-book {res['closed']/n:.2f}, RAG {res['rag']/n:.2f}, oracle passage {res['oracle']/n:.2f} | mean RAG prompt {sum(plen)/n:.0f} tokens | {time.perf_counter()-t:.0f} s")
