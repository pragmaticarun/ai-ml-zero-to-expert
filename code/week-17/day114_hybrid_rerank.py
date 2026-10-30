"""Day 114: lexical (BM25) vs dense vs hybrid (reciprocal rank fusion) retrieval, then cross-encoder reranking of the top 20."""
import time, torch, _rag as R
from transformers import AutoTokenizer, AutoModelForSequenceClassification
D = R.days(); Q = R.eval_set(D); qt, gold = [q for q, _ in Q], [d for _, d in Q]
I = torch.load("out/index_128.pt"); C, ce = I["C"], I["ce"]; docs = [c for _, c in C]
qe = R.embed(qt); dense = qe @ ce.T
t = time.perf_counter(); bm = R.BM25(docs); lex = torch.stack([bm.scores(q) for q in qt]); print(f"BM25 index + scoring {len(qt)} queries: {time.perf_counter()-t:.1f} s")
def ranks_of(S, k=20): return S.topk(k, dim=1).indices
def show(name, idx): r = [[C[i][0] for i in row] for row in idx.tolist()]; print(f"{name:28s} recall@1 {R.metrics(r, gold)[0]:.2f} recall@5 {R.metrics(r, gold)[1]:.2f} MRR {R.metrics(r, gold)[2]:.2f}")
show("dense (MiniLM)", ranks_of(dense)); show("BM25", ranks_of(lex))
def rrf(*Ss, k=60):
    out = torch.zeros_like(Ss[0])
    for S in Ss:
        order = S.argsort(dim=1, descending=True); rank = torch.empty_like(order); rank.scatter_(1, order, torch.arange(S.shape[1]).expand_as(order)); out += 1 / (k + rank + 1)
    return out
hyb = rrf(dense, lex); show("hybrid RRF(dense, BM25)", ranks_of(hyb))
CE = "cross-encoder/ms-marco-MiniLM-L6-v2"; tok = AutoTokenizer.from_pretrained(CE); ced = AutoModelForSequenceClassification.from_pretrained(CE).eval()
for base_name, base in (("dense", dense), ("hybrid", hyb)):
    top = ranks_of(base).tolist(); t = time.perf_counter(); new = []
    with torch.no_grad():
        for q, row in zip(qt, top):
            e = tok([q] * len(row), [docs[i] for i in row], padding=True, truncation=True, max_length=320, return_tensors="pt"); s = ced(**e).logits[:, 0]
            new.append([row[j] for j in s.argsort(descending=True).tolist()])
    print(f"cross-encoder rerank of {base_name} top-20: {time.perf_counter()-t:.1f} s for {len(qt)} queries ({(time.perf_counter()-t)/len(qt)*1000:.0f} ms/query)"); show(f"{base_name} + rerank", torch.tensor(new))
