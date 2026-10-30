"""Day 113: embeddings + brute-force cosine vector search over the course notes (weeks 1-16), comparing chunking strategies."""
import time, torch, _rag as R
D = R.days(); Q = R.eval_set(D); qtext, gold = [q for q, _ in Q], [d for _, d in Q]
print(f"notes: {len(D)} days with content; eval queries (interview questions, their sections removed from the index): {len(Q)}")
print("example query:", repr(qtext[0][:120]), "-> gold day", gold[0])
t = time.perf_counter(); qe = R.embed(qtext); print(f"embedded {len(Q)} queries in {time.perf_counter()-t:.1f} s, dim {qe.shape[1]}")
for name, kw in (("words 64/16", dict(size=64, overlap=16)), ("words 128/32", dict(size=128, overlap=32)), ("words 256/64", dict(size=256, overlap=64)), ("section", dict(size=128, by="section"))):
    C = R.chunks(D, **kw); t = time.perf_counter(); ce = R.embed([c for _, c in C]); te = time.perf_counter() - t
    t = time.perf_counter(); sims = qe @ ce.T; top = sims.topk(20, dim=1).indices; ts = (time.perf_counter() - t) * 1000
    ranks = [[C[i][0] for i in row.tolist()] for row in top]; r1, r5, mrr = R.metrics(ranks, gold)
    print(f"chunking {name:13s}: {len(C):5d} chunks, embed {te:5.1f} s, search all queries {ts:.1f} ms | recall@1 {r1:.2f} recall@5 {r5:.2f} MRR {mrr:.2f}")
    if name == "words 128/32": torch.save({"C": C, "ce": ce}, "out/index_128.pt")
# index memory and an int8 index (preview of Day 116)
C = torch.load("out/index_128.pt")["C"]; ce = torch.load("out/index_128.pt")["ce"]; s = ce.abs().max(1, keepdim=True).values / 127; q8 = (ce / s).round().to(torch.int8)
ranks = [[C[i][0] for i in row.tolist()] for row in (qe @ (q8.float() * s).T).topk(20, dim=1).indices]
print(f"index fp32 {ce.numel()*4/1e6:.2f} MB vs int8 {q8.numel()/1e6:.2f} MB; int8 index recall@1/@5/MRR {R.metrics(ranks, gold)}")
