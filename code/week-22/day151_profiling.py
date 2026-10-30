"""Day 151: profiling where the time goes. Target: the Day 150 capstone's BM25 retrieval (pure Python).
(1) wall-clock timing with warm-up and repeats; (2) cProfile hot spots; (3) tracemalloc memory; (4) two fixes guided by the profile
(inverted index; scipy sparse matrix-vector product), checked for identical scores; (5) torch.profiler op table for a small MLP."""
import time, cProfile, pstats, io, tracemalloc, collections, math, numpy as np, torch, scipy.sparse as sp, _llmops as L
import _rag as G
D = G.days(range(1, 17)); CH = [c for _, c in G.chunks(D, by="section")]; QS = [q for q, _ in G.eval_set(D)]
def bench(fn, reps=3):
    fn(QS[:3]); ts = []
    for _ in range(reps): t = time.perf_counter(); fn(QS); ts.append(time.perf_counter() - t)
    return min(ts) / len(QS) * 1e3
t = time.perf_counter(); bm = G.BM25(CH); print(f"corpus {len(CH)} chunks, {len(QS)} queries; BM25 build {time.perf_counter()-t:.2f} s")
base = lambda qs: [bm.scores(q) for q in qs]
print(f"(1) baseline (Week 17 BM25.scores): {bench(base):.2f} ms/query (best of 3 repeats over {len(QS)} queries)")
pr = cProfile.Profile(); pr.enable(); base(QS); pr.disable(); s = io.StringIO(); st = pstats.Stats(pr, stream=s); st.sort_stats("tottime").print_stats(6)
print("(2) cProfile, top by own time:"); [print("   ", l.strip()[:150]) for l in s.getvalue().splitlines() if l.strip() and (l.strip()[0].isdigit() or "ncalls" in l)][:8]
tracemalloc.start(); bm2 = G.BM25(CH); cur, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
print(f"(3) tracemalloc: building BM25 allocates peak {peak/1e6:.1f} MB (per-doc Counters); per query peak:", end=" ")
tracemalloc.start(); bm.scores(QS[0]); print(f"{tracemalloc.get_traced_memory()[1]/1e3:.0f} kB"); tracemalloc.stop()
# fix A: the profile says we loop over ALL docs for every query; an inverted index touches only docs containing query terms
post = collections.defaultdict(list)
for j, (c, Ln) in enumerate(zip(bm.d, bm.len)):
    for term, f in c.items(): post[term].append((j, bm.idf[term] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * Ln / bm.avg))))
def inv(qs):
    out = []
    for q in qs:
        s = np.zeros(len(CH))
        for term in G.TOK(q):
            for j, w in post.get(term, ()): s[j] += w
        out.append(s)
    return out
# fix B: same weights as a sparse doc x term matrix; a query is a count vector -> one sparse mat-vec
vocab = {t: i for i, t in enumerate(post)}; rows, cols, vals = [], [], []
for term, lst in post.items():
    for j, w in lst: rows.append(j); cols.append(vocab[term]); vals.append(w)
W = sp.csr_matrix((vals, (rows, cols)), shape=(len(CH), len(vocab)))
def sparse(qs):
    out = []
    for q in qs:
        v = np.zeros(len(vocab))
        for term in G.TOK(q):
            if term in vocab: v[vocab[term]] += 1
        out.append(W @ v)
    return out
ref = base(QS); a, b = inv(QS), sparse(QS)
print(f"(4) max |score diff| vs baseline: inverted {max(np.abs(r.numpy()-x).max() for r, x in zip(ref, a)):.1e}, sparse {max(np.abs(r.numpy()-x).max() for r, x in zip(ref, b)):.1e}; "
      f"W is {W.shape[0]}x{W.shape[1]} with {W.nnz} non-zeros ({W.nnz/np.prod(W.shape):.2%} dense)")
tb, ti, ts = bench(base), bench(inv), bench(sparse)
print(f"    baseline {tb:.2f} ms/query | inverted index {ti:.3f} ms ({tb/ti:.0f}x) | sparse mat-vec {ts:.3f} ms ({tb/ts:.0f}x)")
torch.set_num_threads(1); m = torch.nn.Sequential(torch.nn.Linear(384, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 384)); x = torch.randn(256, 384)
with torch.no_grad():
    for _ in range(3): m(x)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
        for _ in range(20): m(x)
print("(5) torch.profiler, MLP 384-1024-384 on batch 256, 20 calls, 1 thread:")
for l in p.key_averages().table(sort_by="self_cpu_time_total", row_limit=5).splitlines()[:9]: print("   ", l[:120])
