"""Day 154 project: speed up the capstone retrieval end to end (tokenise -> score -> top-5) with the Rust kernel, check it returns
the same results, and apply Amdahl's law honestly. Then scale the corpus (tile the chunks 64x) to see where threads start to matter."""
import time, os, numpy as np, scipy.sparse as sp, rust_kernels as R, _llmops  # _llmops puts week-17/20 on sys.path
import _rag as G
D = G.days(range(1, 17)); CH = [c for _, c in G.chunks(D, by="section")]; QS = [q for q, _ in G.eval_set(D)]
bm = G.BM25(CH); vocab = {t: i for i, t in enumerate(bm.idf)}
rows, cols, vals = [], [], []
for j, (c, Ln) in enumerate(zip(bm.d, bm.len)):
    for t, f in c.items(): rows.append(j); cols.append(vocab[t]); vals.append(bm.idf[t] * f * (bm.k1 + 1) / (f + bm.k1 * (1 - bm.b + bm.b * Ln / bm.avg)))
def build(rep):
    n = len(CH); r = np.concatenate([np.array(rows) + k * n for k in range(rep)]); c = np.tile(cols, rep); v = np.tile(np.array(vals, np.float32), rep)
    W = sp.csc_matrix((v, (r, c)), shape=(n * rep, len(vocab)), dtype=np.float32); W.sort_indices()
    return W, W.indptr.astype(np.int64), W.indices.astype(np.int32), W.data
def qids(q): return [vocab[t] for t in G.TOK(q) if t in vocab]
def best(f, n=5):
    f(); ts = []
    for _ in range(n): t = time.perf_counter(); f(); ts.append(time.perf_counter() - t)
    return min(ts)
K = 5
def py_top(qs): return [np.argsort(-bm.scores(q).numpy(), kind="stable")[:K].tolist() for q in qs]
def scipy_top(W, ids):
    out = []
    for q in ids:
        x = np.bincount(q, minlength=W.shape[1]).astype(np.float32); s = W @ x
        p = np.argpartition(-s, K)[:K]; out.append(p[np.lexsort((p, -s[p]))].tolist())
    return out
W, ip, ix, d = build(1); IDS = [qids(q) for q in QS]
ref = py_top(QS); rs = R.bm25_topk(ip, ix, d, IDS, W.shape[0], K); sc = scipy_top(W, IDS)
same = lambda a, b: sum(x == y for x, y in zip(a, b))
print(f"corpus {W.shape[0]} chunks x {W.shape[1]} terms, nnz {W.nnz:,}; {len(QS)} queries, top-{K}")
print(f"agreement with Python baseline (identical ranked top-5 lists): rust {same(ref, rs)}/{len(QS)}, scipy {same(ref, sc)}/{len(QS)}")
diff = [(i, a, b) for i, (a, b) in enumerate(zip(ref, rs)) if a != b]
for i, a, b in diff[:3]:
    s = bm.scores(QS[i]).numpy(); print(f"   q{i}: python {a} rust {b}; python scores {np.round(s[a],5).tolist()} rust {np.round(s[b],5).tolist()}")
n = len(QS); tok = best(lambda: [qids(q) for q in QS]) / n * 1e3
tp = best(lambda: py_top(QS), 3) / n * 1e3; ts_ = best(lambda: scipy_top(W, IDS)) / n * 1e3
tr = best(lambda: R.bm25_topk(ip, ix, d, IDS, W.shape[0], K)) / n * 1e3; trp = best(lambda: R.bm25_topk(ip, ix, d, IDS, W.shape[0], K, True)) / n * 1e3
print(f"per query (ms): tokenise {tok:.4f} | python score+sort {tp:.3f} | scipy {ts_:.4f} | rust {tr:.4f} | rust rayon {trp:.4f}")
e2e = tok + tr; print(f"end to end with rust = tokenise + rust = {e2e:.4f} ms -> {tp/e2e:.0f}x vs python; tokenising is now {tok/e2e:.0%} of the time (Amdahl: "
      f"even an infinitely fast kernel caps the speed-up at {tp/tok:.0f}x)")
# scale: 64x tiled corpus (duplicates tie -> compare score sets, not ids)
W, ip, ix, d = build(64); N = W.shape[0]
a = R.bm25_topk(ip, ix, d, IDS, N, K); b = scipy_top(W, IDS)
ok = sum(np.allclose(np.sort((W[r] @ np.bincount(q, minlength=W.shape[1]).astype(np.float32))), np.sort(W[s] @ np.bincount(q, minlength=W.shape[1]).astype(np.float32)), rtol=1e-5)
         for r, s, q in zip(a, b, IDS))
print(f"\n64x tiled corpus: {N:,} chunks, nnz {W.nnz:,} ({(d.nbytes+ix.nbytes)/1e6:.1f} MB); top-5 score sets match scipy {ok}/{n}")
ts_ = best(lambda: scipy_top(W, IDS), 3) / n * 1e3; tr = best(lambda: R.bm25_topk(ip, ix, d, IDS, N, K), 3) / n * 1e3
trp = best(lambda: R.bm25_topk(ip, ix, d, IDS, N, K, True), 3) / n * 1e3
print(f"per query (ms): scipy {ts_:.3f} | rust {tr:.3f} | rust rayon over queries ({R.n_threads()} threads) {trp:.3f} -> rayon {tr/trp:.1f}x over serial rust")
