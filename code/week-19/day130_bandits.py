"""Day 130: exploration. (1) 10-armed Bernoulli bandit (arm means ~ U(0,1) per run), 2000 pulls, 200 runs: greedy, eps-greedy 0.1,
optimistic init (Q0=1, greedy), UCB1, Thompson sampling (Beta(1,1)). Reports cumulative regret and % optimal arm in the last 500 pulls.
(2) Synthetic recommendation tie-in: 20 items with hidden 5-d vectors, users with random 5-d context; click prob = sigmoid(3*u.v - 1).
Policies: random, eps-greedy per item (ignores context), LinUCB (disjoint, alpha 1.0) on context. 5000 rounds, 20 runs; CTR and regret."""
import time, numpy as np
def bandit(policy, T=2000, runs=200, k=10):
    reg = np.zeros(runs); opt = np.zeros(runs)
    for run in range(runs):
        rng = np.random.default_rng(run); mu = rng.uniform(0, 1, k); n = np.zeros(k); q = np.ones(k) if policy == "optimistic" else np.zeros(k); a_, b_ = np.ones(k), np.ones(k)
        for t in range(T):
            if policy == "greedy" or policy == "optimistic": a = int(rng.choice(np.flatnonzero(q == q.max())))
            elif policy == "eps0.1": a = int(rng.integers(k)) if rng.random() < 0.1 else int(rng.choice(np.flatnonzero(q == q.max())))
            elif policy == "UCB1": a = t if t < k else int(np.argmax(q + np.sqrt(2 * np.log(t) / n)))
            else: a = int(np.argmax(rng.beta(a_, b_)))
            r = float(rng.random() < mu[a]); n[a] += 1
            if policy == "optimistic": q[a] += (r - q[a]) / (n[a] + 1)  # the optimistic prior counts as one pseudo-observation
            else: q[a] += (r - q[a]) / n[a]
            a_[a] += r; b_[a] += 1 - r; reg[run] += mu.max() - mu[a]; opt[run] += (t >= T - 500) and a == mu.argmax()
    return reg.mean(), reg.std() / np.sqrt(runs), opt.mean() / 500
for p in ("greedy", "eps0.1", "optimistic", "UCB1", "Thompson"):
    t0 = time.perf_counter(); m, se, o = bandit(p)
    print(f"bandit {p}: cumulative regret after 2000 pulls {m:.1f} ± {se:.1f} (s.e., 200 runs); optimal arm in last 500 pulls {o:.0%}; {time.perf_counter()-t0:.1f} s", flush=True)
sig = lambda x: 1 / (1 + np.exp(-x))
def recsys(policy, T=5000, runs=20, K=20, d=5):
    ctr = []; reg = []
    for run in range(runs):
        rng = np.random.default_rng(100 + run); Vit = rng.normal(0, 1, (K, d)) / np.sqrt(d); Ainv = np.stack([np.eye(d)] * K); bv = np.zeros((K, d)); n = np.zeros(K); q = np.zeros(K); c = 0; rg = 0
        for t in range(T):
            u = rng.normal(0, 1, d); p = sig(3 * Vit @ u - 1)
            if policy == "random": a = int(rng.integers(K))
            elif policy == "eps-greedy (no context)": a = int(rng.integers(K)) if rng.random() < 0.1 else int(q.argmax())
            else:
                th = np.einsum("kij,kj->ki", Ainv, bv); ucb = th @ u + 1.0 * np.sqrt(np.einsum("i,kij,j->k", u, Ainv, u)); a = int(ucb.argmax())
            r = float(rng.random() < p[a]); c += r; rg += p.max() - p[a]; n[a] += 1; q[a] += (r - q[a]) / n[a]
            Au = Ainv[a] @ u; Ainv[a] -= np.outer(Au, Au) / (1 + u @ Au); bv[a] += r * u   # Sherman-Morrison update
        ctr.append(c / T); reg.append(rg)
    return np.mean(ctr), np.mean(reg)
for p in ("random", "eps-greedy (no context)", "LinUCB"):
    t0 = time.perf_counter(); c, r = recsys(p)
    print(f"recsys {p}: CTR {c:.3f}; cumulative regret over 5000 rounds {r:.0f} (mean of 20 runs); {time.perf_counter()-t0:.1f} s", flush=True)
