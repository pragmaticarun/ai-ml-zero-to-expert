"""Day 138: evaluating agents. Suite: the 24 synthetic tasks. Policies: structured oracle; scripted text-parsing; a 'noisy' scripted policy that
injects LLM-like failure modes at known rates (so we can check the taxonomy recovers them); and SmolLM2-135M ReAct (re-run, as Day 135).
Metrics: success with bootstrap 95% CI, pass@k vs pass^k (all k trials succeed) over 8 trials, and a trace-based failure taxonomy."""
import random, collections, numpy as np, _agent as A
from day135_react import smol_react, text_scripted
T = A.suite()
def noisy(rates, rng):
    def pol(task, hist):
        r = rng.random(); c = 0
        for mode, p in rates.items():
            c += p
            if r < c: break
        else: mode = None
        if mode == "hallucinate" and not hist: return {"final": str(rng.randint(1, 10**6))}
        if mode == "bad_args": return {"tool": "lookup", "args": {"city": task["cities"][0]}}
        if mode == "parse": return None
        if mode == "wrong_math" and len(hist) == len(task["cities"]) and task["op"] != "==": return {"final": hist[0][2]}
        return text_scripted(task, [h for h in hist if h[1] == "ok" and isinstance(h[0], dict)])
    return pol
def taxonomy(r, task):
    if A.correct(r["answer"], task["gold"]): return "success"
    H = r["hist"]
    if r["status"] == "step_limit":
        if H and all(h[0] == "parse_error" for h in H): return "unparseable output"
        if any(h[1] == "error" for h in H): return "stuck on tool errors"
        return "step limit / loop"
    if not any(h[1] == "ok" for h in H): return "answered without evidence"
    return "wrong answer despite evidence"
def boot(x, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x, float); m = rng.choice(x, (n, len(x))).mean(1); return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
RATES = {"hallucinate": 0.05, "bad_args": 0.05, "parse": 0.05, "wrong_math": 0.10}
if __name__ == "__main__":
    runs = {"structured oracle": lambda k: A.scripted_react, "scripted text-parsing": lambda k: text_scripted,
            "noisy scripted (known error rates)": lambda k: noisy(RATES, random.Random(k))}
    K = 8
    for name, mk in runs.items():
        succ = np.zeros((K, len(T))); tax = collections.Counter()
        for k in range(K):
            pol = mk(k)
            for j, t in enumerate(T):
                r = A.run_agent(t, pol, max_steps=6); c = taxonomy(r, t); tax[c] += 1; succ[k, j] = c == "success"
        m, lo, hi = boot(succ[0]); print(f"{name}: trial-1 success {m:.3f} [95% CI {lo:.3f}-{hi:.3f}]; mean over {K} trials {succ.mean():.3f}; "
                                          f"pass@{K} {succ.max(0).mean():.3f}; pass^{K} {succ.min(0).mean():.3f}; taxonomy over {K*len(T)} runs {dict(tax)}", flush=True)
    succ = []; tax = collections.Counter()
    for t in T:
        r = A.run_agent(t, smol_react, max_steps=6); c = taxonomy(r, t); tax[c] += 1; succ.append(c == "success")
    m, lo, hi = boot(succ); print(f"SmolLM2-135M ReAct (greedy, 1 trial): success {m:.3f} [95% CI {lo:.3f}-{hi:.3f}]; taxonomy {dict(tax)}")
