"""Day 136: planning and memory (scripted policies; SmolLM2 failed ReAct on Day 135, so no model here).
(1) Planning under unreliable tools: each tool call fails (transient error) with prob p. Compare plan-then-execute (fixed plan, no feedback),
    plan-then-execute with up to 2 retries per step, and ReAct-style (observe each result, retry/re-decide; step cap 10). 24 tasks x 50 seeds.
(2) Memory: a 60-turn synthetic conversation with 20 facts ('the access code for project P7 is 4821') among 40 distractors; 5 facts are
    later UPDATED. 20 questions at the end. Memories: last-10-turn window, full history, BM25 top-3, MiniLM dense top-3, key-value store."""
import random, re, sys, os, numpy as np, _agent as A
sys.path.insert(0, os.path.join(A.ROOT, "code", "week-17")); import _rag as R
T = A.suite()
def flaky(p, rng):
    def call(c):
        if rng.random() < p: return "error", "TimeoutError: transient"
        return A.call_tool(c)
    return call
def plan(task):  # the whole plan is written up front; later steps refer to earlier results by index
    s = [("lookup", {"city": c, "field": task["field"]}) for c in task["cities"]]
    if task["op"] != "==": s.append(("calc", task["op"]))
    return s
def exec_plan(task, call, retries):
    vals, calls = [], 0
    for kind, a in plan(task):
        for attempt in range(retries + 1):
            calls += 1
            if kind == "lookup": st, r = call({"tool": "lookup", "args": a})
            else: e = f"{vals[0]}{a}{vals[1]}" if len(vals) == 2 else f"{vals[0]}{a}"; st, r = call({"tool": "calculator", "args": {"expression": e}})
            if st == "ok": break
        if st != "ok": return None, calls
        vals.append(r)
    return (("yes" if vals[0] == vals[1] else "no") if task["op"] == "==" else vals[-1]), calls
def react(task, call, cap=10):
    ok, calls = [], 0
    for _ in range(cap):   # decide the next action from what has succeeded so far
        act = A.scripted_react(task, [(None, "ok", v) for v in ok])
        if "final" in act: return act["final"], calls
        calls += 1; st, r = call(act)
        if st == "ok": ok.append(r)
    return None, calls
print("(1) planning with unreliable tools: success rate (mean tool calls per task), 24 tasks x 50 seeds")
for p in (0.0, 0.1, 0.3):
    row = []
    for name, f in (("plan, no retry", lambda t, c: exec_plan(t, c, 0)), ("plan, 2 retries", lambda t, c: exec_plan(t, c, 2)), ("ReAct, cap 10", react)):
        S, C = [], []
        for seed in range(50):
            rng = random.Random(seed); call = flaky(p, rng)
            for t in T: a, n = f(t, call); S.append(A.correct(a, t["gold"])); C.append(n)
        row.append(f"{name} {np.mean(S):.3f} ({np.mean(C):.2f})")
    print(f"  p={p}: " + "; ".join(row), flush=True)
# ---------- memory ----------
rng = random.Random(0); P = [f"P{i}" for i in range(1, 21)]; code = {p: rng.randint(1000, 9999) for p in P}
FILL = ["I had coffee this morning.", "The weather is fine today.", "Let's keep going with the plan.", "Remind me to stretch later.", "That meeting ran long.",
        "I prefer short answers.", "My laptop fan is noisy.", "Lunch was good.", "We should tidy the repo.", "The train was late again."]
turns = [f"Note: the access code for project {p} is {code[p]}." for p in P]
for i in range(40): turns.insert(rng.randint(0, len(turns)), FILL[i % 10] + f" (turn note {i})")
upd = rng.sample(P, 5)
for p in upd: code[p] = rng.randint(1000, 9999); turns.append(f"Update: the access code for project {p} is now {code[p]}.")
Qs = [(f"What is the access code for project {p}?", p) for p in P]
print(f"(2) memory: {len(turns)} turns, {len(P)} facts, {len(upd)} updated near the end; 20 questions")
def extract(texts, p):  # read the LATEST statement about project p among the texts given (texts are in conversation order)
    v = None
    for t in texts:
        m = re.search(rf"project {p} is (?:now )?(\d+)", t)
        if m: v = int(m.group(1))
    return v
E_ = R.embed(turns); bm = R.BM25(turns); kv = {}
for t in turns:
    m = re.search(r"project (P\d+) is (?:now )?(\d+)", t)
    if m: kv[m.group(1)] = int(m.group(2))     # write-time extraction: overwrite on update
def topk_in_order(scores, k): return sorted(np.argsort(-np.asarray(scores))[:k].tolist())
for name, mem in (("window last 10 turns", lambda q: turns[-10:]), ("full history", lambda q: turns),
                  ("BM25 top-3", lambda q: [turns[i] for i in topk_in_order(bm.scores(q).tolist(), 3)]),
                  ("MiniLM dense top-3", lambda q: [turns[i] for i in topk_in_order((R.embed([q]) @ E_.T)[0].tolist(), 3)]),
                  ("BM25 top-1", lambda q: [turns[i] for i in topk_in_order(bm.scores(q).tolist(), 1)]),
                  ("MiniLM dense top-1", lambda q: [turns[i] for i in topk_in_order((R.embed([q]) @ E_.T)[0].tolist(), 1)]),
                  ("key-value store", lambda q: [f"project {re.search(r'P\d+', q).group(0)} is {kv[re.search(r'P\d+', q).group(0)]}"])):
    got = [extract(mem(q), p) for q, p in Qs]; acc = np.mean([g == code[p] for g, (q, p) in zip(got, Qs)])
    accu = np.mean([g == code[p] for g, (q, p) in zip(got, Qs) if p in upd]); words = np.mean([sum(len(t.split()) for t in mem(q)) for q, _ in Qs])
    print(f"  {name:22s}: accuracy {acc:.2f} (on the 5 updated facts {accu:.2f}); mean context {words:.0f} words", flush=True)
