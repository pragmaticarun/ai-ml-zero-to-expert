"""Day 135: a ReAct-style loop (Thought -> Action -> Observation ... -> Final) on the 24-task synthetic suite (_agent.suite()).
Policies: (a) SmolLM2-135M-Instruct answering directly, no tools; (b) SmolLM2 in a ReAct loop with a 1-example transcript;
(c) a scripted text-parsing policy (regex over the question, no model) as a reference; (d) the structured oracle policy (upper bound).
Max 6 steps; greedy decoding."""
import json, re, time, collections, _agent as A
T = A.suite()
EX = [("user", "Task: What is the combined population of Zeta and Omer?"),
      ("assistant", 'Thought: I need the population of Zeta.\nAction: {"tool": "lookup", "args": {"city": "Zeta", "field": "population"}}'),
      ("user", "Observation: 1000"),
      ("assistant", 'Thought: Now the population of Omer.\nAction: {"tool": "lookup", "args": {"city": "Omer", "field": "population"}}'),
      ("user", "Observation: 500"),
      ("assistant", 'Thought: Add them.\nAction: {"tool": "calculator", "args": {"expression": "1000+500"}}'),
      ("user", "Observation: 1500"), ("assistant", "Final: 1500")]
SYS = ("Solve the task step by step. Each turn write 'Thought: ...' then either 'Action: <one JSON tool call>' or 'Final: <answer>'.\nTools:\n"
       + json.dumps(A.tool_specs(["calculator", "lookup"])))
RAW = {}
def smol_react(task, hist):
    m = [{"role": "system", "content": SYS}] + [{"role": r, "content": c} for r, c in EX] + [{"role": "user", "content": "Task: " + task["q"]}]
    for act, st, res in hist:
        m += [{"role": "assistant", "content": "Action: " + json.dumps(act) if isinstance(act, dict) else "(unparseable)"},
              {"role": "user", "content": f"Observation: {res}" if st == "ok" else f"Observation: ERROR {res}"}]
    out = A.generate(m, max_new=60); RAW.setdefault(task["q"], []).append(out)
    f = re.search(r"Final:\s*(.+)", out)
    a = A.parse_json(out.split("Action:", 1)[1]) if "Action:" in out else None
    if a is not None and (f is None or out.index("Action:") < out.index("Final:")): return a
    if f: v = f.group(1).strip().split("\n")[0]; return {"final": v.replace(",", "")}
    return None
def text_scripted(task, hist):
    q = task["q"]; cs = re.findall(r"\b(?:" + "|".join(A.CITIES) + r")\b", q)   # knows the city vocabulary (a gazetteer), not the task structure
    field = "temp_c" if "warmer" in q else "country" if "country" in q else "population"
    if len(hist) < len(cs): return {"tool": "lookup", "args": {"city": cs[len(hist)], "field": field}}
    v = [h[2] for h in hist]
    if field == "country": return {"final": "yes" if v[0] == v[1] else "no"}
    if len(hist) == len(cs):
        e = f"{v[0]}*2" if "twice" in q else f"{v[0]}+{v[1]}" if "combined" in q else f"{v[0]}-{v[1]}"; return {"tool": "calculator", "args": {"expression": e}}
    return {"final": v[-1]}
def direct(task):
    return A.generate([{"role": "user", "content": task["q"] + " Reply with just the answer."}], max_new=20)
if __name__ == "__main__":
    t0 = time.perf_counter(); ok = 0; ex = []
    for t in T:
        a = direct(t); n = re.sub(r"[,\s]", "", a); m = re.search(r"-?\d+(\.\d+)?", n); v = m.group(0) if (m and not isinstance(t["gold"], str)) else a
        ok += A.correct(v, t["gold"]); ex.append((t["q"], a))
    print(f"(a) SmolLM2 direct, no tools: {ok}/{len(T)} correct; {(time.perf_counter()-t0)/len(T):.2f} s/task; e.g. {ex[0][0]!r} -> {ex[0][1]!r} (gold {T[0]['gold']})", flush=True)
    for name, pol in (("(b) SmolLM2 ReAct", smol_react), ("(c) scripted text-parsing", text_scripted), ("(d) structured oracle", A.scripted_react)):
        t0 = time.perf_counter(); res = [A.run_agent(t, pol, max_steps=6) for t in T]; dt = (time.perf_counter() - t0) / len(T)
        succ = [A.correct(r["answer"], t["gold"]) for r, t in zip(res, T)]; by = collections.Counter(t["kind"] for t, s in zip(T, succ) if s)
        errs = sum(1 for r in res for h in r["hist"] if h[1] == "error"); pe = sum(1 for r in res for h in r["hist"] if h[0] == "parse_error")
        print(f"{name}: {sum(succ)}/{len(T)} correct (by kind {dict(by)}); mean steps {sum(r['steps'] for r in res)/len(T):.1f}; "
              f"step-limit {sum(r['status']=='step_limit' for r in res)}; tool errors {errs}; parse errors {pe}; {dt:.2f} s/task", flush=True)
        if pol is smol_react:
            for t, r, s in list(zip(T, res, succ))[:24:6]:
                print(f"   task {t['q']!r} gold {t['gold']} -> answer {r['answer']!r} ({'ok' if s else 'WRONG'}); calls {[h[0] for h in r['hist']][:3]}; obs {[h[2] for h in r['hist']][:3]}")
