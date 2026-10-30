"""Day 134: the agent loop and function/tool calling with JSON schemas.
(1) validator checks on hand-made calls; (2) a scripted policy through the loop; (3) can SmolLM2-135M-Instruct emit a correct
tool call? 18 single-tool requests, greedy decoding, zero-shot vs 3-shot prompt; we score parse / schema-valid / right tool / right result."""
import json, time, _agent as A
print("== (1) validation before execution")
cases = [{"tool": "lookup", "args": {"city": "Corvo", "field": "population"}}, {"tool": "lookup", "args": {"city": "Corvo"}},
         {"tool": "lookup", "args": {"city": "Corvo", "field": "area"}}, {"tool": "lookup", "args": {"city": 3, "field": "country"}},
         {"tool": "calculator", "args": {"expression": "2+2", "unsafe": True}}, {"tool": "weather", "args": {}},
         {"tool": "calculator", "args": {"expression": "open('x').read()"}}, {"tool": "lookup", "args": {"city": "Atlantis", "field": "country"}}, "not a dict"]
for c in cases: print(f"  {json.dumps(c)[:70]:72s} -> {A.call_tool(c)}")
print("== (2) scripted policy in the loop: 'population of Corvo divided by population of Dunmere'")
def scripted(task, hist):
    if len(hist) == 0: return {"tool": "lookup", "args": {"city": "Corvo", "field": "population"}}
    if len(hist) == 1: return {"tool": "lookup", "args": {"city": "Dunmere", "field": "population"}}
    if len(hist) == 2: return {"tool": "calculator", "args": {"expression": f"{hist[0][2]}/{hist[1][2]}"}}
    return {"final": hist[2][2]}
r = A.run_agent("ratio", scripted); [print("  ", h) for h in r["hist"]]; print(f"  final {r['answer']} in {r['steps']} steps ({r['status']})")
print("== (3) SmolLM2-135M-Instruct as the tool caller")
REQ = [(f"What is the population of {c}?", "lookup", None, A.CITIES[c]["population"]) for c in ("Avalon", "Corvo", "Elsby")] + \
      [(f"Which country is {c} in?", "lookup", None, A.CITIES[c]["country"]) for c in ("Brindle", "Dunmere", "Farrow")] + \
      [(f"What is the temperature in {c}?", "lookup", None, A.CITIES[c]["temp_c"]) for c in ("Avalon", "Corvo", "Elsby")] + \
      [(f"Compute {e}.", "calculator", None, A.calc(e)) for e in ("17*23", "1024/16", "3**4", "250-87", "12*12+1", "(7+5)*3", "99/3", "45%7", "2**10")]
SYS = ("You can call tools. Reply with ONLY one JSON object {\"tool\": name, \"args\": {...}}.\nTools:\n" + json.dumps(A.tool_specs(["calculator", "lookup"])))
SHOTS = [("What is the population of Zeta?", {"tool": "lookup", "args": {"city": "Zeta", "field": "population"}}),
         ("Which country is Omer in?", {"tool": "lookup", "args": {"city": "Omer", "field": "country"}}),
         ("Compute 8*9.", {"tool": "calculator", "args": {"expression": "8*9"}})]
for mode in ("zero-shot", "3-shot"):
    n = dict(parsed=0, valid=0, tool=0, result=0); t0 = time.perf_counter(); fails = []
    for q, tool, _, gold in REQ:
        msgs = [{"role": "system", "content": SYS}]
        if mode == "3-shot":
            for sq, sa in SHOTS: msgs += [{"role": "user", "content": sq}, {"role": "assistant", "content": json.dumps(sa)}]
        out = A.generate(msgs + [{"role": "user", "content": q}]); c = A.parse_json(out)
        if c is not None:
            n["parsed"] += 1; st, res = A.call_tool(c)
            n["valid"] += st == "ok" or not str(res).startswith(("malformed", "unknown tool", "$"))
            n["tool"] += c.get("tool") == tool; n["result"] += st == "ok" and res == gold
            if not (st == "ok" and res == gold): fails.append((q, out[:90]))
        else: fails.append((q, out[:90]))
    print(f"  {mode}: of {len(REQ)} requests: JSON parsed {n['parsed']}, schema-valid {n['valid']}, right tool {n['tool']}, correct result {n['result']}; "
          f"{(time.perf_counter()-t0)/len(REQ):.2f} s/request", flush=True)
    for q, o in fails[:4]: print(f"     fail: {q!r} -> {o!r}")
