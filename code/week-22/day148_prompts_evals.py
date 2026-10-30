"""Day 148: LLMOps I. Prompt versions in a content-hashed registry, an eval harness with cached generations,
per-case results, a paired regression gate (exact McNemar), and a determinism check. Model: SmolLM2-135M-Instruct, greedy, CPU."""
import json, os, time, math, _llmops as L
A = L.A
for f in ("prompts.jsonl", "gen_cache.json"):
    p = os.path.join(L.OUT, f); os.path.exists(p) and os.remove(p)
C = A.CITIES
CASES = [(f"What is the population of {c}?", A.CITIES[c]["population"]) for c in ("Avalon", "Corvo", "Elsby")] + \
        [(f"Which country is {c} in?", C[c]["country"]) for c in ("Brindle", "Dunmere", "Farrow")] + \
        [(f"What is the temperature in {c}?", C[c]["temp_c"]) for c in ("Avalon", "Corvo", "Elsby")] + \
        [(f"Compute {e}.", A.calc(e)) for e in ("17*23", "1024/16", "3**4", "250-87", "12*12+1", "(7+5)*3", "99/3", "45%7", "2**10")] + \
        [("How many people live in Brindle?", C["Brindle"]["population"]), ("Dunmere: what country?", C["Dunmere"]["country"]),   # held-out phrasings
         ("How warm is it in Farrow (Celsius)?", C["Farrow"]["temp_c"]), ("Please work out 6*7 for me.", 42),
         ("What's 81 divided by 9?", 9.0), ("Population of Corvo, please.", C["Corvo"]["population"])]
TS = json.dumps(A.tool_specs(["calculator", "lookup"]))
R = L.Registry(os.path.join(L.OUT, "prompts.jsonl"))
SYS = "You can call tools. Reply with ONLY one JSON object {\"tool\": name, \"args\": {...}}.\nTools:\n" + TS
SH3 = [("What is the population of Zeta?", {"tool": "lookup", "args": {"city": "Zeta", "field": "population"}}),
       ("Which country is Omer in?", {"tool": "lookup", "args": {"city": "Omer", "field": "country"}}),
       ("Compute 8*9.", {"tool": "calculator", "args": {"expression": "8*9"}})]
SH4 = SH3 + [("What is the temperature in Zeta?", {"tool": "lookup", "args": {"city": "Zeta", "field": "temp_c"}})]
V = {"v1 zero-shot": R.register("toolcall", SYS, [], "Week 20 zero-shot"),
     "v2 3-shot": R.register("toolcall", SYS, SH3, "Week 20 3-shot"),
     "v3 4-shot (+temp_c shot)": R.register("toolcall", SYS, SH4, "adds a temp_c example"),
     "v4 = v3, system reworded": R.register("toolcall", SYS.replace("Reply with ONLY one JSON object", "Answer with exactly one JSON object and nothing else:"), SH4, "wording only")}
print("registry:", {k: h for k, h in V.items()})
def run_case(h, q, gold, cache):
    out = cache.get({"model": "SmolLM2-135M-Instruct", "prompt": h, "q": q, "max_new": 60, "greedy": True}, lambda: A.generate(R.messages(h, q)))
    c = A.parse_json(out); st, res = A.call_tool(c) if c is not None else ("error", "unparseable")
    return st == "ok" and A.correct(res, gold) if not isinstance(gold, str) else (st == "ok" and res == gold)
res = {}
with L.Cache(os.path.join(L.OUT, "gen_cache.json")) as cache:
    for name, h in V.items():
        t = time.perf_counter(); res[name] = [run_case(h, q, g, cache) for q, g in CASES]
        r = res[name]; print(f"  {name:26s} [{h}]: pass {sum(r)}/{len(r)} (seen 18: {sum(r[:18])}, held-out 6: {sum(r[18:])}); {time.perf_counter()-t:.1f} s")
    m0 = cache.miss; t = time.perf_counter(); again = [run_case(V["v3 4-shot (+temp_c shot)"], q, g, cache) for q, g in CASES]
    print(f"re-run v3 from cache: {cache.hits} hits, {cache.miss - m0} new generations, {time.perf_counter()-t:.3f} s; identical results {again == res['v3 4-shot (+temp_c shot)']}")
# determinism without the cache
outs = [[A.generate(R.messages(V["v2 3-shot"], q)) for q, _ in CASES[:5]] for _ in range(2)]
print(f"determinism: greedy outputs for 5 cases generated twice identical: {outs[0] == outs[1]}")
def mcnemar(base, cand):
    b = sum(x and not y for x, y in zip(base, cand)); c = sum(y and not x for x, y in zip(base, cand)); n = b + c
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n) if n else 1.0; return b, c, p
print("regression gate: candidate must (a) lose no case the baseline passed on the 18 'seen' cases, (b) not be worse overall")
pairs = [("v1 zero-shot", "v2 3-shot"), ("v2 3-shot", "v3 4-shot (+temp_c shot)"), ("v3 4-shot (+temp_c shot)", "v4 = v3, system reworded")]
for a, b in pairs:
    lost, gained, p = mcnemar(res[a], res[b]); lost_seen = [CASES[i][0] for i in range(18) if res[a][i] and not res[b][i]]
    ok = not lost_seen and sum(res[b]) >= sum(res[a])
    print(f"  {a} -> {b}: lost {lost}, gained {gained}, exact McNemar p={p:.3f}; gate {'PASS' if ok else 'FAIL'}" + (f" (regressed: {lost_seen[:3]})" if lost_seen else ""))
