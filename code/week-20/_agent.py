"""Week 20 shared pieces, all from scratch: a JSON-Schema subset validator, a tool registry, safe tools, the agent loop,
and a SmolLM2-135M-Instruct wrapper (local, verified cache; loaded only when asked)."""
import ast, json, operator, os, re, subprocess, sys, tempfile, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# ---------- JSON Schema subset: type, properties, required, enum, additionalProperties, minimum/maximum, items ----------
TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "object": dict, "array": list}
def validate(x, s, path="$"):
    errs = []; t = s.get("type")
    if t:
        ok = isinstance(x, TYPES[t]) and not (t in ("integer", "number") and isinstance(x, bool))
        if not ok: return [f"{path}: expected {t}, got {type(x).__name__}"]
    if "enum" in s and x not in s["enum"]: errs.append(f"{path}: {x!r} not in {s['enum']}")
    if "minimum" in s and x < s["minimum"]: errs.append(f"{path}: {x} < minimum {s['minimum']}")
    if "maximum" in s and x > s["maximum"]: errs.append(f"{path}: {x} > maximum {s['maximum']}")
    if t == "object":
        for k in s.get("required", []):
            if k not in x: errs.append(f"{path}: missing required '{k}'")
        props = s.get("properties", {})
        for k, v in x.items():
            if k in props: errs += validate(v, props[k], f"{path}.{k}")
            elif s.get("additionalProperties", True) is False: errs.append(f"{path}: unexpected property '{k}'")
    if t == "array" and "items" in s:
        for i, v in enumerate(x): errs += validate(v, s["items"], f"{path}[{i}]")
    return errs
# ---------- tools ----------
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}
def calc(expression):
    """Safe arithmetic: walks the AST and allows only numbers and + - * / ** %; never calls eval()."""
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and abs(b) > 100: raise ValueError("exponent too large")
            return OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.operand))
        raise ValueError(f"disallowed syntax: {type(n).__name__}")
    v = ev(ast.parse(expression, mode="eval").body); return round(v, 6) if isinstance(v, float) else v
# a small synthetic "database" (invented fictional cities; not real data)
CITIES = {"Avalon": {"population": 120000, "country": "Northland", "temp_c": 14}, "Brindle": {"population": 45000, "country": "Northland", "temp_c": 9},
          "Corvo": {"population": 310000, "country": "Southmark", "temp_c": 27}, "Dunmere": {"population": 8000, "country": "Southmark", "temp_c": 22},
          "Elsby": {"population": 960000, "country": "Eastreach", "temp_c": 18}, "Farrow": {"population": 27500, "country": "Eastreach", "temp_c": 11}}
def lookup(city, field):
    if city not in CITIES: raise KeyError(f"unknown city {city!r}")
    return CITIES[city][field]
OUTBOX = []
def send_email(to, body): OUTBOX.append((to, body)); return "queued"   # side-effect tool: only records, never sends anything
TOOLS = {
 "calculator": {"fn": calc, "side_effect": False, "description": "Evaluate an arithmetic expression.",
                "schema": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"], "additionalProperties": False}},
 "lookup": {"fn": lookup, "side_effect": False, "description": "Look up a field for a city in the database.",
            "schema": {"type": "object", "properties": {"city": {"type": "string"}, "field": {"type": "string", "enum": ["population", "country", "temp_c"]}},
                       "required": ["city", "field"], "additionalProperties": False}},
 "send_email": {"fn": send_email, "side_effect": True, "description": "Send an email (side effect).",
                "schema": {"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}, "required": ["to", "body"], "additionalProperties": False}},
}
def tool_specs(names=None):
    return [{"name": n, "description": t["description"], "inputSchema": t["schema"]} for n, t in TOOLS.items() if names is None or n in names]
def call_tool(call, allowed=None):
    """call = {'tool': name, 'args': {...}} -> ('ok', result) or ('error', message). Validation happens BEFORE execution."""
    if not isinstance(call, dict) or "tool" not in call or "args" not in call: return "error", "malformed call: need {'tool','args'}"
    n = call["tool"]
    if n not in TOOLS: return "error", f"unknown tool {n!r}"
    if allowed is not None and n not in allowed: return "error", f"tool {n!r} not allowed for this task"
    e = validate(call["args"], TOOLS[n]["schema"])
    if e: return "error", "; ".join(e)
    try: return "ok", TOOLS[n]["fn"](**call["args"])
    except Exception as ex: return "error", f"{type(ex).__name__}: {ex}"
def parse_json(text):
    """Take the first {...} block that parses as JSON."""
    for m in re.finditer(r"\{", text):
        depth = 0
        for j in range(m.start(), len(text)):
            depth += text[j] == "{"; depth -= text[j] == "}"
            if depth == 0:
                try: return json.loads(text[m.start():j + 1])
                except json.JSONDecodeError: break
    return None
# ---------- the agent loop ----------
def run_agent(task, policy, max_steps=6, allowed=None, trace=None):
    """observe -> think/act (policy returns {'tool','args'} or {'final': answer}) -> observe tool result ... until final or step limit."""
    hist = []
    for step in range(max_steps):
        act = policy(task, hist)
        if act is None: hist.append(("parse_error", None, None)); continue
        if "final" in act: return {"answer": act["final"], "steps": step + 1, "hist": hist, "status": "final"}
        st, res = call_tool(act, allowed); hist.append((act, st, res))
    return {"answer": None, "steps": max_steps, "hist": hist, "status": "step_limit"}
# ---------- local small model ----------
_M = {}
def smol():
    if not _M:
        sys.path.insert(0, os.path.join(ROOT, "code", "week-15")); import _hf, torch
        torch.set_num_threads(4); tok, m = _hf.load(_hf.INSTRUCT); _M.update(tok=tok, m=m.eval(), torch=torch)
    return _M["tok"], _M["m"], _M["torch"]
def generate(messages, max_new=60):
    tok, m, torch = smol()
    ids = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    with torch.no_grad(): out = m.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip()
# ---------- synthetic task suite (multi-step; gold answers computed from CITIES) ----------
def suite():
    C = A_ = CITIES; T = []
    pairs = [("Avalon", "Brindle"), ("Corvo", "Dunmere"), ("Elsby", "Farrow"), ("Brindle", "Corvo"), ("Dunmere", "Elsby"), ("Farrow", "Avalon")]
    for a, b in pairs: T.append(dict(kind="sum", q=f"What is the combined population of {a} and {b}?", gold=C[a]["population"] + C[b]["population"], cities=[a, b], field="population", op="+"))
    for a, b in pairs: T.append(dict(kind="diff", q=f"How many degrees warmer is {a} than {b}?", gold=C[a]["temp_c"] - C[b]["temp_c"], cities=[a, b], field="temp_c", op="-"))
    for a, b in pairs: T.append(dict(kind="same", q=f"Are {a} and {b} in the same country? Answer yes or no.", gold="yes" if C[a]["country"] == C[b]["country"] else "no", cities=[a, b], field="country", op="=="))
    for a in CITIES: T.append(dict(kind="double", q=f"What is twice the population of {a}?", gold=2 * C[a]["population"], cities=[a], field="population", op="*2"))
    return T
def correct(ans, gold):
    if isinstance(gold, str): return isinstance(ans, str) and ans.strip().lower().startswith(gold)
    try: return abs(float(ans) - gold) < 1e-6
    except (TypeError, ValueError): return False
def scripted_react(task, hist):
    """An oracle-like scripted policy: reads the task's structured fields (NOT the text), calls lookups, then calculator. Upper bound."""
    cs = task["cities"]
    if len(hist) < len(cs): return {"tool": "lookup", "args": {"city": cs[len(hist)], "field": task["field"]}}
    v = [h[2] for h in hist[:len(cs)]]
    if task["op"] == "==": return {"final": "yes" if v[0] == v[1] else "no"}
    if len(hist) == len(cs):
        e = f"{v[0]}{task['op']}{v[1]}" if len(v) == 2 else f"{v[0]}{task['op']}"; return {"tool": "calculator", "args": {"expression": e}}
    return {"final": hist[-1][2]}
