"""Week 22 LLMOps helpers (CPU scale): a content-hashed prompt registry, a cached eval harness with a paired regression gate,
a tiny tracer (JSONL spans with timings and token counts), a cost model with ASSUMED prices, and rule-based guardrails.
Reuses Week 20's agent tools/SmolLM2 wrapper and Week 17's RAG helpers."""
import os, sys, json, time, hashlib, re, math, uuid, contextlib
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
for w in ("week-20", "week-17"): sys.path.insert(0, os.path.join(ROOT, "code", w))
import _agent as A   # tools, validator, SmolLM2-135M-Instruct generate()
OUT = os.path.join(HERE, "out"); os.makedirs(OUT, exist_ok=True)
def sha(obj): return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:12]
# ---------- prompt registry: every prompt version is immutable and addressed by the hash of its content ----------
class Registry:
    def __init__(s, path): s.path = path; s.items = {}; s.labels = {}
    def register(s, name, system, shots, note=""):
        p = {"name": name, "system": system, "shots": shots}; h = sha(p)
        s.items[h] = dict(p, hash=h, note=note); s.labels.setdefault(name, []).append(h)
        with open(s.path, "a") as f: f.write(json.dumps({"hash": h, "name": name, "note": note}) + "\n")
        return h
    def messages(s, h, user):
        p = s.items[h]; m = [{"role": "system", "content": p["system"]}]
        for q, a in p["shots"]: m += [{"role": "user", "content": q}, {"role": "assistant", "content": json.dumps(a)}]
        return m + [{"role": "user", "content": user}]
# ---------- cached generation: key = (model, prompt hash, input, decoding params) ----------
class Cache:
    def __init__(s, path): s.path = path; s.d = {}; s.hits = s.miss = 0
    def __enter__(s):
        if os.path.exists(s.path): s.d = json.load(open(s.path))
        return s
    def __exit__(s, *a): json.dump(s.d, open(s.path, "w"))
    def get(s, key, fn):
        k = sha(key)
        if k in s.d: s.hits += 1; return s.d[k]
        s.miss += 1; s.d[k] = fn(); return s.d[k]
# ---------- tracing ----------
class Tracer:
    def __init__(s, path): s.path = path; s.spans = []; s.stack = []
    @contextlib.contextmanager
    def span(s, name, **attrs):
        sp = {"trace": s.stack[0]["trace"] if s.stack else uuid.uuid4().hex[:8], "id": uuid.uuid4().hex[:6], "parent": s.stack[-1]["id"] if s.stack else None,
              "name": name, **attrs}; t = time.perf_counter(); s.stack.append(sp)
        try: yield sp
        except Exception as e: sp["error"] = f"{type(e).__name__}: {e}"; raise
        finally:
            sp["ms"] = (time.perf_counter() - t) * 1e3; s.stack.pop(); s.spans.append(sp)
            with open(s.path, "a") as f: f.write(json.dumps(sp, default=str) + "\n")
def pct(xs, q):
    xs = sorted(xs); return xs[min(len(xs) - 1, int(math.ceil(q * len(xs))) - 1)] if xs else float("nan")
# ---------- cost: ASSUMED prices (illustrative, not any vendor's list price) ----------
PRICE_IN, PRICE_OUT = 0.50, 1.50   # $ per million tokens, assumed
def cost(tin, tout): return tin * PRICE_IN / 1e6 + tout * PRICE_OUT / 1e6
# ---------- guardrails ----------
INJ = [r"ignore (all |any )?(previous|prior|above) (instructions|rules)", r"system prompt", r"you are now", r"disregard", r"send (an )?email to", r"reveal"]
PII = {"email": r"[\w.+-]+@[\w-]+\.[\w.]+", "phone": r"\b\d{3}[-. ]?\d{3}[-. ]?\d{4}\b", "card": r"\b(?:\d[ -]?){13,16}\b"}
def input_guard(text):
    hits = [p for p in INJ if re.search(p, text, re.I)]; pii = [k for k, p in PII.items() if re.search(p, text)]
    return {"block": bool(hits), "injection": hits, "pii": pii}
def redact(text):
    for k, p in PII.items(): text = re.sub(p, f"[{k.upper()}]", text)
    return text
def output_guard(call, allowed):
    """Validate a model's tool call before execution: schema + allow-list (side-effect tools never auto-run)."""
    if call is None: return "unparseable"
    st, msg = ("ok", None)
    if call.get("tool") not in allowed: return f"tool {call.get('tool')!r} not allowed"
    if A.TOOLS[call["tool"]]["side_effect"]: return "side-effect tool needs human approval"
    e = A.validate(call.get("args"), A.TOOLS[call["tool"]]["schema"]); return "; ".join(e) if e else None
