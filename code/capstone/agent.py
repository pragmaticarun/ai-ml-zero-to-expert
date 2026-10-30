"""Tiny tool-using agent with allow-list and injection guards."""
import os, re, json, ast
from rag import CourseRAG
TOOLS = {
  "calc": {"schema": {"expr": "str"}, "doc": "Evaluate a Python arithmetic expression (numbers and + - * / ** () only)."},
  "course_search": {"schema": {"query": "str"}, "doc": "Search course notes; returns top-3 chunk ids and snippets."},
  "get_day_title": {"schema": {"day": "int"}, "doc": "Return the SYLLABUS title for a day number if present in notes."},
}
INJECTION_PAT = re.compile(r"(ignore (all |any )?(previous|prior|above) (instructions|rules)|system prompt|reveal (the )?(system|hidden)|jailbreak)", re.I)
def guard_user(text):
    if INJECTION_PAT.search(text or ""): return False, "blocked: injection pattern"
    if len(text or "") > 2000: return False, "blocked: too long"
    return True, "ok"
def run_calc(expr):
    tree = ast.parse(expr, mode="eval")
    allowed = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
               ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.FloorDiv,
               ast.USub, ast.UAdd)
    for n in ast.walk(tree):
        if not isinstance(n, allowed):
            raise ValueError("disallowed")
    return str(eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, {}))
class Agent:
    def __init__(self, rag=None):
        self.rag = rag or CourseRAG(use_lora=False)
        self.syllabus = {}
        # best-effort day titles from week files
        import re as _re
        root = os.path.join(os.path.dirname(__file__), "..", "..", "weeks")
        for wf in os.listdir(root):
            if not wf.endswith(".md"): continue
            for m in _re.finditer(r"## Day (\d+):\s*(.+)", open(os.path.join(root, wf), encoding="utf-8", errors="replace").read()):
                self.syllabus[int(m.group(1))] = m.group(2).strip()
    def call_tool(self, name, args):
        if name not in TOOLS: return {"error": "unknown tool"}
        if name == "calc":
            try: return {"result": run_calc(args["expr"])}
            except Exception as e: return {"error": f"calc rejected: {type(e).__name__}"}
        if name == "course_search":
            hits = self.rag.retrieve(args["query"], 3)
            return {"hits": [{"id": h["id"], "score": h["score"], "snip": h["text"][:160]} for h in hits]}
        if name == "get_day_title":
            d = int(args["day"]); return {"day": d, "title": self.syllabus.get(d, "unknown")}
    def run(self, user):
        ok, why = guard_user(user)
        if not ok: return {"blocked": True, "reason": why, "answer": None}
        # pattern: if looks like math -> calc; if "day N" -> get_day_title; else RAG answer
        m = re.search(r"day\s+(\d+)", user, re.I)
        trace = []
        if re.fullmatch(r"[\d\s\+\-\*/\(\)\.\*%]+", user.strip()) or user.lower().startswith("calc:"):
            expr = user.split(":",1)[-1].strip(); r = self.call_tool("calc", {"expr": expr}); trace.append(("calc", expr, r))
            return {"blocked": False, "trace": trace, "answer": r.get("result", r)}
        if m and "title" in user.lower():
            r = self.call_tool("get_day_title", {"day": int(m.group(1))}); trace.append(("get_day_title", m.group(1), r))
            return {"blocked": False, "trace": trace, "answer": r["title"]}
        # default: course_search + generate
        r = self.call_tool("course_search", {"query": user}); trace.append(("course_search", user, r))
        ans = self.rag.answer(user, k=3, max_new=40, use_context=True)
        return {"blocked": False, "trace": trace, "answer": ans["answer"], "hits": ans["hits"], "grounded_flag": ans["grounded"]}
