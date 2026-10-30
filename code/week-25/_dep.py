"""Week 25 / capstone shared helpers."""
import sys, os, json, re, time
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", ".."))
CAP = os.path.join(H, "..", "capstone")
OUT = os.path.join(H, "out"); COUT = os.path.join(CAP, "out")
os.makedirs(OUT, exist_ok=True); os.makedirs(COUT, exist_ok=True)
sys.path.insert(0, os.path.join(H, ".."))
sys.path.insert(0, CAP)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
WEEKS = os.path.join(ROOT, "weeks")
def week_files():
    return sorted(f for f in os.listdir(WEEKS) if re.match(r"week-\d+\.md$", f))
def chunk_weeks(max_chars=800):
    """Split each week markdown into overlapping-ish sections by ## headers, then by size."""
    chunks = []
    for wf in week_files():
        text = open(os.path.join(WEEKS, wf), encoding="utf-8", errors="replace").read()
        parts = re.split(r"(?=^## )", text, flags=re.M)
        for i, p in enumerate(parts):
            p = p.strip()
            if len(p) < 40: continue
            for j in range(0, len(p), max_chars):
                c = p[j:j+max_chars].strip()
                if len(c) < 40: continue
                chunks.append({"id": f"{wf}#{i}-{j}", "week": wf, "text": c})
    return chunks
def save_json(path, obj):
    open(path, "w").write(json.dumps(obj, indent=2, default=str)); print(f"[saved] {path}")
