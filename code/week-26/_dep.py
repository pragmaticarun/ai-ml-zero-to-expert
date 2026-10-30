"""Week 26 helpers — reuse Week 25 / capstone paths."""
import sys, os, json, time
H = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(H, "..", ".."))
CAP = os.path.join(H, "..", "capstone")
W25 = os.path.join(H, "..", "week-25")
OUT = os.path.join(H, "out"); COUT = os.path.join(CAP, "out"); W25OUT = os.path.join(W25, "out")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, CAP); sys.path.insert(0, W25)
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
def save_json(path, obj):
    open(path, "w").write(json.dumps(obj, indent=2, default=str)); print(f"[saved] {path}")
def load_json(path):
    return json.load(open(path))
