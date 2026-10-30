"""Week 17 RAG helpers: parse the course notes into days/sections, build an eval set, chunk, embed (MiniLM, mean pooling), BM25."""
import os, re, glob, math, collections, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf")); torch.set_num_threads(int(os.environ.get("THREADS", 8)))
EMB = "sentence-transformers/all-MiniLM-L6-v2"
def days(weeks=range(1, 17)):
    """Return {day_id: [(section_title, text), ...]} from weeks/week-XX.md (only days with real content)."""
    out = {}
    for w in weeks:
        t = open(os.path.join(ROOT, "weeks", f"week-{w:02d}.md")).read()
        for blk in re.split(r"^## Day ", t, flags=re.M)[1:]:
            m = re.match(r"(\d+)[:\s]+(.*)", blk); d = int(m.group(1)); title = m.group(2).strip()
            blk = re.split(r"^## Weekly quiz", blk, flags=re.M)[0]
            secs = re.split(r"^### ", blk, flags=re.M); body = [("intro", f"Day {d}: {title}\n" + secs[0].split("\n", 1)[-1])]
            body += [(s.split("\n", 1)[0].strip(), s.split("\n", 1)[-1]) for s in secs[1:]]
            if sum(len(b.split()) for _, b in body) > 300: out[d] = (title, body)
    return out
def eval_set(D):
    """Queries = the interview-style question of each day (first quoted sentence); that section is EXCLUDED from the index."""
    qs = []
    for d, (title, body) in D.items():
        for name, txt in body:
            if name.startswith("Interview-style"):
                m = re.search(r"[\"“]([^\"”]{20,400})[\"”]", txt)
                if m: qs.append((m.group(1).strip(), d))
    return qs
def chunks(D, size=128, overlap=32, by="words"):
    """by='words': fixed windows over each day's text; by='section': one chunk per ### section (split further if > 2*size)."""
    out = []
    for d, (title, body) in D.items():
        secs = [(n, t) for n, t in body if not n.startswith("Interview-style")]
        if by == "section":
            for n, t in secs:
                w = t.split()
                for i in range(0, max(len(w), 1), 2 * size): out.append((d, f"Day {d} {title} / {n}: " + " ".join(w[i:i + 2 * size])))
        else:
            w = " ".join(t for _, t in secs).split()
            for i in range(0, len(w), size - overlap): out.append((d, f"Day {d} {title}: " + " ".join(w[i:i + size])))
    return out
_m = {}
@torch.no_grad()
def embed(texts, bs=64):
    from transformers import AutoTokenizer, AutoModel
    if not _m: _m["tok"] = AutoTokenizer.from_pretrained(EMB); _m["mod"] = AutoModel.from_pretrained(EMB).eval()
    outs = []
    for i in range(0, len(texts), bs):
        e = _m["tok"](texts[i:i + bs], padding=True, truncation=True, max_length=256, return_tensors="pt"); h = _m["mod"](**e).last_hidden_state
        a = e["attention_mask"][..., None].float(); outs.append(torch.nn.functional.normalize((h * a).sum(1) / a.sum(1), dim=-1))
    return torch.cat(outs)
TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())
class BM25:
    def __init__(s, docs, k1=1.5, b=0.75):
        s.d = [collections.Counter(TOK(x)) for x in docs]; s.len = [sum(c.values()) for c in s.d]; s.avg = sum(s.len) / len(s.len); s.k1, s.b = k1, b
        df = collections.Counter(t for c in s.d for t in c); N = len(docs); s.idf = {t: math.log(1 + (N - n + 0.5) / (n + 0.5)) for t, n in df.items()}
    def scores(s, q):
        qt = TOK(q); return torch.tensor([sum(s.idf.get(t, 0) * c[t] * (s.k1 + 1) / (c[t] + s.k1 * (1 - s.b + s.b * L / s.avg)) for t in qt if t in c) for c, L in zip(s.d, s.len)])
def metrics(rank_days, gold):
    """rank_days: list of lists of day ids (best first) per query. Recall@k = gold day among top-k chunks; MRR on first hit."""
    r1 = sum(g in r[:1] for r, g in zip(rank_days, gold)) / len(gold); r5 = sum(g in r[:5] for r, g in zip(rank_days, gold)) / len(gold)
    mrr = sum(1 / (r.index(g) + 1) if g in r else 0 for r, g in zip(rank_days, gold)) / len(gold); return r1, r5, mrr
