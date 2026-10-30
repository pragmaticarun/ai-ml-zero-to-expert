"""Day 137: two tools with real back-ends. (1) search_notes: the Month-4 hybrid RAG (BM25 + MiniLM, RRF, cross-encoder rerank over weeks 1-16;
same code path as Day 120) exposed through a JSON-schema tool; scored on the 58 held-out note questions. (2) run_python: the _sandbox.py tool,
probed with benign and hostile snippets to show what the limits do and do not stop."""
import os, sys, time, torch, _agent as A, _sandbox as S
sys.path.insert(0, os.path.join(A.ROOT, "code", "week-17")); import _rag as R
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print("== (2) sandbox probes (run first; no model needed)")
probes = [("benign: sum of squares", "print(sum(i*i for i in range(1000)))"), ("exception", "1/0"),
          ("infinite loop", "while True: pass"), ("sleep (wall clock, no CPU)", "import time; time.sleep(30)"),
          ("memory bomb 2 GB", "x = bytearray(2*1024**3); print('allocated')"), ("write 5 MB file in cwd", "open('big','wb').write(b'0'*5*2**20); print('wrote')"),
          ("read env vars", "import os; print(len(os.environ), 'env vars')"), ("read a file outside cwd", "print(open('/etc/hostname').read().strip()[:3]+'...')"),
          ("write outside cwd", "open('/tmp/w20_escape_test.txt','w').write('x'); print('escaped write ok')")]
for name, code in probes:
    t0 = time.perf_counter(); r = S.run_python(code); print(f"  {name:28s}: rc={r['returncode']}, {time.perf_counter()-t0:.2f} s, output {r['output'][-80:]!r}", flush=True)
if os.path.exists("/tmp/w20_escape_test.txt"): os.remove("/tmp/w20_escape_test.txt"); print("  (removed /tmp/w20_escape_test.txt again)")
print("== (1) search_notes tool over the Month-4 RAG index")
t = time.perf_counter(); D = R.days(); C = R.chunks(D, size=128, overlap=32); docs = [c for _, c in C]; cday = torch.tensor([d for d, _ in C])
ce = R.embed(docs); bm = R.BM25(docs); CE = "cross-encoder/ms-marco-MiniLM-L6-v2"
ctok = AutoTokenizer.from_pretrained(CE); cem = AutoModelForSequenceClassification.from_pretrained(CE).eval()
print(f"  index: {len(docs)} chunks from {len(D)} days, {time.perf_counter()-t:.1f} s")
@torch.no_grad()
def search_notes(query, k=3):
    dn = (R.embed([query]) @ ce.T)[0]; lx = bm.scores(query); rr = torch.zeros_like(dn)
    for Sx in (dn, lx): rr[Sx.argsort(descending=True)] += 1 / (61 + torch.arange(len(Sx)).float())
    top = rr.topk(20).indices.tolist(); e = ctok([query] * 20, [docs[i] for i in top], padding=True, truncation=True, max_length=320, return_tensors="pt")
    best = [top[j] for j in cem(**e).logits[:, 0].argsort(descending=True)[:k].tolist()]
    return [{"day": int(cday[i]), "text": docs[i][:200]} for i in best]
A.TOOLS["search_notes"] = {"fn": search_notes, "side_effect": False, "description": "Search the course notes (weeks 1-16).",
    "schema": {"type": "object", "properties": {"query": {"type": "string"}, "k": {"type": "integer", "minimum": 1, "maximum": 10}}, "required": ["query"], "additionalProperties": False}}
print("  bad call:", A.call_tool({"tool": "search_notes", "args": {"query": "dropout", "k": 50}}))
EV = R.eval_set(D); t = time.perf_counter(); hit1 = hit3 = 0
for q, g in EV:
    st, res = A.call_tool({"tool": "search_notes", "args": {"query": q, "k": 3}}); days = [r["day"] for r in res]; hit1 += days[0] == g; hit3 += g in days
print(f"  {len(EV)} held-out questions via the tool: recall@1 {hit1/len(EV):.2f}, recall@3 {hit3/len(EV):.2f}; {(time.perf_counter()-t)/len(EV)*1000:.0f} ms/call")
st, res = A.call_tool({"tool": "search_notes", "args": {"query": "What does the importance ratio in PPO clip?", "k": 1}})
print(f"  out-of-index query (PPO is Week 19, not indexed): top day {res[0]['day']}, text {res[0]['text'][:90]!r}")
