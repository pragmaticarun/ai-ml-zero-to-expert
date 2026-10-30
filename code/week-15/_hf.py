"""Shared helpers for weeks 15-17: local HF cache, model loading, Shakespeare token batches."""
import os, sys, math, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
os.environ.setdefault("HF_HOME", os.path.join(ROOT, "data", "cache", "hf"))
torch.set_num_threads(int(os.environ.get("THREADS", "4")))
BASE = "HuggingFaceTB/SmolLM2-135M"; INSTRUCT = "HuggingFaceTB/SmolLM2-135M-Instruct"
sys.path.insert(0, os.path.join(ROOT, "code", "week-14"))

def load(name=BASE, dtype=torch.float32):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(name); m = AutoModelForCausalLM.from_pretrained(name, dtype=dtype); m.eval(); return tok, m

def shakespeare_ids(tok):
    from _gpt import char_data
    text = char_data()[0]; n = int(0.9 * len(text))
    return torch.tensor(tok(text[:n])["input_ids"]), torch.tensor(tok(text[n:])["input_ids"])

@torch.no_grad()
def perplexity(m, ids, T=256, n_windows=16):
    """Mean next-token loss over n_windows non-overlapping windows from the start of ids."""
    tot, cnt = 0.0, 0
    for i in range(n_windows):
        x = ids[i * T:(i + 1) * T][None]
        if x.shape[1] < 2: break
        out = m(input_ids=x, labels=x); tot += out.loss.item() * (x.shape[1] - 1); cnt += x.shape[1] - 1
    return math.exp(tot / cnt), tot / cnt

def batch(ids, T, bs, g):
    ix = torch.randint(len(ids) - T - 1, (bs,), generator=g); return torch.stack([ids[i:i + T] for i in ix])
