"""Shared helpers for Week 16: synthetic preference data, chat rendering, sequence log-probs."""
import os, sys, random, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "week-15"))
import _hf  # sets HF_HOME, threads
from _lora import add_lora, trainable, LoRALinear
from transformers import AutoTokenizer
itok = AutoTokenizer.from_pretrained(_hf.INSTRUCT); IM_END = itok.convert_tokens_to_ids("<|im_end|>")
LIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
WORDS = ("apple river stone cloud green tiger music paper window happy north candle silver orange garden winter bread planet yellow horse "
         "table smile ocean forest light quick brown seven summer castle dream little mountain purple coffee rocket lemon spring shadow").split()
def task(rng):
    w = rng.sample(WORDS, rng.randint(3, 5)); return "Reverse the order of these words: " + " ".join(w), " ".join(w[::-1]), " ".join(w)
def splits(seed=0, n_train=1000, n_test=40):
    rng = random.Random(seed); tr = [task(rng) for _ in range(n_train)]; seen = {t[0] for t in tr}
    te = [t for t in (task(rng) for _ in range(3 * n_test)) if t[0] not in seen][:n_test]; return tr, te
def render(p): return itok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
def pair_ids(p, resp):
    pi = itok(render(p))["input_ids"]; full = itok(render(p) + resp + "<|im_end|>")["input_ids"]; return full, len(pi)
def logp_batch(m, items):
    """items: list of (ids, prompt_len). Returns the summed log-prob of the response tokens for each item."""
    L = max(len(i) for i, _ in items); pad = itok.pad_token_id or 0
    x = torch.tensor([i + [pad] * (L - len(i)) for i, _ in items]); att = torch.tensor([[1] * len(i) + [0] * (L - len(i)) for i, _ in items])
    lp = m(input_ids=x, attention_mask=att).logits[:, :-1].log_softmax(-1); tgt = x[:, 1:]; tok = lp.gather(-1, tgt[..., None])[..., 0]
    mask = torch.zeros_like(tok)
    for b, (i, pl) in enumerate(items): mask[b, pl - 1:len(i) - 1] = 1
    return (tok * mask).sum(1)
@torch.no_grad()
def generate(m, p, n=16):
    x = itok(render(p), return_tensors="pt"); o = m.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=itok.eos_token_id, eos_token_id=[itok.eos_token_id, IM_END])
    return itok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
def exact(m, te): return sum(generate(m, p) == a for p, a, _ in te) / len(te)
