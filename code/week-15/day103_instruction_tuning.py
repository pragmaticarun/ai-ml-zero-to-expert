"""Day 103: chat templates and supervised instruction tuning (SFT) of the BASE SmolLM2-135M with LoRA on a synthetic task
("reverse the order of these words"), with and without masking the prompt out of the loss."""
import os, random, time, torch, _hf
from _lora import add_lora, trainable
from transformers import AutoTokenizer
STEPS = int(os.environ.get("STEPS", 250)); itok = AutoTokenizer.from_pretrained(_hf.INSTRUCT)
msgs = [{"role": "user", "content": "Reverse the order of these words: red fish blue"}]
print("Instruct chat template renders as:\n" + itok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
WORDS = "apple river stone cloud green tiger music paper window happy north candle silver orange garden winter bread planet yellow horse".split() + "table smile ocean forest light quick brown seven summer castle dream little mountain purple coffee rocket lemon spring shadow".split()
rng = random.Random(0)
def ex():
    w = rng.sample(WORDS, rng.randint(3, 5)); return "Reverse the order of these words: " + " ".join(w), " ".join(w[::-1])
train = [ex() for _ in range(2000)]; test = [ex() for _ in range(50)]; test = [t for t in test if t not in train][:40]
import json; os.makedirs("out", exist_ok=True); json.dump(test, open("out/day103_test.json", "w"))
def render(p, a=None):
    s = itok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True); return s if a is None else s + a + "<|im_end|>"
def encode(p, a, mask):
    pi = itok(render(p))["input_ids"]; full = itok(render(p, a))["input_ids"]; lab = list(full)
    if mask: lab[:len(pi)] = [-100] * len(pi)
    return full, lab
def collate(rows):
    L = max(len(f) for f, _ in rows); pad = itok.pad_token_id or 0
    ids = torch.tensor([f + [pad] * (L - len(f)) for f, _ in rows]); lab = torch.tensor([l + [-100] * (L - len(l)) for _, l in rows])
    return ids, (ids != pad) | (lab != -100), lab
@torch.no_grad()
def exact(m, n=40):
    ok = 0
    for p, a in test[:n]:
        x = itok(render(p), return_tensors="pt"); out = m.generate(**x, max_new_tokens=16, do_sample=False, pad_token_id=itok.eos_token_id, eos_token_id=[itok.eos_token_id, itok.convert_tokens_to_ids("<|im_end|>")])
        ok += itok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip() == a
    return ok / n
@torch.no_grad()
def prefix(m, n=40):   # lenient: the first len(answer) words are right, ignoring what follows
    ok = 0
    for p, a in test[:n]:
        x = itok(render(p), return_tensors="pt"); out = m.generate(**x, max_new_tokens=16, do_sample=False, pad_token_id=itok.eos_token_id, eos_token_id=[itok.eos_token_id, itok.convert_tokens_to_ids("<|im_end|>")])
        ok += itok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True).split()[:len(a.split())] == a.split()
    return ok / n
_, base = _hf.load(); print(f"base model, zero-shot with chat template: exact match {exact(base):.2f} on {len(test)} held-out prompts")
_, inst = _hf.load(_hf.INSTRUCT); print(f"SmolLM2-135M-Instruct, zero-shot: exact match {exact(inst):.2f}"); del inst
p, a = train[0]; f, l = encode(p, a, True); print(f"example: {len(f)} tokens, of which {sum(x != -100 for x in l)} are answer tokens counted in the masked loss")
for mask in (True, False):
    _, m = _hf.load(); m = add_lora(m, ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"), r=16)
    opt = torch.optim.AdamW([q for q in m.parameters() if q.requires_grad], lr=1e-3); r2 = random.Random(1); m.train(); t = time.perf_counter()
    for s in range(STEPS):
        ids, att, lab = collate([encode(*r2.choice(train), mask) for _ in range(8)]); loss = m(input_ids=ids, attention_mask=att.long(), labels=lab).loss
        opt.zero_grad(); loss.backward(); opt.step()
    m.eval(); el = time.perf_counter() - t; acc = exact(m); print(f"SFT LoRA r=16, {STEPS} steps x 8, prompt masked={mask}: {el:.0f} s, held-out exact match {acc:.2f}, first-words match {prefix(m):.2f}")
    if mask:
        torch.save({k: v for k, v in m.state_dict().items() if ".A" in k[-2:] or ".B" in k[-2:]}, "out/day103_lora.pt")
        x = itok(render("Reverse the order of these words: coffee tiger window"), return_tensors="pt")
        print("sample:", repr(itok.decode(m.generate(**x, max_new_tokens=16, do_sample=False, pad_token_id=itok.eos_token_id, eos_token_id=[itok.eos_token_id, itok.convert_tokens_to_ids("<|im_end|>")])[0][x['input_ids'].shape[1]:], skip_special_tokens=True)))
