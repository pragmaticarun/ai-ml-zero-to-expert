"""Day 104: evaluating the Day-103 fine-tuned model: task exact match, forgetting (perplexity), multiple-choice
log-likelihood scoring (sum vs length-normalised), and position bias of an LLM judge. Run day103 first."""
import json, torch, _hf
from _lora import add_lora
from transformers import AutoTokenizer
itok = AutoTokenizer.from_pretrained(_hf.INSTRUCT); test = json.load(open("out/day103_test.json")); LIN = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
def render(p): return itok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
_, base = _hf.load(); _, inst = _hf.load(_hf.INSTRUCT); _, tuned = _hf.load(); tuned = add_lora(tuned, LIN, r=16)
missing = tuned.load_state_dict(torch.load("out/day103_lora.pt"), strict=False); print(f"loaded LoRA tensors; unexpected keys: {len(missing.unexpected_keys)}")
models = {"base": base, "instruct": inst, "base+SFT-LoRA": tuned}
@torch.no_grad()
def gen(m, p, n=16):
    x = itok(render(p), return_tensors="pt"); o = m.generate(**x, max_new_tokens=n, do_sample=False, pad_token_id=itok.eos_token_id, eos_token_id=[itok.eos_token_id, itok.convert_tokens_to_ids("<|im_end|>")]); return itok.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
for k, m in models.items():
    outs = [gen(m, p) for p, _ in test]; em = sum(o == a for o, (_, a) in zip(outs, test)) / len(test)
    sameset = sum(sorted(o.split()) == sorted(a.split()) for o, (_, a) in zip(outs, test)) / len(test)
    print(f"{k:14s} reverse-words exact match {em:.2f}; right words in any order {sameset:.2f}")
tr, va = _hf.shakespeare_ids(itok)
for k, m in models.items(): print(f"{k:14s} Shakespeare val ppl {_hf.perplexity(m, va, n_windows=8)[0]:.2f}")
# multiple choice by log-likelihood (questions written for this course; answers are basic facts)
MC = [("The capital of France is", [" Paris", " Berlin", " Madrid", " Rome"]), ("Water freezes at a temperature of", [" 0 degrees Celsius", " 50 degrees Celsius", " 100 degrees Celsius", " 20 degrees Celsius"]),
      ("The largest planet in the Solar System is", [" Jupiter", " Mars", " Venus", " Mercury"]), ("Two plus three equals", [" five", " four", " six", " seven"]),
      ("The chemical symbol for gold is", [" Au", " Ag", " Fe", " Gd"]), ("A week has", [" seven days", " five days", " ten days", " six days"]),
      ("The opposite of hot is", [" cold", " warm", " loud", " fast"]), ("Bees make", [" honey", " milk", " silk", " wool"]),
      ("The Pacific is an", [" ocean", " island", " mountain", " desert"]), ("Shakespeare wrote", [" Hamlet", " War and Peace", " Don Quixote", " The Odyssey"]),
      ("The sun rises in the", [" east", " west", " north", " south"]), ("A triangle has", [" three sides", " four sides", " five sides", " six sides"]),
      ("The primary language spoken in Brazil is", [" Portuguese", " Spanish", " French", " Italian"]), ("Ice is the solid form of", [" water", " air", " salt", " iron"]),
      ("The number of continents usually counted is", [" seven", " three", " twelve", " twenty"]), ("Light travels faster than", [" sound", " nothing", " itself", " time"])]
@torch.no_grad()
def lls(m, q, opts):
    out = []
    for o in opts:
        qi = itok(q)["input_ids"]; full = itok(q + o)["input_ids"]; x = torch.tensor([full]); lp = m(input_ids=x).logits[0, :-1].log_softmax(-1)
        t = torch.tensor(full[1:]); tok_lp = lp[torch.arange(len(t)), t][len(qi) - 1:]; out.append((tok_lp.sum().item(), tok_lp.mean().item()))
    return out
for k, m in models.items():
    s = [lls(m, q, o) for q, o in MC]
    acc_sum = sum(max(range(4), key=lambda i: r[i][0]) == 0 for r in s) / len(MC); acc_mean = sum(max(range(4), key=lambda i: r[i][1]) == 0 for r in s) / len(MC)
    print(f"{k:14s} {len(MC)}-question MC accuracy: sum log-prob {acc_sum:.2f}, length-normalised {acc_mean:.2f} (chance 0.25)")
# judge position bias: instruct model compares a correct and a wrong answer in both orders
pairs = [(q + o[0], q + o[1]) for q, o in MC]; A, B = itok(" A")["input_ids"][-1], itok(" B")["input_ids"][-1]; picksA, consistent, correct = 0, 0, 0
for good, bad in pairs:
    res = []
    for first, second in ((good, bad), (bad, good)):
        p = f"Which statement is true?\nA: {first}\nB: {second}\nAnswer with A or B."
        x = itok(render(p) + "Answer:", return_tensors="pt")
        with torch.no_grad(): lg = inst(**x).logits[0, -1]
        res.append("A" if lg[A] > lg[B] else "B")
    picksA += (res[0] == "A") + (res[1] == "A"); consistent += res[0] != res[1]; correct += (res[0] == "A") + (res[1] == "B")
n = len(pairs); print(f"Instruct-as-judge on {n} pairs x 2 orders: picks A {picksA}/{2*n}; same verdict after swapping (consistent) {consistent}/{n}; correct {correct}/{2*n}")
