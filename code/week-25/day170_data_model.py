"""Day 170: build course corpus index + LoRA fine-tune SmolLM2 on course Q&A (CPU, short)."""
from _dep import *
import torch, time, numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
sys.path.insert(0, CAP); from corpus_qa import QA
chunks = chunk_weeks(); print(f"(1) chunks: {len(chunks)}")
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
t0 = time.perf_counter(); X = enc.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=False, batch_size=64); t_ix = time.perf_counter() - t0
print(f"    index embed: {t_ix:.2f} s for {len(chunks)} x {X.shape[1]}; L2-normalising")
X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
meta = [{"id": c["id"], "week": c["week"], "text": c["text"]} for c in chunks]
np.save(os.path.join(COUT, "embeddings.npy"), X); save_json(os.path.join(COUT, "chunks.json"), meta)
print(f"    saved embeddings.npy ({X.nbytes/1e6:.1f} MB) and chunks.json")
# retrieval sanity on QA
Qe = enc.encode([q for q,_,_ in QA], convert_to_numpy=True); Qe /= (np.linalg.norm(Qe, axis=1, keepdims=True)+1e-9)
hits = (Qe @ X.T).argsort(axis=1)[:, ::-1][:, :3]
rec = 0
for i, (q, a, key) in enumerate(QA):
    weeks = [meta[j]["week"] for j in hits[i]]
    ok = any(key in w for w in weeks) or any(key.replace("week-","") in meta[j]["text"].lower()[:200] for j in hits[i])
    # better: keyword in retrieved text
    ok = any(key in meta[j]["week"] for j in hits[i])
    rec += int(ok)
print(f"    retrieval recall@3 by week-tag on {len(QA)} QA: {rec}/{len(QA)} = {rec/len(QA):.3f}")
# LoRA
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MID); base = AutoModelForCausalLM.from_pretrained(MID)
cfg = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj","v_proj"])
model = get_peft_model(base, cfg); model.print_trainable_parameters()
# build short supervised samples
def pack(q, a):
    msgs = [{"role":"user","content":q},{"role":"assistant","content":a}]
    text = tok.apply_chat_template(msgs, tokenize=False)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=256).input_ids[0]
    return ids
train_ids = [pack(q,a) for q,a,_ in QA]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
model.train(); losses = []; t0 = time.perf_counter(); steps = 60
for s in range(steps):
    ids = train_ids[s % len(train_ids)].unsqueeze(0)
    out = model(ids, labels=ids); loss = out.loss; loss.backward(); opt.step(); opt.zero_grad(); losses.append(loss.item())
    if (s+1) % 20 == 0: print(f"    step {s+1}/{steps} loss {np.mean(losses[-20:]):.3f}")
t_train = time.perf_counter() - t0
path = os.path.join(CAP, "adapters", "course_lora"); model.save_pretrained(path); tok.save_pretrained(path)
print(f"(2) LoRA train {steps} steps in {t_train:.1f} s ({t_train/60:.2f} min); saved {path}")
print(f"    loss start {losses[0]:.3f} end {np.mean(losses[-10:]):.3f}")
# quick gen check
model.eval()
from peft import PeftModel
# already peft
def ask(q):
    prompt = tok.apply_chat_template([{"role":"user","content":q}], add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors="pt").input_ids
    with torch.no_grad(): out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
for q in ("What is RAG?", "What is demographic parity?"):
    print(f"    Q: {q}\n    A: {ask(q)[:160]}")
save_json(os.path.join(OUT, "day170_stats.json"), {"n_chunks":len(chunks),"embed_s":round(t_ix,2),"recall_at_3":round(rec/len(QA),3),"lora_steps":steps,"lora_s":round(t_train,1),"loss_start":round(losses[0],3),"loss_end":round(float(np.mean(losses[-10:])),3)})
