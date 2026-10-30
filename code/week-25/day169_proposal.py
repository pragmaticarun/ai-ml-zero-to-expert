"""Day 169: Capstone proposal, success metrics, and budget table (measured baselines)."""
from _dep import *
import torch, time, numpy as np
from sentence_transformers import SentenceTransformer
# Measure: corpus size, embed time, SmolLM2 forward latency, RSS-ish, MiniLM size
chunks = chunk_weeks(); print(f"corpus: {len(week_files())} week files -> {len(chunks)} chunks")
chars = sum(len(c["text"]) for c in chunks); print(f"total chunk chars: {chars}")
enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
t0 = time.perf_counter(); emb = enc.encode([c["text"] for c in chunks[:64]], convert_to_numpy=True, show_progress_bar=False); t_emb = time.perf_counter() - t0
print(f"embed 64 chunks: {t_emb:.3f} s ({t_emb/64*1000:.1f} ms/chunk); dim {emb.shape[1]}")
# full corpus embed estimate
t_full_est = t_emb / 64 * len(chunks)
from transformers import AutoTokenizer, AutoModelForCausalLM
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
tok = AutoTokenizer.from_pretrained(MID); model = AutoModelForCausalLM.from_pretrained(MID); model.eval()
nparam = sum(p.numel() for p in model.parameters()); print(f"SmolLM2 params: {nparam:,}")
prompt = tok.apply_chat_template([{"role":"user","content":"What is gradient descent in one sentence?"}], add_generation_prompt=True, tokenize=False)
ids = tok(prompt, return_tensors="pt").input_ids
# warm
with torch.no_grad(): model.generate(ids, max_new_tokens=8, do_sample=False, pad_token_id=tok.eos_token_id)
t0 = time.perf_counter()
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
t_gen = time.perf_counter() - t0
n_new = out.shape[1] - ids.shape[1]; print(f"generate {n_new} tokens: {t_gen:.3f} s ({t_gen/max(n_new,1)*1000:.1f} ms/tok)")
# budgets
budget = {
  "name": "Course Assistant capstone",
  "hardware": "CPU-only 8-core Xeon, ~15GB RAM, no GPU",
  "success_metrics": {
    "retrieval_recall_at_3": ">= 0.70 on a 20-question hand-labelled set",
    "grounded_answer_rate": ">= 0.60 (answer cites a retrieved chunk id)",
    "tool_success_rate": ">= 0.80 on allow-listed tool tasks",
    "injection_block_rate": ">= 0.80 on a 10-prompt attack set (0 FP on 10 benign)",
    "p50_rag_latency_s": "<= 8.0 (retrieve+generate 40 toks, CPU)",
    "embed_index_build_s": "<= 120",
    "lora_train_minutes": "<= 15",
    "int8_score_speedup_vs_numpy": ">= 1.5x at batch 256",
    "eval_gate": "FAIL release if recall@3 drops >10% relative or injection block <0.70",
    "rai": "model card + audit report filed; no real-user deployment claim"
  },
  "measured_baselines": {
    "n_chunks": len(chunks), "chunk_chars": chars,
    "embed_64_s": round(t_emb, 3), "embed_ms_per_chunk": round(t_emb/64*1000, 1),
    "embed_full_est_s": round(t_full_est, 1),
    "smol_params": nparam, "gen_40tok_s": round(t_gen, 3), "ms_per_tok": round(t_gen/max(n_new,1)*1000, 1)
  },
  "budget_table": [
    {"item": "RAM for MiniLM + SmolLM2 fp32", "limit": "<= 4 GiB process RSS growth", "note": "fp32 135M ~0.5GiB weights; measure Day 172"},
    {"item": "LoRA rank", "limit": "r<=8, target modules q_proj/v_proj", "note": "keeps adapter <10MB"},
    {"item": "max_new_tokens", "limit": "40 default / 80 hard", "note": "CPU latency budget"},
    {"item": "retrieval top_k", "limit": "3", "note": "prompt size"},
    {"item": "tools", "limit": "calc, course_search, get_day_title", "note": "allow-list only"},
  ]
}
save_json(os.path.join(OUT, "day169_budget.json"), budget)
save_json(os.path.join(COUT, "budget.json"), budget)
print("BUDGET TABLE:")
for row in budget["budget_table"]: print(f"  {row['item']:40s} | {row['limit']}")
print("SUCCESS METRICS:")
for k,v in budget["success_metrics"].items(): print(f"  {k}: {v}")
print("BASELINES:", budget["measured_baselines"])
