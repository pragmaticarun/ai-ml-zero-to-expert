"""Retrieve top-k course chunks; optional generate with SmolLM2 (+ optional LoRA)."""
import os, numpy as np, torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
ROOT = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "out")
MID = "HuggingFaceTB/SmolLM2-135M-Instruct"
class CourseRAG:
    def __init__(self, use_lora=False):
        self.enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.X = np.load(os.path.join(OUT, "embeddings.npy"))
        import json; self.chunks = json.load(open(os.path.join(OUT, "chunks.json")))
        self.tok = AutoTokenizer.from_pretrained(MID)
        self.model = AutoModelForCausalLM.from_pretrained(MID)
        if use_lora:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, os.path.join(ROOT, "adapters", "course_lora"))
        self.model.eval()
    def retrieve(self, q, k=3):
        qe = self.enc.encode([q], convert_to_numpy=True)[0]; qe = qe / (np.linalg.norm(qe)+1e-9)
        scores = self.X @ qe; idx = scores.argsort()[::-1][:k]
        return [{"score": float(scores[i]), **self.chunks[i]} for i in idx]
    def answer(self, q, k=3, max_new=40, use_context=True):
        hits = self.retrieve(q, k)
        if use_context:
            ctx = "\n\n".join(f"[{h['id']}] {h['text'][:400]}" for h in hits)
            user = f"Answer using ONLY the context. Cite chunk ids.\n\nContext:\n{ctx}\n\nQuestion: {q}"
        else:
            user = q
        prompt = self.tok.apply_chat_template([{"role":"user","content":user}], add_generation_prompt=True, tokenize=False)
        ids = self.tok(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids
        with torch.no_grad():
            out = self.model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        return {"answer": text, "hits": hits, "grounded": any(h["id"].split("#")[0] in text or h["id"] in text for h in hits) or any(h["week"].replace(".md","") in text for h in hits)}
