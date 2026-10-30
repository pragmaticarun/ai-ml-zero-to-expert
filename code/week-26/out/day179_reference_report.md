# Course Assistant — Reference Capstone Report (Day 179)

**Title:** On-device-style Course Assistant over `ai-ml-course` notes  
**Author:** reference implementation (course materials)  
**Date:** 2026-10-29 (Asia/Calcutta)  
**Hardware:** CPU-only 8-core Xeon, ~15 GB RAM, no GPU · Day 177 RSS peak **975.0 MiB**

## 1. Problem and non-goals
Build a CPU-feasible study helper that retrieves from this course's markdown notes, answers with a small LM, calls allow-listed tools, and refuses obvious injections — with regression gates and a model card.

Non-goals: web browsing; GPU training; public production deployment; demographic fairness claims; phone latency numbers from server CPUs.

## 2. Architecture
`weeks/*.md` → chunk (1467) → MiniLM-L6-v2 (384-d) → numpy/Rust score → top-3 → SmolLM2-135M-Instruct (optional LoRA r=8) → answer.  
Agent: `guard_user` → route to `calc` | `get_day_title` | `course_search`+RAG.  
Service: stdlib HTTP `:8765` `/health` `/metrics` `/ask`.  
Eval: Day 173 harness + Day 178 ablations. Artefacts under `code/capstone/`.

## 3. Budget vs measured
| Metric | Target | Measured | Pass? |
|--------|--------|----------|-------|
| recall@3 | ≥0.70 | **0.72** (18/25); Day178 **0.72** | YES |
| grounded | ≥0.60 | freeze **1.0**; context-on ablate **1.0** | YES |
| tools | ≥0.80 | **3/3**; Day178 **3/3** | YES |
| injection | ≥0.80; FP=0 | freeze rate **1.0** FP **0**; paraphrases Day177 **1/5** | YES on known-pattern; residual paraphrase risk |
| p50 RAG | ≤8 s | freeze **0.79 s**; Day177 **2.19 s** | YES |
| int8 speedup | ≥1.5× vs numpy-int8 | best **59.38×** | YES |
| embed build | ≤120 s | **16.11 s** (Day 170) | YES |
| LoRA time | ≤15 min | **14.8 s** | YES |
| gates | PASS | freeze **PASS**; polish tests **7/7**; service **10/10** | YES |

## 4. Ablations (Day 178)
- Recall @1/@3/@5 = **0.48 / 0.72 / 0.72** (@5 did not lift past @3).
- Context ON vs OFF grounded heuristic = **1.0 vs 1.0** (caveat: OFF still uses hit overlap — see FINAL_EVAL_REPORT).
- Calc allow/deny battery **6/6** after AST harden.
- LoRA-alone (Day 170): loss fell but answers nonsense — qualitative ablation, RAG kept.

## 5. What worked / what did not
See `code/capstone/out/FINAL_EVAL_REPORT.md`. Highlights: retrieval+gates+tools worked; LoRA-alone, paraphrase regex, tiny-N int8 vs BLAS, and grounded≠fact did not.

## 6. Responsible AI
Threat model (7 rows) + red-team 6/6 (Day 174) + model card `code/capstone/out/MODEL_CARD.md`.  
**Production decision: NO.** Course/portfolio: ACCEPT.

## 7. Demo script (90 s)
```bash
python code/week-25/day175_integrate.py   # or curl :8765
# show: health ok; calc 12*11=132; injection blocked; RAG answer; day173_stats.json PASS
```

## 8. Reproducibility
```bash
source .venv/bin/activate
python code/week-25/day170_data_model.py
python code/week-25/day173_eval.py
python code/week-26/day178_eval_ablate.py
python -c "from service import serve,get_agent; get_agent(); serve('127.0.0.1',8765); import time; time.sleep(3600)"
```
HF ids: `sentence-transformers/all-MiniLM-L6-v2`, `HuggingFaceTB/SmolLM2-135M-Instruct`. Cache: `data/cache/hf`.

## 9. Appendix paths
- Freeze: `code/capstone/out/freeze/`
- Reports: `FINAL_EVAL_REPORT.md`, this file, `MODEL_CARD.md`
- Logs: `code/week-25/out/`, `code/week-26/out/`
- Exam midpoint: `code/week-25/out/day175_exam.md`; final exam Day 180
