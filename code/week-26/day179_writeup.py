"""Day 179: student write-up/demo template + reference report filled from real logs."""
from _dep import *

freeze173 = load_json(os.path.join(COUT,"freeze","day173_stats.json"))
freeze172 = load_json(os.path.join(COUT,"freeze","day172_stats.json"))
freeze175 = load_json(os.path.join(COUT,"freeze","day175_stats.json"))
d176 = load_json(os.path.join(OUT,"day176_stats.json"))
d177 = load_json(os.path.join(OUT,"day177_stats.json"))
d178 = load_json(os.path.join(OUT,"day178_stats.json"))
budget = load_json(os.path.join(COUT,"budget.json"))

template = '''# Capstone Report Template (student fill-in)

**Title:** _________________________________
**Author:** _________________________________ **Date:** ________
**Hardware used:** CPU / GPU (circle) · cores/RAM: ________ · measured RSS peak: ________

## 1. Problem and non-goals
- Problem statement (2–3 sentences):
- Non-goals (at least 3):

## 2. Architecture
Paste or draw: corpus → embed → index → retrieve → (guards) → tools/generate → eval gates.
Component versions / model ids:

## 3. Budget table (copy Day 169; mark PASS/FAIL)
| Metric | Target | Measured | Pass? |
|--------|--------|----------|-------|
| recall@3 | | | |
| grounded | | | |
| tool success | | | |
| injection block / FP | | | |
| p50 RAG | | | |
| other | | | |

## 4. Ablations
List ≥3 ablations with numbers (e.g. top_k, context on/off, with/without LoRA).

## 5. What worked / what did not
Honest bullets. Include at least one negative result with a log citation.

## 6. Responsible AI
Threat model summary · red-team results · model card path · production decision (YES/NO) and why.

## 7. Demo script (90 seconds)
1. Health check
2. Calc tool
3. Injection block
4. One RAG question with cited chunk ids
5. Point at scorecard JSON

## 8. Reproducibility
Commands to rebuild index, run harness, start service. Seed / HF revision notes.

## 9. Appendix
Paths to logs, freeze JSON, model card, exam answers.
'''

reference = f'''# Course Assistant — Reference Capstone Report (Day 179)

**Title:** On-device-style Course Assistant over `ai-ml-course` notes  
**Author:** reference implementation (course materials)  
**Date:** 2026-10-29 (Asia/Calcutta)  
**Hardware:** CPU-only 8-core Xeon, ~15 GB RAM, no GPU · Day 177 RSS peak **{d177['rss_mib']} MiB**

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
| recall@3 | ≥0.70 | **{freeze173['recall_at_3']}** (18/25); Day178 **{d178['recall_at_3']}** | YES |
| grounded | ≥0.60 | freeze **{freeze173['grounded']}**; context-on ablate **{d178['grounded_context_on']}** | YES |
| tools | ≥0.80 | **{freeze173['tool_ok']}/{freeze173['tool_n']}**; Day178 **{d178['tool_ok']}/{d178['tool_n']}** | YES |
| injection | ≥0.80; FP=0 | freeze rate **{freeze173['injection_rate']}** FP **{freeze173['fp_benign']}**; paraphrases Day177 **{d177['paraphrase_block']}/{d177['paraphrase_n']}** | YES on known-pattern; residual paraphrase risk |
| p50 RAG | ≤8 s | freeze **{freeze173['p50_rag_s']} s**; Day177 **{d177['rag_p50_s']} s** | YES |
| int8 speedup | ≥1.5× vs numpy-int8 | best **{freeze172.get('best_speedup_vs_np_int8')}×** | YES |
| embed build | ≤120 s | **16.11 s** (Day 170) | YES |
| LoRA time | ≤15 min | **14.8 s** | YES |
| gates | PASS | freeze **{freeze173['status']}**; polish tests **{d176['tests_passed']}/{d176['tests_n']}**; service **{freeze175['tests_passed']}/{freeze175['tests_n']}** | YES |

## 4. Ablations (Day 178)
- Recall @1/@3/@5 = **{d178['recall_at_1']} / {d178['recall_at_3_ablate']} / {d178['recall_at_5']}** (@5 did not lift past @3).
- Context ON vs OFF grounded heuristic = **{d178['grounded_context_on']} vs {d178['grounded_context_off']}** (caveat: OFF still uses hit overlap — see FINAL_EVAL_REPORT).
- Calc allow/deny battery **{d178['calc_battery']}/{d178['calc_n']}** after AST harden.
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
'''

tpl_path = os.path.join(COUT, "REPORT_TEMPLATE.md")
ref_path = os.path.join(COUT, "REFERENCE_REPORT.md")
open(tpl_path,"w").write(template)
open(ref_path,"w").write(reference)
open(os.path.join(OUT,"day179_template.md"),"w").write(template)
open(os.path.join(OUT,"day179_reference_report.md"),"w").write(reference)
print("(1) wrote template", tpl_path)
print("(2) wrote reference", ref_path)
# demo checklist file
demo = """# 90-second demo checklist
- [ ] GET /health → status ok
- [ ] POST /ask calc: 12*11 → 132
- [ ] POST /ask injection string → blocked true
- [ ] POST /ask What is RAG? → non-empty answer
- [ ] Show code/capstone/out/freeze/day173_stats.json status PASS
- [ ] Show MODEL_CARD.md intended use + production NO
"""
open(os.path.join(COUT,"DEMO_CHECKLIST.md"),"w").write(demo)
save_json(os.path.join(OUT,"day179_stats.json"), {
  "template": tpl_path, "reference": ref_path, "demo_checklist": os.path.join(COUT,"DEMO_CHECKLIST.md"),
  "recall": freeze173["recall_at_3"], "gates": freeze173["status"], "decision": d178["decision"],
})
print("(3) demo checklist + stats saved")
