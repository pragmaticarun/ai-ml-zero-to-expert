# Capstone Report Template (student fill-in)

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
