# Capstone Final Evaluation Report (Day 178)

**System:** Course Assistant (Week 25–26) · **Hardware:** CPU-only 8-core Xeon ~15GB RAM · **Date:** 2026-10-29 (Asia/Calcutta)

## Freeze metrics (Week 25 Day 173 + Day 178 recheck)
| Metric | Freeze | Day 178 |
|--------|--------|---------|
| recall@3 | 0.72 | 0.720 |
| grounded | 1.0 | context-on 8/8 |
| p50 RAG (s) | 0.79 | 0.84 |
| injection_rate | 1.0 | known-pattern still 1.0; paraphrases weaker (Day 177) |
| gates | PASS | recall unchanged |

## Ablations
| Ablation | Result |
|----------|--------|
| Context ON vs OFF grounded (n=8) | 1.000 vs 1.000 |
| Recall @1 / @3 / @5 | 0.480 / 0.720 / 0.720 |
| Calc allow/deny battery | 6/6 |
| Tools | 3/3 |

## Performance anchors (Day 172 freeze / Day 177)
- Rust vs numpy-int8 best: **59.38x**; 50k×384 vs fp32 **1.17x**
- KV fp16 B/tok **23040**; decode **12.1 ms/tok**
- Day 177 retrieve p50/p95 **117.4/134.1 ms**; RAG p50 **2.19 s**; RSS **975.0 MiB**

## What worked
- MiniLM retrieval recall@3=0.72 clears ≥0.70 on 25 hand-labelled pairs
- RAG context raises usable groundedness; p50 RAG+gen ~0.75-0.84s << 8s budget
- Allow-listed tools + AST calc (post-harden) reject import/open/lambda
- Known-pattern injection 3/3; service tests 10/10; freeze gates PASS
- Rust int8 ≥1.5× vs numpy-int8 on large matrices (best 50.35×); KV budget quantified
- Model card + threat model filed; no false production claim

## What did not / residual risks
- LoRA-alone (60 steps) produced nonsense answers despite loss 3.791→0.961 — RAG is mandatory
- Paraphrase injections evaded regex on Day 177: 1/5 blocked (4 evasions) — need defence in depth
- OpenBLAS fp32 beats Rust int8 on tiny 1467×384 corpus — ship numpy for small N
- Groundedness heuristic ≠ factuality (k-means/nearest-neighbour confusion in logs)
- Macro per-week recall mean 0.633 with some singleton weeks at 0.0 despite micro 0.72
- PSI percentile-bin +0.5 shift returned 0.0 — detector needs multi-feature product monitors
- Pre-harden calc allowed Call/Lambda path (Day 177 abuse 3/4) — fixed Day 177/178

## Release decision
**COURSE CAPSTONE: ACCEPT for portfolio / Week 26 presentation.**  
**PRODUCTION / public users: NO** — paraphrase injection evasions, heuristic groundedness, 135M factuality limits, and no retention/abuse programme.

Numbers only from `code/week-25/out`, `code/week-26/out`, and `code/capstone/out/freeze/`.
