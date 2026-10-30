# Model Card — Course Assistant (Week 25 capstone)

## Identity
- **Name:** Course Assistant (capstone)
- **Version:** 0.1.0-week25
- **Date:** 2026-10-29 (Asia/Calcutta build)

## Intended use
CPU-only study helper over ai-ml-course weeks/*.md notes; not a general chatbot or medical/legal advisor.

**Out of scope:** personal advice, code execution beyond calc AST, web browsing, PII processing

## Components
- Retriever: `sentence-transformers/all-MiniLM-L6-v2`
- Generator: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Adapter: `LoRA r=8 on q_proj/v_proj (adapters/course_lora)`
- Tools: calc, course_search, get_day_title
- Scoring: `rust_score int8 (PyO3+rayon)`

## Evaluation (from Day 173 logs)
- recall@3: **0.72** (18/25)
- grounded: **1.0**
- p50 RAG+gen: **0.79 s**
- tools: **3/3**
- injection block: **1.0** (FP=0)
- gates: **PASS**

## Performance (from Day 172 logs)
- corpus rust vs numpy-int8: **5.78x**
- 50k×384 rust vs numpy-int8: **40.85x** (vs fp32 **1.17x**)
- KV fp16 B/tok: **23040**; decode **12.1 ms/tok**
- RSS after SmolLM2: **692.0 MiB**

## Safety & threats
Red-team mitigated **6/6**; benign FP **0/3**.

| Threat | Mitigation |
|--------|------------|
| prompt_injection | INJECTION_PAT + guard_user; block before tools |
| tool_abuse | AST allow-list; no Names/Calls/attrs |
| data_exfiltration | No env/file tools in allow-list |
| hallucination | RAG context required; groundedness gate ≥0.60 |
| privacy | Corpus = public course notes only; no PII by design |
| fairness | Per-week recall reported; no demographic attrs in corpus |
| supply_chain | Pinned HF ids; local HF_HOME cache; checksum optional |

## Fairness proxy
Per-week retrieval recall min=0.00 max=1.00 mean=0.63.

## Limitations
- SmolLM2-135M hallucinates without RAG; LoRA alone insufficient on CPU budget.
- Injection guard is regex-based; paraphrases may evade — defence in depth still needed.
- Fairness measured only as topic/week recall skew, not demographic parity (no demographics in corpus).
- OpenBLAS fp32 can beat int8 Rust on tiny corpora (~1.5k rows); int8 wins vs naive numpy-int8 and at 50k+.

## Ethical considerations
This assistant must cite course chunks and refuse injection/jailbreak attempts. It must not be presented as authoritative outside the course corpus. Log prompts in production only with retention limits and no secrets.
