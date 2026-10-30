# Course Assistant (Week 25 final-capstone start)

CPU-feasible on-device-style study helper over `weeks/*.md`.

## Stack
- **Corpus:** chunked week markdown → `out/chunks.json` + `out/embeddings.npy` (MiniLM-L6-v2, 1467×384)
- **Generator:** `HuggingFaceTB/SmolLM2-135M-Instruct` (+ optional LoRA r=8 in `adapters/course_lora`)
- **RAG / agent:** `rag.py`, `agent.py` (tools: `calc`, `course_search`, `get_day_title`; injection `guard_user`)
- **Int8 scoring:** `rust_score/` PyO3 + rayon (`build_int8_index` / `score_int8`)
- **Service:** `service.py` stdlib HTTP on `:8765` (`/health`, `/metrics`, `/ask`)
- **RAI:** `out/MODEL_CARD.md`, `out/model_card.json`
- **QA freeze set:** `corpus_qa.py` (25 pairs)

## Week scripts
Run one at a time from `code/week-25/` (see logs in `code/week-25/out/`):
`day169_proposal.py` → `day170_data_model.py` → `day171_rag_agent.py` → `day172_perf.py` → `day173_eval.py` → `day174_safety.py` → `day175_integrate.py`.

## Budget (Day 169)
See `out/budget.json`. Release FAIL if recall@3 drops >10% relative or injection block <0.70.
