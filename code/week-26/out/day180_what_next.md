# What next (6–12 months) — Day 180

## Keep skills fresh
- Continue Anki from `flashcards/month-01.csv` … `month-06.csv` using `review-schedule.csv` (+1/+3/+7/+21).
- Re-run `bash run_all.sh` after dependency bumps; treat breakages as labs.
- Once a month: pick one week file, change one hyperparameter, predict, measure, log.

## Reading list (verified official / primary sources only — already in docs/resources.md or standard project homes)
- PyTorch docs: https://docs.pytorch.org/
- Hugging Face Transformers docs: https://huggingface.co/docs/transformers
- ONNX Runtime performance: https://onnxruntime.ai/docs/performance/
- Gymnasium docs: https://gymnasium.farama.org/
- scikit-learn user guide: https://scikit-learn.org/stable/user_guide.html
- NIST AI RMF 1.0: https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf
- EU AI Act overview (not legal advice): https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- ExecuTorch: https://docs.pytorch.org/executorch/stable/getting-started.html
- Opacus: https://opacus.ai/
- Anki: https://ankiweb.net/
- Optional courses (inspiration; see docs/teaching-style.md): https://www.deeplearning.ai/short-courses/ · https://www.coursera.org/specializations/deep-learning

## Research directions (pick one and go deep)
1. Better groundedness / faithfulness metrics for small-LM RAG on CPU.
2. Paraphrase-robust tool-agent guards without LLM-as-judge cost.
3. Structured pruning + int8 that actually wins on your target chip (measure, do not assume).
4. Preference-tuning (DPO) on a domain corpus with frozen behavioural evals.
5. On-device KV/int4 memory planning for a real Android build (complete the UNTESTED path).

## Open-source contribution ideas
- Fix docs/tests in libraries you already use (ORT, ExecuTorch, sentence-transformers, Gymnasium).
- Publish your capstone ablation tables (with hardware labels) as a blog + reproducible script.
- Add a CI job that runs the Day 173 harness on CPU and fails on gate regressions.
- Upstream a minimal PyO3 int8 scoring example with clear BLAS-vs-int8 caveats.

## Career / portfolio
Lead with the **honest** report: freeze recall 0.72, paraphrase gaps, production NO. Show gates, model card, and Rust kernel. Prefer depth in one deployment stack (Android NDK + ORT/ExecuTorch **or** server ORT) over shallow tool tourism.
