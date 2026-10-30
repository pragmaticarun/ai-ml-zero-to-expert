"""Day 180: final exam (30 Q+A), retrospective, skills matrix, what-next guidance."""
from _dep import *

EXAM = [
("M1-Q01","What does gradient descent update toward?",
 "The negative gradient of the loss (scaled by the learning rate)."),
("M1-Q02","Bias–variance trade-off in one line?",
 "High bias underfits; high variance overfits; total error balances both."),
("M1-Q03","Train/validation/test roles?",
 "Train fits params; validation selects hyperparameters; test estimates final generalisation once."),
("M1-Q04","Precision vs recall?",
 "Precision = TP/(TP+FP); recall = TP/(TP+FN)."),
("M1-Q05","What is k-fold cross-validation for?",
 "More stable out-of-sample estimates by rotating held-out folds."),
("M2-Q06","Why ensembles (bagging) reduce variance?",
 "Averaging diverse low-bias learners trained on bootstrap samples."),
("M2-Q07","k-means objective?",
 "Minimise within-cluster sum of squared distances to centroids."),
("M2-Q08","First principal component maximises?",
 "Variance of the projected data (unit-norm constraint)."),
("M3-Q09","Backprop computes what?",
 "Gradients of the loss w.r.t. parameters via reverse-mode autodiff."),
("M3-Q10","Why convolutions share weights?",
 "Parameter efficiency and local/translation-equivariant features."),
("M3-Q11","BatchNorm uses which statistics at train time?",
 "Mini-batch mean and variance (plus learned scale/shift)."),
("M4-Q12","Scaled dot-product attention formula?",
 "softmax(QK^T / sqrt(d_k)) V"),
("M4-Q13","What is RAG?",
 "Retrieve relevant passages then condition generation on them."),
("M4-Q14","BM25 combines which signals?",
 "TF, IDF, and document-length normalisation."),
("M4-Q15","LoRA updates what instead of full matrices?",
 "Small low-rank adapter matrices (e.g. on q_proj/v_proj)."),
("M5-Q16","Bellman optimality uses max or expectation over actions?",
 "Max over actions (expectation is for a fixed policy)."),
("M5-Q17","Q-learning vs SARSA?",
 "Q-learning off-policy (max target); SARSA on-policy (uses taken action)."),
("M5-Q18","PPO clipping purpose?",
 "Limit policy ratio steps to avoid collapse from large updates."),
("M5-Q19","ReAct loop?",
 "Interleave reasoning, tool actions, and observations."),
("M5-Q20","One injection defence that does not rely on detection?",
 "Allow-lists / approval gates / sandboxing (any)."),
("M6-Q21","Why KV cache during decode?",
 "Avoid recomputing K/V for past tokens each new token."),
("M6-Q22","SmolLM2 KV fp16 B/tok measured in this course?",
 "23040 (22.5 KiB/token)."),
("M6-Q23","Demographic parity?",
 "Equal positive prediction rates across groups."),
("M6-Q24","Differential privacy bounds what?",
 "Influence of any single training example on the output distribution."),
("M6-Q25","Capstone freeze recall@3?",
 "0.72 (18/25)."),
("M6-Q26","Capstone production decision?",
 "NO for real users; ACCEPT for course/portfolio with model card."),
("M6-Q27","Rust int8 best speedup vs numpy-int8 logged?",
 "50.35× (100k×256 synthetic)."),
("M6-Q28","Day 177 paraphrase injection result?",
 "1/5 blocked (4 evasions) — regex is incomplete."),
("M6-Q29","ORT/int8 lesson from Week 23 in one line?",
 "Measure: unstructured sparsity ≠ speedup; static int8 can trade accuracy for speed."),
("M6-Q30","What fails a capstone release even if the demo looks good?",
 "Recall drop >10% relative, injection block <0.70, or any budget gate FAIL."),
]

exam_md = ["# Final Exam — Day 180 (cumulative, 30 questions)\n",
           "Closed-book first. Covers Months 1–6. Answers under each item.\n"]
for i,(qid,q,a) in enumerate(EXAM,1):
    exam_md.append(f"## {i}. ({qid}) {q}\n\n**Answer:** {a}\n")
exam_path = os.path.join(OUT, "day180_final_exam.md")
open(exam_path,"w").write("\n".join(exam_md))
open(os.path.join(COUT,"FINAL_EXAM.md"),"w").write("\n".join(exam_md))
print(f"(1) exam {len(EXAM)} Q → {exam_path}")

retrospective = '''# Course Retrospective (Day 180)

## What the 180 days were for
Take an Android/multimedia engineer from ML novice to someone who can specify, build, measure, gate, and refuse to ship an on-device-style assistant — with CPU-honest numbers.

## What compounded
- **Measure before claiming.** Loss curves, demo chats, and marketing cards all lied at least once (LoRA loss, grounded heuristics, Adult "drop sex").
- **Defence in depth beats detectors.** Allow-lists, AST calc, approval gates, sandboxes — Week 20/25/26.
- **Small models need retrieval.** SmolLM2-135M without RAG is a liability; with RAG and gates it is a teaching product.
- **Systems skills matter as much as models.** ONNX/ORT, Rust/PyO3, KV budgets, PSI, model cards.

## What was hard / honest negatives
- Distillation lost every seed (Week 23). Capstone Month-5 release BLOCKED (recall 0.53). Paraphrase injections evade regex. Phone/Android paths UNTESTED. No GPU/Triton numbers.

## Habits to keep
Spaced repetition (+1/+3/+7/+21), learning log, change-one-thing labs, freeze eval sets before prompt tuning, write the non-goals first.
'''
open(os.path.join(OUT,"day180_retrospective.md"),"w").write(retrospective)
open(os.path.join(COUT,"RETROSPECTIVE.md"),"w").write(retrospective)

# skills matrix
skills = [
("Python/NumPy/pandas","M1","strong","used daily in labs"),
("Classical ML (sklearn)","M1-M2","strong","Adult, trees, clustering, PCA"),
("PyTorch training","M3","strong","MLP/CNN; CPU limits noted"),
("NLP / Transformers","M4","strong","SmolLM2, tokenizers, attention labs"),
("RAG / IR","M4-M5","strong","BM25, MiniLM, capstone recall 0.72"),
("Fine-tuning / LoRA","M4-M5","moderate","LoRA ran; quality needed RAG"),
("RL foundations","M5","moderate","DP/MC/TD/DQN/PPO toys on CPU"),
("Agents + tool safety","M5-M6","strong","allow-lists, injection suites, AST calc"),
("MLOps / LLMOps","M5-M6","strong","gates, PSI, prompt hashing, traces"),
("ONNX / quantization","M6","moderate","ORT int8/pruning measured on CPU"),
("Rust/PyO3 kernels","M3/M6","moderate","tiny_infer, rust_score, BM25 kernels"),
("Responsible AI","M6","strong","fairness, DP, cards, audits"),
("On-device / mobile","M6","basic","KV budgets yes; Android app UNTESTED"),
("GPU / CUDA / Triton","M6","basic","marked UNTESTED — no GPU on box"),
]
mat = ["# Self-assessment skills matrix (Day 180)\n",
       "Rate yourself again in 6 months. Levels: basic / moderate / strong.\n\n",
       "| Skill | Where learned | Self level | Evidence |\n|---|---|---|---|\n"]
for s,w,lvl,ev in skills:
    mat.append(f"| {s} | {w} | {lvl} | {ev} |\n")
mat_path = os.path.join(OUT,"day180_skills_matrix.md")
open(mat_path,"w").write("".join(mat))
open(os.path.join(COUT,"SKILLS_MATRIX.md"),"w").write("".join(mat))
print(f"(2) skills matrix {len(skills)} rows")

what_next = '''# What next (6–12 months) — Day 180

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
'''
open(os.path.join(OUT,"day180_what_next.md"),"w").write(what_next)
open(os.path.join(COUT,"WHAT_NEXT.md"),"w").write(what_next)
print("(3) retrospective + what-next written")

save_json(os.path.join(OUT,"day180_stats.json"), {
  "exam_questions": len(EXAM),
  "skills_rows": len(skills),
  "exam_path": exam_path,
  "retrospective": os.path.join(OUT,"day180_retrospective.md"),
  "skills_matrix": mat_path,
  "what_next": os.path.join(OUT,"day180_what_next.md"),
})
print("(4) closeout stats saved")
