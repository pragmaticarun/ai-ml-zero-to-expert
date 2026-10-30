# Final Exam — Day 180 (cumulative, 30 questions)

Closed-book first. Covers Months 1–6. Answers under each item.

## 1. (M1-Q01) What does gradient descent update toward?

**Answer:** The negative gradient of the loss (scaled by the learning rate).

## 2. (M1-Q02) Bias–variance trade-off in one line?

**Answer:** High bias underfits; high variance overfits; total error balances both.

## 3. (M1-Q03) Train/validation/test roles?

**Answer:** Train fits params; validation selects hyperparameters; test estimates final generalisation once.

## 4. (M1-Q04) Precision vs recall?

**Answer:** Precision = TP/(TP+FP); recall = TP/(TP+FN).

## 5. (M1-Q05) What is k-fold cross-validation for?

**Answer:** More stable out-of-sample estimates by rotating held-out folds.

## 6. (M2-Q06) Why ensembles (bagging) reduce variance?

**Answer:** Averaging diverse low-bias learners trained on bootstrap samples.

## 7. (M2-Q07) k-means objective?

**Answer:** Minimise within-cluster sum of squared distances to centroids.

## 8. (M2-Q08) First principal component maximises?

**Answer:** Variance of the projected data (unit-norm constraint).

## 9. (M3-Q09) Backprop computes what?

**Answer:** Gradients of the loss w.r.t. parameters via reverse-mode autodiff.

## 10. (M3-Q10) Why convolutions share weights?

**Answer:** Parameter efficiency and local/translation-equivariant features.

## 11. (M3-Q11) BatchNorm uses which statistics at train time?

**Answer:** Mini-batch mean and variance (plus learned scale/shift).

## 12. (M4-Q12) Scaled dot-product attention formula?

**Answer:** softmax(QK^T / sqrt(d_k)) V

## 13. (M4-Q13) What is RAG?

**Answer:** Retrieve relevant passages then condition generation on them.

## 14. (M4-Q14) BM25 combines which signals?

**Answer:** TF, IDF, and document-length normalisation.

## 15. (M4-Q15) LoRA updates what instead of full matrices?

**Answer:** Small low-rank adapter matrices (e.g. on q_proj/v_proj).

## 16. (M5-Q16) Bellman optimality uses max or expectation over actions?

**Answer:** Max over actions (expectation is for a fixed policy).

## 17. (M5-Q17) Q-learning vs SARSA?

**Answer:** Q-learning off-policy (max target); SARSA on-policy (uses taken action).

## 18. (M5-Q18) PPO clipping purpose?

**Answer:** Limit policy ratio steps to avoid collapse from large updates.

## 19. (M5-Q19) ReAct loop?

**Answer:** Interleave reasoning, tool actions, and observations.

## 20. (M5-Q20) One injection defence that does not rely on detection?

**Answer:** Allow-lists / approval gates / sandboxing (any).

## 21. (M6-Q21) Why KV cache during decode?

**Answer:** Avoid recomputing K/V for past tokens each new token.

## 22. (M6-Q22) SmolLM2 KV fp16 B/tok measured in this course?

**Answer:** 23040 (22.5 KiB/token).

## 23. (M6-Q23) Demographic parity?

**Answer:** Equal positive prediction rates across groups.

## 24. (M6-Q24) Differential privacy bounds what?

**Answer:** Influence of any single training example on the output distribution.

## 25. (M6-Q25) Capstone freeze recall@3?

**Answer:** 0.72 (18/25).

## 26. (M6-Q26) Capstone production decision?

**Answer:** NO for real users; ACCEPT for course/portfolio with model card.

## 27. (M6-Q27) Rust int8 best speedup vs numpy-int8 logged?

**Answer:** 50.35× (100k×256 synthetic).

## 28. (M6-Q28) Day 177 paraphrase injection result?

**Answer:** 1/5 blocked (4 evasions) — regex is incomplete.

## 29. (M6-Q29) ORT/int8 lesson from Week 23 in one line?

**Answer:** Measure: unstructured sparsity ≠ speedup; static int8 can trade accuracy for speed.

## 30. (M6-Q30) What fails a capstone release even if the demo looks good?

**Answer:** Recall drop >10% relative, injection block <0.70, or any budget gate FAIL.
