# Month 6 Midpoint Checkpoint Exam (Day 175)

20 questions covering Weeks 1–24 themes + capstone. Answers below each item.

## Q01. What update rule does gradient descent follow?

**Answer:** Parameters move opposite the loss gradient (negative gradient direction), scaled by a learning rate.

## Q02. Name two metrics derivable from a confusion matrix.

**Answer:** Precision, recall, F1, specificity, false-positive rate (any two).

## Q03. What is overfitting?

**Answer:** Fitting training data well while generalising poorly to unseen data.

## Q04. Why use k-fold cross-validation?

**Answer:** To estimate out-of-sample performance more reliably by rotating held-out folds.

## Q05. How does a random forest reduce variance vs one tree?

**Answer:** Averages many trees trained on bootstrap samples with feature randomness.

## Q06. State the two alternating steps of k-means.

**Answer:** Assign points to nearest centroid; recompute centroids as cluster means.

## Q07. What does the first principal component maximise?

**Answer:** Variance of the projected data (subject to unit-norm).

## Q08. What does backpropagation compute?

**Answer:** Gradients of the loss w.r.t. parameters via reverse-mode autodiff.

## Q09. Why share weights in a convolution?

**Answer:** Parameter efficiency and translation equivariance / local feature detectors.

## Q10. Write the attention output in terms of Q, K, V (softmax form).

**Answer:** softmax(QK^T / sqrt(d_k)) V

## Q11. What problem do residual connections help with in deep nets?

**Answer:** Vanishing gradients / easier optimisation of identity mappings.

## Q12. Define RAG in one sentence.

**Answer:** Retrieve relevant passages then condition generation on them.

## Q13. What does BM25 combine?

**Answer:** Term frequency, inverse document frequency, and document-length normalisation.

## Q14. What does LoRA fine-tune instead of full weights?

**Answer:** Small low-rank adapter matrices (e.g. on q_proj/v_proj).

## Q15. What is a ReAct agent loop?

**Answer:** Interleave reasoning traces with tool actions and observations.

## Q16. Give one prompt-injection defence used in this capstone.

**Answer:** Regex/guard_user block before tools; allow-listed tools; AST-safe calc.

## Q17. Why keep a KV cache during decoding?

**Answer:** Avoid recomputing key/value tensors for past tokens each step.

## Q18. What does ONNX Runtime provide?

**Answer:** Cross-platform ONNX graph execution with fusions/threading optimisations.

## Q19. State demographic parity.

**Answer:** Equal positive prediction rates across groups.

## Q20. What does differential privacy bound?

**Answer:** How much any single training example can change the output distribution.

