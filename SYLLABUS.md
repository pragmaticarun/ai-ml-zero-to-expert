# SYLLABUS: Novice → Expert AI/ML in 180 Days

**Budget:** 180 days × 6 h/day = 1,080 h. **Calendar:** 25 full weeks (Days 1-175) plus **week 26, which has only 5 days (Days 176-180)** and ends with the final capstone presentation on Day 180.
**Week file mapping:** `weeks/week-NN.md` covers Days 7(NN−1)+1 … 7NN. So week-05 = Days 29-35, which straddles Months 1 and 2.

## Month plan
| Month | Days | Theme | Monthly capstone |
|---|---|---|---|
| 1 | 1-30 | Python + maths foundations, first models | Linear + logistic regression from scratch in NumPy, benchmarked against scikit-learn, plus an optional Rust/PyO3 GD kernel (built and verified) |
| 2 | 31-60 | Classical ML | End-to-end tabular ML system (CV, boosting, pipelines, error-analysis report) |
| 3 | 61-90 | Neural nets & deep learning | DL project (CNN or sequence model) with ablations and experiment log |
| 4 | 91-120 | Transformers & LLMs | Fine-tuned (LoRA), quantised, served LLM/RAG app |
| 5 | 121-150 | RL, agents, MLOps | Agent or RL system deployed with tracking, CI, monitoring |
| 6 | 151-180 | Performance, on-device, responsible AI | **Final expert capstone** (below) |

## Final capstone idea (Days 170-180)
**"On-device multimedia assistant."** Build a small model pipeline that runs on an Android phone or edge device. It has two parts: (a) an audio or vision model (for example keyword spotting, or scene classification on camera frames) and (b) a small quantised language model running through llama.cpp/GGUF or ONNX Runtime that turns the detections into a text summary or answers questions about them. Deliverables:
- design doc
- baseline vs final metrics
- a latency/memory benchmark table measured on the device
- a custom Rust or C++ kernel for one hot spot
- a responsible-AI review (model card)
- Day 180 closeout: 30-question cumulative exam, retrospective, skills matrix, and 6–12 month what-next (presentation/demo artefacts shipped Day 179)

An alternative of equal scope is allowed if the student proposes it in the Day 170 design doc.

## Recurring structure
- **Every day (6 h):** 1h15 concept lesson → 45m worked example → 2h coding lab → 1h practice problems → 1h review (spaced repetition + 5-question self-check). See README for the template.
- **Every 7th day:** weekly project + weekly quiz (10+ questions, answers at the end of the week file).
- **Every month:** capstone + checkpoint exam (Days 29-30, 58-60, 88-90, 119-120, 147-149).

## Spaced repetition
- Review each day's material at **+1, +3, +7 and +21 days**. `review-schedule.csv` lists, for every study day, which earlier days to review. For example, Day 8 reviews Days 1, 5 and 7, and Day 22 reviews Days 1, 15, 19 and 21. Each day's "Review" block in the week files already names those days.
- **Flashcards:** `flashcards/month-01.csv` is Anki-importable (fields: Front, Back, Tags; tags look like `m1 d12`). Later phases add `month-02.csv` … `month-06.csv`.

## Day-by-day table
| Day | Week | Month | Topic | Status |
|---|---|---|---|---|
| 1 | 1 | 1 | Setup, Python refresher, what is ML; a hand-tuned line | written |
| 2 | 1 | 1 | NumPy arrays, dtypes, shapes, broadcasting | written |
| 3 | 1 | 1 | Vectorisation and performance mindset (Python vs NumPy vs C++/Rust) | written |
| 4 | 1 | 1 | pandas for tabular data; missing values; one-hot; leakage preview | written |
| 5 | 1 | 1 | matplotlib and exploratory data analysis; residual plots | written |
| 6 | 1 | 1 | Train/validation/test splits, seeds, reproducibility, leakage | written |
| 7 | 1 | 1 | PROJECT W1: EDA report + baselines; Quiz 1 | written |
| 8 | 2 | 1 | Vectors, norms, dot product, cosine similarity | written |
| 9 | 2 | 1 | Matrices and matrix multiplication; naive Rust matmul vs BLAS | written |
| 10 | 2 | 1 | Linear systems, conditioning, least squares | written |
| 11 | 2 | 1 | Eigenvalues/eigenvectors, power iteration, covariance | written |
| 12 | 2 | 1 | SVD and PCA from scratch | written |
| 13 | 2 | 1 | Projections, orthogonality, Gram-Schmidt/QR | written |
| 14 | 2 | 1 | PROJECT W2: SVD image compression; Quiz 2 | written |
| 15 | 3 | 1 | Derivatives and numerical differentiation | written |
| 16 | 3 | 1 | Partial derivatives, gradients, gradient checking | written |
| 17 | 3 | 1 | Chain rule, computational graphs, tiny autograd | written |
| 18 | 3 | 1 | Gradient descent and learning rates | written |
| 19 | 3 | 1 | Probability, conditional probability, Bayes' rule | written |
| 20 | 3 | 1 | Distributions, expectation, variance, LLN and CLT | written |
| 21 | 3 | 1 | PROJECT W3: Maximum likelihood estimation; Quiz 3 | written |
| 22 | 4 | 1 | Linear regression (1 feature) and the cost function | written |
| 23 | 4 | 1 | Gradient descent for linear regression | written |
| 24 | 4 | 1 | Multiple linear regression, vectorised | written |
| 25 | 4 | 1 | Feature scaling and debugging gradient descent | written |
| 26 | 4 | 1 | Logistic regression and log-loss | written |
| 27 | 4 | 1 | Evaluation: confusion matrix, precision/recall, ROC-AUC, bias/variance | written |
| 28 | 4 | 1 | PROJECT W4: housing regression + error analysis; Quiz 4 | written |
| 29 | 5 | 1 | MONTH-1 CAPSTONE: scratch models vs scikit-learn (+ Rust/PyO3 kernel) | written |
| 30 | 5 | 1 | Month-1 checkpoint exam and review | written |
| 31 | 5 | 2 | Regularisation deep dive: ridge, lasso, elastic net | written |
| 32 | 5 | 2 | Polynomial features and the bias-variance trade-off in practice | written |
| 33 | 5 | 2 | Cross-validation and hyperparameter search | written |
| 34 | 5 | 2 | Softmax and multiclass classification | written |
| 35 | 5 | 2 | PROJECT W5: California housing model with CV; Quiz 5 | written |
| 36 | 6 | 2 | k-nearest neighbours; distance metrics; curse of dimensionality | written |
| 37 | 6 | 2 | Naive Bayes; generative vs discriminative models | written |
| 38 | 6 | 2 | Support vector machines and kernels | written |
| 39 | 6 | 2 | Decision trees: entropy, Gini, splitting | written |
| 40 | 6 | 2 | Random forests and bagging | written |
| 41 | 6 | 2 | Gradient boosting from scratch, then HistGradientBoosting | written |
| 42 | 6 | 2 | PROJECT W6: tabular pipeline with boosting; Quiz 6 | written |
| 43 | 7 | 2 | Feature importance, permutation importance, SHAP intuition | written |
| 44 | 7 | 2 | Feature engineering: encodings, dates, text counts | written |
| 45 | 7 | 2 | Imbalanced data: class weights, resampling, threshold tuning | written |
| 46 | 7 | 2 | Pipelines, ColumnTransformer, leakage-proof preprocessing | written |
| 47 | 7 | 2 | k-means clustering and choosing k | written |
| 48 | 7 | 2 | Hierarchical clustering, DBSCAN, Gaussian mixtures (EM intuition) | written |
| 49 | 7 | 2 | PROJECT W7: customer segmentation; Quiz 7 | written |
| 50 | 8 | 2 | PCA in practice; t-SNE/UMAP for visualisation | written |
| 51 | 8 | 2 | Anomaly detection | written |
| 52 | 8 | 2 | Recommender systems: collaborative filtering, matrix factorisation | written |
| 53 | 8 | 2 | Time-series basics: lag features, leakage-safe validation | written |
| 54 | 8 | 2 | Interpretability and probability calibration | written |
| 55 | 8 | 2 | Systematic ML strategy: learning curves, error analysis, what to try next | written |
| 56 | 8 | 2 | PROJECT W8: Rust/PyO3 k-means kernel vs NumPy; Quiz 8 | written |
| 57 | 9 | 2 | Reproducible experiments: seeds, configs, model persistence | written |
| 58 | 9 | 2 | MONTH-2 CAPSTONE (part 1): build and compare, CV only | written |
| 59 | 9 | 2 | Month-2 capstone (part 2): test evaluation, slices, report | written |
| 60 | 9 | 2 | MONTH-2 CAPSTONE delivery + checkpoint exam and review | written |
| 61 | 9 | 3 | The perceptron and the logistic neuron | written |
| 62 | 9 | 3 | Multilayer networks: forward propagation, activations, initialisation | written |
| 63 | 9 | 3 | PROJECT W9: NumPy MLP with hand-derived backprop on digits; Quiz 9 | written |
| 64 | 10 | 3 | PyTorch tensors, autograd, nn.Module, training loop | written |
| 65 | 10 | 3 | Datasets, DataLoaders, batching, devices | written |
| 66 | 10 | 3 | Initialisation; vanishing and exploding gradients | written |
| 67 | 10 | 3 | Optimisers: SGD, momentum, RMSProp, Adam; LR schedules | written |
| 68 | 10 | 3 | Regularisation: L2, dropout, early stopping, augmentation | written |
| 69 | 10 | 3 | Batch norm and layer norm | written |
| 70 | 10 | 3 | PROJECT W10: MNIST MLP in PyTorch with experiment log; Quiz 10 | written |
| 71 | 11 | 3 | Convolutions from scratch (and as im2col matmul) | written |
| 72 | 11 | 3 | CNN architectures: LeNet to ResNet ideas | written |
| 73 | 11 | 3 | Training a CNN on CIFAR-10 (CPU-sized) | written |
| 74 | 11 | 3 | Transfer learning and fine-tuning a pretrained CNN | written |
| 75 | 11 | 3 | Debugging deep nets: overfit one batch, loss curves, sanity checks | written |
| 76 | 11 | 3 | Audio/image pipelines for multimedia devs (spectrograms, augmentation) | written |
| 77 | 11 | 3 | PROJECT W11: image or audio classifier; Quiz 11 | written |
| 78 | 12 | 3 | RNNs and backprop through time | written |
| 79 | 12 | 3 | LSTMs and GRUs | written |
| 80 | 12 | 3 | Embeddings and word2vec intuition | written |
| 81 | 12 | 3 | Seq2seq and the motivation for attention | written |
| 82 | 12 | 3 | Mixed precision; profiling a PyTorch training run | written |
| 83 | 12 | 3 | JAX quick tour (optional): jit, grad, vmap | written |
| 84 | 12 | 3 | PROJECT W12: character-level language model; Quiz 12 | written |
| 85 | 13 | 3 | Hyperparameter tuning and experiment tracking for deep learning | written |
| 86 | 13 | 3 | Month-3 capstone build: small ResNet on CIFAR-10-small with baselines | written |
| 87 | 13 | 3 | Month-3 capstone: ablations | written |
| 88 | 13 | 3 | Month-3 capstone: error analysis | written |
| 89 | 13 | 3 | Cumulative review Months 1-3 (10 runnable drills) | written |
| 90 | 13 | 3 | MONTH-3 CAPSTONE delivery (report) + checkpoint exam | written |
| 91 | 13 | 4 | PROJECT W13: attention from scratch (Q, K, V); Quiz 13 | written |
| 92 | 14 | 4 | Self-attention, multi-head attention, masking | Phase 4 |
| 93 | 14 | 4 | Positional encodings (sinusoidal, RoPE) | Phase 4 |
| 94 | 14 | 4 | The transformer block: residuals, layer norm, MLP | Phase 4 |
| 95 | 14 | 4 | Tiny GPT from scratch | Phase 4 |
| 96 | 14 | 4 | Tokenisation: BPE from scratch | Phase 4 |
| 97 | 14 | 4 | Pretraining objectives and scaling laws | Phase 4 |
| 98 | 14 | 4 | PROJECT W14: train tiny GPT on text; Quiz 14 | Phase 4 |
| 99 | 15 | 4 | Hugging Face transformers and datasets | Phase 4 |
| 100 | 15 | 4 | Fine-tuning a small pretrained model | Phase 4 |
| 101 | 15 | 4 | LoRA from scratch, then PEFT | Phase 4 |
| 102 | 15 | 4 | PEFT configurations and QLoRA concepts | Phase 4 |
| 103 | 15 | 4 | Instruction tuning and chat templates | Phase 4 |
| 104 | 15 | 4 | Evaluating LLMs and the fine-tuned model | Phase 4 |
| 105 | 15 | 4 | PROJECT W15: full fine-tune vs LoRA; Quiz 15 | Phase 4 |
| 106 | 16 | 4 | RLHF overview: preferences, Bradley-Terry, KL-regularised objective | Phase 4 |
| 107 | 16 | 4 | Reward modelling on a small LM, and shortcuts | Phase 4 |
| 108 | 16 | 4 | DPO from scratch | Phase 4 |
| 109 | 16 | 4 | DPO: beta, learning rate and forgetting | Phase 4 |
| 110 | 16 | 4 | LLM evaluation pitfalls | Phase 4 |
| 111 | 16 | 4 | Best-of-n, verifiers and reward exploitation | Phase 4 |
| 112 | 16 | 4 | PROJECT W16: SFT vs DPO vs DPO+NLL; Quiz 16 | Phase 4 |

*Note (2026-10-29): the former Week 16 topics (embeddings and vector search, RAG, prompting/structured output/function calling, quantisation, KV cache/batching/speculative decoding, RAG project) move to later Month 4/5 weeks; the rows below are not yet renumbered.*

| 113 | 17 | 4 | Embeddings, chunking and vector search | Phase 4 |
| 114 | 17 | 4 | Hybrid retrieval, reranking and retrieval evaluation | Phase 4 |
| 115 | 17 | 4 | RAG generation and grounding evaluation | Phase 4 |
| 116 | 17 | 4 | Quantisation from scratch (RTN int8/int4) | Phase 4 |
| 117 | 17 | 4 | GPTQ/AWQ concepts, plus a Rust/PyO3 int8 kernel | Phase 4 |
| 118 | 17 | 4 | Inference serving: KV cache, batching, speculative decoding, vLLM/llama.cpp overview | Phase 4 |
| 119 | 17 | 4 | PROJECT W17: RAG over the notes with a quantised generator; Quiz 17 | Phase 4 |

> Note (Week 17 rewrite): the old Week 17 topics (long context and efficient attention, multimodal overview, Month-4 capstone) move to a later week; serving and llama.cpp/GGUF are now covered in Day 118 (llama.cpp from its README only).

| 120 | 18 | 4 | MONTH-4 CAPSTONE: end-to-end RAG + int8 + own KV-cache decoder; Month-4 checkpoint exam | Phase 4 |
| 121 | 18 | 5 | MDPs, returns and Bellman equations | Phase 5 |
| 122 | 18 | 5 | Dynamic programming: policy evaluation, value and policy iteration | Phase 5 |
| 123 | 18 | 5 | Monte Carlo and TD; Q-learning vs SARSA | Phase 5 |
| 124 | 18 | 5 | Function approximation (linear, Fourier features) | Phase 5 |
| 125 | 18 | 5 | DQN on CartPole (CPU) | Phase 5 |
| 126 | 18 | 5 | PROJECT W18: tabular agents on FrozenLake 8x8 vs DP optimum; Quiz 18 (REINFORCE moves to Week 19) | Phase 5 |
| 127 | 19 | 5 | Policy gradients: REINFORCE and baselines | Phase 5 |
| 128 | 19 | 5 | Actor-critic (A2C), n-step advantages | Phase 5 |
| 129 | 19 | 5 | PPO from scratch (clipped surrogate, GAE) | Phase 5 |
| 130 | 19 | 5 | Exploration: bandits, UCB/Thompson, LinUCB recommendation | Phase 5 |
| 131 | 19 | 5 | Reward shaping pitfalls (potential-based shaping) | Phase 5 |
| 132 | 19 | 5 | RL for LLMs revisited: PPO+KL, GRPO, DPO (toy with exact optimum) | Phase 5 |
| 133 | 19 | 5 | PROJECT W19: REINFORCE vs A2C vs PPO on CartPole, equal step budget; Quiz 19 (agents and MLOps move to Week 20+) | Phase 5 |
| 134 | 20 | 5 | Agent loop and tool calling with JSON schemas + validation | Phase 5 |
| 135 | 20 | 5 | ReAct loop; SmolLM2-135M vs scripted/oracle policies | Phase 5 |
| 136 | 20 | 5 | Planning, retries and agent memory (scripted policies) | Phase 5 |
| 137 | 20 | 5 | Retrieval as a tool (Month-4 RAG) and a sandboxed code-execution tool | Phase 5 |
| 138 | 20 | 5 | Agent evaluation: success CIs, pass@k vs pass^k, failure taxonomy | Phase 5 |
| 139 | 20 | 5 | Prompt injection, safety defences and structured outputs | Phase 5 |
| 140 | 20 | 5 | PROJECT W20: evaluated, defended tool-using agent; Quiz 20 (MLOps moves to Week 21+) | Phase 5 |
| 141 | 21 | 5 | ML lifecycle and scoping: baselines, cost-weighted thresholds, learning curves, slices | Phase 5 |
| 142 | 21 | 5 | Data validation and hash-based dataset versioning (DVC-style) | Phase 5 |
| 143 | 21 | 5 | Experiment tracking (MLflow) and reproducibility | Phase 5 |
| 144 | 21 | 5 | Packaging and serving: ONNX/onnxruntime vs PyTorch, FastAPI, load testing | Phase 5 |
| 145 | 21 | 5 | Monitoring: PSI/KS drift detection; label-error analysis loop | Phase 5 |
| 146 | 21 | 5 | A/B tests, canary rollouts, cost/latency budgets; Docker/CI (untested) | Phase 5 |
| 147 | 21 | 5 | Week-21 project: production-style service with tests, monitoring, model card; Quiz 21 (LLMOps and Month-5 capstone move later) | Phase 5 |
| 148 | 22 | 5 | LLMOps I: prompt versioning, eval harnesses and regression gates (cumulative review moved into the review blocks) | Phase 5 |
| 149 | 22 | 5 | LLMOps II: tracing, latency and cost tracking, guardrails | Phase 5 |
| 150 | 22 | 5 | MONTH-5 CAPSTONE: monitored, tested, tool-using RAG + agent service with release gates and drift monitor; Month-5 checkpoint exam | Phase 5 |
| 151 | 22 | 6 | Profiling Python ML code: timing, cProfile, tracemalloc, torch.profiler | Phase 6 |
| 152 | 22 | 6 | Memory bandwidth, roofline model, memory layout and fusion (NumPy/PyTorch, BLAS, bf16/AMX) | Phase 6 |
| 153 | 22 | 6 | Rust extensions with PyO3: vectorisation and threads (rayon) | Phase 6 |
| 154 | 22 | 6 | PROJECT W22: Rust/PyO3 BM25 kernel speeds up the capstone retrieval (no g++ for C++/pybind11); Triton/CUDA kernels shown UNTESTED; Quiz 22 | Phase 6 |
| 155 | 23 | 6 | ONNX export and ONNX Runtime tuning (graph opts, threads, profiling) | Phase 6 |
| 156 | 23 | 6 | int8 quantization: dynamic, static QDQ, per-channel (ORT, PyTorch) | Phase 6 |
| 157 | 23 | 6 | Pruning and distillation basics | Phase 6 |
| 158 | 23 | 6 | Memory and KV-cache budgeting on devices | Phase 6 |
| 159 | 23 | 6 | Shipping to Android: ORT Mobile, ExecuTorch, LiteRT, llama.cpp (app side UNTESTED) | Phase 6 |
| 160 | 23 | 6 | Rust int8 inference engine (PyO3) vs PyTorch and ORT | Phase 6 |
| 161 | 23 | 6 | PROJECT W23: on-device-style pipeline under a latency/memory/accuracy/size budget; Quiz 23 | Phase 6 |
| 162 | 24 | 6 | Fairness metrics and trade-offs (Adult: DP, equalised odds, calibration) | Phase 6 |
| 163 | 24 | 6 | Bias mitigation: reweighing and group-specific thresholds | Phase 6 |
| 164 | 24 | 6 | Interpretability: permutation importance, Shapley, LIME-style, saliency, probing | Phase 6 |
| 165 | 24 | 6 | Privacy: differential privacy, DP-SGD, membership inference | Phase 6 |
| 166 | 24 | 6 | Robustness, adversarial examples (FGSM/PGD), distribution shift | Phase 6 |
| 167 | 24 | 6 | LLM safety, model cards/datasheets, NIST AI RMF & EU AI Act overview, energy accounting | Phase 6 |
| 168 | 24 | 6 | PROJECT W24: responsible-AI audit report of the Adult model; Quiz 24 | Phase 6 |
| 169 | 25 | 6 | FINAL CAPSTONE: proposal, success metrics and budget table | Phase 6 |
| 170 | 25 | 6 | FINAL CAPSTONE: data, corpus index and LoRA baseline | Phase 6 |
| 171 | 25 | 6 | FINAL CAPSTONE: RAG retrieval and tool-using agent | Phase 6 |
| 172 | 25 | 6 | FINAL CAPSTONE: int8 scoring, Rust/PyO3 kernel and KV budget | Phase 6 |
| 173 | 25 | 6 | FINAL CAPSTONE: evaluation harness, gates and drift | Phase 6 |
| 174 | 25 | 6 | FINAL CAPSTONE: responsible-AI audit and model card | Phase 6 |
| 175 | 25 | 6 | PROJECT W25: integrate service + Month-6 midpoint exam (20Q); Quiz 25 | Phase 6 |
| 176 | 26 | 6 | FINAL CAPSTONE: polish + freeze Week-25 scorecards (week 26 starts; 5 days) | Phase 6 |
| 177 | 26 | 6 | FINAL CAPSTONE: performance/robustness pass (latency percentiles, paraphrase injection, calc AST harden) | Phase 6 |
| 178 | 26 | 6 | FINAL CAPSTONE: final eval + ablations + honest report (ACCEPT_COURSE_NO_PROD) | Phase 6 |
| 179 | 26 | 6 | FINAL CAPSTONE: write-up/demo — student template + reference report from logs | Phase 6 |
| 180 | 26 | 6 | FINAL CAPSTONE closeout: 30Q final exam, retrospective, skills matrix, what-next | Phase 6 |
