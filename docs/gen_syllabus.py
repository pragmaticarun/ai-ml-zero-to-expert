"""Generates the day table in SYLLABUS.md and the spaced-repetition schedule. Edit TOPICS, re-run."""
import csv, os
T = """Setup, Python refresher, what is ML; a hand-tuned line
NumPy arrays, dtypes, shapes, broadcasting
Vectorisation and performance mindset (Python vs NumPy vs C++/Rust)
pandas for tabular data; missing values; one-hot; leakage preview
matplotlib and exploratory data analysis; residual plots
Train/validation/test splits, seeds, reproducibility, leakage
PROJECT W1: EDA report + baselines; Quiz 1
Vectors, norms, dot product, cosine similarity
Matrices and matrix multiplication; naive Rust matmul vs BLAS
Linear systems, conditioning, least squares
Eigenvalues/eigenvectors, power iteration, covariance
SVD and PCA from scratch
Projections, orthogonality, Gram-Schmidt/QR
PROJECT W2: SVD image compression; Quiz 2
Derivatives and numerical differentiation
Partial derivatives, gradients, gradient checking
Chain rule, computational graphs, tiny autograd
Gradient descent and learning rates
Probability, conditional probability, Bayes' rule
Distributions, expectation, variance, LLN and CLT
PROJECT W3: Maximum likelihood estimation; Quiz 3
Linear regression (1 feature) and the cost function
Gradient descent for linear regression
Multiple linear regression, vectorised
Feature scaling and debugging gradient descent
Logistic regression and log-loss
Evaluation: confusion matrix, precision/recall, ROC-AUC, bias/variance
PROJECT W4: housing regression + error analysis; Quiz 4
MONTH-1 CAPSTONE: scratch models vs scikit-learn (+ Rust/PyO3 kernel)
Month-1 checkpoint exam and review
Regularisation deep dive: ridge, lasso, elastic net
Polynomial features and the bias-variance trade-off in practice
Cross-validation and hyperparameter search
Softmax and multiclass classification
PROJECT W5: California housing model with CV; Quiz 5
k-nearest neighbours; distance metrics; curse of dimensionality
Naive Bayes; generative vs discriminative models
Support vector machines and kernels
Decision trees: entropy, Gini, splitting
Random forests and bagging
Gradient boosting from scratch, then HistGradientBoosting
PROJECT W6: tabular pipeline with boosting; Quiz 6
Feature importance, permutation importance, SHAP intuition
Feature engineering: encodings, dates, text counts
Imbalanced data: class weights, resampling, threshold tuning
Pipelines, ColumnTransformer, leakage-proof preprocessing
k-means clustering and choosing k
Hierarchical clustering, DBSCAN, Gaussian mixtures (EM intuition)
PROJECT W7: customer segmentation; Quiz 7
PCA in practice; t-SNE/UMAP for visualisation
Anomaly detection
Recommender systems: collaborative filtering, matrix factorisation
Time-series basics: lag features, leakage-safe validation
Interpretability and probability calibration
Systematic ML strategy: learning curves, error analysis, what to try next
PROJECT W8: Rust/PyO3 k-means kernel vs NumPy; Quiz 8
Model persistence and reproducible experiments
Month-2 capstone build
MONTH-2 CAPSTONE: end-to-end tabular ML system with report
Month-2 checkpoint exam and review
Neurons and the perceptron: from logistic regression to a neural net
Forward propagation and activations; a 2-layer net in NumPy
PROJECT W9: backprop derived, gradient-checked NumPy MLP on digits; Quiz 9
PyTorch tensors, autograd, nn.Module, training loop
Datasets, DataLoaders, batching, devices
Initialisation; vanishing and exploding gradients
Optimisers: SGD, momentum, RMSProp, Adam; LR schedules
Regularisation: L2, dropout, early stopping, augmentation
Batch norm and layer norm
PROJECT W10: MNIST MLP in PyTorch with experiment log; Quiz 10
Convolutions from scratch (and as im2col matmul)
CNN architectures: LeNet to ResNet ideas
Training a CNN on CIFAR-10 (CPU-sized)
Transfer learning and fine-tuning a pretrained CNN
Debugging deep nets: overfit one batch, loss curves, sanity checks
Audio/image pipelines for multimedia devs (spectrograms, augmentation)
PROJECT W11: image or audio classifier; Quiz 11
RNNs and backprop through time
LSTMs and GRUs
Embeddings and word2vec intuition
Seq2seq and the motivation for attention
Mixed precision; profiling a PyTorch training run
JAX quick tour (optional): jit, grad, vmap
PROJECT W12: character-level language model; Quiz 12
Hyperparameter tuning and experiment tracking for deep learning
Month-3 capstone build
Month-3 capstone build: ablations
MONTH-3 CAPSTONE: deep-learning project with ablations
Cumulative review Months 1-3
Month-3 checkpoint exam
PROJECT W13: attention from scratch (Q, K, V); Quiz 13
Self-attention, multi-head attention, masking
Positional encodings (sinusoidal, RoPE)
The transformer block: residuals, layer norm, MLP
Tiny GPT from scratch
Tokenisation: BPE from scratch
Pretraining objectives and scaling laws
PROJECT W14: train tiny GPT on text; Quiz 14
Hugging Face transformers and datasets
Fine-tuning a small pretrained model
LoRA from scratch, then PEFT
Instruction tuning and chat templates
Evaluating LLMs: perplexity, benchmarks, LLM-as-judge pitfalls
RLHF overview: reward models, PPO intuition
PROJECT W15: LoRA fine-tune + evaluation; Quiz 15
DPO and preference optimisation
Embeddings and vector search
RAG: chunking, retrieval, grounding, evaluation
Prompting, structured output, function calling
Quantisation: int8/int4, GPTQ/AWQ ideas, measuring quality loss
KV cache, batching, speculative decoding
PROJECT W16: RAG over your own docs; Quiz 16
Inference serving concepts (vLLM, llama.cpp server)
llama.cpp and GGUF hands-on
Long context and efficient attention variants
Multimodal models overview (vision-language, speech)
Month-4 capstone build
Month-4 capstone build: quantise and serve
MONTH-4 CAPSTONE: fine-tuned, quantised, served LLM app; Quiz 17
Month-4 checkpoint exam
RL basics: MDPs, rewards, returns
Dynamic programming: value and policy iteration
Monte Carlo and TD learning; Q-learning
Deep Q-networks
Policy gradients and REINFORCE
PROJECT W18: gridworld Q-learning + REINFORCE agents; Quiz 18
Actor-critic and PPO
RL for LLMs revisited (RLHF, DPO, GRPO-style ideas)
Agents: tool use, planning, ReAct-style loops
Agent memory and multi-agent systems
Agent evaluation and guardrails
MLOps: lifecycle, scoping, baselines
PROJECT W19: tool-using agent with evals; Quiz 19
Data versioning and experiment tracking (MLflow)
Packaging and serving models (FastAPI, Docker)
CI/CD for ML: testing code, data and models
Monitoring: data drift and concept drift
Feature stores; batch vs online inference
Cost, latency and scaling trade-offs
PROJECT W20: deployed model with monitoring; Quiz 20
Data-centric AI: label quality, slicing, augmentation
Model registry, A/B tests, shadow and canary deployment
LLMOps: tracing and evaluation pipelines for LLM apps
Month-5 capstone build
Month-5 capstone build: deployment
Month-5 capstone build: monitoring and evals
MONTH-5 CAPSTONE: agent or RL system deployed with MLOps; Quiz 21
Cumulative review Months 1-5
Month-5 checkpoint exam
Buffer / catch-up day and Month-6 planning
Profiling Python ML code: where the time goes
NumPy/PyTorch performance: memory layout, fusion, BLAS
Rust extensions with PyO3 for ML kernels
PROJECT W22: C++ pybind11 + PyTorch custom op, speed up a bottleneck; Quiz 22
GPU programming model and CUDA basics
Triton kernels: vector add, softmax, matmul
FlashAttention ideas
Distributed training basics: data, tensor, pipeline parallelism
ONNX export and ONNX Runtime
On-device ML on Android (LiteRT, ExecuTorch, NNAPI overview)
PROJECT W23: on-device model in an Android app; Quiz 23
Quantisation-aware training, pruning, distillation
llama.cpp on mobile/edge; benchmarking tokens/sec
Real-time multimedia ML pipelines (audio/video)
Responsible AI: fairness metrics and audits
Privacy, security, prompt injection, misuse
Explainability and documentation (model cards, datasheets)
PROJECT W24: responsible-AI audit of your capstones; Quiz 24
Governance, regulation, careers; reading papers efficiently
FINAL CAPSTONE: design doc and scoping
FINAL CAPSTONE: data and baseline
FINAL CAPSTONE: model development
FINAL CAPSTONE: error analysis and iteration
FINAL CAPSTONE: optimisation and deployment (on-device or served)
PROJECT W25: FINAL CAPSTONE evaluation + responsible-AI review; Quiz 25
FINAL CAPSTONE: write-up and demo video (week 26 starts)
Cumulative final review
Final exam (cumulative)
FINAL CAPSTONE rehearsal and polish
FINAL CAPSTONE PRESENTATION and course retrospective""".split("\n")
assert len(T) == 180, len(T)
def month(d): return (d - 1) // 30 + 1
rows = [(d, (d-1)//7 + 1, month(d), t) for d, t in enumerate(T, 1)]
table = "| Day | Week | Month | Topic | Status |\n|---|---|---|---|---|\n" + "\n".join(
    f"| {d} | {w} | {m} | {t} | {'written' if d <= 30 else 'Phase ' + str(m)} |" for d, w, m, t in rows)
with open("/tmp/SYLLABUS_TABLE.md.part", "w") as f: f.write(table)
with open(os.path.join(os.path.dirname(__file__), "..", "review-schedule.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["study_day", "review_days(+1,+3,+7,+21)"])
    rev = {d: [] for d in range(1, 181)}
    for d in range(1, 181):
        for k in (1, 3, 7, 21):
            if d + k <= 180: rev[d + k].append(d)
    for d in range(1, 181): w.writerow([d, " ".join(map(str, rev[d]))])
print("ok", len(rows))
