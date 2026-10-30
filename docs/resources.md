# Free / Open Resources (every URL verified with curl → HTTP 200 on 2026-10-29)

Links that did not verify were dropped. Re-check with `bash docs/check_links.sh`.

## Official documentation
| Tool | URL |
|---|---|
| Python tutorial | https://docs.python.org/3/tutorial/ |
| NumPy | https://numpy.org/doc/stable/ |
| pandas | https://pandas.pydata.org/docs/ |
| matplotlib | https://matplotlib.org/stable/ |
| scikit-learn | https://scikit-learn.org/stable/ |
| PyTorch | https://pytorch.org/docs/stable/index.html |
| JAX | https://docs.jax.dev/en/latest/ |
| PyO3 (Rust ↔ Python) | https://pyo3.rs/ |
| pybind11 (C++ ↔ Python) | https://pybind11.readthedocs.io/en/stable/ |
| Triton | https://triton-lang.org/main/index.html |
| ONNX | https://onnx.ai/ |
| ONNX Runtime | https://onnxruntime.ai/docs/ |
| llama.cpp | https://github.com/ggml-org/llama.cpp |
| Hugging Face docs | https://huggingface.co/docs |
| MLflow | https://mlflow.org/docs/latest/index.html |
| Docker | https://docs.docker.com/ |

## Open textbooks & courses
| Resource | URL | Use in course |
|---|---|---|
| Mathematics for Machine Learning (Deisenroth, Faisal, Ong) | https://mml-book.github.io/ | Month 1 maths |
| Dive into Deep Learning | https://d2l.ai/ | Months 3-4 |
| An Introduction to Statistical Learning (ISLR/ISLP) | https://www.statlearning.com/ | Month 2 |
| Deep Learning (Goodfellow, Bengio, Courville) | https://www.deeplearningbook.org/ | Month 3 reference |
| Python Data Science Handbook (VanderPlas) | https://jakevdp.github.io/PythonDataScienceHandbook/ | Month 1 NumPy/pandas |
| 3Blue1Brown linear algebra series | https://www.3blue1brown.com/topics/linear-algebra | Week 2 visuals |
| MIT 18.06 Linear Algebra (Strang) | https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/ | Week 2 depth |
| Hugging Face LLM course | https://huggingface.co/learn/llm-course | Month 4 |
| OpenAI Spinning Up (RL) | https://spinningup.openai.com/ | Month 5 |

Note: the Sutton & Barto RL book page (incompleteideas.net) timed out from this box, so it is not listed. Re-check it in Phase 2.

## Datasets
| Dataset source | URL |
|---|---|
| scikit-learn bundled & fetchable datasets | https://scikit-learn.org/stable/datasets.html |
| torchvision datasets (MNIST, CIFAR-10, …) | https://docs.pytorch.org/vision/stable/datasets.html |
| UCI Machine Learning Repository | https://archive.ics.uci.edu/ |
| Kaggle datasets (free account needed) | https://www.kaggle.com/datasets |
| Hugging Face datasets hub | https://huggingface.co/datasets |
| OpenML (adult, bike-sharing, CIFAR-10-small) | https://www.openml.org/ |
| CIFAR-10-small, OpenML 40926 (Month 3) | https://www.openml.org/d/40926 |
| Tiny Shakespeare (char-rnn repo, Month 3) | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt |
| MovieLens 100K zip (GroupLens) | https://files.grouplens.org/datasets/movielens/ml-100k.zip |
| UCI Wholesale customers | https://archive.ics.uci.edu/dataset/292/wholesale+customers |

## Month 3: deep learning (Weeks 10-13; every link checked with docs/check_links.sh)
| Topic | Link | Used on |
|---|---|---|
| Convolution arithmetic guide (Dumoulin & Visin) | https://arxiv.org/abs/1603.07285 | Day 71 |
| ResNet paper (He et al., 2015) | https://arxiv.org/abs/1512.03385 | Days 72, 86 |
| torchvision models and pretrained weights | https://docs.pytorch.org/vision/stable/models.html | Day 74 |
| A Recipe for Training Neural Networks (Karpathy) | https://karpathy.github.io/2019/04/25/recipe/ | Day 75 |
| SpecAugment (Park et al., 2019) | https://arxiv.org/abs/1904.08779 | Day 76 |
| On the difficulty of training RNNs (Pascanu et al.) | https://arxiv.org/abs/1211.5063 | Day 78 |
| Understanding LSTM Networks (Olah) | https://colah.github.io/posts/2015-08-Understanding-LSTMs/ | Day 79 |
| word2vec with negative sampling (Mikolov et al., 2013) | https://arxiv.org/abs/1310.4546 | Day 80 |
| Neural MT by jointly learning to align and translate (Bahdanau et al.) | https://arxiv.org/abs/1409.0473 | Day 81 |
| PyTorch automatic mixed precision recipe | https://docs.pytorch.org/tutorials/recipes/recipes/amp_recipe.html | Day 82 |
| PyTorch profiler recipe | https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html | Day 82 |
| Perfetto trace viewer | https://perfetto.dev/ | Day 82 |
| PyO3 user guide / maturin | https://pyo3.rs/ , https://www.maturin.rs/ | Day 82 |
| torch.func (grad, vmap) | https://docs.pytorch.org/docs/stable/func.html | Day 83 |
| Thinking in JAX / The Sharp Bits | https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html , https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html | Day 83 |
| The Unreasonable Effectiveness of RNNs (Karpathy) | https://karpathy.github.io/2015/05/21/rnn-effectiveness/ | Day 84 |
| Random search for hyper-parameter optimisation (Bergstra & Bengio, JMLR 2012) | https://www.jmlr.org/papers/v13/bergstra12a.html | Day 85 |
| Practical recommendations for gradient-based training (Bengio, 2012) | https://arxiv.org/abs/1206.5533 | Day 85 |
| On calibration of modern neural networks (Guo et al., 2017) | https://arxiv.org/abs/1706.04599 | Days 77, 88 |
| Attention Is All You Need (Vaswani et al., 2017) | https://arxiv.org/abs/1706.03762 | Day 91 |

## Month 4: transformers and LLMs (Weeks 14-18, Days 92-120; every link checked with docs/check_links.sh)

| Resource | URL | Used on |
|---|---|---|
| Neural Machine Translation of Rare Words with Subword Units (BPE; Sennrich et al.) | https://arxiv.org/abs/1508.07909 | Day 96 |
| RoFormer: Enhanced Transformer with Rotary Position Embedding | https://arxiv.org/abs/2104.09864 | Day 93 |
| Training Compute-Optimal Large Language Models (Chinchilla) | https://arxiv.org/abs/2203.15556 | Day 97 |
| LoRA: Low-Rank Adaptation of Large Language Models | https://arxiv.org/abs/2106.09685 | Day 101 |
| QLoRA: Efficient Finetuning of Quantized LLMs | https://arxiv.org/abs/2305.14314 | Day 102 |
| Training language models to follow instructions with human feedback (InstructGPT) | https://arxiv.org/abs/2203.02155 | Day 106 |
| Direct Preference Optimization: Your Language Model is Secretly a Reward Model | https://arxiv.org/abs/2305.18290 | Day 108 |
| Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks | https://arxiv.org/abs/1908.10084 | Day 113 |
| Efficient and robust approximate nearest neighbor search using HNSW graphs | https://arxiv.org/abs/1603.09320 | Day 113 |
| Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks | https://arxiv.org/abs/2005.11401 | Days 115, 120 |
| Lost in the Middle: How Language Models Use Long Contexts | https://arxiv.org/abs/2307.03172 | Day 115 |
| LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale | https://arxiv.org/abs/2208.07339 | Day 116 |
| GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers | https://arxiv.org/abs/2210.17323 | Day 117 |
| AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration | https://arxiv.org/abs/2306.00978 | Day 117 |
| Fast Inference from Transformers via Speculative Decoding | https://arxiv.org/abs/2211.17192 | Day 118 |
| Efficient Memory Management for LLM Serving with PagedAttention (vLLM) | https://arxiv.org/abs/2309.06180 | Days 118, 120 |
| Hugging Face transformers (repository and docs links) | https://github.com/huggingface/transformers | Weeks 15-17 |
| Hugging Face PEFT docs | https://huggingface.co/docs/peft/index | Days 101-102 |
| Hugging Face datasets docs | https://huggingface.co/docs/datasets/index | Day 99 |
| vLLM docs / repository | https://docs.vllm.ai/ , https://github.com/vllm-project/vllm | Day 118 (README only; not run) |
| vLLM PagedAttention blog post (linked from the vLLM README) | https://blog.vllm.ai/2023/06/20/vllm.html | Day 118 |
| llama.cpp repository | https://github.com/ggml-org/llama.cpp | Day 118 (README only; not built: no g++/cmake on this box) |
| FAISS (vector search library) | https://faiss.ai/ | Day 113 |

## Month 5: RL, agents, MLOps and LLMOps (Weeks 18-22, Days 121-150; every link checked with docs/check_links.sh)
| resource | link | used on |
|---|---|---|
| Gymnasium (RL environments) | https://gymnasium.farama.org/ | Days 121-133 |
| FastAPI | https://fastapi.tiangolo.com/ | Days 144, 147 |
| DVC docs | https://dvc.org/doc | Day 142 |
| Dockerfile reference | https://docs.docker.com/reference/dockerfile/ | Day 147 (Dockerfile UNTESTED: no Docker) |
| GitHub Actions docs | https://docs.github.com/en/actions | Day 147 (CI file UNTESTED) |
| Python cProfile/profile docs | https://docs.python.org/3/library/profile.html | Day 151 |

## Month 6: performance engineering (Week 22-, Days 151-; every link checked with docs/check_links.sh)
| resource | link | used on |
|---|---|---|
| Python tracemalloc docs | https://docs.python.org/3/library/tracemalloc.html | Day 151 |
| PyTorch profiler docs | https://docs.pytorch.org/docs/stable/profiler.html | Day 151 |
| SciPy sparse matrices | https://docs.scipy.org/doc/scipy/reference/sparse.html | Days 151, 154 |
| PyO3 user guide | https://pyo3.rs/ | Days 153-154 |
| maturin guide | https://www.maturin.rs/ | Days 153-154 |
| rayon (Rust data parallelism) | https://docs.rs/rayon/latest/rayon/ | Days 153-154 |
| The Rust Programming Language (book) | https://doc.rust-lang.org/book/ | Day 153 |
| ONNX Runtime install (incl. Android) | https://onnxruntime.ai/docs/install/ | Days 155, 159 |
| ORT Java API getting started | https://onnxruntime.ai/docs/get-started/with-java.html | Day 159 |
| ORT format models | https://onnxruntime.ai/docs/performance/model-optimizations/ort-format-models.html | Day 159 |
| ORT quantization | https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html | Day 156 |
| ORT threading | https://onnxruntime.ai/docs/performance/tune-performance/threading.html | Day 155 |
| ORT graph optimizations | https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html | Day 155 |
| ORT profiling tools | https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html | Day 155 |
| ExecuTorch on Android | https://docs.pytorch.org/executorch/stable/using-executorch-android.html | Day 159 (app UNTESTED) |
| ExecuTorch getting started | https://docs.pytorch.org/executorch/stable/getting-started.html | Day 159 |
| LiteRT for Android | https://ai.google.dev/edge/litert/android | Day 159 (UNTESTED) |
| llama.cpp Android doc | https://github.com/ggml-org/llama.cpp/blob/master/docs/android.md | Day 159 (UNTESTED) |
| NIST AI RMF 1.0 (NIST.AI.100-1) | https://nvlpubs.nist.gov/nistpubs/ai/NIST.AI.100-1.pdf | Day 167 |
| EU Commission AI Act overview | https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai | Day 167 (overview; not legal advice) |
| OpenML Adult (id 1590) | https://www.openml.org/d/1590 | Days 162-168 |
| Opacus documentation | https://opacus.ai/ | Day 165 (RDP accountant cross-check) |
| PyTorch pruning tutorial | https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html | Day 157 |
| Triton docs | https://triton-lang.org/main/index.html | Day 154 (UNTESTED: no GPU, not installed) |
| Triton fused-softmax tutorial | https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html | Day 154 (UNTESTED) |
| NVIDIA CUDA C++ Programming Guide | https://docs.nvidia.com/cuda/cuda-c-programming-guide/ | Day 154 (UNTESTED: no GPU/nvcc) |


## Month 6 closeout (Weeks 25-26, Days 169-180; links already verified above)
| resource | link | used on |
|---|---|---|
| Capstone freeze + reports (local) | `code/capstone/out/` (FINAL_EVAL_REPORT, REFERENCE_REPORT, REPORT_TEMPLATE, DEMO_CHECKLIST, FINAL_EXAM, WHAT_NEXT, SKILLS_MATRIX, RETROSPECTIVE, MODEL_CARD) | Days 176-180 |
| Week 26 logs (local) | `code/week-26/out/` | Days 176-180 |
| Hugging Face SmolLM2-135M-Instruct | https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct | Days 169-180 |
| sentence-transformers MiniLM | https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 | Days 169-180 |
| Anki | https://ankiweb.net/ | Day 180 what-next / spaced repetition |

## Inspiration (Andrew Ng courses; see teaching-style.md)
- https://www.coursera.org/specializations/machine-learning-introduction
- https://www.coursera.org/specializations/deep-learning
- https://www.coursera.org/learn/introduction-to-machine-learning-in-production
- https://www.coursera.org/learn/generative-ai-with-llms
- https://www.coursera.org/learn/ai-for-everyone
- https://www.deeplearning.ai/short-courses/
- https://cs229.stanford.edu/
- https://see.stanford.edu/Course/CS229

## Tools
- Anki (spaced repetition; imports our CSV flashcards): https://ankiweb.net/
