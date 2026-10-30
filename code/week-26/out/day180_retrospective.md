# Course Retrospective (Day 180)

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
