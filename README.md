# AI/ML: Novice to Expert in 180 Days

A complete, original, hands-on course for **Arunkumar**: an Android/multimedia developer comfortable with C++/Rust and new to ML. Design budget: **180 days × ~6 h/day ≈ 1,080 h**. Python is the main language (NumPy, pandas, scikit-learn, PyTorch). Rust appears wherever performance matters (PyO3). There is **no copyrighted course material** copied in; every measured number comes from scripts run on this box.

Teaching style: intuition → tiny example → plain-language maths → code. See [docs/teaching-style.md](docs/teaching-style.md).

## Status
**Complete through Day 180.** All 26 week files are written. **Week 26 has five days only** (176–180) and closes the expert capstone. Flashcards cover Months 1–6. Package verification and zip are the final QA step.

## How to use
1. Read [SYLLABUS.md](SYLLABUS.md) for the 180-day map, then open today’s week file under [weeks/](weeks/).
2. Work the timed blocks (~6 h). Prefer cutting practice over skipping spaced-repetition review.
3. **Run every lab** under `code/week-XX/`. Change one knob, predict, re-measure.
4. Keep a short log (`notes/dayDD.md`). Sit each weekly quiz closed-book, then check answers in the week file.
5. Import Anki decks from `flashcards/month-01.csv` … `month-06.csv`. Due tags each calendar study day are in [review-schedule.csv](review-schedule.csv) (+1/+3/+7/+21).
6. After Day 180 follow `code/capstone/out/WHAT_NEXT.md`.

## Course website
There is a static progress-tracking site in `site/`. It runs offline with no server. Open `site/index.html`, or rebuild it with `.venv/bin/python site/build.py`. See [site/README.md](site/README.md) for GitHub Pages setup and tests.

## Setup (tested: Linux, Python 3.13, CPU-only)
```bash
cd ai-ml-course
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash run_all.sh 1          # Month 1 smoke; use 1..6 for other months
# Optional Rust kernels (https://rustup.rs), examples:
#   cd code/week-05/rust_gd && maturin develop --release
#   cd code/capstone/rust_score && maturin develop --release
```
Labs are plain `.py` (diff-friendly, headless). Figures/logs land in `code/week-XX/out/`. Hugging Face weights (SmolLM2, MiniLM, …) download once into the HF cache — see [data/README.md](data/README.md); recreate with `data/download.py` / Hub, do not ship multi‑100MB weights in the zip.

## Index
| Doc | What |
|---|---|
| [SYLLABUS.md](SYLLABUS.md) | All 180 days, projects, quizzes, spaced repetition |
| [PROGRESS.md](PROGRESS.md) | What was written/verified; month pass/fail from final QA |
| [docs/resources.md](docs/resources.md) | Verified external links only |
| [docs/teaching-style.md](docs/teaching-style.md) | Pedagogy notes |
| [data/README.md](data/README.md) | Datasets/models and how to recreate them |
| [review-schedule.csv](review-schedule.csv) | Daily review days + Anki tags due |
| `flashcards/month-01.csv` … `month-06.csv` | Anki import (Front, Back, Tags) |
| `code/capstone/` | Expert Course Assistant (Weeks 25–26) |

## Weeks (all written)
[01](weeks/week-01.md) · [02](weeks/week-02.md) · [03](weeks/week-03.md) · [04](weeks/week-04.md) · [05](weeks/week-05.md) · [06](weeks/week-06.md) · [07](weeks/week-07.md) · [08](weeks/week-08.md) · [09](weeks/week-09.md) · [10](weeks/week-10.md) · [11](weeks/week-11.md) · [12](weeks/week-12.md) · [13](weeks/week-13.md) · [14](weeks/week-14.md) · [15](weeks/week-15.md) · [16](weeks/week-16.md) · [17](weeks/week-17.md) · [18](weeks/week-18.md) · [19](weeks/week-19.md) · [20](weeks/week-20.md) · [21](weeks/week-21.md) · [22](weeks/week-22.md) · [23](weeks/week-23.md) · [24](weeks/week-24.md) · [25](weeks/week-25.md) · [26](weeks/week-26.md) *(5 days)*

### Day-by-day pointer
| Days | Week file | Month theme (short) |
|---|---|---|
| 1–7 | [week-01](weeks/week-01.md) | Setup, NumPy, first models |
| 8–14 | [week-02](weeks/week-02.md) | Linear algebra / probability labs |
| 15–21 | [week-03](weeks/week-03.md) | Metrics, bias–variance, sklearn |
| 22–28 | [week-04](weeks/week-04.md) | Logistic / from-scratch + Month-1 wrap |
| 29–35 | [week-05](weeks/week-05.md) | Month-1 capstone + checkpoint; start classical ML |
| 36–42 | [week-06](weeks/week-06.md) | Trees, forests, boosting |
| 43–49 | [week-07](weeks/week-07.md) | SVM, clustering, EM |
| 50–56 | [week-08](weeks/week-08.md) | Dimensionality, pipelines |
| 57–63 | [week-09](weeks/week-09.md) | Month-2 tabular system + exam |
| 64–70 | [week-10](weeks/week-10.md) | Autodiff, MLP, training loops |
| 71–77 | [week-11](weeks/week-11.md) | CNNs, regularisation |
| 78–84 | [week-12](weeks/week-12.md) | RNNs/seq, data loading |
| 85–91 | [week-13](weeks/week-13.md) | Month-3 CNN capstone |
| 92–98 | [week-14](weeks/week-14.md) | Attention, GPT bits |
| 99–105 | [week-15](weeks/week-15.md) | HF fine-tune / LoRA |
| 106–112 | [week-16](weeks/week-16.md) | Preference / DPO eval pitfalls |
| 113–119 | [week-17](weeks/week-17.md) | RAG, quant, KV, Rust int8 |
| 120–126 | [week-18](weeks/week-18.md) | Month-4 capstone + RL start |
| 127–133 | [week-19](weeks/week-19.md) | Policy gradients / bandits / RLHF toys |
| 134–140 | [week-20](weeks/week-20.md) | Tool-using agents |
| 141–147 | [week-21](weeks/week-21.md) | MLOps service |
| 148–154 | [week-22](weeks/week-22.md) | LLMOps + Month-5 capstone + perf start |
| 155–161 | [week-23](weeks/week-23.md) | ONNX / quant / on-device budget |
| 162–168 | [week-24](weeks/week-24.md) | Responsible AI |
| 169–175 | [week-25](weeks/week-25.md) | Expert Course Assistant build |
| 176–180 | [week-26](weeks/week-26.md) | Polish, eval, write-up, final exam *(5 days)* |

Full day titles live in [SYLLABUS.md](SYLLABUS.md) and each week file’s `## Day N:` headings (continuous 1–180).

## Honest limitations
- **CPU-only** measurements on a shared box; timings vary (±10–30%). Not phone/GPU claims.
- **No pandoc/PDF** required; PDF export is best-effort in final QA.
- **No g++/C++ pybind** path; Rust/PyO3 is used instead. **llama.cpp** not built.
- **UNTESTED** on this box: Android on-device apps, Docker/CI execution, GPU/Triton/CUDA, some ORT Java thread APIs.
- Large **HF weights**, `.venv/`, `target/`, `__pycache__/`, and bulky `out/` caches are **not** shipped in the zip — recreate via `data/download.py` / Hub / `maturin` / by re-running labs.
- Capstone release decision: **ACCEPT_COURSE_NO_PROD** (freeze recall@3 = 0.72; paraphrase injection gaps remain).

## Daily schedule template (6 h)
| Block | Time | What |
|---|---|---|
| Concept | 1h15 | Intuition → example → maths |
| Worked example | 0h45 | Hand then code |
| Lab | 2h00 | `code/week-XX/dayDD_*.py` + tasks |
| Practice | 1h00 | Problems / quiz prep |
| Review | 1h00 | Spaced repetition + self-check + Anki |

Example (Asia/Calcutta): 07:00–08:15 lesson · 08:30–09:15 example · 09:30–11:30 lab · 14:00–15:00 practice · 18:00–19:00 review.
