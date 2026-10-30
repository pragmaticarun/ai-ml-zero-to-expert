# Teaching Style Guide (inspired by Andrew Ng's courses)

This course borrows a *teaching approach*, not content. Every lesson, example, dataset and lab in this repository is original. Andrew Ng's courses are cited only as inspiration; nothing here is copied from them.

## Sources consulted (all fetched successfully on 2026-10-29)
- Machine Learning Specialization (Coursera / DeepLearning.AI / Stanford Online): https://www.coursera.org/specializations/machine-learning-introduction — its public description states that each lesson starts with a visual picture and a high-level intuition, followed by code, with the deeper maths available separately as optional videos. It also says ungraded notebooks with interactive graphs come before the graded labs, and that a large section covers practical advice on applying ML.
- Machine Learning in Production: https://www.coursera.org/learn/introduction-to-machine-learning-in-production — its syllabus covers establishing a baseline, error analysis, prioritising what to work on, skewed datasets, data-centric development, label consistency, human-level performance and scoping. The videos are short (mostly 3-14 minutes).
- Deep Learning Specialization: https://www.coursera.org/specializations/deep-learning
- Generative AI with LLMs: https://www.coursera.org/learn/generative-ai-with-llms
- AI for Everyone: https://www.coursera.org/learn/ai-for-everyone
- DeepLearning.AI short courses catalogue: https://www.deeplearning.ai/short-courses/
- Stanford CS229: https://cs229.stanford.edu/ and the archived lectures at https://see.stanford.edu/Course/CS229

## The style, in our own words
1. **Intuition before symbols.** Each concept opens with a picture or story you can reason about, such as a line through house prices or a bowl you roll a ball down. The formula comes only after that.
2. **One small concrete example, worked by hand.** Seven houses work better than seven thousand at this stage. When the numbers are small enough to check on paper, you can trust what the code prints.
3. **Plain-language maths.** Every equation gets a sentence saying what it means (for example, "the gradient tells each weight which way is uphill and how steep").
4. **Simple to complex, one rung at a time.** We go from one feature to many features, from regression to classification, from linear models to neural nets to transformers. Each new model is the last one plus a single idea.
5. **Short, focused units.** Each 6-hour day is cut into timed blocks of 30-90 minutes, and each block has one goal.
6. **Hands-on every day.** Every day has runnable code that has actually been run. You learn by changing a number and watching what happens.
7. **Pictures.** We use ASCII or Mermaid diagrams in the notes and matplotlib figures from the labs (saved under `code/week-XX/out/`).
8. **Practical engineering advice.** We cover baselines, train/dev/test discipline, learning curves, bias vs variance, error analysis by slice, and "what to try next" decision lists. Each day ends with a short **What to do next** box.
9. **Debugging as a first-class skill.** Every day has a **Common mistakes / how to debug** section: shape bugs, divergence, leakage, and wrong metrics.
10. **Systems contrast for this learner.** Arunkumar knows C++, Rust and Android multimedia, so we point out where ML code is really about memory layout, BLAS, cache behaviour and data pipelines, just as in media codecs.

## Rules for authors of later phases
- Use only original examples. Do not reproduce course slides, assignments or quiz questions.
- Every number in a lesson must come from a script in `code/` that was actually run, and must be quoted from its output.
- Cite only URLs you have verified (HTTP 200) in `docs/resources.md`.
