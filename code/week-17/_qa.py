"""Hand-written factual questions whose answer string appears in the course notes (checked by day115 at run time)."""
QA = [
 ("In the Week 16 reward-model experiment, what pairwise accuracy did RM B get when the '!' was moved to the wrong answer?", "0.00"),
 ("What exact match did plain DPO reach on Day 108 after training?", "0.17"),
 ("What exact match did SFT on chosen answers reach in the Week 16 project?", "0.70"),
 ("What Shakespeare validation perplexity did the SmolLM2 instruct model get in Week 15?", "32.05"),
 ("What Shakespeare perplexity did the base SmolLM2-135M model get in Week 15?", "26.29"),
 ("How many trainable parameters did the Day 107 reward model have?", "2,442,817"),
 ("What was the bootstrap 95% confidence interval for exact match on Day 110?", "[0.07, 0.30]"),
 ("On Day 111, what pass@8 did the oracle verifier give?", "0.20"),
 ("What is the DPO loss at step 0 when the LoRA B matrices are zero?", "0.693"),
 ("How many of 16 verdicts were consistent when answers were swapped for the 135M judge in Week 15?", "0 of 16"),
 ("On Day 106, what expected reward did the optimal policy get with beta 0.05?", "0.993"),
 ("In Day 109, how much did the rejected log-prob change with beta 0.5?", "−10.3"),
]
