"""Day 19: probability by simulation and Bayes' rule (spam-filter and medical-test examples)."""
import numpy as np
rng = np.random.default_rng(19)
# Medical test: prevalence 1%, sensitivity 95%, false-positive rate 5%
prev, sens, fpr = 0.01, 0.95, 0.05
p_pos = sens*prev + fpr*(1-prev); print(f"Bayes: P(disease | +) = {sens*prev/p_pos:.4f}")
N = 1_000_000; sick = rng.random(N) < prev; pos = np.where(sick, rng.random(N) < sens, rng.random(N) < fpr)
print(f"simulation: P(disease | +) ≈ {sick[pos].mean():.4f}  (from {pos.sum()} positives)")
# Two dice: P(sum=7), conditional P(sum=7 | first die=3)
d = rng.integers(1, 7, (N, 2)); s = d.sum(1)
print(f"P(sum=7) ≈ {(s==7).mean():.4f} (exact {6/36:.4f});  P(sum=7 | d1=3) ≈ {(s[d[:,0]==3]==7).mean():.4f}")
# Tiny naive Bayes by hand: word 'free' in spam vs ham
p_spam, p_free_spam, p_free_ham = 0.3, 0.4, 0.02
print(f"P(spam | 'free') = {p_free_spam*p_spam/(p_free_spam*p_spam + p_free_ham*(1-p_spam)):.4f}")
