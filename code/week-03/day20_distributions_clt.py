"""Day 20: distributions, expectation and variance, law of large numbers and CLT."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
rng = np.random.default_rng(20)
for name, s, Ev, Vv in [("Bernoulli(0.3)", rng.random(10**6) < .3, .3, .21), ("Uniform(0,1)", rng.random(10**6), .5, 1/12),
                        ("Exponential(rate 2)", rng.exponential(.5, 10**6), .5, .25), ("Normal(3, 2^2)", rng.normal(3, 2, 10**6), 3, 4)]:
    print(f"{name:20s} sample mean {s.mean():.4f} (theory {Ev:.4f}) | sample var {s.var():.4f} (theory {Vv:.4f})")
fig, ax = plt.subplots(1, 3, figsize=(12, 3))
for a, n in zip(ax, [1, 5, 50]):
    means = rng.exponential(1, (20000, n)).mean(1); a.hist(means, 60, density=True); a.set_title(f"mean of {n} exponentials")
    print(f"n={n:2d}: std of sample mean {means.std():.4f} vs 1/sqrt(n) = {1/np.sqrt(n):.4f}")
savefig(__file__, "day20_clt.png")
x = rng.normal(0, 1, 10**6); print("Normal: P(|x|<1) ≈", (np.abs(x) < 1).mean().round(4), " P(|x|<2) ≈", (np.abs(x) < 2).mean().round(4))
