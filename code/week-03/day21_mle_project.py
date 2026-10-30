"""Day 21: maximum likelihood. Coin bias, Gaussian mean/var, and 'least squares = Gaussian MLE'."""
import numpy as np
rng = np.random.default_rng(21)
flips = rng.random(200) < 0.7; k, n = flips.sum(), len(flips)
ps = np.linspace(0.01, 0.99, 981); ll = k*np.log(ps) + (n-k)*np.log(1-ps)
print(f"coin: heads {k}/{n}; grid MLE p={ps[ll.argmax()]:.3f}; closed form k/n={k/n:.3f}")
x = rng.normal(5, 2, 500); mu, var = x.mean(), x.var()
print(f"gaussian MLE mu={mu:.3f}, var={var:.3f} (MLE divides by n; unbiased n-1 gives {x.var(ddof=1):.3f})")
# Least squares line == MLE with Gaussian noise: maximise log-lik by gradient ascent and compare to lstsq
X = rng.uniform(0, 10, 100); y = 2.5*X + 1 + rng.normal(0, 1.5, 100)
w = b = 0.0
for _ in range(20000):
    r = y - (w*X + b); w += 1e-3 * (r*X).mean(); b += 1e-3 * r.mean() * 10
print(f"gradient-ascent log-lik: w={w:.4f} b={b:.4f} | lstsq: {np.polyfit(X, y, 1).round(4)}")
