"""Day 5: plotting for EDA: histogram, scatter, residual-style plots."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
rng = np.random.default_rng(5)
area = rng.normal(1200, 350, 300).clip(350); price = 0.05*area + rng.normal(0, 8, 300)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
ax[0].hist(area, bins=30); ax[0].set_title("area histogram"); ax[0].set_xlabel("sq ft")
ax[1].scatter(area, price, s=8, alpha=.6); ax[1].set_title("price vs area"); ax[1].set_xlabel("sq ft"); ax[1].set_ylabel("lakh")
w, b = np.polyfit(area, price, 1); xs = np.linspace(area.min(), area.max(), 50)
ax[1].plot(xs, w*xs + b, "r"); resid = price - (w*area + b)
ax[2].scatter(area, resid, s=8); ax[2].axhline(0, color="k"); ax[2].set_title("residuals")
savefig(__file__, "day05_eda.png")
print(f"fitted slope={w:.4f} intercept={b:.2f}; residual std={resid.std():.2f}")
# A non-linear trap: a straight line on curved data leaves a pattern in residuals
x = rng.uniform(-3, 3, 200); y = x**2 + rng.normal(0, .5, 200)
w2, b2 = np.polyfit(x, y, 1); r2 = y - (w2*x + b2)
plt.figure(figsize=(8, 3)); plt.subplot(1, 2, 1); plt.scatter(x, y, s=6); plt.plot(np.sort(x), w2*np.sort(x)+b2, "r")
plt.subplot(1, 2, 2); plt.scatter(x, r2, s=6); plt.title("U-shaped residuals => model too simple")
savefig(__file__, "day05_residual_pattern.png")
print("corr(|x|, residual) for curved data:", round(np.corrcoef(np.abs(x), r2)[0, 1], 3))
