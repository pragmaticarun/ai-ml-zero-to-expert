"""Day 22: univariate linear regression, the cost function J(w,b), and its bowl shape."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
from housing import make_housing
X, y, _ = make_housing(200, seed=22); x = X[:, 0] / 1000   # area in 1000s of sq ft
def J(w, b): return ((w*x + b - y)**2).mean() / 2
ws, bs = np.linspace(0, 100, 120), np.linspace(-40, 80, 120); W, B = np.meshgrid(ws, bs)
Jg = np.vectorize(J)(W, B); i = np.unravel_index(Jg.argmin(), Jg.shape)
print(f"grid min J={Jg[i]:.2f} at w={W[i]:.2f}, b={B[i]:.2f}; closed form {np.polyfit(x, y, 1).round(2)}")
for w, b in [(0, y.mean()), (30, 30), (W[i], B[i])]: print(f"J(w={w:.1f}, b={b:.1f}) = {J(w, b):.2f}")
fig, ax = plt.subplots(1, 2, figsize=(10, 4)); ax[0].scatter(x, y, s=6); ax[0].plot(np.sort(x), W[i]*np.sort(x)+B[i], "r")
ax[0].set_xlabel("area (1000 sq ft)"); ax[0].set_ylabel("price (lakh)"); c = ax[1].contour(W, B, Jg, 30); ax[1].plot(W[i], B[i], "rx")
ax[1].set_xlabel("w"); ax[1].set_ylabel("b"); savefig(__file__, "day22_cost_contour.png")
