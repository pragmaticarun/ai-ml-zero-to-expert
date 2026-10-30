"""Day 18: gradient descent on 1-D and 2-D bowls; learning rate too small / right / too large."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
f = lambda v: v[0]**2 + 10*v[1]**2; g = lambda v: np.array([2*v[0], 20*v[1]])
plt.figure(figsize=(6, 4)); xx, yy = np.meshgrid(np.linspace(-6, 6, 100), np.linspace(-3, 3, 100))
plt.contour(xx, yy, xx**2 + 10*yy**2, 20)
for lr, c in [(0.01, "b"), (0.09, "g"), (0.105, "r")]:
    v = np.array([-5.0, 2.0]); path = [v.copy()]
    for _ in range(40): v = v - lr*g(v); path.append(v.copy())
    p = np.array(path); plt.plot(p[:, 0], p[:, 1], c+".-", label=f"lr={lr}")
    print(f"lr={lr}: f after 40 steps = {f(v):.4g}")
plt.ylim(-3, 3); plt.legend(); savefig(__file__, "day18_gd_paths.png")
print("divergence threshold for the y-direction: lr < 2/20 =", 2/20)
