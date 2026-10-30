"""Day 14 project: compress an image with truncated SVD (synthetic image, no download)."""
import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from common import plt, savefig
import numpy as np
h, w = 128, 192; yy, xx = np.mgrid[0:h, 0:w]
img = (np.sin(xx/9) * np.cos(yy/13) + ((xx-96)**2 + (yy-64)**2 < 900) * 1.5 + (xx > 150) * .7
       + np.random.default_rng(14).normal(0, .05, (h, w)))
U, S, Vt = np.linalg.svd(img, full_matrices=False)
fig, ax = plt.subplots(1, 4, figsize=(14, 3)); ax[0].imshow(img, cmap="gray"); ax[0].set_title("original")
for a, k in zip(ax[1:], [2, 8, 32]):
    rec = (U[:, :k]*S[:k]) @ Vt[:k]; stored = k*(h+w+1)
    err = np.linalg.norm(img-rec)/np.linalg.norm(img)
    a.imshow(rec, cmap="gray"); a.set_title(f"k={k} rel.err={err:.3f}")
    print(f"k={k:2d}: stores {stored} numbers vs {h*w} ({stored/(h*w)*100:.1f}%), relative error {err:.4f}")
savefig(__file__, "day14_svd_compression.png")
