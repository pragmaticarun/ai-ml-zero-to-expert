"""Day 8: vectors, norms, dot product, cosine similarity (toy 'song embedding' example)."""
import numpy as np
songs = {  # invented features: [tempo, energy, acoustic, vocals]
    "lofi_beat": np.array([0.3, 0.2, 0.6, 0.1]), "edm_drop": np.array([0.9, 0.95, 0.05, 0.3]),
    "unplugged": np.array([0.35, 0.25, 0.9, 0.8]), "gym_mix": np.array([0.85, 0.9, 0.1, 0.4])}
def cos(a, b): return a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
q = songs["edm_drop"]
for k, v in songs.items():
    print(f"{k:10s} dot={q@v:.3f} cos={cos(q, v):.3f} L2dist={np.linalg.norm(q-v):.3f}")
v = np.array([3.0, 4.0]); print("L1", np.abs(v).sum(), "L2", np.linalg.norm(v), "Linf", np.abs(v).max())
u = np.array([1.0, 0.0]); print("projection of v onto u:", (v @ u) * u, " angle(deg):", np.degrees(np.arccos(cos(v, u))).round(2))
print("dot as weighted sum, i.e. a linear model: w=[2,-1], x=[3,4] ->", np.array([2, -1]) @ v)
