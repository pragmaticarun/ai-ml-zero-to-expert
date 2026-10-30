"""Day 121: MDPs, returns and the Bellman expectation equation. Evaluate the uniform-random policy on the 4x4 gridworld
three ways: exact linear solve, iterative policy evaluation, and Monte Carlo sampling; then effect of gamma and of slip noise."""
import time, numpy as np
from _rl import gridworld, model, step
P, nS, nA = gridworld(); T, R = model(P, nS, nA); pi = np.full((nS, nA), 0.25)
def exact(T, R, pi, g):
    Tp = np.einsum("sa,sat->st", pi, T); Rp = (pi * R).sum(1); return np.linalg.solve(np.eye(len(Rp)) - g * Tp, Rp)
def iterative(T, R, pi, g, tol=1e-8):
    V = np.zeros(len(R)); k = 0
    while True:
        k += 1; Vn = (pi * (R + g * T @ V)).sum(1)
        if np.abs(Vn - V).max() < tol: return Vn, k
        V = Vn
g = 1.0; t = time.perf_counter(); Ve = exact(T, R, pi, g); te = time.perf_counter() - t
t = time.perf_counter(); Vi, k = iterative(T, R, pi, g); ti = time.perf_counter() - t
print(f"random policy, gamma=1: exact solve {te*1e3:.2f} ms; iterative evaluation {k} sweeps, {ti*1e3:.1f} ms, max diff {np.abs(Ve-Vi).max():.1e}")
print("V_random (exact), 4x4:"); print(np.round(Ve.reshape(4, 4), 1))
rng = np.random.default_rng(0); s0 = 1; G = []
for _ in range(2000):
    s, ret, d = s0, 0.0, False
    while not d: s, r, d = step(P, s, rng.integers(4), rng); ret += r
    G.append(ret)
G = np.array(G); print(f"Monte Carlo from state 1, 2000 episodes: mean return {G.mean():.2f} ± {1.96*G.std()/np.sqrt(len(G)):.2f} (95% CI), exact {Ve[s0]:.2f}; episode lengths min {int(-G.max())}, max {int(-G.min())}")
for g in (0.5, 0.9, 0.99):
    V = exact(T, R, pi, g); print(f"gamma={g}: V(state 1) {V[1]:.2f}, V(state 7) {V[7]:.2f}  (bounded by -1/(1-gamma) = {-1/(1-g):.1f})")
for slip in (0.0, 0.2, 0.5):
    P2, *_ = gridworld(slip=slip); T2, R2 = model(P2, nS, nA); Vg = exact(T2, R2, np.eye(4)[[2 if s // 4 == 3 else 1 for s in range(nS)]], 1.0)
    print(f"slip={slip}: 'go down, then right' policy, V(state 1) {Vg[1]:.2f}, V(state 4) {Vg[4]:.2f}")
