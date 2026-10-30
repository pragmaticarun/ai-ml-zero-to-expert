"""Day 122: dynamic programming. Value iteration and policy iteration on our FrozenLake 4x4/8x8 (slippery), checked against
gymnasium's FrozenLake transition table; the greedy policy is then run in gymnasium itself to measure its success rate."""
import time, numpy as np, gymnasium as gym
from _rl import lake, LAKE4, LAKE8, model, gridworld
def value_iteration(T, R, g, tol=1e-10):
    V = np.zeros(len(R)); k = 0
    while True:
        k += 1; Q = R + g * T @ V; Vn = Q.max(1)
        if np.abs(Vn - V).max() < tol: return Vn, Q.argmax(1), k
        V = Vn
def policy_iteration(T, R, g):
    nS = len(R); pi = np.zeros(nS, int); it = 0
    while True:
        it += 1; Tp = T[np.arange(nS), pi]; V = np.linalg.solve(np.eye(nS) - g * Tp, R[np.arange(nS), pi])
        new = (R + g * T @ V).argmax(1)
        if (new == pi).all(): return V, pi, it
        pi = new
P, nS, nA = gridworld(); T, R = model(P, nS, nA); V, pi, k = value_iteration(T, R, 1.0)
print(f"gridworld 4x4 optimal V (value iteration, {k} sweeps):"); print(V.reshape(4, 4))
for name, desc, gname in (("4x4", LAKE4, "4x4"), ("8x8", LAKE8, "8x8")):
    P, nS, nA = lake(desc); ge = gym.make("FrozenLake-v1", map_name=gname, is_slippery=True); GP = ge.unwrapped.P
    same = all(sorted((round(p, 6), s2, r, d) for p, s2, r, d in P[s][a]) == sorted((round(p, 6), s2, float(r), d) for p, s2, r, d in GP[s][a]) for s in range(nS) for a in range(4))
    T, R = model(P, nS, nA)
    for g in (0.9, 0.99):
        t = time.perf_counter(); Vv, pv, k = value_iteration(T, R, g); tv = time.perf_counter() - t
        t = time.perf_counter(); Vp, pp, it = policy_iteration(T, R, g); tp = time.perf_counter() - t
        rng_seed = 0; wins = 0; N = 2000
        for ep in range(N):
            s, _ = ge.reset(seed=ep); d = tr = False
            while not (d or tr): s, r, d, tr, _ = ge.step(int(pv[s])); wins += r
        print(f"FrozenLake {name} slippery, gamma={g}: our table == gymnasium's: {same}; VI {k} sweeps {tv*1e3:.1f} ms, PI {it} iterations {tp*1e3:.1f} ms, "
              f"max|V_VI-V_PI| {np.abs(Vv-Vp).max():.1e}; V(start) {Vv[0]:.3f}; greedy policy success in gymnasium {wins/N:.3f} over {N} episodes")
    if name == "4x4": print("  optimal actions 4x4 (gamma 0.99; L D R U):"); print("  " + "\n  ".join("".join("LDRU"[pv[r * 4 + c]] if desc[r][c] in "SF" else desc[r][c] for c in range(4)) for r in range(4)))
P, nS, nA = lake(LAKE4, slippery=False); T, R = model(P, nS, nA); Vv, pv, k = value_iteration(T, R, 0.99); print(f"FrozenLake 4x4 NOT slippery: V(start) {Vv[0]:.3f} (= 0.99^5 = {0.99**5:.3f}, the 6-step shortest path), {k} sweeps")
