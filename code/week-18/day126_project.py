"""Day 126 mini-project: learn FrozenLake 8x8 (slippery) without the model. Tabular Q-learning and SARSA written from scratch
(our own transition sampler), gamma 0.99, alpha 0.1, epsilon 1 -> 0.05 linearly over the first 80% of episodes. Compare with the
DP optimum (value iteration on the known model): V(start), policy agreement, and greedy success rate in gymnasium (2,000 episodes)."""
import time, numpy as np, gymnasium as gym
from _rl import lake, LAKE8, model, step
def value_iteration(T, R, g, tol=1e-10):  # same as day122 (copied so importing does not re-run day122)
    V = np.zeros(len(R)); k = 0
    while True:
        k += 1; Q = R + g * T @ V; Vn = Q.max(1)
        if np.abs(Vn - V).max() < tol: return Vn, Q.argmax(1), k
        V = Vn
P, nS, nA = lake(LAKE8, True)[:3]; T, R = model(P, nS, nA); g = 0.99
Vs, pi_star, _ = value_iteration(T, R, g)
genv = gym.make("FrozenLake-v1", map_name="8x8", is_slippery=True)
def success(pi, n=2000):
    w = 0
    for k in range(n):
        s, _ = genv.reset(seed=5 * 10 ** 5 + k); d = tr = False
        while not (d or tr): s, r, d, tr, _ = genv.step(int(pi[s]))
        w += r
    return w / n
holes_goal = [i for i, ch in enumerate("".join(LAKE8)) if ch in "HG"]; free = [s for s in range(nS) if s not in holes_goal]
def act(Q, s, eps, rng):
    if rng.random() < eps: return int(rng.integers(nA))
    q = Q[s]; return int(rng.choice(np.flatnonzero(q == q.max())))
def learn(alg, n_ep, seed, alpha=0.1):
    rng = np.random.default_rng(seed); Q = np.zeros((nS, nA)); wins = []
    for ep in range(n_ep):
        eps = max(0.05, 1 - ep / (0.8 * n_ep)); s = 0; a = act(Q, s, eps, rng); d = False; t = 0
        while not d and t < 200:
            s2, r, d = step(P, s, a, rng); a2 = act(Q, s2, eps, rng); t += 1
            nxt = 0 if d else (Q[s2].max() if alg == "Q-learning" else Q[s2, a2])
            Q[s, a] += alpha * (r + g * nxt - Q[s, a]); s, a = s2, a2
        wins.append(r)
    return Q, np.array(wins)
print(f"DP optimum (value iteration, gamma {g}): V(start) {Vs[0]:.3f}; greedy success in gymnasium {success(pi_star):.3f}")
for n_ep in (5000, 20000):
    for alg in ("Q-learning", "SARSA"):
        for seed in range(3):
            t0 = time.perf_counter(); Q, wins = learn(alg, n_ep, seed); pi = Q.argmax(1)
            agree = np.mean([np.isclose(T[s, pi[s]] @ Vs, T[s, pi_star[s]] @ Vs, atol=1e-6) or pi[s] == pi_star[s] for s in free])
            print(f"{alg}, {n_ep} episodes, seed {seed}: learned max Q(start) {Q[0].max():.3f} (DP {Vs[0]:.3f}); online success last 1000 eps {wins[-1000:].mean():.3f}; "
                  f"greedy action optimal (ties allowed) in {agree:.0%} of non-terminal states; greedy success in gymnasium {success(pi):.3f}; {time.perf_counter()-t0:.1f} s", flush=True)
