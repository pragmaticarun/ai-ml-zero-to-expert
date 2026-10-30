"""Day 131: reward-shaping pitfalls on a deterministic 10x10 grid (start (0,0), goal (9,9), +1 on reaching goal, 0 otherwise, gamma 0.99,
episodes capped at 200 steps). Rewards compared:
  base      : +1 at goal only
  naive     : base + 0.1 for every step that reduces Manhattan distance to goal (nothing for increasing it)
  potential : base + F, F = gamma*Phi(s') - Phi(s), Phi(s) = -0.1*distance (potential-based; Phi(goal)=0)
For each: (1) exact optimal greedy policy by value iteration on that reward -> does it reach the goal, in how many steps?
(2) tabular Q-learning (alpha 0.5, eps 0.1, Q0 = 0), 10 seeds: first episode at which the greedy policy reaches the goal in 18 steps
(the shortest path), searched over 1000 episodes."""
import numpy as np
N, G, g = 10, (9, 9), 0.99; MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]
dist = lambda s: abs(G[0] - s[0]) + abs(G[1] - s[1])
def nxt(s, a): return (min(max(s[0] + MOVES[a][0], 0), N - 1), min(max(s[1] + MOVES[a][1], 0), N - 1))
def reward(kind, s, s2):
    r = 1.0 if s2 == G else 0.0
    if kind == "naive": r += 0.1 if dist(s2) < dist(s) else 0.0
    if kind == "potential": r += g * (0 if s2 == G else -0.1 * dist(s2)) - (-0.1 * dist(s))
    return r
S = [(i, j) for i in range(N) for j in range(N)]
def rollout(pi, cap=200):
    s = (0, 0); total = 0.0
    for t in range(cap):
        s = nxt(s, pi[s]);
        if s == G: return t + 1
    return None
def vi(kind):
    V = {s: 0.0 for s in S}
    for _ in range(5000):
        Vn = {s: 0.0 if s == G else max(reward(kind, s, nxt(s, a)) + g * V[nxt(s, a)] for a in range(4)) for s in S}
        if max(abs(Vn[s] - V[s]) for s in S) < 1e-10: break
        V = Vn
    return {s: int(np.argmax([reward(kind, s, nxt(s, a)) + g * V[nxt(s, a)] for a in range(4)])) for s in S}
def qlearn(kind, seed, n_ep=1000):
    rng = np.random.default_rng(seed); Q = {s: np.zeros(4) for s in S}
    for ep in range(n_ep):
        s = (0, 0)
        for t in range(200):
            a = int(rng.integers(4)) if rng.random() < 0.1 else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
            s2 = nxt(s, a); r = reward(kind, s, s2); Q[s][a] += 0.5 * (r + (0 if s2 == G else g * Q[s2].max()) - Q[s][a]); s = s2
            if s == G: break
        if rollout({k: int(np.argmax(v)) for k, v in Q.items()}) == 18: return ep + 1
    return None
for kind in ("base", "naive", "potential"):
    steps = rollout(vi(kind)); hits = [qlearn(kind, sd) for sd in range(10)]; ok = [h for h in hits if h]
    print(f"{kind}: optimal policy (VI on this reward) reaches goal: {'yes, in ' + str(steps) + ' steps' if steps else 'NO (loops for 200 steps)'}; "
          f"Q-learning greedy policy = shortest path by episode {hits} -> {len(ok)}/10 seeds, median {np.median(ok) if ok else 'n/a'}", flush=True)
