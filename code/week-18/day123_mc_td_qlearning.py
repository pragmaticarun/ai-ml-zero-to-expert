"""Day 123: Monte Carlo vs TD(0) prediction on the 5-state random walk (true values 1/6..5/6), then Q-learning vs SARSA on cliff
walking (epsilon-greedy 0.1). All from scratch with numpy; 10 seeds each."""
import time, numpy as np, gymnasium as gym
from _rl import cliff, step
true = np.arange(1, 6) / 6
def walk(rng):
    s, traj = 2, []
    while True:
        s2 = s + (1 if rng.random() < 0.5 else -1); r = 1.0 if s2 == 5 else 0.0; traj.append((s, r, s2)); s = s2
        if s in (-1, 5): return traj
def run(method, alpha, n=100, seed=0):
    rng = np.random.default_rng(seed); V = np.full(5, 0.5); err = []
    for _ in range(n):
        tr = walk(rng)
        if method == "MC":
            G = tr[-1][1]
            for s, _, _ in tr: V[s] += alpha * (G - V[s])
        else:
            for s, r, s2 in tr: V[s] += alpha * (r + (0 if s2 in (-1, 5) else V[s2]) - V[s])
        err.append(np.sqrt(((V - true) ** 2).mean()))
    return np.array(err)
for m, al in (("MC", 0.01), ("MC", 0.03), ("TD", 0.05), ("TD", 0.1), ("TD", 0.15)):
    E = np.mean([run(m, al, seed=s) for s in range(10)], 0); print(f"random walk {m} alpha={al}: RMS error after 10 / 50 / 100 episodes {E[9]:.3f} / {E[49]:.3f} / {E[99]:.3f} (mean of 10 seeds)")
P, nS, nA, start = cliff(); goal = nS - 1
def control(alg, eps=0.1, alpha=0.5, n=500, seed=0):
    rng = np.random.default_rng(seed); Q = np.zeros((nS, nA)); rets = []
    pick = lambda s: rng.integers(nA) if rng.random() < eps else int(rng.choice(np.flatnonzero(Q[s] == Q[s].max())))
    for _ in range(n):
        s, ret, a = start, 0.0, None; a = pick(s)
        for t in range(1000):
            s2, r, d = step(P, s, a, rng); ret += r; a2 = pick(s2)
            target = r + (0 if d else (Q[s2].max() if alg == "Q-learning" else Q[s2, a2])); Q[s, a] += alpha * (target - Q[s, a]); s, a = s2, a2
            if d: break
        rets.append(ret)
    s, g, path = start, 0.0, 0                      # greedy rollout of the learned Q
    while s != goal and path < 100: s, r, _ = step(P, s, int(Q[s].argmax()), rng); g += r; path += 1
    return np.array(rets), g, path
for alg in ("Q-learning", "SARSA"):
    t = time.perf_counter(); res = [control(alg, seed=s) for s in range(10)]; dt = time.perf_counter() - t
    R = np.mean([r for r, _, _ in res], 0)
    print(f"cliff {alg}: mean online return per episode, episodes 1-50 {R[:50].mean():.1f}, 401-500 {R[400:].mean():.1f}; greedy path return per seed {[int(g) for _, g, _ in res]} "
          f"(a 100-step cap marks a greedy loop); {dt:.1f} s for 10 seeds")
print("gymnasium FrozenLake time limits (max_episode_steps): 4x4", gym.make("FrozenLake-v1").spec.max_episode_steps, "8x8", gym.make("FrozenLake-v1", map_name="8x8").spec.max_episode_steps)
