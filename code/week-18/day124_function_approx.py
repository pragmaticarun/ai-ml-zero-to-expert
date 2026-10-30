"""Day 124: value-function approximation. (1) 1000-state random walk (jumps of up to 100, reward -1 left / +1 right end): semi-gradient
TD(0) and gradient MC with state aggregation (10 groups of 100), compared with the exact values from a linear solve.
(2) Linear semi-gradient Q-learning on gymnasium CartPole-v1 with random Fourier (RBF-like) features, 3 seeds."""
import time, numpy as np, gymnasium as gym
N = 1000
T = np.zeros((N + 2, N + 2)); 
for s in range(1, N + 1):
    for j in list(range(-100, 0)) + list(range(1, 101)): T[s, min(max(s + j, 0), N + 1)] += 1 / 200
A = np.eye(N) - T[1:N + 1, 1:N + 1]; b = -T[1:N + 1, 0] + T[1:N + 1, N + 1]; Vtrue = np.linalg.solve(A, b)
def episode(rng):
    s, tr = 500, []
    while True:
        j = rng.integers(1, 101) * (1 if rng.random() < 0.5 else -1); s2 = s + j
        if s2 < 1: tr.append((s, -1.0, None)); return tr
        if s2 > N: tr.append((s, 1.0, None)); return tr
        tr.append((s, 0.0, s2)); s = s2
g = lambda s: (s - 1) // 100
for meth, al in (("gradient MC", 5e-4), ("semi-gradient TD(0)", 2e-3)):
    rng = np.random.default_rng(0); w = np.zeros(10); t = time.perf_counter()
    for ep in range(5000):
        tr = episode(rng)
        if meth == "gradient MC":
            G = tr[-1][1]
            for s, _, _ in tr: w[g(s)] += al * (G - w[g(s)])
        else:
            for s, r, s2 in tr: w[g(s)] += al * (r + (0 if s2 is None else w[g(s2)]) - w[g(s)])
    V = w[(np.arange(1, N + 1) - 1) // 100]; best = np.array([Vtrue[i * 100:(i + 1) * 100].mean() for i in range(10)])[(np.arange(N)) // 100]
    print(f"1000-state walk, {meth}, aggregation 10 groups, 5000 episodes: RMS error vs true V {np.sqrt(((V - Vtrue) ** 2).mean()):.3f} "
          f"(best possible with this aggregation {np.sqrt(((best - Vtrue) ** 2).mean()):.3f}); weights {np.round(w, 2).tolist()}; {time.perf_counter()-t:.1f} s")
env = gym.make("CartPole-v1"); D = 256
def run(seed, scale, alpha, n_ep=300, gamma=0.99):
    rng = np.random.default_rng(seed); Wf = rng.normal(0, 1, (D, 4)) / np.array(scale); bf = rng.uniform(0, 2 * np.pi, D)
    phi = lambda s: np.cos(Wf @ s + bf) * np.sqrt(2 / D); W = np.zeros((2, D)); lens = []
    for ep in range(n_ep):
        eps = max(0.01, 1 - ep / 150); s, _ = env.reset(seed=seed * 10000 + ep); f = phi(s); done = trunc = False; L = 0
        while not (done or trunc):
            a = rng.integers(2) if rng.random() < eps else int((W @ f).argmax()); s2, r, done, trunc, _ = env.step(a); f2 = phi(s2); L += 1
            target = r + (0 if done else gamma * (W @ f2).max()); W[a] += alpha * (target - W[a] @ f) * f / (f @ f); f = f2
        lens.append(L)
    ev = []
    for k in range(20):
        s, _ = env.reset(seed=10 ** 6 + k); done = trunc = False; L = 0
        while not (done or trunc): s, r, done, trunc, _ = env.step(int((W @ phi(s)).argmax())); L += 1
        ev.append(L)
    return np.array(lens), np.array(ev)
for scale, alpha in (([2.4, 3.0, 0.21, 3.0], 0.01), ([0.5, 0.5, 0.05, 0.5], 0.1)):
  for seed in range(3):
    t = time.perf_counter(); lens, ev = run(seed, scale, alpha)
    print(f"CartPole linear Q (256 Fourier features, feature scale {scale}, alpha {alpha}), seed {seed}: mean episode length, episodes 1-50 {lens[:50].mean():.1f}, 251-300 {lens[250:].mean():.1f}, "
          f"max {lens.max()}; greedy eval over 20 episodes mean {ev.mean():.1f} (min {ev.min()}, max {ev.max()}; 500 = solved cap); {time.perf_counter()-t:.1f} s")
