"""Day 125: DQN on gymnasium CartPole-v1, CPU scale. MLP 4-64-64-2, replay buffer 20k, batch 64, Adam 1e-3, Huber loss,
target network synced every 500 steps, epsilon 1->0.05 over 10k steps, 30k env steps per seed. Ablation: no target network
(bootstrap from the online net). Greedy evaluation: 20 episodes, 500 = cap."""
import time, numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)
def net(): return nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))
def evaluate(q, env, n=20):
    out = []
    for k in range(n):
        s, _ = env.reset(seed=10 ** 6 + k); d = tr = False; L = 0
        while not (d or tr):
            with torch.no_grad(): a = int(q(torch.tensor(s)).argmax())
            s, r, d, tr, _ = env.step(a); L += 1
        out.append(L)
    return np.array(out)
def run(seed, target=True, steps=30000, gamma=0.99):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); env = gym.make("CartPole-v1")
    q, qt = net(), net(); qt.load_state_dict(q.state_dict()); opt = torch.optim.Adam(q.parameters(), 1e-3)
    N = 20000; S = np.zeros((N, 4), np.float32); A = np.zeros(N, np.int64); R = np.zeros(N, np.float32); S2 = np.zeros((N, 4), np.float32); D = np.zeros(N, np.float32)
    s, _ = env.reset(seed=seed); L = 0; lens = []; best = (-1, None); evals = []
    for t in range(steps):
        eps = max(0.05, 1 - t / 10000)
        if rng.random() < eps: a = int(rng.integers(2))
        else:
            with torch.no_grad(): a = int(q(torch.tensor(s, dtype=torch.float32)).argmax())
        s2, r, d, tr, _ = env.step(a); i = t % N; S[i], A[i], R[i], S2[i], D[i] = s, a, r, s2, float(d); s = s2; L += 1
        if d or tr: lens.append(L); L = 0; s, _ = env.reset()
        if t >= 1000:
            idx = rng.integers(0, min(t + 1, N), 64); bs, bs2 = torch.from_numpy(S[idx]), torch.from_numpy(S2[idx])
            with torch.no_grad(): y = torch.from_numpy(R[idx]) + gamma * (1 - torch.from_numpy(D[idx])) * (qt if target else q)(bs2).max(1).values
            loss = nn.functional.smooth_l1_loss(q(bs).gather(1, torch.from_numpy(A[idx])[:, None]).squeeze(1), y)
            opt.zero_grad(); loss.backward(); opt.step()
        if target and t % 500 == 0: qt.load_state_dict(q.state_dict())
        if (t + 1) % 10000 == 0: evals.append(evaluate(q, gym.make("CartPole-v1")).mean())
    return np.array(lens), evals
for target in (True, False):
    for seed in range(3):
        t0 = time.perf_counter(); lens, evals = run(seed, target)
        print(f"DQN {'with' if target else 'WITHOUT'} target net, seed {seed}: {len(lens)} training episodes, mean length last 20 {lens[-20:].mean():.1f}; "
              f"greedy eval mean at 10k/20k/30k steps {[round(float(e), 1) for e in evals]}; {time.perf_counter()-t0:.0f} s", flush=True)
