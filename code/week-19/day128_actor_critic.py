"""Day 128: advantage actor-critic (A2C-style, one env) on CartPole-v1. Rollouts of n steps, bootstrapped n-step returns,
advantage = return - V(s), loss = policy + 0.5*value - 0.01*entropy, shared Adam 1e-3 on separate actor/critic MLPs.
50k env steps per seed, 3 seeds; n in {1 (one-step TD actor-critic), 32}. Bootstraps through time-limit truncation, not termination."""
import time, numpy as np, torch, gymnasium as gym
from _pg import mlp, evaluate
def train(seed, n, steps=50000, g=0.99):
    torch.manual_seed(seed); env = gym.make("CartPole-v1"); pi, V = mlp(2), mlp(1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(V.parameters()), 1e-3); s, _ = env.reset(seed=seed); L = 0; lens = []; t = 0; evs = []
    while t < steps:
        S, A, R, D = [], [], [], []
        for _ in range(n):
            with torch.no_grad(): a = int(torch.distributions.Categorical(logits=pi(torch.tensor(s))).sample())
            s2, r, d, tr, _ = env.step(a); S.append(s); A.append(a); R.append(r); D.append(d); L += 1; t += 1
            if (t % 10000) == 0: evs.append(evaluate(pi))
            if d or tr:
                lens.append(L); L = 0
                if tr and not d:  # truncation: bootstrap from s2
                    with torch.no_grad(): R[-1] += g * float(V(torch.tensor(s2)))
                D[-1] = True; s, _ = env.reset()
                break
            s = s2
        with torch.no_grad(): nxt = 0.0 if D[-1] else float(V(torch.tensor(s)))
        ret = []; acc = nxt
        for r in reversed(R): acc = r + g * acc; ret.append(acc)  # a rollout stops at episode end, so only its last step can be terminal (handled by nxt)
        ret = torch.tensor(ret[::-1], dtype=torch.float32); St = torch.tensor(np.array(S)); v = V(St).squeeze(1)
        dist = torch.distributions.Categorical(logits=pi(St)); adv = (ret - v).detach()
        loss = -(dist.log_prob(torch.tensor(A)) * adv).mean() + 0.5 * ((ret - v) ** 2).mean() - 0.01 * dist.entropy().mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return np.array(lens), evs
if __name__ == "__main__":
    for n in (1, 32):
        for seed in range(3):
            t0 = time.perf_counter(); lens, evs = train(seed, n)
            print(f"A2C n={n}, seed {seed}: {len(lens)} episodes, mean length last 20 {lens[-20:].mean():.1f}; greedy eval at 10k..50k steps {[round(e,1) for e in evs]}; {time.perf_counter()-t0:.0f} s", flush=True)
