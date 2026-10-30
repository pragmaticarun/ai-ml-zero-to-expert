"""Day 127: REINFORCE on CartPole-v1. Policy MLP 4-64-64-2 (tanh), Adam 1e-3 (policy), gamma 0.99, 300 episodes, 3 seeds.
Variants: (a) no baseline, raw returns G_t; (b) learned value baseline V(s) (separate MLP, Adam 1e-3), advantage G_t - V(s_t).
Also measures gradient variance: std of the policy-gradient estimate over 20 single-episode estimates at initialisation."""
import time, numpy as np, torch, gymnasium as gym
from _pg import mlp, evaluate, discounted
def episode(env, pi, seed=None):
    s, _ = env.reset(seed=seed); S, A, R = [], [], []; d = tr = False
    while not (d or tr):
        with torch.no_grad(): a = int(torch.distributions.Categorical(logits=pi(torch.tensor(s))).sample())
        S.append(s); A.append(a); s, r, d, tr, _ = env.step(a); R.append(r)
    return torch.tensor(np.array(S)), torch.tensor(A), np.array(R)
def grad_variance(seed=0, n=20):
    torch.manual_seed(seed); env = gym.make("CartPole-v1"); pi, V = mlp(2), mlp(1); gs = {"none": [], "constant baseline (mean G)": []}
    eps = [episode(env, pi, seed=1000 + k) for k in range(n)]; meanG = np.mean([discounted(R, 0.99).mean() for _, _, R in eps])
    for S, A, R in eps:
        G = torch.tensor(discounted(R, 0.99), dtype=torch.float32)
        for name, adv in (("none", G), ("constant baseline (mean G)", G - meanG)):
            pi.zero_grad(); lp = torch.distributions.Categorical(logits=pi(S)).log_prob(A); (-(lp * adv).sum()).backward()
            gs[name].append(torch.cat([p.grad.flatten() for p in pi.parameters()]).clone())
    for k, v in gs.items(): v = torch.stack(v); print(f"grad estimate at init, baseline={k}: mean per-parameter std over {n} episodes {v.std(0).mean():.3f}")
def train(seed, baseline, n_ep=300, g=0.99, max_steps=None):
    torch.manual_seed(seed); np.random.seed(seed); env = gym.make("CartPole-v1"); pi, V = mlp(2), mlp(1)
    op, ov = torch.optim.Adam(pi.parameters(), 1e-3), torch.optim.Adam(V.parameters(), 1e-3); lens = []; steps = 0
    for ep in range(n_ep):
        if max_steps and steps >= max_steps: break
        S, A, R = episode(env, pi, seed=seed * 10 ** 5 + ep); G = torch.tensor(discounted(R, g), dtype=torch.float32); lens.append(len(R)); steps += len(R)
        if baseline:
            v = V(S).squeeze(1); adv = (G - v).detach(); ov.zero_grad(); ((v - G) ** 2).mean().backward(); ov.step()
        else: adv = G
        op.zero_grad(); lp = torch.distributions.Categorical(logits=pi(S)).log_prob(A); (-(lp * adv).sum() / len(R)).backward(); op.step()
    return np.array(lens), steps, evaluate(pi)
if __name__ == "__main__":
    grad_variance()
    for baseline in (False, True):
        for seed in range(3):
            t = time.perf_counter(); lens, steps, ev = train(seed, baseline)
            print(f"REINFORCE {'+ value baseline' if baseline else 'no baseline'}, seed {seed}: mean ep length 1-50 {lens[:50].mean():.1f}, 251-300 {lens[250:].mean():.1f}; "
                  f"{steps} env steps; greedy eval (20 eps, cap 500) {ev:.1f}; {time.perf_counter()-t:.0f} s", flush=True)
