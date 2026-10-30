"""Day 129: PPO (clipped surrogate) from scratch on CartPole-v1. One env, rollout 1024 steps, GAE(gamma 0.99, lambda 0.95),
10 epochs x minibatch 64, Adam 3e-4, value coef 0.5, entropy 0.01, advantage normalisation; 50k env steps, 3 seeds.
Ablation: clip eps 0.2 vs 'no clip' (eps = 1e9, i.e. plain multi-epoch policy gradient on the same data).
Logs approx KL(old||new) and fraction of clipped ratios after each update."""
import time, numpy as np, torch, gymnasium as gym
from _pg import mlp, evaluate
def train(seed, clip, steps=50000, T=1024, g=0.99, lam=0.95):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); env = gym.make("CartPole-v1"); pi, V = mlp(2), mlp(1)
    opt = torch.optim.Adam(list(pi.parameters()) + list(V.parameters()), 3e-4); s, _ = env.reset(seed=seed); L = 0; lens = []; t = 0; evs = []; kls = []; cfs = []
    while t < steps:
        S, A, LP, R, Dn, Vs, Vn = [], [], [], [], [], [], []
        for _ in range(T):
            st = torch.tensor(s)
            with torch.no_grad(): dist = torch.distributions.Categorical(logits=pi(st)); a = dist.sample(); v = float(V(st))
            s2, r, d, tr, _ = env.step(int(a)); L += 1; t += 1
            with torch.no_grad(): v2 = 0.0 if d else float(V(torch.tensor(s2)))   # bootstrap through truncation, not termination
            S.append(s); A.append(int(a)); LP.append(float(dist.log_prob(a))); R.append(r); Dn.append(d or tr); Vs.append(v); Vn.append(v2)
            s = s2
            if d or tr: lens.append(L); L = 0; s, _ = env.reset()
        adv = np.zeros(T); acc = 0.0
        for i in reversed(range(T)):
            delta = R[i] + g * Vn[i] - Vs[i]; acc = delta + g * lam * (0 if Dn[i] else acc); adv[i] = acc
        ret = torch.tensor(adv + np.array(Vs), dtype=torch.float32); advt = torch.tensor(adv, dtype=torch.float32); advt = (advt - advt.mean()) / (advt.std() + 1e-8)
        St, At, LPt = torch.tensor(np.array(S)), torch.tensor(A), torch.tensor(LP)
        for _ in range(10):
            for idx in np.array_split(rng.permutation(T), T // 64):
                dist = torch.distributions.Categorical(logits=pi(St[idx])); ratio = torch.exp(dist.log_prob(At[idx]) - LPt[idx])
                pl = -torch.min(ratio * advt[idx], ratio.clamp(1 - clip, 1 + clip) * advt[idx]).mean()
                loss = pl + 0.5 * ((V(St[idx]).squeeze(1) - ret[idx]) ** 2).mean() - 0.01 * dist.entropy().mean()
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(pi.parameters()) + list(V.parameters()), 0.5); opt.step()
        with torch.no_grad():
            lr = torch.distributions.Categorical(logits=pi(St)).log_prob(At) - LPt; kls.append(float((-lr).mean())); cfs.append(float(((lr.exp() - 1).abs() > 0.2).float().mean()))
        evs.append(evaluate(pi))
    return np.array(lens), evs, kls, cfs
if __name__ == "__main__":
    for clip in (0.2, 1e9):
        for seed in range(3):
            t0 = time.perf_counter(); lens, evs, kls, cfs = train(seed, clip)
            print(f"PPO clip={'0.2' if clip < 1 else 'none'}, seed {seed}: {len(lens)} episodes; greedy eval after each of {len(evs)} updates {[round(e) for e in evs]}; "
                  f"approx KL per update max {max(kls):.3f} mean {np.mean(kls):.3f}; frac |ratio-1|>0.2 mean {np.mean(cfs):.2f}; {time.perf_counter()-t0:.0f} s", flush=True)
