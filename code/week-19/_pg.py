"""Shared helpers for Week 19 policy-gradient scripts (CartPole-v1, gymnasium)."""
import numpy as np, torch, torch.nn as nn, gymnasium as gym
torch.set_num_threads(1)
def mlp(o, h=64): return nn.Sequential(nn.Linear(4, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))
def evaluate(pi, n=20, greedy=True):
    env = gym.make("CartPole-v1"); out = []
    for k in range(n):
        s, _ = env.reset(seed=10 ** 6 + k); d = tr = False; L = 0
        while not (d or tr):
            with torch.no_grad(): lg = pi(torch.tensor(s))
            a = int(lg.argmax()) if greedy else int(torch.distributions.Categorical(logits=lg).sample())
            s, r, d, tr, _ = env.step(a); L += 1
        out.append(L)
    return float(np.mean(out))
def discounted(r, g):
    G = np.zeros(len(r)); acc = 0.0
    for t in reversed(range(len(r))): acc = r[t] + g * acc; G[t] = acc
    return G
