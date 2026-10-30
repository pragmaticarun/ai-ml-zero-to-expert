"""Day 133 PROJECT W19: REINFORCE (+value baseline) vs A2C (n=32) vs PPO-clip on CartPole-v1 at an equal budget of ~50k env steps,
5 fresh seeds (10-14) not used on Days 127-129. Final policy scored by greedy AND sampled evaluation (20 episodes each, cap 500).
Reuses the train() functions from day127-129 unchanged."""
import time, numpy as np, torch
from _pg import evaluate
import day127_reinforce as d127, day128_actor_critic as d128, day129_ppo as d129
res = {}
def keep(fn):  # capture the final policy network by wrapping evaluate inside each module
    box = {}
    def ev(pi, n=20, greedy=True): box["pi"] = pi; return evaluate(pi, n, greedy)
    return box, ev
for name in ("REINFORCE+baseline", "A2C n=32", "PPO clip 0.2"):
    for seed in range(10, 15):
        t0 = time.perf_counter()
        if name.startswith("REINFORCE"):
            box, ev = keep(None); d127.evaluate = ev; d127.train(seed, True, n_ep=10 ** 6, max_steps=50000)
        elif name.startswith("A2C"):
            box, ev = keep(None); d128.evaluate = ev; d128.train(seed, 32)
        else:
            box, ev = keep(None); d129.evaluate = ev; d129.train(seed, 0.2, steps=50000)
        pi = box["pi"]; g, s = evaluate(pi, 20, True), evaluate(pi, 20, False); dt = time.perf_counter() - t0
        res.setdefault(name, []).append((g, s, dt)); print(f"{name}, seed {seed}: final greedy eval {g:.1f}, sampled eval {s:.1f}; {dt:.0f} s", flush=True)
for name, v in res.items():
    v = np.array(v); print(f"SUMMARY {name}: greedy mean {v[:,0].mean():.1f} (min {v[:,0].min():.1f}, max {v[:,0].max():.1f}); "
                           f"sampled mean {v[:,1].mean():.1f}; seeds at 500 greedy {int((v[:,0] >= 500).sum())}/5; mean time {v[:,2].mean():.0f} s/seed")
