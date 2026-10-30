"""Day 146: A/B testing (two-proportion z-test, sample size, the cost of peeking), a canary rollout rule, and a cost/latency budget
computed from Day 144's measured numbers (read from out/day144_onnx_serving.log, not typed in)."""
import re, math, numpy as np
from scipy.stats import norm
def ztest(s1, n1, s2, n2):
    p = (s1 + s2) / (n1 + n2); se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2)); z = (s2 / n2 - s1 / n1) / se; return z, 2 * (1 - norm.cdf(abs(z)))
def n_per_arm(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power); pb = (p1 + p2) / 2
    return math.ceil((za * math.sqrt(2 * pb * (1 - pb)) + zb * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2)
print("== sample size per arm (alpha 0.05, power 0.8), baseline conversion 10%")
for p2 in (0.12, 0.11, 0.105): print(f"  detect 10% -> {p2:.1%}: {n_per_arm(0.10, p2)} users per arm")
rng = np.random.default_rng(0); n = n_per_arm(0.10, 0.11); R = 2000
def sim(p2, peek):
    a = rng.random((R, n)) < 0.10; b = rng.random((R, n)) < p2; hits = 0
    looks = np.linspace(n // 10, n, 10).astype(int) if peek else [n]
    for r in range(R):
        if any(ztest(a[r, :k].sum(), k, b[r, :k].sum(), k)[1] < 0.05 for k in looks): hits += 1
    return hits / R
print(f"== {R} simulated experiments, n={n} per arm")
print(f"  A/A (no true effect), test once at the end : false positive rate {sim(0.10, False):.3f}")
print(f"  A/A, peek 10 times and stop at first p<0.05: false positive rate {sim(0.10, True):.3f}")
print(f"  A/B true 10% -> 11%, test once             : power {sim(0.11, False):.3f}")
print("== canary: 5% of traffic to the new model; roll back if its error rate is significantly worse (one-sided z, p<0.01) after each 1000 canary requests")
def canary(err_old, err_new, steps=10, seed=0):
    g = np.random.default_rng(seed); so = sn = no = nn = 0
    for s in range(1, steps + 1):
        so += (g.random(19000) < err_old).sum(); no += 19000; sn += (g.random(1000) < err_new).sum(); nn += 1000
        z, _ = ztest(so, no, sn, nn)
        if z > norm.ppf(0.99): return f"ROLLED BACK after {nn} canary requests (canary err {sn/nn:.3f} vs {so/no:.3f})"
    return f"PROMOTED after {nn} canary requests (canary err {sn/nn:.3f} vs {so/no:.3f})"
for eo, en in ((0.035, 0.035), (0.035, 0.045), (0.035, 0.07)):
    print(f"  true error old {eo}, new {en}: " + "; ".join(canary(eo, en, seed=s) for s in range(2)))
print("== cost/latency budget from out/day144_onnx_serving.log")
L = open("out/day144_onnx_serving.log").read()
rps1 = float(re.findall(r"backend onnx.*?concurrency  1,  1 rows/request: p50 ([\d.]+) ms, p95 ([\d.]+) ms.*?, (\d+) req/s", L, re.S)[0][2])
m = re.findall(r"backend onnx.*?concurrency 16, 64 rows/request: p50 ([\d.]+) ms, p95 ([\d.]+) ms, p99 ([\d.]+) ms, (\d+) req/s \((\d+) rows/s\)", L, re.S)[0]
rows_s = float(m[4]); p95_16 = float(re.findall(r"backend onnx.*?concurrency 16,  1 rows/request: p50 ([\d.]+) ms, p95 ([\d.]+) ms", L, re.S)[0][1])
peak = 200; daily = 2_000_000; price = 0.05   # ASSUMED workload (200 req/s peak, 2M predictions/day) and ASSUMED $0.05 per core-hour (illustrative, not a quote)
cores = math.ceil(peak / (0.6 * rps1)); print(f"  measured: {rps1:.0f} req/s single-row sequential; p95 {p95_16} ms at concurrency 16; {rows_s:.0f} rows/s with 64-row batches")
print(f"  assumed workload: peak {peak} req/s, {daily:,} predictions/day; assumed price ${price}/core-hour")
print(f"  online, 60% target utilisation: {cores} core(s) -> ${cores*24*price:.2f}/day")
print(f"  batch scoring of {daily:,} rows at {rows_s:.0f} rows/s: {daily/rows_s/60:.1f} core-minutes -> ${daily/rows_s/3600*price:.4f}/day")
