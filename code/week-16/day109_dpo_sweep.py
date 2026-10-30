"""Day 109: beta / learning-rate sweep for DPO (runs day108_dpo.py with different settings, 60 steps each), plus forgetting (Shakespeare ppl)."""
import os, subprocess, sys
for beta, lr in ((0.1, 1e-4), (0.5, 1e-4)):
    env = dict(os.environ, BETA=str(beta), LR=str(lr), STEPS="60", PPL="1", TAG=f"_b{beta}_lr{lr}")
    print(f"===== beta {beta}, lr {lr}, 60 steps =====", flush=True)
    out = subprocess.run([sys.executable, "day108_dpo.py"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True).stdout
    print("\n".join(l for l in out.splitlines() if ("Loading weights" not in l and "indices" not in l) and not l.startswith("step") or l.startswith("step  40")), flush=True)
