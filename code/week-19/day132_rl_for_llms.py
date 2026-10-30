"""Day 132: PPO / GRPO / DPO on a one-prompt 'LLM' with 5 possible answers (a KL-regularised bandit, as on Day 106).
Rewards r = [1.0, 0.8, 0.5, 0.2, 0.0]; reference policy pi_ref = [0.05, 0.1, 0.2, 0.3, 0.35]; beta = 0.2.
Target: closed-form optimum pi*(y) ∝ pi_ref(y) exp(r(y)/beta). Policy = softmax(logits), logits initialised at log pi_ref, Adam lr 0.05,
400 iterations, 3 seeds. Methods:
  REINFORCE+KL : 64 samples/iter, reward r - beta*log(pi/pi_ref) (KL in reward), batch-mean baseline, 1 gradient step.
  PPO-clip+KL  : same samples & shaped reward, advantage = shaped reward - batch mean, 4 epochs on the clipped ratio (eps 0.2).
  GRPO         : group of G=8 samples x 8 groups; A = (r - mean_group)/std_group; clipped ratio (eps 0.2), 4 epochs;
                 KL added to the loss with the k3 estimator pi_ref/pi - log(pi_ref/pi) - 1 (as in the DeepSeekMath paper, arXiv 2402.03300).
  DPO          : 64 pairs/iter sampled from pi_ref, labelled by Bradley-Terry P(a>b) = sigmoid(r_a - r_b); to match beta,
                 the BT reward must be r itself, so labels use sigmoid(r_a - r_b); DPO loss -log sigmoid(beta*(log-ratio_w - log-ratio_l)).
Reports total-variation distance to pi*, expected reward, and KL(pi||pi_ref)."""
import numpy as np, torch
torch.set_num_threads(1)
r = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]); ref = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.35]); beta = 0.2
star = ref * torch.exp(r / beta); star = star / star.sum()
def stats(p): return 0.5 * (p - star).abs().sum().item(), (p * r).sum().item(), (p * (p / ref).log()).sum().item()
def run(method, seed, iters=400, B=64, eps=0.2):
    g = torch.Generator().manual_seed(seed); th = ref.log().clone().requires_grad_(True); opt = torch.optim.Adam([th], lr=0.05)
    for it in range(iters):
        with torch.no_grad(): p_old = torch.softmax(th, 0); lp_old = p_old.log()
        if method == "DPO":
            a = torch.multinomial(ref, B, True, generator=g); b = torch.multinomial(ref, B, True, generator=g)
            a_wins = torch.rand(B, generator=g) < torch.sigmoid(r[a] - r[b]); w = torch.where(a_wins, a, b); l = torch.where(a_wins, b, a)
            lp = torch.log_softmax(th, 0); lr_ = lp - ref.log(); loss = -torch.nn.functional.logsigmoid(beta * (lr_[w] - lr_[l])).mean()
            opt.zero_grad(); loss.backward(); opt.step(); continue
        y = torch.multinomial(p_old, B, True, generator=g)
        if method == "GRPO":
            rg = r[y].view(8, 8); A = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-6)).view(-1)
        else:
            shaped = r[y] - beta * (lp_old[y] - ref.log()[y]); A = shaped - shaped.mean()
        for ep in range(1 if method == "REINFORCE+KL" else 4):
            lp = torch.log_softmax(th, 0)
            if method == "REINFORCE+KL": loss = -(lp[y] * A).mean()
            else:
                ratio = torch.exp(lp[y] - lp_old[y]); loss = -torch.min(ratio * A, ratio.clamp(1 - eps, 1 + eps) * A).mean()
                if method == "GRPO":
                    q = ref.log()[y] - lp[y]; loss = loss + beta * (q.exp() - q - 1).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    return torch.softmax(th, 0).detach()
tv, er, kl = stats(star); print(f"closed-form pi* (beta {beta}): {[round(x, 3) for x in star.tolist()]}; reward {er:.3f}; KL to ref {kl:.3f}")
tv, er, kl = stats(ref); print(f"pi_ref: reward {er:.3f}; TV to pi* {tv:.3f}")
for m in ("REINFORCE+KL", "PPO-clip+KL", "GRPO", "DPO"):
    for seed in range(3):
        p = run(m, seed); tv, er, kl = stats(p)
        print(f"{m}, seed {seed}: pi = {[round(x, 3) for x in p.tolist()]}; TV to pi* {tv:.3f}; reward {er:.3f}; KL to ref {kl:.3f}", flush=True)
