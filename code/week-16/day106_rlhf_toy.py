"""Day 106: RLHF building blocks on toys. (1) Bradley-Terry reward model fitted from pairwise preferences.
(2) KL-regularised policy optimisation on a 5-armed bandit: closed form pi* ∝ pi_ref·exp(r/beta) vs policy-gradient (REINFORCE + KL)."""
import torch, math
torch.manual_seed(0)
# (1) hidden true reward r(x) = w·x; preferences sampled with P(a > b) = sigmoid(r(a) - r(b))
D, N = 5, 2000; w_true = torch.tensor([2.0, -1.0, 0.5, 0.0, 1.5]); xa, xb = torch.randn(N, D), torch.randn(N, D)
pref = torch.bernoulli(torch.sigmoid((xa - xb) @ w_true)); w = torch.zeros(D, requires_grad=True); opt = torch.optim.Adam([w], lr=0.05)
for _ in range(500):
    loss = torch.nn.functional.binary_cross_entropy_with_logits((xa - xb) @ w, pref); opt.zero_grad(); loss.backward(); opt.step()
print("Bradley-Terry: true w", w_true.tolist(), "| fitted", [round(v, 2) for v in w.tolist()])
xt, yt = torch.randn(1000, D), torch.randn(1000, D)
print(f"held-out: fitted reward ranks pairs like the true reward in {((((xt - yt) @ w) > 0) == (((xt - yt) @ w_true) > 0)).float().mean():.3f} of pairs; train loss {loss.item():.3f}")
with torch.no_grad():
    bce = torch.nn.functional.binary_cross_entropy_with_logits; l0 = bce(xa @ w - xb @ w, pref); l10 = bce((xa @ w + 10) - (xb @ w + 10), pref)
print(f"reward is identifiable only up to a constant: adding +10 to every reward changes the BT loss by {abs(l10 - l0).item():.1e}")
# (2) KL-regularised bandit
r = torch.tensor([1.0, 0.8, 0.5, 0.2, 0.0]); ref = torch.tensor([0.05, 0.1, 0.2, 0.3, 0.35])
for beta in (0.05, 0.2, 1.0, 5.0):
    pi = ref * torch.exp(r / beta); pi /= pi.sum(); kl = (pi * (pi / ref).log()).sum()
    print(f"beta {beta:4.2f}: optimal pi {[round(v, 3) for v in pi.tolist()]}, expected reward {(pi*r).sum():.3f}, KL to ref {kl:.3f}")
beta = 0.2; logits = ref.log().clone().requires_grad_(True); opt = torch.optim.Adam([logits], lr=0.05); g = torch.Generator().manual_seed(0)
for step in range(3000):
    p = logits.softmax(0); a = torch.multinomial(p, 64, replacement=True, generator=g)
    adv = r[a] - beta * (p[a] / ref[a]).log().detach(); adv = adv - adv.mean()
    loss = -(adv * p[a].log()).mean(); opt.zero_grad(); loss.backward(); opt.step()
pi_star = ref * torch.exp(r / beta); pi_star /= pi_star.sum()
print(f"REINFORCE with KL penalty (beta 0.2, 3000 steps x 64 samples): pi {[round(v, 3) for v in logits.softmax(0).tolist()]} vs closed form {[round(v, 3) for v in pi_star.tolist()]}")
