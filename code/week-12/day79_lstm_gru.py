"""Day 79: RNN vs LSTM vs GRU on a long-memory task (classify the first token after T distractor steps); parameter counts and gate shapes."""
import sys, os, time; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import torch, torch.nn as nn
torch.manual_seed(0)
for cls in (nn.RNN, nn.GRU, nn.LSTM):
    m = cls(10, 32, batch_first=True); print(f"{cls.__name__:4s}(10->32) params {sum(p.numel() for p in m.parameters()):,} | weight_ih shape {tuple(m.weight_ih_l0.shape)}")
def batch(T, n, g, k=4):
    first = torch.randint(0, k, (n,), generator=g); seq = torch.randint(k, 10, (n, T), generator=g); seq[:, 0] = first
    return nn.functional.one_hot(seq, 10).float(), first
class Net(nn.Module):
    def __init__(s, cls, fbias=False):
        super().__init__(); s.r = cls(10, 32, batch_first=True); s.o = nn.Linear(32, 4)
        if fbias:   # forget-gate bias = 1 (gate order i, f, g, o): start out remembering
            with torch.no_grad(): s.r.bias_ih_l0[32:64].fill_(1.0); s.r.bias_hh_l0[32:64].fill_(0.0)
    def forward(s, x): out, _ = s.r(x); return s.o(out[:, -1])
for T in (5, 20, 50):
    for cls, fb in ((nn.RNN, 0), (nn.GRU, 0), (nn.LSTM, 0), (nn.LSTM, 1)):
        torch.manual_seed(0); m = Net(cls, fb); o = torch.optim.Adam(m.parameters(), 3e-3); g = torch.Generator().manual_seed(0); t = time.perf_counter()
        for step in range(1500):
            x, y = batch(T, 64, g); l = nn.functional.cross_entropy(m(x), y); o.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); o.step()
        x, y = batch(T, 2000, torch.Generator().manual_seed(99))
        with torch.no_grad(): acc = (m(x).argmax(1) == y).float().mean().item()
        print(f"T={T:2d} {cls.__name__ + ('+fb1' if fb else ''):8s}: accuracy recalling the first token {acc:.4f} (chance 0.25) [{time.perf_counter()-t:.0f} s]")
# LSTM cell by hand matches nn.LSTMCell
cell = nn.LSTMCell(3, 4); x = torch.randn(2, 3); h = torch.zeros(2, 4); c = torch.zeros(2, 4)
z = x @ cell.weight_ih.T + cell.bias_ih + h @ cell.weight_hh.T + cell.bias_hh; i, f, gg, o_ = z.chunk(4, 1)
c2 = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(gg); h2 = torch.sigmoid(o_) * torch.tanh(c2); h_ref, c_ref = cell(x, (h, c))
print(f"hand-written LSTM step vs nn.LSTMCell: max diff h {(h2 - h_ref).abs().max().item():.1e}, c {(c2 - c_ref).abs().max().item():.1e}")
