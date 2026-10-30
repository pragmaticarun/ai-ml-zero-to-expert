"""Tabular RL helpers written from scratch: a gridworld MDP with explicit transition tables P[s][a] = [(p, s', r, done)],
a FrozenLake-style slippery lake, a cliff-walking world, and an episode sampler. Same table format as gymnasium's toy-text envs."""
import numpy as np
A = [(0, -1), (1, 0), (0, 1), (-1, 0)]  # left, down, right, up (gymnasium FrozenLake order)
def _move(r, c, a, H, W):
    dr, dc = A[a]; return min(max(r + dr, 0), H - 1), min(max(c + dc, 0), W - 1)
def gridworld(H=4, W=4, step=-1.0, slip=0.0):
    """Terminal corners (0,0) and (H-1,W-1); reward `step` per move. slip: prob. of a random perpendicular move."""
    nS = H * W; P = {}
    for s in range(nS):
        r, c = divmod(s, W); term = s in (0, nS - 1); P[s] = {}
        for a in range(4):
            if term: P[s][a] = [(1.0, s, 0.0, True)]; continue
            outs = [(1 - slip, a)] + ([(slip / 2, (a + 1) % 4), (slip / 2, (a + 3) % 4)] if slip else [])
            P[s][a] = [(p, (lambda rr, cc: rr * W + cc)(*_move(r, c, b, H, W)), step, False) for p, b in outs]
            P[s][a] = [(p, s2, rw, s2 in (0, nS - 1)) for p, s2, rw, _ in P[s][a]]
    return P, nS, 4
LAKE4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
LAKE8 = ["SFFFFFFF", "FFFFFFFF", "FFFHFFFF", "FFFFFHFF", "FFFHFFFF", "FHHFFFHF", "FHFFHFHF", "FFFHFFFG"]
def lake(desc=LAKE4, slippery=True):
    """FrozenLake rules: slippery moves go in the intended or either perpendicular direction with prob. 1/3 each; reward 1 at G; H and G end."""
    H, W = len(desc), len(desc[0]); nS = H * W; P = {}
    for s in range(nS):
        r, c = divmod(s, W); P[s] = {}
        for a in range(4):
            if desc[r][c] in "HG": P[s][a] = [(1.0, s, 0.0, True)]; continue
            dirs = [(a + 3) % 4, a, (a + 1) % 4] if slippery else [a]; P[s][a] = []
            for b in dirs:
                rr, cc = _move(r, c, b, H, W); ch = desc[rr][cc]
                P[s][a].append((1 / len(dirs), rr * W + cc, float(ch == "G"), ch in "HG"))
    return P, nS, 4
def cliff(H=4, W=12):
    """Cliff walking: start bottom-left, goal bottom-right, cells between are cliff (reward -100, back to start); -1 per step."""
    nS = H * W; start = (H - 1) * W; goal = nS - 1; P = {}
    for s in range(nS):
        r, c = divmod(s, W); P[s] = {}
        for a in range(4):
            if s == goal: P[s][a] = [(1.0, s, 0.0, True)]; continue
            rr, cc = _move(r, c, a, H, W); s2 = rr * W + cc
            P[s][a] = [(1.0, start, -100.0, False)] if rr == H - 1 and 0 < cc < W - 1 else [(1.0, s2, -1.0, s2 == goal)]
    return P, nS, 4, start
def step(P, s, a, rng):
    outs = P[s][a]; i = rng.choice(len(outs), p=[o[0] for o in outs]) if len(outs) > 1 else 0; _, s2, r, d = outs[i]; return s2, r, d
def model(P, nS, nA):
    """Dense arrays: T[s,a,s'] and expected reward R[s,a]; terminal flags folded in (no value after done)."""
    T = np.zeros((nS, nA, nS)); R = np.zeros((nS, nA))
    for s in range(nS):
        for a in range(nA):
            for p, s2, r, d in P[s][a]:
                R[s, a] += p * r
                if not d: T[s, a, s2] += p
    return T, R
