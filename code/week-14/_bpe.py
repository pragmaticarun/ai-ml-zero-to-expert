"""Byte-level BPE from scratch (Day 96), shared by Days 97-98. Pre-tokenises on a simple regex, so merges never cross word boundaries."""
import re, collections
PAT = re.compile(r" ?[A-Za-z]+| ?[0-9]+| ?[^A-Za-z0-9\s]+|\s+")
def train_bpe(text, vocab_size):
    words = collections.Counter(PAT.findall(text)); seqs = {w: tuple(w.encode()) for w in words}; merges = {}; nxt = 256
    while nxt < vocab_size:
        pairs = collections.Counter()
        for w, c in words.items():
            s = seqs[w]
            for a, b in zip(s, s[1:]): pairs[a, b] += c
        if not pairs: break
        best = max(pairs, key=pairs.get); merges[best] = nxt
        for w in words:
            s = seqs[w]
            if len(s) < 2: continue
            out, i = [], 0
            while i < len(s):
                if i + 1 < len(s) and (s[i], s[i+1]) == best: out.append(nxt); i += 2
                else: out.append(s[i]); i += 1
            seqs[w] = tuple(out)
        nxt += 1
    return merges
def vocab_bytes(merges):
    v = {i: bytes([i]) for i in range(256)}
    for (a, b), i in merges.items(): v[i] = v[a] + v[b]
    return v
def encode_word(bs, merges):
    s = list(bs)
    while len(s) > 1:
        cands = [(merges.get((a, b), 1 << 30), j) for j, (a, b) in enumerate(zip(s, s[1:]))]; r, j = min(cands)
        if r == 1 << 30: break
        s[j:j+2] = [r]
    return s
def encode(text, merges):
    out = []
    for w in PAT.findall(text): out += encode_word(w.encode(), merges)
    return out
def decode(ids, merges): v = vocab_bytes(merges); return b"".join(v[i] for i in ids).decode("utf-8", errors="replace")
