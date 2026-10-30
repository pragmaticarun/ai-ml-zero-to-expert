"""Day 96: byte-level BPE from scratch: train, encode, decode, compression; compare with a pretrained tokenizer; Rust encoder speed."""
import sys, os, time, json; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common import load_shakespeare, outdir
from _bpe import train_bpe, encode, decode, vocab_bytes, PAT
text = load_shakespeare(); n = int(0.9 * len(text)); tr, va = text[:n], text[n:]
t = time.perf_counter(); merges = train_bpe(tr, 1024); print(f"trained BPE to vocab 1024 ({len(merges)} merges) on {len(tr):,} chars in {time.perf_counter()-t:.1f} s")
v = vocab_bytes(merges); print("first 10 merges:", [v[i].decode() for i in range(256, 266)]); print("last 5 merges:", [v[i].decode() for i in range(1019, 1024)])
for name, s in (("train", tr), ("val", va)):
    ids = encode(s, merges); print(f"{name}: {len(s.encode()):,} bytes -> {len(ids):,} tokens, {len(s.encode())/len(ids):.2f} bytes/token")
for s in ("To be, or not to be", "Hello, world! 12345", "naïve café 東京", "   indented"):
    ids = encode(s, merges); print(f"{s!r:24s} -> {len(ids):2d} tokens {[v[i].decode('utf-8', 'replace') for i in ids]} | roundtrip ok: {decode(ids, merges) == s}")
for vs in (300, 512):
    m2 = train_bpe(tr, vs); print(f"vocab {vs}: val {len(va.encode())/len(encode(va, m2)):.2f} bytes/token")
json.dump([[a, b, i] for (a, b), i in merges.items()], open(os.path.join(outdir(__file__), "bpe1024.json"), "w"))
try:
    os.environ.setdefault("HF_HOME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "cache", "hf"))
    from transformers import AutoTokenizer; tk = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    ids = tk(va)["input_ids"]; print(f"SmolLM2-135M tokenizer (vocab {len(tk):,}): val {len(va.encode())/len(ids):.2f} bytes/token")
    print("  'Hello, world! 12345' ->", tk.convert_ids_to_tokens(tk("Hello, world! 12345")["input_ids"]))
except Exception as e: print("pretrained tokenizer skipped:", type(e).__name__, e)
words = [w.encode() for w in PAT.findall(va)]
from _bpe import encode_word
t = time.perf_counter(); py_ids = [i for w in words for i in encode_word(w, merges)]; tp = time.perf_counter() - t
print(f"Python encoder: {len(va):,} chars in {tp*1000:.0f} ms")
try:
    import rust_bpe; enc = rust_bpe.Encoder([((a, b), i) for (a, b), i in merges.items()])
    for par in (False, True):
        t = time.perf_counter(); r = enc.encode_words(words, par); tr_ = time.perf_counter() - t
        print(f"Rust encoder ({'rayon' if par else 'serial'}): {tr_*1000:.1f} ms, identical to Python: {r == py_ids}, speed-up {tp/tr_:.0f}x")
except ImportError: print("rust_bpe not built: cd code/week-14/rust_bpe && maturin develop --release")
