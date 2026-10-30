"""Day 144: export the MLP to ONNX, check numerical agreement and speed vs PyTorch, then serve it with FastAPI/uvicorn and load-test over HTTP."""
import os, sys, time, json, subprocess, asyncio, numpy as np, torch, httpx, onnx, onnxruntime as ort, _mlops as M
Xtr, Xte, ytr, yte, _ = M.data(); m = M.train(Xtr, ytr)
torch.save(m.state_dict(), os.path.join(M.OUT, "model.pt"))
json.dump({"weights_sha": M.weights_hash(m), "test_acc": M.acc(m, Xte, yte)}, open(os.path.join(M.OUT, "model_meta.json"), "w"))
p = os.path.join(M.OUT, "model.onnx")
torch.onnx.export(m, (torch.zeros(1, 30),), p, input_names=["x"], output_names=["logits"], dynamic_axes={"x": {0: "batch"}}, dynamo=False)
onnx.checker.check_model(onnx.load(p)); print(f"onnx {onnx.__version__}, onnxruntime {ort.__version__}; exported {os.path.getsize(p)} bytes; checker OK")
so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(p, so)
X = Xte.astype(np.float32)
with torch.no_grad(): zt = m(torch.from_numpy(X)).numpy()
zo = s.run(None, {"x": X})[0]; print(f"max |logit diff| torch vs onnx on {len(X)} rows: {np.abs(zt-zo).max():.2e}; same predictions {(zt.argmax(1)==zo.argmax(1)).all()}")
def bench(f, x, n):
    for _ in range(20): f(x)
    t = time.perf_counter()
    for _ in range(n): f(x)
    return (time.perf_counter() - t) / n * 1e6
tf = lambda x: m(torch.from_numpy(x)); of = lambda x: s.run(None, {"x": x})
with torch.no_grad():
    for bs in (1, 64, 1024):
        x = np.random.default_rng(0).normal(size=(bs, 30)).astype(np.float32) * Xtr.std(0) + Xtr.mean(0); x = x.astype(np.float32)
        a, b = bench(tf, x, 2000 if bs < 1024 else 300), bench(of, x, 2000 if bs < 1024 else 300)
        print(f"  batch {bs:4d}: torch {a:8.1f} us, onnxruntime {b:8.1f} us per call  (speed-up {a/b:.1f}x; 1 thread each)")
print("== HTTP serving (uvicorn, 1 worker, localhost) + load test")
async def load(n, conc, bs):
    lat = []; body = {"rows": Xte[:bs].tolist()}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:8765", timeout=30) as c:
        sem = asyncio.Semaphore(conc)
        async def one():
            async with sem:
                t = time.perf_counter(); r = await c.post("/predict", json=body); r.raise_for_status(); lat.append(time.perf_counter() - t)
        t0 = time.perf_counter(); await asyncio.gather(*[one() for _ in range(n)]); wall = time.perf_counter() - t0
    l = np.array(lat) * 1000; return np.percentile(l, 50), np.percentile(l, 95), np.percentile(l, 99), n / wall
for be in ("torch", "onnx"):
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "serve_app:app", "--port", "8765", "--log-level", "warning"], env={**os.environ, "BACKEND": be}, cwd=os.path.dirname(os.path.abspath(__file__)))
    try:
        for _ in range(100):
            try: h = httpx.get("http://127.0.0.1:8765/health").json(); break
            except Exception: time.sleep(0.1)
        bad = httpx.post("http://127.0.0.1:8765/predict", json={"rows": [[1, 2]]}).json(); bad2 = httpx.post("http://127.0.0.1:8765/predict", json={"rows": "x"}).status_code
        print(f" backend {be}: health {h}; bad shape -> {bad}; bad type -> HTTP {bad2}")
        for n, conc, bs in ((300, 1, 1), (600, 16, 1), (300, 16, 64)):
            p50, p95, p99, rps = asyncio.run(load(n, conc, bs)); print(f"   {n} requests, concurrency {conc:2d}, {bs:2d} rows/request: p50 {p50:.1f} ms, p95 {p95:.1f} ms, p99 {p99:.1f} ms, {rps:.0f} req/s ({rps*bs:.0f} rows/s)", flush=True)
    finally: srv.terminate(); srv.wait()
