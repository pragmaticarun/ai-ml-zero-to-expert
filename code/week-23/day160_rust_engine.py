"""Day 160: a tiny MLP inference engine in Rust (code/week-23/tiny_infer, PyO3, built with RUSTFLAGS='-C target-cpu=native'),
compared against PyTorch and ONNX Runtime on numerics and latency. (1) fp32 path vs torch; (2) int8 path vs a NumPy re-implementation
of the SAME algorithm (should match almost exactly) and vs fp32; (3) latency at batch 1 and 256, serial vs rayon-parallel."""
import os, numpy as np, torch, onnxruntime as ort, tiny_infer as TI, _dep as D
Xtr, Xte, ytr, yte = D.data(); m = D.base_model(); lin = [l for l in m if isinstance(l, torch.nn.Linear)]
ws = [l.weight.detach().numpy().astype(np.float32).copy() for l in lin]; bs = [l.bias.detach().numpy().astype(np.float32).copy() for l in lin]
ef, eq = TI.MlpF32(ws, bs), TI.MlpInt8(ws, bs)
with torch.no_grad(): ref = m(torch.from_numpy(Xte)).numpy()
of = ef.forward(Xte); oq = eq.forward(Xte); oqp = eq.forward(Xte, parallel=True)
print(f"(1) Rust fp32: max|diff| vs torch {np.abs(of-ref).max():.2e}; acc {(of.argmax(1)==yte).mean():.4f}; weights {ef.nbytes()/1e3:.0f} kB")
def np_int8(X):  # same algorithm as tiny_infer::row_q, in NumPy (float64 dequant math, so tiny rounding differences are possible)
    h = X.astype(np.float32)
    for k, (w, b) in enumerate(zip(ws, bs)):
        sw = np.where(np.abs(w).max(1) > 0, np.abs(w).max(1) / 127, 1).astype(np.float32); wq = np.clip(np.round(w / sw[:, None]), -127, 127).astype(np.int32)
        am = np.abs(h).max(1); sx = np.where(am > 0, am / 127, 1).astype(np.float32); hq = np.clip(np.round(h / sx[:, None]), -127, 127).astype(np.int32)
        h = (hq @ wq.T).astype(np.float32) * sx[:, None] * sw[None, :] + b
        if k < len(ws) - 1: h = np.maximum(h, 0)
    return h
on = np_int8(Xte)
print(f"(2) Rust int8: max|diff| vs NumPy same-algorithm {np.abs(oq-on).max():.2e}; serial==parallel {np.array_equal(oq, oqp)}; max|dlogit| vs fp32 {np.abs(oq-ref).max():.3f}; "
      f"acc {(oq.argmax(1)==yte).mean():.4f}; argmax agree with fp32 {(oq.argmax(1)==ref.argmax(1)).mean():.4f}; weights {eq.nbytes()/1e3:.0f} kB")
so = ort.SessionOptions(); so.intra_op_num_threads = 1; sf = ort.InferenceSession(D.P("w23_mlp.onnx"), so, providers=["CPUExecutionProvider"])
sd = ort.InferenceSession(D.P("w23_dyn.onnx"), so, providers=["CPUExecutionProvider"])
od = sd.run(None, {"x": Xte})[0]; print(f"    Rust int8 vs ORT dynamic int8 logits: max|diff| {np.abs(oq-od).max():.3f} (different activation quantization scheme from ORT's DynamicQuantizeMatMul, so not bit-identical)")
x1, x256 = np.ascontiguousarray(Xte[:1]), np.ascontiguousarray(Xte[:256])
torch.set_num_threads(1)
rows = [("torch eager fp32 (1 thread)", lambda x: m(torch.from_numpy(x))), ("ORT fp32 (1 thread)", lambda x: sf.run(None, {"x": x})), ("ORT dynamic int8 (1 thread)", lambda x: sd.run(None, {"x": x})),
        ("Rust fp32 serial", lambda x: ef.forward(x)), ("Rust int8 serial", lambda x: eq.forward(x)),
        ("Rust fp32 rayon (8 threads)", lambda x: ef.forward(x, parallel=True)), ("Rust int8 rayon (8 threads)", lambda x: eq.forward(x, parallel=True))]
print("(3) median latency, us/call:")
with torch.no_grad():
    for name, f in rows: print(f"    {name:30s} b1 {D.bench(lambda: f(x1)):8.1f}   b256 {D.bench(lambda: f(x256), reps=50):9.1f}")
print("note: the Rust engine does one row at a time (matrix-vector); torch/ORT call a blocked GEMM for b256 (weights reused from cache across rows).")
