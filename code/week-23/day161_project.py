"""Day 161 PROJECT: an on-device-style inference pipeline under an explicit budget (digits classifier, batch 1, 1 thread,
the way a phone app would call it per input). Each variant runs in a FRESH subprocess so its RSS is measured cleanly.
Budget (chosen up front): test acc >= 0.97, p95 latency <= 20 us, model file <= 400 kB, load RSS increase <= 30 MiB."""
import os, sys, json, subprocess, copy, numpy as np, torch, _dep as D
from onnxruntime.quantization import quantize_dynamic, QuantType
BUDGET = dict(acc=0.97, p95_us=20.0, size_kb=400, rss_mib=30)
def build():
    Xtr, Xte, ytr, yte = D.data(); T = D.base_model()
    l1, l2, l3 = [l for l in T if isinstance(l, torch.nn.Linear)]; k = 128  # structured prune to 128 hidden (Day 157), then finetune
    i1 = l1.weight.norm(dim=1).topk(k).indices.sort().values; i2 = l2.weight[:, i1].norm(dim=1).topk(k).indices.sort().values; n = D.mlp(h=k)
    with torch.no_grad():
        n[0].weight.copy_(l1.weight[i1]); n[0].bias.copy_(l1.bias[i1]); n[2].weight.copy_(l2.weight[i2][:, i1]); n[2].bias.copy_(l2.bias[i2])
        n[4].weight.copy_(l3.weight[:, i2]); n[4].bias.copy_(l3.bias)
    D.train(n.train(), Xtr, ytr, epochs=5, lr=5e-4); torch.save(n.state_dict(), D.P("w23_pruned128.pt"))
    D.export_onnx(T, D.P("w23_mlp.onnx")); D.export_onnx(n, D.P("w23_pruned128.onnx"))
    quantize_dynamic(D.P("w23_mlp.onnx"), D.P("w23_dyn.onnx"), weight_type=QuantType.QInt8)
    quantize_dynamic(D.P("w23_pruned128.onnx"), D.P("w23_pruned128_dyn.onnx"), weight_type=QuantType.QInt8)
VARIANTS = [("torch eager fp32", "torch", "w23_mlp.pt"), ("ORT fp32", "ort", "w23_mlp.onnx"), ("ORT dynamic int8", "ort", "w23_dyn.onnx"),
            ("ORT static QDQ int8", "ort", "w23_ort_static_QDQ_pertensor_MinMax.onnx"), ("ORT-format .ort int8", "ort", "day159_ort/w23_dyn.ort"),
            ("ExecuTorch XNNPACK fp32", "et", "w23_mlp_xnnpack.pte"), ("Rust int8 (tiny_infer)", "rust", "w23_mlp.pt"),
            ("pruned-128 ORT fp32", "ort", "w23_pruned128.onnx"), ("pruned-128 ORT dyn int8", "ort", "w23_pruned128_dyn.onnx")]
def child(kind, f):
    import time, logging; logging.disable(logging.WARNING)
    Xtr, Xte, ytr, yte = D.data(); r0 = D.rss_mb(); p = D.P(f)
    if kind == "torch":
        torch.set_num_threads(1); m = D.mlp(); m.load_state_dict(torch.load(p)); m.eval(); fn = lambda x: m(torch.from_numpy(x)).detach().numpy()
    elif kind == "ort":
        import onnxruntime as ort; so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]); fn = lambda x: s.run(None, {"x": x})[0]
    elif kind == "et":
        from executorch.runtime import Runtime; meth = Runtime.get().load_program(p).load_method("forward"); fn = lambda x: meth.execute([torch.from_numpy(x)])[0].numpy()
    else:
        import tiny_infer as TI; sd = torch.load(p); ws = [sd[f"{i}.weight"].numpy().copy() for i in (0, 2, 4)]; bs = [sd[f"{i}.bias"].numpy().copy() for i in (0, 2, 4)]
        e = TI.MlpInt8(ws, bs); fn = lambda x: e.forward(x)
    r1 = D.rss_mb(); pred = np.array([fn(Xte[i:i+1]).argmax() for i in range(len(Xte))]); ts = []
    for _ in range(50): fn(Xte[:1])
    for i in range(1000): x = Xte[i % len(Xte): i % len(Xte) + 1]; t = time.perf_counter(); fn(x); ts.append((time.perf_counter() - t) * 1e6)
    print(json.dumps(dict(acc=float((pred == yte).mean()), p50=float(np.percentile(ts, 50)), p95=float(np.percentile(ts, 95)), rss=r1 - r0)))
if __name__ == "__main__" and len(sys.argv) == 3: child(sys.argv[1], sys.argv[2]); sys.exit()
build(); print(f"budget: {BUDGET}  (batch 1, 1 thread, 1000 timed calls after 50 warm-up; this x86 server CPU, NOT a phone)")
print(f"{'variant':26s} {'file kB':>8s} {'acc':>7s} {'p50 us':>7s} {'p95 us':>7s} {'load RSS MiB':>12s}  verdict")
for name, kind, f in VARIANTS:
    r = subprocess.run([sys.executable, __file__, kind, f], capture_output=True, text=True, env={**os.environ, "PYTHONWARNINGS": "ignore"})
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    if not line: print(f"{name:26s} FAILED: {r.stderr[-300:]}"); continue
    j = json.loads(line[0]); size = os.path.getsize(D.P(f)) / 1e3 if kind != "rust" else 308.0
    fails = [k for k, ok in [("acc", j["acc"] >= BUDGET["acc"]), ("p95", j["p95"] <= BUDGET["p95_us"]), ("size", size <= BUDGET["size_kb"]), ("rss", j["rss"] <= BUDGET["rss_mib"])] if not ok]
    print(f"{name:26s} {size:8.0f} {j['acc']:7.4f} {j['p50']:7.1f} {j['p95']:7.1f} {j['rss']:12.1f}  {'PASS' if not fails else 'FAIL: ' + ','.join(fails)}")
print("Rust row: file size = in-memory int8 weights + f32 scales/biases from MlpInt8.nbytes() (Day 160 log: 308 kB); it loads the fp32 .pt and quantizes at start-up.")
