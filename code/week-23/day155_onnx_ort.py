"""Day 155: ONNX export and ONNX Runtime tuning on CPU.
(1) export the digits MLP, check the graph and numerics vs PyTorch; (2) graph optimisation levels (and save the optimised graph);
(3) intra-op thread sweep at batch 1 and batch 256; (4) sequential vs parallel execution mode; (5) ORT's built-in profiler."""
import os, json, numpy as np, torch, onnx, onnxruntime as ort, _dep as D
torch.set_num_threads(8)
Xtr, Xte, ytr, yte = D.data(); m = D.base_model()
print(f"fp32 MLP 64-512-512-10: {sum(p.numel() for p in m.parameters())} params; torch test acc {D.acc(m, Xte, yte):.4f}")
path = D.export_onnx(m, D.P("w23_mlp.onnx")); g = onnx.load(path); onnx.checker.check_model(g)
print(f"(1) exported {os.path.basename(path)}: {os.path.getsize(path)/1e3:.0f} kB, opset {g.opset_import[0].version}, nodes {[n.op_type for n in g.graph.node]}")
def sess(level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL, intra=1, mode=ort.ExecutionMode.ORT_SEQUENTIAL, opt_path=None, prof=False):
    so = ort.SessionOptions(); so.graph_optimization_level = level; so.intra_op_num_threads = intra; so.execution_mode = mode
    if opt_path: so.optimized_model_filepath = opt_path
    if prof: so.enable_profiling = True; so.profile_file_prefix = D.P("day155_ortprof")
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
s = sess(); o = s.run(None, {"x": Xte})[0]
with torch.no_grad(): t = m(torch.from_numpy(Xte)).numpy()
print(f"    numerics vs torch on {len(Xte)} test rows: max |diff| {np.abs(o-t).max():.2e}; argmax agreement {(o.argmax(1)==t.argmax(1)).mean():.4f}; ORT acc {(o.argmax(1)==yte).mean():.4f}")
x1, x256 = Xte[:1], Xte[:256]
print("(2) graph optimisation levels (1 thread), median us/call:")
for name, lv in [("DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL), ("ENABLE_BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
                 ("ENABLE_EXTENDED", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED), ("ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL)]:
    op = D.P(f"day155_opt_{name}.onnx"); ss = sess(lv, opt_path=op); nodes = [n.op_type for n in onnx.load(op).graph.node]
    print(f"    {name:16s} b1 {D.bench(lambda: ss.run(None, {'x': x1})):7.1f}  b256 {D.bench(lambda: ss.run(None, {'x': x256}), reps=50):8.1f}  saved graph ops: {nodes}")
print("(3) intra_op_num_threads sweep (ENABLE_ALL), median us/call:")
with torch.no_grad(): tb1 = D.bench(lambda: m(torch.from_numpy(x1))); tb256 = D.bench(lambda: m(torch.from_numpy(x256)), reps=50)
for n in [1, 2, 4, 8]:
    ss = sess(intra=n); print(f"    threads={n}: b1 {D.bench(lambda: ss.run(None, {'x': x1})):7.1f}   b256 {D.bench(lambda: ss.run(None, {'x': x256}), reps=50):8.1f}")
print(f"    reference torch eager (8 threads): b1 {tb1:7.1f}   b256 {tb256:8.1f}")
ss = sess(intra=1, mode=ort.ExecutionMode.ORT_PARALLEL)
print(f"(4) ORT_PARALLEL execution mode (1 intra thread): b1 {D.bench(lambda: ss.run(None, {'x': x1})):.1f} us (a chain graph has no independent branches to run in parallel)")
ss = sess(prof=True)
for _ in range(50): ss.run(None, {"x": x256})
pf = ss.end_profiling(); ev = json.load(open(pf)); agg = {}
for e in ev:
    if e.get("cat") == "Node" and e["name"].endswith("_kernel_time"): agg[e["args"]["op_name"]] = agg.get(e["args"]["op_name"], 0) + e["dur"]
tot = sum(agg.values()); print(f"(5) ORT profiler ({os.path.basename(pf)}, 50 runs at b256): kernel time by op:", {k: f"{v/tot*100:.0f}%" for k, v in sorted(agg.items(), key=lambda kv: -kv[1])})
