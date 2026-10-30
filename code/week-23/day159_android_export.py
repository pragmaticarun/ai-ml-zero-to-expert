"""Day 159: the three model files an Android app would ship, produced and run HERE on x86-64 Linux (not on a phone).
(1) ONNX -> ORT format (.ort) with ORT's own converter (what ONNX Runtime Mobile's docs recommend for minimal builds); load and compare;
(2) PyTorch -> ExecuTorch .pte (torch.export -> XNNPACK partitioner -> to_executorch), run with the ExecuTorch Python runtime; compare;
(3) LiteRT: the ai-edge-litert 2.3.0 interpreter imports, but converting PyTorch -> .tflite needs a separate converter package not installed
    here, so LiteRT conversion/run is UNTESTED. Everything about the phone side (Gradle, Kotlin, JNI) is UNTESTED on this box."""
import os, glob, subprocess, sys, shutil, numpy as np, torch, onnxruntime as ort, _dep as D
torch.set_num_threads(1)
Xtr, Xte, ytr, yte = D.data(); m = D.base_model(); x = torch.from_numpy(Xte[:1])
with torch.no_grad(): ref = m(torch.from_numpy(Xte)).numpy()
fp = D.P("w23_mlp.onnx"); D.export_onnx(m, fp)
od = D.P("day159_ort"); shutil.rmtree(od, ignore_errors=True); os.makedirs(od); shutil.copy(fp, od); shutil.copy(D.P("w23_dyn.onnx"), od)
r = subprocess.run([sys.executable, "-m", "onnxruntime.tools.convert_onnx_models_to_ort", od], capture_output=True, text=True)
print(f"(1) convert_onnx_models_to_ort exit {r.returncode}; files: {sorted(os.path.basename(f) + ' ' + str(os.path.getsize(f)//1000) + 'kB' for f in glob.glob(od + '/*') if not f.endswith('.onnx'))}")
if r.returncode: print(r.stderr[-800:])
for f in sorted(glob.glob(od + "/*.ort")):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"]); o = s.run(None, {"x": Xte})[0]
    print(f"    {os.path.basename(f):22s} acc {(o.argmax(1)==yte).mean():.4f}; max|diff| vs torch {np.abs(o-ref).max():.3e}; b1 {D.bench(lambda: s.run(None, {'x': Xte[:1]})):.1f} us (1 thread)")
cfgs = glob.glob(od + "/*.config"); print("    required-operators config (input to a reduced-size custom build):", open(cfgs[0]).read().strip().replace("\n", " | ") if cfgs else "none")
try:
    from executorch.exir import to_edge_transform_and_lower
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.runtime import Runtime
    rt = Runtime.get()
    for name, parts in [("portable (no delegate)", []), ("XNNPACK delegate", [XnnpackPartitioner()])]:
        ep = torch.export.export(m, (x,)); prog = to_edge_transform_and_lower(ep, partitioner=parts).to_executorch()
        pte = D.P("w23_mlp_" + ("xnnpack" if parts else "portable") + ".pte"); open(pte, "wb").write(prog.buffer)
        meth = rt.load_program(pte).load_method("forward")
        o = np.concatenate([meth.execute([torch.from_numpy(Xte[i:i+1])])[0].numpy() for i in range(len(Xte))])
        print(f"(2) ExecuTorch {name:22s}: {os.path.basename(pte)} {os.path.getsize(pte)//1000} kB; acc {(o.argmax(1)==yte).mean():.4f}; max|diff| vs torch {np.abs(o-ref).max():.3e}; "
              f"b1 {D.bench(lambda: meth.execute([x])):.1f} us (Python binding overhead included; static batch=1 export)")
except Exception as e: print("(2) ExecuTorch FAILED:", type(e).__name__, str(e)[:300])
try:
    from ai_edge_litert.interpreter import Interpreter; import ai_edge_litert
    print(f"(3) LiteRT: ai_edge_litert {getattr(ai_edge_litert, '__version__', '2.3.0 (pip)')} Interpreter importable = True; .tflite conversion from PyTorch UNTESTED here (converter not installed)")
except Exception as e: print("(3) LiteRT import FAILED:", e)
with torch.no_grad(): print(f"reference torch eager b1 {D.bench(lambda: m(x)):.1f} us (1 thread)")
