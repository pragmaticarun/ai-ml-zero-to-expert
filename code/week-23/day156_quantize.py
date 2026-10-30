"""Day 156: int8 quantization for deployment, measured honestly (accuracy, size, latency) on this CPU.
ORT dynamic (weights int8, activations quantized at run time), ORT static (calibrated activation ranges; QDQ format,
per-tensor vs per-channel; MinMax vs Percentile), and PyTorch dynamic quantization."""
import os, warnings, numpy as np, torch, onnxruntime as ort, _dep as D
from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType, QuantFormat, CalibrationDataReader, CalibrationMethod
warnings.filterwarnings("ignore"); torch.set_num_threads(4)
Xtr, Xte, ytr, yte = D.data(); m = D.base_model(); fp = D.P("w23_mlp.onnx")
if not os.path.exists(fp): D.export_onnx(m, fp)
class Reader(CalibrationDataReader):
    def __init__(s, X): s.it = iter([{"x": X[i:i+1]} for i in range(len(X))])
    def get_next(s): return next(s.it, None)
cal = Xtr[:200]  # calibration set is drawn from TRAIN data, never test
variants = {"fp32": fp}
quantize_dynamic(fp, D.P("w23_dyn.onnx"), weight_type=QuantType.QInt8); variants["ort dynamic int8"] = D.P("w23_dyn.onnx")
for name, pc, meth in [("ort static QDQ per-tensor MinMax", False, CalibrationMethod.MinMax), ("ort static QDQ per-channel MinMax", True, CalibrationMethod.MinMax),
                       ("ort static QDQ per-channel Percentile", True, CalibrationMethod.Percentile)]:
    out = D.P("w23_" + name.replace(" ", "_").replace("-", "") + ".onnx")
    quantize_static(fp, out, Reader(cal), quant_format=QuantFormat.QDQ, per_channel=pc, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, calibrate_method=meth)
    variants[name] = out
with torch.no_grad(): ref = m(torch.from_numpy(Xte)).numpy()
print(f"{'variant':40s} {'size kB':>8s} {'acc':>7s} {'agree':>6s} {'max|dlogit|':>11s} {'b1 us(1t)':>9s} {'b256 us(1t)':>11s} {'b256 us(4t)':>11s}")
for name, f in variants.items():
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s1 = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    so4 = ort.SessionOptions(); so4.intra_op_num_threads = 4; s4 = ort.InferenceSession(f, so4, providers=["CPUExecutionProvider"])
    o = s1.run(None, {"x": Xte})[0]
    print(f"{name:40s} {os.path.getsize(f)/1e3:8.0f} {(o.argmax(1)==yte).mean():7.4f} {(o.argmax(1)==ref.argmax(1)).mean():6.3f} {np.abs(o-ref).max():11.3f} "
          f"{D.bench(lambda: s1.run(None, {'x': Xte[:1]})):9.1f} {D.bench(lambda: s1.run(None, {'x': Xte[:256]}), reps=50):11.1f} {D.bench(lambda: s4.run(None, {'x': Xte[:256]}), reps=50):11.1f}")
try:
    from torch.ao.quantization import quantize_dynamic as tqd
    qm = tqd(m, {torch.nn.Linear}, dtype=torch.qint8)
    with torch.no_grad(): o = qm(torch.from_numpy(Xte)).numpy()
    torch.save(qm.state_dict(), D.P("w23_torch_dyn.pt")); torch.save(m.state_dict(), D.P("w23_mlp.pt"))
    with torch.no_grad():
        b1 = D.bench(lambda: qm(torch.from_numpy(Xte[:1]))); b256 = D.bench(lambda: qm(torch.from_numpy(Xte[:256])), reps=50)
        f1 = D.bench(lambda: m(torch.from_numpy(Xte[:1]))); f256 = D.bench(lambda: m(torch.from_numpy(Xte[:256])), reps=50)
    print(f"torch fp32 eager (4 threads): b1 {f1:.1f} us, b256 {f256:.1f} us; state_dict {os.path.getsize(D.P('w23_mlp.pt'))/1e3:.0f} kB")
    print(f"torch dynamic qint8 (4 threads, engine {torch.backends.quantized.engine}): acc {(o.argmax(1)==yte).mean():.4f}, agree {(o.argmax(1)==ref.argmax(1)).mean():.3f}, "
          f"max|dlogit| {np.abs(o-ref).max():.3f}, b1 {b1:.1f} us, b256 {b256:.1f} us; state_dict {os.path.getsize(D.P('w23_torch_dyn.pt'))/1e3:.0f} kB")
except Exception as e: print("torch dynamic quantization FAILED:", type(e).__name__, str(e)[:200])
print("CPU flags relevant to int8:", [f for f in open('/proc/cpuinfo').read().split('flags')[1].split('\n')[0].split() if f in ('avx512_vnni', 'avx_vnni', 'amx_int8', 'avx512bw')])
