//! Week 23: a tiny MLP inference engine in Rust (PyO3). fp32 and int8 paths for a ReLU MLP stored as row-major [out, in] weights.
//! int8 path = "dynamic quantization": weights int8 with one f32 scale per output row (per-channel, symmetric), activations
//! quantized per input row at run time (symmetric, scale = max|x|/127), i32 accumulation, dequantize, add bias, ReLU.
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods, IntoPyArray};
use pyo3::prelude::*;
use rayon::prelude::*;

struct LayerF { w: Vec<f32>, b: Vec<f32>, o: usize, i: usize }
struct LayerQ { w: Vec<i8>, s: Vec<f32>, b: Vec<f32>, o: usize, i: usize }

fn dot_f(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16)); let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for k in 0..16 { acc[k] += x[k] * y[k]; } }
    acc.iter().sum::<f32>() + ra.iter().zip(rb).map(|(x, y)| x * y).sum::<f32>()
}
fn dot_q(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = [0i32; 32];
    let (ca, cb) = (a.chunks_exact(32), b.chunks_exact(32)); let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for k in 0..32 { acc[k] += x[k] as i32 * y[k] as i32; } }
    acc.iter().sum::<i32>() + ra.iter().zip(rb).map(|(x, y)| *x as i32 * *y as i32).sum::<i32>()
}
fn row_f(layers: &[LayerF], x: &[f32]) -> Vec<f32> {
    let mut h = x.to_vec(); let n = layers.len();
    for (li, l) in layers.iter().enumerate() {
        let mut y: Vec<f32> = (0..l.o).map(|r| dot_f(&l.w[r * l.i..(r + 1) * l.i], &h) + l.b[r]).collect();
        if li + 1 < n { for v in y.iter_mut() { if *v < 0.0 { *v = 0.0; } } }
        h = y;
    }
    h
}
fn row_q(layers: &[LayerQ], x: &[f32]) -> Vec<f32> {
    let mut h = x.to_vec(); let n = layers.len(); let mut hq = Vec::new();
    for (li, l) in layers.iter().enumerate() {
        let amax = h.iter().fold(0f32, |m, v| m.max(v.abs())); let sx = if amax > 0.0 { amax / 127.0 } else { 1.0 };
        hq.clear(); hq.extend(h.iter().map(|v| (v / sx).round().clamp(-127.0, 127.0) as i8));
        let mut y: Vec<f32> = (0..l.o).map(|r| dot_q(&l.w[r * l.i..(r + 1) * l.i], &hq) as f32 * sx * l.s[r] + l.b[r]).collect();
        if li + 1 < n { for v in y.iter_mut() { if *v < 0.0 { *v = 0.0; } } }
        h = y;
    }
    h
}
fn run<F: Fn(&[f32]) -> Vec<f32> + Sync>(x: &[f32], rows: usize, cols: usize, out: usize, par: bool, f: F) -> Vec<f32> {
    let res: Vec<Vec<f32>> = if par { (0..rows).into_par_iter().map(|r| f(&x[r * cols..(r + 1) * cols])).collect() }
                             else { (0..rows).map(|r| f(&x[r * cols..(r + 1) * cols])).collect() };
    let mut flat = Vec::with_capacity(rows * out); for r in res { flat.extend(r); } flat
}

#[pyclass] struct MlpF32 { layers: Vec<LayerF> }
#[pymethods] impl MlpF32 {
    #[new] fn new(ws: Vec<PyReadonlyArray2<f32>>, bs: Vec<PyReadonlyArray1<f32>>) -> PyResult<Self> {
        let mut layers = Vec::new();
        for (w, b) in ws.iter().zip(bs.iter()) { let s = w.shape(); layers.push(LayerF { w: w.as_slice()?.to_vec(), b: b.as_slice()?.to_vec(), o: s[0], i: s[1] }); }
        Ok(Self { layers }) }
    #[pyo3(signature = (x, parallel=false))]
    fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>, parallel: bool) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (r, c) = (x.shape()[0], x.shape()[1]); let xs = x.as_slice()?; let o = self.layers.last().unwrap().o; let ls = &self.layers;
        let flat = py.allow_threads(|| run(xs, r, c, o, parallel, |row| row_f(ls, row)));
        Ok(numpy::ndarray::Array2::from_shape_vec((r, o), flat).unwrap().into_pyarray(py)) }
    fn nbytes(&self) -> usize { self.layers.iter().map(|l| 4 * (l.w.len() + l.b.len())).sum() }
}
#[pyclass] struct MlpInt8 { layers: Vec<LayerQ> }
#[pymethods] impl MlpInt8 {
    /// Quantize fp32 weights here (per output row, symmetric): w_q = round(w / s), s = max|w_row| / 127.
    #[new] fn new(ws: Vec<PyReadonlyArray2<f32>>, bs: Vec<PyReadonlyArray1<f32>>) -> PyResult<Self> {
        let mut layers = Vec::new();
        for (w, b) in ws.iter().zip(bs.iter()) {
            let sh = w.shape(); let (o, i) = (sh[0], sh[1]); let wf = w.as_slice()?; let mut q = vec![0i8; o * i]; let mut s = vec![0f32; o];
            for r in 0..o { let row = &wf[r * i..(r + 1) * i]; let m = row.iter().fold(0f32, |a, v| a.max(v.abs())); s[r] = if m > 0.0 { m / 127.0 } else { 1.0 };
                for k in 0..i { q[r * i + k] = (row[k] / s[r]).round().clamp(-127.0, 127.0) as i8; } }
            layers.push(LayerQ { w: q, s, b: b.as_slice()?.to_vec(), o, i });
        }
        Ok(Self { layers }) }
    #[pyo3(signature = (x, parallel=false))]
    fn forward<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'py, f32>, parallel: bool) -> PyResult<Bound<'py, PyArray2<f32>>> {
        let (r, c) = (x.shape()[0], x.shape()[1]); let xs = x.as_slice()?; let o = self.layers.last().unwrap().o; let ls = &self.layers;
        let flat = py.allow_threads(|| run(xs, r, c, o, parallel, |row| row_q(ls, row)));
        Ok(numpy::ndarray::Array2::from_shape_vec((r, o), flat).unwrap().into_pyarray(py)) }
    fn nbytes(&self) -> usize { self.layers.iter().map(|l| l.w.len() + 4 * (l.s.len() + l.b.len())).sum() }
}
#[pymodule] fn tiny_infer(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<MlpF32>()?; m.add_class::<MlpInt8>()?; Ok(()) }
