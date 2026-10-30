//! int8 weight-only matmul: y[b, o] = scale[o] * sum_i x[b, i] * w[o, i], with w stored as i8 (per-output-channel scale).
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use pyo3::prelude::*;
use rayon::prelude::*;

fn dot_i8(x: &[f32], w: &[i8]) -> f32 {
    // 8 independent accumulators so the compiler can vectorise
    let mut acc = [0f32; 8];
    let n = x.len() / 8 * 8;
    for i in (0..n).step_by(8) {
        for k in 0..8 { acc[k] += x[i + k] * w[i + k] as f32; }
    }
    let mut s: f32 = acc.iter().sum();
    for i in n..x.len() { s += x[i] * w[i] as f32; }
    s
}

#[pyfunction]
fn matmul_i8<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f32>, w: PyReadonlyArray2<'py, i8>, scale: PyReadonlyArray1<'py, f32>, parallel: bool) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (xb, wb, sb) = (x.as_array(), w.as_array(), scale.as_slice()?);
    let (bsz, d_in) = (xb.shape()[0], xb.shape()[1]);
    let d_out = wb.shape()[0];
    if wb.shape()[1] != d_in { return Err(pyo3::exceptions::PyValueError::new_err("shape mismatch")); }
    let xs = xb.as_standard_layout().to_owned(); let ws = wb.as_standard_layout().to_owned();
    let (xs, ws) = (xs.as_slice().unwrap(), ws.as_slice().unwrap());
    let mut out = vec![0f32; bsz * d_out];
    let row = |(o, chunk): (usize, &mut [f32])| {
        let wr = &ws[o * d_in..(o + 1) * d_in];
        for b in 0..bsz { chunk[b] = sb[o] * dot_i8(&xs[b * d_in..(b + 1) * d_in], wr); }
    };
    // compute in [out, batch] layout, transpose at the end
    let mut tmp = vec![0f32; d_out * bsz];
    py.allow_threads(|| {
        if parallel { tmp.par_chunks_mut(bsz).enumerate().for_each(row); } else { tmp.chunks_mut(bsz).enumerate().for_each(row); }
    });
    for o in 0..d_out { for b in 0..bsz { out[b * d_out + o] = tmp[o * bsz + b]; } }
    Ok(PyArray2::from_vec2(py, &out.chunks(d_out).map(|r| r.to_vec()).collect::<Vec<_>>())?)
}

#[pymodule]
fn rust_qmatmul(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_function(wrap_pyfunction!(matmul_i8, m)?)?; Ok(()) }
