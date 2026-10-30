//! Week-12 systems lab: a batch "collate" kernel in Rust. Given uint8 images (N, D) and a batch of
//! row indices, gather the rows and convert to normalised float32 in one pass (serial and rayon versions).
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ndarray::Array2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn work(x: &[u8], d: usize, idx: &[i64], mean: f32, std: f32, par: bool) -> Vec<f32> {
    let inv = 1.0 / (255.0 * std); let off = mean / std;
    let mut out = vec![0f32; idx.len() * d];
    let f = |(r, o): (usize, &mut [f32])| { let s = idx[r] as usize * d; for (t, v) in o.iter_mut().enumerate() { *v = x[s + t] as f32 * inv - off; } };
    if par { out.par_chunks_mut(d).enumerate().for_each(f); } else { out.chunks_mut(d).enumerate().for_each(f); }
    out
}
fn run<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, u8>, idx: PyReadonlyArray1<'py, i64>, mean: f32, std: f32, par: bool) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let xa = x.as_array(); let (n, d) = (xa.nrows(), xa.ncols());
    let xs = xa.as_slice().ok_or_else(|| pyo3::exceptions::PyValueError::new_err("images must be C-contiguous"))?;
    let iv: Vec<i64> = idx.as_array().to_vec();
    if iv.iter().any(|&i| i < 0 || i as usize >= n) { return Err(pyo3::exceptions::PyIndexError::new_err("index out of range")); }
    let b = iv.len(); let out = py.allow_threads(|| work(xs, d, &iv, mean, std, par));
    Ok(Array2::from_shape_vec((b, d), out).unwrap().into_pyarray(py))
}
#[pyfunction]
fn collate<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, u8>, idx: PyReadonlyArray1<'py, i64>, mean: f32, std: f32) -> PyResult<Bound<'py, PyArray2<f32>>> { run(py, x, idx, mean, std, false) }
#[pyfunction]
fn collate_par<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, u8>, idx: PyReadonlyArray1<'py, i64>, mean: f32, std: f32) -> PyResult<Bound<'py, PyArray2<f32>>> { run(py, x, idx, mean, std, true) }
#[pymodule]
fn rust_loader(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_function(wrap_pyfunction!(collate, m)?)?; m.add_function(wrap_pyfunction!(collate_par, m)?)?; Ok(()) }
