//! Week-8 project: k-means kernels in Rust, exposed to Python with PyO3.
//! `assign`: nearest centroid per row (serial); `assign_par`: the same with rayon threads;
//! `lloyd`: full Lloyd iterations (assign + update) without returning to Python.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

fn nearest(row: &[f64], c: &[f64], k: usize, n: usize) -> (i64, f64) {
    let mut best = (0i64, f64::INFINITY);
    for j in 0..k {
        let cj = &c[j * n..(j + 1) * n];
        let mut d = 0.0; for t in 0..n { let e = row[t] - cj[t]; d += e * e; }
        if d < best.1 { best = (j as i64, d); }
    }
    best
}
fn to_vec(a: &PyReadonlyArray2<f64>) -> (Vec<f64>, usize, usize) {
    let v = a.as_array(); (v.iter().cloned().collect(), v.nrows(), v.ncols())
}
#[pyfunction]
fn assign<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, c: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<i64>> {
    let (xv, m, n) = to_vec(&x); let (cv, k, _) = to_vec(&c);
    let lab: Vec<i64> = py.allow_threads(|| (0..m).map(|i| nearest(&xv[i * n..(i + 1) * n], &cv, k, n).0).collect());
    lab.into_pyarray(py)
}
#[pyfunction]
fn assign_par<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, c: PyReadonlyArray2<'py, f64>) -> Bound<'py, PyArray1<i64>> {
    let (xv, m, n) = to_vec(&x); let (cv, k, _) = to_vec(&c);
    let lab: Vec<i64> = py.allow_threads(|| (0..m).into_par_iter().map(|i| nearest(&xv[i * n..(i + 1) * n], &cv, k, n).0).collect());
    lab.into_pyarray(py)
}
/// Run `iters` Lloyd iterations from initial centroids c0. Returns (centroids k x n, labels, inertia, iterations used).
#[pyfunction]
fn lloyd<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, c0: PyReadonlyArray2<'py, f64>, iters: usize)
    -> (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray1<i64>>, f64, usize) {
    let (xv, m, n) = to_vec(&x); let (mut c, k, _) = to_vec(&c0);
    let (lab, inertia, used) = py.allow_threads(|| {
        let mut lab = vec![0i64; m]; let mut used = 0; let mut inertia = 0.0;
        for it in 0..iters {
            let res: Vec<(i64, f64)> = (0..m).into_par_iter().map(|i| nearest(&xv[i * n..(i + 1) * n], &c, k, n)).collect();
            let mut sum = vec![0.0; k * n]; let mut cnt = vec![0usize; k]; inertia = 0.0;
            for (i, &(l, d)) in res.iter().enumerate() {
                lab[i] = l; inertia += d; cnt[l as usize] += 1;
                for t in 0..n { sum[l as usize * n + t] += xv[i * n + t]; }
            }
            let mut moved = 0.0;
            for j in 0..k { if cnt[j] > 0 { for t in 0..n { let v = sum[j * n + t] / cnt[j] as f64; moved += (v - c[j * n + t]).abs(); c[j * n + t] = v; } } }
            used = it + 1; if moved < 1e-12 { break; }
        }
        (lab, inertia, used)
    });
    let ca = numpy::ndarray::Array2::from_shape_vec((k, n), c).unwrap();
    (ca.into_pyarray(py), lab.into_pyarray(py), inertia, used)
}
#[pymodule]
fn rust_kmeans(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(assign, m)?)?; m.add_function(wrap_pyfunction!(assign_par, m)?)?; m.add_function(wrap_pyfunction!(lloyd, m)?)?; Ok(())
}
