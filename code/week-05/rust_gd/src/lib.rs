//! Month-1 capstone (optional): batch gradient descent for linear regression in Rust, exposed via PyO3.
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, IntoPyArray};
use pyo3::prelude::*;

/// Fit w, b by full-batch gradient descent on X (m x n, row-major) and y (m). Returns (w, b).
#[pyfunction]
fn linreg_gd<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f64>, y: PyReadonlyArray1<'py, f64>,
                  lr: f64, steps: usize) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let x = x.as_array(); let y = y.as_array();
    let (m, n) = (x.nrows(), x.ncols());
    let mut w = vec![0.0f64; n]; let mut b = 0.0f64; let mut gw = vec![0.0f64; n];
    for _ in 0..steps {
        gw.iter_mut().for_each(|g| *g = 0.0); let mut gb = 0.0;
        for i in 0..m {
            let row = x.row(i);
            let mut pred = b; for j in 0..n { pred += w[j] * row[j]; }
            let err = pred - y[i];
            for j in 0..n { gw[j] += err * row[j]; } gb += err;
        }
        for j in 0..n { w[j] -= lr * gw[j] / m as f64; } b -= lr * gb / m as f64;
    }
    Ok((w.into_pyarray(py), b))
}

#[pymodule]
fn rust_gd(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_function(wrap_pyfunction!(linreg_gd, m)?)?; Ok(()) }
