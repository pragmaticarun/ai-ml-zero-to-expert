use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use std::sync::Mutex;

struct Index {
    q: Vec<i8>,
    scales: Vec<f32>,
    rows: usize,
    cols: usize,
}

static IDX: Mutex<Option<Index>> = Mutex::new(None);

#[inline]
fn dot_q(a: &[i8], b: &[i8]) -> i32 {
    let mut acc = [0i32; 32];
    let (ca, cb) = (a.chunks_exact(32), b.chunks_exact(32));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for k in 0..32 {
            acc[k] += x[k] as i32 * y[k] as i32;
        }
    }
    acc.iter().sum::<i32>() + ra.iter().zip(rb).map(|(x, y)| *x as i32 * *y as i32).sum::<i32>()
}

fn quantize_rows(x: &[f32], rows: usize, cols: usize) -> (Vec<i8>, Vec<f32>) {
    let mut q = vec![0i8; rows * cols];
    let mut scales = vec![0f32; rows];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let m = row.iter().copied().map(f32::abs).fold(0f32, f32::max).max(1e-12);
        let s = m / 127.0;
        scales[r] = s;
        for c in 0..cols {
            q[r * cols + c] = (row[c] / s).round().clamp(-127.0, 127.0) as i8;
        }
    }
    (q, scales)
}

#[pyfunction]
fn build_int8_index(db: PyReadonlyArray2<f32>) -> PyResult<usize> {
    let db = db.as_array();
    let rows = db.nrows();
    let cols = db.ncols();
    let flat: Vec<f32> = db.iter().copied().collect();
    let (q, scales) = quantize_rows(&flat, rows, cols);
    let nbytes = q.len() + scales.len() * 4;
    *IDX.lock().unwrap() = Some(Index { q, scales, rows, cols });
    Ok(nbytes)
}

#[pyfunction]
fn score_int8<'py>(
    py: Python<'py>,
    query: PyReadonlyArray1<f32>,
    parallel: bool,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let guard = IDX.lock().unwrap();
    let idx = guard
        .as_ref()
        .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("call build_int8_index first"))?;
    let query = query.as_array();
    assert_eq!(query.len(), idx.cols);
    let qmax = query.iter().copied().map(f32::abs).fold(0f32, f32::max).max(1e-12);
    let qs = qmax / 127.0;
    let qv: Vec<i8> = query
        .iter()
        .map(|v| (v / qs).round().clamp(-127.0, 127.0) as i8)
        .collect();
    let rows = idx.rows;
    let cols = idx.cols;
    let qdb = &idx.q;
    let scales = &idx.scales;
    let scores: Vec<f32> = if parallel {
        (0..rows)
            .into_par_iter()
            .map(|r| {
                let row = &qdb[r * cols..(r + 1) * cols];
                dot_q(row, &qv) as f32 * scales[r] * qs
            })
            .collect()
    } else {
        (0..rows)
            .map(|r| {
                let row = &qdb[r * cols..(r + 1) * cols];
                dot_q(row, &qv) as f32 * scales[r] * qs
            })
            .collect()
    };
    Ok(PyArray1::from_vec(py, scores))
}

#[pyfunction]
fn int8_dot_scores<'py>(
    py: Python<'py>,
    db: PyReadonlyArray2<f32>,
    query: PyReadonlyArray1<f32>,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, usize)> {
    let nb = build_int8_index(db)?;
    let s = score_int8(py, query, parallel)?;
    Ok((s, nb))
}

#[pymodule]
fn rust_score(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_int8_index, m)?)?;
    m.add_function(wrap_pyfunction!(score_int8, m)?)?;
    m.add_function(wrap_pyfunction!(int8_dot_scores, m)?)?;
    Ok(())
}
