//! Week 22 CPU kernels for Python via PyO3. Every kernel works on contiguous f32 slices and releases the GIL.
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods, IntoPyArray, PyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// One accumulator, indexed loop: a strict left-to-right f32 sum, which the compiler may NOT reorder, so it cannot vectorise.
fn dot_naive_s(a: &[f32], b: &[f32]) -> f32 { let mut s = 0f32; for i in 0..a.len() { s += a[i] * b[i]; } s }
/// 16 independent accumulators over exact chunks: no bounds checks in the hot loop, and lanes the compiler maps to SIMD registers.
fn dot_simd_s(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 16];
    let (ca, cb) = (a.chunks_exact(16), b.chunks_exact(16));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) { for k in 0..16 { acc[k] += x[k] * y[k]; } }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) { s += x * y; }
    s
}
#[pyfunction] fn dot_naive(py: Python<'_>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let (a, b) = (a.as_slice()?, b.as_slice()?); Ok(py.allow_threads(|| dot_naive_s(a, b))) }
#[pyfunction] fn dot_simd(py: Python<'_>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let (a, b) = (a.as_slice()?, b.as_slice()?); Ok(py.allow_threads(|| dot_simd_s(a, b))) }
#[pyfunction] fn dot_par(py: Python<'_>, a: PyReadonlyArray1<f32>, b: PyReadonlyArray1<f32>) -> PyResult<f32> {
    let (a, b) = (a.as_slice()?, b.as_slice()?);
    Ok(py.allow_threads(|| a.par_chunks(1 << 16).zip(b.par_chunks(1 << 16)).map(|(x, y)| dot_simd_s(x, y)).sum())) }

/// Row softmax, three passes over memory (max, exp+sum into the output, scale) -- what a chain of numpy ops does, minus temporaries.
fn softmax3(x: &[f32], o: &mut [f32]) {
    let m = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32; for (oi, xi) in o.iter_mut().zip(x) { *oi = (xi - m).exp(); s += *oi; }
    let inv = 1.0 / s; for oi in o.iter_mut() { *oi *= inv; }
}
/// "Online" softmax: running max and rescaled running sum in ONE read pass, then one write pass (the FlashAttention trick, 1-D).
fn softmax_online(x: &[f32], o: &mut [f32]) {
    let (mut m, mut s) = (f32::NEG_INFINITY, 0f32);
    for &xi in x { if xi > m { s = s * (m - xi).exp() + 1.0; m = xi; } else { s += (xi - m).exp(); } }
    let inv = 1.0 / s; for (oi, xi) in o.iter_mut().zip(x) { *oi = (xi - m).exp() * inv; }
}
#[pyfunction] #[pyo3(signature = (x, online=false, parallel=false))]
fn softmax_rows<'py>(py: Python<'py>, x: PyReadonlyArray2<'py, f32>, online: bool, parallel: bool) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (r, c) = (x.shape()[0], x.shape()[1]); let xs = x.as_slice()?; let mut out = vec![0f32; r * c];
    py.allow_threads(|| {
        let f = |(xr, or): (&[f32], &mut [f32])| if online { softmax_online(xr, or) } else { softmax3(xr, or) };
        if parallel { xs.par_chunks(c).zip(out.par_chunks_mut(c)).for_each(f) } else { xs.chunks(c).zip(out.chunks_mut(c)).for_each(f) }
    });
    Ok(numpy::ndarray::Array2::from_shape_vec((r, c), out).unwrap().into_pyarray(py))
}

/// BM25 over a CSC (term-major) weight matrix: for each query term id, add that term's column into the doc scores; then top-k.
/// indptr/indices/data are scipy's CSC arrays (indices = doc ids).
fn bm25_one(indptr: &[i64], indices: &[i32], data: &[f32], q: &[i64], ndoc: usize, k: usize) -> Vec<i64> {
    let mut s = vec![0f32; ndoc];
    for &t in q { let (a, b) = (indptr[t as usize] as usize, indptr[t as usize + 1] as usize);
        for j in a..b { s[indices[j] as usize] += data[j]; } }
    let mut idx: Vec<usize> = (0..ndoc).collect(); let k = k.min(ndoc);
    idx.select_nth_unstable_by(k - 1, |&x, &y| s[y].partial_cmp(&s[x]).unwrap());
    let mut top = idx[..k].to_vec(); top.sort_by(|&x, &y| s[y].partial_cmp(&s[x]).unwrap().then(x.cmp(&y)));
    top.into_iter().map(|i| i as i64).collect()
}
#[pyfunction] #[pyo3(signature = (indptr, indices, data, queries, ndoc, k, parallel=false))]
fn bm25_topk<'py>(py: Python<'py>, indptr: PyReadonlyArray1<'py, i64>, indices: PyReadonlyArray1<'py, i32>, data: PyReadonlyArray1<'py, f32>,
                  queries: Vec<Vec<i64>>, ndoc: usize, k: usize, parallel: bool) -> PyResult<Vec<Vec<i64>>> {
    let (ip, ix, d) = (indptr.as_slice()?, indices.as_slice()?, data.as_slice()?);
    Ok(py.allow_threads(|| if parallel { queries.par_iter().map(|q| bm25_one(ip, ix, d, q, ndoc, k)).collect() }
                           else { queries.iter().map(|q| bm25_one(ip, ix, d, q, ndoc, k)).collect() }))
}
#[pyfunction] fn n_threads() -> usize { rayon::current_num_threads() }
#[allow(dead_code)] fn _unused(py: Python<'_>) -> Bound<'_, PyArray1<f32>> { vec![0f32].into_pyarray(py) }

#[pymodule]
fn rust_kernels(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(dot_naive, m)?, wrap_pyfunction!(dot_simd, m)?, wrap_pyfunction!(dot_par, m)?, wrap_pyfunction!(softmax_rows, m)?,
              wrap_pyfunction!(bm25_topk, m)?, wrap_pyfunction!(n_threads, m)?] { m.add_function(f)?; }
    Ok(())
}
