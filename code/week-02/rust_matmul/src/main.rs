// Day 9 systems contrast: naive i-p-j matmul in Rust, same algorithm as the Python loop.
use std::time::Instant;
fn matmul(a: &[f64], b: &[f64], c: &mut [f64], n: usize) {
    for i in 0..n { for p in 0..n { let aip = a[i*n+p];
        for j in 0..n { c[i*n+j] += aip * b[p*n+j]; } } }
}
fn main() {
    for &n in &[128usize, 512] {
        // deterministic pseudo-random fill (LCG) to avoid external crates
        let mut s: u64 = 42; let mut r = || { s = s.wrapping_mul(6364136223846793005).wrapping_add(1); (s >> 11) as f64 / (1u64<<53) as f64 };
        let a: Vec<f64> = (0..n*n).map(|_| r()).collect(); let b: Vec<f64> = (0..n*n).map(|_| r()).collect();
        let mut c = vec![0.0; n*n]; let t = Instant::now(); matmul(&a, &b, &mut c, n); let dt = t.elapsed().as_secs_f64();
        println!("rust naive n={}: {:.2} ms = {:.2} GFLOP/s (checksum {:.3})", n, dt*1e3, 2.0*(n as f64).powi(3)/dt/1e9, c.iter().sum::<f64>());
    }
}
