// Day 96: BPE encoder in Rust. Same algorithm as _bpe.encode_word (lowest-rank merge first), applied to pre-split words.
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::HashMap;

#[pyclass]
struct Encoder { ranks: HashMap<(u32, u32), u32> }

fn encode_word(ranks: &HashMap<(u32, u32), u32>, w: &[u8]) -> Vec<u32> {
    let mut s: Vec<u32> = w.iter().map(|&b| b as u32).collect();
    while s.len() > 1 {
        let mut best = (u32::MAX, 0usize);
        for j in 0..s.len() - 1 {
            if let Some(&r) = ranks.get(&(s[j], s[j + 1])) { if r < best.0 { best = (r, j); } }
        }
        if best.0 == u32::MAX { break; }
        s[best.1] = best.0; s.remove(best.1 + 1);
    }
    s
}

#[pymethods]
impl Encoder {
    #[new]
    fn new(merges: Vec<((u32, u32), u32)>) -> Self { Encoder { ranks: merges.into_iter().collect() } }
    /// Encode a list of pre-tokenised words (as bytes). parallel=true uses rayon across words.
    fn encode_words(&self, py: Python<'_>, words: Vec<Vec<u8>>, parallel: bool) -> Vec<u32> {
        py.allow_threads(|| {
            if parallel { words.par_iter().map(|w| encode_word(&self.ranks, w)).collect::<Vec<_>>().concat() }
            else { words.iter().flat_map(|w| encode_word(&self.ranks, w)).collect() }
        })
    }
}

#[pymodule]
fn rust_bpe(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Encoder>()?; Ok(()) }
