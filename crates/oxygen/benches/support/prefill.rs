//! The prefill sweep's pure helpers (FR-42, Spec §12), unit-tested from
//! `tests/bench_support.rs`.

use serde_json::{json, Value};

/// The prompt lengths the sweep times, in tokens. 67 and 644 are the first and
/// last turn of release 0.3.0's ten-turn chat, so chat time to first token is
/// read from the rows at those lengths.
pub const LENGTHS: [usize; 5] = [32, 67, 128, 512, 644];

/// The first `n` ids of `base`, repeated from its start when `base` is shorter.
/// Timing depends on the prompt's length and not on its text, so repeating a
/// fixed prompt gives every length the same tokens in every run.
pub fn prompt_ids(base: &[u32], n: usize) -> Vec<u32> {
    base.iter().copied().cycle().take(if base.is_empty() { 0 } else { n }).collect()
}

/// The largest absolute difference between two logit vectors, measured in f64;
/// infinite when their lengths differ. The sweep checks that the two paths agree
/// on the last logits before it times them (FR-41).
pub fn logits_gap(a: &[f32], b: &[f32]) -> f64 {
    if a.len() != b.len() {
        return f64::INFINITY;
    }
    a.iter().zip(b).map(|(x, y)| (f64::from(*x) - f64::from(*y)).abs()).fold(0.0, f64::max)
}

/// The row fields of one path at one prompt length.
pub fn fields(path: &str, prompt_tokens: usize) -> Value {
    json!({"group": "prefill", "path": path, "prompt_tokens": prompt_tokens})
}
