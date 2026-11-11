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

/// The largest difference the sweep accepts between its two paths' last logits:
/// the batched-against-one-token bound of Specification §11, 1e-3 since release
/// 0.4.2 (the paths sum in a different order, so they differ by about 2e-5 on
/// these prompts; 0.4.0 and 0.4.1 measured exactly 0 and used 1e-5).
pub const AGREEMENT_BOUND: f64 = 1e-3;

/// Whether a gap between the two paths' last logits is acceptable. NaN and
/// infinity (paths of different shapes) never are.
pub fn agree(gap: f64) -> bool {
    gap <= AGREEMENT_BOUND
}

/// The row fields of one path at one prompt length.
pub fn fields(path: &str, prompt_tokens: usize) -> Value {
    json!({"group": "prefill", "path": path, "prompt_tokens": prompt_tokens})
}
