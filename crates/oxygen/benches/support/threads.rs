//! The thread sweep's pure helpers (FR-44, FR-45, Spec §12), unit-tested from
//! `tests/bench_support.rs`.

use serde_json::{json, Value};

/// The thread counts the sweep times (initial; tune with use).
pub const COUNTS: [usize; 6] = [1, 2, 4, 8, 12, 16];

/// Prompt lengths timed alone, in tokens.
pub const PROMPT_LENGTHS: [usize; 2] = [128, 644];

/// Prompt lengths after which decoding is timed.
pub const DECODE_CONTEXTS: [usize; 2] = [32, 512];

/// Tokens generated in each decode timing.
pub const DECODE_TOKENS: usize = 128;

/// The sweep's counts that the machine can run, in order: a count above the
/// cores available is left out, because it would measure the scheduler.
pub fn counts_for(cores: usize) -> Vec<usize> {
    COUNTS.into_iter().filter(|&n| n <= cores).collect()
}

/// The default thread count (FR-44), from rows of `(count, decode ms per token
/// at 512 tokens, prompt ms at 644 tokens)`: the count with the highest decode
/// speed; counts within 2% of the best decode time are ties, and a tie goes to
/// the faster prompt, then to the smaller count (initial; tune with use).
pub fn choose_default(rows: &[(usize, f64, f64)]) -> usize {
    let best = rows.iter().map(|&(_, decode, _)| decode).fold(f64::INFINITY, f64::min);
    let (count, _) = rows
        .iter()
        .filter(|&&(_, decode, _)| decode <= best * 1.02)
        .map(|&(n, _, prompt)| (n, prompt))
        .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
        .expect("at least one count");
    count
}

/// Whether two last-logit vectors are identical. The sweep requires it of every
/// count before timing anything: the output does not depend on the count (FR-43).
pub fn logits_identical(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
}

/// The row fields of one thread count, phase (`prompt` or `decode`) and context length.
pub fn fields(threads: usize, phase: &str, context_tokens: usize) -> Value {
    json!({"group": "threads", "threads": threads, "phase": phase, "context_tokens": context_tokens})
}

/// `fields` with the thread count in use added, for the other groups' rows (FR-44, FR-9).
pub fn with_threads(mut fields: Value, threads: usize) -> Value {
    fields["threads"] = json!(threads);
    fields
}
