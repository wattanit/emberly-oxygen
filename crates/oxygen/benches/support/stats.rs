//! Median, minimum and maximum over repeated runs (S-7). Pure, so it is
//! unit-tested from `tests/bench_support.rs`.

/// A measurement over repeated runs: the median is the result, and the
/// minimum and maximum show how noisy the runs were.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spread {
    pub median: f64,
    pub min: f64,
    pub max: f64,
}

/// The spread of `values`, in any order. The median of an odd count is the
/// middle value; of an even count, the mean of the two middle values. `None`
/// for no values or a value that is not a number, which no run should give.
pub fn spread(values: &[f64]) -> Option<Spread> {
    if values.is_empty() || values.iter().any(|v| v.is_nan()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 1 { sorted[n / 2] } else { (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0 };
    Some(Spread { median, min: sorted[0], max: sorted[n - 1] })
}
