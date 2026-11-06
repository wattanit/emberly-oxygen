//! Unit tests of the benchmark harness's pure helpers, which live in
//! `benches/support/`: `cargo test` does not run a `harness = false` bench's
//! own tests, so they are included here by path.

#[path = "../benches/support/stats.rs"]
mod stats;

use stats::{spread, Spread};

// S-7, FR-8: the median of an odd count is the middle value, whatever the order.
#[test]
fn median_of_an_odd_count_is_the_middle_value() {
    let s = spread(&[20.3, 19.9, 20.8, 20.1, 20.5]).unwrap();
    assert_eq!(s, Spread { median: 20.3, min: 19.9, max: 20.8 });
    assert_eq!(spread(&[7.0]).unwrap(), Spread { median: 7.0, min: 7.0, max: 7.0 });
}

// S-7: an even count takes the mean of the two middle values.
#[test]
fn median_of_an_even_count_is_the_mean_of_the_middle_two() {
    assert_eq!(spread(&[4.0, 1.0, 3.0, 2.0]).unwrap(), Spread { median: 2.5, min: 1.0, max: 4.0 });
}

// S-7: ties are kept, so a repeated value can be the median and the extremes.
#[test]
fn ties_are_counted() {
    assert_eq!(spread(&[5.0, 5.0, 1.0, 5.0, 9.0]).unwrap(), Spread { median: 5.0, min: 1.0, max: 9.0 });
    assert_eq!(spread(&[3.0; 5]).unwrap(), Spread { median: 3.0, min: 3.0, max: 3.0 });
}

// S-7: nothing to summarize, or a value that is not a number, gives no spread rather than a wrong one.
#[test]
fn empty_or_nan_gives_none() {
    assert_eq!(spread(&[]), None);
    assert_eq!(spread(&[1.0, f64::NAN, 2.0]), None);
}
