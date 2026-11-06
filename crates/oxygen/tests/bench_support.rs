//! Unit tests of the benchmark harness's pure helpers, which live in
//! `benches/support/`: `cargo test` does not run a `harness = false` bench's
//! own tests, so they are included here by path.

#[path = "../benches/support/stats.rs"]
mod stats;

#[path = "../benches/support/record.rs"]
mod record;

use record::{rfc3339, row, Conditions, Machine};
use serde_json::json;
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

fn conditions() -> Conditions {
    Conditions {
        version: "0.1.0".into(),
        commit: Some("abc123".into()),
        dirty: Some(false),
        timestamp: "2026-11-06T09:30:00Z".into(),
        machine: Machine {
            cpu: Some("Apple M4 Max".into()),
            cores: Some(16),
            performance_cores: Some(12),
            efficiency_cores: Some(4),
            memory_bytes: Some(137_438_953_472),
            os: "macos".into(),
            os_version: Some("26.3".into()),
            arch: "aarch64".into(),
        },
        profile: "bench".into(),
        debug_assertions: false,
        warmup_runs: 1,
        recorded_runs: 5,
    }
}

// FR-9, S-7, Spec §12: a row carries every condition with its results, each result as median,
// minimum and maximum, and the row's own fields (its group and workload).
#[test]
fn a_row_carries_its_conditions_and_results() {
    let results = [("generation_tokens_per_s", Spread { median: 20.3, min: 20.2, max: 20.4 })];
    let got = row(&conditions(), json!({"group": "workloads", "workload": "short", "prompt_tokens": 32, "new_tokens": 128}), &results);
    let want = json!({
        "version": "0.1.0", "commit": "abc123", "dirty": false, "timestamp": "2026-11-06T09:30:00Z",
        "machine": {
            "cpu": "Apple M4 Max", "cores": 16, "performance_cores": 12, "efficiency_cores": 4,
            "memory_bytes": 137_438_953_472u64, "os": "macos", "os_version": "26.3", "arch": "aarch64",
        },
        "build": {"profile": "bench", "debug_assertions": false},
        "warmup_runs": 1, "recorded_runs": 5,
        "group": "workloads", "workload": "short", "prompt_tokens": 32, "new_tokens": 128,
        "results": {"generation_tokens_per_s": {"median": 20.3, "min": 20.2, "max": 20.4}},
    });
    assert_eq!(got, want);
    // One line of the record: no line break inside it.
    assert!(!got.to_string().contains('\n'));
}

// FR-9: what the system does not report is recorded as null, not guessed or dropped.
#[test]
fn unknown_conditions_are_null() {
    let mut c = conditions();
    c.commit = None;
    c.dirty = None;
    c.machine.cpu = None;
    c.machine.performance_cores = None;
    let got = row(&c, json!({}), &[]);
    for pointer in ["/commit", "/dirty", "/machine/cpu", "/machine/performance_cores"] {
        assert_eq!(got.pointer(pointer), Some(&serde_json::Value::Null), "{pointer}");
    }
}

// FR-9: timestamps are RFC 3339 in UTC, checked at the epoch, a leap day and a known instant.
#[test]
fn timestamps_are_rfc3339_utc() {
    assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
    assert_eq!(rfc3339(951_782_400), "2000-02-29T00:00:00Z");
    assert_eq!(rfc3339(1_793_957_400), "2026-11-06T09:30:00Z");
    assert_eq!(rfc3339(1_793_957_400 + 86_399 - 34_200), "2026-11-06T23:59:59Z");
}
