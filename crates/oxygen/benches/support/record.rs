//! One line of the cumulative record (FR-9, Spec §12): every result carries
//! the conditions it was measured under, so rows from different releases can
//! be compared. Pure, so it is unit-tested from `tests/bench_support.rs`.

use serde_json::{json, Map, Value};

use super::stats::Spread;

/// The machine a row was measured on. A field the system does not report is
/// `None` and is recorded as `null`, never guessed.
#[derive(Debug, Clone, PartialEq)]
pub struct Machine {
    pub cpu: Option<String>,
    pub cores: Option<u64>,
    pub performance_cores: Option<u64>,
    pub efficiency_cores: Option<u64>,
    pub memory_bytes: Option<u64>,
    pub os: String,
    pub os_version: Option<String>,
    pub arch: String,
}

/// What every row of one benchmark run shares.
#[derive(Debug, Clone, PartialEq)]
pub struct Conditions {
    pub version: String,
    /// `None` when `git` is not available.
    pub commit: Option<String>,
    pub dirty: Option<bool>,
    pub timestamp: String,
    pub machine: Machine,
    pub profile: String,
    pub debug_assertions: bool,
    pub warmup_runs: usize,
    pub recorded_runs: usize,
}

/// The kernel variants the kernel group times besides the engine's own kernel
/// (`in_house`) and the library baseline (`matrixmultiply`): the 0.1.0 loop and
/// the partial-sum kernel at each width the release tries (FR-25).
pub const KERNEL_VARIANTS: [&str; 5] = ["index_order", "lanes_4", "lanes_8", "lanes_16", "lanes_32"];

/// The two metric names of one kernel in a kernel row. `in_house` and
/// `matrixmultiply` give the four names release 0.1.0's rows already have, so
/// the series stays comparable (FR-9); each variant adds two.
pub fn kernel_metrics(kernel: &str) -> [String; 2] {
    [format!("{kernel}_ns_per_call"), format!("{kernel}_gflops")]
}

/// A row: the shared conditions, the row's own `fields` (its group and what
/// it measured), and each metric's spread over the recorded runs.
pub fn row(conditions: &Conditions, fields: Value, results: &[(&str, Spread)]) -> Value {
    let c = conditions;
    let m = &c.machine;
    let mut row = json!({
        "version": c.version,
        "commit": c.commit,
        "dirty": c.dirty,
        "timestamp": c.timestamp,
        "machine": {
            "cpu": m.cpu,
            "cores": m.cores,
            "performance_cores": m.performance_cores,
            "efficiency_cores": m.efficiency_cores,
            "memory_bytes": m.memory_bytes,
            "os": m.os,
            "os_version": m.os_version,
            "arch": m.arch,
        },
        "build": {"profile": c.profile, "debug_assertions": c.debug_assertions},
        "warmup_runs": c.warmup_runs,
        "recorded_runs": c.recorded_runs,
    });
    let object = row.as_object_mut().expect("a row is an object");
    if let Value::Object(own) = fields {
        object.extend(own);
    }
    let mut measured = Map::new();
    for (metric, s) in results {
        measured.insert((*metric).to_owned(), json!({"median": s.median, "min": s.min, "max": s.max}));
    }
    object.insert("results".into(), Value::Object(measured));
    row
}

/// Seconds since the Unix epoch as an RFC 3339 UTC timestamp, such as
/// `2026-11-06T09:30:00Z`. Days are converted to a civil date with Howard
/// Hinnant's algorithm, valid for every date after 1970.
pub fn rfc3339(unix_seconds: u64) -> String {
    let days = (unix_seconds / 86_400) as i64;
    let secs = unix_seconds % 86_400;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        secs / 3_600,
        secs % 3_600 / 60,
        secs % 60
    )
}
