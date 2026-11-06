//! The benchmark harness (Spec §12), run with `cargo bench`.
//!
//! Each workload in `prompts.json` runs once to warm up and then
//! `RECORDED_RUNS` times, every run in a fresh child process: this program
//! starts itself again with `--child <workload>`, and the child loads the
//! model, generates, and prints its measurements as one JSON line. A fresh
//! process per run keeps each run's peak memory its own. The harness runs
//! offline and without Python (FR-7), and refuses a build with debug
//! assertions (FR-10). No log subscriber is installed, so the engine runs at
//! the default log level (S-8).
//!
//! Metrics (FR-8): prompt throughput, generation throughput and time to first
//! token come from `generate`'s `Stats` (the first token counts as prompt
//! work, plan §9); model load time is measured separately; peak resident
//! memory is the child's own high-water mark. Each is summarized over the
//! recorded runs as median, minimum and maximum (S-7).

//!
//! The record (FR-9): one JSON line per workload is appended to
//! `results/benchmarks.jsonl` once every workload has finished, each carrying
//! the version, commit, dirty flag, machine, build and timestamp. With
//! `cargo bench -- --dry-run` the lines are printed instead of appended.
//!
//! The kernel group (HC-2) times the in-house matrix-vector product against
//! `matrixmultiply`'s at every matrix size in the model, in this process, and
//! records one line per size with `"group": "kernels"`.

#[path = "support/record.rs"]
mod record;
#[path = "support/stats.rs"]
mod stats;

use std::io::Write;
use std::path::PathBuf;
use std::process::{exit, Command};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use std::hint::black_box;
use std::time::Duration;

use emberly_oxygen::{generate, kernels, Model, Tensor, Tokenizer};
use serde_json::{json, Value};

const WARMUP_RUNS: usize = 1;
const RECORDED_RUNS: usize = 5;

/// The measurements a child reports, in the order they are summarized.
const METRICS: [&str; 5] = [
    "time_to_first_token_ms",
    "prompt_tokens_per_s",
    "generation_tokens_per_s",
    "load_ms",
    "peak_rss_bytes",
];

struct Workload {
    id: String,
    text: String,
    prompt_tokens: usize,
    new_tokens: usize,
}

fn main() {
    if cfg!(debug_assertions) {
        fail("refusing to benchmark a build with debug assertions on: its timings measure the build settings, not the engine (FR-10); run `cargo bench`");
    }
    let args: Vec<String> = std::env::args().collect();
    match args.iter().position(|a| a == "--child") {
        Some(at) => {
            let id = args.get(at + 1).unwrap_or_else(|| fail("--child needs a workload"));
            child(&workload(id));
        }
        None => parent(),
    }
}

/// Prints the reason and exits non-zero: the benchmark stops, and nothing is
/// recorded from a run that did not complete.
fn fail(message: &str) -> ! {
    eprintln!("benchmark: {message}");
    exit(1)
}

fn root() -> PathBuf {
    PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
}

fn model_dir() -> PathBuf {
    let dir = root().join("models/smollm2-135m");
    if !dir.exists() {
        fail(&format!("{}: missing; run `cd oracle && uv run python download.py`", dir.display()));
    }
    dir
}

fn workloads() -> Vec<Workload> {
    let doc: Value = serde_json::from_str(include_str!("prompts.json")).expect("prompts.json is valid JSON");
    let list = doc["workloads"].as_array().expect("prompts.json lists workloads");
    list.iter()
        .map(|w| Workload {
            id: w["id"].as_str().expect("workload id").to_owned(),
            text: w["text"].as_str().expect("workload text").to_owned(),
            prompt_tokens: w["prompt_tokens"].as_u64().expect("prompt_tokens") as usize,
            new_tokens: w["new_tokens"].as_u64().expect("new_tokens") as usize,
        })
        .collect()
}

fn workload(id: &str) -> Workload {
    workloads()
        .into_iter()
        .find(|w| w.id == id)
        .unwrap_or_else(|| fail(&format!("no workload {id:?} in prompts.json")))
}

/// Runs every workload in child processes, reports each run, and records the
/// summaries.
fn parent() {
    let dry_run = std::env::args().any(|a| a == "--dry-run");
    // Taken before any run, so the commit and dirty flag describe the code measured.
    let conditions = conditions();
    let workloads = workloads();
    // The prompts must be exactly the lengths Spec §12 names, or the rows
    // would not measure the same work as the series they join.
    let tokenizer = Tokenizer::load(&model_dir()).unwrap_or_else(|e| fail(&e.to_string()));
    for w in &workloads {
        let n = tokenizer.encode(&w.text).unwrap_or_else(|e| fail(&e.to_string())).len();
        if n != w.prompt_tokens {
            fail(&format!("workload {}: the prompt is {n} tokens, prompts.json says {}", w.id, w.prompt_tokens));
        }
    }

    let mut rows = Vec::new();
    for w in &workloads {
        println!("workload {}: {} prompt tokens, {} new tokens", w.id, w.prompt_tokens, w.new_tokens);
        let mut recorded = Vec::with_capacity(RECORDED_RUNS);
        for run in 0..WARMUP_RUNS + RECORDED_RUNS {
            let measured = run_child(&w.id);
            let label = if run < WARMUP_RUNS { "warm-up".to_string() } else { format!("run {}", run - WARMUP_RUNS + 1) };
            println!("  {label:<8} {measured}");
            if run >= WARMUP_RUNS {
                recorded.push(measured);
            }
        }
        println!("  {:<24} {:>14} {:>14} {:>14}", "over 5 runs", "median", "min", "max");
        let mut results = Vec::new();
        for metric in METRICS {
            let s = summarize(&recorded, metric);
            let places = if metric.ends_with("_bytes") { 0 } else { 3 };
            println!("  {metric:<24} {:>14.places$} {:>14.places$} {:>14.places$}", s.median, s.min, s.max);
            results.push((metric, s));
        }
        let fields = json!({
            "group": "workloads",
            "workload": w.id,
            "prompt_tokens": w.prompt_tokens,
            "new_tokens": w.new_tokens,
        });
        rows.push(record::row(&conditions, fields, &results));
    }
    rows.extend(kernel_rows(&conditions));
    write_rows(&rows, dry_run);
}

/// Every matrix shape in the forward pass of SmolLM2-135M, as `[rows, columns]`
/// with the projections that use it.
const KERNEL_SIZES: [(usize, usize, &str); 5] = [
    (576, 576, "query and attention output"),
    (192, 576, "key and value"),
    (1536, 576, "MLP gate and up"),
    (576, 1536, "MLP down"),
    (49152, 576, "tied output projection"),
];

/// The kernel group (HC-2): the in-house `matvec` against `matrixmultiply`'s
/// `sgemm` with one column, both single-threaded, on the same inputs.
fn kernel_rows(conditions: &record::Conditions) -> Vec<Value> {
    println!("kernels: matrix-vector product, in-house against matrixmultiply sgemm (both single-threaded)");
    println!("  {:>12} {:>10} {:>10} {:>10} {:>10}  (medians over 5 runs)", "shape", "ours ns", "ours GF/s", "base ns", "base GF/s");
    let mut rows = Vec::new();
    for (seed, (m, k, role)) in KERNEL_SIZES.into_iter().enumerate() {
        let weight = Tensor::new(vec![m, k], pseudo_random(m * k, 2 * seed as u64 + 1)).expect("shape matches data");
        let input = pseudo_random(k, 2 * seed as u64 + 2);
        let mut ours = vec![0f32; m];
        let mut baseline = vec![0f32; m];
        let sgemm = |out: &mut [f32]| {
            // SAFETY: out (m x 1) = weight (m x k, row-major) * input (k x 1); every
            // pointer covers exactly the elements those strides reach.
            unsafe {
                matrixmultiply::sgemm(
                    m, k, 1, 1.0,
                    weight.data().as_ptr(), k as isize, 1,
                    input.as_ptr(), 1, 1,
                    0.0,
                    out.as_mut_ptr(), 1, 1,
                );
            }
        };

        // A baseline that computes something else is not timed.
        kernels::matvec(&weight, &input, &mut ours);
        sgemm(&mut baseline);
        for (i, (a, b)) in ours.iter().zip(&baseline).enumerate() {
            if (a - b).abs() > 1e-4 * (1.0 + b.abs()) {
                fail(&format!("kernels {m}x{k}: row {i} is {a} in-house and {b} in matrixmultiply"));
            }
        }

        let ours_ns = time_calls(|| kernels::matvec(&weight, black_box(&input), black_box(&mut ours)));
        let base_ns = time_calls(|| sgemm(black_box(&mut baseline)));
        let flops = 2.0 * (m * k) as f64;
        let gflops = |ns: &[f64]| ns.iter().map(|t| flops / t).collect::<Vec<f64>>();
        let spread = |values: &[f64]| stats::spread(values).unwrap_or_else(|| fail("kernels: no usable timings"));
        let results = [
            ("in_house_ns_per_call", spread(&ours_ns)),
            ("in_house_gflops", spread(&gflops(&ours_ns))),
            ("matrixmultiply_ns_per_call", spread(&base_ns)),
            ("matrixmultiply_gflops", spread(&gflops(&base_ns))),
        ];
        println!(
            "  {:>12} {:>10.0} {:>10.2} {:>10.0} {:>10.2}",
            format!("{m}x{k}"),
            results[0].1.median,
            results[1].1.median,
            results[2].1.median,
            results[3].1.median
        );
        let fields = json!({
            "group": "kernels",
            "kernel": "matvec",
            "rows": m,
            "cols": k,
            "role": role,
            "baseline": "matrixmultiply sgemm with one column, single-threaded",
        });
        rows.push(record::row(conditions, fields, &results));
    }
    rows
}

/// Nanoseconds per call of `f`, once per recorded run. The number of calls per
/// run is calibrated first (doubling until a batch takes 50 ms), which also
/// warms caches, and then set so that each run takes about a second.
fn time_calls(mut f: impl FnMut()) -> Vec<f64> {
    let mut calls = 1u64;
    loop {
        let start = Instant::now();
        (0..calls).for_each(|_| f());
        let took = start.elapsed();
        if took >= Duration::from_millis(50) {
            calls = ((calls as f64) * Duration::from_secs(1).as_secs_f64() / took.as_secs_f64()).ceil() as u64;
            break;
        }
        calls *= 2;
    }
    (0..RECORDED_RUNS)
        .map(|_| {
            let start = Instant::now();
            (0..calls).for_each(|_| f());
            start.elapsed().as_nanos() as f64 / calls as f64
        })
        .collect()
}

/// Deterministic values in [-1, 1), so every run times the same inputs.
fn pseudo_random(len: usize, seed: u64) -> Vec<f32> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Appends the rows to the record, one JSON line each, or prints them for a
/// dry run. Called once at the end, so a run that stops early records nothing.
fn write_rows(rows: &[Value], dry_run: bool) {
    let lines: String = rows.iter().map(|r| format!("{r}\n")).collect();
    let path = root().join("results/benchmarks.jsonl");
    if dry_run {
        print!("dry run, not appended to {}:\n{lines}", path.display());
        return;
    }
    let appended = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .and_then(|mut file| file.write_all(lines.as_bytes()));
    if let Err(e) = appended {
        fail(&format!("{}: {e}", path.display()));
    }
    println!("appended {} rows to {}", rows.len(), path.display());
}

/// The conditions every row of this run shares (FR-9).
fn conditions() -> record::Conditions {
    let git = |args: &[&str]| -> Option<String> {
        let out = Command::new("git").args(args).current_dir(root()).output().ok()?;
        out.status.success().then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    let now = SystemTime::now().duration_since(UNIX_EPOCH).expect("the clock is after 1970");
    record::Conditions {
        version: env!("CARGO_PKG_VERSION").to_owned(),
        commit: git(&["rev-parse", "HEAD"]),
        dirty: git(&["status", "--porcelain"]).map(|s| !s.is_empty()),
        timestamp: record::rfc3339(now.as_secs()),
        machine: machine(),
        // `cargo bench` builds the bench profile, and main has already refused
        // a build with debug assertions.
        profile: "bench".to_owned(),
        debug_assertions: cfg!(debug_assertions),
        warmup_runs: WARMUP_RUNS,
        recorded_runs: RECORDED_RUNS,
    }
}

/// The machine, from `sysctlbyname` on macOS. Elsewhere only what the
/// standard library knows is filled in, and the rest is `null`.
fn machine() -> record::Machine {
    let cores = std::thread::available_parallelism().ok().map(|n| n.get() as u64);
    let mut m = record::Machine {
        cpu: None,
        cores,
        performance_cores: None,
        efficiency_cores: None,
        memory_bytes: None,
        os: std::env::consts::OS.to_owned(),
        os_version: None,
        arch: std::env::consts::ARCH.to_owned(),
    };
    #[cfg(target_os = "macos")]
    {
        m.cpu = sysctl::string("machdep.cpu.brand_string");
        m.cores = sysctl::number("hw.ncpu").or(m.cores);
        m.performance_cores = sysctl::number("hw.perflevel0.physicalcpu");
        m.efficiency_cores = sysctl::number("hw.perflevel1.physicalcpu");
        m.memory_bytes = sysctl::number("hw.memsize");
        m.os_version = sysctl::string("kern.osproductversion");
    }
    m
}

#[cfg(target_os = "macos")]
mod sysctl {
    use std::ffi::{c_void, CString};
    use std::ptr::null_mut;

    /// The raw value of a `sysctl` entry, or `None` if the system has no such entry.
    fn raw(name: &str) -> Option<Vec<u8>> {
        let name = CString::new(name).ok()?;
        let mut len = 0usize;
        // SAFETY: with a null buffer, sysctlbyname only writes the value's size
        // into `len`; the second call writes at most `len` bytes into a buffer
        // of that size and updates `len` to what it wrote.
        unsafe {
            if libc::sysctlbyname(name.as_ptr(), null_mut(), &mut len, null_mut(), 0) != 0 {
                return None;
            }
            let mut buf = vec![0u8; len];
            if libc::sysctlbyname(name.as_ptr(), buf.as_mut_ptr().cast::<c_void>(), &mut len, null_mut(), 0) != 0 {
                return None;
            }
            buf.truncate(len);
            Some(buf)
        }
    }

    pub fn string(name: &str) -> Option<String> {
        let mut bytes = raw(name)?;
        while bytes.last() == Some(&0) {
            bytes.pop();
        }
        String::from_utf8(bytes).ok()
    }

    /// A 4- or 8-byte little-endian integer entry.
    pub fn number(name: &str) -> Option<u64> {
        let bytes = raw(name)?;
        match bytes.len() {
            4 => Some(u64::from(u32::from_le_bytes(bytes.try_into().ok()?))),
            8 => Some(u64::from_le_bytes(bytes.try_into().ok()?)),
            _ => None,
        }
    }
}

/// One metric over the recorded runs.
fn summarize(runs: &[Value], metric: &str) -> stats::Spread {
    let values: Vec<f64> = runs
        .iter()
        .map(|r| r[metric].as_f64().unwrap_or_else(|| fail(&format!("a run did not report {metric}: {r}"))))
        .collect();
    stats::spread(&values).unwrap_or_else(|| fail(&format!("{metric}: no usable values")))
}

/// One run in a fresh process; its measurements, or the benchmark stops.
fn run_child(id: &str) -> Value {
    let exe = std::env::current_exe().unwrap_or_else(|e| fail(&format!("cannot find this program: {e}")));
    let out = Command::new(exe)
        .args(["--child", id])
        .output()
        .unwrap_or_else(|e| fail(&format!("cannot start a child process: {e}")));
    if !out.status.success() {
        fail(&format!("workload {id}: the child failed ({}):\n{}", out.status, String::from_utf8_lossy(&out.stderr)));
    }
    let text = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str(text.trim())
        .unwrap_or_else(|e| fail(&format!("workload {id}: unreadable child output ({e}): {text:?}")))
}

/// In the child: load, generate, and print the measurements as one JSON line.
fn child(w: &Workload) {
    let dir = model_dir();
    let start = Instant::now();
    let model = Model::load(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let tokenizer = Tokenizer::load(&dir).unwrap_or_else(|e| fail(&e.to_string()));
    let load = start.elapsed();

    let generation = generate(&model, &tokenizer, &w.text, w.new_tokens, &mut |_| {}).unwrap_or_else(|e| fail(&e.to_string()));
    let stats = generation.stats;
    if stats.generated_tokens != w.new_tokens {
        fail(&format!(
            "workload {}: generation stopped at the end-of-sequence token after {} of {} tokens",
            w.id, stats.generated_tokens, w.new_tokens
        ));
    }
    let first = stats.time_to_first_token.expect("tokens were generated");
    println!(
        "{}",
        json!({
            "load_ms": load.as_secs_f64() * 1000.0,
            "prompt_tokens": stats.prompt_tokens,
            "generated_tokens": stats.generated_tokens,
            "time_to_first_token_ms": first.as_secs_f64() * 1000.0,
            "prompt_tokens_per_s": stats.prompt_tokens_per_second(),
            "generation_tokens_per_s": stats.generation_tokens_per_second(),
            "peak_rss_bytes": peak_rss_bytes(),
        })
    );
}

/// This process's peak resident memory in bytes, or `None` where the unit of
/// `ru_maxrss` is not known. It is bytes on macOS and kibibytes on Linux.
fn peak_rss_bytes() -> Option<u64> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: getrusage writes a whole `rusage` into the pointer it is given,
    // which points at one; it is read only when the call reports success.
    let usage = unsafe {
        if libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) != 0 {
            return None;
        }
        usage.assume_init()
    };
    let max = u64::try_from(usage.ru_maxrss).ok()?;
    if cfg!(target_os = "macos") {
        Some(max)
    } else if cfg!(target_os = "linux") {
        Some(max * 1024)
    } else {
        None
    }
}
