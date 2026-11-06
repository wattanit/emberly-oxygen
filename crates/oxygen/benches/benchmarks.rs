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

#[path = "support/stats.rs"]
mod stats;

use std::path::PathBuf;
use std::process::{exit, Command};
use std::time::Instant;

use emberly_oxygen::{generate, Model, Tokenizer};
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

fn model_dir() -> PathBuf {
    let dir = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/smollm2-135m"));
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

/// Runs every workload in child processes and reports each run.
fn parent() {
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
        for metric in METRICS {
            let s = summarize(&recorded, metric);
            let places = if metric.ends_with("_bytes") { 0 } else { 3 };
            println!("  {metric:<24} {:>14.places$} {:>14.places$} {:>14.places$}", s.median, s.min, s.max);
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
