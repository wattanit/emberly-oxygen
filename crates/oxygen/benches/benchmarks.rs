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
//!
//! The sampling group (Spec §12) times one sampling step on a vocabulary-sized
//! logit vector for each control alone and for all four, in this process, one
//! line per setting with `"group": "sampling"`. It measures the sampling code
//! only: the two workloads still run greedy, so their rows stay comparable with
//! every earlier release's (FR-9, FR-29). `cargo bench -- --sampling-only
//! --dry-run` runs just this group and prints its rows.
//!
//! The prefill group (FR-42) times the prompt alone, from its first token to
//! the logits of its last position, at five lengths, in this process: one `step`
//! per token (the way releases before 0.4.0 processed a prompt) and one
//! `prefill`, one line per path and length with `"group": "prefill"`. The two
//! are checked to agree on the last logits first. `cargo bench -- --prefill-only
//! --dry-run` runs just this group and prints its rows.
//!
//! The thread sweep (FR-45) runs the released engine at 1, 2, 4, 8, 12 and 16
//! threads (counts above the machine's cores are left out), in this process,
//! and times the prompt alone at 128 and 644 tokens and the generation of 128
//! tokens after prompts of 32 and 512 tokens, one line per count, phase and
//! length with `"group": "threads"`. Every count must give the same last
//! logits as one thread before anything is timed. It also prints the thread
//! count the sweep picks as the default (FR-44). `cargo bench -- --threads-only
//! --dry-run` runs just this group. Every row of every group carries `threads`,
//! the count that produced it (FR-9, FR-44).

#[path = "support/batched.rs"]
mod batched;
#[path = "support/prefill.rs"]
mod prefill;
#[path = "support/threads.rs"]
mod threads;

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

use emberly_oxygen::{generate, kernels, Logits, Model, Sampling, Tensor, Tokenizer};
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
    if std::env::args().any(|a| a == "--sampling-only") {
        write_rows(&sampling_rows(&conditions), dry_run);
        return;
    }
    if std::env::args().any(|a| a == "--prefill-only") {
        write_rows(&prefill_rows(&conditions), dry_run);
        return;
    }
    if std::env::args().any(|a| a == "--threads-only") {
        write_rows(&thread_rows(&conditions), dry_run);
        return;
    }
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
        // The count the children ran at (the default), as they reported it.
        let used = recorded[0]["threads"].as_u64().unwrap_or_else(|| fail("a child did not report its thread count")) as usize;
        rows.push(record::row(&conditions, threads::with_threads(fields, used), &results));
    }
    rows.extend(kernel_rows(&conditions));
    rows.extend(batched_kernel_rows(&conditions));
    rows.extend(sampling_rows(&conditions));
    rows.extend(prefill_rows(&conditions));
    rows.extend(thread_rows(&conditions));
    write_rows(&rows, dry_run);
}

/// The prefill group (FR-42): the time to process a prompt of each length, up to
/// the logits of its last position, by one `step` per token and by one `prefill`,
/// in this process. Each path runs once to warm up and then `RECORDED_RUNS`
/// times; the row's metric is the prompt time in milliseconds. Before anything
/// is timed the two paths must agree on the last logits (FR-41), so a number is
/// never recorded for a path that computes something else.
fn prefill_rows(conditions: &record::Conditions) -> Vec<Value> {
    let model = Model::load(&model_dir()).unwrap_or_else(|e| fail(&e.to_string()));
    let tokenizer = Tokenizer::load(&model_dir()).unwrap_or_else(|e| fail(&e.to_string()));
    let long = workloads().into_iter().max_by_key(|w| w.prompt_tokens).unwrap_or_else(|| fail("prompts.json lists no workload"));
    let base = tokenizer.encode(&long.text).unwrap_or_else(|e| fail(&e.to_string()));

    let by_steps = |ids: &[u32]| {
        let mut session = model.session(false);
        let mut last = Vec::new();
        for &id in ids {
            last = session.step(id).unwrap_or_else(|e| fail(&e.to_string()));
        }
        last
    };
    let batched = |ids: &[u32]| model.session(false).prefill(ids, Logits::Last).unwrap_or_else(|e| fail(&e.to_string()));

    println!("prefill: the prompt alone, median milliseconds over {RECORDED_RUNS} runs");
    let mut rows = Vec::new();
    for n in prefill::LENGTHS {
        let ids = prefill::prompt_ids(&base, n);
        let gap = prefill::logits_gap(&by_steps(&ids), &batched(&ids));
        if gap > 1e-5 {
            fail(&format!("prefill: at {n} tokens the two paths differ by {gap:e} in the last logits; nothing is recorded"));
        }
        for (path, run) in [("one_token", &by_steps as &dyn Fn(&[u32]) -> Vec<f32>), ("batched", &batched)] {
            black_box(run(&ids));
            let ms: Vec<f64> = (0..RECORDED_RUNS)
                .map(|_| {
                    let start = Instant::now();
                    black_box(run(black_box(&ids)));
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            let spread = stats::spread(&ms).unwrap_or_else(|| fail("prefill: no usable timings"));
            println!("  {n:>4} tokens, {path:<10} {:>9.1} ms  (min {:.1}, max {:.1})", spread.median, spread.min, spread.max);
            rows.push(record::row(conditions, threads::with_threads(prefill::fields(path, n), model.threads()), &[("prompt_ms", spread)]));
        }
    }
    rows
}

/// The thread sweep (FR-45, Spec §12): the prompt alone at each length and the
/// generation after each context, at every count the machine can run. Each
/// timing runs once to warm up and then `RECORDED_RUNS` times. Before anything
/// is timed, every count must give exactly the last logits one thread gives
/// (FR-43): a speed is never recorded for an engine that computes something else.
fn thread_rows(conditions: &record::Conditions) -> Vec<Value> {
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let tokenizer = Tokenizer::load(&model_dir()).unwrap_or_else(|e| fail(&e.to_string()));
    let long = workloads().into_iter().max_by_key(|w| w.prompt_tokens).unwrap_or_else(|| fail("prompts.json lists no workload"));
    let base = tokenizer.encode(&long.text).unwrap_or_else(|e| fail(&e.to_string()));
    let lengths: Vec<usize> = threads::PROMPT_LENGTHS.iter().chain(&threads::DECODE_CONTEXTS).copied().collect();
    let argmax = |logits: &[f32]| logits.iter().enumerate().fold((0, f32::MIN), |b, (i, &v)| if v > b.1 { (i, v) } else { b }).0 as u32;

    println!("threads: median over {RECORDED_RUNS} runs, counts {:?} of {cores} cores", threads::counts_for(cores));
    let mut baseline: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut rows = Vec::new();
    let (mut decode_at_512, mut prompt_at_644) = (Vec::new(), Vec::new());
    for count in threads::counts_for(cores) {
        let model = Model::load_with_threads(&model_dir(), count).unwrap_or_else(|e| fail(&e.to_string()));
        // Identical logits first: the prompt's last logits at every length, and one step after it.
        for &n in &lengths {
            let ids = prefill::prompt_ids(&base, n);
            let mut session = model.session(false);
            let mut last = session.prefill(&ids, Logits::Last).unwrap_or_else(|e| fail(&e.to_string()));
            last.extend(session.step(argmax(&last)).unwrap_or_else(|e| fail(&e.to_string())));
            match baseline.iter().find(|(m, _)| *m == n) {
                None => baseline.push((n, last)),
                Some((_, one)) => {
                    if !threads::logits_identical(one, &last) {
                        fail(&format!("threads: {count} threads give different logits from one thread at {n} tokens; nothing is recorded"));
                    }
                }
            }
        }
        let time = |mut run: Box<dyn FnMut() -> f64 + '_>| -> stats::Spread {
            run();
            let values: Vec<f64> = (0..RECORDED_RUNS).map(|_| run()).collect();
            stats::spread(&values).unwrap_or_else(|| fail("threads: no usable timings"))
        };
        for n in threads::PROMPT_LENGTHS {
            let ids = prefill::prompt_ids(&base, n);
            let spread = time(Box::new(|| {
                let start = Instant::now();
                black_box(model.session(false).prefill(black_box(&ids), Logits::Last).unwrap_or_else(|e| fail(&e.to_string())));
                start.elapsed().as_secs_f64() * 1000.0
            }));
            println!("  {count:>2} threads, prompt {n:>4} tokens {:>9.1} ms  (min {:.1}, max {:.1})", spread.median, spread.min, spread.max);
            if n == 644 {
                prompt_at_644.push(spread.median);
            }
            rows.push(record::row(conditions, threads::fields(count, "prompt", n), &[("prompt_ms", spread)]));
        }
        for n in threads::DECODE_CONTEXTS {
            let ids = prefill::prompt_ids(&base, n);
            let spread = time(Box::new(|| {
                let mut session = model.session(false);
                let mut next = argmax(&session.prefill(&ids, Logits::Last).unwrap_or_else(|e| fail(&e.to_string())));
                let start = Instant::now();
                for _ in 0..threads::DECODE_TOKENS {
                    next = argmax(&session.step(black_box(next)).unwrap_or_else(|e| fail(&e.to_string())));
                }
                black_box(next);
                start.elapsed().as_secs_f64() * 1000.0 / threads::DECODE_TOKENS as f64
            }));
            println!("  {count:>2} threads, decode after {n:>3} tokens {:>7.3} ms/token ({:.1} tokens/s; min {:.3}, max {:.3})", spread.median, 1000.0 / spread.median, spread.min, spread.max);
            if n == 512 {
                decode_at_512.push((count, spread.median));
            }
            rows.push(record::row(conditions, threads::fields(count, "decode", n), &[("ms_per_token", spread)]));
        }
    }
    let candidates: Vec<(usize, f64, f64)> = decode_at_512.iter().zip(&prompt_at_644).map(|(&(n, decode), &prompt)| (n, decode, prompt)).collect();
    println!("threads: the default by the sweep's rule (highest decode speed at 512 tokens; ties within 2% to the faster prompt at 644 tokens) is {}", threads::choose_default(&candidates));
    rows
}

/// The batched kernel rows (FR-48, Spec §12): at the three shapes of the batched
/// products with 128 input rows, single-threaded and in this process, the 0.4.1
/// `matmul`, the blocked kernel and `matrixmultiply`'s sgemm, as GFLOP/s. The
/// blocked kernel must equal its definition written out plainly before anything
/// is timed, and the other two must agree with it within rounding, so a rate is
/// never recorded for a kernel that computes something else.
fn batched_kernel_rows(conditions: &record::Conditions) -> Vec<Value> {
    println!("kernels: matrix-matrix product, {} inputs, median GFLOP/s over {RECORDED_RUNS} runs (single-threaded)", batched::INPUTS);
    println!("  {:>12} {:>9} {:>9} {:>15}", "shape", "matmul", "blocked", "matrixmultiply");
    let mut rows = Vec::new();
    let n = batched::INPUTS;
    for (seed, (m, k, role)) in batched::SHAPES.into_iter().enumerate() {
        let weight = Tensor::new(vec![m, k], pseudo_random(m * k, 2 * seed as u64 + 11)).expect("shape matches data");
        let input = pseudo_random(n * k, 2 * seed as u64 + 12);
        let mut old = vec![0f32; n * m];
        let mut blocked = vec![0f32; n * m];
        let mut baseline = vec![0f32; n * m];
        let sgemm = |out: &mut [f32]| {
            // SAFETY: out (n x m) = input (n x k) * weight^T (k x m), the weight being row-major m x k:
            // every pointer covers exactly the elements those strides reach.
            unsafe {
                matrixmultiply::sgemm(
                    n, k, m, 1.0,
                    input.as_ptr(), k as isize, 1,
                    weight.data().as_ptr(), 1, k as isize,
                    0.0,
                    out.as_mut_ptr(), m as isize, 1,
                );
            }
        };
        kernels::matmul_blocked(&weight, &input, n, &mut blocked);
        if blocked != batched::chunked_reference(weight.data(), m, k, &input, n) {
            fail(&format!("batched kernels {m}x{k}: the blocked kernel differs from its definition; nothing is recorded"));
        }
        kernels::matmul(&weight, &input, n, &mut old);
        sgemm(&mut baseline);
        for (what, other) in [("matmul", &old), ("matrixmultiply", &baseline)] {
            for (i, (a, b)) in blocked.iter().zip(other.iter()).enumerate() {
                if (a - b).abs() > 1e-3 * (1.0 + b.abs()) {
                    fail(&format!("batched kernels {m}x{k}: element {i} is {a} in the blocked kernel and {b} in {what}"));
                }
            }
        }

        let flops = batched::flops(m, k, n);
        let mut measured: Vec<(String, stats::Spread)> = Vec::new();
        let mut median_gflops = Vec::new();
        let mut record_kernel = |name: &str, ns: &[f64]| {
            let gflops: Vec<f64> = ns.iter().map(|t| flops / t).collect();
            let [ns_name, gflops_name] = record::kernel_metrics(name);
            let spread = |v: &[f64]| stats::spread(v).unwrap_or_else(|| fail("batched kernels: no usable timings"));
            median_gflops.push(spread(&gflops).median);
            measured.push((ns_name, spread(ns)));
            measured.push((gflops_name, spread(&gflops)));
        };
        record_kernel("matmul", &time_calls(|| kernels::matmul(&weight, black_box(&input), n, black_box(&mut old))));
        record_kernel("blocked", &time_calls(|| kernels::matmul_blocked(&weight, black_box(&input), n, black_box(&mut blocked))));
        record_kernel("matrixmultiply", &time_calls(|| sgemm(black_box(&mut baseline))));
        println!("  {:>12} {:>9.2} {:>9.2} {:>15.2}", format!("{m}x{k}"), median_gflops[0], median_gflops[1], median_gflops[2]);

        let results: Vec<(&str, stats::Spread)> = measured.iter().map(|(name, spread)| (name.as_str(), *spread)).collect();
        rows.push(record::row(conditions, batched::fields(m, k, role), &results));
    }
    rows
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

/// The kernel group (HC-2, FR-25): at every matrix size in the model, the
/// engine's own `matvec` against `matrixmultiply`'s `sgemm` with one column and
/// against the 0.1.0 index-order loop and the partial-sum kernel at widths 4,
/// 8, 16 and 32, all single-threaded and on the same inputs. A row keeps the
/// four field names 0.1.0's rows have (`in_house_*` is the engine's kernel,
/// `matrixmultiply_*` the library) and adds two per variant (FR-9).
fn kernel_rows(conditions: &record::Conditions) -> Vec<Value> {
    type Kernel = fn(&Tensor, &[f32], &mut [f32]);
    let variants: [(&str, Kernel); 5] = [
        ("index_order", kernels::matvec_index_order),
        ("lanes_4", kernels::matvec_lanes::<4>),
        ("lanes_8", kernels::matvec_lanes::<8>),
        ("lanes_16", kernels::matvec_lanes::<16>),
        ("lanes_32", kernels::matvec_lanes::<32>),
    ];
    assert!(variants.iter().map(|v| v.0).eq(record::KERNEL_VARIANTS), "variants and their record names must agree");

    println!("kernels: matrix-vector product, median GFLOP/s over 5 runs (single-threaded)");
    print!("  {:>12} {:>8} {:>8}", "shape", "engine", "matmul");
    for (name, _) in &variants {
        print!(" {name:>11}");
    }
    println!();
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

        // A baseline or a variant that computes something else is not timed.
        kernels::matvec(&weight, &input, &mut ours);
        let agrees = |what: &str, other: &[f32]| {
            for (i, (a, b)) in ours.iter().zip(other).enumerate() {
                if (a - b).abs() > 1e-4 * (1.0 + b.abs()) {
                    fail(&format!("kernels {m}x{k}: row {i} is {a} in the engine and {b} in {what}"));
                }
            }
        };
        sgemm(&mut baseline);
        agrees("matrixmultiply", &baseline);
        let mut outputs: Vec<Vec<f32>> = Vec::new();
        for (name, kernel) in &variants {
            let mut out = vec![0f32; m];
            kernel(&weight, &input, &mut out);
            agrees(name, &out);
            outputs.push(out);
        }

        let flops = 2.0 * (m * k) as f64;
        let spread = |values: &[f64]| stats::spread(values).unwrap_or_else(|| fail("kernels: no usable timings"));
        let mut measured: Vec<(String, stats::Spread)> = Vec::new();
        let mut median_gflops = Vec::new();
        let mut record_kernel = |name: &str, ns: &[f64]| {
            let gflops: Vec<f64> = ns.iter().map(|t| flops / t).collect();
            let [ns_name, gflops_name] = record::kernel_metrics(name);
            median_gflops.push(spread(&gflops).median);
            measured.push((ns_name, spread(ns)));
            measured.push((gflops_name, spread(&gflops)));
        };
        record_kernel("in_house", &time_calls(|| kernels::matvec(&weight, black_box(&input), black_box(&mut ours))));
        record_kernel("matrixmultiply", &time_calls(|| sgemm(black_box(&mut baseline))));
        for ((name, kernel), out) in variants.iter().zip(outputs.iter_mut()) {
            record_kernel(name, &time_calls(|| kernel(&weight, black_box(&input), black_box(&mut out[..]))));
        }

        print!("  {:>12}", format!("{m}x{k}"));
        for (n, gflops) in median_gflops.iter().enumerate() {
            print!(" {gflops:>w$.2}", w = if n < 2 { 8 } else { 11 });
        }
        println!();

        let results: Vec<(&str, stats::Spread)> = measured.iter().map(|(n, s)| (n.as_str(), *s)).collect();
        let fields = json!({
            "group": "kernels",
            "threads": 1,
            "kernel": "matvec",
            "rows": m,
            "cols": k,
            "role": role,
            "baseline": "matrixmultiply sgemm with one column, single-threaded",
            "variants": record::KERNEL_VARIANTS,
        });
        rows.push(record::row(conditions, fields, &results));
    }
    rows
}

/// The size of SmolLM2-135M's vocabulary: the length of one step's logits.
const VOCAB: usize = 49152;

/// One sampling step (the controls, the softmax and the draw) for each setting of
/// the sampling reference's kind, on logits spread over a realistic range and a
/// 32-token context (Spec §12). The forward step it follows takes about 8.5 ms
/// (the short workload's generation throughput), so the release note can state
/// what a sampled step adds.
fn sampling_rows(conditions: &record::Conditions) -> Vec<Value> {
    let logits: Vec<f32> = pseudo_random(VOCAB, 77).iter().map(|v| v * 6.0).collect();
    let context: Vec<u32> = (0..32).map(|i| (i * 1531 % VOCAB) as u32).collect();
    let neutral = Sampling { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: Some(1) };
    let settings: [(&str, Sampling); 6] = [
        ("neutral", neutral),
        ("temperature-0.7", Sampling { temperature: 0.7, ..neutral }),
        ("top-k-40", Sampling { top_k: 40, ..neutral }),
        ("top-p-0.9", Sampling { top_p: 0.9, ..neutral }),
        ("penalty-1.3", Sampling { repetition_penalty: 1.3, ..neutral }),
        ("combined", Sampling { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.3, seed: Some(1) }),
    ];
    println!("sampling: one step on {VOCAB} logits, median microseconds over {RECORDED_RUNS} runs");
    let mut rows = Vec::new();
    for (name, sampling) in settings {
        let ns = time_calls(|| {
            black_box(kernels::sample_token(black_box(&logits), black_box(&context), &sampling, 1));
        });
        let spread = stats::spread(&ns).unwrap_or_else(|| fail("sampling: no usable timings"));
        println!("  {name:<16} {:>10.1} us  (min {:.1}, max {:.1})", spread.median / 1000.0, spread.min / 1000.0, spread.max / 1000.0);
        let fields = json!({
            "group": "sampling",
            "threads": 1,
            "setting": name,
            "vocab": VOCAB,
            "context_tokens": context.len(),
        });
        rows.push(record::row(conditions, fields, &[("ns_per_step", spread)]));
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
            "threads": model.threads(),
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
