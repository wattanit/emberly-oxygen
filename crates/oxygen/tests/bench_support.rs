//! Unit tests of the benchmark harness's pure helpers, which live in
//! `benches/support/`: `cargo test` does not run a `harness = false` bench's
//! own tests, so they are included here by path.

#[path = "../benches/support/stats.rs"]
mod stats;

#[path = "../benches/support/record.rs"]
mod record;

#[path = "../benches/support/prefill.rs"]
mod prefill;

#[path = "../benches/support/threads.rs"]
mod threads;

#[path = "../benches/support/batched.rs"]
mod batched;

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

// FR-9, FR-25: a kernel row keeps the four field names release 0.1.0's rows
// have, so the series stays comparable, and gains two per variant.
#[test]
fn kernel_row_keeps_the_0_1_0_fields_and_adds_the_variants() {
    let spread = Spread { median: 2.0, min: 1.0, max: 3.0 };
    let mut names: Vec<String> = Vec::new();
    for kernel in ["in_house", "matrixmultiply"].into_iter().chain(record::KERNEL_VARIANTS) {
        names.extend(record::kernel_metrics(kernel));
    }
    let results: Vec<(&str, Spread)> = names.iter().map(|n| (n.as_str(), spread)).collect();
    let line = row(&conditions(), json!({"group": "kernels", "kernel": "matvec"}), &results);
    let present = line["results"].as_object().expect("results is an object");
    for old in ["in_house_ns_per_call", "in_house_gflops", "matrixmultiply_ns_per_call", "matrixmultiply_gflops"] {
        assert!(present.contains_key(old), "0.1.0's field {old} is missing");
    }
    for variant in record::KERNEL_VARIANTS {
        for name in record::kernel_metrics(variant) {
            assert!(present.contains_key(&name), "the variant field {name} is missing");
        }
    }
    assert_eq!(present.len(), 4 + 2 * record::KERNEL_VARIANTS.len());
}

// FR-42, Spec §12: the prefill sweep's prompts are the first `n` ids of one fixed list, so every run
// of a length feeds the same tokens; a list shorter than `n` is repeated from its start, which
// changes nothing the timing depends on (it depends on the length, not on the text).
#[test]
fn fr42_the_sweeps_prompts_are_the_first_n_ids_of_one_list() {
    let base = [10u32, 11, 12, 13];
    assert_eq!(prefill::prompt_ids(&base, 3), [10, 11, 12]);
    assert_eq!(prefill::prompt_ids(&base, 4), [10, 11, 12, 13]);
    assert_eq!(prefill::prompt_ids(&base, 6), [10, 11, 12, 13, 10, 11]);
    assert_eq!(prefill::prompt_ids(&base, 9), [10, 11, 12, 13, 10, 11, 12, 13, 10]);
    assert!(prefill::prompt_ids(&base, 0).is_empty());
    // The same call gives the same ids, whatever else has run.
    assert_eq!(prefill::prompt_ids(&base, 7), prefill::prompt_ids(&base, 7));
}

// FR-42, Spec §12: the lengths of the sweep include the first and last turn of release 0.3.0's ten-turn
// chat (67 and 644 tokens), so chat time to first token is read from the sweep's rows.
#[test]
fn fr42_the_sweep_includes_the_chat_lengths() {
    assert_eq!(prefill::LENGTHS, [32, 67, 128, 512, 644]);
}

// FR-41, Spec §12: the two paths are checked to agree on the last logits before anything is timed;
// the gap is the largest absolute difference, measured in f64, and a different length is infinite.
#[test]
fn fr41_the_gap_between_two_logit_vectors_is_the_largest_absolute_difference() {
    assert_eq!(prefill::logits_gap(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]), 0.0);
    assert_eq!(prefill::logits_gap(&[1.0, 2.0, 3.0], &[1.0, 2.5, 2.75]), 0.5);
    assert_eq!(prefill::logits_gap(&[1.0, 2.0], &[1.0]), f64::INFINITY);
}

// FR-42, FR-9: a prefill row names its path and prompt length and carries the conditions and a timing.
#[test]
fn fr42_a_prefill_row_names_its_path_and_length() {
    let spread = Spread { median: 620.0, min: 610.0, max: 650.0 };
    let line = row(&conditions(), prefill::fields("batched", 512), &[("prompt_ms", spread)]);
    assert_eq!(line["group"], "prefill");
    assert_eq!(line["path"], "batched");
    assert_eq!(line["prompt_tokens"], 512);
    assert_eq!(line["results"]["prompt_ms"]["median"], 620.0);
    assert_eq!(line["version"], "0.1.0");
}

// FR-45, Spec §12: the sweep's counts are 1, 2, 4, 8, 12 and 16, and a count above the machine's
// available cores is left out; a machine with fewer cores than the largest keeps the rest.
#[test]
fn fr45_the_sweep_leaves_out_counts_above_the_cores() {
    assert_eq!(threads::COUNTS, [1, 2, 4, 8, 12, 16]);
    assert_eq!(threads::counts_for(16), [1, 2, 4, 8, 12, 16]);
    assert_eq!(threads::counts_for(12), [1, 2, 4, 8, 12]);
    assert_eq!(threads::counts_for(6), [1, 2, 4]);
    assert_eq!(threads::counts_for(1), [1]);
    assert_eq!(threads::counts_for(64), [1, 2, 4, 8, 12, 16]);
}

// FR-45, Spec §12: prompt time at 128 and 644 tokens, decode after prompts of 32 and 512 tokens.
#[test]
fn fr45_the_sweeps_lengths_and_decode_tokens() {
    assert_eq!(threads::PROMPT_LENGTHS, [128, 644]);
    assert_eq!(threads::DECODE_CONTEXTS, [32, 512]);
    assert_eq!(threads::DECODE_TOKENS, 128);
}

// FR-44, FR-45, Spec §12: the default is the count with the highest decode speed at 512 tokens; counts
// within 2% of the best decode time are ties, and a tie goes to the count with the faster prompt at 644
// tokens. Rows are `(count, decode ms per token at 512, prompt ms at 644)`; the numbers are illustrative.
#[test]
fn fr44_the_default_is_the_fastest_decode_with_ties_to_the_faster_prompt() {
    // A clear decode winner, whatever its prompt time.
    assert_eq!(threads::choose_default(&[(1, 12.1, 4900.0), (2, 8.5, 3400.0), (4, 7.0, 2100.0), (12, 7.6, 880.0)]), 4);
    // 4 and 8 are within 2% in decode (7.34 * 1.02 > 7.275 and the reverse); 8 has the faster prompt.
    assert_eq!(threads::choose_default(&[(1, 12.1, 4900.0), (4, 7.340, 2131.0), (8, 7.275, 1034.0), (12, 7.451, 884.0)]), 8);
    // 12 is outside the 2% band (7.451 > 7.275 * 1.02), so its faster prompt does not count.
    assert_eq!(threads::choose_default(&[(8, 7.275, 1034.0), (12, 7.451, 884.0)]), 8);
    // Inside the band the faster prompt wins even at a higher count.
    assert_eq!(threads::choose_default(&[(8, 7.275, 1034.0), (12, 7.40, 884.0)]), 12);
    // Equal prompt times go to the smaller count; the order of the rows does not matter.
    assert_eq!(threads::choose_default(&[(8, 7.0, 900.0), (4, 7.1, 900.0)]), 4);
    assert_eq!(threads::choose_default(&[(12, 7.40, 884.0), (8, 7.275, 1034.0)]), 12);
    // One count only.
    assert_eq!(threads::choose_default(&[(1, 8.7, 4900.0)]), 1);
}

// FR-43, Spec §12: the counts are checked to give identical last logits before any timing: the gap
// must be exactly 0, because the engine's output does not depend on the count.
#[test]
fn fr43_the_sweep_accepts_only_identical_logits() {
    assert!(threads::logits_identical(&[1.0, 2.0], &[1.0, 2.0]));
    assert!(!threads::logits_identical(&[1.0, 2.0], &[1.0, 2.000001]));
    assert!(!threads::logits_identical(&[1.0], &[1.0, 2.0]));
}

// FR-44, FR-45, FR-9: a thread-sweep row names its count, phase and context length; the group and
// the metric are in it. Every other group's row also carries its count, so a speed is never
// recorded without the count that produced it.
#[test]
fn fr44_fr45_a_thread_row_names_its_count_phase_and_context() {
    let spread = Spread { median: 5.1, min: 5.0, max: 5.3 };
    let line = row(&conditions(), threads::fields(8, "decode", 512), &[("ms_per_token", spread)]);
    assert_eq!(line["group"], "threads");
    assert_eq!(line["threads"], 8);
    assert_eq!(line["phase"], "decode");
    assert_eq!(line["context_tokens"], 512);
    assert_eq!(line["results"]["ms_per_token"]["median"], 5.1);
    assert_eq!(threads::with_threads(json!({"group": "kernels"}), 1)["threads"], 1);
    assert_eq!(threads::with_threads(json!({"group": "kernels"}), 1)["group"], "kernels");
}

// FR-48, Spec §12: the batched kernel rows use 128 input rows at the three shapes of the batched
// products, and count 2 * rows * cols * inputs floating-point operations.
#[test]
fn fr48_the_batched_kernel_rows_use_the_three_shapes_and_128_inputs() {
    assert_eq!(batched::INPUTS, 128);
    let shapes: Vec<(usize, usize)> = batched::SHAPES.iter().map(|s| (s.0, s.1)).collect();
    assert_eq!(shapes, [(1536, 576), (576, 1536), (576, 576)]);
    assert_eq!(batched::flops(2, 3, 4), 48.0);
    assert_eq!(batched::flops(1536, 576, 128), 2.0 * 1536.0 * 576.0 * 128.0);
}

// FR-46, FR-48, Spec §12: the checking sum the blocked kernel is compared with before it is timed is
// the chunked fused sum: hand-worked small cases, and a length that crosses a chunk boundary.
#[test]
fn fr46_the_chunked_reference_is_the_chunked_fused_sum() {
    // [[1, 2, 3], [4, 5, 6]] times the inputs [1, 1, 1] and [2, 0, 1]: [[6, 15], [5, 14]], laid out [input][row].
    let weight = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let input = [1.0, 1.0, 1.0, 2.0, 0.0, 1.0];
    assert_eq!(batched::chunked_reference(&weight, 2, 3, &input, 2), [6.0, 15.0, 5.0, 14.0]);
    // 65 products of 1: a chunk of 64 and a chunk of 1, 65 in all.
    let ones = vec![1.0f32; 65];
    assert_eq!(batched::chunked_reference(&ones, 1, 65, &ones, 1), [65.0]);
    // The chunk sums are added after each chunk is complete: 2^24 + 1 + 1 in one chunk loses both
    // ones when added one by one to a large total, but a chunk's own sum of ones is exact and
    // keeps them. Columns: 1 large value, then 64 ones in the next chunk.
    let mut w = vec![1.0f32; 65];
    let mut x = vec![1.0f32; 65];
    w[0] = 16_777_216.0;
    x[0] = 1.0;
    // chunk 1: 16777216 + 63 ones (the 63 ones are lost one at a time: each add of 1 rounds away); chunk 2: 1.
    let got = batched::chunked_reference(&w, 1, 65, &x, 1)[0];
    let mut chunk1 = 16_777_216.0f32;
    for _ in 0..63 {
        chunk1 = 1.0f32.mul_add(1.0, chunk1);
    }
    assert_eq!(got, chunk1 + 1.0);
}

// FR-48, FR-9: a batched kernel row names its shape, the input count and the thread count, and carries
// each kernel's two metrics.
#[test]
fn fr48_a_batched_kernel_row_names_its_shape_and_kernels() {
    let spread = Spread { median: 100.0, min: 99.0, max: 101.0 };
    let results: Vec<(String, Spread)> = batched::KERNELS.iter().flat_map(|k| record::kernel_metrics(k)).map(|name| (name, spread)).collect();
    let results: Vec<(&str, Spread)> = results.iter().map(|(n, s)| (n.as_str(), *s)).collect();
    let line = row(&conditions(), batched::fields(1536, 576, "MLP gate and up"), &results);
    assert_eq!(line["group"], "kernels");
    assert_eq!(line["kernel"], "matmul");
    assert_eq!((line["rows"].as_u64(), line["cols"].as_u64(), line["inputs"].as_u64(), line["threads"].as_u64()), (Some(1536), Some(576), Some(128), Some(1)));
    for kernel in ["matmul", "blocked", "matrixmultiply"] {
        assert_eq!(line["results"][format!("{kernel}_gflops")]["median"], 100.0, "{kernel}");
    }
    assert_eq!(batched::KERNELS, ["matmul", "blocked", "matrixmultiply"]);
}

// FR-41, FR-47, Spec §11, §12: the prefill sweep requires its two paths to agree on the last logits within
// the batched-against-one-token bound, 1e-3 since 0.4.2 (they sum in a different order, so they differ by
// about 2e-5 on the benchmark's prompts); a difference of defect size is refused.
#[test]
fn fr41_fr47_the_sweeps_agreement_bound_is_the_batched_against_one_token_bound() {
    assert_eq!(prefill::AGREEMENT_BOUND, 1e-3);
    assert!(prefill::agree(0.0));
    assert!(prefill::agree(1.67e-5), "the observed rounding difference passes");
    assert!(prefill::agree(1e-3), "the bound itself passes");
    assert!(!prefill::agree(2e-3));
    assert!(!prefill::agree(f64::INFINITY), "paths of different shapes never agree");
    assert!(!prefill::agree(f64::NAN));
}
