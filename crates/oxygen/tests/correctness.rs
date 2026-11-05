//! The correctness harness (Spec §11): compares the engine against every
//! stored expected output. Verifies FR-5 (full comparison), FR-13 to FR-15
//! (tokenizer: encoding, decoding and streaming), S-1 to S-4 and S-5 (a release
//! passes only if every prompt category passes).
//!
//! A failing run shows the table in its failure message; a passing run prints
//! it only with `cargo test -- --nocapture`.
//!
//! Each prompt's checks run in a thread of their own. This is test code only:
//! the engine stays single-threaded, each thread has its own session, and
//! results are collected in prompt order, so the table is the same as a
//! sequential run's (checked below).

mod common;

use std::thread;

use common::compare::greedy_check;
use common::fixtures::{discover, require_model_dir, workspace_root, Fixtures, Prompt};
use common::harness::{check_prompt, Check, CheckResult};
use common::report::render;
use emberly_oxygen::{generate, Model, Tokenizer};

#[test]
fn correctness_harness() {
    let results = check_all(|_| true, true);

    let report = render(&results);
    assert!(
        report.failed.is_empty(),
        "{} of {} categories failed (S-5)\n\n{}",
        report.failed.len(),
        distinct_categories(&results),
        report.text
    );
    println!("{}", report.text);
}

/// FR-5, S-5: running the prompts in parallel changes nothing in the table,
/// errors included. Checked on the three shortest prompts to keep it quick.
#[test]
fn parallel_table_equals_sequential_table() {
    let short = |p: &Prompt| ["english-short", "utf8-split-emoji", "english-medium"].contains(&p.id.as_str());
    let sequential = check_all(short, false);
    assert_eq!(sequential.len(), 3 * Check::ALL.len(), "three prompts, every check");
    let (sequential, parallel) = (render(&sequential).text, render(&check_all(short, true)).text);
    assert_eq!(parallel, sequential);
}

/// FR-6, S-4: a continuation shifted by one token fails at its first token,
/// and the table's failure line names the position and both tokens.
#[test]
fn a_shifted_continuation_names_the_first_diverging_position() {
    let root = workspace_root();
    let all = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));
    let fixtures = &all[0];
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == "english-short").unwrap();
    let (model, tokenizer) = load(&root, fixtures);

    let n = prompt.continuation_ids.len();
    let ids = generate(&model, &tokenizer, &prompt.text, n + 1, &mut |_| {}).unwrap();
    let gaps = &prompt.continuation_top2_gaps;
    assert!(!greedy_check(&ids[..n], &prompt.continuation_ids, gaps).is_failure(), "unshifted passes");

    let verdict = greedy_check(&ids[1..], &prompt.continuation_ids, gaps);
    let result = CheckResult {
        model_id: fixtures.manifest.model_id.clone(),
        prompt: prompt.id.clone(),
        category: prompt.category.clone(),
        check: Check::Greedy,
        verdict,
        max_error: None,
    };
    // The first index where the shifted sequence differs: not always 0, since a
    // continuation can repeat a token (english-short's starts with two newlines).
    let at = (0..n).find(|&i| ids[i + 1] != ids[i]).unwrap();
    let line = format!(
        "english-short / greedy: first diverges at position {at}: token {} at continuation index {at}, reference {}",
        ids[at + 1], ids[at]
    );
    let text = render(&[result]).text;
    assert!(text.contains(&line), "expected {line:?} in:\n{text}");
}

/// Runs every check on the prompts `keep` selects, one thread per prompt when
/// `parallel`, and returns the results in manifest order either way.
fn check_all(keep: impl Fn(&Prompt) -> bool, parallel: bool) -> Vec<CheckResult> {
    let root = workspace_root();
    let all = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));

    let mut results: Vec<CheckResult> = Vec::new();
    for fixtures in &all {
        let (model, tokenizer) = load(&root, fixtures);
        let prompts: Vec<&Prompt> = fixtures.manifest.prompts.iter().filter(|p| keep(p)).collect();
        let check = |prompt: &Prompt| {
            let reference = fixtures.load_tensors(prompt).unwrap_or_else(|e| panic!("{e}"));
            check_prompt(fixtures, prompt, &reference, &tokenizer, &model)
        };
        if parallel {
            thread::scope(|s| {
                let handles: Vec<_> = prompts.iter().map(|&p| s.spawn(move || check(p))).collect();
                for handle in handles {
                    results.extend(handle.join().unwrap_or_else(|panic| std::panic::resume_unwind(panic)));
                }
            });
        } else {
            for prompt in prompts {
                results.extend(check(prompt));
            }
        }
    }
    results
}

fn load(root: &std::path::Path, fixtures: &Fixtures) -> (Model, Tokenizer) {
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let tokenizer = Tokenizer::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    (model, tokenizer)
}

fn distinct_categories(results: &[CheckResult]) -> usize {
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for r in results {
        if !seen.contains(&(r.model_id.as_str(), r.category.as_str())) {
            seen.push((&r.model_id, &r.category));
        }
    }
    seen.len()
}
