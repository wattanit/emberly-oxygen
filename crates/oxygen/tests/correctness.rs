//! The correctness harness (Spec §11): compares the engine against every
//! stored expected output. Verifies FR-5 (full comparison), FR-13 to FR-15
//! (tokenizer: encoding, decoding and streaming), S-1 to S-4 and S-5 (a release
//! passes only if every prompt category passes).
//!
//! A failing run shows the table in its failure message; a passing run prints
//! it only with `cargo test -- --nocapture`.

mod common;

use common::fixtures::{discover, require_model_dir, workspace_root};
use common::harness::{check_prompt, CheckResult};
use common::report::render;
use emberly_oxygen::{Model, Tokenizer};

#[test]
fn correctness_harness() {
    let root = workspace_root();
    let all = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));

    let mut results: Vec<CheckResult> = Vec::new();
    for fixtures in &all {
        let model_dir = root.join("models").join(&fixtures.manifest.model_id);
        require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
        let model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
        let tokenizer = Tokenizer::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
        for prompt in &fixtures.manifest.prompts {
            let reference = fixtures.load_tensors(prompt).unwrap_or_else(|e| panic!("{e}"));
            results.extend(check_prompt(fixtures, prompt, &reference, &tokenizer, &model));
        }
    }

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

fn distinct_categories(results: &[CheckResult]) -> usize {
    let mut seen: Vec<(&str, &str)> = Vec::new();
    for r in results {
        if !seen.contains(&(r.model_id.as_str(), r.category.as_str())) {
            seen.push((&r.model_id, &r.category));
        }
    }
    seen.len()
}
