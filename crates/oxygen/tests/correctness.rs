//! The correctness harness (Spec §11): compares the engine against every
//! stored expected output. Verifies FR-5 (full comparison), FR-13 to FR-15
//! (tokenizer, as far as the checks cover them), S-1 to S-4 and S-5 (a release
//! passes only if every prompt category passes).

mod common;

use common::fixtures::{discover, workspace_root};
use common::harness::{check_prompt, CheckResult};
use emberly_oxygen::Model;

#[test]
fn correctness_harness() {
    let root = workspace_root();
    let all = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));

    let mut results: Vec<CheckResult> = Vec::new();
    for fixtures in &all {
        let model_dir = root.join("models").join(&fixtures.manifest.model_id);
        let model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
        for prompt in &fixtures.manifest.prompts {
            let reference = fixtures.load_tensors(prompt).unwrap_or_else(|e| panic!("{e}"));
            results.extend(check_prompt(fixtures, prompt, &reference, &model_dir, &model));
        }
    }

    let failures: Vec<String> = results
        .iter()
        .filter(|r| r.verdict.is_failure())
        .map(|r| format!("{} / {} / {}: {:?}", r.model_id, r.prompt, r.check.name(), r.verdict))
        .collect();
    assert!(failures.is_empty(), "{} of {} checks failed:\n{}", failures.len(), results.len(), failures.join("\n"));
}
