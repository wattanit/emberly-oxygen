//! Tests of the S-9 check in the harness (FR-5, FR-31): the engine's adjusted
//! distribution at a prompt's last position is compared with the stored one.
//! The fixtures here are hand-made, so the expected values are computed by hand.

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use common::compare::Verdict;
use common::fixtures::{discover, Fixtures};
use common::harness::{check_sampling, EngineRun, SamplingResult};
use emberly_oxygen::{distribution, Sampling};

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxygen-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// A manifest with a three-token vocabulary and one prompt whose token IDs are
/// `[0, 1]`, with these sampling settings.
fn fixtures_with(name: &str, settings: serde_json::Value) -> (Fixtures, PathBuf) {
    let root = temp_dir(name);
    std::fs::create_dir_all(root.join("m")).unwrap();
    let margins: serde_json::Map<String, serde_json::Value> = settings
        .as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["name"].as_str().unwrap().to_string(),
                serde_json::json!({"top_k_margin_logit": 0.25, "top_p_margin_prob": null, "support": 3}),
            )
        })
        .collect();
    let manifest = serde_json::json!({
        "model_id": "m", "dims": {"layers": 1, "hidden": 1, "vocab": 3},
        "continuation_tokens": 1, "eos_token_id": 0,
        "prompts": [{
            "id": "p", "category": "english", "text": "x", "token_ids": [0, 1],
            "prompt_top2_gaps": [1.0, 1.0], "continuation_ids": [1], "continuation_text": "x",
            "continuation_top2_gaps": [1.0], "stopped_at_eos": false, "tensors_file": "p.safetensors",
            "f64_tensors_file": "p.f64.safetensors", "f64_prompt_top2_gaps": [1.0, 1.0],
            "f64_continuation_ids": [1], "f64_continuation_text": "x",
            "f64_continuation_top2_gaps": [1.0], "f64_stopped_at_eos": false,
            "sampling_file": "p.sampling.safetensors"
        }],
        "sampling": {"settings": settings, "context": "x", "margins": {"p": margins}}
    });
    std::fs::write(root.join("m/manifest.json"), manifest.to_string()).unwrap();
    (discover(&root).unwrap().remove(0), root)
}

/// Two positions of logits; the last row, [1, 2, 3], is the one S-9 uses.
fn run() -> EngineRun {
    EngineRun {
        logits: vec![9.0, 9.0, 9.0, 1.0, 2.0, 3.0],
        embeddings: vec![],
        layers: vec![],
        final_norm: vec![],
    }
}

fn softmax(x: [f64; 3]) -> Vec<f64> {
    let m = x.iter().cloned().fold(f64::MIN, f64::max);
    let e: Vec<f64> = x.iter().map(|v| (v - m).exp()).collect();
    let t: f64 = e.iter().sum();
    e.iter().map(|v| v / t).collect()
}

fn the_library(l: &[f32], c: &[u32], s: &Sampling) -> Result<Vec<f64>, emberly_oxygen::Error> {
    distribution(l, c, s)
}

fn one(results: Vec<SamplingResult>) -> SamplingResult {
    assert_eq!(results.len(), 1);
    results.into_iter().next().unwrap()
}

#[test]
fn s9_a_matching_stored_distribution_passes_and_carries_the_margins() {
    let (f, root) = fixtures_with("s9_a_matching_stored_distribution_passes_and_carries_the_margins", serde_json::json!([{"name": "t", "temperature": 0.5}]));
    // Temperature 0.5 on [1, 2, 3] is the softmax of [2, 4, 6].
    let stored = HashMap::from([("t".to_string(), softmax([2.0, 4.0, 6.0]))]);
    let r = one(check_sampling(&f, &f.manifest.prompts[0], &stored, &run(), &the_library).unwrap());
    assert_eq!(r.verdict, Verdict::Pass, "{:?}", r.verdict);
    assert!(r.max_error < 1e-12, "{:e}", r.max_error);
    assert_eq!((r.setting.as_str(), r.prompt.as_str(), r.category.as_str()), ("t", "p", "english"));
    assert_eq!(r.top_k_margin, Some(0.25));
    assert_eq!(r.top_p_margin, None);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn s9_a_stored_distribution_that_differs_fails_naming_the_setting_and_token() {
    let (f, root) = fixtures_with("s9_a_stored_distribution_that_differs_fails_naming_the_setting_and_token", serde_json::json!([{"name": "t", "temperature": 0.5}]));
    let mut wrong = softmax([2.0, 4.0, 6.0]);
    wrong[0] += 0.1;
    wrong[2] -= 0.1;
    let stored = HashMap::from([("t".to_string(), wrong)]);
    let r = one(check_sampling(&f, &f.manifest.prompts[0], &stored, &run(), &the_library).unwrap());
    let Verdict::Fail { position, message } = &r.verdict else { panic!("expected a failure: {:?}", r.verdict) };
    assert!(position.is_some() && message.contains("t"), "{message}");
    assert!((r.max_error - 0.1).abs() < 1e-6, "{:e}", r.max_error);
    std::fs::remove_dir_all(&root).unwrap();
}

// The context for the penalty is the prompt's own token IDs ([0, 1] here).
#[test]
fn fr27_the_penalty_uses_the_prompts_token_ids_as_context() {
    let (f, root) = fixtures_with("fr27_the_penalty_uses_the_prompts_token_ids_as_context", serde_json::json!([{"name": "pen", "repetition_penalty": 2.0}]));
    // Tokens 0 and 1 are penalized: [1, 2, 3] becomes [0.5, 1.0, 3.0]; temperature stays 1.
    let stored = HashMap::from([("pen".to_string(), softmax([0.5, 1.0, 3.0]))]);
    let r = one(check_sampling(&f, &f.manifest.prompts[0], &stored, &run(), &the_library).unwrap());
    assert_eq!(r.verdict, Verdict::Pass, "{:?}", r.verdict);
    // With no context the same stored distribution must not match.
    let no_context = |l: &[f32], _c: &[u32], s: &Sampling| distribution(l, &[], s);
    let r = one(check_sampling(&f, &f.manifest.prompts[0], &stored, &run(), &no_context).unwrap());
    assert!(r.verdict.is_failure(), "a check that ignored the context would pass");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn s9_the_check_judges_the_function_it_is_given() {
    let (f, root) = fixtures_with("s9_the_check_judges_the_function_it_is_given", serde_json::json!([{"name": "t", "temperature": 0.5}]));
    let stored = HashMap::from([("t".to_string(), softmax([2.0, 4.0, 6.0]))]);
    let uniform = |l: &[f32], _c: &[u32], _s: &Sampling| Ok(vec![1.0 / l.len() as f64; l.len()]);
    let r = one(check_sampling(&f, &f.manifest.prompts[0], &stored, &run(), &uniform).unwrap());
    assert!(r.verdict.is_failure());
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_a_missing_stored_distribution_or_an_engine_without_logits_is_an_error_not_a_pass() {
    let (f, root) = fixtures_with("fr31_a_missing_stored_distribution_or_an_engine_without_logits_is_an_error_not_a_pass", serde_json::json!([{"name": "t", "temperature": 0.5}]));
    let empty = HashMap::new();
    let e = check_sampling(&f, &f.manifest.prompts[0], &empty, &run(), &the_library).unwrap_err();
    assert!(e.contains("t"), "{e}");
    let stored = HashMap::from([("t".to_string(), softmax([2.0, 4.0, 6.0]))]);
    let no_logits = EngineRun { logits: vec![], embeddings: vec![], layers: vec![], final_norm: vec![] };
    assert!(check_sampling(&f, &f.manifest.prompts[0], &stored, &no_logits, &the_library).is_err());
    std::fs::remove_dir_all(&root).unwrap();
}
