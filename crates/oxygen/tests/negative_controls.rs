//! Negative controls (Spec §17): the harness must be seen to fail. Each test
//! breaks the engine in one known way, runs a real prompt through the same
//! `check_prompt` the correctness harness uses, and asserts that the right check
//! fails and says where. With no fault the same prompt passes, so a fault that
//! did nothing would itself be caught. Verifies S-1 and S-2 can fail (a gate
//! that has never failed has not been shown to work) and FR-6 (the report names
//! the first diverging layer and position).

mod common;

use common::compare::{Verdict, LOGIT_TOLERANCE};
use common::fixtures::{discover, require_model_dir, workspace_root};
use common::harness::{check_prompt, run_engine, Check, CheckResult};
use emberly_oxygen::{Fault, Model, Tokenizer};

/// A short prompt, so each control takes a couple of seconds.
const PROMPT: &str = "english-short";

fn run(fault: Option<Fault>) -> Vec<CheckResult> {
    let root = workspace_root();
    let fixtures = discover(&root.join("fixtures"))
        .unwrap_or_else(|e| panic!("{e}"))
        .into_iter()
        .next()
        .expect("no fixtures");
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let mut model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    if let Some(fault) = fault {
        model.inject_fault(fault);
    }
    let tokenizer = Tokenizer::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == PROMPT).expect("prompt missing");
    let reference = fixtures.load_reference(prompt).unwrap_or_else(|e| panic!("{e}"));
    let f32_reference = fixtures.load_tensors(prompt).unwrap_or_else(|e| panic!("{e}"));
    check_prompt(&fixtures, prompt, &reference, &f32_reference, &tokenizer, &model)
}

fn result(results: &[CheckResult], check: Check) -> &CheckResult {
    results.iter().find(|r| r.check == check).expect("check missing")
}

fn passes(results: &[CheckResult], check: Check) -> bool {
    result(results, check).verdict == Verdict::Pass
}

fn failure(results: &[CheckResult], check: Check) -> (Option<usize>, &str) {
    match &result(results, check).verdict {
        Verdict::Fail { position, message } => (*position, message.as_str()),
        other => panic!("{} should have failed, got {other:?}", check.name()),
    }
}

// S-1, S-2, S-3: with no fault the prompt passes the three numeric checks and the tokenizer
// checks, so the failures below are caused by the faults.
#[test]
fn s1_s2_without_a_fault_the_numeric_checks_pass() {
    let results = run(None);
    for check in [Check::Encode, Check::Decode, Check::Streaming, Check::Logits, Check::Top1, Check::Intermediates] {
        assert!(passes(&results, check), "{} should pass: {:?}", check.name(), result(&results, check).verdict);
    }
}

// S-2, FR-6: pairing dimension 2i with 2i+1 instead of i with i+d/2 fails the intermediates, and the
// report names the first diverging layer and position. Position 0 is unrotated, so it is position 1.
#[test]
fn s2_interleaved_rope_fails_the_intermediates() {
    let results = run(Some(Fault::InterleavedRope));
    let (position, message) = failure(&results, Check::Intermediates);
    assert_eq!(position, Some(1), "{message}");
    assert!(message.starts_with("layer 0"), "the embedding is right and layer 0 is the first wrong tensor: {message}");
    assert!(passes(&results, Check::Encode) && passes(&results, Check::Decode), "only the engine's math was broken");
}

// S-1, FR-6: a 2e-3 error in one logit, just over S-1's 1e-3, fails the logits and names that position,
// while every intermediate (which the fault never touches) still passes.
#[test]
fn s1_a_perturbed_logit_fails_the_logits_at_its_position() {
    let results = run(Some(Fault::PerturbedLogit { position: 5 }));
    let (position, message) = failure(&results, Check::Logits);
    assert_eq!(position, Some(5), "{message}");
    let error = result(&results, Check::Logits).max_error.expect("the logits check measures an error");
    assert!(error > 1e-3 && error < 3e-3, "measured {error}, expected about 2e-3");
    assert!(passes(&results, Check::Intermediates), "the intermediates are untouched by this fault");
}

// S-1, S-2, FR-6: computing logits without the final norm fails the logits and the final-norm
// intermediate, and every layer before it still passes.
#[test]
fn s1_s2_a_dropped_final_norm_fails_logits_and_the_final_norm() {
    let results = run(Some(Fault::NoFinalNorm));
    failure(&results, Check::Logits);
    let (_, message) = failure(&results, Check::Intermediates);
    assert!(message.starts_with("final norm"), "every layer is right; the final norm is the first wrong tensor: {message}");
}

// FR-40, FR-41, S-1, S-2, FR-6: a batched pass with no causal mask lets position 0 see the whole prompt, so
// the logits fail at position 0 and the first wrong tensor is layer 0 (the embeddings are right). The
// harness reaches the engine through `prefill`, the only place the fault acts, so this also shows
// the gate judges the batched pass and not the one-token path.
#[test]
fn fr40_s1_s2_an_unmasked_attention_fails_the_logits_and_the_first_layer() {
    let results = run(Some(Fault::UnmaskedAttention));
    let (position, message) = failure(&results, Check::Logits);
    assert_eq!(position, Some(0), "{message}");
    let (position, message) = failure(&results, Check::Intermediates);
    assert_eq!(position, Some(0), "{message}");
    assert!(message.starts_with("layer 0"), "the embeddings are right and layer 0 is the first wrong tensor: {message}");
    assert!(passes(&results, Check::Encode) && passes(&results, Check::Decode), "only the engine's math was broken");
}

// S-1, HC-4: the perturbation is sized against the limit in force. Its size is measured, not assumed: the same
// prompt is run with and without the fault and the one logit it touches is compared. If S-1 is ever loosened
// to the perturbation or past it, this fails loudly instead of leaving a control that can no longer fail; if S-1
// is tightened until the perturbation is no longer "just over" it, it fails too, so the control is re-sized.
#[test]
fn s1_the_perturbation_exceeds_the_limit_in_force_and_touches_one_logit() {
    let root = workspace_root();
    let fixtures = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}")).into_iter().next().unwrap();
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == PROMPT).expect("prompt missing");
    let position = 5;

    let clean = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let mut faulted = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    faulted.inject_fault(Fault::PerturbedLogit { position });
    let (clean, faulted) = (run_engine(&clean, &prompt.token_ids).unwrap(), run_engine(&faulted, &prompt.token_ids).unwrap());

    let differing: Vec<usize> = (0..clean.logits.len()).filter(|&i| clean.logits[i] != faulted.logits[i]).collect();
    let vocab = fixtures.manifest.dims.vocab;
    assert_eq!(differing, [position * vocab], "the fault touches the first logit at its position and nothing else");
    let size = f64::from(faulted.logits[position * vocab] - clean.logits[position * vocab]);
    assert!(size > LOGIT_TOLERANCE, "perturbation {size:e} does not exceed S-1's limit {LOGIT_TOLERANCE:e}");
    assert!(size < 3.0 * LOGIT_TOLERANCE, "perturbation {size:e} is no longer just over S-1's limit {LOGIT_TOLERANCE:e}: re-size it");
}

// --- Sampling controls (S-9) -----------------------------------------------------
//
// The distribution check must be seen to fail (Spec §17). Each fault changes how
// the engine's adjusted distribution is computed, and S-9 must fail on a setting
// that exercises it. With no fault the same prompt passes every setting, so a
// fault that did nothing would itself be caught.

use common::harness::{check_sampling, SamplingResult};

/// The prompt the sampling controls use. Not `english-short`: there the inverted
/// penalty moves the distribution by only 7.7e-3, because the tokens it hits
/// carry little probability, while on `thai-medium` it moves it by 0.85 and the
/// combined setting keeps 7 tokens, so a swapped order shows (plan 0.2.0 §5).
const SAMPLING_PROMPT: &str = "thai-medium";

/// One prompt's S-9 results with the given fault (or none) in the distribution function.
fn sampling_run(fault: Option<Fault>) -> Vec<SamplingResult> {
    let root = workspace_root();
    let fixtures = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}")).into_iter().next().expect("no fixtures");
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == SAMPLING_PROMPT).expect("prompt missing");
    let stored = fixtures.load_sampling(prompt).unwrap_or_else(|e| panic!("{e}"));
    let engine = run_engine(&model, &prompt.token_ids).unwrap_or_else(|e| panic!("{e}"));
    let distribution = move |l: &[f32], c: &[u32], s: &emberly_oxygen::Sampling| match fault {
        Some(fault) => emberly_oxygen::distribution_with_fault(l, c, s, fault),
        None => emberly_oxygen::distribution(l, c, s),
    };
    check_sampling(&fixtures, prompt, &stored, &engine, &distribution).unwrap_or_else(|e| panic!("{e}"))
}

fn failing_settings(results: &[SamplingResult]) -> Vec<&str> {
    results.iter().filter(|r| r.verdict.is_failure()).map(|r| r.setting.as_str()).collect()
}

#[test]
fn s9_without_a_fault_every_sampling_setting_passes() {
    let results = sampling_run(None);
    assert!(results.len() >= 5, "each control alone and a combination");
    assert!(failing_settings(&results).is_empty(), "{:?}", failing_settings(&results));
}

#[test]
fn s9_a_swapped_sampling_order_fails_the_combined_setting() {
    let results = sampling_run(Some(Fault::SwappedSamplingOrder));
    // Top-p before temperature only differs where both are set.
    assert_eq!(failing_settings(&results), ["combined"]);
    let combined = results.iter().find(|r| r.setting == "combined").unwrap();
    assert!(combined.max_error > 1e-2, "the fault should be far over any tolerance: {:e}", combined.max_error);
}

#[test]
fn s9_an_inverted_penalty_fails_the_penalty_settings() {
    let results = sampling_run(Some(Fault::PenaltyWrongSign));
    let failing = failing_settings(&results);
    assert!(failing.contains(&"penalty-1.3") && failing.contains(&"combined"), "{failing:?}");
    assert!(!failing.contains(&"temperature-0.7") && !failing.contains(&"top-k-40"), "{failing:?}");
}
