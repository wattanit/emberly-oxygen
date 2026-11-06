//! Tests that a failing intermediate comparison names the first diverging
//! tensor and position. They verify FR-6 and S-2 with synthetic captures.

mod common;

use common::compare::Verdict;
use common::fixtures::{PromptTensors, Tensor};
use common::harness::{check_intermediates, EngineRun};

const LAYERS: usize = 4;
const POSITIONS: usize = 3;
const HIDDEN: usize = 2;

fn tensor(shape: &[usize], f: impl Fn(usize) -> f32) -> Tensor {
    Tensor { shape: shape.to_vec(), data: (0..shape.iter().product()).map(f).collect() }
}

/// A reference and an engine run that agrees with it exactly.
fn matching() -> (EngineRun, PromptTensors) {
    let value = |i: usize| 1.0 + i as f32;
    let reference = PromptTensors {
        logits: tensor(&[POSITIONS, 1], value),
        embeddings: tensor(&[POSITIONS, HIDDEN], value),
        layer_outputs: tensor(&[LAYERS, POSITIONS, HIDDEN], value),
        final_norm_output: tensor(&[POSITIONS, HIDDEN], value),
    };
    let per_layer = POSITIONS * HIDDEN;
    let run = EngineRun {
        logits: reference.logits.data.clone(),
        embeddings: reference.embeddings.data.clone(),
        layers: reference.layer_outputs.data.chunks(per_layer).map(<[f32]>::to_vec).collect(),
        final_norm: reference.final_norm_output.data.clone(),
    };
    (run, reference)
}

fn check(run: &EngineRun, reference: &PromptTensors) -> Verdict {
    check_intermediates(run, reference, LAYERS, HIDDEN).verdict
}

fn fail_message(v: Verdict) -> (Option<usize>, String) {
    match v {
        Verdict::Fail { position, message } => (position, message),
        other => panic!("expected a failure, got {other:?}"),
    }
}

#[test]
fn fr6_matching_capture_passes() {
    let (run, reference) = matching();
    assert_eq!(check(&run, &reference), Verdict::Pass);
}

#[test]
fn fr6_reports_the_first_diverging_layer_and_position() {
    let (mut run, reference) = matching();
    // Layer 2 holds 13 to 18, so 0.01 is about 5.6e-4 of its largest value: over S-2.
    run.layers[2][HIDDEN] += 0.01; // layer 2, position 1
    run.layers[3][0] += 5.0; // a later, larger divergence must not be reported
    let (position, message) = fail_message(check(&run, &reference));
    assert_eq!(position, Some(1));
    assert!(message.starts_with("layer 2:"), "{message}");
}

#[test]
fn fr6_names_the_first_position_within_the_layer() {
    let (mut run, reference) = matching();
    run.layers[0][2 * HIDDEN] += 1.0; // position 2
    run.layers[0][HIDDEN] += 1.0; // position 1, earlier in the matrix
    let (position, message) = fail_message(check(&run, &reference));
    assert_eq!(position, Some(1));
    assert!(message.starts_with("layer 0:"), "{message}");
}

#[test]
fn fr6_embeddings_come_before_layers_and_the_final_norm_comes_last() {
    let (mut run, reference) = matching();
    run.embeddings[0] += 1.0;
    run.layers[0][0] += 1.0;
    assert!(fail_message(check(&run, &reference)).1.starts_with("embeddings:"));

    let (mut run, reference) = matching();
    run.final_norm[HIDDEN * 2] += 1.0;
    let (position, message) = fail_message(check(&run, &reference));
    assert_eq!(position, Some(2));
    assert!(message.starts_with("final norm:"), "{message}");
}

#[test]
fn s2_divergence_just_under_the_threshold_passes() {
    let (mut run, reference) = matching();
    // Layer 1 holds 7 to 12 (each layer is judged by its own largest value);
    // 0.001 is 8.3e-5 of 12.
    run.layers[1][0] += 0.001;
    assert_eq!(check(&run, &reference), Verdict::Pass);
}

#[test]
fn fr6_a_wrong_layer_count_is_a_failure_not_a_panic() {
    let (mut run, reference) = matching();
    run.layers.pop();
    assert!(check(&run, &reference).is_failure());
    run.layers.clear();
    assert!(check(&run, &reference).is_failure());
}
