//! Unit tests of the comparison functions. Each is named for the S item it
//! verifies; boundary tests sit just under and just over each threshold.

mod common;

use common::compare::*;

fn failed_at(v: &Verdict) -> Option<usize> {
    match v {
        Verdict::Fail { position, .. } => *position,
        other => panic!("expected a failure, got {other:?}"),
    }
}

// --- S-1: logits -----------------------------------------------------------

#[test]
fn s1_identical_logits_pass_with_zero_error() {
    let r = [0.5, -1.0, 2.0, 3.0, 4.0, 5.0];
    let n = logit_error(&r, &r, 3);
    assert_eq!(n.verdict, Verdict::Pass);
    assert_eq!(n.max_error, 0.0);
}

#[test]
fn s1_boundary_at_the_tolerance() {
    let reference = [0.0; 4];
    let at = [0.0, 0.0, LOGIT_TOLERANCE, 0.0];
    assert_eq!(logit_error(&at, &reference, 2).verdict, Verdict::Pass, "exactly the tolerance passes");
    let over = [0.0, 0.0, 1.0001 * LOGIT_TOLERANCE, 0.0];
    let n = logit_error(&over, &reference, 2);
    assert_eq!(failed_at(&n.verdict), Some(1));
    let under = [0.0, 0.0, 0.9999 * LOGIT_TOLERANCE, 0.0];
    assert_eq!(logit_error(&under, &reference, 2).verdict, Verdict::Pass);
}

#[test]
fn s1_reports_the_first_diverging_position_and_the_worst_error() {
    let reference = [0.0; 6];
    let engine = [0.0, 0.0, 0.002, 0.0, 0.0, 0.5];
    let n = logit_error(&engine, &reference, 2);
    assert_eq!(failed_at(&n.verdict), Some(1), "first position, not the worst");
    assert_eq!(n.max_error, 0.5, "worst error over all positions is still reported");
}

#[test]
fn s1_nan_and_infinity_always_fail() {
    let reference = [1.0, 2.0, 3.0, 4.0];
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let n = logit_error(&[1.0, 2.0, 3.0, bad], &reference, 2);
        assert_eq!(failed_at(&n.verdict), Some(1));
    }
}

#[test]
fn s1_shape_mismatch_is_a_failure_not_a_panic() {
    assert!(logit_error(&[], &[1.0, 2.0], 2).verdict.is_failure(), "engine returned nothing");
    assert!(logit_error(&[1.0; 3], &[1.0; 4], 2).verdict.is_failure(), "length differs");
    assert!(logit_error(&[1.0; 3], &[1.0; 3], 2).verdict.is_failure(), "not a multiple of the width");
    assert!(logit_error(&[], &[], 2).verdict.is_failure(), "empty reference");
    assert!(logit_error(&[1.0], &[1.0], 0).verdict.is_failure(), "zero width");
}

// --- S-2: intermediates ----------------------------------------------------

#[test]
fn s2_error_is_relative_to_the_largest_reference_magnitude() {
    let reference = [100.0, 1.0, -50.0, 2.0];
    // An absolute error of 0.0099 is 9.9e-5 of 100: under the bound.
    let under = [100.0099, 1.0, -50.0, 2.0];
    assert_eq!(intermediate_error(&under, &reference, 2).verdict, Verdict::Pass);
    // 0.0101 is 1.01e-4 of 100: over it.
    let over = [100.0, 1.0, -50.0101, 2.0];
    assert_eq!(failed_at(&intermediate_error(&over, &reference, 2).verdict), Some(1));
}

#[test]
fn s2_the_same_absolute_error_is_judged_by_the_tensor_scale() {
    let engine_error = 0.005;
    let small = [1.0, 0.0];
    let large = [1000.0, 0.0];
    assert!(intermediate_error(&[1.0 + engine_error, 0.0], &small, 2).verdict.is_failure());
    assert_eq!(intermediate_error(&[1000.0 + engine_error, 0.0], &large, 2).verdict, Verdict::Pass);
}

#[test]
fn s2_reports_the_measured_relative_error() {
    let n = intermediate_error(&[1.0, 2.0], &[1.0, 4.0], 2);
    assert_eq!(n.max_error, 0.5);
}

#[test]
fn s2_zero_reference_nan_and_shape_problems_fail() {
    assert!(intermediate_error(&[0.0, 0.0], &[0.0, 0.0], 2).verdict.is_failure(), "all-zero reference");
    assert!(intermediate_error(&[f32::NAN, 1.0], &[1.0, 1.0], 2).verdict.is_failure());
    assert!(intermediate_error(&[1.0], &[1.0, 1.0], 2).verdict.is_failure());
    assert!(intermediate_error(&[], &[1.0, 1.0], 2).verdict.is_failure());
}

// --- S-3: top token --------------------------------------------------------

#[test]
fn s3_matching_top_tokens_pass() {
    let reference = [0.0, 5.0, 1.0, 9.0, 0.0, 0.0];
    let engine = [0.1, 4.9, 1.1, 8.0, 0.2, 0.0];
    assert_eq!(top1_check(&engine, &reference, 3, &[5.0, 9.0]), Verdict::Pass);
}

#[test]
fn s3_a_different_top_token_fails_at_its_position() {
    let reference = [0.0, 5.0, 1.0, 9.0, 0.0, 0.0];
    let engine = [0.0, 5.0, 1.0, 0.0, 9.0, 0.0];
    assert_eq!(failed_at(&top1_check(&engine, &reference, 3, &[5.0, 9.0])), Some(1));
}

#[test]
fn s3_near_tie_boundary_on_the_reference_gap() {
    let reference = [0.0, 1.0];
    let engine = [1.0, 0.0];
    // Gap just under the threshold: either token passes.
    assert_eq!(top1_check(&engine, &reference, 2, &[0.9999 * NEAR_TIE_GAP]), Verdict::Pass);
    // Gap exactly at the threshold is not a near-tie ("less than").
    assert!(top1_check(&engine, &reference, 2, &[NEAR_TIE_GAP]).is_failure());
}

#[test]
fn s3_exact_tie_takes_the_lowest_index() {
    let reference = [3.0, 3.0, 1.0];
    assert_eq!(top1_check(&[3.0, 3.0, 1.0], &reference, 3, &[0.0]), Verdict::Pass);
    // Engine picks index 1 where the reference picks 0, but the gap is 0: near-tie.
    assert_eq!(top1_check(&[1.0, 2.0, 0.0], &reference, 3, &[0.0]), Verdict::Pass);
    // With a large stated gap the same disagreement fails.
    assert!(top1_check(&[1.0, 2.0, 0.0], &reference, 3, &[1.0]).is_failure());
}

#[test]
fn s3_nan_and_shape_problems_fail() {
    assert!(top1_check(&[f32::NAN, 1.0], &[0.0, 1.0], 2, &[1.0]).is_failure());
    assert!(top1_check(&[1.0], &[0.0, 1.0], 2, &[1.0]).is_failure());
    assert!(top1_check(&[0.0, 1.0], &[0.0, 1.0], 2, &[]).is_failure(), "gap list too short");
}

// --- S-4: greedy continuation ---------------------------------------------

#[test]
fn s4_identical_continuations_pass() {
    assert_eq!(greedy_check(&[1, 2, 3], &[1, 2, 3], &[1.0, 1.0, 1.0]), Verdict::Pass);
}

#[test]
fn s4_a_different_token_fails_at_its_index() {
    let v = greedy_check(&[1, 9, 3], &[1, 2, 3], &[1.0, 1.0, 1.0]);
    assert_eq!(failed_at(&v), Some(1));
}

#[test]
fn s4_near_tie_at_the_start_stops_comparison_there() {
    let gaps = [0.0, 1.0, 1.0];
    assert_eq!(greedy_check(&[9, 9, 9], &[1, 2, 3], &gaps), Verdict::NearTie { position: 0 });
}

#[test]
fn s4_near_tie_at_the_end_still_checks_what_came_before() {
    let gaps = [1.0, 1.0, 0.0];
    assert_eq!(greedy_check(&[1, 2, 9], &[1, 2, 3], &gaps), Verdict::NearTie { position: 2 });
    assert_eq!(failed_at(&greedy_check(&[1, 9, 9], &[1, 2, 3], &gaps)), Some(1), "mismatch before the tie is a failure");
}

#[test]
fn s4_near_tie_boundary_on_the_gap() {
    assert_eq!(greedy_check(&[9], &[1], &[0.9999 * NEAR_TIE_GAP]), Verdict::NearTie { position: 0 });
    assert!(greedy_check(&[9], &[1], &[NEAR_TIE_GAP]).is_failure());
}

#[test]
fn s4_length_mismatches_fail() {
    let gaps = [1.0, 1.0, 1.0];
    assert_eq!(failed_at(&greedy_check(&[1, 2], &[1, 2, 3], &gaps)), Some(2), "engine stopped early");
    assert_eq!(failed_at(&greedy_check(&[], &[1, 2, 3], &gaps)), Some(0));
    assert_eq!(failed_at(&greedy_check(&[1, 2, 3, 4], &[1, 2, 3], &gaps)), Some(3), "engine ran on");
    assert!(greedy_check(&[1, 2, 3], &[1, 2, 3], &[1.0]).is_failure(), "gap list too short");
}
