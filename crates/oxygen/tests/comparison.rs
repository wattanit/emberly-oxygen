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

// The tolerance is a decimal (1e-3), which no binary float holds exactly, so
// "an f32 value exactly at the tolerance" cannot be built; the boundary is
// tested just under and just over it instead.
#[test]
fn s1_boundary_at_the_tolerance() {
    let reference = [0.0; 4];
    let over = [0.0, 0.0, (1.0001 * LOGIT_TOLERANCE) as f32, 0.0];
    let n = logit_error(&over, &reference, 2);
    assert_eq!(failed_at(&n.verdict), Some(1));
    let under = [0.0, 0.0, (0.9999 * LOGIT_TOLERANCE) as f32, 0.0];
    assert_eq!(logit_error(&under, &reference, 2).verdict, Verdict::Pass);
}

// FR-2, S-1: the reference is kept whole. 1 + 1e-9 is not an f32; a check that
// narrowed the reference to f32 would see no error at all here.
#[test]
fn s1_the_f64_reference_is_not_narrowed() {
    let n = logit_error(&[1.0f32, 2.0], &[1.0 + 1e-9, 2.0], 2);
    assert!(n.max_error > 0.9e-9 && n.max_error < 1.1e-9, "measured {:e}", n.max_error);
}

// The engine's f32 is widened before the difference, so the comparison adds no
// rounding: 0.1f32 against the same f32 value as an f64 is exactly zero, and
// against the f64 0.1 it is the f32's own representation error, about 1.5e-9.
#[test]
fn s1_the_engines_f32_is_widened_exactly() {
    let engine = [0.1f32, 2.0];
    let same = [f64::from(0.1f32), 2.0];
    assert_eq!(logit_error(&engine, &same, 2).max_error, 0.0);
    let n = logit_error(&engine, &[0.1f64, 2.0], 2);
    assert!(n.max_error > 1.4e-9 && n.max_error < 1.6e-9, "measured {:e}", n.max_error);
}

// FR-5: the same functions measure the distance to the f32 reference, which the
// harness reports; an f32 reference gives the same figure as its widened copy.
#[test]
fn s1_an_f32_reference_is_measured_by_the_same_function() {
    let engine = [1.0f32, 2.5, -3.0, 0.25];
    let reference32 = [1.0005f32, 2.5, -3.0, 0.2504];
    let widened: Vec<f64> = reference32.iter().map(|&v| f64::from(v)).collect();
    let a = logit_error(&engine, &reference32, 2);
    let b = logit_error(&engine, &widened, 2);
    assert_eq!(a.max_error, b.max_error);
    assert!(a.max_error > 0.0);
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
    assert!(logit_error(&[], &[] as &[f64], 2).verdict.is_failure(), "empty reference");
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

// FR-2, S-2: as for the logits, the reference is not narrowed.
#[test]
fn s2_the_f64_reference_is_not_narrowed() {
    let n = intermediate_error(&[1.0f32, 2.0], &[1.0, 2.0 + 2e-9], 2);
    assert!(n.max_error > 0.9e-9 && n.max_error < 1.1e-9, "measured {:e}", n.max_error);
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

// S-3: the gap is an f64 from the reference, so a gap one part in a million over
// the threshold is told from one under it; an f32 gap could not.
#[test]
fn s3_the_near_tie_rule_reads_the_f64_gap() {
    let (reference, engine) = ([0.0, 1.0], [1.0f32, 0.0]);
    assert_eq!(top1_check(&engine, &reference, 2, &[NEAR_TIE_GAP - 1e-12]), Verdict::Pass);
    assert!(top1_check(&engine, &reference, 2, &[NEAR_TIE_GAP + 1e-12]).is_failure());
}

#[test]
fn s3_the_reference_may_be_f32_or_f64() {
    let engine = [0.0f32, 5.0, 1.0];
    assert_eq!(top1_check(&engine, &[0.0f32, 5.0, 1.0], 3, &[1.0]), Verdict::Pass);
    assert_eq!(top1_check(&engine, &[0.0f64, 5.0, 1.0], 3, &[1.0]), Verdict::Pass);
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

// S-4: comparison stops at a near-tie of the f64 reference, told from a gap just over it.
#[test]
fn s4_the_near_tie_rule_reads_the_f64_gap() {
    assert_eq!(greedy_check(&[9], &[1], &[NEAR_TIE_GAP - 1e-12]), Verdict::NearTie { position: 0 });
    assert!(greedy_check(&[9], &[1], &[NEAR_TIE_GAP + 1e-12]).is_failure());
}

#[test]
fn s4_length_mismatches_fail() {
    let gaps = [1.0, 1.0, 1.0];
    assert_eq!(failed_at(&greedy_check(&[1, 2], &[1, 2, 3], &gaps)), Some(2), "engine stopped early");
    assert_eq!(failed_at(&greedy_check(&[], &[1, 2, 3], &gaps)), Some(0));
    assert_eq!(failed_at(&greedy_check(&[1, 2, 3, 4], &[1, 2, 3], &gaps)), Some(3), "engine ran on");
    assert!(greedy_check(&[1, 2, 3], &[1, 2, 3], &[1.0]).is_failure(), "gap list too short");
}

// --- S-9: sampling distributions ---------------------------------------------

#[test]
fn s9_identical_distributions_pass_with_zero_error() {
    let d = [0.25, 0.75, 0.0];
    let n = distribution_error(&d, &d);
    assert_eq!(n.verdict, Verdict::Pass);
    assert_eq!(n.max_error, 0.0);
}

#[test]
fn s9_boundary_at_the_tolerance() {
    let reference = [0.5, 0.5, 0.0];
    let over = [0.5 + 1.0001 * SAMPLING_TOLERANCE, 0.5 - 1.0001 * SAMPLING_TOLERANCE, 0.0];
    assert_eq!(failed_at(&distribution_error(&over, &reference).verdict), Some(0));
    let under = [0.5 + 0.9999 * SAMPLING_TOLERANCE, 0.5 - 0.9999 * SAMPLING_TOLERANCE, 0.0];
    let n = distribution_error(&under, &reference);
    assert_eq!(n.verdict, Verdict::Pass);
    assert!((n.max_error - 0.9999 * SAMPLING_TOLERANCE).abs() < 1e-15);
}

#[test]
fn s9_the_failure_names_the_token_with_the_largest_difference() {
    let reference = [0.1, 0.2, 0.7];
    let engine = [0.1, 0.2 + 0.5 * SAMPLING_TOLERANCE, 0.7 - 5.0 * SAMPLING_TOLERANCE];
    let n = distribution_error(&engine, &reference);
    // Token 2 differs most; the first token over the tolerance is also token 2.
    assert_eq!(failed_at(&n.verdict), Some(2));
    assert!((n.max_error - 5.0 * SAMPLING_TOLERANCE).abs() < 1e-15);
}

#[test]
fn s9_a_missing_probability_is_a_difference_not_a_skip() {
    // The engine kept a token the reference removed (or the reverse): the whole probability counts.
    let reference = [0.0, 1.0];
    let engine = [0.5, 0.5];
    let n = distribution_error(&engine, &reference);
    assert_eq!(failed_at(&n.verdict), Some(0));
    assert_eq!(n.max_error, 0.5);
}

#[test]
fn s9_shape_mismatch_and_non_finite_values_fail_without_panicking() {
    assert!(distribution_error(&[1.0], &[0.5, 0.5]).verdict.is_failure());
    assert!(distribution_error(&[], &[]).verdict.is_failure());
    assert!(distribution_error(&[f64::NAN, 1.0], &[0.5, 0.5]).verdict.is_failure());
    assert!(distribution_error(&[f64::INFINITY, 0.0], &[0.5, 0.5]).verdict.is_failure());
}

// --- S-11: chat template text and ids ----------------------------------------

#[test]
fn s11_identical_text_passes() {
    assert_eq!(text_check("<|im_start|>user\nhi", "<|im_start|>user\nhi"), Verdict::Pass);
    assert_eq!(text_check("", ""), Verdict::Pass);
}

#[test]
fn s11_one_differing_character_fails_at_its_position() {
    let v = text_check("abcXe", "abcde");
    assert_eq!(failed_at(&v), Some(3));
    let Verdict::Fail { message, .. } = v else { unreachable!() };
    assert!(message.contains("character 3") && message.contains('X') && message.contains('d'), "{message}");
}

#[test]
fn s11_a_text_that_is_longer_or_shorter_fails_where_it_ends() {
    assert_eq!(failed_at(&text_check("abc\n", "abc")), Some(3));
    assert_eq!(failed_at(&text_check("ab", "abc")), Some(2));
    assert_eq!(failed_at(&text_check("", "a")), Some(0));
}

// The position counts characters, not bytes: a Thai letter is three bytes.
#[test]
fn s11_the_position_counts_characters_not_bytes() {
    let thai = "\u{e2a}\u{e27}\u{e31}";
    assert_eq!(failed_at(&text_check(&format!("{thai}X"), &format!("{thai}Y"))), Some(3));
}

#[test]
fn s11_identical_ids_pass() {
    assert_eq!(ids_check(&[1, 5, 2], &[1, 5, 2]), Verdict::Pass);
    assert_eq!(ids_check(&[], &[]), Verdict::Pass);
}

#[test]
fn s11_a_differing_id_fails_at_its_position() {
    let v = ids_check(&[1, 6, 2], &[1, 5, 2]);
    assert_eq!(failed_at(&v), Some(1));
    let Verdict::Fail { message, .. } = v else { unreachable!() };
    assert!(message.contains("position 1") && message.contains('6') && message.contains('5'), "{message}");
}

#[test]
fn s11_ids_of_another_length_fail_where_one_ends() {
    assert_eq!(failed_at(&ids_check(&[1, 5, 2, 198], &[1, 5, 2])), Some(3));
    assert_eq!(failed_at(&ids_check(&[1, 5], &[1, 5, 2])), Some(2));
}
