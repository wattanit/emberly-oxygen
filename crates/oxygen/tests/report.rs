//! Tests of the pass/fail table. They verify S-5 (judged per category, never by
//! average) and FR-5 and FR-6 (every check reported, with the first diverging
//! position).

mod common;

use common::compare::Verdict;
use common::harness::{Check, CheckResult};
use common::report::render;

fn results(prompt: &str, category: &str, verdicts: [Verdict; Check::ALL.len()]) -> Vec<CheckResult> {
    Check::ALL
        .into_iter()
        .zip(verdicts)
        .map(|(check, verdict)| CheckResult {
            model_id: "m".into(),
            prompt: prompt.into(),
            category: category.into(),
            check,
            verdict,
            max_error: matches!(check, Check::Logits).then_some(2.5e-4),
        })
        .collect()
}

/// Collapses runs of whitespace so assertions do not depend on column widths.
fn squeeze(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn passing() -> [Verdict; Check::ALL.len()] {
    std::array::from_fn(|_| Verdict::Pass)
}

/// Where a check's verdict sits in the array, whatever the column order.
fn at(check: Check) -> usize {
    Check::ALL.iter().position(|&c| c == check).unwrap()
}

fn failing_at(position: usize) -> Verdict {
    Verdict::Fail { position: Some(position), message: "bad".into() }
}

#[test]
fn s5_one_failing_prompt_fails_its_category_only() {
    let mut all = results("a1", "english", passing());
    all.extend(results("a2", "english", passing()));
    let mut bad = passing();
    bad[at(Check::Logits)] = failing_at(7);
    all.extend(results("t1", "thai", bad));
    all.extend(results("t2", "thai", passing()));

    let report = render(&all);
    assert_eq!(report.failed, [("m".to_string(), "thai".to_string())]);
    let summary = squeeze(&report.text);
    assert!(summary.contains("english 2/2 pass"), "{}", report.text);
    assert!(summary.contains("thai 1/2 FAIL"), "{}", report.text);
}

#[test]
fn s5_a_category_is_never_passed_by_average() {
    // Nine passing prompts do not rescue a category with one failing prompt.
    let mut all = Vec::new();
    for i in 0..9 {
        all.extend(results(&format!("p{i}"), "code", passing()));
    }
    let mut bad = passing();
    bad[at(Check::Greedy)] = failing_at(0);
    all.extend(results("p9", "code", bad));
    assert_eq!(render(&all).failed.len(), 1);
}

#[test]
fn s5_all_passing_means_no_failed_category() {
    let all = results("a1", "english", passing());
    assert!(render(&all).failed.is_empty());
}

#[test]
fn fr5_every_check_has_a_column_and_a_cell() {
    let report = render(&results("a1", "english", passing()));
    for check in Check::ALL {
        assert!(report.text.contains(check.name()), "missing column {}", check.name());
    }
    let row = report.text.lines().find(|l| l.starts_with("a1")).unwrap();
    assert_eq!(row.matches("pass").count(), Check::ALL.len(), "{row}");
}

#[test]
fn fr6_a_failure_line_gives_the_first_diverging_position() {
    let mut bad = passing();
    bad[at(Check::Intermediates)] = Verdict::Fail { position: Some(12), message: "layer 3: too large".into() };
    let report = render(&results("a1", "english", bad));
    assert!(report.text.contains("a1 / intermediates: first diverges at position 12: layer 3: too large"), "{}", report.text);
}

#[test]
fn s4_a_near_tie_passes_but_is_reported() {
    let mut tie = passing();
    tie[at(Check::Greedy)] = Verdict::NearTie { position: 9 };
    let report = render(&results("a1", "english", tie));
    assert!(report.failed.is_empty());
    assert!(report.text.contains("near-tie, position 9"), "{}", report.text);
}

#[test]
fn the_measured_error_is_printed() {
    let report = render(&results("a1", "english", passing()));
    assert!(report.text.contains("largest logit error: 2.5e-4"), "{}", report.text);
}
