//! Tests of the pass/fail table. They verify S-5 (judged per category, never by
//! average) and FR-5 and FR-6 (every check reported, with the first diverging
//! position).

mod common;

use common::compare::Verdict;
use common::harness::{Check, CheckResult, ChatReported, ChatResult};
use common::report::{render, render_chat};

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
            f32_error: None,
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

// S-1, S-2, FR-5: the category summary gives each category's own worst error, not only the overall one.
#[test]
fn the_measured_error_is_printed_per_category() {
    let mut all = results("a1", "english", passing());
    // `results` gives the logits check 2.5e-4; give the thai prompt a larger one.
    let mut thai = results("t1", "thai", passing());
    for r in &mut thai {
        if r.check == Check::Logits {
            r.max_error = Some(7.5e-4);
        }
        if r.check == Check::Intermediates {
            r.max_error = Some(3e-6);
        }
    }
    all.extend(thai);
    let text = squeeze(&render(&all).text);
    assert!(text.contains("english 1/1 pass 2.5e-4 -"), "{text}");
    assert!(text.contains("thai 1/1 pass 7.5e-4 3e-6"), "{text}");
}

// FR-5: the f32 distance is printed under its own label and fails nothing, even
// above S-1's limit, while the judged error stays the one the verdict used.
#[test]
fn fr5_the_f32_distance_is_labelled_reported_and_judges_nothing() {
    let mut all = results("a1", "english", passing());
    for r in &mut all {
        if r.check == Check::Logits {
            r.f32_error = Some(7.1e-3);
        }
        if r.check == Check::Intermediates {
            r.f32_error = Some(4e-4);
        }
    }
    let report = render(&all);
    assert!(report.failed.is_empty(), "an f32 distance over the limits does not fail the category");
    assert!(report.text.contains("largest logit error: 2.5e-4"), "{}", report.text);
    assert!(
        report.text.contains("largest logit error against the f32 reference (reported, never judged): 7.1e-3"),
        "{}",
        report.text
    );
    let text = squeeze(&report.text);
    assert!(text.contains("english 1/1 pass 2.5e-4 - | 7.1e-3 4e-4"), "{text}");
}

// --- S-9 in the table ------------------------------------------------------------

use common::harness::SamplingResult;
use common::report::render_with_sampling;

fn sampling_result(prompt: &str, category: &str, setting: &str, error: f64, verdict: Verdict) -> SamplingResult {
    SamplingResult {
        model_id: "m".into(),
        prompt: prompt.into(),
        category: category.into(),
        setting: setting.into(),
        max_error: error,
        verdict,
        top_k_margin: Some(2.5e-4),
        top_p_margin: None,
    }
}

#[test]
fn s9_the_table_reports_the_worst_sampling_error_per_category_and_setting() {
    let all = results("a1", "english", passing());
    let sampling = vec![
        sampling_result("a1", "english", "top-k-40", 3.0e-6, Verdict::Pass),
        sampling_result("a1", "english", "combined", 7.0e-5, Verdict::Pass),
    ];
    let report = render_with_sampling(&all, &sampling);
    assert!(report.failed.is_empty());
    let text = squeeze(&report.text);
    assert!(text.contains("sampling"), "{}", report.text);
    assert!(text.contains("top-k-40") && text.contains("combined"), "{}", report.text);
    assert!(text.contains("7e-5"), "worst error of the category: {}", report.text);
}

#[test]
fn s5_s9_a_failing_sampling_setting_fails_its_category_and_is_named() {
    let mut all = results("a1", "english", passing());
    all.extend(results("t1", "thai", passing()));
    let sampling = vec![
        sampling_result("a1", "english", "combined", 1e-6, Verdict::Pass),
        sampling_result("t1", "thai", "combined", 0.2, failing_at(5)),
    ];
    let report = render_with_sampling(&all, &sampling);
    assert_eq!(report.failed, vec![("m".to_string(), "thai".to_string())]);
    let text = squeeze(&report.text);
    assert!(text.contains("t1 / sampling combined"), "{}", report.text);
    assert!(text.contains("bad"), "the failure message is shown: {}", report.text);
}

#[test]
fn s9_the_smallest_cutoff_margins_are_reported() {
    let all = results("a1", "english", passing());
    let mut near = sampling_result("a1", "english", "top-k-40", 1e-6, Verdict::Pass);
    near.top_k_margin = Some(4.4e-4);
    let mut far = sampling_result("a2", "english", "top-k-40", 1e-6, Verdict::Pass);
    far.top_k_margin = Some(9.0e-3);
    let text = squeeze(&render_with_sampling(&all, &[near, far]).text);
    assert!(text.contains("4.4e-4") && text.contains("a1"), "the smaller margin and its prompt appear: {text}");
    assert!(!text.contains("9e-3"), "only the smallest is reported: {text}");
}

// --- S-11 and the reported-only table (FR-39) ----------------------------------

fn chat_result(conversation: &str, text: Verdict, tokens: Verdict) -> ChatResult {
    ChatResult { model_id: "i".into(), conversation: conversation.into(), text, tokens }
}

fn reported(conversation: &str, error: f64, reply: Verdict, engine: usize, reference: usize) -> ChatReported {
    ChatReported {
        model_id: "i".into(),
        conversation: conversation.into(),
        max_logit_error: error,
        reply,
        engine_reply_len: engine,
        reference_reply_len: reference,
    }
}

#[test]
fn s11_passing_conversations_are_listed_and_fail_nothing() {
    let results = [chat_result("no-system", Verdict::Pass, Verdict::Pass), chat_result("thai", Verdict::Pass, Verdict::Pass)];
    let report = render_chat(&results, &[]);
    assert!(report.failed.is_empty());
    let text = squeeze(&report.text);
    assert!(text.contains("no-system pass pass") && text.contains("thai pass pass"), "{}", report.text);
    assert!(text.contains("S-11"), "{}", report.text);
}

#[test]
fn s11_a_failing_conversation_fails_the_run_and_its_failure_is_located() {
    let text = Verdict::Fail { position: Some(12), message: "character 12: engine has 'u', reference has 's'".into() };
    let results = [chat_result("no-system", Verdict::Pass, Verdict::Pass), chat_result("thai", text, failing_at(0))];
    let report = render_chat(&results, &[]);
    assert_eq!(report.failed, [("i".to_string(), "thai".to_string())]);
    let squeezed = squeeze(&report.text);
    assert!(squeezed.contains("thai FAIL FAIL"), "{}", report.text);
    assert!(report.text.contains("thai / text") && report.text.contains("character 12"), "{}", report.text);
    assert!(report.text.contains("thai / tokens"), "{}", report.text);
}

// FR-39: the reported figures are printed, labelled, and cannot fail the run.
#[test]
fn fr39_a_reported_figure_never_fails_the_run() {
    let results = [chat_result("c", Verdict::Pass, Verdict::Pass)];
    let reported = [
        reported("c", 0.5, Verdict::Fail { position: Some(3), message: "token 9 at continuation index 3, reference 8".into() }, 6, 8),
        reported("d", f64::INFINITY, Verdict::NearTie { position: 2 }, 0, 4),
    ];
    let report = render_chat(&results, &reported);
    assert!(report.failed.is_empty(), "a reported figure judges nothing");
    let text = squeeze(&report.text);
    assert!(text.contains("reported only (instruct)"), "{}", report.text);
    assert!(text.contains("c 5e-1") && text.contains("differs at 3"), "{}", report.text);
    assert!(text.contains("d inf") && text.contains("tie at 2"), "{}", report.text);
    let after = report.text.split("reported only (instruct)").nth(1).unwrap();
    assert!(!after.contains("FAIL"), "reported rows must not look like failures: {after}");
}

#[test]
fn fr39_an_agreeing_reply_shows_its_lengths() {
    let reported = [reported("c", 7.5e-5, Verdict::Pass, 8, 8)];
    let report = render_chat(&[chat_result("c", Verdict::Pass, Verdict::Pass)], &reported);
    let text = squeeze(&report.text);
    assert!(text.contains("c 7.5e-5 agrees 8/8"), "{}", report.text);
}
