//! The pass/fail table (FR-5, S-5): rows are prompts, columns are checks, and a
//! summary by category follows. A category passes only if every prompt in it
//! passes every check; no averages are used anywhere.

use std::fmt::Write;

use super::compare::Verdict;
use super::harness::{Check, CheckResult, ChatReported, ChatResult, SamplingResult};

pub struct Report {
    pub text: String,
    /// (model ID, category) pairs with at least one failing check.
    pub failed: Vec<(String, String)>,
}

fn cell(v: &Verdict) -> &'static str {
    match v {
        Verdict::Pass => "pass",
        Verdict::NearTie { .. } => "tie",
        Verdict::Fail { .. } => "FAIL",
    }
}

/// Distinct values in order of first appearance.
fn distinct<'a>(items: impl Iterator<Item = &'a str>) -> Vec<&'a str> {
    let mut out: Vec<&str> = Vec::new();
    for i in items {
        if !out.contains(&i) {
            out.push(i);
        }
    }
    out
}

pub fn render(results: &[CheckResult]) -> Report {
    render_with_sampling(results, &[])
}

/// The table, with S-9's results beside it: per category the worst probability
/// error for each setting, the smallest cutoff margins (so a cutoff the engine's
/// rounding could cross is visible), and each failing setting named. A category
/// with a failing setting fails (S-5).
pub fn render_with_sampling(results: &[CheckResult], sampling: &[SamplingResult]) -> Report {
    let mut text = String::new();
    let mut failed = Vec::new();

    for model in distinct(results.iter().map(|r| r.model_id.as_str())) {
        let rows: Vec<&CheckResult> = results.iter().filter(|r| r.model_id == model).collect();
        writeln!(text, "model {model}").unwrap();

        write!(text, "{:<20} {:<11}", "prompt", "category").unwrap();
        for c in Check::ALL {
            write!(text, " {:<13}", c.name()).unwrap();
        }
        text.push('\n');
        for prompt in distinct(rows.iter().map(|r| r.prompt.as_str())) {
            let of_prompt: Vec<_> = rows.iter().filter(|r| r.prompt == prompt).collect();
            write!(text, "{:<20} {:<11}", prompt, of_prompt[0].category).unwrap();
            for c in Check::ALL {
                let c = of_prompt.iter().find(|r| r.check == c).map_or("-", |r| cell(&r.verdict));
                write!(text, " {c:<13}").unwrap();
            }
            text.push('\n');
        }

        writeln!(
            text,
            "\ncategory     prompts  result  max logit error  max intermediate error  | vs f32 reference, reported only: logit  intermediate"
        )
        .unwrap();
        for category in distinct(rows.iter().map(|r| r.category.as_str())) {
            let of_category: Vec<_> = rows.iter().filter(|r| r.category == category).collect();
            let prompts = distinct(of_category.iter().map(|r| r.prompt.as_str()));
            let bad = prompts
                .iter()
                .filter(|p| of_category.iter().any(|r| r.prompt == **p && r.verdict.is_failure()))
                .count();
            let verdict = if bad == 0 { "pass" } else { "FAIL" };
            // Worst error measured within the category, for the checks that measure one (S-1, S-2).
            let worst = |check: Check, reported: bool| {
                let e = of_category
                    .iter()
                    .filter(|r| r.check == check)
                    .filter_map(|r| if reported { r.f32_error } else { r.max_error });
                e.fold(None, |m: Option<f64>, e| Some(m.map_or(e, |m| m.max(e)))).map_or("-".to_string(), |e| format!("{e:e}"))
            };
            writeln!(
                text,
                "{category:<12} {:>2}/{:<5} {verdict:<7} {:<16} {:<23} | {:<9} {}",
                prompts.len() - bad,
                prompts.len(),
                worst(Check::Logits, false),
                worst(Check::Intermediates, false),
                worst(Check::Logits, true),
                worst(Check::Intermediates, true)
            )
            .unwrap();
            if bad > 0 {
                failed.push((model.to_string(), category.to_string()));
            }
        }

        let measured = |check: Check, reported: bool| {
            rows.iter()
                .filter(|r| r.check == check)
                .filter_map(|r| if reported { r.f32_error } else { r.max_error })
                .fold(None, |m: Option<f64>, e| Some(m.map_or(e, |m| m.max(e))))
        };
        text.push('\n');
        for (check, label) in [(Check::Logits, "logit error"), (Check::Intermediates, "relative intermediate error")] {
            if let Some(e) = measured(check, false) {
                writeln!(text, "largest {label}: {e:e}").unwrap();
            }
        }
        for (check, label) in [(Check::Logits, "logit error"), (Check::Intermediates, "relative intermediate error")] {
            if let Some(e) = measured(check, true) {
                writeln!(text, "largest {label} against the f32 reference (reported, never judged): {e:e}").unwrap();
            }
        }

        let of_model: Vec<&SamplingResult> = sampling.iter().filter(|r| r.model_id == model).collect();
        if !of_model.is_empty() {
            let settings = distinct(of_model.iter().map(|r| r.setting.as_str()));
            writeln!(text, "sampling distributions (S-9), worst probability error against the stored distribution:").unwrap();
            write!(text, "{:<12}", "category").unwrap();
            for setting in &settings {
                write!(text, " {setting:<16}").unwrap();
            }
            text.push('\n');
            for category in distinct(of_model.iter().map(|r| r.category.as_str())) {
                write!(text, "{category:<12}").unwrap();
                for setting in &settings {
                    let worst = of_model
                        .iter()
                        .filter(|r| r.category == category && r.setting == *setting)
                        .map(|r| r.max_error)
                        .fold(None, |m: Option<f64>, e| Some(m.map_or(e, |m| m.max(e))));
                    write!(text, " {:<16}", worst.map_or("-".to_string(), |e| format!("{e:e}"))).unwrap();
                }
                text.push('\n');
                let category_failed = of_model.iter().any(|r| r.category == category && r.verdict.is_failure());
                let key = (model.to_string(), category.to_string());
                if category_failed && !failed.contains(&key) {
                    failed.push(key);
                }
            }
            // The smallest margin to each cutoff: a cutoff nearer than the engine's rounding can flip.
            let smallest = |margin: fn(&SamplingResult) -> Option<f64>| {
                of_model
                    .iter()
                    .filter_map(|r| margin(r).map(|m| (m, *r)))
                    .min_by(|a, b| a.0.total_cmp(&b.0))
            };
            if let Some((m, r)) = smallest(|r| r.top_k_margin) {
                writeln!(text, "smallest top-k cutoff margin: {m:e} logits ({} / {})", r.prompt, r.setting).unwrap();
            }
            if let Some((m, r)) = smallest(|r| r.top_p_margin) {
                writeln!(text, "smallest top-p cutoff margin: {m:e} probability ({} / {})", r.prompt, r.setting).unwrap();
            }
            text.push('\n');
        }

        let failures: Vec<_> = rows.iter().filter(|r| r.verdict.is_failure()).collect();
        if !failures.is_empty() {
            writeln!(text, "\nfailures ({}):", failures.len()).unwrap();
            for r in failures {
                let Verdict::Fail { position, message } = &r.verdict else { continue };
                let at = position.map_or(String::new(), |p| format!("first diverges at position {p}: "));
                writeln!(text, "  {} / {}: {at}{message}", r.prompt, r.check.name()).unwrap();
            }
        }
        for r in of_model.iter().filter(|r| r.verdict.is_failure()) {
            let Verdict::Fail { position, message } = &r.verdict else { continue };
            let at = position.map_or(String::new(), |t| format!("largest difference at token {t}: "));
            writeln!(text, "  {} / sampling {}: {at}{message}", r.prompt, r.setting).unwrap();
        }
        let ties: Vec<_> = rows.iter().filter(|r| matches!(r.verdict, Verdict::NearTie { .. })).collect();
        for r in ties {
            let Verdict::NearTie { position } = r.verdict else { continue };
            writeln!(text, "note: {} / {}: comparison stopped at a near-tie, position {position}", r.prompt, r.check.name()).unwrap();
        }
        text.push('\n');
    }
    let text = text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n") + "\n";
    Report { text, failed }
}

/// The chat tables (S-11, FR-39): per conversation, whether the rendered text and
/// the token ids equal the reference's, and then, labelled *reported only
/// (instruct)*, the logit difference and greedy reply agreement that judge
/// nothing. `failed` holds the (model, conversation) pairs whose S-11 failed and
/// nothing else: the reported figures never reach it.
pub fn render_chat(results: &[ChatResult], reported: &[ChatReported]) -> Report {
    let mut text = String::new();
    let mut failed = Vec::new();
    let models = distinct(results.iter().map(|r| r.model_id.as_str()).chain(reported.iter().map(|r| r.model_id.as_str())));
    for model in models {
        writeln!(text, "model {model}").unwrap();
        let rows: Vec<&ChatResult> = results.iter().filter(|r| r.model_id == model).collect();
        if !rows.is_empty() {
            writeln!(text, "chat template (S-11), rendered text and token ids against the reference:").unwrap();
            writeln!(text, "{:<16} {:<8} {:<8}", "conversation", "text", "tokens").unwrap();
            for r in &rows {
                writeln!(text, "{:<16} {:<8} {:<8}", r.conversation, cell(&r.text), cell(&r.tokens)).unwrap();
                if r.is_failure() {
                    failed.push((model.to_string(), r.conversation.clone()));
                }
            }
            for r in &rows {
                for (what, verdict) in [("text", &r.text), ("tokens", &r.tokens)] {
                    if let Verdict::Fail { message, .. } = verdict {
                        writeln!(text, "  {} / {what}: {message}", r.conversation).unwrap();
                    }
                }
            }
            text.push('\n');
        }
        let of_model: Vec<&ChatReported> = reported.iter().filter(|r| r.model_id == model).collect();
        if !of_model.is_empty() {
            writeln!(text, "reported only (instruct), judges nothing:").unwrap();
            writeln!(text, "{:<16} {:<20} greedy reply against the reference", "conversation", "largest logit error")
                .unwrap();
            for r in of_model {
                let lengths = format!("{}/{}", r.engine_reply_len, r.reference_reply_len);
                let reply = match &r.reply {
                    Verdict::Pass => format!("agrees {lengths}"),
                    Verdict::NearTie { position } => format!("tie at {position} {lengths}"),
                    Verdict::Fail { position: Some(p), .. } => format!("differs at {p} {lengths}"),
                    Verdict::Fail { position: None, .. } => format!("differs {lengths}"),
                };
                writeln!(text, "{:<16} {:<20} {reply}", r.conversation, format!("{:e}", r.max_logit_error)).unwrap();
            }
            text.push('\n');
        }
    }
    let text = text.lines().map(str::trim_end).collect::<Vec<_>>().join("\n") + "\n";
    Report { text, failed }
}
