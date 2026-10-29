//! The pass/fail table (FR-5, S-5): rows are prompts, columns are checks, and a
//! summary by category follows. A category passes only if every prompt in it
//! passes every check; no averages are used anywhere.

use std::fmt::Write;

use super::compare::Verdict;
use super::harness::{Check, CheckResult};

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

        writeln!(text, "\ncategory     prompts  result  max logit error  max intermediate error").unwrap();
        for category in distinct(rows.iter().map(|r| r.category.as_str())) {
            let of_category: Vec<_> = rows.iter().filter(|r| r.category == category).collect();
            let prompts = distinct(of_category.iter().map(|r| r.prompt.as_str()));
            let bad = prompts
                .iter()
                .filter(|p| of_category.iter().any(|r| r.prompt == **p && r.verdict.is_failure()))
                .count();
            let verdict = if bad == 0 { "pass" } else { "FAIL" };
            // Worst error measured within the category, for the checks that measure one (S-1, S-2).
            let worst = |check: Check| {
                let e = of_category.iter().filter(|r| r.check == check).filter_map(|r| r.max_error);
                e.fold(None, |m: Option<f32>, e| Some(m.map_or(e, |m| m.max(e)))).map_or("-".to_string(), |e| format!("{e:e}"))
            };
            writeln!(
                text,
                "{category:<12} {:>2}/{:<5} {verdict:<7} {:<16} {}",
                prompts.len() - bad,
                prompts.len(),
                worst(Check::Logits),
                worst(Check::Intermediates)
            )
            .unwrap();
            if bad > 0 {
                failed.push((model.to_string(), category.to_string()));
            }
        }

        let measured = |check: Check| {
            rows.iter().filter(|r| r.check == check).filter_map(|r| r.max_error).fold(None, |m: Option<f32>, e| Some(m.map_or(e, |m| m.max(e))))
        };
        text.push('\n');
        for (check, label) in [(Check::Logits, "largest logit error"), (Check::Intermediates, "largest relative intermediate error")] {
            if let Some(e) = measured(check) {
                writeln!(text, "{label}: {e:e}").unwrap();
            }
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
