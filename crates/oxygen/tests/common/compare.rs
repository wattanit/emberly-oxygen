//! The correctness checks as pure functions (S-1 to S-4). Each reports the
//! first diverging position (FR-6), not only pass or fail. Bad shapes and
//! non-finite values are failures with a message, never panics.

/// S-1: largest absolute logit difference allowed at every position.
pub const LOGIT_TOLERANCE: f32 = 1e-3;
/// S-2: largest difference allowed, relative to the largest magnitude in the
/// reference tensor.
pub const INTERMEDIATE_TOLERANCE: f32 = 1e-4;
/// S-3: two reference logits closer than this form a near-tie, where either
/// token passes.
pub const NEAR_TIE_GAP: f32 = 1e-3;

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Pass,
    /// Greedy comparison stopped at a near-tie (S-4); everything before it matched.
    NearTie { position: usize },
    Fail { position: Option<usize>, message: String },
}

impl Verdict {
    pub fn is_failure(&self) -> bool {
        matches!(self, Verdict::Fail { .. })
    }
}

/// A numeric check's verdict with the worst error it measured, so the first
/// real measurement can be recorded (Plan §8).
#[derive(Debug, Clone, PartialEq)]
pub struct Numeric {
    pub max_error: f32,
    pub verdict: Verdict,
}

fn fail(position: impl Into<Option<usize>>, message: impl Into<String>) -> Verdict {
    Verdict::Fail { position: position.into(), message: message.into() }
}

fn numeric_fail(position: impl Into<Option<usize>>, message: impl Into<String>) -> Numeric {
    Numeric { max_error: f32::INFINITY, verdict: fail(position, message) }
}

/// Checks two `[positions, width]` matrices have the same, non-empty shape.
// `%` rather than `is_multiple_of`, which would raise the minimum Rust version.
#[allow(clippy::manual_is_multiple_of)]
fn check_shape(engine: &[f32], reference: &[f32], width: usize) -> Result<(), String> {
    if width == 0 || reference.is_empty() || reference.len() % width != 0 {
        return Err(format!("reference has {} values, not a multiple of width {width}", reference.len()));
    }
    if engine.len() != reference.len() {
        return Err(format!(
            "engine has {} values, reference has {} ({} positions of {width})",
            engine.len(),
            reference.len(),
            reference.len() / width
        ));
    }
    Ok(())
}

fn first_non_finite(engine: &[f32], width: usize) -> Option<usize> {
    engine.iter().position(|v| !v.is_finite()).map(|i| i / width)
}

/// S-1: at every position the largest absolute logit difference is at most
/// [`LOGIT_TOLERANCE`].
pub fn logit_error(engine: &[f32], reference: &[f32], vocab: usize) -> Numeric {
    if let Err(m) = check_shape(engine, reference, vocab) {
        return numeric_fail(None, m);
    }
    if let Some(p) = first_non_finite(engine, vocab) {
        return numeric_fail(p, "engine logits contain NaN or infinity");
    }
    let mut max_error = 0f32;
    let mut first_bad = None;
    for (p, (e, r)) in engine.chunks(vocab).zip(reference.chunks(vocab)).enumerate() {
        let worst = e.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        max_error = max_error.max(worst);
        if worst > LOGIT_TOLERANCE && first_bad.is_none() {
            first_bad = Some((p, worst));
        }
    }
    let verdict = match first_bad {
        None => Verdict::Pass,
        Some((p, worst)) => fail(
            p,
            format!("logit error {worst:e} at position {p} exceeds {LOGIT_TOLERANCE:e} (largest over all positions {max_error:e})"),
        ),
    };
    Numeric { max_error, verdict }
}

/// S-2: one stored intermediate (a `[positions, width]` matrix) matches within
/// [`INTERMEDIATE_TOLERANCE`] relative to the largest magnitude in the
/// reference matrix.
pub fn intermediate_error(engine: &[f32], reference: &[f32], width: usize) -> Numeric {
    if let Err(m) = check_shape(engine, reference, width) {
        return numeric_fail(None, m);
    }
    if let Some(p) = first_non_finite(engine, width) {
        return numeric_fail(p, "engine values contain NaN or infinity");
    }
    let scale = reference.iter().fold(0f32, |m, v| m.max(v.abs()));
    if !(scale > 0.0 && scale.is_finite()) {
        return numeric_fail(None, "reference tensor is all zero or not finite, so a relative error is undefined");
    }
    let mut max_error = 0f32;
    let mut first_bad = None;
    for (p, (e, r)) in engine.chunks(width).zip(reference.chunks(width)).enumerate() {
        let worst = e.iter().zip(r).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / scale;
        max_error = max_error.max(worst);
        if worst > INTERMEDIATE_TOLERANCE && first_bad.is_none() {
            first_bad = Some((p, worst));
        }
    }
    let verdict = match first_bad {
        None => Verdict::Pass,
        Some((p, worst)) => fail(
            p,
            format!("relative error {worst:e} at position {p} exceeds {INTERMEDIATE_TOLERANCE:e} (largest over all positions {max_error:e})"),
        ),
    };
    Numeric { max_error, verdict }
}

/// Index of the largest value; the lowest index wins an exact tie, as the
/// reference's argmax does.
fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

/// S-3: the engine's top token matches the reference at every position, except
/// where the reference's two highest logits differ by less than
/// [`NEAR_TIE_GAP`]. `gaps` holds that difference per position.
pub fn top1_check(engine: &[f32], reference: &[f32], vocab: usize, gaps: &[f32]) -> Verdict {
    if let Err(m) = check_shape(engine, reference, vocab) {
        return fail(None, m);
    }
    let positions = reference.len() / vocab;
    if gaps.len() != positions {
        return fail(None, format!("{} gaps for {positions} positions", gaps.len()));
    }
    if let Some(p) = first_non_finite(engine, vocab) {
        return fail(p, "engine logits contain NaN or infinity");
    }
    for (p, (e, r)) in engine.chunks(vocab).zip(reference.chunks(vocab)).enumerate() {
        let (got, want) = (argmax(e), argmax(r));
        if got != want && gaps[p] >= NEAR_TIE_GAP {
            return fail(p, format!("top token {got} at position {p}, reference {want} (gap {:e})", gaps[p]));
        }
    }
    Verdict::Pass
}

/// S-4: the greedy continuation matches token for token. Comparison stops at
/// the first near-tie in `gaps` (the reference's top-two gap before each
/// token), because the two sequences may legitimately diverge from there.
pub fn greedy_check(engine: &[u32], reference: &[u32], gaps: &[f32]) -> Verdict {
    if gaps.len() < reference.len() {
        return fail(None, format!("{} gaps for {} reference tokens", gaps.len(), reference.len()));
    }
    for (i, want) in reference.iter().enumerate() {
        if gaps[i] < NEAR_TIE_GAP {
            return Verdict::NearTie { position: i };
        }
        match engine.get(i) {
            None => return fail(i, format!("engine stopped after {} tokens, reference has {}", engine.len(), reference.len())),
            Some(got) if got != want => {
                return fail(i, format!("token {got} at continuation index {i}, reference {want}"));
            }
            Some(_) => {}
        }
    }
    if engine.len() > reference.len() {
        return fail(reference.len(), format!("engine produced {} tokens, reference has {}", engine.len(), reference.len()));
    }
    Verdict::Pass
}
