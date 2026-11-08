//! Sampling settings and the sampler (Spec §8, FR-26 to FR-30).

use crate::Error;

/// How the next token is chosen (FR-26). The default is greedy (FR-29).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for Sampling {
    fn default() -> Self {
        Sampling { temperature: 0.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: None }
    }
}

impl Sampling {
    /// Rejects an invalid value naming the option and the value (FR-30). A
    /// clamped value would run a different distribution from the one asked for.
    pub fn validate(&self) -> Result<(), Error> {
        let reject = |option: &str, value: f32, rule: &str| {
            Err(Error::Rejected(format!("invalid {option} {value}: {rule}")))
        };
        if !(self.temperature.is_finite() && self.temperature >= 0.0) {
            return reject("temperature", self.temperature, "must be a finite number, at least 0");
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return reject("top-p", self.top_p, "must be greater than 0 and at most 1");
        }
        if !(self.repetition_penalty.is_finite() && self.repetition_penalty > 0.0) {
            return reject("repetition-penalty", self.repetition_penalty, "must be a finite number greater than 0");
        }
        Ok(())
    }
}

/// Repetition penalty (FR-27): for each distinct token already in the context,
/// a positive logit is divided by the penalty and a negative one multiplied by
/// it. Ids outside the vocabulary are ignored, as the reference ignores them.
pub(crate) fn apply_repetition_penalty(logits: &mut [f32], context: &[u32], penalty: f32) {
    let mut seen = vec![false; logits.len()];
    for &id in context {
        let id = id as usize;
        if id < logits.len() && !seen[id] {
            seen[id] = true;
            let l = &mut logits[id];
            *l = if *l < 0.0 { *l * penalty } else { *l / penalty };
        }
    }
}

/// Temperature (FR-26): every logit is divided by it.
pub(crate) fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}

/// Top-k (FR-26): tokens whose logit is below the k-th highest are removed, so
/// tokens tied with the k-th stay, as in the reference library. A k of 0, or at
/// least the vocabulary, removes nothing.
pub(crate) fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut copy = logits.to_vec();
    let (_, &mut kth, _) = copy.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    for l in logits.iter_mut() {
        if *l < kth {
            *l = f32::NEG_INFINITY;
        }
    }
}

/// Top-p (FR-26), computed as the reference library does: the probabilities are
/// sorted ascending and a token is removed when the running sum up to and
/// including it is at most 1 - p. That keeps the smallest set of highest
/// probability tokens whose sum is at least p, and the highest token always
/// stays. The sums are in f64.
pub(crate) fn apply_top_p(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let probs = softmax_f64(logits);
    let mut ascending: Vec<usize> = (0..logits.len()).collect();
    ascending.sort_by(|&a, &b| logits[a].total_cmp(&logits[b]));
    let threshold = 1.0 - f64::from(p);
    let mut cumulative = 0.0f64;
    let last = ascending.len() - 1;
    for (rank, &i) in ascending.iter().enumerate() {
        cumulative += probs[i];
        if rank < last && cumulative <= threshold {
            logits[i] = f32::NEG_INFINITY;
        }
    }
}

/// Softmax in f64 of f32 logits; a logit of negative infinity gets probability 0.
pub(crate) fn softmax_f64(logits: &[f32]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f64> = logits.iter().map(|&l| f64::from(l - max).exp()).collect();
    let total: f64 = probs.iter().sum();
    for p in &mut probs {
        *p /= total;
    }
    probs
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rejected(s: Sampling) -> String {
        match s.validate() {
            Err(Error::Rejected(message)) => message,
            other => panic!("expected a rejected request, got {other:?}"),
        }
    }

    // FR-30: an invalid value is rejected naming the option and the value.
    #[test]
    fn fr30_a_negative_or_non_finite_temperature_is_rejected_naming_it() {
        for bad in [-0.5f32, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let message = rejected(Sampling { temperature: bad, ..Sampling::default() });
            assert!(message.contains("temperature") && message.contains(&bad.to_string()), "{message}");
        }
    }

    #[test]
    fn fr30_a_top_p_outside_zero_exclusive_to_one_is_rejected_naming_it() {
        for bad in [0.0f32, -0.1, 1.0001, f32::NAN, f32::INFINITY] {
            let message = rejected(Sampling { top_p: bad, ..Sampling::default() });
            assert!(message.contains("top-p") && message.contains(&bad.to_string()), "{message}");
        }
    }

    #[test]
    fn fr30_a_penalty_of_zero_or_less_or_non_finite_is_rejected_naming_it() {
        for bad in [0.0f32, -1.0, f32::NAN, f32::INFINITY] {
            let message = rejected(Sampling { repetition_penalty: bad, ..Sampling::default() });
            assert!(message.contains("repetition-penalty") && message.contains(&bad.to_string()), "{message}");
        }
    }

    #[test]
    fn fr30_boundary_values_are_accepted() {
        let ok = |s: Sampling| s.validate().unwrap();
        ok(Sampling::default());
        ok(Sampling { temperature: 0.0, ..Sampling::default() });
        ok(Sampling { temperature: 1e-6, ..Sampling::default() });
        ok(Sampling { top_p: 1.0, ..Sampling::default() });
        ok(Sampling { top_p: f32::MIN_POSITIVE, ..Sampling::default() });
        ok(Sampling { repetition_penalty: f32::MIN_POSITIVE, ..Sampling::default() });
        ok(Sampling { top_k: 0, ..Sampling::default() });
        ok(Sampling { top_k: usize::MAX, seed: Some(u64::MAX), ..Sampling::default() });
    }

    #[test]
    fn fr29_the_default_is_greedy_with_every_other_control_neutral() {
        let d = Sampling::default();
        assert_eq!((d.temperature, d.top_k, d.top_p, d.repetition_penalty, d.seed), (0.0, 0, 1.0, 1.0, None));
    }

    // FR-27: positive logits are divided, negative multiplied, once per distinct token.
    #[test]
    fn fr27_the_penalty_divides_positive_and_multiplies_negative_logits() {
        let mut l = [2.0f32, -1.0, 0.5, 3.0, 0.0];
        apply_repetition_penalty(&mut l, &[0, 1, 1, 4, 99], 2.0);
        // token 1 appears twice and is penalized once; zero stays zero; the
        // out-of-range id 99 is ignored; tokens 2 and 3 are not in the context.
        assert_eq!(l, [1.0, -2.0, 0.5, 3.0, 0.0]);
    }

    #[test]
    fn fr27_a_penalty_below_one_raises_the_context_tokens() {
        let mut l = [2.0f32, -1.0];
        apply_repetition_penalty(&mut l, &[0, 1], 0.5);
        assert_eq!(l, [4.0, -0.5]);
    }

    // FR-26: temperature divides the logits.
    #[test]
    fn fr26_temperature_divides_every_logit() {
        let mut l = [2.0f32, 4.0, -2.0];
        apply_temperature(&mut l, 2.0);
        assert_eq!(l, [1.0, 2.0, -1.0]);
    }

    fn kept(l: &[f32]) -> Vec<usize> {
        l.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, _)| i).collect()
    }

    #[test]
    fn fr26_top_k_keeps_the_k_highest_and_removes_the_rest() {
        let mut l = [1.0f32, 5.0, 3.0, 4.0, 2.0];
        apply_top_k(&mut l, 2);
        assert_eq!(kept(&l), [1, 3]);
        assert_eq!((l[1], l[3]), (5.0, 4.0));
        assert!(l[0] == f32::NEG_INFINITY && l[2] == f32::NEG_INFINITY && l[4] == f32::NEG_INFINITY);
    }

    // The reference library keeps every token tied with the k-th (Spec §8).
    #[test]
    fn fr26_top_k_keeps_tokens_tied_with_the_kth() {
        let mut l = [3.0f32, 3.0, 3.0, 1.0];
        apply_top_k(&mut l, 2);
        assert_eq!(kept(&l), [0, 1, 2]);
    }

    #[test]
    fn fr26_top_k_zero_or_at_least_the_vocabulary_changes_nothing() {
        for k in [0, 4, 5, 1000] {
            let mut l = [1.0f32, 5.0, 3.0, 4.0];
            apply_top_k(&mut l, k);
            assert_eq!(l, [1.0, 5.0, 3.0, 4.0], "k = {k}");
        }
        let mut l = [1.0f32, 5.0, 3.0, 4.0];
        apply_top_k(&mut l, 1);
        assert_eq!(kept(&l), [1]);
    }

    // Probabilities 0.5, 0.3, 0.15, 0.05 as logits, in a shuffled order.
    fn logits_for_halves() -> [f32; 4] {
        [0.15f32.ln(), 0.5f32.ln(), 0.05f32.ln(), 0.3f32.ln()]
    }

    #[test]
    fn fr26_top_p_keeps_the_smallest_set_whose_probabilities_sum_to_at_least_p() {
        for (p, want) in [(0.4f32, vec![1]), (0.79, vec![1, 3]), (0.81, vec![0, 1, 3]), (0.96, vec![0, 1, 2, 3])] {
            let mut l = logits_for_halves();
            apply_top_p(&mut l, p);
            assert_eq!(kept(&l), want, "p = {p}");
        }
        // 0.5 + 0.3 + 0.15 = 0.95 < 0.99, so every token is needed.
        let mut l = logits_for_halves();
        apply_top_p(&mut l, 0.99);
        assert_eq!(kept(&l), [0, 1, 2, 3]);
    }

    #[test]
    fn fr26_top_p_always_keeps_at_least_one_token() {
        let mut l = logits_for_halves();
        apply_top_p(&mut l, 1e-9);
        assert_eq!(kept(&l), [1]);
    }

    #[test]
    fn fr26_top_p_of_one_changes_nothing() {
        let mut l = logits_for_halves();
        apply_top_p(&mut l, 1.0);
        assert_eq!(l, logits_for_halves());
    }

    #[test]
    fn fr26_top_p_counts_only_the_tokens_top_k_left() {
        // After top-k keeps {0.5, 0.3}, their probabilities renormalize to 0.625 and 0.375.
        let mut l = logits_for_halves();
        apply_top_k(&mut l, 2);
        apply_top_p(&mut l, 0.6);
        assert_eq!(kept(&l), [1]);
        let mut l = logits_for_halves();
        apply_top_k(&mut l, 2);
        apply_top_p(&mut l, 0.7);
        assert_eq!(kept(&l), [1, 3]);
    }
}
