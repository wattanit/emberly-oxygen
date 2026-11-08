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
}
