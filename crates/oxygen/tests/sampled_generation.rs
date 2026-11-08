//! Sampled generation (FR-26 to FR-29): seeded runs repeat, greedy is unchanged,
//! temperature 0 ignores the other controls, and invalid or incomplete requests
//! are rejected before any computation. These run the real model.

mod common;

use common::fixtures::{require_model_dir, workspace_root};
use emberly_oxygen::{generate, generate_with, Error, Model, Sampling, Tokenizer};

const PROMPT: &str = "The quick brown fox jumps over the lazy dog, and then";
const NEW_TOKENS: usize = 16;

fn load() -> (Model, Tokenizer) {
    let dir = workspace_root().join("models").join("smollm2-135m");
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    (Model::load(&dir).unwrap(), Tokenizer::load(&dir).unwrap())
}

fn sampled(seed: u64) -> Sampling {
    Sampling { temperature: 0.8, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, seed: Some(seed) }
}

fn ids(model: &Model, tokenizer: &Tokenizer, sampling: &Sampling) -> Vec<u32> {
    generate_with(model, tokenizer, PROMPT, NEW_TOKENS, sampling, &mut |_| {}).unwrap().ids
}

#[test]
fn fr28_the_same_seed_gives_the_same_tokens() {
    let (m, t) = load();
    let first = ids(&m, &t, &sampled(1234));
    assert_eq!(first, ids(&m, &t, &sampled(1234)));
    assert!(!first.is_empty());
}

#[test]
fn fr28_different_seeds_give_different_tokens() {
    let (m, t) = load();
    let a = ids(&m, &t, &sampled(1));
    let b = ids(&m, &t, &sampled(2));
    assert_ne!(a, b, "two seeds at temperature 0.8 over {NEW_TOKENS} tokens should differ");
}

#[test]
fn fr28_sampled_generation_reports_its_seed_and_greedy_reports_none() {
    let (m, t) = load();
    let run = generate_with(&m, &t, PROMPT, 4, &sampled(99), &mut |_| {}).unwrap();
    assert_eq!(run.seed, Some(99));
    let run = generate_with(&m, &t, PROMPT, 4, &Sampling::default(), &mut |_| {}).unwrap();
    assert_eq!(run.seed, None);
}

#[test]
fn fr29_the_default_settings_give_the_greedy_output() {
    let (m, t) = load();
    let greedy = generate(&m, &t, PROMPT, NEW_TOKENS, &mut |_| {}).unwrap().ids;
    assert_eq!(ids(&m, &t, &Sampling::default()), greedy);
}

#[test]
fn fr29_temperature_zero_ignores_every_other_control() {
    let (m, t) = load();
    let greedy = generate(&m, &t, PROMPT, NEW_TOKENS, &mut |_| {}).unwrap().ids;
    let all_set = Sampling { temperature: 0.0, top_k: 5, top_p: 0.3, repetition_penalty: 5.0, seed: Some(7) };
    assert_eq!(ids(&m, &t, &all_set), greedy);
}

// The controls are wired in: top-k 1 keeps only the highest token, so a sampled
// run with it is the greedy run whatever the seed. A sampler that ignored top-k
// would not repeat the greedy output.
#[test]
fn fr26_top_k_one_samples_the_greedy_tokens() {
    let (m, t) = load();
    let greedy = generate(&m, &t, PROMPT, NEW_TOKENS, &mut |_| {}).unwrap().ids;
    for seed in [1, 2, 3] {
        let s = Sampling { top_k: 1, ..sampled(seed) };
        assert_eq!(ids(&m, &t, &s), greedy, "seed {seed}");
    }
}

// The penalty is wired in and counts the prompt. After "cat cat ... cat" the
// highest token is a token of the prompt itself (2644, " cat"); with a large
// penalty, and only the prompt as context (one new token, so nothing generated
// yet), the first token must change to one outside the prompt. A sampler that
// left the prompt out of the penalty's context would repeat 2644.
#[test]
fn fr27_the_penalty_counts_the_prompts_own_tokens() {
    let (m, t) = load();
    let prompt = "cat cat cat cat cat cat cat";
    let prompt_ids = t.encode(prompt).unwrap();
    let first = |penalty: f32| {
        let s = Sampling { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: penalty, seed: Some(1) };
        generate_with(&m, &t, prompt, 1, &s, &mut |_| {}).unwrap().ids[0]
    };
    let plain = first(1.0);
    assert!(prompt_ids.contains(&plain), "the plain first token {plain} should repeat the prompt");
    let penalized = first(50.0);
    assert!(!prompt_ids.contains(&penalized), "the penalized first token {penalized} should leave the prompt's tokens");
}

// Generated tokens join the context too: with a large penalty and top-k 1, no
// token is chosen twice in a short run.
#[test]
fn fr27_the_penalty_counts_the_tokens_already_generated() {
    let (m, t) = load();
    let s = Sampling { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 50.0, seed: Some(1) };
    let ids = generate_with(&m, &t, PROMPT, 12, &s, &mut |_| {}).unwrap().ids;
    let mut seen = std::collections::HashSet::new();
    assert!(ids.iter().all(|id| seen.insert(*id)), "a token repeated under a penalty of 50: {ids:?}");
}

#[test]
fn fr28_sampling_without_a_seed_is_rejected_naming_the_seed() {
    let (m, t) = load();
    let s = Sampling { seed: None, ..sampled(0) };
    match generate_with(&m, &t, PROMPT, 4, &s, &mut |_| {}) {
        Err(Error::Rejected(message)) => assert!(message.contains("seed"), "{message}"),
        other => panic!("expected a rejected request, got {other:?}"),
    }
}

#[test]
fn fr30_an_invalid_setting_is_rejected_before_any_computation() {
    let (m, t) = load();
    let s = Sampling { temperature: -1.0, ..sampled(1) };
    let mut emitted = false;
    let result = generate_with(&m, &t, PROMPT, 4, &s, &mut |_| emitted = true);
    assert!(matches!(result, Err(Error::Rejected(m)) if m.contains("temperature")));
    assert!(!emitted);
}

#[test]
fn fr26_sampled_generation_stops_at_the_maximum_number_of_tokens() {
    let (m, t) = load();
    for n in [0, 1, 5] {
        let run = generate_with(&m, &t, PROMPT, n, &sampled(5), &mut |_| {}).unwrap();
        assert!(run.ids.len() <= n);
        assert_eq!(run.stats.generated_tokens, run.ids.len());
    }
}

/// The settings of the guard below: greedy, two plain seeds, and all four controls together.
fn guard_cases() -> Vec<(&'static str, Sampling)> {
    vec![
        ("greedy", Sampling::default()),
        ("seed 1", sampled(1)),
        ("seed 2", sampled(2)),
        (
            "combined",
            Sampling { temperature: 0.7, top_k: 40, top_p: 0.9, repetition_penalty: 1.3, seed: Some(7) },
        ),
    ]
}

// FR-28, FR-29: a guard for refactoring the step loop (plan 0.3.0 task 16.1). The ids below
// were recorded from the code before the loop was reorganized so a chat turn could own its
// generator; they must not change with the reorganization. They pin this build on the
// benchmark machine (FR-28 promises reproduction within a build only): if a deliberate change
// to the generator, the controls or the arithmetic moves them, record them again.
#[test]
fn fr28_fr29_greedy_and_seeded_output_is_unchanged_by_the_step_loop_refactor() {
    let (m, t) = load();
    let expected: [(&str, &[u32]); 4] = [
        ("greedy", &[260, 2767, 27003, 690, 260, 2644, 30, 198, 198, 504, 2767, 314, 253, 2767, 28, 284]),
        ("seed 1", &[656, 260, 13917, 6391, 47, 198, 198, 504, 16438, 553, 17986, 12364, 284, 5988, 288, 3134]),
        ("seed 2", &[260, 4255, 359, 12027, 281, 260, 4480, 30, 216, 15442, 3528, 339, 3363, 3590, 10761, 198]),
        ("combined", &[665, 314, 253, 27216, 6621, 351, 253, 1805, 7706, 30, 378, 6715, 30858, 1187, 288, 1138]),
    ];
    for ((name, sampling), (expected_name, want)) in guard_cases().iter().zip(expected) {
        assert_eq!(*name, expected_name);
        let got = ids(&m, &t, sampling);
        assert_eq!(got.as_slice(), want, "{name}: recorded as {got:?}");
    }
}
