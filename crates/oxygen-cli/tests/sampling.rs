//! Runs the built binary with sampling options and checks what a user sees
//! (Spec §9, §8): the options reach the engine, invalid values are rejected naming
//! the option, the seed is always reported on standard error when sampling and
//! replays the run, and a greedy run is as it always was.

mod common;

use common::{generate, Run};

const PROMPT: &str = "The quick brown fox jumps over the lazy dog, and then";

fn run(extra: &[&str]) -> Run {
    let mut args = vec!["--max-new-tokens", "16"];
    args.extend(extra);
    args.push(PROMPT);
    generate(&args)
}

fn text(r: &Run) -> String {
    String::from_utf8(r.stdout.clone()).unwrap()
}

/// The number in a `seed: <n>` line of standard error.
fn seed_of(r: &Run) -> u64 {
    let line = r.stderr.lines().find(|l| l.starts_with("seed: ")).unwrap_or_else(|| panic!("no seed line in {:?}", r.stderr));
    line["seed: ".len()..].trim().parse().unwrap()
}

// FR-30: an invalid value exits 1, prints nothing to standard output, and names the option and value.
#[test]
fn fr30_invalid_values_are_rejected_naming_the_option_and_the_value() {
    for (arg, option, value) in [
        ("--temperature=-1", "temperature", "-1"),
        ("--top-p=1.5", "top-p", "1.5"),
        ("--top-p=0", "top-p", "0"),
        ("--repetition-penalty=0", "repetition-penalty", "0"),
    ] {
        let r = run(&[arg, "--seed", "1"]);
        assert_eq!(r.code, 1, "{arg}: {}", r.stderr);
        assert!(r.stdout.is_empty(), "{arg}: standard output {:?}", text(&r));
        assert!(r.stderr.starts_with("error: ") && r.stderr.contains(option) && r.stderr.contains(value), "{arg}: {}", r.stderr);
    }
}

// FR-28, FR-20: sampling prints the seed it used to standard error, given or not.
#[test]
fn fr28_a_given_seed_is_reported_on_standard_error() {
    let r = run(&["--temperature", "0.8", "--seed", "4242"]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(seed_of(&r), 4242);
    assert!(!text(&r).contains("seed"), "standard output is generated text only");
}

#[test]
fn fr28_a_run_without_a_seed_reports_the_one_it_took_and_replays_with_it() {
    let first = run(&["--temperature", "0.8"]);
    assert_eq!(first.code, 0, "{}", first.stderr);
    let seed = seed_of(&first);
    let replay = run(&["--temperature", "0.8", "--seed", &seed.to_string()]);
    assert_eq!(text(&replay), text(&first), "the reported seed replays the run");
}

#[test]
fn fr28_the_seed_decides_the_text() {
    let a = run(&["--temperature", "0.8", "--seed", "1"]);
    let b = run(&["--temperature", "0.8", "--seed", "2"]);
    assert_eq!(text(&a), text(&run(&["--temperature", "0.8", "--seed", "1"])));
    assert_ne!(text(&a), text(&b));
}

// FR-29: with no sampling option the output is the greedy output and standard error is empty.
#[test]
fn fr29_a_run_without_sampling_options_is_greedy_and_prints_no_seed() {
    let r = run(&[]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(r.stderr, "");
    let explicit = run(&["--temperature", "0"]);
    assert_eq!(text(&explicit), text(&r));
    assert_eq!(explicit.stderr, "", "temperature 0 is greedy: no seed line");
}

// Spec §8: temperature 0 ignores the other controls, and says so.
#[test]
fn fr29_temperature_zero_ignores_other_controls_with_a_warning() {
    let greedy = run(&[]);
    let r = run(&["--temperature", "0", "--top-k", "3", "--repetition-penalty", "5"]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(text(&r), text(&greedy));
    assert!(r.stderr.contains("ignored") && r.stderr.contains("temperature 0"), "{}", r.stderr);
}

// FR-26, FR-30: the options reach the engine: top-k 1 samples the greedy text.
#[test]
fn fr26_top_k_one_reaches_the_engine() {
    let greedy = run(&[]);
    let r = run(&["--temperature", "1", "--top-k", "1", "--seed", "9"]);
    assert_eq!(text(&r), text(&greedy));
}

#[test]
fn fr30_the_help_lists_every_sampling_option() {
    let r = common::run(&["generate", "--help"], None);
    for option in ["--temperature", "--top-k", "--top-p", "--repetition-penalty", "--seed"] {
        assert!(String::from_utf8_lossy(&r.stdout).contains(option), "{option} missing from the help");
    }
}
