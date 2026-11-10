//! Logging through the chat command: standard output is the same at every
//! verbosity level (S-8), and each turn is logged like a generation (FR-23).

mod common;

use common::{chat, Run};

/// Open-ended messages and a hot temperature, so the draws matter: a log level that consumed or
/// skipped one draw would change the replies.
const INPUT: &str = "Tell me about the sea.\nAnd about the sky?\n";

/// A seeded, sampled chat of two turns at each level: default, -v, -vv, -vvv.
fn runs() -> [Run; 4] {
    ["", "-v", "-vv", "-vvv"].map(|flag| {
        let mut args = vec!["--max-new-tokens", "16", "--seed", "5", "--temperature", "1.5", "--top-p", "1"];
        if !flag.is_empty() {
            args.insert(0, flag);
        }
        let r = chat(&args, INPUT);
        assert_eq!(r.code, 0, "{flag}: {}", r.stderr);
        r
    })
}

fn field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let at = line.find(&format!(" {name}="))? + name.len() + 2;
    Some(line[at..].split(' ').next().unwrap())
}

fn number(line: &str, name: &str) -> usize {
    field(line, name).unwrap_or_else(|| panic!("{name} missing: {line}")).parse().unwrap()
}

fn step_events(stderr: &str) -> Vec<&str> {
    stderr.lines().filter(|l| l.contains(" DEBUG ") && l.contains("generate: step phase=")).collect()
}

// S-8, FR-20: a sampled chat's standard output is byte-identical at the default level, -v, -vv and
// -vvv, and the default level writes only the seed and the prompt markers to standard error.
#[test]
fn s8_chat_output_is_identical_at_every_level() {
    let runs = runs();
    assert!(!runs[0].stdout.is_empty());
    for (flag, r) in ["-v", "-vv", "-vvv"].iter().zip(&runs[1..]) {
        assert_eq!(r.stdout, runs[0].stdout, "{flag}");
    }
    assert_eq!(runs[0].stderr, "seed: 5\n> > > \n", "nothing is logged by default");
}

// FR-23: -v logs each turn's prompt and run summary, and the history makes the second prompt longer.
#[test]
fn fr23_each_turn_logs_its_prompt_and_summary_at_info_level() {
    let [_, v, _, _] = runs();
    let prompts: Vec<usize> =
        v.stderr.lines().filter(|l| l.contains("prompt encoded")).map(|l| number(l, "prompt_tokens")).collect();
    let summaries: Vec<&str> = v.stderr.lines().filter(|l| l.contains("run summary")).collect();
    assert_eq!(prompts.len(), 2, "{}", v.stderr);
    assert_eq!(summaries.len(), 2);
    assert!(prompts[1] > prompts[0], "the second prompt holds the first turn too: {prompts:?}");
    assert!(!v.stderr.contains(" DEBUG "));
}

// FR-23, FR-40: -vv gives one prefill event per turn and one step event per decode step, inside
// generate -> prefill{tokens} and generate -> step{pos}, with the probability and kept tokens of each
// sampled choice.
#[test]
fn fr23_each_turn_logs_its_steps_at_debug_level() {
    let [_, v, vv, _] = runs();
    let prompts: Vec<usize> =
        v.stderr.lines().filter(|l| l.contains("prompt encoded")).map(|l| number(l, "prompt_tokens")).collect();
    let generated: Vec<usize> =
        v.stderr.lines().filter(|l| l.contains("run summary")).map(|l| number(l, "generated_tokens")).collect();
    let prefills: Vec<&str> = vv.stderr.lines().filter(|l| l.contains(" DEBUG ") && l.contains(": prefill tokens=")).collect();
    assert_eq!(prefills.len(), prompts.len(), "one prefill per turn");
    for (prefill, prompt) in prefills.iter().zip(&prompts) {
        assert_eq!(number(prefill, "tokens"), *prompt, "{prefill}");
    }
    let steps = step_events(&vv.stderr);
    let expected: usize = generated.iter().map(|g| g - 1).sum();
    assert_eq!(steps.len(), expected, "the last generated token of a turn is not fed");
    let chosen: Vec<&&str> = prefills.iter().chain(&steps).collect();
    assert_eq!(chosen.len(), generated.iter().sum::<usize>(), "every generated token is chosen once");
    for event in chosen {
        assert!(
            field(event, "probability").is_some() && field(event, "kept").is_some(),
            "{event}"
        );
    }
    // The prompt is processed again each turn, so each turn's first decode step is at its own prompt's length.
    assert!(steps.iter().any(|s| s.contains(&format!(" generate:step{{pos={}}}: ", prompts[0]))));
    assert!(steps.iter().any(|s| s.contains(&format!(" generate:step{{pos={}}}: ", prompts[1]))));
}

// FR-23: -vvv adds the per-layer events, and still no change to standard output.
#[test]
fn fr23_trace_level_adds_the_layer_events_in_chat() {
    let [_, _, vv, vvv] = runs();
    assert!(!vv.stderr.contains(" TRACE "));
    assert!(vvv.stderr.contains(" TRACE ") && vvv.stderr.contains("layer"), "{}", &vvv.stderr[..400.min(vvv.stderr.len())]);
}
