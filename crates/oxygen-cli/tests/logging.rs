//! Logging through the built binary: standard output is the same at every
//! verbosity level (S-8), and each level shows what it promises (FR-23).

mod common;

use common::{generate, Run};

const PROMPT: &str = "The quick brown fox";
const NEW_TOKENS: usize = 6;

/// One run at each level: default, -v, -vv, -vvv.
fn runs() -> [Run; 4] {
    let n = NEW_TOKENS.to_string();
    ["", "-v", "-vv", "-vvv"].map(|flag| {
        let mut args = vec!["--max-new-tokens", &n, PROMPT];
        if !flag.is_empty() {
            args.insert(0, flag);
        }
        let r = generate(&args);
        assert_eq!(r.code, 0, "{flag}: {}", r.stderr);
        r
    })
}

/// The value of `name=` on a formatted log line, up to the next space.
fn field<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let at = line.find(&format!(" {name}="))? + name.len() + 2;
    Some(line[at..].split(' ').next().unwrap())
}

fn number(line: &str, name: &str) -> usize {
    field(line, name).unwrap_or_else(|| panic!("{name} missing: {line}")).parse().unwrap()
}

/// The debug step events, in order.
fn step_events(stderr: &str) -> Vec<&str> {
    stderr.lines().filter(|l| l.contains(" DEBUG ") && l.contains("generate: step phase=")).collect()
}

// S-8, FR-20: standard output is byte-identical at the default level, -v, -vv and -vvv, and
// the default level writes nothing to standard error.
#[test]
fn output_is_identical_at_every_level() {
    let runs = runs();
    assert!(!runs[0].stdout.is_empty());
    for (flag, r) in ["-v", "-vv", "-vvv"].iter().zip(&runs[1..]) {
        assert_eq!(r.stdout, runs[0].stdout, "{flag}");
    }
    assert_eq!(runs[0].stderr, "");
}

// FR-23, Spec §14: each level shows what it promises and nothing of the next one.
#[test]
fn each_level_shows_what_it_promises() {
    let [_, v, vv, vvv] = runs();

    // -v: the model summary, the prompt's token count and the run summary; no steps.
    for message in ["model loaded", "prompt encoded", "run summary"] {
        assert!(v.stderr.contains(message), "-v lacks {message:?}: {}", v.stderr);
    }
    assert!(!v.stderr.contains(" DEBUG ") && !v.stderr.contains(" TRACE "), "{}", v.stderr);
    let summary = v.stderr.lines().find(|l| l.contains("run summary")).unwrap();
    let prompt_tokens = number(summary, "prompt_tokens");
    assert_eq!(number(summary, "generated_tokens"), NEW_TOKENS);

    // -vv: one event per forward step, in position order inside generate -> step{pos}. Every
    // prompt token appears once as `token`, marked as a prompt step, and every generated token
    // once as `next`; each step feeds the token the step before it chose.
    assert!(!vv.stderr.contains(" TRACE "));
    let steps = step_events(&vv.stderr);
    assert_eq!(steps.len(), prompt_tokens + NEW_TOKENS - 1, "the last generated token is not fed");
    for (pos, step) in steps.iter().enumerate() {
        assert!(step.contains(&format!(" generate:step{{pos={pos}}}: ")), "{step}");
        let phase = if pos < prompt_tokens { "\"prompt\"" } else { "\"generate\"" };
        assert_eq!(field(step, "phase"), Some(phase), "{step}");
        for name in ["text", "top5", "step_ms"] {
            assert!(field(step, name).is_some(), "{name} missing: {step}");
        }
        assert_eq!(field(step, "next").is_some(), pos + 1 >= prompt_tokens, "only steps whose result is taken: {step}");
    }
    let nexts: Vec<usize> = steps.iter().filter_map(|s| field(s, "next")).map(|n| n.parse().unwrap()).collect();
    assert_eq!(nexts.len(), NEW_TOKENS);
    let fed: Vec<usize> = steps[prompt_tokens..].iter().map(|s| number(s, "token")).collect();
    assert_eq!(fed, nexts[..NEW_TOKENS - 1], "each generated token but the last is fed back");

    // -vvv: one event per layer per step, each carrying the full span path, with both norms and
    // a most-attended position for each of the 9 heads.
    let layers: Vec<&str> = vvv.stderr.lines().filter(|l| l.contains(" TRACE ")).collect();
    assert_eq!(step_events(&vvv.stderr).len(), steps.len());
    assert_eq!(layers.len(), steps.len() * 30);
    for (n, line) in layers.iter().enumerate() {
        let (pos, idx) = (n / 30, n % 30);
        assert!(line.contains(&format!(" generate:step{{pos={pos}}}:layer{{idx={idx}}}: ")), "{line}");
        for name in ["residual_norm_after_attention", "residual_norm_after_mlp"] {
            assert!(field(line, name).unwrap().parse::<f32>().unwrap() > 0.0, "{line}");
        }
        let heads = line.split("most_attended=[").nth(1).unwrap().trim_end_matches(']');
        let heads: Vec<usize> = heads.split(", ").map(|h| h.parse().unwrap()).collect();
        assert_eq!(heads.len(), 9, "{line}");
        assert!(heads.iter().all(|&h| h <= pos), "{line}");
    }
}
