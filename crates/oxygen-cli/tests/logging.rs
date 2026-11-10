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

/// The debug prefill events, in order.
fn prefill_events(stderr: &str) -> Vec<&str> {
    stderr.lines().filter(|l| l.contains(" DEBUG ") && l.contains(": prefill tokens=")).collect()
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

    // -vv: one `prefill` event for the whole prompt, then one event per decode step, in position
    // order inside generate -> step{pos} from the first position after the prompt. The prefill
    // reports the first generated token as `next`, and every generated token appears once as
    // `next`; each step feeds the token the event before it chose.
    assert!(!vv.stderr.contains(" TRACE "));
    let prefills = prefill_events(&vv.stderr);
    assert_eq!(prefills.len(), 1, "{}", vv.stderr);
    assert!(prefills[0].contains(&format!(" generate:prefill{{tokens={prompt_tokens}}}: ")), "{}", prefills[0]);
    assert_eq!(number(prefills[0], "tokens"), prompt_tokens);
    for name in ["top5", "prefill_ms", "next"] {
        assert!(field(prefills[0], name).is_some(), "{name} missing: {}", prefills[0]);
    }
    let steps = step_events(&vv.stderr);
    assert_eq!(steps.len(), NEW_TOKENS - 1, "the last generated token is not fed");
    for (i, step) in steps.iter().enumerate() {
        assert!(step.contains(&format!(" generate:step{{pos={}}}: ", prompt_tokens + i)), "{step}");
        assert_eq!(field(step, "phase"), Some("\"generate\""), "{step}");
        for name in ["text", "top5", "step_ms", "next"] {
            assert!(field(step, name).is_some(), "{name} missing: {step}");
        }
    }
    let nexts: Vec<usize> =
        prefills.iter().chain(&steps).filter_map(|s| field(s, "next")).map(|n| n.parse().unwrap()).collect();
    assert_eq!(nexts.len(), NEW_TOKENS);
    let fed: Vec<usize> = steps.iter().map(|s| number(s, "token")).collect();
    assert_eq!(fed, nexts[..NEW_TOKENS - 1], "each generated token but the last is fed back");

    // -vvv: one event per layer for the prefill (describing the prompt's last row) and one per layer
    // per decode step, each carrying the full span path, with both norms and a most-attended
    // position for each of the 9 heads.
    let layers: Vec<&str> = vvv.stderr.lines().filter(|l| l.contains(" TRACE ")).collect();
    assert_eq!(step_events(&vvv.stderr).len(), steps.len());
    assert_eq!(layers.len(), (1 + steps.len()) * 30);
    for (n, line) in layers.iter().enumerate() {
        let (unit, idx) = (n / 30, n % 30);
        let (unit_path, last_position) = if unit == 0 {
            (format!("prefill{{tokens={prompt_tokens}}}"), prompt_tokens - 1)
        } else {
            (format!("step{{pos={}}}", prompt_tokens + unit - 1), prompt_tokens + unit - 1)
        };
        assert!(line.contains(&format!(" generate:{unit_path}:layer{{idx={idx}}}: ")), "{line}");
        for name in ["residual_norm_after_attention", "residual_norm_after_mlp"] {
            assert!(field(line, name).unwrap().parse::<f32>().unwrap() > 0.0, "{line}");
        }
        let heads = line.split("most_attended=[").nth(1).unwrap().trim_end_matches(']');
        let heads: Vec<usize> = heads.split(", ").map(|h| h.parse().unwrap()).collect();
        assert_eq!(heads.len(), 9, "{line}");
        assert!(heads.iter().all(|&h| h <= last_position), "{line}");
    }
}
