//! Runs the built binary's chat command and checks what a user sees (Spec §9,
//! FR-37, FR-38): replies alone on standard output, everything else on standard
//! error, the loop's ends and skips, the defaults, and the seed.

mod common;

use common::{chat, Run};

const FRANCE: &str = "What is the capital of France?";

fn out(r: &Run) -> String {
    String::from_utf8(r.stdout.clone()).unwrap()
}

/// Greedy chat with a generous reply limit, so replies are the same every run.
const GREEDY: [&str; 4] = ["--temperature", "0", "--max-new-tokens", "64"];

fn greedy(input: &str) -> Run {
    chat(&GREEDY, input)
}

/// The number in the `seed: <n>` line of standard error.
fn seed_of(r: &Run) -> u64 {
    let line = r.stderr.lines().find(|l| l.contains("seed: ")).unwrap_or_else(|| panic!("no seed line in {:?}", r.stderr));
    line[line.find("seed: ").unwrap() + "seed: ".len()..].trim().parse().unwrap()
}

/// The `prompt_tokens` of the first info event ("prompt encoded") in a `-v` log.
fn prompt_tokens(r: &Run) -> usize {
    let line = r.stderr.lines().find(|l| l.contains("prompt encoded")).unwrap_or_else(|| panic!("no event in {:?}", r.stderr));
    let at = line.find("prompt_tokens=").unwrap() + "prompt_tokens=".len();
    line[at..].split_whitespace().next().unwrap().parse().unwrap()
}

// FR-38, FR-20: standard output holds the reply and its newline and nothing else.
#[test]
fn fr38_standard_output_holds_only_the_reply() {
    let r = greedy(&format!("{FRANCE}\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    let text = out(&r);
    assert!(text.contains("Paris") && text.ends_with('\n'), "{text:?}");
    assert_eq!(text.matches('\n').count(), 1, "one reply, one line: {text:?}");
    assert!(!text.contains('>') && !text.contains("seed") && !text.contains("<|im_end|>"), "{text:?}");
}

// FR-38: the prompt marker goes to standard error, once per line read and once at the end of input.
#[test]
fn fr38_the_prompt_marker_goes_to_standard_error() {
    let r = greedy(&format!("{FRANCE}\n"));
    assert_eq!(r.stderr.matches("> ").count(), 2, "{:?}", r.stderr);
    assert!(r.stderr.starts_with("> "), "{:?}", r.stderr);
    assert!(r.stderr.ends_with("> \n"), "the last marker is followed by a newline at end of input: {:?}", r.stderr);
}

// FR-38, FR-35: each line is a message and the conversation goes on from one turn to the next.
#[test]
fn fr38_each_line_is_a_message_and_each_gets_a_reply() {
    let r = greedy(&format!("{FRANCE}\nAnd of Italy?\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    let text = out(&r);
    assert_eq!(text.lines().count(), 2, "{text:?}");
    assert!(text.lines().next().unwrap().contains("Paris"), "{text:?}");
    assert_eq!(r.stderr.matches("> ").count(), 3);
}

// FR-38: `/exit` ends the chat with code 0, and later lines are never read as messages.
#[test]
fn fr38_slash_exit_ends_the_chat() {
    let r = greedy(&format!("{FRANCE}\n/exit\nAnd of Italy?\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(out(&r).lines().count(), 1, "{:?}", out(&r));
}

// FR-38: only the exact line `/exit` ends it; the same text inside a message is a message.
#[test]
fn fr38_only_the_exact_line_slash_exit_exits() {
    let r = greedy("say /exit\n");
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(out(&r).lines().count(), 1, "a reply was given: {:?}", out(&r));
}

// FR-38: a blank line sends nothing: no reply, and the chat goes on.
#[test]
fn fr38_a_blank_line_sends_nothing() {
    let r = greedy(&format!("\n   \n{FRANCE}\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(out(&r).lines().count(), 1, "{:?}", out(&r));
    assert_eq!(r.stderr.matches("> ").count(), 4, "{:?}", r.stderr);
}

// FR-38: end of input with no message at all is a success with nothing on standard output.
#[test]
fn fr38_end_of_input_ends_the_chat_with_success() {
    let r = greedy("");
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(r.stdout.is_empty());
}

// FR-36, FR-38: a turn that cannot fit prints its error to standard error and the chat continues.
#[test]
fn fr38_a_rejected_turn_prints_its_error_and_the_chat_continues() {
    let long = "word ".repeat(9000);
    let r = greedy(&format!("{long}\n{FRANCE}\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(r.stderr.contains("error: ") && r.stderr.contains("context length"), "{}", &r.stderr[..r.stderr.len().min(300)]);
    let text = out(&r);
    assert_eq!(text.lines().count(), 1, "only the second message got a reply: {text:?}");
    assert!(text.contains("Paris"));
}

// FR-28, FR-37: a sampled chat reports its seed once, before the first marker, given or taken from the clock.
#[test]
fn fr28_a_sampled_chat_reports_its_seed_once_before_the_first_marker() {
    let r = chat(&["--max-new-tokens", "16"], &format!("{FRANCE}\nAnd of Italy?\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(r.stderr.matches("seed: ").count(), 1, "{:?}", r.stderr);
    assert!(r.stderr.find("seed: ").unwrap() < r.stderr.find("> ").unwrap(), "{:?}", r.stderr);
    let given = chat(&["--max-new-tokens", "16", "--seed", "4242"], FRANCE);
    assert_eq!(seed_of(&given), 4242);
}

// FR-28: the reported seed replays the whole conversation.
#[test]
fn fr28_the_reported_seed_replays_the_conversation() {
    let input = format!("{FRANCE}\nAnd of Italy?\n");
    let first = chat(&["--max-new-tokens", "24"], &input);
    let seed = seed_of(&first).to_string();
    let replay = chat(&["--max-new-tokens", "24", "--seed", &seed], &input);
    assert_eq!(out(&replay), out(&first));
}

// FR-37: a greedy chat reports no seed.
#[test]
fn fr37_a_greedy_chat_reports_no_seed() {
    let r = greedy(&format!("{FRANCE}\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(out(&r).contains("Paris"), "the chat ran");
    assert!(!r.stderr.contains("seed"), "{:?}", r.stderr);
}

// FR-37: the defaults are temperature 0.2 and top-p 0.9: spelling them out gives the same
// replies for the same seed, and another temperature does not.
#[test]
fn fr37_the_defaults_are_temperature_point_two_and_top_p_point_nine() {
    let input = "Tell me about the sea.\n";
    let base = ["--max-new-tokens", "32", "--seed", "11"];
    let default = chat(&base, input);
    let mut explicit = base.to_vec();
    explicit.extend(["--temperature", "0.2", "--top-p", "0.9"]);
    assert_eq!(out(&chat(&explicit, input)), out(&default));
    let mut hotter = base.to_vec();
    hotter.extend(["--temperature", "1.0"]);
    assert_ne!(out(&chat(&hotter, input)), out(&default));
}

// FR-37: the default top-p really filters: at debug level the tokens left after the controls
// number in the tens, and with `--top-p 1` the whole vocabulary stays.
#[test]
fn fr37_the_default_top_p_filters_the_vocabulary() {
    fn kept(r: &Run) -> Vec<usize> {
        r.stderr
            .lines()
            .filter_map(|l| l.find(" kept=").map(|at| l[at + 6..].split_whitespace().next().unwrap().parse().unwrap()))
            .collect()
    }
    let input = "Tell me about the sea.\n";
    let default = chat(&["-vv", "--max-new-tokens", "6", "--seed", "3"], input);
    assert!(kept(&default).len() >= 6 && kept(&default).iter().all(|&k| k < 1000), "{:?}", kept(&default));
    let open = chat(&["-vv", "--max-new-tokens", "6", "--seed", "3", "--top-p", "1"], input);
    assert!(kept(&open).iter().any(|&k| k == 49152), "{:?}", kept(&open));
}

// FR-37, Spec §9: `--temperature 0` alone is greedy with no warning about top-p, which the user never set.
#[test]
fn fr37_temperature_zero_alone_gives_no_warning() {
    let r = greedy(&format!("{FRANCE}\n"));
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(out(&r).contains("Paris"), "the chat ran");
    assert!(!r.stderr.to_lowercase().contains("ignored") && !r.stderr.contains("WARN"), "{:?}", r.stderr);
}

// FR-26: a control the user did set is still reported ignored under temperature 0.
#[test]
fn fr37_a_control_the_user_set_is_reported_ignored_under_temperature_zero() {
    let r = chat(&["--temperature", "0", "--top-k", "40", "--max-new-tokens", "8"], FRANCE);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(r.stderr.contains("ignored"), "{:?}", r.stderr);
}

// FR-33: `--system` replaces the default system message.
#[test]
fn fr33_system_replaces_the_default_system_message() {
    let tokens = |extra: &[&str]| {
        let mut args = vec!["-v", "--temperature", "0", "--max-new-tokens", "2"];
        args.extend(extra);
        prompt_tokens(&chat(&args, "hi\n"))
    };
    let default = tokens(&[]);
    let same_text = tokens(&["--system", "You are a helpful AI assistant named SmolLM, trained by Hugging Face"]);
    let other = tokens(&["--system", "Be brief."]);
    assert_eq!(default, same_text, "the default message given explicitly is the same prompt");
    assert_ne!(default, other);
}

// FR-30: an invalid sampling value exits 1 naming the option and the value, with nothing on standard output.
#[test]
fn fr30_an_invalid_value_is_rejected_naming_the_option_and_the_value() {
    for (arg, option, value) in [("--temperature=-1", "temperature", "-1"), ("--top-p=1.5", "top-p", "1.5")] {
        let r = chat(&[arg, "--seed", "1"], "hi\n");
        assert_eq!(r.code, 1, "{arg}: {}", r.stderr);
        assert!(r.stdout.is_empty());
        assert!(r.stderr.contains("error: ") && r.stderr.contains(option) && r.stderr.contains(value), "{arg}: {}", r.stderr);
    }
}

// FR-12: a model directory that is missing is an error, not a chat.
#[test]
fn fr38_a_missing_model_is_an_error_with_code_one() {
    let r = common::run(&["chat", "--model", "/nonexistent/model"], Some(b"hi\n"));
    assert_eq!(r.code, 1, "{}", r.stderr);
    assert!(r.stdout.is_empty() && r.stderr.contains("error: "), "{}", r.stderr);
}
