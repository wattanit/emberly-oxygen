//! Runs the built `emberly-oxygen` binary with the real model and checks what
//! a user sees: standard output, standard error and the exit code (Spec §9).
//! The fixtures supply the expected text.

mod common;

use std::path::Path;
use std::time::Duration;

use common::{generate, model_dir, root, run, Run};
use serde_json::Value;

/// `english-short` from the fixture manifest: its prompt text, the oracle's continuation, and
/// the prompt's and the continuation's token counts.
fn english_short() -> (String, String, usize, usize) {
    let path = root().join("fixtures/smollm2-135m/manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}; run `cd oracle && uv run python generate.py`", path.display()));
    let manifest: Value = serde_json::from_str(&text).unwrap();
    let prompt = manifest["prompts"].as_array().unwrap().iter().find(|p| p["id"] == "english-short").unwrap();
    (
        prompt["text"].as_str().unwrap().to_owned(),
        prompt["continuation_text"].as_str().unwrap().to_owned(),
        prompt["token_ids"].as_array().unwrap().len(),
        prompt["continuation_ids"].as_array().unwrap().len(),
    )
}

/// An error exit: code 1, nothing on standard output, the message on standard error.
fn assert_error(r: &Run, names: &[&str]) {
    assert_eq!(r.code, 1, "{}", r.stderr);
    assert!(r.stdout.is_empty(), "standard output: {:?}", String::from_utf8_lossy(&r.stdout));
    assert!(r.stderr.starts_with("error: "), "{}", r.stderr);
    for name in names {
        assert!(r.stderr.contains(name), "{name:?} not in {:?}", r.stderr);
    }
}

// FR-19, FR-17, FR-20, Spec §9: standard output is exactly the oracle's continuation and
// nothing else; at the default level standard error is empty.
#[test]
fn output_is_exactly_the_oracle_continuation() {
    let (prompt, continuation, _, n) = english_short();
    assert_eq!(n, 32);
    let r = generate(&["--max-new-tokens", "32", &prompt]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(String::from_utf8(r.stdout).unwrap(), continuation);
    assert_eq!(r.stderr, "");
}

// FR-19, Spec §9: a prompt on standard input gives the same output as the same prompt as an argument.
#[test]
fn standard_input_and_argument_give_the_same_output() {
    let (prompt, _, _, _) = english_short();
    let model = model_dir();
    let piped = run(&["generate", "--model", &model, "--max-new-tokens", "8"], Some(prompt.as_bytes()));
    let argument = generate(&["--max-new-tokens", "8", &prompt]);
    assert_eq!((piped.code, argument.code), (0, 0), "{}{}", piped.stderr, argument.stderr);
    assert!(!piped.stdout.is_empty());
    assert_eq!(piped.stdout, argument.stdout);
}

// FR-21, Spec §9, §12: --stats reports the counts, time to first token and generation throughput
// (with model load time) on standard error, and standard output is unchanged.
#[test]
fn stats_go_to_standard_error_only() {
    let (prompt, continuation, prompt_tokens, _) = english_short();
    let r = generate(&["--max-new-tokens", "32", "--stats", &prompt]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert_eq!(String::from_utf8(r.stdout).unwrap(), continuation);
    let field = |label: &str| -> String {
        let line = r.stderr.lines().find(|l| l.starts_with(label)).unwrap_or_else(|| panic!("{label} missing: {}", r.stderr));
        line[label.len()..].trim().to_owned()
    };
    assert!(field("model load:").ends_with(" s"));
    assert_eq!(field("prompt tokens:"), prompt_tokens.to_string());
    assert_eq!(field("generated tokens:"), "32");
    let first: f64 = field("time to first token:").trim_end_matches(" s").parse().unwrap();
    assert!(first > 0.0);
    let rate: f64 = field("generation:").trim_end_matches(" tokens/s").parse().unwrap();
    assert!(rate > 0.0);
}

// FR-18, Spec §9: a request beyond the context is rejected naming the numbers, before any step
// is computed (no step event even at -vv), and so returns quickly.
#[test]
fn a_request_beyond_the_context_is_rejected_before_any_work() {
    let r = generate(&["--max-new-tokens", "100000", "-vv", "Hello"]);
    assert_eq!(r.code, 1, "{}", r.stderr);
    assert!(r.stdout.is_empty());
    let error = r.stderr.lines().find(|l| l.starts_with("error: ")).unwrap_or_else(|| panic!("{}", r.stderr));
    for number in ["1", "100000", "8192"] {
        assert!(error.contains(number), "{number} not in {error:?}");
    }
    assert!(r.stderr.contains("model loaded"), "logging is on at -vv: {}", r.stderr);
    assert!(!r.stderr.contains(" step "), "no step was run: {}", r.stderr);
    assert!(r.elapsed < Duration::from_secs(5), "took {:?}", r.elapsed);
}

// FR-18, FR-20, Spec §9: an empty prompt is an error exit with nothing on standard output.
#[test]
fn an_empty_prompt_is_an_error() {
    assert_error(&generate(&["--max-new-tokens", "4", ""]), &["empty"]);
    // The same through standard input, which is empty here.
    assert_error(&generate(&["--max-new-tokens", "4"]), &["empty"]);
}

// FR-20, Spec §9, §14: a missing model directory is an error naming the path.
#[test]
fn a_missing_model_directory_names_the_path() {
    let r = run(&["generate", "--model", "/nonexistent-model-dir", "--max-new-tokens", "4", "Hello"], None);
    assert_error(&r, &["/nonexistent-model-dir"]);
}

// FR-20, Spec §9, §14: a configuration outside the supported set is an error naming the field.
#[test]
fn an_unsupported_configuration_names_the_field() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("unsupported-config");
    std::fs::create_dir_all(&dir).unwrap();
    let mut config: Value = serde_json::from_str(&std::fs::read_to_string(root().join("models/smollm2-135m/config.json")).unwrap()).unwrap();
    config["model_type"] = Value::from("gpt2");
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
    let r = run(&["generate", "--model", dir.to_str().unwrap(), "--max-new-tokens", "4", "Hello"], None);
    assert_error(&r, &["model_type", "gpt2"]);
}

// FR-19, FR-20, Spec §9: standard input that is not UTF-8 is an error, not repaired.
#[test]
fn invalid_utf8_on_standard_input_is_an_error() {
    let model = model_dir();
    let r = run(&["generate", "--model", &model, "--max-new-tokens", "4"], Some(b"ab\xffcd"));
    assert_error(&r, &["UTF-8"]);
}

// FR-20, Spec §9, §14: usage errors exit 2, with nothing on standard output.
#[test]
fn usage_errors_exit_2() {
    let model = model_dir();
    for args in [
        vec!["generate", "--max-new-tokens", "4", "Hello"],
        vec!["generate", "--model", &model, "--max-new-tokens", "many", "Hello"],
        vec!["generate", "--model", &model, "--max-new-tokens", "4", "--frobnicate", "Hello"],
    ] {
        let r = run(&args, None);
        assert_eq!(r.code, 2, "{args:?}: {}", r.stderr);
        assert!(r.stdout.is_empty(), "{args:?}");
        assert!(r.stderr.contains("Usage:") || r.stderr.contains("--help"), "{args:?}: {}", r.stderr);
    }
}

// FR-24, FR-20, Spec §9: --log-file takes the log instead of standard error, leaving standard
// output as it is; a path that cannot be created is an error naming it.
#[test]
fn the_log_file_takes_the_log() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cli-log-file.txt");
    let _ = std::fs::remove_file(&path);
    let log = path.to_str().unwrap();
    let plain = generate(&["--max-new-tokens", "4", "Hello"]);
    let logged = generate(&["--max-new-tokens", "4", "-vv", "--log-file", log, "Hello"]);
    assert_eq!(logged.code, 0, "{}", logged.stderr);
    assert_eq!(logged.stderr, "");
    assert_eq!(logged.stdout, plain.stdout);
    let written = std::fs::read_to_string(&path).unwrap();
    assert!(written.contains("run summary") && written.contains("prefill{tokens=") && written.contains("step{pos="), "{written}");

    let r = generate(&["--max-new-tokens", "4", "--log-file", "/nonexistent-dir/log.txt", "Hello"]);
    assert_error(&r, &["/nonexistent-dir/log.txt"]);
}

// FR-44, Spec §9, §14: 0 threads is an error exit naming the option and the value, nothing on standard
// output, before any computation; the same for chat. A value that is not a whole number is a usage error.
#[test]
fn fr44_zero_threads_is_rejected_naming_the_option_and_value() {
    let r = generate(&["--max-new-tokens", "4", "--threads", "0", "Hello"]);
    assert_error(&r, &["threads", "0"]);
    assert!(r.elapsed < Duration::from_secs(5), "took {:?}", r.elapsed);
    assert_error(&common::chat(&["--threads", "0"], ""), &["threads", "0"]);
    for bad in ["many", "-1", "2.5"] {
        let r = generate(&["--max-new-tokens", "4", "--threads", bad, "Hello"]);
        assert_eq!(r.code, 2, "{bad}: {}", r.stderr);
        assert!(r.stdout.is_empty());
    }
}

// FR-43, FR-44, S-6, Spec §9: the thread count does not change the output: the oracle's continuation at
// 1, 3 and 16 threads and at the default.
#[test]
fn fr43_fr44_output_does_not_depend_on_the_thread_count() {
    let (prompt, continuation, _, _) = english_short();
    for threads in ["1", "3", "16"] {
        let r = generate(&["--max-new-tokens", "32", "--threads", threads, &prompt]);
        assert_eq!(r.code, 0, "{threads}: {}", r.stderr);
        assert_eq!(String::from_utf8(r.stdout).unwrap(), continuation, "{threads} threads");
        assert_eq!(r.stderr, "", "the default level is quiet at {threads} threads");
    }
}

// FR-44, Spec §14: -v reports the thread count in use, the default being the cores available.
#[test]
fn fr44_the_run_summary_reports_the_thread_count() {
    let r = generate(&["-v", "--max-new-tokens", "2", "--threads", "3", "Hello"]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    let line = r.stderr.lines().find(|l| l.contains("model loaded")).unwrap_or_else(|| panic!("{}", r.stderr));
    assert!(line.contains(" threads=3"), "{line}");
    let cores = std::thread::available_parallelism().unwrap().get();
    let r = generate(&["-v", "--max-new-tokens", "2", "Hello"]);
    let line = r.stderr.lines().find(|l| l.contains("model loaded")).unwrap();
    assert!(line.contains(&format!(" threads={cores}")), "{line}");
}

// FR-43, FR-44: a greedy chat reply is the same at 1 and 5 threads.
#[test]
fn fr43_fr44_a_chat_reply_does_not_depend_on_the_thread_count() {
    let reply = |threads: &str| {
        let r = common::chat(&["--temperature", "0", "--max-new-tokens", "12", "--threads", threads], "Hello\n");
        assert_eq!(r.code, 0, "{threads}: {}", r.stderr);
        r.stdout
    };
    let one = reply("1");
    assert!(!one.is_empty());
    assert_eq!(reply("5"), one);
}
