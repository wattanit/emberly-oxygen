//! Tests of the streaming check itself (FR-15, Spec §11): that it passes on the
//! stored `utf8-split` prompts, and that it fails when it should, so a pass
//! means something.

mod common;

use common::compare::Verdict;
use common::fixtures::{discover, require_model_dir, workspace_root, Prompt};
use common::harness::streaming_check;
use emberly_oxygen::Tokenizer;

fn tokenizer(model_id: &str) -> Tokenizer {
    let dir = workspace_root().join("models").join(model_id);
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"))
}

/// A prompt whose reference data is what the tokenizer itself produces.
fn prompt(tokenizer: &Tokenizer, category: &str, text: &str, continuation: &str) -> Prompt {
    let ids = |t: &str| tokenizer.encode(t).unwrap();
    Prompt {
        id: "synthetic".into(),
        category: category.into(),
        text: text.into(),
        token_ids: ids(text),
        prompt_top2_gaps: vec![],
        continuation_ids: ids(continuation),
        continuation_text: continuation.into(),
        continuation_top2_gaps: vec![],
        stopped_at_eos: false,
        tensors_file: String::new(),
    }
}

// FR-15: every stored prompt streams back to its reference text, and the
// utf8-split prompts really split a character.
#[test]
fn fr15_stored_prompts_stream_correctly() {
    let root = workspace_root();
    for fixtures in discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}")) {
        let tokenizer = tokenizer(&fixtures.manifest.model_id);
        let split = fixtures.manifest.prompts.iter().filter(|p| p.category == "utf8-split").count();
        assert!(split >= 2, "the prompt set needs at least two utf8-split prompts");
        for p in &fixtures.manifest.prompts {
            assert_eq!(streaming_check(&tokenizer, p), Verdict::Pass, "{}", p.id);
        }
    }
}

// FR-15: a utf8-split prompt that splits nothing is a failure, not a pass.
#[test]
fn fr15_a_prompt_that_splits_nothing_fails_as_utf8_split() {
    let t = tokenizer("smollm2-135m");
    let plain = prompt(&t, "utf8-split", "hello world", "again");
    match streaming_check(&t, &plain) {
        Verdict::Fail { message, .. } => assert!(message.contains("does not test streaming"), "{message}"),
        other => panic!("expected failure, got {other:?}"),
    }
    // The same prompt in another category has no such requirement.
    let plain = prompt(&t, "english", "hello world", "again");
    assert_eq!(streaming_check(&t, &plain), Verdict::Pass);
}

// FR-15: streamed text that differs from the reference is a failure naming where.
#[test]
fn fr15_a_wrong_reference_text_fails() {
    let t = tokenizer("smollm2-135m");
    let mut wrong = prompt(&t, "english", "hello world", "again");
    wrong.text = "hello there".into();
    match streaming_check(&t, &wrong) {
        Verdict::Fail { position, message } => {
            assert!(position.is_some() && message.contains("prompt"), "{message}");
        }
        other => panic!("expected failure, got {other:?}"),
    }
    let mut wrong = prompt(&t, "english", "hello world", "again");
    wrong.continuation_text = "again!".into();
    assert!(matches!(streaming_check(&t, &wrong), Verdict::Fail { message, .. } if message.contains("continuation")));
}

// FR-15: a Thai prompt that does split characters passes, with the split visible.
#[test]
fn fr15_a_split_thai_prompt_passes() {
    let t = tokenizer("smollm2-135m");
    // Tokens of this text cut characters in the middle (checked by the check itself).
    let thai = prompt(&t, "utf8-split", "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}\u{e04}\u{e23}\u{e31}\u{e1a}", "\u{e01}\u{e47}");
    assert_eq!(streaming_check(&t, &thai), Verdict::Pass);
}
