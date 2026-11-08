//! Negative controls for S-11 (Spec §17): the template check must be seen to fail.
//! Each control gives `check_chat_template` a faulty renderer or encoder, kept
//! here and not in the library (the template is not model arithmetic, so no
//! faulty code is compiled into the engine), and runs it over the real stored
//! conversations. With the real functions every conversation passes, so a
//! control that did nothing would itself be caught.

mod common;

use common::compare::Verdict;
use common::fixtures::{discover_chat, workspace_root, ChatFixtures, Conversation};
use common::harness::{check_chat_template, ChatResult};
use emberly_oxygen::{render, Error, Message, Tokenizer};

const DEFAULT: &str =
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

fn setup() -> (ChatFixtures, Tokenizer) {
    let root = workspace_root();
    let fixtures = discover_chat(&root.join("fixtures"))
        .unwrap_or_else(|e| panic!("{e}"))
        .into_iter()
        .next()
        .expect("no chat fixtures");
    let dir = root.join("models").join(&fixtures.chat.model_id);
    common::fixtures::require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    let tokenizer = Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"));
    (fixtures, tokenizer)
}

fn check_all(
    fixtures: &ChatFixtures,
    renderer: &dyn Fn(&[Message], bool) -> String,
    encoder: &dyn Fn(&str) -> Result<Vec<u32>, Error>,
) -> Vec<ChatResult> {
    fixtures
        .chat
        .conversations
        .iter()
        .map(|c| check_chat_template(&fixtures.chat.model_id, c, renderer, encoder))
        .collect()
}

fn of<'a>(results: &'a [ChatResult], id: &str) -> &'a ChatResult {
    results.iter().find(|r| r.conversation == id).unwrap_or_else(|| panic!("conversation {id} missing"))
}

fn has_default_system(c: &Conversation) -> bool {
    c.messages[0].role != "system"
}

/// A renderer that leaves out the default system message the template adds.
fn without_default_system(messages: &[Message], generation_prompt: bool) -> String {
    let text = render(messages, generation_prompt);
    text.strip_prefix(DEFAULT).map(str::to_string).unwrap_or(text)
}

/// A renderer that forgets the newline after each `<|im_end|>`.
fn without_newline_after_end(messages: &[Message], generation_prompt: bool) -> String {
    render(messages, generation_prompt).replace("<|im_end|>\n", "<|im_end|>")
}

/// An encoder that spells each marker out as ordinary characters (`<` then
/// `|im_start|>`), which is how a tokenizer without special tokens would read it.
fn spelling_markers_out(tokenizer: &Tokenizer, text: &str) -> Result<Vec<u32>, Error> {
    let mut ids = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let next = ["<|im_start|>", "<|im_end|>"].iter().filter_map(|m| rest.find(m).map(|at| (at, *m))).min();
        match next {
            Some((at, marker)) => {
                ids.extend(tokenizer.encode(&rest[..at])?);
                ids.extend(tokenizer.encode("<")?);
                ids.extend(tokenizer.encode(&marker[1..])?);
                rest = &rest[at + marker.len()..];
            }
            None => {
                ids.extend(tokenizer.encode(rest)?);
                break;
            }
        }
    }
    Ok(ids)
}

fn failed(v: &Verdict) -> bool {
    v.is_failure()
}

// S-11: with the real functions every stored conversation passes, so the failures
// below are caused by the faults.
#[test]
fn s11_without_a_fault_every_conversation_passes() {
    let (fixtures, t) = setup();
    let results = check_all(&fixtures, &render, &|text| t.encode(text));
    assert_eq!(results.len(), fixtures.chat.conversations.len());
    for r in &results {
        assert_eq!((&r.text, &r.tokens), (&Verdict::Pass, &Verdict::Pass), "{}", r.conversation);
    }
}

// S-11: no default system message fails text and ids exactly where the template adds one.
#[test]
fn s11_a_missing_default_system_message_fails_where_the_template_adds_it() {
    let (fixtures, t) = setup();
    let results = check_all(&fixtures, &without_default_system, &|text| t.encode(text));
    for c in &fixtures.chat.conversations {
        let r = of(&results, &c.id);
        if has_default_system(c) {
            assert!(failed(&r.text) && failed(&r.tokens), "{} should fail both", c.id);
        } else {
            assert!(!r.is_failure(), "{} has its own system message, so the fault changes nothing", c.id);
        }
    }
    assert!(results.iter().filter(|r| r.is_failure()).count() >= 3, "no-system, multi-turn and thai");
}

// S-11: a missing newline after the end marker fails every conversation, text and ids.
#[test]
fn s11_a_missing_newline_after_the_end_marker_fails_every_conversation() {
    let (fixtures, t) = setup();
    let results = check_all(&fixtures, &without_newline_after_end, &|text| t.encode(text));
    for r in &results {
        assert!(failed(&r.text) && failed(&r.tokens), "{} should fail both", r.conversation);
    }
}

// S-11, FR-33: markers spelled out as characters leave the text right and fail the
// ids, which is why both are checked.
#[test]
fn s11_markers_spelled_out_as_characters_fail_the_ids_and_not_the_text() {
    let (fixtures, t) = setup();
    let results = check_all(&fixtures, &render, &|text| spelling_markers_out(&t, text));
    for r in &results {
        assert_eq!(r.text, Verdict::Pass, "{}: the text is the real renderer's", r.conversation);
        let Verdict::Fail { position, .. } = &r.tokens else { panic!("{} should fail on ids", r.conversation) };
        assert_eq!(*position, Some(0), "{}: the first marker is already wrong", r.conversation);
    }
}
