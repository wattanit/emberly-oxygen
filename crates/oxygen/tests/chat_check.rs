//! Tests of the S-11 check in the harness (FR-39): the engine's rendering and
//! encoding of a stored conversation against the stored text and token ids.
//! The conversations here are hand-made; the text is written out by hand.

mod common;

use common::compare::Verdict;
use common::fixtures::{require_model_dir, workspace_root, ChatMessage, Conversation};
use common::harness::check_chat_template;
use emberly_oxygen::{render, Error, Tokenizer};

const DEFAULT: &str =
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

fn tokenizer() -> Tokenizer {
    let dir = workspace_root().join("models").join("smollm2-135m-instruct");
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    Tokenizer::load(&dir).unwrap_or_else(|e| panic!("{e}"))
}

/// A stored conversation whose rendering is written by hand and whose ids are
/// the tokenizer's encoding of it.
fn conversation(tokenizer: &Tokenizer, roles_and_texts: &[(&str, &str)], rendered: &str) -> Conversation {
    Conversation {
        id: "hand".into(),
        messages: roles_and_texts
            .iter()
            .map(|(role, content)| ChatMessage { role: role.to_string(), content: content.to_string() })
            .collect(),
        rendered: rendered.into(),
        token_ids: tokenizer.encode(rendered).unwrap(),
        prompt_top2_gaps: vec![],
        reply_ids: vec![],
        reply_text: String::new(),
        reply_top2_gaps: vec![],
        stopped_at_eos: false,
        tensors_file: String::new(),
    }
}

fn hand() -> (Tokenizer, Conversation) {
    let t = tokenizer();
    let rendered = format!("{DEFAULT}<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n");
    let c = conversation(&t, &[("user", "hi")], &rendered);
    (t, c)
}

fn failed(v: &Verdict) -> bool {
    v.is_failure()
}

// S-11: the real renderer and encoder pass on a hand-written conversation.
#[test]
fn s11_the_real_functions_pass() {
    let (t, c) = hand();
    let r = check_chat_template("m", &c, &render, &|text| t.encode(text));
    assert_eq!((r.text.clone(), r.tokens.clone()), (Verdict::Pass, Verdict::Pass));
    assert!(!r.is_failure());
    assert_eq!((r.model_id.as_str(), r.conversation.as_str()), ("m", "hand"));
}

// S-11: a rendering that differs fails the text, and its encoding then fails the ids.
#[test]
fn s11_a_wrong_rendering_fails_text_and_tokens() {
    let (t, c) = hand();
    let wrong = |_: &[emberly_oxygen::Message], _: bool| "<|im_start|>user\nhi<|im_end|>\n".to_string();
    let r = check_chat_template("m", &c, &wrong, &|text| t.encode(text));
    assert!(failed(&r.text) && failed(&r.tokens) && r.is_failure());
    let Verdict::Fail { position, .. } = r.text else { unreachable!() };
    assert_eq!(position, Some(12), "both texts open with the 12-character marker; the stored one then says system, the engine's says user");
}

// S-11: text and ids are judged separately: a right rendering with a wrong encoder fails only the ids.
#[test]
fn s11_a_wrong_encoder_fails_only_the_tokens() {
    let (t, c) = hand();
    let wrong = |text: &str| -> Result<Vec<u32>, Error> {
        let mut ids = t.encode(text)?;
        ids.pop();
        Ok(ids)
    };
    let r = check_chat_template("m", &c, &render, &wrong);
    assert_eq!(r.text, Verdict::Pass);
    let Verdict::Fail { position, .. } = r.tokens else { panic!("expected the ids to fail") };
    assert_eq!(position, Some(c.token_ids.len() - 1));
    assert!(r.is_failure());
}

// S-11: an encoder that errors is a failure with its message, not a pass or a panic.
#[test]
fn s11_an_encoder_error_is_a_failure() {
    let (_, c) = hand();
    let broken = |_: &str| -> Result<Vec<u32>, Error> { Err(Error::Rejected("no encoder".into())) };
    let r = check_chat_template("m", &c, &render, &broken);
    let Verdict::Fail { message, .. } = r.tokens else { panic!("expected a failure") };
    assert!(message.contains("no encoder"), "{message}");
}

// S-11: a stored role the engine does not know is a failure of both, not a panic.
#[test]
fn s11_an_unknown_stored_role_is_a_failure() {
    let t = tokenizer();
    let c = conversation(&t, &[("robot", "hi")], "x");
    let r = check_chat_template("m", &c, &render, &|text| t.encode(text));
    for v in [&r.text, &r.tokens] {
        let Verdict::Fail { message, .. } = v else { panic!("expected a failure") };
        assert!(message.contains("robot"), "{message}");
    }
}

// S-11: the stored messages, not the stored rendering, are what the engine renders.
#[test]
fn s11_the_engine_renders_the_stored_messages_with_the_generation_prompt() {
    let t = tokenizer();
    let rendered = "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";
    let c = conversation(&t, &[("system", "Be brief."), ("user", "hi")], rendered);
    let seen = std::cell::RefCell::new(None);
    let spy = |messages: &[emberly_oxygen::Message], generation_prompt: bool| {
        *seen.borrow_mut() = Some((messages.len(), generation_prompt));
        render(messages, generation_prompt)
    };
    let r = check_chat_template("m", &c, &spy, &|text| t.encode(text));
    assert!(!r.is_failure());
    assert_eq!(*seen.borrow(), Some((2, true)));
}
