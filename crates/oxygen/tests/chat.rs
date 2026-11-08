//! Conversations on the real instruct model (FR-33 to FR-37). Greedy unless a
//! test is about sampling, so each reply is the same every run.

mod common;

use common::fixtures::{require_model_dir, workspace_root};
use emberly_oxygen::{generate, render, Chat, Error, Message, Model, Role, Sampling, Tokenizer};

fn load() -> (Model, Tokenizer) {
    let dir = workspace_root().join("models").join("smollm2-135m-instruct");
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    (Model::load(&dir).unwrap(), Tokenizer::load(&dir).unwrap())
}

fn greedy() -> Sampling {
    Sampling::default()
}

/// Runs one turn and returns the generation and the text streamed.
fn turn(chat: &mut Chat, user: &str) -> (emberly_oxygen::Generation, String) {
    let mut streamed = String::new();
    let generation = chat.turn(user, &mut |t| streamed.push_str(t)).unwrap_or_else(|e| panic!("{e}"));
    (generation, streamed)
}

// FR-34: a reply ends at <|im_end|> (ID 2), which is generated but never streamed.
#[test]
fn fr34_a_reply_ends_at_the_end_marker_and_it_is_not_streamed() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 64).unwrap();
    let (generation, streamed) = turn(&mut chat, "What is the capital of France?");
    assert_eq!(generation.ids.last(), Some(&2), "stopped at <|im_end|>, not at the maximum");
    assert!(generation.ids.len() < 64);
    assert!(streamed.contains("Paris"), "{streamed:?}");
    assert!(!streamed.contains("<|im_end|>"), "the end marker is a signal, not text: {streamed:?}");
}

// FR-35: a turn adds exactly a user message and a reply message holding the text streamed.
#[test]
fn fr35_a_turn_appends_the_user_message_and_the_reply_as_streamed() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 64).unwrap();
    assert!(chat.messages().is_empty(), "the default system message is the template's, not stored");
    let (_, streamed) = turn(&mut chat, "What is the capital of France?");
    assert_eq!(
        chat.messages(),
        [Message::new(Role::User, "What is the capital of France?"), Message::new(Role::Assistant, streamed)]
    );
}

// FR-33: a system message given to the conversation is its first message.
#[test]
fn fr33_a_system_message_is_the_first_message() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, Some("Be brief.".into()), &greedy(), 8).unwrap();
    assert_eq!(chat.messages(), [Message::new(Role::System, "Be brief.")]);
    let (generation, _) = turn(&mut chat, "hi");
    let expected = render(&[Message::new(Role::System, "Be brief."), Message::new(Role::User, "hi")], true);
    assert_eq!(generation.stats.prompt_tokens, t.encode(&expected).unwrap().len());
    assert_eq!(chat.messages().len(), 3);
}

// FR-35: the second turn's prompt is the whole conversation so far, rendered again.
#[test]
fn fr35_the_second_prompt_contains_both_earlier_messages() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 64).unwrap();
    let (_, first_reply) = turn(&mut chat, "Hi! My name is Ploy.");
    let (second, _) = turn(&mut chat, "What is 2 plus 2?");
    let expected = render(
        &[
            Message::new(Role::User, "Hi! My name is Ploy."),
            Message::new(Role::Assistant, first_reply),
            Message::new(Role::User, "What is 2 plus 2?"),
        ],
        true,
    );
    assert_eq!(second.stats.prompt_tokens, t.encode(&expected).unwrap().len());
    assert_eq!(chat.messages().len(), 4);
}

// FR-34: a reply cut by the maximum has no end marker generated, and the next prompt closes it.
#[test]
fn fr34_a_reply_cut_by_the_maximum_is_closed_by_the_next_rendering() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 3).unwrap();
    let (generation, cut) = turn(&mut chat, "What is the capital of France?");
    assert_eq!(generation.ids.len(), 3);
    assert!(!generation.ids.contains(&2), "cut before the model ended its turn");
    let history = render(chat.messages(), false);
    assert!(history.ends_with(&format!("<|im_start|>assistant\n{cut}<|im_end|>\n")), "{history:?}");
    // And the conversation goes on from the closed turn.
    let (next, _) = turn(&mut chat, "And of Italy?");
    assert!(next.stats.prompt_tokens > generation.stats.prompt_tokens);
    assert_eq!(chat.messages().len(), 4);
}

// FR-29: temperature 0 gives the ids `generate` gives on the rendered text.
#[test]
fn fr29_a_greedy_chat_turn_equals_generate_on_the_rendered_prompt() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 32).unwrap();
    let (generation, _) = turn(&mut chat, "What is the capital of France?");
    let prompt = render(&[Message::new(Role::User, "What is the capital of France?")], true);
    let direct = generate(&m, &t, &prompt, 32, &mut |_| {}).unwrap();
    assert_eq!(generation.ids, direct.ids);
}

// FR-30, FR-28: invalid settings and sampling without a seed are rejected when the conversation starts.
#[test]
fn fr30_fr28_bad_settings_and_a_missing_seed_are_rejected_at_the_start() {
    let (m, t) = load();
    let sampled = Sampling { temperature: 0.2, top_p: 0.9, ..Sampling::default() };
    let error = Chat::new(&m, &t, None, &sampled, 8).err().expect("no seed");
    assert!(matches!(&error, Error::Rejected(m) if m.contains("seed")), "{error}");
    let negative = Sampling { temperature: -1.0, ..Sampling::default() };
    let error = Chat::new(&m, &t, None, &negative, 8).err().expect("bad temperature");
    assert!(matches!(&error, Error::Rejected(m) if m.contains("temperature")), "{error}");
    let seeded = Sampling { seed: Some(5), ..sampled };
    let chat = Chat::new(&m, &t, None, &seeded, 8).unwrap();
    assert_eq!(chat.seed(), Some(5));
    assert_eq!(Chat::new(&m, &t, None, &greedy(), 8).unwrap().seed(), None);
}

// FR-35: the reply is stored as the user saw it, edge whitespace included, because the
// next rendering must reproduce the prompt the reference would build from that text.
#[test]
fn fr35_a_reply_is_stored_exactly_as_streamed_whitespace_included() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 6).unwrap();
    let (_, streamed) = turn(&mut chat, "List three fruits.");
    assert_ne!(streamed, streamed.trim(), "this prompt and maximum end the reply in a newline: {streamed:?}");
    assert_eq!(chat.messages()[1].content, streamed);
}

/// A user message of about `n` tokens: "word " repeated.
fn long_message(n: usize) -> String {
    "word ".repeat(n)
}

// FR-36, FR-18: a turn that cannot fit is rejected before anything is computed, the
// conversation is as it was, and a later turn works.
#[test]
fn fr36_a_turn_that_cannot_fit_is_rejected_at_once_and_leaves_the_conversation_alone() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, None, &greedy(), 64).unwrap();
    let (_, first) = turn(&mut chat, "What is the capital of France?");
    let before = chat.messages().to_vec();

    let started = std::time::Instant::now();
    let mut streamed = String::new();
    let error = chat.turn(&long_message(9000), &mut |x| streamed.push_str(x)).unwrap_err();
    assert!(matches!(&error, Error::Rejected(m) if m.contains("context length")), "{error}");
    assert!(started.elapsed() < std::time::Duration::from_secs(5), "rejected before computing 9000 positions");
    assert!(streamed.is_empty(), "nothing was generated");
    assert_eq!(chat.messages(), before, "the rejected turn left no trace");

    let (_, again) = turn(&mut chat, "And of Italy?");
    assert_eq!(chat.messages().len(), 4);
    assert!(!first.is_empty() && !again.is_empty());
}

// FR-36: a rejected turn draws nothing from the conversation's generator, so a seeded chat
// replies exactly as one that never had the rejected turn.
#[test]
fn fr36_a_rejected_turn_leaves_the_generator_where_it_was() {
    let (m, t) = load();
    let seeded = Sampling { temperature: 0.8, seed: Some(42), ..Sampling::default() };
    let mut plain = Chat::new(&m, &t, None, &seeded, 24).unwrap();
    let (a, _) = turn(&mut plain, "Tell me about the sea.");

    let mut rejected = Chat::new(&m, &t, None, &seeded, 24).unwrap();
    rejected.turn(&long_message(9000), &mut |_| {}).unwrap_err();
    let (b, _) = turn(&mut rejected, "Tell me about the sea.");
    assert_eq!(a.ids, b.ids);
}

// FR-36: the limit is on the whole rendered conversation, so a conversation that has grown
// long enough rejects a message that would have fitted at the start.
#[test]
fn fr36_the_limit_counts_the_history_not_only_the_new_message() {
    let (m, t) = load();
    let mut chat = Chat::new(&m, &t, Some(long_message(8150)), &greedy(), 64).unwrap();
    let error = chat.turn("hi", &mut |_| {}).unwrap_err();
    assert!(matches!(&error, Error::Rejected(m) if m.contains("context length")), "{error}");
    assert_eq!(chat.messages().len(), 1);
}
