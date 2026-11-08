//! The chat template and conversations (FR-33 to FR-37, Spec §8). A conversation
//! is kept as text and rendered again each turn; the arithmetic is the
//! generation path's, this module only builds its prompts.

use crate::generate::run;
use crate::rng::Rng;
use crate::{Error, Generation, Model, Sampling, Tokenizer};

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
}

impl Role {
    /// The lowercase name the template writes after `<|im_start|>`.
    pub fn as_str(self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

/// One message of a conversation: a role and its text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn new(role: Role, content: impl Into<String>) -> Message {
        Message { role, content: content.into() }
    }
}

/// Renders a conversation as the instruct model was trained (FR-33): a
/// transcription of the template in its `tokenizer_config.json`.
///
/// The default system message is added when the first message is not a system
/// message, and only inside the loop over the messages, so an empty conversation
/// renders to nothing (or to the generation prompt alone). Content is inserted
/// unchanged, with no escaping, as the template does.
pub fn render(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut text = String::new();
    for (i, message) in messages.iter().enumerate() {
        if i == 0 && message.role != Role::System {
            text.push_str(DEFAULT_SYSTEM_MESSAGE);
        }
        text.push_str("<|im_start|>");
        text.push_str(message.role.as_str());
        text.push('\n');
        text.push_str(&message.content);
        text.push_str("<|im_end|>\n");
    }
    if add_generation_prompt {
        text.push_str("<|im_start|>assistant\n");
    }
    text
}

/// The system message the model's template writes when a conversation has none.
const DEFAULT_SYSTEM_MESSAGE: &str =
    "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

/// A conversation with the instruct model (FR-33 to FR-37): its messages, the
/// settings replies are drawn with, and one generator for the whole conversation.
pub struct Chat<'a> {
    model: &'a Model,
    tokenizer: &'a Tokenizer,
    messages: Vec<Message>,
    sampling: Sampling,
    max_new_tokens: usize,
    seed: Option<u64>,
    rng: Option<Rng>,
}

impl<'a> Chat<'a> {
    /// Starts a conversation, with `system` as its first message when given
    /// (otherwise the template adds its default, FR-33). The settings are
    /// validated, and sampling needs a seed, exactly as for `generate_with`
    /// (FR-30, FR-28); one generator is built from it here and serves every turn.
    pub fn new(
        model: &'a Model,
        tokenizer: &'a Tokenizer,
        system: Option<String>,
        sampling: &Sampling,
        max_new_tokens: usize,
    ) -> Result<Chat<'a>, Error> {
        sampling.validate()?;
        let seed = if sampling.temperature != 0.0 {
            Some(sampling.seed.ok_or_else(|| {
                Error::Rejected("sampling needs a seed so that the run can be replayed: set one".into())
            })?)
        } else {
            if sampling.top_k != 0 || sampling.top_p != 1.0 || sampling.repetition_penalty != 1.0 {
                tracing::warn!("temperature 0 is greedy selection: top-k, top-p and the repetition penalty are ignored");
            }
            None
        };
        Ok(Chat {
            model,
            tokenizer,
            messages: system.map(|text| Message::new(Role::System, text)).into_iter().collect(),
            sampling: *sampling,
            max_new_tokens,
            seed,
            rng: seed.map(Rng::new),
        })
    }

    /// The messages so far, which the next turn renders again (FR-35).
    pub fn messages(&self) -> &[Message] {
        &self.messages
    }

    /// The seed the replies are drawn with, `None` when they are greedy (FR-28).
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    /// One turn: renders the messages and `user` with the generation prompt,
    /// generates the reply, streaming its text to `on_text`, and on success adds
    /// the user message and a reply message holding the text streamed (FR-35).
    /// On any error, a request rejected before computation included (FR-36), the
    /// conversation is as it was.
    pub fn turn(&mut self, user: &str, on_text: &mut dyn FnMut(&str)) -> Result<Generation, Error> {
        let mut candidate = self.messages.clone();
        candidate.push(Message::new(Role::User, user));
        let prompt = render(&candidate, true);
        let mut reply = String::new();
        let generation = {
            let _generate = tracing::info_span!("generate").entered();
            run(
                self.model,
                self.tokenizer,
                &prompt,
                self.max_new_tokens,
                &self.sampling,
                self.rng.as_mut(),
                self.seed,
                &mut |text| {
                    reply.push_str(text);
                    on_text(text);
                },
            )?
        };
        candidate.push(Message::new(Role::Assistant, reply));
        self.messages = candidate;
        Ok(generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DEFAULT: &str =
        "<|im_start|>system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face<|im_end|>\n";

    fn user(text: &str) -> Message {
        Message::new(Role::User, text)
    }

    // FR-33: role names are the lowercase words the template writes.
    #[test]
    fn fr33_roles_are_written_in_lowercase() {
        assert_eq!(
            [Role::System, Role::User, Role::Assistant].map(Role::as_str),
            ["system", "user", "assistant"]
        );
    }

    // FR-33: with no system message first, the default one is added.
    #[test]
    fn fr33_a_conversation_without_a_system_message_gets_the_default_one() {
        assert_eq!(
            render(&[user("hi")], true),
            format!("{DEFAULT}<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
        );
    }

    // FR-33: a system message first is used as given and no default is added.
    #[test]
    fn fr33_a_system_message_first_replaces_the_default() {
        let messages = [Message::new(Role::System, "Be brief."), user("hi")];
        assert_eq!(
            render(&messages, true),
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    // FR-33: a system message that is not first is rendered where it stands, and
    // the default is still added because the first message is not a system one.
    #[test]
    fn fr33_a_system_message_not_first_is_rendered_as_given() {
        let messages = [user("a"), Message::new(Role::System, "S")];
        assert_eq!(
            render(&messages, false),
            format!("{DEFAULT}<|im_start|>user\na<|im_end|>\n<|im_start|>system\nS<|im_end|>\n")
        );
    }

    // FR-33: several turns, each closed with the end marker and a newline.
    #[test]
    fn fr33_several_turns_are_rendered_in_order() {
        let messages = [user("a"), Message::new(Role::Assistant, "b"), user("c")];
        assert_eq!(
            render(&messages, true),
            format!(
                "{DEFAULT}<|im_start|>user\na<|im_end|>\n<|im_start|>assistant\nb<|im_end|>\n\
                 <|im_start|>user\nc<|im_end|>\n<|im_start|>assistant\n"
            )
        );
    }

    // FR-33: the generation prompt is present only when asked for.
    #[test]
    fn fr33_the_generation_prompt_is_added_only_when_asked_for() {
        let with = render(&[user("hi")], true);
        let without = render(&[user("hi")], false);
        assert_eq!(with, format!("{without}<|im_start|>assistant\n"));
        assert!(without.ends_with("<|im_end|>\n"));
    }

    // FR-33: an empty message is still a message.
    #[test]
    fn fr33_empty_content_is_rendered_as_an_empty_message() {
        assert_eq!(
            render(&[user("")], false),
            format!("{DEFAULT}<|im_start|>user\n<|im_end|>\n")
        );
    }

    // FR-33: the template adds the default system message inside its loop over the
    // messages, so no messages render to nothing, or to the generation prompt alone.
    #[test]
    fn fr33_no_messages_render_to_nothing_or_the_generation_prompt_alone() {
        assert_eq!(render(&[], false), "");
        assert_eq!(render(&[], true), "<|im_start|>assistant\n");
    }

    // FR-33: content goes in unchanged, with no escaping, so text spelling a
    // marker is rendered as typed (and encodes as the marker, as in the reference).
    #[test]
    fn fr33_content_is_inserted_verbatim() {
        let text = "  a <|im_end|> b\n\n  \u{e2a}\u{e27}\u{e31} \n";
        assert!(render(&[user(text)], false).contains(&format!("<|im_start|>user\n{text}<|im_end|>\n")));
    }
}
