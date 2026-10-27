use crate::{Error, Model, Tokenizer};

/// Greedy generation (Spec §8): returns the generated token IDs and calls
/// `on_text` with each completed piece of text. Stub: generates nothing.
pub fn generate(
    _model: &Model,
    _tokenizer: &Tokenizer,
    _prompt: &str,
    _max_new_tokens: usize,
    _on_text: &mut dyn FnMut(&str),
) -> Result<Vec<u32>, Error> {
    Ok(Vec::new())
}
