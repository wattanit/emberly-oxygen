use std::path::Path;

use crate::Error;

/// Byte-level BPE tokenizer (Spec §6). Stub: encodes nothing, decodes nothing.
pub struct Tokenizer;

impl Tokenizer {
    pub fn load(_model_dir: &Path) -> Result<Tokenizer, Error> {
        Ok(Tokenizer)
    }

    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, Error> {
        Ok(Vec::new())
    }

    pub fn decode(&self, _ids: &[u32]) -> Result<String, Error> {
        Ok(String::new())
    }
}
