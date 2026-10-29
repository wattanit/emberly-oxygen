mod byte_map;
mod components;
mod pretokenize;
mod unicode_classes;

use std::path::Path;

use crate::Error;

/// Byte-level BPE tokenizer (Spec §6). Stub: encodes nothing, decodes nothing.
pub struct Tokenizer;

impl Tokenizer {
    /// Reads `tokenizer.json` and rejects any component outside what is
    /// implemented (FR-12). The rest of loading arrives with tasks 3.7 to 3.9.
    pub fn load(model_dir: &Path) -> Result<Tokenizer, Error> {
        let path = model_dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let json = serde_json::from_str(&text).map_err(|e| Error::Malformed {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        components::check(&json)?;
        Ok(Tokenizer)
    }

    pub fn encode(&self, _text: &str) -> Result<Vec<u32>, Error> {
        Ok(Vec::new())
    }

    pub fn decode(&self, _ids: &[u32]) -> Result<String, Error> {
        Ok(String::new())
    }
}
