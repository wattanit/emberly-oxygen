mod bpe;
mod byte_map;
mod components;
mod pretokenize;
mod unicode_classes;

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::Error;
use bpe::Bpe;
use byte_map::ByteMap;

/// Byte-level BPE tokenizer (Spec §6). Encoding is real; decoding is still a
/// stub until task 3.10.
pub struct Tokenizer {
    bpe: Bpe,
    byte_map: ByteMap,
    /// Added tokens, matched in raw text before anything else is done to it.
    added: Vec<(String, u32)>,
}

impl Tokenizer {
    /// Reads `tokenizer.json`, rejecting any component outside what is
    /// implemented (FR-12), and builds the vocabulary and merge table.
    pub fn load(model_dir: &Path) -> Result<Tokenizer, Error> {
        let path = model_dir.join("tokenizer.json");
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        let json: Value = serde_json::from_str(&text).map_err(|e| Error::Malformed {
            path: path.clone(),
            reason: e.to_string(),
        })?;
        components::check(&json)?;
        Tokenizer::from_json(&json, &path)
    }

    fn from_json(json: &Value, path: &Path) -> Result<Tokenizer, Error> {
        let malformed = |reason: String| Error::Malformed {
            path: path.to_owned(),
            reason,
        };

        let vocab_json = json["model"]["vocab"]
            .as_object()
            .ok_or_else(|| malformed("model.vocab is not an object".into()))?;
        let mut vocab = HashMap::with_capacity(vocab_json.len());
        for (piece, id) in vocab_json {
            let id = id
                .as_u64()
                .and_then(|n| u32::try_from(n).ok())
                .ok_or_else(|| malformed(format!("vocabulary entry {piece:?} has no valid ID")))?;
            vocab.insert(piece.clone(), id);
        }

        // Merges are either "left right" strings or [left, right] pairs.
        let merges_json = json["model"]["merges"]
            .as_array()
            .ok_or_else(|| malformed("model.merges is not a list".into()))?;
        let mut merges = Vec::with_capacity(merges_json.len());
        for (rank, merge) in merges_json.iter().enumerate() {
            let pair = match merge {
                Value::String(s) => s.split_once(' ').map(|(a, b)| (a.to_owned(), b.to_owned())),
                Value::Array(v) => match v.as_slice() {
                    [Value::String(a), Value::String(b)] => Some((a.clone(), b.clone())),
                    _ => None,
                },
                _ => None,
            };
            merges.push(pair.ok_or_else(|| malformed(format!("merge {rank} is not a pair of pieces")))?);
        }
        let bpe = Bpe::new(vocab, &merges).map_err(malformed)?;

        let mut added = Vec::new();
        for token in json["added_tokens"].as_array().map(Vec::as_slice).unwrap_or_default() {
            let content = token["content"].as_str().filter(|c| !c.is_empty());
            let id = token["id"].as_u64().and_then(|n| u32::try_from(n).ok());
            let (Some(content), Some(id)) = (content, id) else {
                return Err(malformed(format!("added token {token} has no content or ID")));
            };
            added.push((content.to_owned(), id));
        }
        Ok(Tokenizer { bpe, byte_map: ByteMap::new(), added })
    }

    /// Text to token IDs (FR-13). Added tokens are matched first, then the
    /// rest is split, mapped to the byte-level alphabet and merged. SmolLM2
    /// has no post-processor, so nothing is added around the result.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
        let mut ids = Vec::new();
        let mut rest = text;
        loop {
            // The earliest added token in `rest`, the longest if several start together.
            let next = self
                .added
                .iter()
                .filter_map(|(content, id)| rest.find(content.as_str()).map(|at| (at, content.len(), *id)))
                .min_by_key(|&(at, len, _)| (at, std::cmp::Reverse(len)));
            let (plain, special) = match next {
                Some((at, len, id)) => (&rest[..at], Some((len, id))),
                None => (rest, None),
            };
            self.encode_plain(plain, &mut ids)?;
            let Some((len, id)) = special else {
                return Ok(ids);
            };
            ids.push(id);
            rest = &rest[plain.len() + len..];
        }
    }

    fn encode_plain(&self, text: &str, ids: &mut Vec<u32>) -> Result<(), Error> {
        for piece in pretokenize::split(text) {
            let written: String = piece.bytes().map(|b| self.byte_map.char_for(b)).collect();
            ids.extend(self.bpe.encode_piece(&written)?);
        }
        Ok(())
    }

    pub fn decode(&self, _ids: &[u32]) -> Result<String, Error> {
        Ok(String::new())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    fn model_dir() -> PathBuf {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        assert!(dir.exists(), "{}: missing; run `cd oracle && uv run python download.py`", dir.display());
        dir
    }

    /// A complete tokenizer.json with a made-up vocabulary: the 256 byte
    /// characters at IDs 0 to 255, then the given merged tokens, then added tokens.
    fn tiny(merges: &[&str], merged: &[&str], added: &[&str]) -> Tokenizer {
        let map = ByteMap::new();
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(map.char_for(b).to_string(), Value::from(b));
        }
        for (i, token) in merged.iter().enumerate() {
            vocab.insert(token.to_string(), Value::from(256 + i));
        }
        let first_added = 256 + merged.len();
        let added: Vec<Value> = added
            .iter()
            .enumerate()
            .map(|(i, c)| serde_json::json!({"id": first_added + i, "content": c}))
            .collect();
        let json = serde_json::json!({
            "model": {"vocab": vocab, "merges": merges},
            "added_tokens": added,
        });
        Tokenizer::from_json(&json, Path::new("tokenizer.json")).unwrap()
    }

    // FR-13, Spec §6: special tokens are matched before splitting, longest first, anywhere in the text.
    #[test]
    fn added_tokens_are_matched_first() {
        let t = tiny(&["h i"], &["hi"], &["<s>", "<s><s>", "<e>"]);
        let (s, ss, e) = (257, 258, 259);
        assert_eq!(t.encode("hi").unwrap(), [256]);
        assert_eq!(t.encode("<s>").unwrap(), [s]);
        assert_eq!(t.encode("<s><s>").unwrap(), [ss], "the longer token wins at the same position");
        assert_eq!(t.encode("hi<e>hi").unwrap(), [256, e, 256]);
        assert_eq!(t.encode("<e><s>x").unwrap(), [e, s, u32::from(b'x')]);
        assert_eq!(t.encode("<s").unwrap(), [u32::from(b'<'), u32::from(b's')]);
        assert_eq!(t.encode("").unwrap(), Vec::<u32>::new());
    }

    // FR-13: the engine's loader accepts both merge spellings.
    #[test]
    fn merges_may_be_strings_or_pairs() {
        let map = ByteMap::new();
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(map.char_for(b).to_string(), Value::from(b));
        }
        vocab.insert("hi".into(), Value::from(256));
        for merges in [serde_json::json!(["h i"]), serde_json::json!([["h", "i"]])] {
            let json = serde_json::json!({"model": {"vocab": vocab, "merges": merges}});
            let t = Tokenizer::from_json(&json, Path::new("t.json")).unwrap();
            assert_eq!(t.encode("hi").unwrap(), [256]);
        }
    }

    // Spec §14: a broken file is an error naming what is wrong.
    #[test]
    fn broken_files_are_malformed() {
        let path = Path::new("t.json");
        let bad = [
            serde_json::json!({"model": {"vocab": {"a": 0}, "merges": ["a"]}}),
            serde_json::json!({"model": {"vocab": {"a": -1}, "merges": []}}),
            serde_json::json!({"model": {"vocab": 3, "merges": []}}),
        ];
        for json in bad {
            let err = Tokenizer::from_json(&json, path).err().expect("expected rejection");
            assert!(matches!(err, Error::Malformed { .. }), "{err}");
        }
    }

    // FR-13: text whose byte has no vocabulary entry loses that byte, like the reference.
    #[test]
    fn bytes_without_an_entry_are_dropped() {
        let map = ByteMap::new();
        let mut vocab = serde_json::Map::new();
        for b in (0..=255u8).filter(|&b| b != 4) {
            vocab.insert(map.char_for(b).to_string(), Value::from(b));
        }
        let json = serde_json::json!({"model": {"vocab": vocab, "merges": []}});
        let t = Tokenizer::from_json(&json, Path::new("t.json")).unwrap();
        assert_eq!(t.encode("a\u{4}b").unwrap(), [u32::from(b'a'), u32::from(b'b')]);
        assert_eq!(t.encode("\u{4}").unwrap(), Vec::<u32>::new());
    }

    // FR-13: the real tokenizer reproduces the reference's IDs on the whole corpus
    // (prompts, edge cases, special-token text, seeded random mixtures).
    #[test]
    fn matches_the_reference_on_the_corpus() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/smollm2-135m/tokenizer-corpus.json");
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("{path}: {e}; run `cd oracle && uv run python tokenizer_corpus.py`")
        });
        let corpus: Value = serde_json::from_str(&text).unwrap();
        let tokenizer = Tokenizer::load(&model_dir()).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        let mut failures = Vec::new();
        for case in cases {
            let input = case["text"].as_str().unwrap();
            let expected: Vec<u32> = case["ids"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as u32).collect();
            let ours = tokenizer.encode(input).unwrap();
            if ours != expected {
                failures.push(format!("{input:?}: ours {ours:?}, reference {expected:?}"));
            }
        }
        assert!(
            failures.is_empty(),
            "{} of {} differ, first: {:#?}",
            failures.len(),
            cases.len(),
            &failures[..failures.len().min(5)]
        );
    }

    #[test]
    fn real_special_tokens() {
        let t = Tokenizer::load(&model_dir()).unwrap();
        assert_eq!(t.encode("<|endoftext|>").unwrap(), [0]);
        assert_eq!(t.encode("<|im_start|><|im_end|>").unwrap(), [1, 2]);
    }
}
