mod bpe;
mod byte_map;
mod components;
mod pretokenize;
mod stream;
mod unicode_classes;

use std::collections::HashMap;
use std::path::Path;

use serde_json::Value;

use crate::Error;
use bpe::Bpe;
use byte_map::ByteMap;
pub use stream::StreamDecoder;

/// Byte-level BPE tokenizer (Spec §6).
pub struct Tokenizer {
    bpe: Bpe,
    /// The piece (in the byte-level alphabet) of each token ID.
    pieces: Vec<String>,
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
        // Decoding needs ID to piece, so IDs must be exactly 0 to n-1, each once.
        let mut pieces = vec![String::new(); vocab.len()];
        let mut seen = vec![false; vocab.len()];
        for (piece, &id) in &vocab {
            let slot = id as usize;
            if slot >= pieces.len() || std::mem::replace(&mut seen[slot], true) {
                return Err(malformed(format!(
                    "vocabulary IDs are not exactly 0 to {}: {piece:?} has ID {id}",
                    pieces.len() - 1
                )));
            }
            pieces[slot] = piece.clone();
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
            if pieces.get(id as usize).map(String::as_str) != Some(content) {
                return Err(malformed(format!("added token {content:?} is not in the vocabulary with ID {id}")));
            }
            added.push((content.to_owned(), id));
        }
        Ok(Tokenizer { bpe, pieces, byte_map: ByteMap::new(), added })
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

    /// Token IDs to text (FR-14). The bytes of all tokens are joined before
    /// they are read as UTF-8, so a character split across tokens comes out
    /// whole; bytes that form no valid sequence become U+FFFD, as in the
    /// reference. Added tokens print as their text, like any other token.
    pub fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            self.append_bytes(id, &mut bytes)?;
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    /// The raw bytes one token stands for. A character outside the byte-level
    /// alphabet stands for no byte and is skipped, as the reference does.
    fn append_bytes(&self, id: u32, out: &mut Vec<u8>) -> Result<(), Error> {
        let piece = self
            .pieces
            .get(id as usize)
            .ok_or_else(|| Error::Rejected(format!("token ID {id} is not in the vocabulary (size {})", self.pieces.len())))?;
        out.extend(piece.chars().filter_map(|c| self.byte_map.byte_for(c)));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    pub(super) fn model_dir() -> PathBuf {
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
        for (i, content) in added.iter().enumerate() {
            vocab.insert(content.to_string(), Value::from(first_added + i));
        }
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

    // FR-14: decoding reverses the byte-level mapping, including multi-byte characters.
    #[test]
    fn decoding_reverses_the_byte_mapping() {
        let t = tiny(&[], &[], &["<s>"]);
        let ids = |text: &str| text.bytes().map(u32::from).collect::<Vec<_>>();
        for text in ["", "hello world", "line\n\ttab", "\u{e01}\u{e47}", "caf\u{e9}", "\u{1f600}", "a\u{1f468}\u{200d}\u{1f469}b"] {
            assert_eq!(t.decode(&ids(text)).unwrap(), text);
        }
        assert_eq!(t.decode(&[256]).unwrap(), "<s>", "added tokens print as their text");
        assert_eq!(t.decode(&[u32::from(b'x'), 256, u32::from(b'y')]).unwrap(), "x<s>y");
    }

    // FR-14: bytes of one character may arrive in separate tokens; invalid bytes become U+FFFD.
    #[test]
    fn split_and_invalid_bytes() {
        let t = tiny(&[], &[], &[]);
        let thai = "\u{e01}".as_bytes(); // three bytes
        let whole: Vec<u32> = thai.iter().map(|&b| u32::from(b)).collect();
        assert_eq!(t.decode(&whole).unwrap(), "\u{e01}");
        assert_eq!(t.decode(&whole[..2]).unwrap(), "\u{fffd}", "an incomplete sequence is one replacement");
        assert_eq!(t.decode(&[0x80]).unwrap(), "\u{fffd}");
        assert_eq!(t.decode(&[0xFF, u32::from(b'a')]).unwrap(), "\u{fffd}a");
        assert_eq!(t.decode(&[0xE0, 0xB8, u32::from(b'a')]).unwrap(), "\u{fffd}a");
    }

    // Spec §14: an unknown ID is a rejected request, never a panic.
    #[test]
    fn out_of_range_ids_are_rejected_by_name() {
        let t = tiny(&[], &[], &[]);
        let err = t.decode(&[65, 256]).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("256"), "{err}");
        assert!(t.decode(&[u32::MAX]).is_err());
    }

    // FR-14: the vocabulary must map IDs to pieces unambiguously.
    #[test]
    fn vocabularies_with_gaps_or_repeats_are_malformed() {
        let map = ByteMap::new();
        let base = |extra: (&str, u64)| {
            let mut vocab = serde_json::Map::new();
            for b in 0..=255u8 {
                vocab.insert(map.char_for(b).to_string(), Value::from(b));
            }
            vocab.insert(extra.0.into(), Value::from(extra.1));
            serde_json::json!({"model": {"vocab": vocab, "merges": []}})
        };
        for extra in [("gap", 300), ("repeat", 7)] {
            let err = Tokenizer::from_json(&base(extra), Path::new("t.json")).err().expect("rejected");
            assert!(matches!(err, Error::Malformed { .. }) && err.to_string().contains("IDs"), "{err}");
        }
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
        // Contiguous IDs, with the character of byte 4 left out.
        for (id, b) in (0..=255u8).filter(|&b| b != 4).enumerate() {
            vocab.insert(map.char_for(b).to_string(), Value::from(id));
        }
        let (a, b) = (vocab["a"].as_u64().unwrap() as u32, vocab["b"].as_u64().unwrap() as u32);
        let json = serde_json::json!({"model": {"vocab": vocab, "merges": []}});
        let t = Tokenizer::from_json(&json, Path::new("t.json")).unwrap();
        assert_eq!(t.encode("a\u{4}b").unwrap(), [a, b]);
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

    pub(super) fn corpus() -> Value {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/smollm2-135m/tokenizer-corpus.json");
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("{path}: {e}; run `cd oracle && uv run python tokenizer_corpus.py`")
        });
        serde_json::from_str(&text).unwrap()
    }

    pub(super) fn ids_of(value: &Value) -> Vec<u32> {
        value.as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as u32).collect()
    }

    // FR-14: decoding matches the oracle's decoder for the reference IDs of every
    // corpus string, for every single token in the vocabulary, and for random ID
    // sequences that split characters across tokens and form invalid UTF-8.
    #[test]
    fn decoding_matches_the_reference() {
        let corpus = corpus();
        let tokenizer = Tokenizer::load(&model_dir()).unwrap();
        let mut failures = Vec::new();
        let mut check = |what: String, ids: Vec<u32>, expected: &str| {
            let ours = tokenizer.decode(&ids).unwrap();
            if ours != expected {
                failures.push(format!("{what} {ids:?}: ours {ours:?}, reference {expected:?}"));
            }
        };

        let cases = corpus["cases"].as_array().unwrap();
        for case in cases {
            check("case".into(), ids_of(&case["ids"]), case["decoded"].as_str().unwrap());
        }
        let singles = corpus["single_tokens"].as_array().unwrap();
        assert_eq!(singles.len(), 49152);
        for (id, expected) in singles.iter().enumerate() {
            check("token".into(), vec![id as u32], expected.as_str().unwrap());
        }
        let sequences = corpus["sequences"].as_array().unwrap();
        assert!(sequences.len() >= 4000);
        for seq in sequences {
            check("sequence".into(), ids_of(&seq["ids"]), seq["decoded"].as_str().unwrap());
        }
        assert!(
            failures.is_empty(),
            "{} of {} differ, first: {:#?}",
            failures.len(),
            cases.len() + singles.len() + sequences.len(),
            &failures[..failures.len().min(5)]
        );
    }

    // FR-13, FR-14: text survives encode then decode, except bytes the vocabulary cannot hold.
    #[test]
    fn encode_then_decode_round_trips_ordinary_text() {
        let tokenizer = Tokenizer::load(&model_dir()).unwrap();
        for text in ["Hello, world!", "  indented\n\tcode()", "\u{e2a}\u{e27}\u{e31}\u{e2a}\u{e14}\u{e35}", "\u{1f468}\u{200d}\u{1f469} <|im_start|>x"] {
            assert_eq!(tokenizer.decode(&tokenizer.encode(text).unwrap()).unwrap(), text);
        }
    }

    #[test]
    fn real_special_tokens() {
        let t = Tokenizer::load(&model_dir()).unwrap();
        assert_eq!(t.encode("<|endoftext|>").unwrap(), [0]);
        assert_eq!(t.encode("<|im_start|><|im_end|>").unwrap(), [1, 2]);
    }
}
