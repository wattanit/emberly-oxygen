//! Streaming decoder (Spec §6, FR-15). A character can arrive in pieces: a
//! Thai character is three bytes and tokens routinely split it. The decoder
//! keeps the bytes of a character that is not complete yet and emits it once
//! the rest arrives, so streamed text never holds half a character.

use super::Tokenizer;
use crate::Error;

/// Decodes one token at a time. Everything pushed and flushed, joined, equals
/// what [`Tokenizer::decode`] returns for the same IDs.
pub struct StreamDecoder<'a> {
    tokenizer: &'a Tokenizer,
    /// The start of a character that is not complete yet: at most 3 bytes.
    pending: Vec<u8>,
}

impl Tokenizer {
    /// Starts a stream with nothing held back.
    pub fn stream_decoder(&self) -> StreamDecoder<'_> {
        StreamDecoder {
            tokenizer: self,
            pending: Vec::new(),
        }
    }
}

impl StreamDecoder<'_> {
    /// Feeds one token and returns the text that is now complete, which is
    /// empty while a character is still being assembled. Bytes that cannot
    /// begin or continue a valid sequence come out as U+FFFD at once, as in
    /// the one-shot decode. An ID outside the vocabulary is rejected and the
    /// stream is left as it was.
    pub fn push(&mut self, id: u32) -> Result<String, Error> {
        let mut bytes = std::mem::take(&mut self.pending);
        let held = bytes.len();
        if let Err(e) = self.tokenizer.append_bytes(id, &mut bytes) {
            bytes.truncate(held);
            self.pending = bytes;
            return Err(e);
        }

        let mut text = String::new();
        let mut at = 0;
        while at < bytes.len() {
            match std::str::from_utf8(&bytes[at..]) {
                Ok(valid) => {
                    text.push_str(valid);
                    at = bytes.len();
                }
                Err(e) => {
                    let (valid, rest) = bytes[at..].split_at(e.valid_up_to());
                    text.push_str(std::str::from_utf8(valid).expect("checked by valid_up_to"));
                    match e.error_len() {
                        // The bytes are invalid wherever the stream goes next.
                        Some(len) => {
                            text.push('\u{fffd}');
                            at += valid.len() + len;
                        }
                        // The end of a character is still to come.
                        None => {
                            self.pending = rest.to_vec();
                            debug_assert!(self.pending.len() <= 3);
                            return Ok(text);
                        }
                    }
                }
            }
        }
        Ok(text)
    }

    /// Ends the stream. Held bytes can no longer be completed and come out as
    /// U+FFFD, as the one-shot decode treats a truncated character.
    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::*;
    use crate::tokenizer::tests::{corpus, ids_of, model_dir};

    /// A tokenizer whose token IDs are the byte values, so any bytes can be pushed.
    fn bytes_tokenizer() -> Tokenizer {
        let map = crate::tokenizer::byte_map::ByteMap::new();
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(map.char_for(b).to_string(), Value::from(b));
        }
        let json = serde_json::json!({"model": {"vocab": vocab, "merges": []}});
        Tokenizer::from_json(&json, Path::new("t.json")).unwrap()
    }

    /// Pushes each chunk of bytes as one step (one token per byte), returning what each step emitted.
    fn stream(t: &Tokenizer, steps: &[&[u8]]) -> (Vec<String>, String) {
        let mut decoder = t.stream_decoder();
        let mut emitted = Vec::new();
        for step in steps {
            let mut text = String::new();
            for &b in *step {
                text.push_str(&decoder.push(u32::from(b)).unwrap());
            }
            emitted.push(text);
        }
        let last = decoder.finish();
        (emitted, last)
    }

    // FR-15, Spec §17: a three-byte Thai character split across every possible boundary.
    #[test]
    fn a_three_byte_character_waits_for_its_last_byte() {
        let t = bytes_tokenizer();
        let thai = "\u{e01}".as_bytes();
        assert_eq!(thai.len(), 3);
        for steps in [
            vec![&thai[..1], &thai[1..]],
            vec![&thai[..2], &thai[2..]],
            vec![&thai[..1], &thai[1..2], &thai[2..]],
        ] {
            let (emitted, last) = stream(&t, &steps);
            let (done, early) = emitted.split_last().unwrap();
            assert!(early.iter().all(String::is_empty), "nothing before the last byte: {emitted:?}");
            assert_eq!(done, "\u{e01}");
            assert_eq!(last, "");
        }
    }

    // FR-15: the same for a four-byte emoji, at every way of cutting it into steps.
    #[test]
    fn a_four_byte_character_at_every_cut() {
        let t = bytes_tokenizer();
        let emoji = "\u{1f600}".as_bytes();
        for mask in 0..8u32 {
            let mut steps: Vec<&[u8]> = Vec::new();
            let mut from = 0;
            for cut in 1..4 {
                if mask >> (cut - 1) & 1 == 1 {
                    steps.push(&emoji[from..cut]);
                    from = cut;
                }
            }
            steps.push(&emoji[from..]);
            let (emitted, last) = stream(&t, &steps);
            assert_eq!(emitted.concat(), "\u{1f600}", "{steps:?}");
            assert!(emitted[..emitted.len() - 1].iter().all(String::is_empty), "{steps:?}: {emitted:?}");
            assert_eq!(last, "");
        }
    }

    // FR-15: complete text is emitted at once and only the incomplete tail is held.
    #[test]
    fn complete_text_is_emitted_at_once_with_the_incomplete_tail_held() {
        let t = bytes_tokenizer();
        let mut d = t.stream_decoder();
        assert_eq!(d.push(u32::from(b'h')).unwrap(), "h");
        assert_eq!(d.push(0xE0).unwrap(), "");
        assert_eq!(d.pending, [0xE0]);
        assert_eq!(d.push(0xB8).unwrap(), "");
        assert_eq!(d.pending, [0xE0, 0xB8]);
        assert_eq!(d.push(0x81).unwrap(), "\u{e01}");
        assert!(d.pending.is_empty());
        assert_eq!(d.finish(), "");
    }

    // FR-15, Spec §6: bytes that cannot be part of a valid sequence become U+FFFD at once.
    #[test]
    fn invalid_bytes_become_replacements() {
        let t = bytes_tokenizer();
        let one = |bytes: &[u8]| stream(&t, &[bytes]);
        // A continuation byte with nothing to continue.
        assert_eq!(one(&[0x80]), (vec!["\u{fffd}".to_string()], String::new()));
        // A byte that is never valid.
        assert_eq!(one(&[0xFF, b'a']).0, ["\u{fffd}a"]);
        // A truncated four-byte sequence followed by ASCII: replaced when the ASCII arrives.
        assert_eq!(one(&[0xF0, 0x9F, 0x98, b'a']).0, ["\u{fffd}a"]);
        // The same when the pieces arrive one by one.
        let (emitted, last) = stream(&t, &[&[0xF0], &[0x9F], &[0x98], b"a"]);
        assert_eq!(emitted, ["", "", "", "\u{fffd}a"]);
        assert_eq!(last, "");
        // An overlong lead byte is invalid straight away.
        assert_eq!(one(&[0xC0, 0x80]).0, ["\u{fffd}\u{fffd}"]);
    }

    // FR-15: held bytes at the end are flushed as the one-shot decode treats them.
    #[test]
    fn finish_flushes_held_bytes() {
        let t = bytes_tokenizer();
        for held in [&[0xE0][..], &[0xE0, 0xB8], &[0xF0, 0x9F, 0x98]] {
            let (emitted, last) = stream(&t, &[held]);
            assert_eq!(emitted, [""]);
            assert_eq!(last, "\u{fffd}");
            let ids: Vec<u32> = held.iter().map(|&b| u32::from(b)).collect();
            assert_eq!(t.decode(&ids).unwrap(), "\u{fffd}", "one-shot agrees");
        }
    }

    // Spec §14: an unknown ID is rejected and the held bytes survive.
    #[test]
    fn a_rejected_id_leaves_the_stream_unchanged() {
        let t = bytes_tokenizer();
        let mut d = t.stream_decoder();
        assert_eq!(d.push(0xE0).unwrap(), "");
        assert!(matches!(d.push(9999), Err(Error::Rejected(_))));
        assert_eq!(d.pending, [0xE0]);
        assert_eq!(d.push(0xB8).unwrap(), "");
        assert_eq!(d.push(0x81).unwrap(), "\u{e01}");
    }

    // FR-15: streamed text equals the one-shot decode, never holds more than three
    // bytes, and only ever holds the start of a valid character. Checked over the
    // reference IDs of every corpus string, every random sequence and every single token.
    #[test]
    fn streaming_equals_one_shot_decoding() {
        let corpus = corpus();
        let tokenizer = Tokenizer::load(&model_dir()).unwrap();
        let mut sequences: Vec<Vec<u32>> = corpus["cases"].as_array().unwrap().iter().map(|c| ids_of(&c["ids"])).collect();
        sequences.extend(corpus["sequences"].as_array().unwrap().iter().map(|s| ids_of(&s["ids"])));
        sequences.extend((0..49152).map(|id| vec![id]));

        for ids in &sequences {
            let mut decoder = tokenizer.stream_decoder();
            let mut streamed = String::new();
            for &id in ids {
                streamed.push_str(&decoder.push(id).unwrap());
                assert!(decoder.pending.len() <= 3, "{ids:?}");
                if !decoder.pending.is_empty() {
                    let err = std::str::from_utf8(&decoder.pending).unwrap_err();
                    assert!(err.error_len().is_none() && err.valid_up_to() == 0, "{ids:?}: holds {:x?}", decoder.pending);
                }
            }
            streamed.push_str(&decoder.finish());
            assert_eq!(streamed, tokenizer.decode(ids).unwrap(), "{ids:?}");
        }
    }
}
