//! The byte-level alphabet (Spec §6). Every byte gets one visible character,
//! so any byte string can be written as text and BPE merges never meet a
//! control character or a space. Printable bytes map to themselves; the rest
//! map, in byte order, to code points from U+0100 upward.

// Nothing calls the map until BPE encoding and decoding (tasks 3.9, 3.10),
// which remove this.
#![allow(dead_code)]

pub(super) struct ByteMap {
    to_char: [char; 256],
    /// Indexed by code point; `None` for characters outside the alphabet.
    to_byte: Vec<Option<u8>>,
}

impl ByteMap {
    pub fn new() -> ByteMap {
        let printable = |b: u32| matches!(b, 0x21..=0x7E | 0xA1..=0xAC | 0xAE..=0xFF);
        let mut to_char = ['\0'; 256];
        let mut next = 0x100;
        for b in 0..256u32 {
            let code = if printable(b) {
                b
            } else {
                next += 1;
                next - 1
            };
            to_char[b as usize] = char::from_u32(code).expect("below the surrogate range");
        }
        let max = to_char.iter().map(|&c| c as usize).max().expect("256 entries");
        let mut to_byte = vec![None; max + 1];
        for (b, &c) in to_char.iter().enumerate() {
            to_byte[c as usize] = Some(b as u8);
        }
        ByteMap { to_char, to_byte }
    }

    pub fn char_for(&self, byte: u8) -> char {
        self.to_char[byte as usize]
    }

    /// The byte a character stands for, or `None` if it is not in the alphabet.
    pub fn byte_for(&self, c: char) -> Option<u8> {
        self.to_byte.get(c as usize).copied().flatten()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // FR-13, FR-14, Spec §17: every byte survives the round trip.
    #[test]
    fn all_256_bytes_round_trip() {
        let map = ByteMap::new();
        for b in 0..=255u8 {
            assert_eq!(map.byte_for(map.char_for(b)), Some(b), "byte {b}");
        }
    }

    // Spec §6: the map is a bijection onto 256 distinct characters.
    #[test]
    fn the_alphabet_has_256_distinct_characters() {
        let map = ByteMap::new();
        let chars: HashSet<char> = (0..=255u8).map(|b| map.char_for(b)).collect();
        assert_eq!(chars.len(), 256);
        let decodable = (0..=0x200u32)
            .filter_map(char::from_u32)
            .filter(|&c| map.byte_for(c).is_some())
            .count();
        assert_eq!(decodable, 256, "no character outside the alphabet decodes");
    }

    // FR-13: known values of the standard byte-level alphabet.
    #[test]
    fn known_values() {
        let map = ByteMap::new();
        assert_eq!(map.char_for(b'a'), 'a');
        assert_eq!(map.char_for(b'~'), '~');
        assert_eq!(map.char_for(0xE9), 'é'); // printable Latin-1 maps to itself
        assert_eq!(map.char_for(b' '), 'Ġ'); // U+0120
        assert_eq!(map.char_for(b'\n'), 'Ċ'); // U+010A
        assert_eq!(map.char_for(0), '\u{100}');
        assert_eq!(map.char_for(0xAD), '\u{143}'); // the last remapped byte
        assert_eq!(map.byte_for('Ġ'), Some(b' '));
        assert_eq!(map.byte_for('€'), None);
        assert_eq!(map.byte_for('\u{144}'), None);
    }

    // FR-13: the real vocabulary is written in this alphabet, apart from added tokens.
    #[test]
    fn every_vocabulary_entry_is_in_the_alphabet() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/smollm2-135m/tokenizer.json");
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("{path}: {e}; run `cd oracle && uv run python download.py`"));
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        let added: HashSet<&str> = json["added_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| t["content"].as_str().unwrap())
            .collect();
        let map = ByteMap::new();
        let vocab = json["model"]["vocab"].as_object().unwrap();
        let mut checked = 0;
        for piece in vocab.keys().filter(|p| !added.contains(p.as_str())) {
            assert!(piece.chars().all(|c| map.byte_for(c).is_some()), "{piece:?}");
            checked += 1;
        }
        assert_eq!(checked, 49152 - 17);
    }
}
