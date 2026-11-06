//! Byte-pair encoding (Spec §6): a piece written in the byte-level alphabet
//! starts as one symbol per character, and the adjacent pair with the lowest
//! merge rank is joined, again and again, until no ranked pair remains. Ties
//! go to the leftmost pair. The simplest correct form: a rescan per merge.

use std::collections::HashMap;

use crate::Error;

pub(super) struct Bpe {
    vocab: HashMap<String, u32>,
    /// (left, right) token IDs to (rank, merged token ID).
    merges: HashMap<(u32, u32), (u32, u32)>,
}

impl Bpe {
    /// `merges` are in rank order, lowest first. Every merge must name tokens
    /// that exist, including the token its two halves make.
    pub fn new(vocab: HashMap<String, u32>, merges: &[(String, String)]) -> Result<Bpe, String> {
        let mut table = HashMap::with_capacity(merges.len());
        for (rank, (left, right)) in merges.iter().enumerate() {
            let id = |piece: &str| {
                vocab
                    .get(piece)
                    .copied()
                    .ok_or_else(|| format!("merge {rank} ({left:?} {right:?}) names {piece:?}, which is not in the vocabulary"))
            };
            let pair = (id(left)?, id(right)?);
            let merged = id(&format!("{left}{right}"))?;
            // The first listing of a pair is the one that counts.
            table.entry(pair).or_insert((rank as u32, merged));
        }
        Ok(Bpe { vocab, merges: table })
    }

    pub fn id(&self, piece: &str) -> Option<u32> {
        self.vocab.get(piece).copied()
    }

    /// Token IDs for one piece, which must be written in the byte-level alphabet.
    pub fn encode_piece(&self, piece: &str) -> Result<Vec<u32>, Error> {
        let mut symbols = Vec::with_capacity(piece.len());
        for c in piece.chars() {
            // SmolLM2's vocabulary has no entry for 21 of the 256 byte characters
            // (six ASCII control characters among them), and the reference drops
            // a byte it cannot find. Matching it exactly is FR-13, so we do too.
            if let Some(id) = self.id(c.encode_utf8(&mut [0; 4])) {
                symbols.push(id);
            }
        }
        loop {
            let mut best: Option<(u32, usize, u32)> = None;
            for (i, pair) in symbols.windows(2).enumerate() {
                if let Some(&(rank, merged)) = self.merges.get(&(pair[0], pair[1])) {
                    if best.is_none_or(|(r, _, _)| rank < r) {
                        best = Some((rank, i, merged));
                    }
                }
            }
            let Some((_, at, merged)) = best else {
                return Ok(symbols);
            };
            symbols[at] = merged;
            symbols.remove(at + 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bpe(tokens: &[&str], merges: &[(&str, &str)]) -> Bpe {
        let vocab = tokens.iter().enumerate().map(|(i, t)| (t.to_string(), i as u32)).collect();
        let merges: Vec<_> = merges.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect();
        Bpe::new(vocab, &merges).unwrap()
    }

    // FR-13, Spec §6: the lowest rank is merged first, whatever its position.
    #[test]
    fn rank_order_decides() {
        let tokens = ["a", "b", "c", "ab", "bc"];
        // IDs: a 0, b 1, c 2, ab 3, bc 4.
        assert_eq!(bpe(&tokens, &[("a", "b"), ("b", "c")]).encode_piece("abc").unwrap(), [3, 2]);
        assert_eq!(bpe(&tokens, &[("b", "c"), ("a", "b")]).encode_piece("abc").unwrap(), [0, 4]);
    }

    // FR-13: equal pairs merge leftmost first; merges cascade; unranked pairs stay apart.
    #[test]
    fn ties_cascades_and_no_merges() {
        let tokens = ["a", "b", "aa", "ab", "aab"];
        let b = bpe(&tokens, &[("a", "a"), ("aa", "b")]);
        assert_eq!(b.encode_piece("aaa").unwrap(), [2, 0]); // leftmost pair first
        assert_eq!(b.encode_piece("aab").unwrap(), [4]); // aa, then aa+b
        assert_eq!(b.encode_piece("ba").unwrap(), [1, 0]); // no ranked pair
        assert_eq!(b.encode_piece("").unwrap(), Vec::<u32>::new());
        assert_eq!(bpe(&tokens, &[("a", "b")]).encode_piece("abab").unwrap(), [3, 3]);
    }

    // Spec §14: a vocabulary that disagrees with its merges is an error, not a panic.
    #[test]
    fn inconsistent_merges_are_rejected() {
        let vocab: HashMap<String, u32> = [("a", 0), ("b", 1)].map(|(t, i)| (t.to_string(), i)).into();
        let err = Bpe::new(vocab, &[("a".into(), "b".into())]).err().unwrap();
        assert!(err.contains("\"ab\""), "{err}");
    }

    // FR-13: a character with no vocabulary entry is dropped, as the reference does.
    #[test]
    fn characters_without_an_entry_are_dropped() {
        let b = bpe(&["a", "b"], &[]);
        assert_eq!(b.encode_piece("azb").unwrap(), [0, 1]);
        assert_eq!(b.encode_piece("z").unwrap(), Vec::<u32>::new());
    }
}
