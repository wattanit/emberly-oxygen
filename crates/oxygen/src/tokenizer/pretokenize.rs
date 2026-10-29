//! The pre-tokenizer SmolLM2 declares (Spec §6): digits are isolated one by
//! one, then each remaining stretch is split by the byte-level pattern
//!
//! ```text
//! 's|'t|'re|'ve|'m|'ll|'d | ?L+ | ?N+ | ?[^\s L N]+ | \s+(?!\S) | \s+
//! ```
//!
//! where L is a Unicode letter, N a number and `\s` white space. It is a
//! hand-written scanner: the pattern needs lookahead, and a regex crate is
//! not a dependency. At each position the alternatives are tried in the order
//! above and the first that matches wins, as in the reference.
//!
//! Character classes come from `unicode_classes`, measured from the reference
//! itself rather than from Rust's standard library, so the two cannot disagree
//! about what a letter is (Thai vowel and tone marks are not letters).

use super::unicode_classes::{LETTERS, NUMBERS, SPACES};

fn in_class(table: &[(u32, u32)], c: char) -> bool {
    let c = c as u32;
    table
        .binary_search_by(|&(lo, hi)| {
            if c < lo {
                std::cmp::Ordering::Greater
            } else if c > hi {
                std::cmp::Ordering::Less
            } else {
                std::cmp::Ordering::Equal
            }
        })
        .is_ok()
}

fn is_letter(c: char) -> bool {
    in_class(LETTERS, c)
}

fn is_number(c: char) -> bool {
    in_class(NUMBERS, c)
}

fn is_space(c: char) -> bool {
    in_class(SPACES, c)
}

/// Splits `text` into pieces that, joined, give back `text` exactly.
pub(super) fn split(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    for stretch in isolate_numbers(text) {
        split_byte_level(stretch, &mut pieces);
    }
    pieces
}

/// The digit stage: every number character becomes a piece of its own.
fn isolate_numbers(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (i, c) in text.char_indices() {
        if is_number(c) {
            if start < i {
                out.push(&text[start..i]);
            }
            out.push(&text[i..i + c.len_utf8()]);
            start = i + c.len_utf8();
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The byte-level stage over one stretch.
fn split_byte_level<'a>(text: &'a str, out: &mut Vec<&'a str>) {
    let mut at = 0;
    while at < text.len() {
        let len = match_at(&text[at..]);
        debug_assert!(len > 0, "the pattern always consumes something");
        out.push(&text[at..at + len]);
        at += len;
    }
}

/// Length in bytes of the match at the start of `rest` (non-empty).
fn match_at(rest: &str) -> usize {
    // 's|'t|'re|'ve|'m|'ll|'d
    if let Some(after) = rest.strip_prefix('\'') {
        for suffix in ["s", "t", "re", "ve", "m", "ll", "d"] {
            if after.starts_with(suffix) {
                return 1 + suffix.len();
            }
        }
    }

    // ` ?` then a run of one class, tried for letters, numbers, then everything else.
    let lead = if rest.starts_with(' ') { 1 } else { 0 };
    let body = &rest[lead..];
    for class in [is_letter as fn(char) -> bool, is_number, is_other] {
        let run = run_len(body, class);
        if run > 0 {
            return lead + run;
        }
    }

    // `\s+(?!\S)` then `\s+`: the whole white-space run, except that when
    // something other than white space follows, the last character is left
    // behind so it can lead the next piece.
    let run = run_len(rest, is_space);
    debug_assert!(run > 0, "any other character is matched above");
    let end = rest[run..].chars().next().is_none();
    if end {
        return run;
    }
    let last = rest[..run].chars().next_back().expect("run is non-empty").len_utf8();
    if run > last { run - last } else { run }
}

/// Neither white space, a letter nor a number.
fn is_other(c: char) -> bool {
    !is_space(c) && !is_letter(c) && !is_number(c)
}

fn run_len(text: &str, class: fn(char) -> bool) -> usize {
    text.char_indices()
        .find(|&(_, c)| !class(c))
        .map_or(text.len(), |(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::byte_map::ByteMap;

    // FR-13, Spec §6: pieces always join back to the input, and match the expected split.
    fn check(text: &str, expected: &[&str]) {
        let pieces = split(text);
        assert_eq!(pieces, expected, "{text:?}");
        assert_eq!(pieces.concat(), text);
    }

    #[test]
    fn words_and_leading_spaces() {
        check("Hello world", &["Hello", " world"]);
        check("Hello  world", &["Hello", " ", " world"]);
        check("a b c", &["a", " b", " c"]);
        check("", &[]);
    }

    // The contraction alternatives come first and are case-sensitive.
    #[test]
    fn contractions() {
        check("it's", &["it", "'s"]);
        check("I'm you're we've they'll he'd don't", &["I", "'m", " you", "'re", " we", "'ve", " they", "'ll", " he", "'d", " don", "'t"]);
        check("IT'S", &["IT", "'", "S"]);
        check("'ss", &["'s", "s"]);
        check("'", &["'"]);
        check("rock 'n' roll", &["rock", " '", "n", "'", " roll"]);
    }

    // The digit stage runs first, so every digit stands alone and takes no space.
    #[test]
    fn digits() {
        check("a123", &["a", "1", "2", "3"]);
        check("12 34", &["1", "2", " ", "3", "4"]);
        check("1,234.56", &["1", ",", "2", "3", "4", ".", "5", "6"]);
        check("\u{0661}\u{2167}", &["\u{0661}", "\u{2167}"]);
    }

    #[test]
    fn punctuation_runs() {
        check("!!", &["!!"]);
        check(" !!", &[" !!"]);
        check("a !b", &["a", " !", "b"]);
        check("...", &["..."]);
        check("(a)", &["(", "a", ")"]);
    }

    // `\s+(?!\S)`: the last white-space character before a word leads the word.
    #[test]
    fn white_space_runs() {
        check("a  b", &["a", " ", " b"]);
        check("a \n\n x", &["a", " \n\n", " x"]);
        check("a\n\nb", &["a", "\n", "\n", "b"]);
        check("a   ", &["a", "   "]);
        check("\t\t x", &["\t\t", " x"]);
        check("\n", &["\n"]);
        check("a\u{a0}b", &["a", "\u{a0}", "b"]);
    }

    // Thai vowel and tone marks are not letters, so they split off.
    #[test]
    fn thai() {
        check("\u{0e01}\u{0e47}", &["\u{0e01}", "\u{0e47}"]);
        check("\u{0e44}\u{0e14}\u{0e49}", &["\u{0e44}\u{0e14}", "\u{0e49}"]);
        check("\u{0e51}\u{0e52}", &["\u{0e51}", "\u{0e52}"]);
        check("a\u{0e31}b", &["a", "\u{0e31}", "b"]);
    }

    #[test]
    fn emoji_and_mixed() {
        check("a\u{1f600}b", &["a", "\u{1f600}", "b"]);
        check("\u{1f468}\u{200d}\u{1f469}", &["\u{1f468}\u{200d}\u{1f469}"]);
        check("x=1+2", &["x", "=", "1", "+", "2"]);
    }

    // Spec §14: any input is split without panicking and without losing text.
    #[test]
    fn every_scalar_value_is_covered() {
        for cp in (0..=0x10FFFFu32).filter_map(char::from_u32) {
            let text = format!("a{cp} {cp}\u{0301}");
            assert_eq!(split(&text).concat(), text, "U+{:04X}", cp as u32);
        }
    }

    // FR-13: the splitter agrees with the reference on the whole differential
    // corpus (prompts, edge cases and seeded random mixtures).
    #[test]
    fn matches_the_reference_on_the_corpus() {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/smollm2-135m/tokenizer-corpus.json");
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
            panic!("{path}: {e}; run `cd oracle && uv run python tokenizer_corpus.py`")
        });
        let corpus: serde_json::Value = serde_json::from_str(&text).unwrap();
        let cases = corpus["cases"].as_array().unwrap();
        assert!(cases.len() > 3000, "corpus looks truncated: {} cases", cases.len());

        let map = ByteMap::new();
        let mut failures = Vec::new();
        for case in cases {
            let input = case["text"].as_str().unwrap();
            let expected: Vec<&str> = case["pieces"]
                .as_array()
                .unwrap()
                .iter()
                .map(|p| p.as_str().unwrap())
                .collect();
            let ours: Vec<String> = split(input)
                .iter()
                .map(|piece| piece.bytes().map(|b| map.char_for(b)).collect())
                .collect();
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
}
