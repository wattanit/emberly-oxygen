//! The tokenizer components the engine implements (Spec §6). Only what
//! SmolLM2's `tokenizer.json` actually uses is supported; any other type, or
//! an option outside what is implemented, is rejected naming it (FR-12).

use serde_json::Value;

use crate::Error;

/// Checks a parsed `tokenizer.json` against the supported set.
pub(super) fn check(json: &Value) -> Result<(), Error> {
    let field = |name: &str| json.get(name).unwrap_or(&Value::Null);

    // Nothing is implemented for these, so only "absent" is supported.
    for name in ["normalizer", "post_processor", "truncation", "padding"] {
        if let Some(found) = component_label(field(name)) {
            return Err(unsupported(name, found));
        }
    }

    check_model(field("model"))?;
    check_pre_tokenizer(field("pre_tokenizer"))?;

    // The decoder only reverses the byte-level alphabet (Spec §6); the
    // library's ByteLevel decoder ignores its other options, so do we.
    match type_of(field("decoder")).as_deref() {
        Some("ByteLevel") => {}
        other => return Err(unsupported("decoder", other.unwrap_or("absent").into())),
    }

    check_added_tokens(field("added_tokens"))
}

fn check_model(model: &Value) -> Result<(), Error> {
    if type_of(model).as_deref() != Some("BPE") {
        return Err(unsupported("model", type_of(model).unwrap_or_else(|| "absent".into())));
    }
    // (option, the one supported value): every option that would change
    // how pieces are formed is fixed at its plain-BPE setting.
    for (option, supported) in [
        ("dropout", Value::Null),
        ("unk_token", Value::Null),
        ("continuing_subword_prefix", Value::Null),
        ("end_of_word_suffix", Value::Null),
        ("fuse_unk", Value::Bool(false)),
        ("byte_fallback", Value::Bool(false)),
        ("ignore_merges", Value::Bool(false)),
    ] {
        let found = model.get(option).unwrap_or(&Value::Null);
        // fuse_unk, byte_fallback and ignore_merges may be absent, meaning false.
        let absent_is_false = supported == Value::Bool(false) && found.is_null();
        if *found != supported && !absent_is_false {
            return Err(unsupported(&format!("model {option}"), found.to_string()));
        }
    }
    Ok(())
}

/// Digits split individually, then the byte-level pattern, in that order.
fn check_pre_tokenizer(pre: &Value) -> Result<(), Error> {
    if type_of(pre).as_deref() != Some("Sequence") {
        return Err(unsupported("pre_tokenizer", type_of(pre).unwrap_or_else(|| "absent".into())));
    }
    let stages = pre
        .get("pretokenizers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let expected = ["Digits", "ByteLevel"];
    for (i, stage) in stages.iter().enumerate() {
        let kind = type_of(stage).unwrap_or_else(|| "absent".into());
        if expected.get(i) != Some(&kind.as_str()) {
            return Err(unsupported("pre_tokenizer stage", kind));
        }
        // (option, supported value)
        let options: &[(&str, bool)] = if kind == "Digits" {
            &[("individual_digits", true)]
        } else {
            &[("add_prefix_space", false), ("use_regex", true)]
        };
        for &(option, supported) in options {
            let found = stage.get(option).and_then(Value::as_bool);
            if found != Some(supported) {
                return Err(unsupported(
                    &format!("{kind} {option}"),
                    stage.get(option).map_or("absent".into(), Value::to_string),
                ));
            }
        }
    }
    if stages.len() != expected.len() {
        return Err(unsupported(
            "pre_tokenizer stages",
            format!("{} (expected Digits then ByteLevel)", stages.len()),
        ));
    }
    Ok(())
}

/// Added tokens are matched literally before splitting, so options that
/// change matching (surrounding whitespace, whole word, normalization) are not supported.
fn check_added_tokens(added: &Value) -> Result<(), Error> {
    for token in added.as_array().map(Vec::as_slice).unwrap_or_default() {
        let content = token.get("content").and_then(Value::as_str).unwrap_or("?");
        for option in ["single_word", "lstrip", "rstrip", "normalized"] {
            if token.get(option).and_then(Value::as_bool) != Some(false) {
                return Err(unsupported(&format!("added token {content} {option}"), "true".into()));
            }
        }
    }
    Ok(())
}

fn type_of(component: &Value) -> Option<String> {
    component.get("type").and_then(Value::as_str).map(str::to_owned)
}

/// `None` for an absent component; otherwise its type, or a rendering of the value.
fn component_label(component: &Value) -> Option<String> {
    if component.is_null() {
        None
    } else {
        Some(type_of(component).unwrap_or_else(|| component.to_string()))
    }
}

fn unsupported(component: &str, value: String) -> Error {
    Error::Unsupported {
        component: format!("tokenizer {component}"),
        value,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const REAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/smollm2-135m/tokenizer.json");

    fn real() -> Value {
        let text = std::fs::read_to_string(REAL).unwrap_or_else(|e| {
            panic!("{REAL}: {e}; run `cd oracle && uv run python download.py`")
        });
        serde_json::from_str(&text).unwrap()
    }

    // Spec §6, §19: the facts recorded for task 3.6, pinned so a model
    // revision cannot silently invalidate them.
    #[test]
    fn smollm2_uses_exactly_the_recorded_components() {
        let t = real();
        assert!(t["normalizer"].is_null());
        assert!(t["post_processor"].is_null());
        assert!(t["truncation"].is_null() && t["padding"].is_null());
        assert_eq!(t["model"]["type"], "BPE");
        assert_eq!(t["model"]["vocab"].as_object().unwrap().len(), 49152);
        assert_eq!(t["model"]["merges"].as_array().unwrap().len(), 48900);
        assert!(t["model"]["merges"][0].is_string(), "merges are \"a b\" strings");
        assert_eq!(
            t["pre_tokenizer"],
            json!({"type": "Sequence", "pretokenizers": [
                {"type": "Digits", "individual_digits": true},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            ]})
        );
        assert_eq!(t["decoder"]["type"], "ByteLevel");

        // 17 added tokens at IDs 0 to 16, all special, and also present in the vocabulary.
        let added = t["added_tokens"].as_array().unwrap();
        assert_eq!(added.len(), 17);
        for (i, token) in added.iter().enumerate() {
            assert_eq!(token["id"], i);
            assert_eq!(token["special"], true);
            assert_eq!(t["model"]["vocab"][token["content"].as_str().unwrap()], i);
        }
        assert_eq!(added[0]["content"], "<|endoftext|>");
    }

    #[test]
    fn the_real_file_is_supported() {
        check(&real()).unwrap();
    }

    fn rejected(path: &[&str], value: Value) -> String {
        let mut t = real();
        let mut at = &mut t;
        for key in &path[..path.len() - 1] {
            at = match key.parse::<usize>() {
                Ok(i) => &mut at[i],
                Err(_) => &mut at[*key],
            };
        }
        at[path[path.len() - 1]] = value;
        check(&t).err().expect("expected rejection").to_string()
    }

    // FR-12, Spec §6: any other component type is rejected naming it.
    #[test]
    fn other_components_are_rejected_by_name() {
        let cases = [
            (vec!["normalizer"], json!({"type": "NFC"}), "normalizer", "NFC"),
            (vec!["post_processor"], json!({"type": "TemplateProcessing"}), "post_processor", "TemplateProcessing"),
            (vec!["decoder"], json!({"type": "Metaspace"}), "decoder", "Metaspace"),
            (vec!["decoder"], Value::Null, "decoder", "absent"),
            (vec!["pre_tokenizer"], json!({"type": "Whitespace"}), "pre_tokenizer", "Whitespace"),
            (vec!["model", "type"], json!("Unigram"), "model", "Unigram"),
            (vec!["truncation"], json!({"max_length": 8}), "truncation", "max_length"),
        ];
        for (path, value, component, shown) in cases {
            let msg = rejected(&path, value);
            assert!(msg.contains(component) && msg.contains(shown), "{path:?}: {msg}");
        }
    }

    // FR-12, Spec §6: options outside what is implemented are rejected naming option and value.
    #[test]
    fn unsupported_options_are_rejected_by_name() {
        let cases = [
            (vec!["pre_tokenizer", "pretokenizers", "1", "add_prefix_space"], json!(true), "ByteLevel add_prefix_space", "true"),
            (vec!["pre_tokenizer", "pretokenizers", "1", "use_regex"], json!(false), "ByteLevel use_regex", "false"),
            (vec!["pre_tokenizer", "pretokenizers", "0", "individual_digits"], json!(false), "Digits individual_digits", "false"),
            (vec!["pre_tokenizer", "pretokenizers", "0", "type"], json!("Punctuation"), "pre_tokenizer stage", "Punctuation"),
            (vec!["pre_tokenizer", "pretokenizers"], json!([]), "pre_tokenizer stages", "0"),
            (vec!["model", "byte_fallback"], json!(true), "model byte_fallback", "true"),
            (vec!["model", "unk_token"], json!("<unk>"), "model unk_token", "<unk>"),
            (vec!["model", "dropout"], json!(0.1), "model dropout", "0.1"),
            (vec!["added_tokens", "3", "lstrip"], json!(true), "added token <repo_name> lstrip", "true"),
        ];
        for (path, value, component, shown) in cases {
            let msg = rejected(&path, value);
            assert!(msg.contains(component) && msg.contains(shown), "{path:?}: {msg}");
        }
    }
}
