//! The instruct model differs from the base model only where Specification §3
//! says, so it loads with no instruct-specific code (FR-32), and the chat
//! markers are single tokens (FR-33). These pin facts about the real files: a
//! revision that differed would fail here, not run.

mod common;

use common::fixtures::{require_model_dir, workspace_root};
use emberly_oxygen::{Model, Tokenizer};
use serde_json::{json, Value};

fn instruct_dir() -> std::path::PathBuf {
    let dir = workspace_root().join("models").join("smollm2-135m-instruct");
    require_model_dir(&dir).unwrap_or_else(|e| panic!("{e}"));
    dir
}

fn read(path: std::path::PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

// FR-32: the tokenizer is the base model's, so nothing in tokenization changes.
#[test]
fn fr32_the_instruct_tokenizer_equals_the_base_models() {
    let base = workspace_root().join("models").join("smollm2-135m");
    assert_eq!(read(instruct_dir().join("tokenizer.json")), read(base.join("tokenizer.json")));
}

// FR-32: the configuration differs only in the fields Spec §3 lists, with these values.
#[test]
fn fr32_the_instruct_config_differs_from_the_base_models_only_where_the_spec_says() {
    let base = read(workspace_root().join("models").join("smollm2-135m").join("config.json"));
    let instruct = read(instruct_dir().join("config.json"));
    let allowed: [(&str, Value); 6] = [
        ("bos_token_id", json!(1)),
        ("eos_token_id", json!(2)),
        ("pad_token_id", json!(2)),
        ("mlp_bias", json!(false)),
        ("transformers_version", instruct["transformers_version"].clone()),
        ("transformers.js_config", instruct["transformers.js_config"].clone()),
    ];
    let (base, instruct) = (base.as_object().unwrap(), instruct.as_object().unwrap());
    let keys: std::collections::BTreeSet<_> = base.keys().chain(instruct.keys()).collect();
    for key in keys {
        if base.get(key) != instruct.get(key) {
            let (_, want) = allowed
                .iter()
                .find(|(k, _)| k == key)
                .unwrap_or_else(|| panic!("config field {key} differs and Spec §3 does not list it"));
            assert_eq!(instruct.get(key), Some(want), "{key}");
        }
    }
}

// FR-32, FR-16: the model loads through the ordinary path and its end-of-sequence token is <|im_end|>.
#[test]
fn fr32_the_instruct_model_loads_and_stops_at_im_end() {
    let model = Model::load(&instruct_dir()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(model.config().eos_token_id, 2);
    assert_eq!(model.config().vocab_size, 49152);
}

// FR-33: each marker is one token with the ID the vocabulary gives it, wherever it appears.
#[test]
fn fr33_the_chat_markers_encode_as_single_tokens() {
    let t = Tokenizer::load(&instruct_dir()).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(t.encode("<|im_start|>").unwrap(), vec![1]);
    assert_eq!(t.encode("<|im_end|>").unwrap(), vec![2]);
    let ids = t.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n").unwrap();
    assert_eq!((ids.iter().filter(|&&i| i == 1).count(), ids.iter().filter(|&&i| i == 2).count()), (2, 1));
    assert_eq!((ids[0], ids[ids.len() - 1]), (1, 198), "starts with the marker and ends with the newline token");
}
