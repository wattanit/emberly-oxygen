//! Tests of the fixture reader. They verify FR-4 (stored fixtures are read as
//! they were written) and FR-7 (a missing input fails naming the command that
//! creates it).

mod common;

use std::path::PathBuf;

use common::fixtures::{
    discover, discover_chat, parse_safetensors, parse_safetensors_f64, require_model_dir, workspace_root, CHAT_COMMAND,
    DOWNLOAD_COMMAND, GENERATE_COMMAND, SAMPLING_COMMAND,
};

/// Builds a safetensors file in memory: (name, dtype, shape, values).
fn build(tensors: &[(&str, &str, &[usize], &[f32])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, values) in tensors {
        let begin = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    file
}

/// The same, with 8-byte values.
fn build64(tensors: &[(&str, &str, &[usize], &[f64])]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, dtype, shape, values) in tensors {
        let begin = data.len();
        for v in *values {
            data.extend_from_slice(&v.to_le_bytes());
        }
        header.insert(
            name.to_string(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len()]}),
        );
    }
    let header = serde_json::to_vec(&header).unwrap();
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(data);
    file
}

#[test]
fn fr4_reads_f64_tensors_exactly() {
    // 0.1 and a value one part in 1e12 off 1.0 are not representable in f32: an f64 reader must keep them whole.
    let values = [0.1f64, 1.0 + 1e-12, -2.5, 3.0e-300];
    let file = build64(&[("a", "F64", &[2, 2], &values)]);
    let tensors = parse_safetensors_f64(&file).unwrap();
    assert_eq!(tensors["a"].shape, [2, 2]);
    assert_eq!(tensors["a"].data, values, "values are bit-exact, not narrowed to f32");
}

#[test]
fn fr4_a_dtype_other_than_the_one_asked_for_is_refused_naming_both() {
    let f64_file = build64(&[("a", "F64", &[1], &[1.0])]);
    let error = parse_safetensors(&f64_file).unwrap_err();
    assert!(error.contains("F64") && error.contains("F32"), "{error}");

    let f32_file = build(&[("a", "F32", &[1], &[1.0])]);
    let error = parse_safetensors_f64(&f32_file).unwrap_err();
    assert!(error.contains("F32") && error.contains("F64"), "{error}");
}

#[test]
fn fr4_bad_f64_files_fail_without_panicking() {
    let good = build64(&[("a", "F64", &[2], &[1.0, 2.0])]);
    assert!(parse_safetensors_f64(&good[..4]).is_err(), "shorter than the length prefix");
    assert!(parse_safetensors_f64(&good[..good.len() - 1]).is_err(), "data cut short");
    let wrong_count = build64(&[("a", "F64", &[3], &[1.0, 2.0])]);
    assert!(parse_safetensors_f64(&wrong_count).is_err(), "shape disagrees with byte range");
}

#[test]
fn fr4_reads_f32_tensors_by_name() {
    let file = build(&[
        ("a", "F32", &[2, 2], &[1.0, -2.5, 3.0, 4.25]),
        ("b", "F32", &[3], &[0.5, 0.25, -1.0]),
    ]);
    let tensors = parse_safetensors(&file).unwrap();
    assert_eq!(tensors["a"].shape, [2, 2]);
    assert_eq!(tensors["a"].data, [1.0, -2.5, 3.0, 4.25]);
    assert_eq!(tensors["b"].data, [0.5, 0.25, -1.0]);
    assert_eq!(tensors.len(), 2, "__metadata__ is not a tensor");
}

#[test]
fn fr4_rejects_other_dtypes_and_bad_files_without_panicking() {
    let bf16 = build(&[("a", "BF16", &[1], &[1.0])]);
    assert!(parse_safetensors(&bf16).unwrap_err().contains("BF16"));

    let good = build(&[("a", "F32", &[2], &[1.0, 2.0])]);
    assert!(parse_safetensors(&good[..4]).is_err(), "shorter than the length prefix");
    assert!(parse_safetensors(&good[..12]).is_err(), "header cut short");
    assert!(parse_safetensors(&good[..good.len() - 1]).is_err(), "data cut short");

    let wrong_count = build(&[("a", "F32", &[3], &[1.0, 2.0])]);
    assert!(parse_safetensors(&wrong_count).is_err(), "shape disagrees with byte range");
}

/// A fresh empty directory for one test.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oxygen-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn fr7_missing_fixtures_name_the_oracle_command() {
    let empty = temp_dir("empty-fixtures");
    let error = discover(&empty).unwrap_err();
    assert!(error.contains(GENERATE_COMMAND), "{error}");

    let absent = empty.join("nope");
    assert!(discover(&absent).unwrap_err().contains(GENERATE_COMMAND));
    std::fs::remove_dir_all(&empty).unwrap();
}

#[test]
fn fr7_a_folder_without_a_manifest_is_not_a_fixture_folder() {
    let root = temp_dir("no-manifest");
    std::fs::create_dir_all(root.join("some-model")).unwrap();
    assert!(discover(&root).unwrap_err().contains(GENERATE_COMMAND));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr7_a_malformed_manifest_names_the_file() {
    let root = temp_dir("bad-manifest");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), "{ not json").unwrap();
    let error = discover(&root).unwrap_err();
    assert!(error.contains("manifest.json") && error.contains("malformed"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

/// A one-prompt manifest, with or without the oracle's `f64_` fields (Spec §10).
fn tiny_manifest(with_f64: bool) -> serde_json::Value {
    let mut prompt = serde_json::json!({
        "id": "p", "category": "english", "text": "x", "token_ids": [1],
        "prompt_top2_gaps": [1.0], "continuation_ids": [1], "continuation_text": "x",
        "continuation_top2_gaps": [1.0], "stopped_at_eos": false,
        "tensors_file": "p.safetensors"
    });
    if with_f64 {
        let extra = serde_json::json!({
            "f64_tensors_file": "p.f64.safetensors", "f64_prompt_top2_gaps": [1.0],
            "f64_continuation_ids": [1], "f64_continuation_text": "x",
            "f64_continuation_top2_gaps": [1.0], "f64_stopped_at_eos": false
        });
        prompt.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
    }
    serde_json::json!({
        "model_id": "m",
        "dims": {"layers": 1, "hidden": 1, "vocab": 1},
        "continuation_tokens": 1,
        "eos_token_id": 0,
        "prompts": [prompt]
    })
}

#[test]
fn fr7_a_manifest_without_the_f64_fields_names_the_oracle_command() {
    let root = temp_dir("no-f64-fields");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(false).to_string()).unwrap();
    let error = discover(&root).unwrap_err();
    assert!(error.contains("f64_") && error.contains(GENERATE_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr7_a_missing_f64_file_fails_naming_it_and_the_command() {
    let root = temp_dir("missing-f64");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(true).to_string()).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_reference(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("p.f64.safetensors") && error.contains(GENERATE_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr4_an_f32_file_where_the_f64_file_is_named_is_refused() {
    let root = temp_dir("f32-as-f64");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(true).to_string()).unwrap();
    std::fs::write(root.join("m/p.f64.safetensors"), build(&[("logits", "F32", &[1, 1], &[1.0])])).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_reference(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("F32") && error.contains("F64") && error.contains(GENERATE_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr7_a_manifest_naming_a_missing_tensors_file_fails_naming_it() {
    let root = temp_dir("missing-tensors");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(true).to_string()).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_tensors(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("p.safetensors") && error.contains(GENERATE_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr7_missing_model_files_name_the_download_command() {
    let dir = temp_dir("model");
    let error = require_model_dir(&dir).unwrap_err();
    assert!(error.contains("config.json") && error.contains(DOWNLOAD_COMMAND), "{error}");

    std::fs::write(dir.join("config.json"), "{}").unwrap();
    std::fs::write(dir.join("model.safetensors"), "").unwrap();
    let error = require_model_dir(&dir).unwrap_err();
    assert!(error.contains("tokenizer.json"), "names the file that is missing: {error}");

    std::fs::write(dir.join("tokenizer.json"), "{}").unwrap();
    require_model_dir(&dir).unwrap();
    assert!(require_model_dir(&dir.join("absent")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn fr4_real_fixture_shapes_match_the_manifest() {
    let all = discover(&workspace_root().join("fixtures")).unwrap();
    for fixtures in &all {
        let prompt = fixtures
            .manifest
            .prompts
            .iter()
            .min_by_key(|p| p.token_ids.len())
            .expect("manifest has prompts");
        let t = fixtures.load_tensors(prompt).unwrap();
        let positions = prompt.token_ids.len();
        assert_eq!(t.logits.data.len(), positions * fixtures.manifest.dims.vocab);
        assert_eq!(t.layer_outputs.shape[0], fixtures.manifest.dims.layers);
        assert_eq!(prompt.prompt_top2_gaps.len(), positions);
        assert!(t.logits.data.iter().all(|v| v.is_finite()));

        // The f64 reference has the same shapes and is F64 on disk (FR-2, FR-4).
        let r = fixtures.load_reference(prompt).unwrap();
        assert_eq!(r.logits.shape, t.logits.shape);
        assert_eq!(r.layer_outputs.shape, t.layer_outputs.shape);
        assert_eq!(r.embeddings.shape, t.embeddings.shape);
        assert_eq!(r.final_norm_output.shape, t.final_norm_output.shape);
        assert_eq!(prompt.f64_prompt_top2_gaps.len(), positions);
        assert_eq!(prompt.f64_continuation_top2_gaps.len(), prompt.f64_continuation_ids.len());
        assert!(r.logits.data.iter().all(|v| v.is_finite()));
    }
}

/// `tiny_manifest` plus the oracle's `sampling` entry and the prompt's file name
/// (Spec §10, FR-31).
fn tiny_manifest_with_sampling() -> serde_json::Value {
    let mut m = tiny_manifest(true);
    m["prompts"][0]["sampling_file"] = "p.sampling.safetensors".into();
    m["sampling"] = serde_json::json!({
        "settings": [
            {"name": "hot", "temperature": 0.7},
            {"name": "all", "temperature": 0.7, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.3}
        ],
        "context": "the prompt's token ids",
        "margins": {"p": {
            "hot": {"top_k_margin_logit": null, "top_p_margin_prob": null, "support": 1},
            "all": {"top_k_margin_logit": 0.25, "top_p_margin_prob": 0.125, "support": 1}
        }}
    });
    m
}

#[test]
fn fr31_a_manifest_without_the_sampling_entry_names_the_oracle_command() {
    let root = temp_dir("no-sampling-entry");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(true).to_string()).unwrap();
    // The greedy harness must still run on fixtures that predate sampling.
    let all = discover(&root).unwrap();
    let error = all[0].sampling().unwrap_err();
    assert!(error.contains("sampling") && error.contains(SAMPLING_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_a_missing_sampling_file_fails_naming_it_and_the_command() {
    let root = temp_dir("missing-sampling-file");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest_with_sampling().to_string()).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_sampling(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("p.sampling.safetensors") && error.contains(SAMPLING_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_a_sampling_distribution_of_the_wrong_shape_is_refused() {
    let root = temp_dir("sampling-wrong-shape");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest_with_sampling().to_string()).unwrap();
    // vocab is 1 in the tiny manifest; a two-entry distribution does not fit.
    let file = build64(&[("hot", "F64", &[2], &[0.5, 0.5]), ("all", "F64", &[1], &[1.0])]);
    std::fs::write(root.join("m/p.sampling.safetensors"), file).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_sampling(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("hot") && error.contains("shape"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_a_setting_without_its_distribution_is_named() {
    let root = temp_dir("sampling-missing-setting");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest_with_sampling().to_string()).unwrap();
    std::fs::write(root.join("m/p.sampling.safetensors"), build64(&[("hot", "F64", &[1], &[1.0])])).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_sampling(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("all") && error.contains("missing"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_a_sampling_file_in_the_wrong_precision_is_refused() {
    let root = temp_dir("sampling-f32");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest_with_sampling().to_string()).unwrap();
    let file = build(&[("hot", "F32", &[1], &[1.0]), ("all", "F32", &[1], &[1.0])]);
    std::fs::write(root.join("m/p.sampling.safetensors"), file).unwrap();
    let all = discover(&root).unwrap();
    let error = all[0].load_sampling(&all[0].manifest.prompts[0]).unwrap_err();
    assert!(error.contains("F32") && error.contains("F64"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_the_settings_margins_and_distributions_are_read_exactly() {
    let root = temp_dir("sampling-reads");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest_with_sampling().to_string()).unwrap();
    let file = build64(&[("hot", "F64", &[1], &[1.0]), ("all", "F64", &[1], &[1.0])]);
    std::fs::write(root.join("m/p.sampling.safetensors"), file).unwrap();
    let all = discover(&root).unwrap();
    let prompt = &all[0].manifest.prompts[0];
    let entry = all[0].sampling().unwrap();
    assert_eq!(entry.settings.len(), 2);
    let all_setting = entry.settings.iter().find(|s| s.name == "all").unwrap();
    assert_eq!(
        (all_setting.temperature, all_setting.top_k, all_setting.top_p, all_setting.repetition_penalty),
        (Some(0.7), Some(40), Some(0.9), Some(1.3))
    );
    let hot = entry.settings.iter().find(|s| s.name == "hot").unwrap();
    assert_eq!((hot.top_k, hot.top_p, hot.repetition_penalty), (None, None, None));
    let margins = &entry.margins["p"]["all"];
    assert_eq!((margins.top_k_margin_logit, margins.top_p_margin_prob), (Some(0.25), Some(0.125)));
    assert_eq!(entry.margins["p"]["hot"].top_k_margin_logit, None);
    let loaded = all[0].load_sampling(prompt).unwrap();
    assert_eq!(loaded["hot"], vec![1.0]);
    assert_eq!(loaded["all"], vec![1.0]);
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr31_real_sampling_distributions_sum_to_one() {
    let all = discover(&workspace_root().join("fixtures")).unwrap();
    for fixtures in &all {
        let entry = fixtures.sampling().unwrap();
        assert!(entry.settings.len() >= 5, "each control alone and a combination");
        for prompt in &fixtures.manifest.prompts {
            let loaded = fixtures.load_sampling(prompt).unwrap();
            for setting in &entry.settings {
                let dist = &loaded[&setting.name];
                assert_eq!(dist.len(), fixtures.manifest.dims.vocab);
                let total: f64 = dist.iter().sum();
                assert!((total - 1.0).abs() < 1e-12, "{} {}: sums to {total}", prompt.id, setting.name);
                assert!(dist.iter().all(|p| p.is_finite() && *p >= 0.0));
                assert!(entry.margins[&prompt.id].contains_key(&setting.name));
            }
        }
    }
}

/// A `chat.json` as the oracle writes it (Spec §10, FR-39): one conversation of
/// two positions over a vocabulary of three.
fn tiny_chat() -> serde_json::Value {
    serde_json::json!({
        "model_id": "i",
        "dims": {"vocab": 3},
        "reply_tokens": 4,
        "eos_token_id": 2,
        "conversations": [{
            "id": "c",
            "messages": [{"role": "user", "content": "hi"}],
            "rendered": "x",
            "token_ids": [1, 5],
            "prompt_top2_gaps": [0.5, 0.25],
            "reply_ids": [7, 2],
            "reply_text": "a",
            "reply_top2_gaps": [0.5, 0.5],
            "stopped_at_eos": true,
            "tensors_file": "c.f64.safetensors"
        }]
    })
}

fn chat_root(name: &str, chat: &serde_json::Value) -> PathBuf {
    let root = temp_dir(name);
    std::fs::create_dir_all(root.join("i")).unwrap();
    std::fs::write(root.join("i/chat.json"), chat.to_string()).unwrap();
    root
}

#[test]
fn fr39_missing_chat_fixtures_name_the_oracle_command() {
    let root = temp_dir("no-chat");
    let error = discover_chat(&root).unwrap_err();
    assert!(error.contains(CHAT_COMMAND), "{error}");
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), "{}").unwrap();
    let error = discover_chat(&root).unwrap_err();
    assert!(error.contains(CHAT_COMMAND), "a manifest folder is not a chat folder: {error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_the_manifest_discovery_passes_over_the_chat_folder_and_not_the_reverse() {
    let root = chat_root("both-kinds", &tiny_chat());
    std::fs::create_dir_all(root.join("m")).unwrap();
    std::fs::write(root.join("m/manifest.json"), tiny_manifest(true).to_string()).unwrap();
    let manifests = discover(&root).unwrap();
    assert_eq!(manifests.len(), 1);
    assert!(manifests[0].dir.ends_with("m"));
    let chats = discover_chat(&root).unwrap();
    assert_eq!(chats.len(), 1);
    assert!(chats[0].dir.ends_with("i"));
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_a_chat_file_missing_a_field_names_the_command() {
    let mut chat = tiny_chat();
    chat["conversations"][0].as_object_mut().unwrap().remove("rendered");
    let root = chat_root("chat-missing-field", &chat);
    let error = discover_chat(&root).unwrap_err();
    assert!(error.contains("malformed") && error.contains("rendered") && error.contains(CHAT_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_a_missing_logits_file_fails_naming_it_and_the_command() {
    let root = chat_root("chat-missing-file", &tiny_chat());
    let all = discover_chat(&root).unwrap();
    let error = all[0].load_logits(&all[0].chat.conversations[0]).unwrap_err();
    assert!(error.contains("c.f64.safetensors") && error.contains(CHAT_COMMAND), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_logits_of_the_wrong_shape_are_refused() {
    let root = chat_root("chat-wrong-shape", &tiny_chat());
    // Two positions over three tokens are expected; three positions are stored.
    std::fs::write(root.join("i/c.f64.safetensors"), build64(&[("logits", "F64", &[3, 3], &[0.0; 9])])).unwrap();
    let all = discover_chat(&root).unwrap();
    let error = all[0].load_logits(&all[0].chat.conversations[0]).unwrap_err();
    assert!(error.contains("logits") && error.contains("shape") && error.contains("[2, 3]"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_a_file_without_logits_or_in_the_wrong_precision_is_refused() {
    let root = chat_root("chat-bad-file", &tiny_chat());
    std::fs::write(root.join("i/c.f64.safetensors"), build64(&[("other", "F64", &[2, 3], &[0.0; 6])])).unwrap();
    let all = discover_chat(&root).unwrap();
    let error = all[0].load_logits(&all[0].chat.conversations[0]).unwrap_err();
    assert!(error.contains("logits") && error.contains("missing"), "{error}");

    std::fs::write(root.join("i/c.f64.safetensors"), build(&[("logits", "F32", &[2, 3], &[0.0; 6])])).unwrap();
    let error = all[0].load_logits(&all[0].chat.conversations[0]).unwrap_err();
    assert!(error.contains("F32") && error.contains("F64"), "{error}");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_the_conversations_and_logits_are_read_exactly() {
    let root = chat_root("chat-reads", &tiny_chat());
    let values = [0.1f64, 1.0 + 1e-12, -2.5, 3.0e-300, 4.0, 5.5];
    std::fs::write(root.join("i/c.f64.safetensors"), build64(&[("logits", "F64", &[2, 3], &values)])).unwrap();
    let all = discover_chat(&root).unwrap();
    let chat = &all[0].chat;
    assert_eq!((chat.model_id.as_str(), chat.dims.vocab, chat.reply_tokens, chat.eos_token_id), ("i", 3, 4, 2));
    let c = &chat.conversations[0];
    assert_eq!((c.id.as_str(), c.messages[0].role.as_str(), c.messages[0].content.as_str()), ("c", "user", "hi"));
    assert_eq!((c.token_ids.as_slice(), c.reply_ids.as_slice(), c.stopped_at_eos), (&[1, 5][..], &[7, 2][..], true));
    assert_eq!((c.prompt_top2_gaps.as_slice(), c.reply_top2_gaps.as_slice()), (&[0.5, 0.25][..], &[0.5, 0.5][..]));
    let logits = all[0].load_logits(c).unwrap();
    assert_eq!(logits.shape, vec![2, 3]);
    assert_eq!(logits.data, values.to_vec(), "f64 values are kept whole");
    std::fs::remove_dir_all(&root).unwrap();
}

#[test]
fn fr39_real_chat_fixtures_are_consistent() {
    let root = workspace_root().join("fixtures");
    let all = discover_chat(&root).unwrap_or_else(|e| panic!("{e}"));
    assert!(discover(&root).unwrap().iter().all(|f| f.manifest.model_id != all[0].chat.model_id), "the chat model has no manifest");
    for fixtures in &all {
        let chat = &fixtures.chat;
        let ids: Vec<&str> = chat.conversations.iter().map(|c| c.id.as_str()).collect();
        for wanted in ["no-system", "with-system", "multi-turn", "thai"] {
            assert!(ids.contains(&wanted), "conversation {wanted} is missing: {ids:?}");
        }
        for c in &chat.conversations {
            let positions = c.token_ids.len();
            assert!(c.rendered.starts_with("<|im_start|>") && c.rendered.ends_with("<|im_start|>assistant\n"), "{}", c.id);
            assert_eq!(c.prompt_top2_gaps.len(), positions, "{}", c.id);
            assert_eq!(c.reply_top2_gaps.len(), c.reply_ids.len(), "{}", c.id);
            assert_eq!(c.reply_ids.last() == Some(&chat.eos_token_id), c.stopped_at_eos, "{}", c.id);
            assert!(c.reply_ids.len() <= chat.reply_tokens, "{}", c.id);
            let logits = fixtures.load_logits(c).unwrap();
            assert_eq!(logits.shape, vec![positions, chat.dims.vocab], "{}", c.id);
            assert!(logits.data.iter().all(|v| v.is_finite()), "{}", c.id);
        }
    }
}
