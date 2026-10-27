//! Tests of the fixture reader. They verify FR-4 (stored fixtures are read as
//! they were written) and FR-7 (a missing input fails naming the command that
//! creates it).

mod common;

use std::path::PathBuf;

use common::fixtures::{
    discover, parse_safetensors, require_model_dir, workspace_root, DOWNLOAD_COMMAND, GENERATE_COMMAND,
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

#[test]
fn fr7_a_manifest_naming_a_missing_tensors_file_fails_naming_it() {
    let root = temp_dir("missing-tensors");
    std::fs::create_dir_all(root.join("m")).unwrap();
    let manifest = serde_json::json!({
        "model_id": "m",
        "dims": {"layers": 1, "hidden": 1, "vocab": 1},
        "continuation_tokens": 1,
        "eos_token_id": 0,
        "prompts": [{
            "id": "p", "category": "english", "text": "x", "token_ids": [1],
            "prompt_top2_gaps": [1.0], "continuation_ids": [1], "continuation_text": "x",
            "continuation_top2_gaps": [1.0], "stopped_at_eos": false,
            "tensors_file": "p.safetensors"
        }]
    });
    std::fs::write(root.join("m/manifest.json"), manifest.to_string()).unwrap();
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
    }
}
