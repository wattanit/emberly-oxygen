//! Tests of the fixture reader. They verify FR-4 (stored fixtures are read as
//! they were written) and FR-7 (a missing input fails naming the command that
//! creates it).

mod common;

use common::fixtures::{discover, parse_safetensors, workspace_root, GENERATE_COMMAND};

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

#[test]
fn fr7_missing_fixtures_name_the_oracle_command() {
    let empty = std::env::temp_dir().join(format!("oxygen-empty-{}", std::process::id()));
    std::fs::create_dir_all(&empty).unwrap();
    let error = discover(&empty).unwrap_err();
    assert!(error.contains(GENERATE_COMMAND), "{error}");

    let absent = empty.join("nope");
    assert!(discover(&absent).unwrap_err().contains(GENERATE_COMMAND));
    std::fs::remove_dir_all(&empty).unwrap();
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
