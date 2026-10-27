//! Reads the oracle's stored outputs (FR-4). This reader is separate from the
//! engine's loader on purpose: the reference data must not be read by the
//! code under test. Errors are plain strings that name what to do about them.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const GENERATE_COMMAND: &str = "cd oracle && uv run python generate.py";

#[derive(Debug, Deserialize)]
pub struct Dims {
    pub layers: usize,
    pub hidden: usize,
    pub vocab: usize,
}

#[derive(Debug, Deserialize)]
pub struct Prompt {
    pub id: String,
    pub category: String,
    pub text: String,
    pub token_ids: Vec<u32>,
    pub prompt_top2_gaps: Vec<f32>,
    pub continuation_ids: Vec<u32>,
    pub continuation_text: String,
    pub continuation_top2_gaps: Vec<f32>,
    pub stopped_at_eos: bool,
    pub tensors_file: String,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub dims: Dims,
    pub continuation_tokens: usize,
    pub eos_token_id: u32,
    pub prompts: Vec<Prompt>,
}

/// A fixture folder: where it is and what its manifest says.
#[derive(Debug)]
pub struct Fixtures {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

/// The stored tensors of one prompt, shapes already checked against the manifest.
#[derive(Debug)]
pub struct PromptTensors {
    pub logits: Tensor,
    pub embeddings: Tensor,
    pub layer_outputs: Tensor,
    pub final_norm_output: Tensor,
}

pub fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Finds every `<root>/*/manifest.json`, so a new model's folder needs no code
/// change (Requirements §2.2).
pub fn discover(fixtures_root: &Path) -> Result<Vec<Fixtures>, String> {
    let missing = || {
        format!(
            "no fixtures found in {}; generate them with: {GENERATE_COMMAND}",
            fixtures_root.display()
        )
    };
    let mut dirs: Vec<PathBuf> = fs::read_dir(fixtures_root)
        .map_err(|_| missing())?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    if dirs.is_empty() {
        return Err(missing());
    }
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let path = dir.join("manifest.json");
            let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let manifest = serde_json::from_str(&text)
                .map_err(|e| format!("{}: malformed manifest: {e}", path.display()))?;
            Ok(Fixtures { dir, manifest })
        })
        .collect()
}

impl Fixtures {
    /// Loads one prompt's tensors. Only one prompt is held at a time: the long
    /// ones are about 135 MB each.
    pub fn load_tensors(&self, prompt: &Prompt) -> Result<PromptTensors, String> {
        let path = self.dir.join(&prompt.tensors_file);
        if !path.is_file() {
            return Err(format!(
                "{}: file named by the manifest is missing; regenerate with: {GENERATE_COMMAND}",
                path.display()
            ));
        }
        let mut tensors = read_safetensors(&path)?;
        let positions = prompt.token_ids.len();
        let Dims { layers, hidden, vocab } = self.manifest.dims;
        let mut take = |name: &str, shape: &[usize]| -> Result<Tensor, String> {
            let t = tensors
                .remove(name)
                .ok_or_else(|| format!("{}: tensor {name} is missing", path.display()))?;
            if t.shape != shape {
                return Err(format!(
                    "{}: tensor {name} has shape {:?}, manifest implies {shape:?}",
                    path.display(),
                    t.shape
                ));
            }
            Ok(t)
        };
        Ok(PromptTensors {
            logits: take("logits", &[positions, vocab])?,
            embeddings: take("embeddings", &[positions, hidden])?,
            layer_outputs: take("layer_outputs", &[layers, positions, hidden])?,
            final_norm_output: take("final_norm_output", &[positions, hidden])?,
        })
    }
}

/// Reads every tensor of a safetensors file: an 8-byte little-endian header
/// length, a JSON header, then raw data. Only F32 is accepted.
pub fn read_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_safetensors(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

pub fn parse_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    let len_bytes: [u8; 8] = bytes.get(..8).ok_or("file shorter than the header length")?.try_into().unwrap();
    let header_len = usize::try_from(u64::from_le_bytes(len_bytes)).map_err(|_| "header length overflows")?;
    let header_end = 8usize.checked_add(header_len).ok_or("header length overflows")?;
    let header = bytes.get(8..header_end).ok_or("header extends past the end of the file")?;
    let data = &bytes[header_end..];
    let header: serde_json::Value =
        serde_json::from_slice(header).map_err(|e| format!("malformed header: {e}"))?;
    let entries = header.as_object().ok_or("header is not a JSON object")?;

    let mut tensors = HashMap::new();
    for (name, info) in entries {
        if name == "__metadata__" {
            continue;
        }
        let dtype = info["dtype"].as_str().ok_or_else(|| format!("tensor {name}: no dtype"))?;
        if dtype != "F32" {
            return Err(format!("tensor {name}: dtype {dtype} is not supported, only F32"));
        }
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .ok_or_else(|| format!("tensor {name}: no shape"))?
            .iter()
            .map(|d| d.as_u64().map(|d| d as usize))
            .collect::<Option<_>>()
            .ok_or_else(|| format!("tensor {name}: bad shape"))?;
        let offsets = info["data_offsets"].as_array().filter(|o| o.len() == 2);
        let (begin, end) = match offsets.map(|o| (o[0].as_u64(), o[1].as_u64())) {
            Some((Some(b), Some(e))) => (b as usize, e as usize),
            _ => return Err(format!("tensor {name}: bad data_offsets")),
        };
        let raw = data
            .get(begin..end)
            .ok_or_else(|| format!("tensor {name}: data range {begin}..{end} is outside the file"))?;
        let count: usize = shape.iter().product();
        if raw.len() != count * 4 {
            return Err(format!("tensor {name}: {} bytes for {count} F32 elements", raw.len()));
        }
        let data = raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        tensors.insert(name.clone(), Tensor { shape, data });
    }
    Ok(tensors)
}
