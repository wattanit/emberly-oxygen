//! Reads the oracle's stored outputs (FR-4). This reader is separate from the
//! engine's loader on purpose: the reference data must not be read by the
//! code under test. Errors are plain strings that name what to do about them.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

pub const GENERATE_COMMAND: &str = "cd oracle && uv run python generate.py";
pub const SAMPLING_COMMAND: &str = "cd oracle && uv run python sampling_reference.py";
pub const CHAT_COMMAND: &str = "cd oracle && uv run python chat_reference.py";
pub const DOWNLOAD_COMMAND: &str = "cd oracle && uv run python download.py";

/// The files the engine loads from a model directory (Spec §5).
const MODEL_FILES: [&str; 3] = ["config.json", "model.safetensors", "tokenizer.json"];

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
    /// The 64-bit reference's side of the record (Spec §10): the file of its
    /// tensors, its continuation and the top-two gaps S-3 and S-4 use.
    pub f64_tensors_file: String,
    pub f64_prompt_top2_gaps: Vec<f64>,
    pub f64_continuation_ids: Vec<u32>,
    pub f64_continuation_text: String,
    pub f64_continuation_top2_gaps: Vec<f64>,
    pub f64_stopped_at_eos: bool,
    /// The file of this prompt's stored sampling distributions (FR-31); absent
    /// in fixtures made before sampling, so the greedy checks still run on them.
    #[serde(default)]
    pub sampling_file: Option<String>,
}

/// One sampling setting the oracle stored distributions for (Spec §10). A
/// control that is absent takes its neutral value.
#[derive(Debug, Clone, Deserialize)]
pub struct SamplingSetting {
    pub name: String,
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub repetition_penalty: Option<f64>,
}

/// How near a prompt's setting sits to its cutoffs, and how many tokens it keeps.
#[derive(Debug, Clone, Deserialize)]
pub struct SamplingMargins {
    pub top_k_margin_logit: Option<f64>,
    pub top_p_margin_prob: Option<f64>,
    pub support: usize,
}

/// The manifest's `sampling` entry: the settings, and per prompt ID and setting
/// name the margins.
#[derive(Debug, Clone, Deserialize)]
pub struct SamplingEntry {
    pub settings: Vec<SamplingSetting>,
    pub margins: HashMap<String, HashMap<String, SamplingMargins>>,
}

/// One message of a stored conversation (Spec §10).
#[derive(Debug, Clone, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// One stored conversation (FR-39): the messages, the library's rendering of
/// them with the generation prompt, its token ids, the f64 reference's greedy
/// reply, and the top-two gaps that decide a near-tie.
#[derive(Debug, Clone, Deserialize)]
pub struct Conversation {
    pub id: String,
    pub messages: Vec<ChatMessage>,
    pub rendered: String,
    pub token_ids: Vec<u32>,
    pub prompt_top2_gaps: Vec<f64>,
    pub reply_ids: Vec<u32>,
    pub reply_text: String,
    pub reply_top2_gaps: Vec<f64>,
    pub stopped_at_eos: bool,
    pub tensors_file: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatDims {
    pub vocab: usize,
}

/// `chat.json`: the instruct model's chat fixtures, which are not a manifest and
/// are never judged by S-1 to S-4 (Requirements §13).
#[derive(Debug, Deserialize)]
pub struct Chat {
    pub model_id: String,
    pub dims: ChatDims,
    pub reply_tokens: usize,
    pub eos_token_id: u32,
    pub conversations: Vec<Conversation>,
}

/// A fixture folder holding `chat.json`.
#[derive(Debug)]
pub struct ChatFixtures {
    pub dir: PathBuf,
    pub chat: Chat,
}

#[derive(Debug, Deserialize)]
pub struct Manifest {
    pub model_id: String,
    pub dims: Dims,
    pub continuation_tokens: usize,
    pub eos_token_id: u32,
    pub prompts: Vec<Prompt>,
    #[serde(default)]
    pub sampling: Option<SamplingEntry>,
}

/// A fixture folder: where it is and what its manifest says.
#[derive(Debug)]
pub struct Fixtures {
    pub dir: PathBuf,
    pub manifest: Manifest,
}

/// A stored value: the two precisions the oracle writes (Spec §10).
pub trait Element: Copy {
    /// The safetensors dtype name.
    const DTYPE: &'static str;
    const WIDTH: usize;
    fn from_le(bytes: &[u8]) -> Self;
}

impl Element for f32 {
    const DTYPE: &'static str = "F32";
    const WIDTH: usize = 4;
    fn from_le(bytes: &[u8]) -> Self {
        f32::from_le_bytes(bytes.try_into().unwrap())
    }
}

impl Element for f64 {
    const DTYPE: &'static str = "F64";
    const WIDTH: usize = 8;
    fn from_le(bytes: &[u8]) -> Self {
        f64::from_le_bytes(bytes.try_into().unwrap())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorOf<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

pub type Tensor = TensorOf<f32>;
pub type Tensor64 = TensorOf<f64>;

/// The stored tensors of one prompt in one precision, shapes already checked
/// against the manifest.
#[derive(Debug)]
pub struct Tensors<T> {
    pub logits: TensorOf<T>,
    pub embeddings: TensorOf<T>,
    pub layer_outputs: TensorOf<T>,
    pub final_norm_output: TensorOf<T>,
}

/// The 32-bit run's tensors.
pub type PromptTensors = Tensors<f32>;
/// The 64-bit run's tensors: the reference (Requirements FR-2).
pub type ReferenceTensors = Tensors<f64>;

pub fn workspace_root() -> PathBuf {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    root.canonicalize().unwrap_or(root)
}

/// Checks the model directory holds the published files; the harness never
/// downloads (FR-7), so a missing one fails naming the command that fetches it.
pub fn require_model_dir(dir: &Path) -> Result<(), String> {
    for file in MODEL_FILES {
        if !dir.join(file).is_file() {
            return Err(format!(
                "{} is missing; download the model with: {DOWNLOAD_COMMAND}",
                dir.join(file).display()
            ));
        }
    }
    Ok(())
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
                .map_err(|e| {
                    format!(
                        "{}: malformed manifest: {e}; fixtures made before the f64 reference lack its fields, regenerate with: {GENERATE_COMMAND}",
                        path.display()
                    )
                })?;
            Ok(Fixtures { dir, manifest })
        })
        .collect()
}

/// Finds every `<root>/*/chat.json` (FR-39). A folder with a manifest is not
/// looked at here, and a folder with only `chat.json` is not looked at by
/// `discover`, so no model is judged by a check meant for the other.
pub fn discover_chat(fixtures_root: &Path) -> Result<Vec<ChatFixtures>, String> {
    let mut dirs: Vec<PathBuf> = fs::read_dir(fixtures_root)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.join("chat.json").is_file())
        .collect();
    if dirs.is_empty() {
        return Err(format!(
            "no chat fixtures found in {}; generate them with: {CHAT_COMMAND}",
            fixtures_root.display()
        ));
    }
    dirs.sort();
    dirs.into_iter()
        .map(|dir| {
            let path = dir.join("chat.json");
            let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
            let chat = serde_json::from_str(&text).map_err(|e| {
                format!("{}: malformed chat file: {e}; regenerate with: {CHAT_COMMAND}", path.display())
            })?;
            Ok(ChatFixtures { dir, chat })
        })
        .collect()
}

impl ChatFixtures {
    /// Loads one conversation's f64 logits, `[positions, vocab]`.
    pub fn load_logits(&self, conversation: &Conversation) -> Result<Tensor64, String> {
        let path = self.dir.join(&conversation.tensors_file);
        if !path.is_file() {
            return Err(format!(
                "{}: file named by chat.json is missing; regenerate with: {CHAT_COMMAND}",
                path.display()
            ));
        }
        let mut tensors =
            read_typed::<f64>(&path).map_err(|e| format!("{e}; regenerate with: {CHAT_COMMAND}"))?;
        let logits = tensors
            .remove("logits")
            .ok_or_else(|| format!("{}: tensor logits is missing; regenerate with: {CHAT_COMMAND}", path.display()))?;
        let shape = [conversation.token_ids.len(), self.chat.dims.vocab];
        if logits.shape != shape {
            return Err(format!(
                "{}: tensor logits has shape {:?}, chat.json implies {shape:?}; regenerate with: {CHAT_COMMAND}",
                path.display(),
                logits.shape
            ));
        }
        Ok(logits)
    }
}

impl Fixtures {
    /// Loads one prompt's tensors. Only one prompt is held at a time: the long
    /// ones are about 135 MB each.
    pub fn load_tensors(&self, prompt: &Prompt) -> Result<PromptTensors, String> {
        self.load(prompt, &prompt.tensors_file)
    }

    /// Loads one prompt's 64-bit reference tensors, the ones S-1 to S-4 judge by.
    pub fn load_reference(&self, prompt: &Prompt) -> Result<ReferenceTensors, String> {
        self.load(prompt, &prompt.f64_tensors_file)
    }

    /// The manifest's sampling entry. Fixtures made before the sampling reference
    /// have none, which fails here, naming the command, and not in `discover`.
    pub fn sampling(&self) -> Result<&SamplingEntry, String> {
        self.manifest.sampling.as_ref().ok_or_else(|| {
            format!(
                "{}: the manifest has no sampling entry; generate the sampling reference with: {SAMPLING_COMMAND}",
                self.dir.join("manifest.json").display()
            )
        })
    }

    /// Loads one prompt's stored adjusted distributions, one per setting, each
    /// the length of the vocabulary (FR-31).
    pub fn load_sampling(&self, prompt: &Prompt) -> Result<HashMap<String, Vec<f64>>, String> {
        let entry = self.sampling()?;
        let file = prompt.sampling_file.as_deref().ok_or_else(|| {
            format!("{}: prompt {} names no sampling file; generate it with: {SAMPLING_COMMAND}", self.dir.display(), prompt.id)
        })?;
        let path = self.dir.join(file);
        if !path.is_file() {
            return Err(format!(
                "{}: file named by the manifest is missing; generate it with: {SAMPLING_COMMAND}",
                path.display()
            ));
        }
        let mut tensors = read_typed::<f64>(&path).map_err(|e| format!("{e}; generate it with: {SAMPLING_COMMAND}"))?;
        let mut out = HashMap::new();
        for setting in &entry.settings {
            let t = tensors.remove(&setting.name).ok_or_else(|| {
                format!("{}: the distribution for setting {} is missing", path.display(), setting.name)
            })?;
            if t.shape != [self.manifest.dims.vocab] {
                return Err(format!(
                    "{}: distribution {} has shape {:?}, the vocabulary implies [{}]",
                    path.display(),
                    setting.name,
                    t.shape,
                    self.manifest.dims.vocab
                ));
            }
            out.insert(setting.name.clone(), t.data);
        }
        Ok(out)
    }

    fn load<T: Element>(&self, prompt: &Prompt, file: &str) -> Result<Tensors<T>, String> {
        let path = self.dir.join(file);
        if !path.is_file() {
            return Err(format!(
                "{}: file named by the manifest is missing; regenerate with: {GENERATE_COMMAND}",
                path.display()
            ));
        }
        let mut tensors = read_typed::<T>(&path)
            .map_err(|e| format!("{e}; regenerate with: {GENERATE_COMMAND}"))?;
        let positions = prompt.token_ids.len();
        let Dims { layers, hidden, vocab } = self.manifest.dims;
        let mut take = |name: &str, shape: &[usize]| -> Result<TensorOf<T>, String> {
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
        Ok(Tensors {
            logits: take("logits", &[positions, vocab])?,
            embeddings: take("embeddings", &[positions, hidden])?,
            layer_outputs: take("layer_outputs", &[layers, positions, hidden])?,
            final_norm_output: take("final_norm_output", &[positions, hidden])?,
        })
    }
}

/// Reads every tensor of an F32 safetensors file: an 8-byte little-endian
/// header length, a JSON header, then raw data.
pub fn read_safetensors(path: &Path) -> Result<HashMap<String, Tensor>, String> {
    read_typed(path)
}

fn read_typed<T: Element>(path: &Path) -> Result<HashMap<String, TensorOf<T>>, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    parse_typed(&bytes).map_err(|e| format!("{}: {e}", path.display()))
}

/// Parses an in-memory safetensors file whose tensors are all F32.
pub fn parse_safetensors(bytes: &[u8]) -> Result<HashMap<String, Tensor>, String> {
    parse_typed(bytes)
}

/// Parses an in-memory safetensors file whose tensors are all F64, keeping
/// every value whole (the reference is never narrowed).
pub fn parse_safetensors_f64(bytes: &[u8]) -> Result<HashMap<String, Tensor64>, String> {
    parse_typed(bytes)
}

fn parse_typed<T: Element>(bytes: &[u8]) -> Result<HashMap<String, TensorOf<T>>, String> {
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
        if dtype != T::DTYPE {
            return Err(format!("tensor {name}: dtype {dtype} is not supported here, only {}", T::DTYPE));
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
        if raw.len() != count * T::WIDTH {
            return Err(format!("tensor {name}: {} bytes for {count} {} elements", raw.len(), T::DTYPE));
        }
        let data = raw.chunks_exact(T::WIDTH).map(T::from_le).collect();
        tensors.insert(name.clone(), TensorOf { shape, data });
    }
    Ok(tensors)
}
