//! Runs the engine over the fixtures and collects one result per prompt and
//! check (FR-5). Nothing is skipped: a check the engine cannot perform yet is
//! run anyway and fails, so the gate is seen to fail.

use std::path::Path;

use emberly_oxygen::{generate, Capture, Model, Tokenizer};

use super::compare::{
    greedy_check, intermediate_error, logit_error, top1_check, Numeric, Verdict,
};
use super::fixtures::{Fixtures, Prompt, PromptTensors};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Check {
    /// FR-13: the prompt encodes to the reference token IDs.
    Encode,
    /// FR-14: the reference continuation IDs decode to the reference text.
    Decode,
    /// S-1: logits at every position.
    Logits,
    /// S-3: top token at every position.
    Top1,
    /// S-2: embeddings, every layer's output and the final-norm output.
    Intermediates,
    /// S-4: greedy continuation.
    Greedy,
}

impl Check {
    pub const ALL: [Check; 6] =
        [Check::Encode, Check::Decode, Check::Logits, Check::Top1, Check::Intermediates, Check::Greedy];

    pub fn name(self) -> &'static str {
        match self {
            Check::Encode => "encode",
            Check::Decode => "decode",
            Check::Logits => "logits",
            Check::Top1 => "top-1",
            Check::Intermediates => "intermediates",
            Check::Greedy => "greedy",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub model_id: String,
    pub prompt: String,
    pub category: String,
    pub check: Check,
    pub verdict: Verdict,
    /// Worst error measured, for the checks that measure one.
    pub max_error: Option<f32>,
}

/// What the engine produced for one prompt, fed one token at a time.
pub struct EngineRun {
    /// `[positions, vocab]`, flattened.
    pub logits: Vec<f32>,
    pub embeddings: Vec<f32>,
    /// Per layer, `[positions, hidden]` flattened.
    pub layers: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
}

/// The harness owns the token loop, so later phases use the same path.
pub fn run_engine(model: &Model, ids: &[u32]) -> Result<EngineRun, String> {
    let mut session = model.session(true);
    let mut run = EngineRun { logits: vec![], embeddings: vec![], layers: vec![], final_norm: vec![] };
    for (position, &token) in ids.iter().enumerate() {
        let logits = session.step(token).map_err(|e| format!("position {position}: {e}"))?;
        run.logits.extend(logits);
        let Capture { embedding, layers, final_norm } =
            session.capture().ok_or("capture is off")?.clone();
        run.embeddings.extend(embedding);
        if run.layers.len() < layers.len() {
            run.layers.resize(layers.len(), vec![]);
        }
        for (all, one) in run.layers.iter_mut().zip(layers) {
            all.extend(one);
        }
        run.final_norm.extend(final_norm);
    }
    Ok(run)
}

fn fail(message: impl Into<String>) -> Numeric {
    Numeric { max_error: f32::INFINITY, verdict: Verdict::Fail { position: None, message: message.into() } }
}

/// S-2 over every stored intermediate, in order: embeddings, layers 0 to N-1,
/// then the final-norm output. The first tensor over the threshold is the
/// verdict, so a failure names the first diverging layer (FR-6).
pub fn check_intermediates(run: &EngineRun, reference: &PromptTensors, layers: usize, hidden: usize) -> Numeric {
    if run.layers.len() != layers {
        return fail(format!("engine captured {} layers, reference has {layers}", run.layers.len()));
    }
    let mut max_error = 0f32;
    let mut first_bad: Option<Verdict> = None;
    let mut tensors: Vec<(String, &[f32], &[f32])> = vec![("embeddings".into(), &run.embeddings, &reference.embeddings.data)];
    let per_layer = reference.layer_outputs.data.len() / layers.max(1);
    for (i, engine) in run.layers.iter().enumerate() {
        let reference = &reference.layer_outputs.data[i * per_layer..(i + 1) * per_layer];
        tensors.push((format!("layer {i}"), engine, reference));
    }
    tensors.push(("final norm".into(), &run.final_norm, &reference.final_norm_output.data));
    for (name, engine, reference) in tensors {
        let n = intermediate_error(engine, reference, hidden);
        max_error = max_error.max(n.max_error);
        if let (Verdict::Fail { position, message }, None) = (&n.verdict, &first_bad) {
            first_bad = Some(Verdict::Fail { position: *position, message: format!("{name}: {message}") });
        }
    }
    Numeric { max_error, verdict: first_bad.unwrap_or(Verdict::Pass) }
}

fn encode_check(model_dir: &Path, prompt: &Prompt) -> Verdict {
    let encoded = Tokenizer::load(model_dir).and_then(|t| t.encode(&prompt.text));
    match encoded {
        Err(e) => Verdict::Fail { position: None, message: e.to_string() },
        Ok(ids) if ids == prompt.token_ids => Verdict::Pass,
        Ok(ids) => {
            let first = ids.iter().zip(&prompt.token_ids).position(|(a, b)| a != b);
            Verdict::Fail {
                position: first.or(Some(ids.len().min(prompt.token_ids.len()))),
                message: format!("encoded {} tokens, reference has {}", ids.len(), prompt.token_ids.len()),
            }
        }
    }
}

fn decode_check(model_dir: &Path, prompt: &Prompt) -> Verdict {
    match Tokenizer::load(model_dir).and_then(|t| t.decode(&prompt.continuation_ids)) {
        Err(e) => Verdict::Fail { position: None, message: e.to_string() },
        Ok(text) if text == prompt.continuation_text => Verdict::Pass,
        Ok(text) => Verdict::Fail {
            position: None,
            message: format!("decoded {text:?}, reference {:?}", prompt.continuation_text),
        },
    }
}

fn greedy(model_dir: &Path, model: &Model, prompt: &Prompt) -> Verdict {
    let run = Tokenizer::load(model_dir)
        .and_then(|t| generate(model, &t, &prompt.text, prompt.continuation_ids.len(), &mut |_| {}));
    match run {
        Err(e) => Verdict::Fail { position: None, message: e.to_string() },
        Ok(ids) => greedy_check(&ids, &prompt.continuation_ids, &prompt.continuation_top2_gaps),
    }
}

/// Runs every check for one prompt.
pub fn check_prompt(
    fixtures: &Fixtures,
    prompt: &Prompt,
    reference: &PromptTensors,
    model_dir: &Path,
    model: &Model,
) -> Vec<CheckResult> {
    let dims = &fixtures.manifest.dims;
    let result = |check, verdict, max_error| CheckResult {
        model_id: fixtures.manifest.model_id.clone(),
        prompt: prompt.id.clone(),
        category: prompt.category.clone(),
        check,
        verdict,
        max_error,
    };
    // The forward pass is fed the reference token IDs, so a tokenizer failure
    // does not hide the model's results.
    let run = run_engine(model, &prompt.token_ids);
    let (logits, top1, inter) = match &run {
        Ok(run) => (
            logit_error(&run.logits, &reference.logits.data, dims.vocab),
            top1_check(&run.logits, &reference.logits.data, dims.vocab, &prompt.prompt_top2_gaps),
            check_intermediates(run, reference, dims.layers, dims.hidden),
        ),
        Err(e) => (fail(e.clone()), Verdict::Fail { position: None, message: e.clone() }, fail(e.clone())),
    };
    vec![
        result(Check::Encode, encode_check(model_dir, prompt), None),
        result(Check::Decode, decode_check(model_dir, prompt), None),
        result(Check::Logits, logits.verdict, Some(logits.max_error)),
        result(Check::Top1, top1, None),
        result(Check::Intermediates, inter.verdict, Some(inter.max_error)),
        result(Check::Greedy, greedy(model_dir, model, prompt), None),
    ]
}
