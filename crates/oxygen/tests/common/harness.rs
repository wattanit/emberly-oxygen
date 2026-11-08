//! Runs the engine over the fixtures and collects one result per prompt and
//! check (FR-5). Nothing is skipped: a check the engine cannot perform yet is
//! run anyway and fails, so the gate is seen to fail.

use emberly_oxygen::{generate, Capture, Model, Tokenizer};

use super::compare::{
    greedy_check, intermediate_error, logit_error, top1_check, Numeric, Reference, Verdict,
};
use super::fixtures::{Fixtures, Prompt, PromptTensors, ReferenceTensors, Tensors};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Check {
    /// FR-13: the prompt encodes to the reference token IDs.
    Encode,
    /// FR-14: the reference continuation IDs decode to the reference text.
    Decode,
    /// FR-15: streaming the reference IDs one token at a time gives the same
    /// text as decoding them at once, and a `utf8-split` prompt really does
    /// make the decoder hold a character back.
    Streaming,
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
    pub const ALL: [Check; 7] = [
        Check::Encode,
        Check::Decode,
        Check::Streaming,
        Check::Logits,
        Check::Top1,
        Check::Intermediates,
        Check::Greedy,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Check::Encode => "encode",
            Check::Decode => "decode",
            Check::Streaming => "streaming",
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
    /// Worst error against the 64-bit reference, for the checks that measure one
    /// (S-1, S-2). This is the figure the verdict is judged by.
    pub max_error: Option<f64>,
    /// The same error against the 32-bit run, reported beside it and never
    /// judged (FR-5): no verdict reads this field.
    pub f32_error: Option<f64>,
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
    Numeric { max_error: f64::INFINITY, verdict: Verdict::Fail { position: None, message: message.into() } }
}

/// S-2 over every stored intermediate, in order: embeddings, layers 0 to N-1,
/// then the final-norm output. The first tensor over the threshold is the
/// verdict, so a failure names the first diverging layer (FR-6).
pub fn check_intermediates<R: Reference>(run: &EngineRun, reference: &Tensors<R>, layers: usize, hidden: usize) -> Numeric {
    if run.layers.len() != layers {
        return fail(format!("engine captured {} layers, reference has {layers}", run.layers.len()));
    }
    let mut max_error = 0f64;
    let mut first_bad: Option<Verdict> = None;
    let mut tensors: Vec<(String, &[f32], &[R])> = vec![("embeddings".into(), &run.embeddings, &reference.embeddings.data)];
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

fn encode_check(tokenizer: &Tokenizer, prompt: &Prompt) -> Verdict {
    match tokenizer.encode(&prompt.text) {
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

fn decode_check(tokenizer: &Tokenizer, prompt: &Prompt) -> Verdict {
    match tokenizer.decode(&prompt.continuation_ids) {
        Err(e) => Verdict::Fail { position: None, message: e.to_string() },
        Ok(text) if text == prompt.continuation_text => Verdict::Pass,
        Ok(text) => Verdict::Fail {
            position: None,
            message: format!("decoded {text:?}, reference {:?}", prompt.continuation_text),
        },
    }
}

/// FR-15: streams the prompt's reference IDs and its reference continuation
/// through the decoder one token at a time. Each must give back the reference
/// text, and for a `utf8-split` prompt the prompt's tokens must make the
/// decoder hold bytes back at least once, or the prompt tests nothing.
pub fn streaming_check(tokenizer: &Tokenizer, prompt: &Prompt) -> Verdict {
    let fail = |position, message: String| Verdict::Fail { position: Some(position), message };
    let mut held_back = false;
    for (what, ids, expected) in [
        ("prompt", &prompt.token_ids, &prompt.text),
        ("continuation", &prompt.continuation_ids, &prompt.continuation_text),
    ] {
        let mut decoder = tokenizer.stream_decoder();
        let mut text = String::new();
        for (step, &id) in ids.iter().enumerate() {
            let chunk = match decoder.push(id) {
                Ok(chunk) => chunk,
                Err(e) => return fail(step, format!("{what}: {e}")),
            };
            // A token with bytes of its own that emitted nothing left a character unfinished.
            if what == "prompt" && chunk.is_empty() && tokenizer.decode(&[id]).is_ok_and(|t| !t.is_empty()) {
                held_back = true;
            }
            text.push_str(&chunk);
            if !expected.starts_with(&text) {
                return fail(step, format!("{what}: streamed {text:?}, which is not a prefix of {expected:?}"));
            }
        }
        text.push_str(&decoder.finish());
        if text != *expected {
            return fail(ids.len(), format!("{what}: streamed {text:?}, reference {expected:?}"));
        }
    }
    if prompt.category == "utf8-split" && !held_back {
        return fail(0, "no token of this prompt split a character, so it does not test streaming".into());
    }
    Verdict::Pass
}

/// S-4 against the 64-bit reference's continuation and gaps.
fn greedy(tokenizer: &Tokenizer, model: &Model, prompt: &Prompt) -> Verdict {
    let run = generate(model, tokenizer, &prompt.text, prompt.f64_continuation_ids.len(), &mut |_| {});
    match run {
        Err(e) => Verdict::Fail { position: None, message: e.to_string() },
        Ok(run) => greedy_check(&run.ids, &prompt.f64_continuation_ids, &prompt.f64_continuation_top2_gaps),
    }
}

/// Runs every check for one prompt. S-1 to S-4 are judged against `reference`,
/// the 64-bit run; `f32_reference`, the 32-bit run, only supplies the distance
/// reported beside the errors (FR-5).
pub fn check_prompt(
    fixtures: &Fixtures,
    prompt: &Prompt,
    reference: &ReferenceTensors,
    f32_reference: &PromptTensors,
    tokenizer: &Tokenizer,
    model: &Model,
) -> Vec<CheckResult> {
    let dims = &fixtures.manifest.dims;
    let result = |check, verdict, max_error, f32_error| CheckResult {
        model_id: fixtures.manifest.model_id.clone(),
        prompt: prompt.id.clone(),
        category: prompt.category.clone(),
        check,
        verdict,
        max_error,
        f32_error,
    };
    // The forward pass is fed the reference token IDs, so a tokenizer failure
    // does not hide the model's results.
    let run = run_engine(model, &prompt.token_ids);
    let (logits, top1, inter, f32_logits, f32_inter) = match &run {
        Ok(run) => (
            logit_error(&run.logits, &reference.logits.data, dims.vocab),
            top1_check(&run.logits, &reference.logits.data, dims.vocab, &prompt.f64_prompt_top2_gaps),
            check_intermediates(run, reference, dims.layers, dims.hidden),
            // Reported, not judged: only the measured error is read from these two.
            logit_error(&run.logits, &f32_reference.logits.data, dims.vocab).max_error,
            check_intermediates(run, f32_reference, dims.layers, dims.hidden).max_error,
        ),
        Err(e) => (
            fail(e.clone()),
            Verdict::Fail { position: None, message: e.clone() },
            fail(e.clone()),
            f64::INFINITY,
            f64::INFINITY,
        ),
    };
    vec![
        result(Check::Encode, encode_check(tokenizer, prompt), None, None),
        result(Check::Decode, decode_check(tokenizer, prompt), None, None),
        result(Check::Streaming, streaming_check(tokenizer, prompt), None, None),
        result(Check::Logits, logits.verdict, Some(logits.max_error), Some(f32_logits)),
        result(Check::Top1, top1, None, None),
        result(Check::Intermediates, inter.verdict, Some(inter.max_error), Some(f32_inter)),
        result(Check::Greedy, greedy(tokenizer, model, prompt), None, None),
    ]
}
