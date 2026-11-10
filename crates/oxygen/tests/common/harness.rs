//! Runs the engine over the fixtures and collects one result per prompt and
//! check (FR-5). Nothing is skipped: a check the engine cannot perform yet is
//! run anyway and fails, so the gate is seen to fail.

use std::collections::HashMap;

use emberly_oxygen::{generate, Capture, Error, Logits, Message, Model, Role, Sampling, Tokenizer};

use super::compare::{
    greedy_check, intermediate_error, logit_error, top1_check, Numeric, Reference, Verdict,
};
use super::fixtures::{ChatFixtures, Conversation, Fixtures, Prompt, PromptTensors, ReferenceTensors, Tensors};

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

/// What the engine produced for one prompt.
pub struct EngineRun {
    /// `[positions, vocab]`, flattened.
    pub logits: Vec<f32>,
    pub embeddings: Vec<f32>,
    /// Per layer, `[positions, hidden]` flattened.
    pub layers: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
}

/// The engine's values for one prompt from the batched pass, the one generation
/// uses (FR-41, Spec §11): every position's logits and the capture of the whole
/// prompt, in one `prefill`. The harness owns this call, so every check reaches
/// the engine the same way.
pub fn run_engine(model: &Model, ids: &[u32]) -> Result<EngineRun, String> {
    let mut session = model.session(true);
    let logits = session.prefill(ids, Logits::All).map_err(|e| e.to_string())?;
    let Capture { embedding, layers, final_norm } = session.capture().ok_or("capture is off")?.clone();
    Ok(EngineRun { logits, embeddings: embedding, layers, final_norm })
}

/// The same values from the one-token path, `step` once per token: the 0.3.0 way,
/// kept as the other side of the batched-against-one-token comparison (FR-41,
/// Spec §11).
#[allow(dead_code)]
pub fn run_engine_by_steps(model: &Model, ids: &[u32]) -> Result<EngineRun, String> {
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

/// How far apart two runs of the engine are (FR-41, Spec §11).
#[allow(dead_code)]
pub struct Difference {
    /// The largest absolute difference between any two logits.
    pub logits: f64,
    /// The largest absolute difference between any two values of the embeddings, the layers or the final norm.
    pub intermediates: f64,
}

/// The two runs' largest absolute differences, measured in f64 so the subtraction adds no rounding.
/// Runs of different shapes are infinitely far apart: a pass that returned too few values must not
/// pass by having its prefix compared.
#[allow(dead_code)]
pub fn engine_difference(a: &EngineRun, b: &EngineRun) -> Difference {
    fn largest(x: &[f32], y: &[f32]) -> f64 {
        if x.len() != y.len() {
            return f64::INFINITY;
        }
        x.iter().zip(y).map(|(p, q)| (f64::from(*p) - f64::from(*q)).abs()).fold(0.0, f64::max)
    }
    let mut intermediates = largest(&a.embeddings, &b.embeddings).max(largest(&a.final_norm, &b.final_norm));
    if a.layers.len() != b.layers.len() {
        intermediates = f64::INFINITY;
    }
    for (x, y) in a.layers.iter().zip(&b.layers) {
        intermediates = intermediates.max(largest(x, y));
    }
    Difference { logits: largest(&a.logits, &b.logits), intermediates }
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
    // The forward pass is fed the reference token IDs, so a tokenizer failure
    // does not hide the model's results.
    let run = run_engine(model, &prompt.token_ids);
    check_prompt_with_run(fixtures, prompt, reference, f32_reference, tokenizer, model, &run)
}

/// [`check_prompt`] on an engine run already made, so the sampling check (S-9)
/// can read the same run's logits instead of running the prompt twice.
pub fn check_prompt_with_run(
    fixtures: &Fixtures,
    prompt: &Prompt,
    reference: &ReferenceTensors,
    f32_reference: &PromptTensors,
    tokenizer: &Tokenizer,
    model: &Model,
    run: &Result<EngineRun, String>,
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
    let (logits, top1, inter, f32_logits, f32_inter) = match run {
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

/// One setting's S-9 result for one prompt (FR-5, S-9): the largest probability
/// difference from the stored distribution, and how near the setting sits to its
/// cutoffs, so a cutoff the engine's rounding could cross is visible.
#[derive(Debug, Clone)]
pub struct SamplingResult {
    pub model_id: String,
    pub prompt: String,
    pub category: String,
    pub setting: String,
    pub max_error: f64,
    pub verdict: Verdict,
    pub top_k_margin: Option<f64>,
    pub top_p_margin: Option<f64>,
}

/// How the engine's adjusted distribution is computed: the library's
/// `distribution`, or a faulted one for the negative controls.
pub type DistributionFn = dyn Fn(&[f32], &[u32], &Sampling) -> Result<Vec<f64>, Error>;

/// S-9 for one prompt: for every stored setting, the engine's adjusted
/// distribution at the prompt's last position (computed from the engine's own
/// logits, with the prompt's token IDs as the repetition-penalty context)
/// against the stored one. A missing stored distribution or an engine run
/// without logits is an error, not a pass.
pub fn check_sampling(
    fixtures: &Fixtures,
    prompt: &Prompt,
    stored: &HashMap<String, Vec<f64>>,
    run: &EngineRun,
    distribution: &DistributionFn,
) -> Result<Vec<SamplingResult>, String> {
    let entry = fixtures.sampling()?;
    let vocab = fixtures.manifest.dims.vocab;
    if run.logits.len() < vocab || run.logits.len() % vocab != 0 {
        return Err(format!("{}: the engine produced {} logits, not whole positions of {vocab}", prompt.id, run.logits.len()));
    }
    let last = &run.logits[run.logits.len() - vocab..];
    let margins = entry.margins.get(&prompt.id);
    let mut results = Vec::new();
    for setting in &entry.settings {
        let reference = stored
            .get(&setting.name)
            .ok_or_else(|| format!("{}: no stored distribution for setting {}", prompt.id, setting.name))?;
        let sampling = Sampling {
            temperature: setting.temperature.unwrap_or(1.0) as f32,
            top_k: setting.top_k.unwrap_or(0),
            top_p: setting.top_p.unwrap_or(1.0) as f32,
            repetition_penalty: setting.repetition_penalty.unwrap_or(1.0) as f32,
            seed: None,
        };
        let numeric = match distribution(last, &prompt.token_ids, &sampling) {
            Ok(engine) => super::compare::distribution_error(&engine, reference),
            Err(e) => fail(e.to_string()),
        };
        let verdict = match numeric.verdict {
            Verdict::Fail { position, message } => {
                Verdict::Fail { position, message: format!("setting {}: {message}", setting.name) }
            }
            other => other,
        };
        let m = margins.and_then(|m| m.get(&setting.name));
        results.push(SamplingResult {
            model_id: fixtures.manifest.model_id.clone(),
            prompt: prompt.id.clone(),
            category: prompt.category.clone(),
            setting: setting.name.clone(),
            max_error: numeric.max_error,
            verdict,
            top_k_margin: m.and_then(|m| m.top_k_margin_logit),
            top_p_margin: m.and_then(|m| m.top_p_margin_prob),
        });
    }
    Ok(results)
}

/// One conversation's S-11 result (FR-39): the rendered text and the token ids
/// are judged separately, because a text that matches can still encode wrongly
/// (FR-33), so each gets its own verdict.
#[derive(Debug, Clone)]
pub struct ChatResult {
    pub model_id: String,
    pub conversation: String,
    pub text: Verdict,
    pub tokens: Verdict,
}

impl ChatResult {
    pub fn is_failure(&self) -> bool {
        self.text.is_failure() || self.tokens.is_failure()
    }
}

/// How the engine renders a conversation: the library's `render`, or a faulty
/// one for the negative controls.
pub type RenderFn<'a> = dyn Fn(&[Message], bool) -> String + 'a;
/// How the engine encodes text: the tokenizer's `encode`, or a faulty one.
pub type EncodeFn<'a> = dyn Fn(&str) -> Result<Vec<u32>, Error> + 'a;

/// S-11 for one conversation: render the stored messages with the generation
/// prompt, compare the text with the stored rendering, then encode the engine's
/// rendering and compare its ids with the stored ones.
pub fn check_chat_template(
    model_id: &str,
    conversation: &Conversation,
    render: &RenderFn<'_>,
    encode: &EncodeFn<'_>,
) -> ChatResult {
    let result = |text, tokens| ChatResult {
        model_id: model_id.to_string(),
        conversation: conversation.id.clone(),
        text,
        tokens,
    };
    let mut messages = Vec::new();
    for m in &conversation.messages {
        let role = match m.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                let failure = || Verdict::Fail {
                    position: None,
                    message: format!("conversation {}: stored role {other:?} is not one the engine knows", conversation.id),
                };
                return result(failure(), failure());
            }
        };
        messages.push(Message::new(role, m.content.clone()));
    }
    let text = render(&messages, true);
    let text_verdict = super::compare::text_check(&text, &conversation.rendered);
    let tokens_verdict = match encode(&text) {
        Ok(ids) => super::compare::ids_check(&ids, &conversation.token_ids),
        Err(e) => Verdict::Fail { position: None, message: format!("the encoder failed: {e}") },
    };
    result(text_verdict, tokens_verdict)
}

/// What the harness reports, without judging, about the instruct model on one
/// conversation (FR-39, Requirements §13): the engine's largest logit
/// difference from the stored f64 logits over the rendered prompt, and how its
/// greedy reply compares with the stored one under S-4's near-tie rule. No
/// verdict anywhere reads these fields: the type has no way to fail a run.
#[derive(Debug, Clone)]
pub struct ChatReported {
    pub model_id: String,
    pub conversation: String,
    pub max_logit_error: f64,
    pub reply: Verdict,
    pub engine_reply_len: usize,
    pub reference_reply_len: usize,
}

/// Measures the reported-only figures for one conversation: feeds the stored
/// token ids through the forward pass, and generates a greedy reply from the
/// stored rendering, up to the stored reply's own limit.
pub fn report_chat(
    model: &Model,
    tokenizer: &Tokenizer,
    fixtures: &ChatFixtures,
    conversation: &Conversation,
) -> Result<ChatReported, String> {
    let reference = fixtures.load_logits(conversation)?;
    let run = run_engine(model, &conversation.token_ids)?;
    let max_logit_error = logit_error(&run.logits, &reference.data, fixtures.chat.dims.vocab).max_error;
    let generation = generate(model, tokenizer, &conversation.rendered, fixtures.chat.reply_tokens, &mut |_| {})
        .map_err(|e| format!("{}: {e}", conversation.id))?;
    Ok(ChatReported {
        model_id: fixtures.chat.model_id.clone(),
        conversation: conversation.id.clone(),
        max_logit_error,
        reply: greedy_check(&generation.ids, &conversation.reply_ids, &conversation.reply_top2_gaps),
        engine_reply_len: generation.ids.len(),
        reference_reply_len: conversation.reply_ids.len(),
    })
}
