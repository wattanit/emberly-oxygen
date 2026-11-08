use std::time::{Duration, Instant};

use tracing::Level;

use crate::sampling::Sampler;
use crate::{Error, Model, Sampling, Session, Tokenizer};

/// What [`generate`] returns: the generated token IDs and how long it took.
#[derive(Debug, Clone, PartialEq)]
pub struct Generation {
    pub ids: Vec<u32>,
    pub stats: Stats,
    /// The seed the draws used; `None` for greedy generation (FR-28).
    pub seed: Option<u64>,
}

/// Timings of one generation (FR-21, Spec §12). The first generated token's
/// cost belongs to the prompt: `time_to_first_token` runs from receiving the
/// prompt string (encoding included) to choosing the first token, and
/// `generation_time` is everything after it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Stats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    /// `None` when no token was asked for.
    pub time_to_first_token: Option<Duration>,
    pub generation_time: Duration,
}

impl Stats {
    /// Prompt tokens over the time to first token.
    pub fn prompt_tokens_per_second(&self) -> Option<f64> {
        self.time_to_first_token.map(|t| self.prompt_tokens as f64 / t.as_secs_f64())
    }

    /// The tokens after the first over the time after it; `None` for fewer
    /// than two tokens, since nothing was generated after the first.
    pub fn generation_tokens_per_second(&self) -> Option<f64> {
        let after_first = self.generated_tokens.checked_sub(1).filter(|&n| n > 0)?;
        Some(after_first as f64 / self.generation_time.as_secs_f64())
    }
}

/// Greedy generation (Spec §8): returns the generated token IDs with timings
/// and calls `on_text` with each completed piece of text as soon as it exists.
///
/// The request is checked before any computation (FR-18). Generation stops
/// after `max_new_tokens` tokens or at the end-of-sequence token, which is
/// returned with the others but whose text is not emitted: standard output
/// carries the generated text, and the end-of-sequence token is a signal,
/// not text. `on_text` is never called with an empty string.
///
/// Logging (Spec §14): everything happens in a `generate` span, each forward
/// step in a `step{pos}` span inside it. Info gives the prompt's token count
/// and a run summary; debug gives one event per step.
pub fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    on_text: &mut dyn FnMut(&str),
) -> Result<Generation, Error> {
    generate_with(model, tokenizer, prompt, max_new_tokens, &Sampling::default(), on_text)
}

/// Generation with sampling settings (Spec §8, FR-26 to FR-29). With the
/// default settings, or a temperature of 0, it is greedy and identical to
/// [`generate`]. Otherwise each token is drawn from the adjusted distribution by
/// a generator seeded from `sampling.seed`, which is then required (so the run
/// can be replayed, FR-28) and returned in [`Generation::seed`]. Invalid
/// settings are rejected before any computation (FR-30).
pub fn generate_with(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    sampling: &Sampling,
    on_text: &mut dyn FnMut(&str),
) -> Result<Generation, Error> {
    let _generate = tracing::info_span!("generate").entered();
    sampling.validate()?;
    let sampling_on = sampling.temperature != 0.0;
    let seed = if sampling_on {
        Some(sampling.seed.ok_or_else(|| {
            Error::Rejected("sampling needs a seed so that the run can be replayed: set one".into())
        })?)
    } else {
        if sampling.top_k != 0 || sampling.top_p != 1.0 || sampling.repetition_penalty != 1.0 {
            tracing::warn!("temperature 0 is greedy selection: top-k, top-p and the repetition penalty are ignored");
        }
        None
    };
    let start = Instant::now();
    let config = model.config();
    let prompt_ids = tokenizer.encode(prompt)?;
    if prompt_ids.is_empty() {
        return Err(Error::Rejected(
            "the prompt is empty: it encodes to no tokens".into(),
        ));
    }
    let needed = prompt_ids.len().saturating_add(max_new_tokens);
    if needed > config.context_length {
        return Err(Error::Rejected(format!(
            "{} prompt tokens plus {max_new_tokens} new tokens exceed the context length of {}",
            prompt_ids.len(),
            config.context_length
        )));
    }
    tracing::info!(prompt_tokens = prompt_ids.len(), max_new_tokens, "prompt encoded");
    if max_new_tokens == 0 {
        let stats = Stats {
            prompt_tokens: prompt_ids.len(),
            generated_tokens: 0,
            time_to_first_token: None,
            generation_time: Duration::ZERO,
        };
        return Ok(Generation { ids: Vec::new(), stats, seed });
    }

    let mut steps = Steps {
        session: model.session(false),
        tokenizer,
        debug: tracing::enabled!(Level::DEBUG),
        sampler: seed.map(|seed| Sampler::new(sampling, seed, &prompt_ids)),
    };
    let last = prompt_ids.len() - 1;
    let mut next = 0;
    for (pos, &id) in prompt_ids.iter().enumerate() {
        next = steps.step(pos, id, "prompt", pos == last)?;
    }
    let first_token = start.elapsed();

    let mut decoder = tokenizer.stream_decoder();
    let mut ids = Vec::with_capacity(max_new_tokens);
    loop {
        ids.push(next);
        if next == config.eos_token_id {
            break;
        }
        let text = decoder.push(next)?;
        if !text.is_empty() {
            on_text(&text);
        }
        if ids.len() == max_new_tokens {
            break;
        }
        next = steps.step(prompt_ids.len() + ids.len() - 1, next, "generate", true)?;
    }
    let rest = decoder.finish();
    if !rest.is_empty() {
        on_text(&rest);
    }

    let stats = Stats {
        prompt_tokens: prompt_ids.len(),
        generated_tokens: ids.len(),
        time_to_first_token: Some(first_token),
        generation_time: start.elapsed() - first_token,
    };
    tracing::info!(
        prompt_tokens = stats.prompt_tokens,
        generated_tokens = stats.generated_tokens,
        stopped_at_eos = ids.last() == Some(&config.eos_token_id),
        time_to_first_token_ms = %ms(first_token),
        prompt_tokens_per_s = stats.prompt_tokens_per_second().map(rate),
        generation_ms = %ms(stats.generation_time),
        generation_tokens_per_s = stats.generation_tokens_per_second().map(rate),
        "run summary"
    );
    Ok(Generation { ids, stats, seed })
}

/// The decode loop's forward steps, each in its own span and, at debug
/// level, described by one event.
struct Steps<'a> {
    session: Session<'a>,
    tokenizer: &'a Tokenizer,
    debug: bool,
    /// `None` for greedy generation, which takes the argmax as it always has.
    sampler: Option<Sampler>,
}

impl Steps<'_> {
    /// Feeds `token` at position `pos` and returns the highest-scoring next
    /// token. `chooses` says whether the caller takes it, which the event
    /// shows as `next`; a prompt step before the last does not.
    fn step(&mut self, pos: usize, token: u32, phase: &'static str, chooses: bool) -> Result<u32, Error> {
        let _step = tracing::debug_span!("step", pos).entered();
        let start = self.debug.then(Instant::now);
        let logits = self.session.step(token)?;
        let step_ms = start.map(|s| ms(s.elapsed()));
        let mut choice = None;
        let next = match &mut self.sampler {
            None => argmax(&logits),
            // A prompt step before the last chooses nothing, so it draws nothing.
            Some(_) if !chooses => 0,
            Some(sampler) => {
                let c = sampler.choose(&logits);
                choice = Some(c);
                c.token
            }
        };
        if let Some(step_ms) = step_ms {
            let text = self.text(token);
            let top5 = self.top5(&logits);
            if let Some(c) = choice {
                // Sampling: the chosen token's probability and the tokens the controls left.
                let next_text = self.text(next);
                let probability = format!("{:.6}", c.probability);
                tracing::debug!(phase, token, text = ?text, top5 = %top5, step_ms = %step_ms, next, next_text = ?next_text, probability = %probability, kept = c.kept, "step");
            } else if chooses {
                let next_text = self.text(next);
                tracing::debug!(phase, token, text = ?text, top5 = %top5, step_ms = %step_ms, next, next_text = ?next_text, "step");
            } else {
                tracing::debug!(phase, token, text = ?text, top5 = %top5, step_ms = %step_ms, "step");
            }
        }
        Ok(next)
    }

    fn text(&self, id: u32) -> String {
        self.tokenizer.decode(&[id]).unwrap_or_else(|e| e.to_string())
    }

    /// The five highest logits as `id "text" logit`, highest first; on a tie
    /// the lower ID comes first, as in [`argmax`].
    fn top5(&self, logits: &[f32]) -> String {
        let mut best: Vec<usize> = Vec::with_capacity(6);
        for (id, &x) in logits.iter().enumerate() {
            let at = best.iter().position(|&b| x > logits[b]).unwrap_or(best.len());
            if at < 5 {
                best.insert(at, id);
                best.truncate(5);
            }
        }
        let shown: Vec<String> = best
            .iter()
            .map(|&id| format!("{id} {:?} {:.3}", self.text(id as u32), logits[id]))
            .collect();
        shown.join(", ")
    }
}

/// A duration in milliseconds to the microsecond, for the log.
fn ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1000.0)
}

/// A throughput to two decimals, for the log.
fn rate(r: f64) -> tracing::field::DisplayValue<String> {
    tracing::field::display(format!("{r:.2}"))
}

/// The ID of the highest logit; on an exact tie the lowest ID wins, as the
/// reference's argmax does (FR-16).
fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0;
    for (id, &x) in logits.iter().enumerate() {
        if x > logits[best] {
            best = id;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use super::*;
    use crate::test_log::{collect, Logged};
    use crate::tokenizer::byte_map::ByteMap;
    use crate::weights::{LayerWeights, Weights};
    use crate::{Config, Tensor};

    /// Token IDs 0 to 255 are the bytes of the same value; 256 is the end-of-sequence token.
    const EOS: u32 = 256;
    const VOCAB: usize = 257;

    fn tokenizer() -> Tokenizer {
        let map = ByteMap::new();
        let mut vocab = serde_json::Map::new();
        for b in 0..=255u8 {
            vocab.insert(map.char_for(b).to_string(), Value::from(b));
        }
        vocab.insert("<|endoftext|>".into(), Value::from(EOS));
        let json = serde_json::json!({
            "model": {"vocab": vocab, "merges": []},
            "added_tokens": [{"id": EOS, "content": "<|endoftext|>"}],
        });
        Tokenizer::from_json(&json, Path::new("tokenizer.json")).unwrap()
    }

    fn config(context_length: usize) -> Config {
        Config {
            hidden_size: VOCAB,
            layers: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 2,
            intermediate_size: VOCAB,
            vocab_size: VOCAB,
            context_length,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            eos_token_id: EOS,
        }
    }

    fn zeros(rows: usize, cols: usize) -> Tensor {
        Tensor::new(vec![rows, cols], vec![0.0; rows * cols]).unwrap()
    }

    fn identity(n: usize) -> Tensor {
        let mut t = vec![0.0; n * n];
        (0..n).for_each(|i| t[i * n + i] = 1.0);
        Tensor::new(vec![n, n], t).unwrap()
    }

    fn ones(n: usize) -> Tensor {
        Tensor::new(vec![n], vec![1.0; n]).unwrap()
    }

    /// Every weight zero: every logit is zero at every step.
    fn zero_model(context_length: usize) -> Model {
        let layer = LayerWeights {
            attn_norm: ones(VOCAB),
            q_proj: zeros(2, VOCAB),
            k_proj: zeros(2, VOCAB),
            v_proj: zeros(2, VOCAB),
            o_proj: zeros(VOCAB, 2),
            mlp_norm: ones(VOCAB),
            gate_proj: zeros(VOCAB, VOCAB),
            up_proj: zeros(VOCAB, VOCAB),
            down_proj: zeros(VOCAB, VOCAB),
        };
        let weights = Weights { embed_tokens: zeros(VOCAB, VOCAB), layers: vec![layer], final_norm: ones(VOCAB) };
        Model::from_parts(config(context_length), weights)
    }

    /// A model whose next token depends only on the current one: `next[t]`
    /// follows `t`, and a token not listed is followed by itself. Embeddings are
    /// one-hot and attention contributes nothing; the MLP of the one layer is
    /// a lookup table. After the norm the current token's coordinate is about
    /// 16, the gate and up projections pass it through (SiLU(16) is about 16),
    /// and the down projection writes about 257 into the next token's
    /// coordinate, against 1 left on the current one, so its logit is highest.
    fn bigram_model(next: &[(u32, u32)], context_length: usize) -> Model {
        let mut down = vec![0.0; VOCAB * VOCAB];
        for &(from, to) in next {
            down[to as usize * VOCAB + from as usize] = 1.0;
        }
        let layer = LayerWeights {
            attn_norm: ones(VOCAB),
            q_proj: zeros(2, VOCAB),
            k_proj: zeros(2, VOCAB),
            v_proj: zeros(2, VOCAB),
            o_proj: zeros(VOCAB, 2),
            mlp_norm: ones(VOCAB),
            gate_proj: identity(VOCAB),
            up_proj: identity(VOCAB),
            down_proj: Tensor::new(vec![VOCAB, VOCAB], down).unwrap(),
        };
        let weights = Weights { embed_tokens: identity(VOCAB), layers: vec![layer], final_norm: ones(VOCAB) };
        Model::from_parts(config(context_length), weights)
    }

    /// Runs `generate`, returning the result and every chunk passed to `on_text`.
    fn run(model: &Model, prompt: &str, max_new_tokens: usize) -> (Result<Vec<u32>, Error>, Vec<String>) {
        let mut chunks = Vec::new();
        let ids = generate(model, &tokenizer(), prompt, max_new_tokens, &mut |s| chunks.push(s.to_owned())).map(|g| g.ids);
        assert!(chunks.iter().all(|c| !c.is_empty()), "on_text was called with an empty string: {chunks:?}");
        (ids, chunks)
    }

    /// "ก" (U+0E01) is E0 B8 81: the chain a -> E0 -> B8 -> 81 -> b -> EOS
    /// generates the Thai character over three tokens.
    const THAI_CHAIN: [(u32, u32); 5] = [(0x61, 0xE0), (0xE0, 0xB8), (0xB8, 0x81), (0x81, 0x62), (0x62, EOS)];

    // FR-16, Spec §8: equal logits everywhere, so the lowest ID wins every step.
    #[test]
    fn an_exact_tie_goes_to_the_lowest_id() {
        let model = zero_model(16);
        assert_eq!(model.session(false).step(5).unwrap(), vec![0.0; VOCAB], "every logit is equal");
        let (ids, _) = run(&model, "hello", 4);
        assert_eq!(ids.unwrap(), [0, 0, 0, 0]);
    }

    // FR-16: argmax scans in ascending order and replaces only on a strictly greater logit.
    #[test]
    fn argmax_takes_the_first_of_equal_maxima() {
        assert_eq!(argmax(&[1.0, 3.0, 2.0, 3.0]), 1);
        assert_eq!(argmax(&[-2.0, -1.0, -1.0]), 1);
        assert_eq!(argmax(&[0.5]), 0);
        assert_eq!(argmax(&[f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0]), 2);
    }

    // FR-16, Spec §8: the bigram model generates the chain it was built with.
    #[test]
    fn the_highest_logit_is_taken_each_step() {
        let model = bigram_model(&[(0x61, 0x62), (0x62, 0x63), (0x63, 0x61)], 16);
        let (ids, chunks) = run(&model, "a", 5);
        assert_eq!(ids.unwrap(), [0x62, 0x63, 0x61, 0x62, 0x63]);
        assert_eq!(chunks.concat(), "bcabc");
    }

    // Spec §8: the next token comes from the last prompt position's logits.
    #[test]
    fn the_next_token_follows_the_last_prompt_token() {
        let model = bigram_model(&[(0x61, 0x78), (0x62, 0x79)], 16);
        assert_eq!(run(&model, "ab", 1).0.unwrap(), [0x79]);
        assert_eq!(run(&model, "ba", 1).0.unwrap(), [0x78]);
    }

    // Spec §8, FR-17: generation stops at EOS; EOS is the last ID returned and its text is not emitted.
    #[test]
    fn eos_stops_generation_and_is_returned_but_not_emitted() {
        let model = bigram_model(&THAI_CHAIN, 32);
        let (ids, chunks) = run(&model, "a", 20);
        let ids = ids.unwrap();
        assert_eq!(ids, [0xE0, 0xB8, 0x81, 0x62, EOS]);
        assert_eq!(chunks, ["\u{e01}", "b"]);
        assert!(!chunks.concat().contains("<|endoftext|>"));
    }

    // Spec §8: without EOS, generation stops after the maximum number of new tokens.
    #[test]
    fn the_maximum_stops_generation() {
        let model = bigram_model(&THAI_CHAIN, 32);
        for max in 1..=4 {
            let ids = run(&model, "a", max).0.unwrap();
            assert_eq!(ids, [0xE0, 0xB8, 0x81, 0x62][..max]);
        }
        // A token that is followed by itself never reaches EOS.
        assert_eq!(run(&model, "z", 7).0.unwrap(), [0x7A; 7]);
    }

    // FR-17, FR-15: chunks arrive as text completes, a split Thai character only once whole,
    // and the chunks joined equal the one-shot decode of the IDs (EOS excluded).
    #[test]
    fn streamed_text_equals_the_decode_and_never_holds_half_a_character() {
        let model = bigram_model(&THAI_CHAIN, 32);
        let mut order = Vec::new();
        let ids = generate(&model, &tokenizer(), "a", 20, &mut |s| order.push(s.to_owned())).unwrap().ids;
        assert_eq!(order, ["\u{e01}", "b"], "one chunk when the character completes, one for b");
        let text = tokenizer().decode(&ids[..ids.len() - 1]).unwrap();
        assert_eq!(order.concat(), text);

        // Cut off inside the character: the held bytes are flushed at the end as the
        // one-shot decode reads them, a single U+FFFD.
        for max in 1..=2 {
            let (ids, chunks) = run(&model, "a", max);
            let ids = ids.unwrap();
            assert_eq!(chunks, ["\u{fffd}"]);
            assert_eq!(chunks.concat(), tokenizer().decode(&ids).unwrap());
        }
    }

    // FR-18, Spec §8: an empty prompt is rejected before anything is computed or emitted.
    #[test]
    fn an_empty_prompt_is_rejected() {
        let (result, chunks) = run(&bigram_model(&THAI_CHAIN, 32), "", 4);
        let msg = result.unwrap_err().to_string();
        assert!(msg.contains("empty"), "{msg}");
        assert!(chunks.is_empty());
    }

    // FR-18, FR-13: text whose every byte the vocabulary lacks encodes to nothing and is
    // rejected the same way (byte 4 is absent from SmolLM2's vocabulary, plan §9 task 3.9).
    #[test]
    fn a_prompt_of_dropped_bytes_is_empty() {
        let real = Tokenizer::load(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m")).unwrap();
        assert_eq!(real.encode("\x04\x04").unwrap(), [0u32; 0]);
        let model = zero_model(16);
        let err = generate(&model, &real, "\x04\x04", 4, &mut |_| panic!("nothing is emitted")).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("empty"), "{err}");
    }

    // FR-18, Spec §8: prompt length plus the maximum above the context is rejected, naming the numbers.
    #[test]
    fn a_request_beyond_the_context_is_rejected() {
        let model = bigram_model(&THAI_CHAIN, 8);
        // Exactly the context is allowed.
        assert_eq!(run(&model, "abc", 5).0.unwrap().len(), 5);
        for (prompt, max, numbers) in [("abc", 6, ["3", "6", "8"]), ("a", usize::MAX, ["1", &usize::MAX.to_string(), "8"])] {
            let (result, chunks) = run(&model, prompt, max);
            let err = result.unwrap_err();
            let msg = err.to_string();
            assert!(matches!(err, Error::Rejected(_)), "{msg}");
            assert!(numbers.iter().all(|n| msg.contains(*n)), "{msg}");
            assert!(chunks.is_empty());
        }
    }

    // FR-18, Spec §8: a maximum of zero checks the prompt and generates nothing.
    #[test]
    fn zero_new_tokens_generates_nothing() {
        let model = bigram_model(&THAI_CHAIN, 8);
        let (ids, chunks) = run(&model, "a", 0);
        assert_eq!(ids.unwrap(), [0u32; 0]);
        assert!(chunks.is_empty());
        assert!(run(&model, "", 0).0.is_err(), "the prompt is still checked");
    }

    /// Every logged event of a run of `generate` at `level`, with the IDs and chunks.
    fn logged(model: &Model, prompt: &str, max: usize, level: Option<Level>) -> (Vec<u32>, Vec<String>, Vec<Logged>) {
        let ((ids, chunks), events) = collect(level, || run(model, prompt, max));
        (ids.unwrap(), chunks, events)
    }

    fn steps(events: &[Logged]) -> Vec<&Logged> {
        events.iter().filter(|e| e.message() == "step").collect()
    }

    /// a -> b -> c -> a ...
    const CYCLE: [(u32, u32); 3] = [(0x61, 0x62), (0x62, 0x63), (0x63, 0x61)];

    // FR-22, FR-23, Spec §14: at the default level the library logs nothing; at info, the prompt's
    // token count and a run summary inside the `generate` span, and nothing per step.
    #[test]
    fn info_gives_the_prompt_count_and_a_run_summary_only() {
        let model = bigram_model(&CYCLE, 16);
        assert!(logged(&model, "ab", 3, Some(Level::WARN)).2.is_empty());

        let (_, _, events) = logged(&model, "ab", 3, Some(Level::INFO));
        let messages: Vec<&str> = events.iter().map(Logged::message).collect();
        assert_eq!(messages, ["prompt encoded", "run summary"]);
        assert!(events.iter().all(|e| e.level == Level::INFO && e.path == ["generate"]), "{events:?}");
        assert_eq!(events[0].get("prompt_tokens"), Some("2"));
        let summary = &events[1];
        assert_eq!(summary.get("prompt_tokens"), Some("2"));
        assert_eq!(summary.get("generated_tokens"), Some("3"));
        assert_eq!(summary.get("stopped_at_eos"), Some("false"));
        for field in ["time_to_first_token_ms", "prompt_tokens_per_s", "generation_ms", "generation_tokens_per_s"] {
            let value: f64 = summary.get(field).unwrap_or_else(|| panic!("{field} missing")).parse().unwrap();
            assert!(value.is_finite() && value >= 0.0, "{field} = {value}");
        }
    }

    // Spec §14: with a single generated token there are no tokens after the first, so there is
    // no generation throughput to report, rather than a division by zero.
    #[test]
    fn one_token_has_no_generation_throughput() {
        let (_, _, events) = logged(&bigram_model(&CYCLE, 16), "a", 1, Some(Level::INFO));
        let summary = events.iter().find(|e| e.message() == "run summary").unwrap();
        assert_eq!(summary.get("generated_tokens"), Some("1"));
        assert_eq!(summary.get("generation_tokens_per_s"), None);
    }

    // FR-23, Spec §14: at debug there is one event per forward step, inside `generate` ->
    // `step{pos}`, with the token fed, its text, the top 5 candidates and the step time; prompt
    // steps are marked. Each prompt token appears once as `token` and each generated token once
    // as `next`. The last generated token is never fed back (plan §9, task 5.1), so a run has
    // one step fewer than prompt plus generated tokens.
    #[test]
    fn debug_gives_one_event_per_step() {
        let model = bigram_model(&CYCLE, 16);
        let (ids, _, events) = logged(&model, "ab", 3, Some(Level::DEBUG));
        assert_eq!(ids, [0x63, 0x61, 0x62]);
        let steps = steps(&events);
        assert_eq!(steps.len(), 2 + 3 - 1);
        assert!(events.iter().all(|e| e.level <= Level::DEBUG), "no trace events at debug");

        let expected = [
            ("prompt", 0x61, "\"a\"", None),
            ("prompt", 0x62, "\"b\"", Some(0x63)),
            ("generate", 0x63, "\"c\"", Some(0x61)),
            ("generate", 0x61, "\"a\"", Some(0x62)),
        ];
        for (pos, (step, (phase, token, text, next))) in steps.iter().zip(expected).enumerate() {
            assert_eq!(step.path, ["generate".to_string(), format!("step{{pos={pos}}}")]);
            assert_eq!(step.get("phase"), Some(phase), "{step:?}");
            assert_eq!(step.get("token"), Some(token.to_string().as_str()));
            assert_eq!(step.get("text"), Some(text));
            assert_eq!(step.get("next"), next.map(|n: u32| n.to_string()).as_deref());
            assert!(step.get("step_ms").unwrap().parse::<f64>().unwrap() >= 0.0);
            let top5: Vec<&str> = step.get("top5").unwrap().split(", ").collect();
            assert_eq!(top5.len(), 5, "{top5:?}");
            let first: u32 = top5[0].split(' ').next().unwrap().parse().unwrap();
            assert_eq!(Some(first), next.or(Some(0x62)), "the top candidate is the one chosen");
        }
        let nexts: Vec<u32> = steps.iter().filter_map(|s| s.get("next")).map(|n| n.parse().unwrap()).collect();
        assert_eq!(nexts, ids, "every generated token once, as `next`");
    }

    // Spec §14, FR-16: the top 5 are in descending order and, among equal logits, by ascending ID.
    #[test]
    fn top5_orders_ties_by_id() {
        let (model, tokenizer) = (zero_model(4), tokenizer());
        let steps = Steps { session: model.session(false), tokenizer: &tokenizer, debug: true, sampler: None };
        let mut logits = vec![0.0; VOCAB];
        logits[200] = 2.0;
        logits[100] = 1.0;
        logits[7] = 1.0;
        let ids: Vec<u32> = steps.top5(&logits).split(", ").map(|c| c.split(' ').next().unwrap().parse().unwrap()).collect();
        assert_eq!(ids, [200, 7, 100, 0, 1]);
    }

    // FR-23, Spec §14: at trace each step adds one event per layer, inside `generate` ->
    // `step{pos}` -> `layer{idx}`, with both residual norms and a position for every head.
    #[test]
    fn trace_gives_one_event_per_layer_per_step() {
        let model = bigram_model(&CYCLE, 16);
        let (_, _, events) = logged(&model, "ab", 3, Some(Level::TRACE));
        let layers: Vec<&Logged> = events.iter().filter(|e| e.level == Level::TRACE).collect();
        assert_eq!(layers.len(), steps(&events).len() * model.config().layers);
        for (pos, event) in layers.iter().enumerate() {
            assert_eq!(event.path, ["generate".to_string(), format!("step{{pos={pos}}}"), "layer{idx=0}".to_string()]);
            for field in ["residual_norm_after_attention", "residual_norm_after_mlp"] {
                assert!(event.get(field).unwrap().parse::<f32>().unwrap() > 0.0, "{field}: {event:?}");
            }
            // One head; its query and keys are zero here, so the weights are equal and the
            // earliest position wins.
            assert_eq!(event.get("most_attended"), Some("[0]"));
        }
    }

    // S-8, FR-23: the IDs and the streamed text are the same with no subscriber and at every level.
    #[test]
    fn output_is_the_same_at_every_level() {
        let model = bigram_model(&THAI_CHAIN, 32);
        let (ids, chunks, _) = logged(&model, "a", 20, None);
        for level in [Level::WARN, Level::INFO, Level::DEBUG, Level::TRACE] {
            let (i, c, _) = logged(&model, "a", 20, Some(level));
            assert_eq!((&i, &c), (&ids, &chunks), "{level}");
        }
    }

    // FR-21, Spec §12: the returned timings follow the definitions, and agree with the run summary.
    #[test]
    fn stats_follow_the_definitions() {
        let model = bigram_model(&CYCLE, 16);
        let (generation, events) = collect(Some(Level::INFO), || generate(&model, &tokenizer(), "ab", 4, &mut |_| {}).unwrap());
        let stats = generation.stats;
        assert_eq!((stats.prompt_tokens, stats.generated_tokens), (2, 4));
        let first = stats.time_to_first_token.unwrap();
        let summary = events.iter().find(|e| e.message() == "run summary").unwrap();
        assert_eq!(summary.get("time_to_first_token_ms").unwrap(), format!("{:.3}", first.as_secs_f64() * 1000.0));
        assert_eq!(summary.get("generation_tokens_per_s").unwrap(), format!("{:.2}", stats.generation_tokens_per_second().unwrap()));
        assert_eq!(summary.get("prompt_tokens_per_s").unwrap(), format!("{:.2}", stats.prompt_tokens_per_second().unwrap()));

        let at = |generated_tokens, ms| Stats {
            prompt_tokens: 10,
            generated_tokens,
            time_to_first_token: Some(Duration::from_millis(500)),
            generation_time: Duration::from_millis(ms),
        };
        assert_eq!(at(5, 2000).prompt_tokens_per_second(), Some(20.0));
        assert_eq!(at(5, 2000).generation_tokens_per_second(), Some(2.0), "4 tokens after the first in 2 s");
        assert_eq!(at(1, 0).generation_tokens_per_second(), None);
        assert_eq!(at(0, 0).generation_tokens_per_second(), None);

        let zero = generate(&model, &tokenizer(), "ab", 0, &mut |_| {}).unwrap().stats;
        assert_eq!((zero.prompt_tokens, zero.generated_tokens, zero.time_to_first_token), (2, 0, None));
    }

    /// Every logged event of a sampled run of `generate_with` at `level`.
    fn logged_sampled(
        model: &Model,
        prompt: &str,
        max: usize,
        sampling: &Sampling,
        level: Option<Level>,
    ) -> (Vec<u32>, Vec<String>, Vec<Logged>) {
        let ((ids, chunks), events) = collect(level, || {
            let mut chunks = Vec::new();
            let ids = generate_with(model, &tokenizer(), prompt, max, sampling, &mut |s| chunks.push(s.to_owned()))
                .unwrap()
                .ids;
            (ids, chunks)
        });
        (ids, chunks, events)
    }

    // FR-23, Spec §14: when sampling, the step that chooses a token also reports its
    // probability and how many tokens the controls left; steps before the last prompt
    // token, and greedy steps, report neither.
    #[test]
    fn debug_reports_the_probability_and_kept_tokens_of_a_sampled_choice() {
        let model = bigram_model(&CYCLE, 16);
        let sampling = Sampling { temperature: 1.0, top_k: 1, top_p: 1.0, repetition_penalty: 1.0, seed: Some(3) };
        let (ids, _, events) = logged_sampled(&model, "ab", 3, &sampling, Some(Level::DEBUG));
        assert_eq!(ids, [0x63, 0x61, 0x62], "top-k 1 follows the cycle");
        let steps = steps(&events);
        assert_eq!(steps.len(), 4);
        assert_eq!(steps[0].get("probability"), None, "a prompt step before the last chooses nothing");
        for step in &steps[1..] {
            assert_eq!(step.get("probability"), Some("1.000000"), "{step:?}");
            assert_eq!(step.get("kept"), Some("1"), "{step:?}");
        }
        let (_, _, greedy) = logged(&model, "ab", 3, Some(Level::DEBUG));
        assert!(steps_of(&greedy).iter().all(|s| s.get("probability").is_none() && s.get("kept").is_none()));
    }

    fn steps_of(events: &[Logged]) -> Vec<&Logged> {
        steps(events)
    }

    // S-8: a sampled run gives the same IDs and text with no subscriber and at every
    // level, so what the log computes never feeds back into the draw. All logits are
    // equal in `zero_model`, so the tokens really are random.
    #[test]
    fn sampled_output_is_the_same_at_every_level() {
        let model = zero_model(64);
        let sampling = Sampling { temperature: 1.0, top_k: 0, top_p: 0.9, repetition_penalty: 1.1, seed: Some(11) };
        let (ids, chunks, _) = logged_sampled(&model, "a", 20, &sampling, None);
        assert!(ids.windows(2).any(|w| w[0] != w[1]), "the tokens should be random, not one repeated: {ids:?}");
        for level in [Level::WARN, Level::INFO, Level::DEBUG, Level::TRACE] {
            let (i, c, _) = logged_sampled(&model, "a", 20, &sampling, Some(level));
            assert_eq!((&i, &c), (&ids, &chunks), "{level}");
        }
    }
}
