use crate::{Error, Model, Tokenizer};

/// Greedy generation (Spec §8): returns the generated token IDs and calls
/// `on_text` with each completed piece of text as soon as it exists.
///
/// The request is checked before any computation (FR-18). Generation stops
/// after `max_new_tokens` tokens or at the end-of-sequence token, which is
/// returned with the others but whose text is not emitted: standard output
/// carries the generated text, and the end-of-sequence token is a signal,
/// not text. `on_text` is never called with an empty string.
pub fn generate(
    model: &Model,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    on_text: &mut dyn FnMut(&str),
) -> Result<Vec<u32>, Error> {
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
    if max_new_tokens == 0 {
        return Ok(Vec::new());
    }

    let mut session = model.session(false);
    let mut logits = Vec::new();
    for &id in &prompt_ids {
        logits = session.step(id)?;
    }

    let mut decoder = tokenizer.stream_decoder();
    let mut ids = Vec::with_capacity(max_new_tokens);
    loop {
        let next = argmax(&logits);
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
        logits = session.step(next)?;
    }
    let rest = decoder.finish();
    if !rest.is_empty() {
        on_text(&rest);
    }
    Ok(ids)
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
        let ids = generate(model, &tokenizer(), prompt, max_new_tokens, &mut |s| chunks.push(s.to_owned()));
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
        let ids = generate(&model, &tokenizer(), "a", 20, &mut |s| order.push(s.to_owned())).unwrap();
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
}
