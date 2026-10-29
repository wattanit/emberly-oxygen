//! The forward pass for one token (Spec §7): embedding, then for each layer an
//! attention block and an MLP block that each add to the residual stream, then
//! the final norm and the tied output projection. The stream `x` is never
//! normalized in place; only the copies fed to attention, the MLP and the
//! output are.

use crate::kv_cache::KvCache;
use crate::ops::{add_assign, matvec, mul_assign, rms_norm, rope, silu, softmax};
use crate::weights::Weights;
use crate::{Capture, Config, Error};

/// Processes `token` at the next position and returns the logits for it.
///
/// A token outside the vocabulary and a full context are rejected before
/// anything is touched, so a rejected step leaves the cache and the capture
/// as they were. With `capture`, the embedding, the raw output of every layer
/// and the final-norm output are recorded (Spec §7).
pub(crate) fn step(
    config: &Config,
    weights: &Weights,
    cache: &mut KvCache,
    token: u32,
    mut capture: Option<&mut Capture>,
) -> Result<Vec<f32>, Error> {
    if token as usize >= config.vocab_size {
        return Err(Error::Rejected(format!(
            "token ID {token} is not in the vocabulary (size {})",
            config.vocab_size
        )));
    }
    cache.ensure_room()?;
    let position = cache.len();

    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps as f32;
    let theta = config.rope_theta as f32;
    let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);

    // 1. x = the embedding row of the token.
    let mut x = weights.embed_tokens.row(token as usize).to_vec();
    if let Some(c) = capture.as_deref_mut() {
        c.embedding = x.clone();
        c.layers.clear();
    }

    // 2. Each layer.
    let mut h = vec![0f32; hidden];
    for (index, layer) in weights.layers.iter().enumerate() {
        // Attention block.
        rms_norm(&x, layer.attn_norm.data(), eps, &mut h);
        let mut q = vec![0f32; heads * head_dim];
        let mut k = vec![0f32; kv_heads * head_dim];
        let mut v = vec![0f32; kv_heads * head_dim];
        matvec(&layer.q_proj, &h, &mut q);
        matvec(&layer.k_proj, &h, &mut k);
        matvec(&layer.v_proj, &h, &mut v);
        rope(&mut q, position, heads, head_dim, theta);
        rope(&mut k, position, kv_heads, head_dim, theta);
        cache.append(index, &k, &v)?;

        let mut attended = vec![0f32; heads * head_dim];
        attention(&q, cache, index, position + 1, heads, kv_heads, head_dim, &mut attended);
        let mut projected = vec![0f32; hidden];
        matvec(&layer.o_proj, &attended, &mut projected);
        add_assign(&mut x, &projected);

        // MLP block.
        rms_norm(&x, layer.mlp_norm.data(), eps, &mut h);
        let mut gate = vec![0f32; config.intermediate_size];
        let mut up = vec![0f32; config.intermediate_size];
        matvec(&layer.gate_proj, &h, &mut gate);
        matvec(&layer.up_proj, &h, &mut up);
        silu(&mut gate);
        mul_assign(&mut gate, &up);
        let mut down = vec![0f32; hidden];
        matvec(&layer.down_proj, &gate, &mut down);
        add_assign(&mut x, &down);

        if let Some(c) = capture.as_deref_mut() {
            c.layers.push(x.clone());
        }
    }

    // 3. Logits = the tied embedding table times the final-norm output.
    rms_norm(&x, weights.final_norm.data(), eps, &mut h);
    if let Some(c) = capture {
        c.final_norm = h.clone();
    }
    let mut logits = vec![0f32; config.vocab_size];
    matvec(&weights.embed_tokens, &h, &mut logits);
    Ok(logits)
}

/// Grouped-query attention for one token over the first `positions` cached
/// positions of one layer. Query head `i` reads key-value head
/// `i / (heads / kv_heads)`; scores are dot products scaled by
/// `1/sqrt(head_dim)`, then softmax, then a weighted sum of the values.
/// The heads' outputs are concatenated into `out`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention(
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    assert_eq!(q.len(), heads * head_dim, "q holds `heads` heads");
    assert_eq!(out.len(), heads * head_dim, "out holds `heads` heads");
    assert!(heads % kv_heads == 0, "heads must be a multiple of kv_heads");
    let group = heads / kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0f32; positions];
    for head in 0..heads {
        let kv = head / group;
        let span = kv * head_dim..(kv + 1) * head_dim;
        let query = &q[head * head_dim..(head + 1) * head_dim];

        for (t, score) in scores.iter_mut().enumerate() {
            let key = &cache.key(layer, t)[span.clone()];
            let mut dot = 0f32;
            for (a, b) in query.iter().zip(key) {
                dot += a * b;
            }
            *score = dot * scale;
        }
        softmax(&mut scores);

        let result = &mut out[head * head_dim..(head + 1) * head_dim];
        result.fill(0.0);
        for (t, &weight) in scores.iter().enumerate() {
            let value = &cache.value(layer, t)[span.clone()];
            for (r, v) in result.iter_mut().zip(value) {
                *r += weight * v;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::LayerWeights;
    use crate::{Model, Tensor};

    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "element {i}: {a} vs {e} (tolerance {tolerance}); {actual:?}");
        }
    }

    // FR-1, Spec §7, §11: grouped-query attention by hand. Four query heads share two key-value
    // heads, head_dim 2, two cached positions. A score of ln 3 against 0 gives weights 3/4 and
    // 1/4; equal scores give 1/2 each. Heads 0 and 1 read key-value head 0, heads 2 and 3 read
    // head 1; reading by `head % kv_heads` instead would swap heads 1 and 2's sources.
    #[test]
    fn attention_groups_query_heads_over_key_value_heads() {
        let mut cache = KvCache::new(1, 2, 2, 8);
        // Keys:   position 0: head 0 [1, 0], head 1 [0, 1];  position 1: all zero.
        // Values: position 0: head 0 [10, 0], head 1 [0, 20]; position 1: head 0 [0, 10], head 1 [20, 0].
        cache.append(0, &[1., 0., 0., 1.], &[10., 0., 0., 20.]).unwrap();
        cache.append(0, &[0., 0., 0., 0.], &[0., 10., 20., 0.]).unwrap();
        let big = std::f32::consts::SQRT_2 * 3f32.ln(); // after the 1/sqrt(2) scale, a score of ln 3
        let q = [big, 0., 0., 0., 0., big, 0., 0.];
        let mut out = [0f32; 8];
        attention(&q, &cache, 0, 2, 4, 2, 2, &mut out);
        close(&out, &[7.5, 2.5, 5., 5., 5., 15., 10., 10.], 1e-5);
    }

    // FR-1, Spec §7: only positions up to the current one are read, whatever else the cache holds.
    #[test]
    fn attention_reads_only_the_positions_it_is_given() {
        let mut cache = KvCache::new(1, 1, 2, 8);
        cache.append(0, &[1., 0.], &[3., 4.]).unwrap();
        cache.append(0, &[0., 1.], &[5., 6.]).unwrap();
        cache.append(0, &[100., 100.], &[1e6, 1e6]).unwrap();
        let mut first = [0f32; 2];
        attention(&[1., 1.], &cache, 0, 1, 1, 1, 2, &mut first);
        assert_eq!(first, [3., 4.], "one position: the softmax of a single score is 1");
        let mut two = [0f32; 2];
        attention(&[1., 1.], &cache, 0, 2, 1, 1, 2, &mut two);
        close(&two, &[4., 5.], 1e-5); // equal scores: the mean of the first two values
    }

    /// The tiny model of the reference below: integer-formula weights both sides can compute.
    fn matrix(rows: usize, cols: usize, seed: usize) -> Tensor {
        let data = (0..rows * cols)
            .map(|n| (((n / cols) * 5 + (n % cols) * 3 + seed) % 7) as f32 / 7.0 - 0.5)
            .collect();
        Tensor::new(vec![rows, cols], data).unwrap()
    }

    fn norm(len: usize, seed: usize) -> Tensor {
        Tensor::new(vec![len], (0..len).map(|j| 1.0 + ((j + seed) % 3) as f32 * 0.25).collect()).unwrap()
    }

    fn tiny_config(context_length: usize) -> Config {
        Config {
            hidden_size: 4,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            intermediate_size: 3,
            vocab_size: 4,
            context_length,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    fn tiny_model(context_length: usize) -> Model {
        let mut seed = 10;
        let mut next = || {
            seed += 1;
            seed - 1
        };
        let layers = (0..2)
            .map(|_| LayerWeights {
                attn_norm: norm(4, next()),
                q_proj: matrix(4, 4, next()),
                k_proj: matrix(2, 4, next()),
                v_proj: matrix(2, 4, next()),
                o_proj: matrix(4, 4, next()),
                mlp_norm: norm(4, next()),
                gate_proj: matrix(3, 4, next()),
                up_proj: matrix(3, 4, next()),
                down_proj: matrix(4, 3, next()),
            })
            .collect();
        let weights = Weights { embed_tokens: matrix(4, 4, 1), layers, final_norm: norm(4, next()) };
        Model::from_parts(tiny_config(context_length), weights)
    }

    // FR-1, Spec §7: a two-layer model fed three tokens, against an independent numpy
    // implementation of Spec §7 in float64 (weights from the same integer formula: row i,
    // column j of a matrix is ((5i + 3j + seed) mod 7) / 7 - 0.5; a norm weight is
    // 1 + ((j + seed) mod 3) / 4). Covers RoPE at positions 0, 1 and 2, the cache across
    // steps, grouped-query attention (2 heads over 1), the residual stream and the tied output.
    #[test]
    fn a_tiny_model_matches_an_independent_reference() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        let expected: [(u32, [f32; 4], [[f32; 4]; 2], [f32; 4]); 3] = [
            (2, [-0.478599321, 0.556913018, 1.654692089, -0.721665280],
             [[-0.276911449, -2.182372800, 0.893603523, 0.488145256], [0.296345671, -2.282515693, 0.286672257, 0.585264398]],
             [0.309704445, -2.862489287, 0.239675976, 0.611647153]),
            (0, [0.414470565, -0.232617106, 1.413414301, -0.064865065],
             [[-0.470090844, -0.641421639, -0.072362666, 0.219245127], [-0.378449033, -0.664430486, -0.183649913, 0.224889104]],
             [-1.156706917, -2.436950627, -0.449052014, 0.687360147]),
            (3, [-0.181693408, -0.631061703, 1.246354816, -0.133866666],
             [[-0.128820170, -0.352348034, -0.009600638, 0.576087067], [-0.288078163, -0.350019463, 0.215527230, 0.645147227]],
             [-0.881036694, -1.284567999, 0.527321882, 1.973070001]),
        ];
        for (token, logits, layers, final_norm) in expected {
            let got = session.step(token).unwrap();
            close(&got, &logits, 1e-5);
            let capture = session.capture().unwrap();
            assert_eq!(capture.layers.len(), 2);
            close(&capture.layers[0], &layers[0], 1e-5);
            close(&capture.layers[1], &layers[1], 1e-5);
            close(&capture.final_norm, &final_norm, 1e-5);
        }
        assert_eq!(session.cache.len(), 3, "one cached position per step");
    }

    // FR-1, Spec §7: the capture is exactly what the pass says it is.
    #[test]
    fn capture_records_embedding_layers_and_final_norm() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        let logits = session.step(1).unwrap();
        let capture = session.capture().unwrap().clone();
        assert_eq!(capture.embedding, model.weights.embed_tokens.row(1), "the embedding row of the token");
        assert_eq!(capture.layers.len(), 2);
        assert!(capture.layers.iter().all(|l| l.len() == 4));

        // The final-norm output is the last layer's output through the final norm,
        // and the logits are the tied embedding table times it.
        let mut norm_out = vec![0f32; 4];
        rms_norm(&capture.layers[1], model.weights.final_norm.data(), 1e-5, &mut norm_out);
        assert_eq!(capture.final_norm, norm_out);
        let mut again = vec![0f32; 4];
        matvec(&model.weights.embed_tokens, &capture.final_norm, &mut again);
        assert_eq!(logits, again);
    }

    // Spec §7: capture is off by default and changes nothing about the output.
    #[test]
    fn capture_off_gives_the_same_logits() {
        let model = tiny_model(8);
        let (mut on, mut off) = (model.session(true), model.session(false));
        assert!(off.capture().is_none());
        for token in [0, 3, 1, 2] {
            assert_eq!(on.step(token).unwrap(), off.step(token).unwrap());
        }
        assert!(off.capture().is_none());
    }

    // FR-1, Spec §7: causal by construction: a position's logits depend only on the tokens up to it.
    #[test]
    fn earlier_positions_do_not_depend_on_later_tokens() {
        let model = tiny_model(8);
        let (mut a, mut b) = (model.session(false), model.session(false));
        let first = a.step(2).unwrap();
        assert_eq!(first, b.step(2).unwrap());
        a.step(1).unwrap();
        b.step(3).unwrap();
        // Same prefix, different later tokens: a fresh session reproduces the first answer.
        assert_eq!(model.session(false).step(2).unwrap(), first);
    }

    // Spec §14: an unknown token is rejected naming it, and the session is left as it was.
    #[test]
    fn an_unknown_token_is_rejected_and_changes_nothing() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        let err = session.step(4).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains('4'), "{err}");
        assert!(session.step(u32::MAX).is_err());
        assert_eq!(session.cache.len(), 0);
        // The next step behaves as the first step of a fresh session.
        assert_eq!(session.step(2).unwrap(), model.session(false).step(2).unwrap());
    }

    // Spec §14, FR-18: past the context length a step is rejected, and the session is left as it was.
    #[test]
    fn a_full_context_rejects_the_step_and_changes_nothing() {
        let model = tiny_model(2);
        let mut session = model.session(true);
        session.step(0).unwrap();
        let last = session.step(1).unwrap();
        let capture_before = session.capture().unwrap().clone();
        let err = session.step(2).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("context"), "{err}");
        assert_eq!(session.cache.len(), 2);
        assert_eq!(session.capture().unwrap().layers, capture_before.layers);
        // The cache is intact: a fresh session fed the same two tokens agrees.
        let mut fresh = model.session(false);
        fresh.step(0).unwrap();
        assert_eq!(fresh.step(1).unwrap(), last);
    }
}
