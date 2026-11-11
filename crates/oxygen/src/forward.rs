//! The forward pass for one token (Spec §7): embedding, then for each layer an
//! attention block and an MLP block that each add to the residual stream, then
//! the final norm and the tied output projection. The stream `x` is never
//! normalized in place; only the copies fed to attention, the MLP and the
//! output are.
//!
//! At the trace level each layer reports, inside a `layer{idx}` span, the
//! residual-stream norm after attention and after the MLP and the position
//! each head attends to most (Spec §14). They are computed only when trace is
//! enabled and are never read back, so the result is the same at every level.

use crate::fault::{self, Fault};
use crate::kv_cache::KvCache;
use std::sync::Arc;

use crate::pool::{Job, Pool};
#[cfg(feature = "bench-kernels")]
use crate::ops::matmul_pooled;
use crate::ops::{add_assign, matmul_blocked_pooled, matvec_pooled, mul_assign, rms_norm, rope, silu, softmax};
use crate::weights::Weights;
use crate::{Capture, Config, Error, Logits};

use tracing::Level;

/// A batched product: the blocked one (FR-46), or, only where a test or the
/// benchmark asked for it with the `bench-kernels` feature, the 0.4.1 loop that
/// FR-47's drift check compares it with.
fn product(pool: &Pool, blocked: bool, weight: &crate::Tensor, input: &[f32], count: usize, output: &mut [f32]) {
    #[cfg(feature = "bench-kernels")]
    if !blocked {
        return matmul_pooled(pool, weight, input, count, output);
    }
    let _ = blocked;
    matmul_blocked_pooled(pool, weight, input, count, output);
}

/// Processes `token` at the next position and returns the logits for it.
///
/// A token outside the vocabulary and a full context are rejected before
/// anything is touched, so a rejected step leaves the cache and the capture
/// as they were. With `capture`, the embedding, the raw output of every layer
/// and the final-norm output are recorded (Spec §7).
pub(crate) fn step(
    config: &Config,
    weights: &Weights,
    pool: &Pool,
    cache: &mut KvCache,
    token: u32,
    mut capture: Option<&mut Capture>,
    fault: Option<Fault>,
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
    let stats = tracing::enabled!(Level::TRACE);
    let mut most_attended = vec![0usize; if stats { heads } else { 0 }];
    let mut h = vec![0f32; hidden];
    for (index, layer) in weights.layers.iter().enumerate() {
        let _layer = tracing::trace_span!("layer", idx = index).entered();

        // Attention block.
        rms_norm(&x, layer.attn_norm.data(), eps, &mut h);
        let mut q = vec![0f32; heads * head_dim];
        let mut k = vec![0f32; kv_heads * head_dim];
        let mut v = vec![0f32; kv_heads * head_dim];
        matvec_pooled(pool, &layer.q_proj, &h, &mut q);
        matvec_pooled(pool, &layer.k_proj, &h, &mut k);
        matvec_pooled(pool, &layer.v_proj, &h, &mut v);
        if let Some(f) = fault.filter(|f| f.interleaves_rope()) {
            fault::rope_interleaved(f, &mut q, position, heads, head_dim, theta);
            fault::rope_interleaved(f, &mut k, position, kv_heads, head_dim, theta);
        } else {
            rope(&mut q, position, heads, head_dim, theta);
            rope(&mut k, position, kv_heads, head_dim, theta);
        }
        cache.append(index, &k, &v)?;

        let mut attended = vec![0f32; heads * head_dim];
        if stats {
            // The trace's most-attended positions come from the serial path; the output is the same.
            attention(&q, cache, index, position + 1, heads, kv_heads, head_dim, &mut attended, Some(&mut most_attended[..]));
        } else {
            attention_pooled(pool, &q, cache, index, position + 1, heads, kv_heads, head_dim, &mut attended, ATTENTION_MIN_POSITIONS);
        }
        let mut projected = vec![0f32; hidden];
        matvec_pooled(pool, &layer.o_proj, &attended, &mut projected);
        add_assign(&mut x, &projected);
        let after_attention = if stats { norm(&x) } else { 0.0 };

        // MLP block.
        rms_norm(&x, layer.mlp_norm.data(), eps, &mut h);
        let mut gate = vec![0f32; config.intermediate_size];
        let mut up = vec![0f32; config.intermediate_size];
        matvec_pooled(pool, &layer.gate_proj, &h, &mut gate);
        matvec_pooled(pool, &layer.up_proj, &h, &mut up);
        silu(&mut gate);
        mul_assign(&mut gate, &up);
        let mut down = vec![0f32; hidden];
        matvec_pooled(pool, &layer.down_proj, &gate, &mut down);
        add_assign(&mut x, &down);

        if stats {
            tracing::trace!(
                residual_norm_after_attention = %after_attention,
                residual_norm_after_mlp = %norm(&x),
                most_attended = ?most_attended,
                "layer"
            );
        }
        if let Some(c) = capture.as_deref_mut() {
            c.layers.push(x.clone());
        }
    }

    // 3. Logits = the tied embedding table times the final-norm output.
    if fault.is_some_and(Fault::skips_final_norm) {
        h.copy_from_slice(&x);
    } else {
        rms_norm(&x, weights.final_norm.data(), eps, &mut h);
    }
    if let Some(c) = capture {
        c.final_norm = h.clone();
    }
    let mut logits = vec![0f32; config.vocab_size];
    matvec_pooled(pool, &weights.embed_tokens, &h, &mut logits);
    fault::perturb_logits(fault, position, &mut logits);
    Ok(logits)
}

/// Processes a whole prompt in one pass over an empty cache (Spec §7, FR-40).
///
/// An empty prompt, a token outside the vocabulary and a prompt longer than
/// the context are rejected before anything is touched, so a rejected call
/// leaves the cache and the capture as they were. A cache that already holds
/// positions is an internal bug: cache reuse is deferred (Spec §2).
pub(crate) fn prefill(
    config: &Config,
    weights: &Weights,
    pool: &Pool,
    cache: &mut KvCache,
    tokens: &[u32],
    logits: Logits,
    mut capture: Option<&mut Capture>,
    fault: Option<Fault>,
    blocked: bool,
) -> Result<Vec<f32>, Error> {
    assert_eq!(cache.len(), 0, "prefill needs an empty cache");
    if tokens.is_empty() {
        return Err(Error::Rejected("the prompt is empty".into()));
    }
    if let Some(&token) = tokens.iter().find(|&&t| t as usize >= config.vocab_size) {
        return Err(Error::Rejected(format!(
            "token ID {token} is not in the vocabulary (size {})",
            config.vocab_size
        )));
    }
    if tokens.len() > config.context_length {
        return Err(Error::Rejected(format!(
            "the prompt of {} tokens does not fit in the model's context of {} positions (FR-18)",
            tokens.len(),
            config.context_length
        )));
    }

    let n = tokens.len();
    let hidden = config.hidden_size;
    let eps = config.rms_norm_eps as f32;
    let theta = config.rope_theta as f32;
    let (heads, kv_heads, head_dim) = (config.heads, config.kv_heads, config.head_dim);
    let (q_width, kv_width) = (heads * head_dim, kv_heads * head_dim);

    // 1. X = the embedding rows of the tokens, one row per token.
    let mut x = Vec::with_capacity(n * hidden);
    for &token in tokens {
        x.extend_from_slice(weights.embed_tokens.row(token as usize));
    }
    if let Some(c) = capture.as_deref_mut() {
        c.embedding = x.clone();
        c.layers.clear();
    }

    // 2. Each layer, every operation over all rows. At trace level each layer reports the
    // statistics of the last row, the one whose logits generation uses (Spec §14).
    let stats = tracing::enabled!(Level::TRACE);
    let mut most_attended = vec![0usize; if stats { heads } else { 0 }];
    let mut h = vec![0f32; n * hidden];
    for (index, layer) in weights.layers.iter().enumerate() {
        let _layer = tracing::trace_span!("layer", idx = index).entered();
        // Attention block.
        for (xr, hr) in x.chunks_exact(hidden).zip(h.chunks_exact_mut(hidden)) {
            rms_norm(xr, layer.attn_norm.data(), eps, hr);
        }
        let mut q = vec![0f32; n * q_width];
        let mut k = vec![0f32; n * kv_width];
        let mut v = vec![0f32; n * kv_width];
        product(pool, blocked, &layer.q_proj, &h, n, &mut q);
        product(pool, blocked, &layer.k_proj, &h, n, &mut k);
        product(pool, blocked, &layer.v_proj, &h, n, &mut v);
        for t in 0..n {
            let (qr, kr) = (&mut q[t * q_width..(t + 1) * q_width], &mut k[t * kv_width..(t + 1) * kv_width]);
            if let Some(f) = fault.filter(|f| f.interleaves_rope()) {
                fault::rope_interleaved(f, qr, t, heads, head_dim, theta);
                fault::rope_interleaved(f, kr, t, kv_heads, head_dim, theta);
            } else {
                rope(qr, t, heads, head_dim, theta);
                rope(kr, t, kv_heads, head_dim, theta);
            }
            cache.append(index, kr, &v[t * kv_width..(t + 1) * kv_width])?;
        }

        // The whole prompt is in the cache, so the count is what keeps the future out:
        // row `t` reads positions 0..=t and no more (Spec §7, causal mask).
        let mut attended = vec![0f32; n * q_width];
        let unmasked = fault.is_some_and(Fault::unmasks_attention);
        if stats {
            for t in 0..n {
                let out = &mut attended[t * q_width..(t + 1) * q_width];
                // The unmasked fault reads the whole prompt from every row (Spec §17).
                let positions = if unmasked { n } else { t + 1 };
                let record = (t == n - 1).then_some(&mut most_attended[..]);
                attention(&q[t * q_width..(t + 1) * q_width], cache, index, positions, heads, kv_heads, head_dim, out, record);
            }
        } else {
            attention_rows_pooled(pool, &q, cache, index, n, unmasked, heads, kv_heads, head_dim, &mut attended);
        }
        let mut projected = vec![0f32; n * hidden];
        product(pool, blocked, &layer.o_proj, &attended, n, &mut projected);
        add_assign(&mut x, &projected);
        let last_row = n * hidden - hidden..;
        let after_attention = if stats { norm(&x[last_row.clone()]) } else { 0.0 };

        // MLP block.
        for (xr, hr) in x.chunks_exact(hidden).zip(h.chunks_exact_mut(hidden)) {
            rms_norm(xr, layer.mlp_norm.data(), eps, hr);
        }
        let mut gate = vec![0f32; n * config.intermediate_size];
        let mut up = vec![0f32; n * config.intermediate_size];
        product(pool, blocked, &layer.gate_proj, &h, n, &mut gate);
        product(pool, blocked, &layer.up_proj, &h, n, &mut up);
        silu(&mut gate);
        mul_assign(&mut gate, &up);
        let mut down = vec![0f32; n * hidden];
        product(pool, blocked, &layer.down_proj, &gate, n, &mut down);
        add_assign(&mut x, &down);

        if stats {
            tracing::trace!(
                residual_norm_after_attention = %after_attention,
                residual_norm_after_mlp = %norm(&x[last_row]),
                most_attended = ?most_attended,
                "layer"
            );
        }
        if let Some(c) = capture.as_deref_mut() {
            c.layers.push(x.clone());
        }
    }

    // 3. Logits = the tied embedding table times the final-norm output: of every row, or
    // of the last row alone, which is all generation uses.
    let first = if logits == Logits::All { 0 } else { n - 1 };
    let rows = n - first;
    let mut normed = vec![0f32; rows * hidden];
    for (xr, hr) in x[first * hidden..].chunks_exact(hidden).zip(normed.chunks_exact_mut(hidden)) {
        if fault.is_some_and(Fault::skips_final_norm) {
            hr.copy_from_slice(xr);
        } else {
            rms_norm(xr, weights.final_norm.data(), eps, hr);
        }
    }
    if let Some(c) = capture {
        c.final_norm = normed.clone();
    }
    let vocab = config.vocab_size;
    let mut out = vec![0f32; rows * vocab];
    if logits == Logits::All {
        product(pool, blocked, &weights.embed_tokens, &normed, rows, &mut out);
    } else {
        matvec_pooled(pool, &weights.embed_tokens, &normed, &mut out);
    }
    for (r, row) in out.chunks_exact_mut(vocab).enumerate() {
        fault::perturb_logits(fault, first + r, row);
    }
    Ok(out)
}

/// Grouped-query attention for one token over the first `positions` cached
/// positions of one layer. Query head `i` reads key-value head
/// `i / (heads / kv_heads)`; scores are dot products scaled by
/// `1/sqrt(head_dim)`, then softmax, then a weighted sum of the values.
/// The heads' outputs are concatenated into `out`. With `most_attended`, each
/// head's most heavily weighted position is written there (the earliest on a
/// tie), for the trace log.
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
    most_attended: Option<&mut [usize]>,
) {
    assert_eq!(q.len(), heads * head_dim, "q holds `heads` heads");
    assert_eq!(out.len(), heads * head_dim, "out holds `heads` heads");
    assert!(heads % kv_heads == 0, "heads must be a multiple of kv_heads");
    assert!(positions * kv_heads * head_dim <= cache.keys(layer).len(), "position {positions} is not in the cache");
    attention_heads(q, cache.keys(layer), cache.values(layer), positions, 0..heads, heads, kv_heads, head_dim, out, most_attended);
}

/// The heads in `range` of [`attention`], over a layer's flat keys and values
/// (`[position][kv_heads × head_dim]`), written to `out` (`range.len() ×
/// head_dim` values). Each head is computed whole, in the same order whatever
/// the range, which is what lets threads divide the heads without changing a
/// bit (FR-43). `most_attended` is indexed by head.
#[allow(clippy::too_many_arguments)]
fn attention_heads(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    positions: usize,
    range: std::ops::Range<usize>,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
    mut most_attended: Option<&mut [usize]>,
) {
    let group = heads / kv_heads;
    let width = kv_heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let mut scores = vec![0f32; positions];
    for (slot, head) in range.enumerate() {
        let kv = head / group;
        let span = kv * head_dim..(kv + 1) * head_dim;
        let query = &q[head * head_dim..(head + 1) * head_dim];

        for (t, score) in scores.iter_mut().enumerate() {
            let key = &keys[t * width..(t + 1) * width][span.clone()];
            let mut dot = 0f32;
            for (a, b) in query.iter().zip(key) {
                dot += a * b;
            }
            *score = dot * scale;
        }
        softmax(&mut scores);
        if let Some(m) = most_attended.as_deref_mut() {
            let mut best = 0;
            for (t, &weight) in scores.iter().enumerate() {
                if weight > scores[best] {
                    best = t;
                }
            }
            m[head] = best;
        }

        let result = &mut out[slot * head_dim..(slot + 1) * head_dim];
        result.fill(0.0);
        for (t, &weight) in scores.iter().enumerate() {
            let value = &values[t * width..(t + 1) * width][span.clone()];
            for (r, v) in result.iter_mut().zip(value) {
                *r += weight * v;
            }
        }
    }
}

/// Below this many positions a decode step's attention runs on the calling
/// thread: the hand-over costs about as much as the work (plan 0.4.1 §5, task
/// 20.1: parity near 128 positions, 1.6× at 512). Initial; tune with use.
pub(crate) const ATTENTION_MIN_POSITIONS: usize = 256;

/// [`attention`] with the query heads divided among the pool's threads
/// (FR-43). Each head is computed whole by one thread with the same code, so
/// the result equals [`attention`]'s bit for bit at every thread count.
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_pooled(
    pool: &Pool,
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    positions: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
    min_positions: usize,
) {
    if pool.threads() == 1 || positions < min_positions {
        return attention(q, cache, layer, positions, heads, kv_heads, head_dim, out, None);
    }
    assert_eq!(q.len(), heads * head_dim, "q holds `heads` heads");
    assert_eq!(out.len(), heads * head_dim, "out holds `heads` heads");
    assert!(heads % kv_heads == 0, "heads must be a multiple of kv_heads");
    assert!(positions * kv_heads * head_dim <= cache.keys(layer).len(), "position {positions} is not in the cache");
    let (keys, values) = cache.shared(layer);
    let q = Arc::new(q.to_vec());
    let job: Job = Arc::new(move |range| {
        let mut part = vec![0f32; range.len() * head_dim];
        attention_heads(&q, &keys, &values, positions, range, heads, kv_heads, head_dim, &mut part, None);
        part
    });
    out.copy_from_slice(&pool.run(heads, &job));
}

/// The attention of every row of a prompt (Spec §7), `q` holding `n` rows, row
/// `t` over `t + 1` cached positions (or all `n` with `unmasked`, the fault),
/// the rows divided among the pool's threads (FR-43).
#[allow(clippy::too_many_arguments)]
pub(crate) fn attention_rows_pooled(
    pool: &Pool,
    q: &[f32],
    cache: &KvCache,
    layer: usize,
    n: usize,
    unmasked: bool,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) {
    let q_width = heads * head_dim;
    assert_eq!(q.len(), n * q_width, "q holds `n` rows");
    assert_eq!(out.len(), n * q_width, "out holds `n` rows");
    assert!(heads % kv_heads == 0, "heads must be a multiple of kv_heads");
    assert!(n * kv_heads * head_dim <= cache.keys(layer).len(), "the prompt is not in the cache");
    let row_positions = move |t: usize| if unmasked { n } else { t + 1 };
    if pool.threads() == 1 || n < 2 {
        for t in 0..n {
            let row = t * q_width..(t + 1) * q_width;
            attention(&q[row.clone()], cache, layer, row_positions(t), heads, kv_heads, head_dim, &mut out[row], None);
        }
        return;
    }
    // Row `t` costs `t + 1` positions, so equal runs of rows would leave the last
    // thread with most of the work. The pool's ranges run over a zigzag order
    // (first row, last row, second row, ...), which gives each range a mix of cheap
    // and dear rows; the rows are put back in order below. Which thread computes a
    // row changes no bit of it (FR-43).
    let row_of = move |i: usize| if i % 2 == 0 { i / 2 } else { n - 1 - i / 2 };
    let (keys, values) = cache.shared(layer);
    let q = Arc::new(q.to_vec());
    let job: Job = Arc::new(move |range| {
        let mut part = vec![0f32; range.len() * q_width];
        for (slot, i) in range.enumerate() {
            let t = row_of(i);
            let out = &mut part[slot * q_width..(slot + 1) * q_width];
            attention_heads(&q[t * q_width..(t + 1) * q_width], &keys, &values, row_positions(t), 0..heads, heads, kv_heads, head_dim, out, None);
        }
        part
    });
    let zigzag = pool.run(n, &job);
    for i in 0..n {
        let t = row_of(i);
        out[t * q_width..(t + 1) * q_width].copy_from_slice(&zigzag[i * q_width..(i + 1) * q_width]);
    }
}

/// The Euclidean norm of the residual stream, for the trace log.
fn norm(x: &[f32]) -> f32 {
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    sum.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::matvec;
    use crate::Session;
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
        attention(&q, &cache, 0, 2, 4, 2, 2, &mut out, None);
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
        attention(&[1., 1.], &cache, 0, 1, 1, 1, 2, &mut first, None);
        assert_eq!(first, [3., 4.], "one position: the softmax of a single score is 1");
        let mut two = [0f32; 2];
        attention(&[1., 1.], &cache, 0, 2, 1, 1, 2, &mut two, None);
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
            eos_token_id: 0,
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

    // FR-40, FR-18, Spec §7: an empty prompt is rejected, not a panic, and nothing is touched.
    #[test]
    fn prefill_rejects_an_empty_prompt() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        let err = session.prefill(&[], Logits::Last).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("empty"), "{err}");
        assert_eq!(session.cache.len(), 0);
        assert!(session.capture().unwrap().layers.is_empty());
    }

    // FR-40, FR-12, Spec §7: a token outside the vocabulary anywhere in the prompt is rejected
    // before any position is processed, so the cache and the capture are as they were.
    #[test]
    fn prefill_rejects_an_unknown_token_and_changes_nothing() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        for prompt in [[9u32, 1, 2], [1, 2, 4], [1, 9, 2]] {
            let err = session.prefill(&prompt, Logits::All).unwrap_err();
            assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("not in the vocabulary"), "{err}");
            assert_eq!(session.cache.len(), 0, "{prompt:?}");
            let capture = session.capture().unwrap();
            assert!(capture.embedding.is_empty() && capture.layers.is_empty() && capture.final_norm.is_empty());
        }
    }

    // FR-40, FR-18, Spec §7: a prompt longer than the context is rejected before any computation,
    // and the same call with a prompt of exactly the context length is not rejected for its length.
    #[test]
    fn prefill_rejects_a_prompt_longer_than_the_context() {
        let model = tiny_model(4);
        let mut session = model.session(true);
        let err = session.prefill(&[0, 1, 2, 3, 0], Logits::Last).unwrap_err();
        assert!(matches!(err, Error::Rejected(_)) && err.to_string().contains("context"), "{err}");
        assert_eq!(session.cache.len(), 0);
        assert!(session.capture().unwrap().layers.is_empty());
    }

    // FR-40, Spec §2, §14: prefill is for an empty session; a session that already holds
    // positions is an internal bug (cache reuse is deferred), so it panics.
    #[test]
    #[should_panic(expected = "empty")]
    fn prefill_on_a_session_with_positions_panics() {
        let model = tiny_model(8);
        let mut session = model.session(false);
        session.step(1).unwrap();
        let _ = session.prefill(&[1, 2], Logits::Last);
    }

    /// The one-token route over a prompt: logits row by row, and what each step captured.
    fn by_steps<'a>(model: &'a Model, tokens: &[u32]) -> (Session<'a>, Vec<Vec<f32>>, Vec<Capture>) {
        let mut session = model.session(true);
        let mut logits = Vec::new();
        let mut captures = Vec::new();
        for &token in tokens {
            logits.push(session.step(token).unwrap());
            captures.push(session.capture().unwrap().clone());
        }
        (session, logits, captures)
    }

    // FR-40, FR-41, Spec §7: the batched pass equals the one-token pass at every position within
    // 1e-5: its logits, the embedding, every layer's output and the final-norm output (each
    // `[positions, hidden]`), and the keys and values it leaves in the cache.
    #[test]
    fn prefill_equals_the_one_token_pass_at_every_position() {
        let model = tiny_model(8);
        let tokens = [2u32, 0, 3, 1, 1];
        let (stepped, want_logits, want_captures) = by_steps(&model, &tokens);
        let mut session = model.session(true);
        let got = session.prefill(&tokens, Logits::All).unwrap();
        let vocab = 4;
        assert_eq!(got.len(), tokens.len() * vocab);
        for (t, want) in want_logits.iter().enumerate() {
            close(&got[t * vocab..(t + 1) * vocab], want, 1e-5);
        }
        let capture = session.capture().unwrap();
        let rows = |pick: &dyn Fn(&Capture) -> Vec<f32>| -> Vec<f32> { want_captures.iter().flat_map(|c| pick(c)).collect() };
        close(&capture.embedding, &rows(&|c| c.embedding.clone()), 1e-5);
        close(&capture.final_norm, &rows(&|c| c.final_norm.clone()), 1e-5);
        assert_eq!(capture.layers.len(), 2);
        for layer in 0..2 {
            close(&capture.layers[layer], &rows(&|c| c.layers[layer].clone()), 1e-5);
        }
        assert_eq!(session.cache.len(), tokens.len());
        for layer in 0..2 {
            for t in 0..tokens.len() {
                close(session.cache.key(layer, t), stepped.cache.key(layer, t), 1e-5);
                close(session.cache.value(layer, t), stepped.cache.value(layer, t), 1e-5);
            }
        }
    }

    // FR-1, FR-40, Spec §7: independent of the one-token route, the batched pass over the three
    // tokens of the numpy reference above gives that reference's logits, both layers' outputs
    // and final-norm output at every position (the same values as the one-token test).
    #[test]
    fn prefill_matches_the_independent_reference() {
        let model = tiny_model(8);
        let mut session = model.session(true);
        let got = session.prefill(&[2, 0, 3], Logits::All).unwrap();
        let logits = [
            [-0.478599321, 0.556913018, 1.654692089, -0.721665280],
            [0.414470565, -0.232617106, 1.413414301, -0.064865065],
            [-0.181693408, -0.631061703, 1.246354816, -0.133866666],
        ];
        let layers = [
            [[-0.276911449, -2.182372800, 0.893603523, 0.488145256], [0.296345671, -2.282515693, 0.286672257, 0.585264398]],
            [[-0.470090844, -0.641421639, -0.072362666, 0.219245127], [-0.378449033, -0.664430486, -0.183649913, 0.224889104]],
            [[-0.128820170, -0.352348034, -0.009600638, 0.576087067], [-0.288078163, -0.350019463, 0.215527230, 0.645147227]],
        ];
        let final_norm = [
            [0.309704445, -2.862489287, 0.239675976, 0.611647153],
            [-1.156706917, -2.436950627, -0.449052014, 0.687360147],
            [-0.881036694, -1.284567999, 0.527321882, 1.973070001],
        ];
        let capture = session.capture().unwrap();
        for t in 0..3 {
            let row = t * 4..(t + 1) * 4;
            close(&got[row.clone()], &logits[t], 1e-5);
            close(&capture.layers[0][row.clone()], &layers[t][0], 1e-5);
            close(&capture.layers[1][row.clone()], &layers[t][1], 1e-5);
            close(&capture.final_norm[row], &final_norm[t], 1e-5);
        }
    }

    // FR-40, Spec §7: the mask is causal. Changing a token after position `t` leaves the logits
    // at positions up to `t` bit for bit equal; the changed position itself differs.
    #[test]
    fn prefill_is_causal() {
        let model = tiny_model(8);
        let mut a = model.session(false);
        let mut b = model.session(false);
        let one = a.prefill(&[2, 0, 3, 1], Logits::All).unwrap();
        let two = b.prefill(&[2, 0, 1, 3], Logits::All).unwrap();
        let same = |x: &[f32], y: &[f32]| x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits());
        assert!(same(&one[..2 * 4], &two[..2 * 4]), "positions 0 and 1 cannot see position 2");
        assert!(!same(&one[2 * 4..3 * 4], &two[2 * 4..3 * 4]), "position 2 reads its own token");
    }

    // FR-40, Spec §7: a prompt of one token is a pass of one row, and equals `step`.
    #[test]
    fn prefill_of_one_token_equals_step() {
        let model = tiny_model(8);
        let want = model.session(false).step(3).unwrap();
        for logits in [Logits::Last, Logits::All] {
            let got = model.session(false).prefill(&[3], logits).unwrap();
            close(&got, &want, 1e-5);
        }
    }

    // FR-40, FR-18, Spec §7: a prompt that fills the context exactly is processed, and the next
    // token is then rejected as `step` rejects it.
    #[test]
    fn prefill_can_fill_the_context_exactly() {
        let model = tiny_model(4);
        let mut session = model.session(false);
        session.prefill(&[1, 2, 3, 0], Logits::Last).unwrap();
        assert_eq!(session.cache.len(), 4);
        assert!(matches!(session.step(1), Err(Error::Rejected(_))));
    }

    // FR-40, FR-46, Spec §7: `Last` is the last row of `All` within rounding (its unembedding is a matrix-vector
    // product with 16 partial sums and `All`'s a blocked product, so since 0.4.2 not bit for bit), and its
    // capture's final norm is that row only.
    #[test]
    fn last_logits_are_the_last_row_of_all() {
        let model = tiny_model(8);
        let tokens = [2u32, 0, 3];
        let all = model.session(false).prefill(&tokens, Logits::All).unwrap();
        let mut session = model.session(true);
        let last = session.prefill(&tokens, Logits::Last).unwrap();
        assert_eq!(last.len(), 4);
        close(&last, &all[2 * 4..], 1e-5);
        let capture = session.capture().unwrap();
        assert_eq!(capture.final_norm.len(), 4, "the final norm of the last row only");
        assert_eq!(capture.embedding.len(), 3 * 4, "the embedding and layers cover every row");
        assert_eq!(capture.layers[1].len(), 3 * 4);
    }

    // FR-40, Spec §7: after a batched pass, decoding continues from the cache it filled exactly
    // as it would after one-token steps.
    #[test]
    fn decoding_continues_from_a_prefilled_cache() {
        let model = tiny_model(8);
        let (mut stepped, _, _) = by_steps(&model, &[2, 0, 3]);
        let mut session = model.session(false);
        session.prefill(&[2, 0, 3], Logits::Last).unwrap();
        for token in [1u32, 2] {
            close(&session.step(token).unwrap(), &stepped.step(token).unwrap(), 1e-5);
        }
    }

    // HC-4, FR-40, Spec §17: every fault changes what the batched pass computes (so the harness,
    // which reaches the engine through it, has something to notice), and with none injected the
    // feature changes nothing.
    #[cfg(feature = "negative-controls")]
    #[test]
    fn each_fault_changes_the_batched_pass() {
        use crate::Fault;
        let tokens = [2u32, 0, 3, 1];
        let faults = [
            Fault::InterleavedRope,
            Fault::PerturbedLogit { position: 2 },
            Fault::NoFinalNorm,
            Fault::UnmaskedAttention,
        ];
        // The head dimension is 4, not the tiny model's 2: with two values per head the
        // interleaved and the half-split pairing are the same pairing, so that fault is
        // invisible there.
        let honest = wide_head_model().session(false).prefill(&tokens, Logits::All).unwrap();
        for fault in faults {
            let mut model = wide_head_model();
            model.inject_fault(fault);
            let got = model.session(false).prefill(&tokens, Logits::All).unwrap();
            assert_ne!(got, honest, "{fault:?} changes the logits");
        }
    }

    /// A tiny model with a head dimension of 4 (hidden 8, two heads over one key-value head).
    fn wide_head_model() -> Model {
        let mut seed = 30;
        let mut next = || {
            seed += 1;
            seed - 1
        };
        let layers = (0..2)
            .map(|_| LayerWeights {
                attn_norm: norm(8, next()),
                q_proj: matrix(8, 8, next()),
                k_proj: matrix(4, 8, next()),
                v_proj: matrix(4, 8, next()),
                o_proj: matrix(8, 8, next()),
                mlp_norm: norm(8, next()),
                gate_proj: matrix(3, 8, next()),
                up_proj: matrix(3, 8, next()),
                down_proj: matrix(8, 3, next()),
            })
            .collect();
        let weights = Weights { embed_tokens: matrix(4, 8, 1), layers, final_norm: norm(8, next()) };
        let config = Config { hidden_size: 8, head_dim: 4, ..tiny_config(8) };
        Model::from_parts(config, weights)
    }

    // HC-4, FR-40, Spec §17: the unmasked fault is a missing causal mask, so row 0 now sees the whole
    // prompt: its logits depend on a later token, which the honest pass never allows.
    #[cfg(feature = "negative-controls")]
    #[test]
    fn the_unmasked_fault_lets_a_row_see_the_future() {
        use crate::Fault;
        let mut model = tiny_model(8);
        model.inject_fault(Fault::UnmaskedAttention);
        let one = model.session(false).prefill(&[2, 0, 3, 1], Logits::All).unwrap();
        let two = model.session(false).prefill(&[2, 0, 1, 3], Logits::All).unwrap();
        assert_ne!(&one[..4], &two[..4], "row 0 depends on tokens after it");
    }

    // HC-4, Spec §17: a perturbed logit lands on the row of its position, and only there, whether
    // every row is asked for or the last alone.
    #[cfg(feature = "negative-controls")]
    #[test]
    fn a_perturbed_logit_lands_on_its_own_row() {
        use crate::Fault;
        let tokens = [2u32, 0, 3, 1];
        let honest = tiny_model(8).session(false).prefill(&tokens, Logits::All).unwrap();
        let mut model = tiny_model(8);
        model.inject_fault(Fault::PerturbedLogit { position: 2 });
        let all = model.session(false).prefill(&tokens, Logits::All).unwrap();
        for t in 0..4 {
            let (a, b) = (&all[t * 4..(t + 1) * 4], &honest[t * 4..(t + 1) * 4]);
            assert_eq!(a != b, t == 2, "row {t}");
        }
        assert!((all[2 * 4] - honest[2 * 4] - 2e-3).abs() < 1e-6);
        let mut last = tiny_model(8);
        last.inject_fault(Fault::PerturbedLogit { position: 3 });
        let got = last.session(false).prefill(&tokens, Logits::Last).unwrap();
        assert!((got[0] - honest[3 * 4] - 2e-3).abs() < 1e-6, "the last row is position 3");
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

    // Spec §14: each head's most heavily weighted position, the earliest on a tie, and the
    // attention output is the same whether it is asked for or not. Three query heads over one
    // key-value head: keys at positions 0, 1, 2 are [0, 0], [1, 0], [0, 1]; head 0 asks for
    // [1, 0] (position 1), head 1 for [0, 1] (position 2), head 2 for nothing (all equal: 0).
    #[test]
    fn attention_reports_the_most_attended_position_per_head() {
        let mut cache = KvCache::new(1, 1, 2, 8);
        for key in [[0., 0.], [1., 0.], [0., 1.]] {
            cache.append(0, &key, &[1., 2.]).unwrap();
        }
        let q = [4., 0., 0., 4., 0., 0.];
        let (mut out, mut plain) = ([0f32; 6], [0f32; 6]);
        let mut most = [9usize; 3];
        attention(&q, &cache, 0, 3, 3, 1, 2, &mut out, Some(&mut most));
        attention(&q, &cache, 0, 3, 3, 1, 2, &mut plain, None);
        assert_eq!(most, [1, 2, 0]);
        assert_eq!(out, plain);
    }

    // S-8, FR-23, Spec §14: the logits and capture are bit-identical with no subscriber and at
    // every level, and the trace values are what they claim: one event per layer per step, the
    // norm after the MLP is the norm of the layer's output, and every head names a position
    // already processed.
    #[test]
    fn trace_statistics_are_right_and_change_nothing() {
        use crate::test_log::collect;
        use tracing::Level;

        let model = tiny_model(8);
        let run = |level| {
            collect(level, || {
                let mut session = model.session(true);
                [2, 0, 3].map(|t| (session.step(t).unwrap(), session.capture().unwrap().clone()))
            })
        };
        let (plain, _) = run(None);
        for level in [Level::WARN, Level::DEBUG, Level::TRACE] {
            let (steps, events) = run(Some(level));
            for ((logits, capture), (want_logits, want_capture)) in steps.iter().zip(&plain) {
                assert_eq!(logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want_logits.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{level}");
                assert_eq!(capture.layers, want_capture.layers, "{level}");
            }
            if level != Level::TRACE {
                assert!(events.is_empty(), "{level}: {events:?}");
                continue;
            }
            assert_eq!(events.len(), 3 * 2, "one per layer per step");
            for (n, event) in events.iter().enumerate() {
                let (pos, layer) = (n / 2, n % 2);
                assert_eq!(event.path, [format!("layer{{idx={layer}}}")]);
                let after_mlp: f32 = event.get("residual_norm_after_mlp").unwrap().parse().unwrap();
                assert_eq!(after_mlp, super::norm(&plain[pos].1.layers[layer]));
                let after_attention: f32 = event.get("residual_norm_after_attention").unwrap().parse().unwrap();
                assert!(after_attention > 0.0 && after_attention != after_mlp, "{event:?}");
                let heads: Vec<usize> = event.get("most_attended").unwrap().trim_matches(['[', ']']).split(", ").map(|h| h.parse().unwrap()).collect();
                assert_eq!(heads.len(), 2, "one per head");
                assert!(heads.iter().all(|&h| h <= pos), "{heads:?} at position {pos}");
            }
        }
    }

    // S-8, Spec §14 (measurement, not a check): the real model's time per step, to show that
    // logging costs nothing visible when no subscriber is listening. Run with
    //   cargo test -p emberly-oxygen --lib -- --ignored --nocapture step_speed
    // Five rounds of 128 steps each from a fresh session; prints the median and the range.
    #[test]
    #[ignore = "a measurement: run with --ignored --nocapture"]
    fn step_speed() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        let model = Model::load(&dir).unwrap();
        let mut per_step: Vec<f64> = (0..5)
            .map(|_| {
                let mut session = model.session(false);
                let start = std::time::Instant::now();
                for token in 0..128u32 {
                    std::hint::black_box(session.step(1000 + token * 7).unwrap());
                }
                start.elapsed().as_secs_f64() * 1000.0 / 128.0
            })
            .collect();
        per_step.sort_by(f64::total_cmp);
        println!("ms per step over 128 positions: median {:.2}, min {:.2}, max {:.2}", per_step[2], per_step[0], per_step[4]);
    }

    /// A cache of `positions` random positions for one layer, and a random query row.
    fn random_attention_case(positions: usize, heads: usize, kv_heads: usize, head_dim: usize, seed: u64) -> (KvCache, Vec<f32>) {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
        };
        let width = kv_heads * head_dim;
        let mut cache = KvCache::new(1, kv_heads, head_dim, positions + 1);
        for _ in 0..positions {
            let k: Vec<f32> = (0..width).map(|_| next()).collect();
            let v: Vec<f32> = (0..width).map(|_| next()).collect();
            cache.append(0, &k, &v).unwrap();
        }
        let q: Vec<f32> = (0..heads * head_dim * positions.max(1)).map(|_| next()).collect();
        (cache, q)
    }

    // FR-43, Spec §4: threaded decode attention equals the single-threaded one exactly at every
    // count, for the model's shape (9 heads over 3, head dimension 64) and an odd one.
    #[test]
    fn pooled_attention_equals_attention_exactly_at_every_count() {
        for &(heads, kv_heads, head_dim) in &[(9, 3, 64), (6, 2, 4), (4, 4, 3)] {
            for positions in [1, 2, 8, 300] {
                let (cache, q) = random_attention_case(positions, heads, kv_heads, head_dim, positions as u64 + heads as u64);
                let q = &q[..heads * head_dim];
                let mut want = vec![0.0; heads * head_dim];
                attention(q, &cache, 0, positions, heads, kv_heads, head_dim, &mut want, None);
                for threads in [1, 2, 3, 7, 16] {
                    let pool = Pool::new(threads).unwrap();
                    for min in [0, usize::MAX] {
                        let mut got = vec![f32::NAN; heads * head_dim];
                        attention_pooled(&pool, q, &cache, 0, positions, heads, kv_heads, head_dim, &mut got, min);
                        assert_eq!(got, want, "threads {threads} positions {positions} heads {heads} min {min}");
                    }
                }
            }
        }
    }

    // Spec §3, §4: workers hand back their handles on the cache before a call returns, so the
    // next append extends it in place and does not copy.
    #[test]
    fn workers_do_not_keep_the_cache_after_a_call() {
        let pool = Pool::new(4).unwrap();
        let mut out = vec![0.0; 9 * 8];
        for round in 0..50 {
            let (mut cache, q) = random_attention_case(40, 9, 3, 8, round);
            attention_pooled(&pool, &q[..72], &cache, 0, cache.len(), 9, 3, 8, &mut out, 0);
            assert_eq!(cache.handles(0), 1, "a worker still holds the cache");
            let mut rows = vec![0.0; 40 * 72];
            attention_rows_pooled(&pool, &q[..40 * 72], &cache, 0, 40, false, 9, 3, 8, &mut rows);
            assert_eq!(cache.handles(0), 1, "a worker still holds the cache after the row pass");
            cache.append(0, &[0.0; 24], &[0.0; 24]).unwrap();
            assert_eq!(cache.handles(0), 1);
        }
    }

    // FR-43, FR-41, Spec §7: threaded prompt attention equals the row-by-row one exactly at every
    // count, masked and with the unmasked fault.
    #[test]
    fn pooled_row_attention_equals_the_row_loop_exactly() {
        let (heads, kv_heads, head_dim) = (9, 3, 8);
        let q_width = heads * head_dim;
        for n in [1, 2, 5, 37] {
            let (cache, q) = random_attention_case(n, heads, kv_heads, head_dim, 90 + n as u64);
            for unmasked in [false, true] {
                let mut want = vec![0.0; n * q_width];
                for t in 0..n {
                    let positions = if unmasked { n } else { t + 1 };
                    attention(&q[t * q_width..(t + 1) * q_width], &cache, 0, positions, heads, kv_heads, head_dim, &mut want[t * q_width..(t + 1) * q_width], None);
                }
                for threads in [1, 2, 3, 7, 16] {
                    let pool = Pool::new(threads).unwrap();
                    let mut got = vec![f32::NAN; n * q_width];
                    attention_rows_pooled(&pool, &q[..n * q_width], &cache, 0, n, unmasked, heads, kv_heads, head_dim, &mut got);
                    assert_eq!(got, want, "threads {threads} n {n} unmasked {unmasked}");
                }
            }
        }
    }
}
