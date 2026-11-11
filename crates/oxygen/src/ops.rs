//! The tensor operations the forward pass needs (Spec §4). Each is a plain
//! loop over slices with f32 accumulation in index order, except the
//! matrix-vector product, which keeps independent partial sums (FR-25); none
//! uses `unsafe`, SIMD intrinsics or threads (HC-1). The reference sums in a
//! different order, so results differ in the last bits; S-1 and S-2 bound
//! that difference.

use std::sync::Arc;

use crate::pool::{Job, Pool};
use crate::Tensor;

/// How many independent partial sums the engine's matrix-vector product keeps
/// (Spec §4). 16, chosen by the width sweep of release 0.1.1 (plan §4): of 4,
/// 8, 16 and 32 it had the lowest estimated kernel time per token in two runs,
/// 20% under 8 sums. Tune with use; `tests/kernel.rs` pins the value.
pub(crate) const LANES: usize = 16;

/// `output[i]` = dot product of weight row `i` with `input`, for a weight in
/// the published `[out, in]` layout (Spec §3), so each row is contiguous.
/// Computed with [`LANES`] partial sums, see [`matvec_lanes`].
///
/// Panics if the weight is not 2-D or the lengths disagree: the engine builds
/// every call from the configuration, so a mismatch is an internal bug and
/// never the effect of user input (Spec §14).
pub(crate) fn matvec(weight: &Tensor, input: &[f32], output: &mut [f32]) {
    matvec_lanes::<LANES>(weight, input, output);
}

/// The matrix-vector product with `N` independent f32 partial sums per output
/// element (FR-25, Spec §4). In index order, every add waits for the one
/// before it; `N` separate sums have no such chain, so the processor can work
/// on them at once and the compiler can keep them in vector registers.
///
/// Sum `j` adds the products whose index is `j` modulo `N`, each in index
/// order. The sums are then combined sequentially, lane 0 first
/// (`((l0 + l1) + l2) + ...`), and the last `len mod N` products are added in
/// index order after them. The order is fixed, so a call is bit-identical to
/// the next (S-6), and `tests/kernel.rs` pins it with hand-worked cases.
///
/// Panics as [`matvec`] does.
pub(crate) fn matvec_lanes<const N: usize>(weight: &Tensor, input: &[f32], output: &mut [f32]) {
    let &[rows, cols] = weight.shape() else {
        panic!("matvec needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), cols, "input length must equal the weight's columns");
    assert_eq!(output.len(), rows, "output length must equal the weight's rows");
    for (i, out) in output.iter_mut().enumerate() {
        *out = dot_lanes::<N>(weight.row(i), input);
    }
}

/// One dot product with `N` independent partial sums, in the order described
/// at [`matvec_lanes`]. The matrix-vector and matrix-matrix products both call
/// it, so there is one summation order (Spec §4).
#[inline(always)]
fn dot_lanes<const N: usize>(row: &[f32], input: &[f32]) -> f32 {
    let mut sums = [0f32; N];
    let mut row = row.chunks_exact(N);
    let mut x = input.chunks_exact(N);
    for (w, x) in row.by_ref().zip(x.by_ref()) {
        for j in 0..N {
            sums[j] += w[j] * x[j];
        }
    }
    let mut sum = 0f32;
    for lane in sums {
        sum += lane;
    }
    for (w, x) in row.remainder().iter().zip(x.remainder()) {
        sum += w * x;
    }
    sum
}

/// The matrix-matrix product for a batch of inputs (FR-40, Spec §4): `input`
/// holds `count` rows of the weight's column count, and `output` receives
/// `count` rows of the weight's row count, so `output[t][i]` is the dot
/// product of weight row `i` with input row `t`.
///
/// The loops put the weight row outermost: each row is read from memory once
/// and reused for every input while it is in cache, where `count` separate
/// matrix-vector products would read it `count` times. Each element is the
/// same [`LANES`]-sum dot product as [`matvec`]'s, so the result equals it bit
/// for bit. That reuse is the whole technique: no input block is held in
/// registers and nothing is tiled, threaded or vectorized by hand.
///
/// Panics if the weight is not 2-D or the lengths disagree (internal bug,
/// Spec §14).
#[cfg(any(test, feature = "bench-kernels"))]
pub(crate) fn matmul(weight: &Tensor, input: &[f32], count: usize, output: &mut [f32]) {
    let &[rows, cols] = weight.shape() else {
        panic!("matmul needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), count * cols, "input length must be count times the weight's columns");
    assert_eq!(output.len(), count * rows, "output length must be count times the weight's rows");
    for i in 0..rows {
        let row = weight.row(i);
        for t in 0..count {
            output[t * rows + i] = dot_lanes::<LANES>(row, &input[t * cols..(t + 1) * cols]);
        }
    }
}

/// Below this many multiply-adds (rows × columns × inputs) a product runs on
/// the calling thread: handing it to workers costs more than it saves. From
/// the measurements of plan 0.4.1 §5 (task 20.1): a 192×576 or 576×576 product
/// is not helped by threads, a 1536×576 one is. Initial; tune with use.
pub(crate) const MIN_THREADED_MADDS: usize = 500_000;

/// Whether a product of this size is divided among the pool's threads.
pub(crate) fn threads_pay(rows: usize, cols: usize, count: usize, threads: usize, min_madds: usize) -> bool {
    threads > 1 && rows * cols * count >= min_madds
}

/// [`matvec`] with the weight's rows divided among the pool's threads
/// (FR-43). Equal to [`matvec`] bit for bit at every thread count.
pub(crate) fn matvec_pooled(pool: &Pool, weight: &Tensor, input: &[f32], output: &mut [f32]) {
    matvec_pooled_min(pool, weight, input, output, MIN_THREADED_MADDS);
}

pub(crate) fn matvec_pooled_min(pool: &Pool, weight: &Tensor, input: &[f32], output: &mut [f32], min_madds: usize) {
    let &[rows, cols] = weight.shape() else {
        panic!("matvec needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), cols, "input length must equal the weight's columns");
    assert_eq!(output.len(), rows, "output length must equal the weight's rows");
    if !threads_pay(rows, cols, 1, pool.threads(), min_madds) {
        return matvec(weight, input, output);
    }
    // Each worker owns whole output elements and calls the same dot product,
    // so no sum is split and the result does not depend on the thread count.
    let (weight, input) = (weight.clone(), Arc::new(input.to_vec()));
    let job: Job = Arc::new(move |range| range.map(|i| dot_lanes::<LANES>(weight.row(i), &input)).collect());
    output.copy_from_slice(&pool.run(rows, &job));
}

/// [`matmul`] with the weight's rows divided among the pool's threads (FR-43).
#[cfg(any(test, feature = "bench-kernels"))]
pub(crate) fn matmul_pooled(pool: &Pool, weight: &Tensor, input: &[f32], count: usize, output: &mut [f32]) {
    matmul_pooled_min(pool, weight, input, count, output, MIN_THREADED_MADDS);
}

#[cfg(any(test, feature = "bench-kernels"))]
pub(crate) fn matmul_pooled_min(pool: &Pool, weight: &Tensor, input: &[f32], count: usize, output: &mut [f32], min_madds: usize) {
    let &[rows, cols] = weight.shape() else {
        panic!("matmul needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), count * cols, "input length must be count times the weight's columns");
    assert_eq!(output.len(), count * rows, "output length must be count times the weight's rows");
    if !threads_pay(rows, cols, count, pool.threads(), min_madds) {
        return matmul(weight, input, count, output);
    }
    // A worker's range of weight rows gives `count` outputs per row, row by
    // row; the pool joins the ranges in order and the result is laid out as the
    // single-threaded product's `[count][rows]`.
    let (weight, input) = (weight.clone(), Arc::new(input.to_vec()));
    let job: Job = Arc::new(move |range| {
        let mut part = Vec::with_capacity(range.len() * count);
        for i in range {
            let row = weight.row(i);
            part.extend((0..count).map(|t| dot_lanes::<LANES>(row, &input[t * cols..(t + 1) * cols])));
        }
        part
    });
    let by_row = pool.run(rows, &job);
    for i in 0..rows {
        for t in 0..count {
            output[t * rows + i] = by_row[i * count + t];
        }
    }
}

/// Weight rows per tile of the blocked product (FR-46, Spec §4). Initial; the
/// profile of plan 0.4.2 §5 found only some tile shapes vectorize, so this is
/// pinned by measurement and not chosen freely.
pub(crate) const TILE_ROWS: usize = 4;

/// Input rows per tile of the blocked product, and so the width of a packed
/// input panel (initial; see [`TILE_ROWS`]).
pub(crate) const TILE_INPUTS: usize = 16;

/// Products summed before the running total takes their sum (FR-46, Spec §4).
/// One long sequential sum accumulates more rounding than the 16 partial sums
/// of [`matmul`] did (measured on the real model: worst logit error against the
/// float64 reference 6.5e-4 against 3.5e-4); adding the sums of chunks of 64
/// brought it back for about 3% of the kernel's speed (plan 0.4.2 §5). Initial;
/// tune with use.
pub(crate) const TILE_DEPTH: usize = 64;

/// [`matmul`] in register tiles (FR-46, FR-47, Spec §4): each output element
/// is a sum over the shared dimension in index order, taken in chunks of
/// [`TILE_DEPTH`] products: each chunk is summed with fused multiply-adds
/// (rounded once per product) and the chunk sums are added in order. So an
/// element's value does not depend on the tile it falls in, on the tile shape
/// or on how rows are divided among threads. The result differs from [`matmul`]'s in the last bits (its sums are
/// 16 partial sums with a separate multiply and add).
///
/// Panics as [`matmul`] does.
pub(crate) fn matmul_blocked(weight: &Tensor, input: &[f32], count: usize, output: &mut [f32]) {
    let (rows, cols) = blocked_shape(weight, input, count, output);
    let packed = pack_inputs(input, count, cols);
    let by_row = blocked_rows(weight, 0..rows, &packed, input, count);
    transpose_into(&by_row, rows, count, output);
}

/// Checks a blocked product's sizes (the engine builds every call from the
/// configuration, so a mismatch is an internal bug, Spec §14) and returns
/// the weight's rows and columns.
fn blocked_shape(weight: &Tensor, input: &[f32], count: usize, output: &[f32]) -> (usize, usize) {
    let &[rows, cols] = weight.shape() else {
        panic!("matmul needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), count * cols, "input length must be count times the weight's columns");
    assert_eq!(output.len(), count * rows, "output length must be count times the weight's rows");
    (rows, cols)
}

/// Packs the full panels of [`TILE_INPUTS`] input rows into `[k][TILE_INPUTS]`
/// order: panel `p`, position `k`, lane `j` holds element `k` of input row
/// `p * TILE_INPUTS + j`, so the tile reads the inputs it needs for one weight
/// value as one contiguous run. The leftover inputs (`count mod TILE_INPUTS`)
/// are not packed; they are summed from `input` directly.
fn pack_inputs(input: &[f32], count: usize, cols: usize) -> Vec<f32> {
    let panels = count / TILE_INPUTS;
    let mut packed = vec![0f32; panels * cols * TILE_INPUTS];
    for p in 0..panels {
        for k in 0..cols {
            for j in 0..TILE_INPUTS {
                packed[(p * cols + k) * TILE_INPUTS + j] = input[(p * TILE_INPUTS + j) * cols + k];
            }
        }
    }
    packed
}

/// One tile: `R` weight rows against one packed panel. Every accumulator is a
/// sum over one chunk of `k` in index order with a fused multiply-add; the chunk
/// sums go into the running totals in order. The shape (4 rows by 16 inputs,
/// and one row for the leftovers) is the one the compiler turns into vector
/// fused multiply-adds from this safe code (plan 0.4.2 §5).
#[inline(always)]
fn tile<const R: usize>(rows: [&[f32]; R], panel: &[f32], cols: usize) -> [[f32; TILE_INPUTS]; R] {
    let mut total = [[0f32; TILE_INPUTS]; R];
    let mut start = 0;
    while start < cols {
        let end = (start + TILE_DEPTH).min(cols);
        let mut acc = [[0f32; TILE_INPUTS]; R];
        for k in start..end {
            let x: &[f32; TILE_INPUTS] = panel[k * TILE_INPUTS..(k + 1) * TILE_INPUTS].try_into().expect("a panel row");
            for i in 0..R {
                let w = rows[i][k];
                for j in 0..TILE_INPUTS {
                    acc[i][j] = w.mul_add(x[j], acc[i][j]);
                }
            }
        }
        for i in 0..R {
            for j in 0..TILE_INPUTS {
                total[i][j] += acc[i][j];
            }
        }
        start = end;
    }
    total
}

/// One element by the same chunked sum as the tile's, for the inputs that do not fill a panel.
#[inline(always)]
fn fused_dot(row: &[f32], input: &[f32]) -> f32 {
    let mut total = 0f32;
    for (w, x) in row.chunks(TILE_DEPTH).zip(input.chunks(TILE_DEPTH)) {
        let mut chunk = 0f32;
        for (w, x) in w.iter().zip(x) {
            chunk = w.mul_add(*x, chunk);
        }
        total += chunk;
    }
    total
}

/// The blocked product for the weight rows in `range`, written `[row][input]`
/// (row-major by weight row, so ranges join by concatenation). Rows are taken
/// four at a time, then one at a time for what is left; the leftover inputs are
/// summed singly. Every element is the same index-order fused sum whichever
/// path computes it, so the grouping changes no bit.
fn blocked_rows(weight: &Tensor, range: std::ops::Range<usize>, packed: &[f32], input: &[f32], count: usize) -> Vec<f32> {
    let cols = weight.shape()[1];
    let panels = count / TILE_INPUTS;
    let singles = panels * TILE_INPUTS..count;
    let first = range.start;
    let mut out = vec![0f32; range.len() * count];
    let mut i = range.start;
    while i < range.end {
        let take = if range.end - i >= TILE_ROWS { TILE_ROWS } else { 1 };
        for p in 0..panels {
            let panel = &packed[p * cols * TILE_INPUTS..(p + 1) * cols * TILE_INPUTS];
            if take == TILE_ROWS {
                let acc = tile::<TILE_ROWS>(std::array::from_fn(|r| weight.row(i + r)), panel, cols);
                for (r, lanes) in acc.iter().enumerate() {
                    let at = (i - first + r) * count + p * TILE_INPUTS;
                    out[at..at + TILE_INPUTS].copy_from_slice(lanes);
                }
            } else {
                let acc = tile::<1>([weight.row(i)], panel, cols);
                let at = (i - first) * count + p * TILE_INPUTS;
                out[at..at + TILE_INPUTS].copy_from_slice(&acc[0]);
            }
        }
        for t in singles.clone() {
            for r in 0..take {
                out[(i - first + r) * count + t] = fused_dot(weight.row(i + r), &input[t * cols..(t + 1) * cols]);
            }
        }
        i += take;
    }
    out
}

/// Puts `[row][input]` results into the products' `[input][row]` layout.
fn transpose_into(by_row: &[f32], rows: usize, count: usize, output: &mut [f32]) {
    for i in 0..rows {
        for t in 0..count {
            output[t * rows + i] = by_row[i * count + t];
        }
    }
}

/// [`matmul_blocked`] with the weight's rows divided among the pool's threads,
/// in whole tiles where the size allows (FR-46, FR-43). Equal to it bit for bit
/// at every thread count.
pub(crate) fn matmul_blocked_pooled(pool: &Pool, weight: &Tensor, input: &[f32], count: usize, output: &mut [f32]) {
    matmul_blocked_pooled_min(pool, weight, input, count, output, MIN_THREADED_MADDS);
}

pub(crate) fn matmul_blocked_pooled_min(pool: &Pool, weight: &Tensor, input: &[f32], count: usize, output: &mut [f32], min_madds: usize) {
    matmul_blocked_threaded(pool, weight, input, count, output, min_madds);
    // The negative control (Spec §17): the inputs outside the whole panels are left unwritten.
    #[cfg(feature = "negative-controls")]
    if pool.tiles_unfinished() {
        let rows = weight.shape()[0];
        for t in count / TILE_INPUTS * TILE_INPUTS..count {
            output[t * rows..(t + 1) * rows].fill(0.0);
        }
    }
}

fn matmul_blocked_threaded(pool: &Pool, weight: &Tensor, input: &[f32], count: usize, output: &mut [f32], min_madds: usize) {
    let (rows, cols) = blocked_shape(weight, input, count, output);
    if !threads_pay(rows, cols, count, pool.threads(), min_madds) {
        return matmul_blocked(weight, input, count, output);
    }
    // The pool divides tile groups, so a range is a whole number of tiles except at
    // the last row; the packed input is made once and shared with the workers.
    let groups = rows.div_ceil(TILE_ROWS);
    let (weight, input, packed) = (weight.clone(), Arc::new(input.to_vec()), Arc::new(pack_inputs(input, count, cols)));
    let job: Job = Arc::new(move |range| {
        blocked_rows(&weight, range.start * TILE_ROWS..(range.end * TILE_ROWS).min(rows), &packed, &input, count)
    });
    transpose_into(&pool.run(groups, &job), rows, count, output);
}

/// RMSNorm: `weight * (x / sqrt(mean(x^2) + eps))`. The mean of squares is a
/// sum in index order divided by the length, and the reciprocal square root
/// is taken once, then multiplied in, as the reference does in f32.
pub(crate) fn rms_norm(x: &[f32], weight: &[f32], eps: f32, out: &mut [f32]) {
    assert_eq!(weight.len(), x.len(), "weight length must equal the input's");
    assert_eq!(out.len(), x.len(), "output length must equal the input's");
    let mut sum = 0f32;
    for v in x {
        sum += v * v;
    }
    let scale = 1.0 / (sum / x.len() as f32 + eps).sqrt();
    for ((o, v), w) in out.iter_mut().zip(x).zip(weight) {
        *o = w * (v * scale);
    }
}

/// Softmax in place. The maximum is subtracted first, so a large score cannot
/// overflow `exp`, and a very small one underflows to 0 instead of NaN.
/// Scores are finite: the engine never masks with infinity (attention only
/// reads cached positions).
pub(crate) fn softmax(x: &mut [f32]) {
    let Some(max) = x.iter().copied().reduce(f32::max) else {
        return;
    };
    let mut sum = 0f32;
    for v in x.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    for v in x.iter_mut() {
        *v /= sum;
    }
}

/// SiLU in place: `x * sigmoid(x) = x / (1 + exp(-x))`. For a large negative
/// `x`, `exp(-x)` is infinite and the result is -0, not NaN.
pub(crate) fn silu(x: &mut [f32]) {
    for v in x.iter_mut() {
        *v /= 1.0 + (-*v).exp();
    }
}

/// `a[i] += b[i]`.
pub(crate) fn add_assign(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "operands must have equal lengths");
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// `a[i] *= b[i]`.
pub(crate) fn mul_assign(a: &mut [f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "operands must have equal lengths");
    for (x, y) in a.iter_mut().zip(b) {
        *x *= y;
    }
}

/// The cosine and sine of each rotation angle at one position, one pair per
/// frequency: `position * theta^(-2i/head_dim)`, all in f32.
pub(crate) fn rope_angles(position: usize, head_dim: usize, theta: f32) -> Vec<(f32, f32)> {
    (0..head_dim / 2)
        .map(|i| {
            let frequency = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let angle = position as f32 * frequency;
            (angle.cos(), angle.sin())
        })
        .collect()
}

/// Rotary position embedding in place, for `heads` heads of `head_dim` values
/// at one position. Half-split pairing: dimension `i` rotates with dimension
/// `i + head_dim/2`, by the angle `position * theta^(-2i/head_dim)`. The
/// frequency, angle, sine and cosine are all f32, because the reference
/// computes them in f32 and a more precise angle is a different angle: at
/// position 5000 the f32 and f64 cosines differ by 8e-5 (Spec §4).
pub(crate) fn rope(x: &mut [f32], position: usize, heads: usize, head_dim: usize, theta: f32) {
    assert!(head_dim % 2 == 0, "head_dim must be even, got {head_dim}");
    assert_eq!(x.len(), heads * head_dim, "x must hold `heads` heads of `head_dim` values");
    let half = head_dim / 2;
    let angles = rope_angles(position, head_dim, theta);
    for head in x.chunks_exact_mut(head_dim) {
        let (first, second) = head.split_at_mut(half);
        for ((a, b), &(cos, sin)) in first.iter_mut().zip(second.iter_mut()).zip(&angles) {
            let (x1, x2) = (*a, *b);
            *a = x1 * cos - x2 * sin;
            *b = x2 * cos + x1 * sin;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;

    /// Deterministic values in [-1, 1) from a seed (xorshift), so no rand crate is needed.
    fn random(seed: u64, n: usize) -> Vec<f32> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn weight(rows: usize, cols: usize, seed: u64) -> Tensor {
        Tensor::new(vec![rows, cols], random(seed, rows * cols)).unwrap()
    }

    /// The same product by the test-only linear algebra crate (HC-2).
    fn reference(weight: &Tensor, input: &[f32]) -> Vec<f32> {
        let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
        let mut out = vec![0f32; rows];
        // out (rows x 1) = weight (rows x cols) * input (cols x 1)
        unsafe {
            matrixmultiply::sgemm(
                rows, cols, 1, 1.0,
                weight.data().as_ptr(), cols as isize, 1,
                input.as_ptr(), 1, 1,
                0.0,
                out.as_mut_ptr(), 1, 1,
            );
        }
        out
    }


    fn close(actual: &[f32], expected: &[f32], tolerance: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!((a - e).abs() <= tolerance, "element {i}: {a} vs {e} (tolerance {tolerance})");
        }
    }

    // Spec §4, §17: RMSNorm of [3, 4] has mean square 12.5, so the scale is 1/sqrt(12.5).
    #[test]
    fn rms_norm_hand_computed() {
        let mut out = [0f32; 2];
        rms_norm(&[3., 4.], &[1., 1.], 0.0, &mut out);
        close(&out, &[0.848_528_1, 1.131_370_9], 1e-6);
        rms_norm(&[3., 4.], &[2., 0.5], 0.0, &mut out);
        close(&out, &[1.697_056_3, 0.565_685_4], 1e-6);
    }

    // Spec §4, §17: with unit weights the output has unit mean square, whatever the input's scale.
    #[test]
    fn rms_norm_output_has_unit_rms() {
        for scale in [1e-3f32, 1.0, 1e3] {
            let x: Vec<f32> = random(5, 64).iter().map(|v| v * scale).collect();
            let mut out = vec![0f32; 64];
            rms_norm(&x, &vec![1.0; 64], 0.0, &mut out);
            let mean_square: f32 = out.iter().map(|v| v * v).sum::<f32>() / 64.0;
            assert!((mean_square - 1.0).abs() < 1e-5, "scale {scale}: {mean_square}");
        }
    }

    // Spec §4: eps is added inside the square root, so it shows, and it keeps a zero input finite.
    #[test]
    fn rms_norm_epsilon() {
        let mut out = [0f32; 1];
        rms_norm(&[1.], &[1.], 1.0, &mut out);
        close(&out, &[std::f32::consts::FRAC_1_SQRT_2], 1e-7); // 1 / sqrt(1 + 1)
        let mut out = [9f32; 3];
        rms_norm(&[0., 0., 0.], &[1., 2., 3.], 1e-5, &mut out);
        assert_eq!(out, [0., 0., 0.]);
        rms_norm(&[], &[], 1e-5, &mut []);
    }

    // Spec §4, §17: softmax of [1, 2, 3] is the textbook [0.0900, 0.2447, 0.6652]; it sums to 1.
    #[test]
    fn softmax_hand_computed() {
        let mut x = [1f32, 2., 3.];
        softmax(&mut x);
        close(&x, &[0.090_030_57, 0.244_728_47, 0.665_240_96], 1e-7);
        assert!((x.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let mut equal = [5f32, 5.];
        softmax(&mut equal);
        assert_eq!(equal, [0.5, 0.5]);
        let mut one = [-123f32];
        softmax(&mut one);
        assert_eq!(one, [1.0]);
        softmax(&mut []);
    }

    // Spec §4, §17: subtracting the maximum keeps large scores finite and small ones at 0.
    #[test]
    fn softmax_does_not_overflow() {
        let mut x = [1000f32, 1000., 999.];
        softmax(&mut x);
        close(&x, &[0.422_318_8, 0.422_318_8, 0.155_362_4], 1e-6);
        assert!(x.iter().all(|v| v.is_finite()));
        let mut x = [0f32, -1000.];
        softmax(&mut x);
        assert_eq!(x, [1.0, 0.0]);
        let mut x = [3e38f32, 3e38];
        softmax(&mut x);
        assert_eq!(x, [0.5, 0.5]);
    }

    // Spec §4: adding a constant to every score changes nothing, which is why the maximum can be removed.
    #[test]
    fn softmax_is_shift_invariant() {
        let base = random(9, 32);
        let mut a = base.clone();
        softmax(&mut a);
        for shift in [-500.0f32, 7.5, 500.0] {
            let mut b: Vec<f32> = base.iter().map(|v| v + shift).collect();
            softmax(&mut b);
            close(&b, &a, 1e-5);
        }
    }

    // Spec §4, §17: SiLU at 0, +-1, and far out on both sides.
    #[test]
    fn silu_hand_computed_and_limits() {
        let mut x = [0f32, 1., -1., 100., -1000., 1e30, -1e30];
        silu(&mut x);
        assert_eq!(x[0], 0.0);
        close(&x[1..3], &[0.731_058_6, -0.268_941_4], 1e-7);
        assert_eq!(x[3], 100.0, "large positive values pass through");
        assert!(x[4..].iter().all(|v| !v.is_nan()), "{x:?}");
        assert!(x[4].abs() < 1e-30 && x[6].abs() < 1e-30, "large negative values go to zero: {x:?}");
        assert_eq!(x[5], 1e30);
    }

    // Spec §4: elementwise operations on equal lengths; a mismatch is an internal bug.
    #[test]
    fn elementwise_add_and_multiply() {
        let mut a = [1f32, 2., 3.];
        add_assign(&mut a, &[10., 20., 30.]);
        assert_eq!(a, [11., 22., 33.]);
        mul_assign(&mut a, &[2., 0., -1.]);
        assert_eq!(a, [22., 0., -33.]);
        add_assign(&mut [], &[]);
        mul_assign(&mut [], &[]);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "equal lengths")]
    fn add_with_unequal_lengths_panics() {
        add_assign(&mut [1., 2.], &[1.]);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "equal lengths")]
    fn multiply_with_unequal_lengths_panics() {
        mul_assign(&mut [1.], &[1., 2.]);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "weight length")]
    fn rms_norm_with_a_short_weight_panics() {
        rms_norm(&[1., 2.], &[1.], 0.0, &mut [0.; 2]);
    }


    // Reference values: cosines and sines of the rotation angles at positions 1 and 5000, for
    // SmolLM2-135M's head_dim 64 and rope_theta 100000, from transformers 5.21.0's
    // LlamaRotaryEmbedding in float32 (torch 2.14.1). Regenerate with
    // `cd oracle && uv run python rope_reference.py`.
    const POS1_COS: [f32; 32] = [
        5.403023362e-01, 7.662379742e-01, 8.837559819e-01, 9.428144097e-01,
        9.720144868e-01, 9.863391519e-01, 9.933397770e-01, 9.967548251e-01,
        9.984192848e-01, 9.992301464e-01, 9.996250868e-01, 9.998174310e-01,
        9.999110699e-01, 9.999567270e-01, 9.999789000e-01, 9.999897480e-01,
        9.999949932e-01, 9.999975562e-01, 9.999988079e-01, 9.999994636e-01,
        9.999997020e-01, 9.999998808e-01, 9.999999404e-01, 1.000000000e+00,
        1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00,
        1.000000000e+00, 1.000000000e+00, 1.000000000e+00, 1.000000000e+00,
    ];
    const POS1_SIN: [f32; 32] = [
        8.414709568e-01, 6.425569057e-01, 4.679480791e-01, 3.333181739e-01,
        2.349210680e-01, 1.647274643e-01, 1.152217165e-01, 8.049702644e-02,
        5.620449409e-02, 3.923182935e-02, 2.738077380e-02, 1.910836808e-02,
        1.333482005e-02, 9.305586107e-03, 6.493770983e-03, 4.531568382e-03,
        3.162272507e-03, 2.206732286e-03, 1.539925928e-03, 1.074607600e-03,
        7.498941268e-04, 5.232990952e-04, 3.651741135e-04, 2.548296761e-04,
        1.778279402e-04, 1.240937709e-04, 8.659643208e-05, 6.042963651e-05,
        4.216965317e-05, 2.942727224e-05, 2.053524986e-05, 1.433012676e-05,
    ];
    const POS5000_COS: [f32; 32] = [
        1.546684057e-01, -4.024487436e-01, -9.946651459e-01, -8.787785769e-01,
        -2.613872588e-01, -3.904827535e-01, 7.887293696e-01, 6.987187266e-01,
        -1.888198312e-03, 1.395955384e-01, 2.587175369e-01, 2.676206231e-01,
        -7.631481886e-01, -8.279874921e-01, 4.948315620e-01, -7.858231068e-01,
        -9.946563244e-01, 3.808692470e-02, 1.537368596e-01, 6.136304140e-01,
        -8.208616376e-01, -8.652752042e-01, -2.523171902e-01, 2.923162580e-01,
        6.300802827e-01, 8.136059642e-01, 9.077185392e-01, 9.546994567e-01,
        9.778537154e-01, 9.891949892e-01, 9.947333932e-01, 9.974341989e-01,
    ];
    const POS5000_SIN: [f32; 32] = [
        -9.879664183e-01, 9.154425263e-01, -1.031561196e-01, 4.772297442e-01,
        -9.652340412e-01, -9.206102490e-01, -6.147406101e-01, 7.153964639e-01,
        -9.999982119e-01, 9.902086258e-01, -9.659530520e-01, 9.635243416e-01,
        -6.462234855e-01, 5.607466102e-01, 8.689889312e-01, -6.184512973e-01,
        -1.032414138e-01, -9.992744327e-01, 9.881118536e-01, -7.895933986e-01,
        -5.711271167e-01, 5.012971759e-01, 9.676445723e-01, 9.563217163e-01,
        7.765299678e-01, 5.814166665e-01, 4.195796251e-01, 2.975717783e-01,
        2.092894614e-01, 1.466060430e-01, 1.024959385e-01, 7.158934325e-02,
    ];

    /// Rotating `[1; 32] ++ [0; 32]` gives the cosines in the first half and the sines in the second.
    fn rotation_table(position: usize) -> (Vec<f32>, Vec<f32>) {
        let mut x = vec![0f32; 64];
        x[..32].fill(1.0);
        rope(&mut x, position, 1, 64, 100000.0);
        (x[..32].to_vec(), x[32..].to_vec())
    }

    // Spec §4, §17: the f32 angles reproduce the reference's cosines and sines, including at a
    // high position where an f64 angle would already differ by 8e-5.
    #[test]
    fn rope_matches_the_reference_rotation_table() {
        for (position, cos, sin) in [(1, POS1_COS, POS1_SIN), (5000, POS5000_COS, POS5000_SIN)] {
            let (our_cos, our_sin) = rotation_table(position);
            close(&our_cos, &cos, 1e-6);
            close(&our_sin, &sin, 1e-6);
        }
    }

    // Spec §4, §17: position 0 rotates by nothing, exactly.
    #[test]
    fn rope_is_the_identity_at_position_zero() {
        let original = random(21, 3 * 8);
        let mut x = original.clone();
        rope(&mut x, 0, 3, 8, 10000.0);
        assert_eq!(x, original);
    }

    // Spec §4, §17: hand-computed rotation at position 1 for head_dim 4, theta 10000. The two
    // frequencies are 10000^0 = 1 and 10000^(-1/2) = 0.01, so the pairs (x0, x2) and (x1, x3)
    // turn by 1 and 0.01 radians.
    #[test]
    fn rope_hand_computed_at_position_one() {
        let mut x = [1f32, 2., 3., 4.];
        rope(&mut x, 1, 1, 4, 10000.0);
        close(&x, &[-1.984_110_6, 1.959_900_7, 2.462_378_0, 4.019_799_7], 1e-6);
    }

    // Spec §4: the first pair turns by exactly `position` radians, since its frequency is 1.
    #[test]
    fn rope_first_pair_turns_by_the_position() {
        for position in [1usize, 2, 7, 100] {
            let mut x = [1f32, 0., 0., 0.];
            rope(&mut x, position, 1, 4, 10000.0);
            let angle = position as f64;
            close(&x[..1], &[angle.cos() as f32], 1e-5);
            close(&x[2..3], &[angle.sin() as f32], 1e-5);
        }
    }

    // Spec §4: a rotation never changes the length of a pair, and every head turns independently
    // by the same angles.
    #[test]
    fn rope_preserves_pair_lengths_and_treats_heads_alike() {
        let head = random(33, 8);
        let mut two = [head.clone(), head.clone()].concat();
        rope(&mut two, 9, 2, 8, 10000.0);
        assert_eq!(&two[..8], &two[8..], "identical heads stay identical");
        let mut one = head.clone();
        rope(&mut one, 9, 1, 8, 10000.0);
        assert_eq!(&two[..8], &one[..], "a head does not depend on its neighbour");
        for i in 0..4 {
            let before = head[i].hypot(head[i + 4]);
            let after = one[i].hypot(one[i + 4]);
            assert!((before - after).abs() < 1e-6, "pair {i}: {before} vs {after}");
        }
    }

    // Spec §14: a head_dim that cannot be paired, or a wrong length, is an internal bug.
    #[test]
    #[should_panic(expected = "even")]
    fn rope_with_odd_head_dim_panics() {
        rope(&mut [0.; 3], 1, 1, 3, 10000.0);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "heads")]
    fn rope_with_wrong_length_panics() {
        rope(&mut [0.; 7], 1, 2, 4, 10000.0);
    }

    // HC-1, HC-2, Spec §4: hand-computed 2x3 case.
    #[test]
    fn hand_computed_product() {
        let w = Tensor::new(vec![2, 3], vec![1., 2., 3., 4., 5., 6.]).unwrap();
        let mut out = [0f32; 2];
        matvec(&w, &[1., 0., -1.], &mut out);
        assert_eq!(out, [-2., -2.]);
        matvec(&w, &[1., 1., 1.], &mut out);
        assert_eq!(out, [6., 15.]);
    }

    // HC-1, Spec §4: the identity matrix returns its input, and rows pick out their own dot product.
    #[test]
    fn identity_and_row_order() {
        let n = 5;
        let mut data = vec![0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        let id = Tensor::new(vec![n, n], data).unwrap();
        let x = [3., -1., 4., -1., 5.];
        let mut out = [0f32; 5];
        matvec(&id, &x, &mut out);
        assert_eq!(out, x);
        // Output element i belongs to row i, not column i.
        let w = Tensor::new(vec![2, 2], vec![0., 1., 0., 0.]).unwrap();
        let mut out = [0f32; 2];
        matvec(&w, &[7., 9.], &mut out);
        assert_eq!(out, [9., 0.]);
    }

    // Spec §4, §14: empty dimensions are valid and give empty or zero outputs.
    #[test]
    fn zero_sized_edge_cases() {
        let none = Tensor::new(vec![0, 3], vec![]).unwrap();
        matvec(&none, &[1., 2., 3.], &mut []);
        let no_columns = Tensor::new(vec![2, 0], vec![]).unwrap();
        let mut out = [9f32; 2];
        matvec(&no_columns, &[], &mut out);
        assert_eq!(out, [0., 0.]);
    }

    // Spec §14: a length that disagrees with the weight is an internal bug and panics.
    #[test]
    #[should_panic(expected = "input length")]
    fn wrong_input_length_panics() {
        matvec(&weight(2, 3, 1), &[1., 2.], &mut [0.; 2]);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "output length")]
    fn wrong_output_length_panics() {
        matvec(&weight(2, 3, 1), &[1., 2., 3.], &mut [0.; 3]);
    }

    // Spec §4, §14: a length mismatch in a kernel call is an internal bug and panics.
    #[test]
    #[should_panic(expected = "2-D")]
    fn a_vector_is_not_a_weight() {
        let v = Tensor::new(vec![3], vec![1., 2., 3.]).unwrap();
        matvec(&v, &[1., 2., 3.], &mut [0.; 3]);
    }

    // HC-2, Spec §17: against the test-only linear algebra crate on random inputs, at every
    // shape the model uses. The bound is the standard worst case for summing k products in
    // any order, gamma_k * sum|w_i x_i| with gamma_k = k u / (1 - k u), u = 2^-24, applied to
    // both implementations: far below S-1's 1e-3 for values of this size.
    #[test]
    fn matches_the_linear_algebra_crate_at_every_model_shape() {
        let shapes = [(576, 576), (192, 576), (1536, 576), (576, 1536), (49152, 576)];
        for (n, (rows, cols)) in shapes.into_iter().enumerate() {
            let w = weight(rows, cols, 10 + n as u64);
            let x = random(100 + n as u64, cols);
            let mut ours = vec![0f32; rows];
            matvec(&w, &x, &mut ours);
            let theirs = reference(&w, &x);

            let u = 2f64.powi(-24);
            let gamma = cols as f64 * u / (1.0 - cols as f64 * u);
            let mut worst_ratio = 0f64;
            for i in 0..rows {
                let abs_sum: f64 = w.row(i).iter().zip(&x).map(|(a, b)| (f64::from(*a) * f64::from(*b)).abs()).sum();
                let bound = 2.0 * gamma * abs_sum;
                let diff = f64::from(ours[i] - theirs[i]).abs();
                assert!(diff <= bound, "[{rows}, {cols}] row {i}: {diff} > {bound}");
                worst_ratio = worst_ratio.max(diff / bound);
            }
            // The observed difference is a small part of even this loose bound.
            assert!(worst_ratio < 0.5, "[{rows}, {cols}] worst ratio {worst_ratio}");
        }
    }

    // Spec §4 (measurement, not a check): how fast the plain index-order loop runs and what that
    // means for a full harness run. Run with
    //   cargo test -p emberly-oxygen --lib -- --ignored --nocapture matvec_speed
    // It also times two variants with independent partial sums, to inform the owner's choice
    // recorded in the Phase 4 to-do; those variants are not used by the engine.
    #[test]
    #[ignore = "a measurement: run with --ignored --nocapture"]
    fn matvec_speed() {
        fn dot_in_order(a: &[f32], b: &[f32]) -> f32 {
            let mut s = 0f32;
            for (x, y) in a.iter().zip(b) {
                s += x * y;
            }
            s
        }
        fn dot_with<const N: usize>(a: &[f32], b: &[f32]) -> f32 {
            let mut parts = [0f32; N];
            let (ca, cb) = (a.chunks_exact(N), b.chunks_exact(N));
            let (ra, rb) = (ca.remainder(), cb.remainder());
            for (x, y) in ca.zip(cb) {
                for k in 0..N {
                    parts[k] += x[k] * y[k];
                }
            }
            let mut s: f32 = parts.iter().sum();
            for (x, y) in ra.iter().zip(rb) {
                s += x * y;
            }
            s
        }
        fn run(w: &Tensor, x: &[f32], dot: fn(&[f32], &[f32]) -> f32) -> f64 {
            let rows = w.shape()[0];
            let mut out = vec![0f32; rows];
            let reps = (200_000_000 / w.data().len()).max(3);
            let start = Instant::now();
            for _ in 0..reps {
                for (i, o) in out.iter_mut().enumerate() {
                    *o = dot(w.row(i), x);
                }
                std::hint::black_box(&out);
            }
            start.elapsed().as_secs_f64() / reps as f64
        }

        // Per token: 30 layers of q, k, v, o, gate, up, down, plus the output projection.
        let layer = [(576, 576, 2), (192, 576, 2), (1536, 576, 2), (576, 1536, 1)];
        let variants: [(&str, fn(&[f32], &[f32]) -> f32); 3] =
            [("index order", dot_in_order), ("4 partial sums", dot_with::<4>), ("8 partial sums", dot_with::<8>)];
        for (name, dot) in variants {
            let mut per_token = 0f64;
            let mut macs = 0f64;
            for (rows, cols, count) in layer.into_iter().chain([(49152, 576, 0)]) {
                let w = weight(rows, cols, 7);
                let x = random(8, cols);
                let t = run(&w, &x, dot);
                let times = if count == 0 { 1.0 } else { 30.0 * count as f64 };
                per_token += t * times;
                macs += (rows * cols) as f64 * times;
                println!("{name:>15}  [{rows:>5}, {cols:>4}]  {:>7.1} us  {:>5.2} G multiply-adds/s", t * 1e6, (rows * cols) as f64 / t / 1e9);
            }
            let positions = 2596.0;
            println!(
                "{name:>15}  per token {:.1} ms ({:.0} M multiply-adds); matrix-vector part of a full harness run: {:.0} s\n",
                per_token * 1e3,
                macs / 1e6,
                per_token * positions
            );
        }
    }

    // FR-43, Spec §4: the threaded products equal the single-threaded ones exactly, at counts
    // that divide the rows and counts that do not, at the model's shapes and at odd ones.
    const SHAPES: [(usize, usize); 9] =
        [(192, 576), (576, 576), (1536, 576), (576, 1536), (1, 37), (5, 16), (3, 7), (100, 33), (49, 15)];

    #[test]
    fn pooled_matvec_equals_matvec_exactly_at_every_count() {
        for threads in [1, 2, 3, 7, 16] {
            let pool = Pool::new(threads).unwrap();
            for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
                let w = weight(rows, cols, 100 + i as u64);
                let x = random(200 + i as u64, cols);
                let mut want = vec![0.0; rows];
                matvec(&w, &x, &mut want);
                for min in [0, MIN_THREADED_MADDS, usize::MAX] {
                    let mut got = vec![f32::NAN; rows];
                    matvec_pooled_min(&pool, &w, &x, &mut got, min);
                    assert_eq!(got, want, "threads {threads} shape {rows}x{cols} min {min}");
                }
            }
        }
    }

    #[test]
    fn pooled_matmul_equals_matmul_exactly_at_every_count() {
        for threads in [1, 2, 3, 7, 16] {
            let pool = Pool::new(threads).unwrap();
            for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
                for count in [1, 3, 8] {
                    let w = weight(rows, cols, 300 + i as u64);
                    let x = random(400 + i as u64, count * cols);
                    let mut want = vec![0.0; count * rows];
                    matmul(&w, &x, count, &mut want);
                    for min in [0, usize::MAX] {
                        let mut got = vec![f32::NAN; count * rows];
                        matmul_pooled_min(&pool, &w, &x, count, &mut got, min);
                        assert_eq!(got, want, "threads {threads} shape {rows}x{cols} count {count} min {min}");
                    }
                }
            }
        }
    }

    // FR-43, Spec §4: a small product, or a single thread, stays on the calling thread.
    #[test]
    fn small_products_and_one_thread_do_not_use_workers() {
        assert!(!threads_pay(1536, 576, 1, 1, MIN_THREADED_MADDS), "one thread");
        assert!(!threads_pay(192, 576, 1, 8, MIN_THREADED_MADDS), "key projection is below the threshold");
        assert!(!threads_pay(576, 576, 1, 8, MIN_THREADED_MADDS), "576x576 is below the threshold");
        assert!(threads_pay(1536, 576, 1, 8, MIN_THREADED_MADDS), "gate projection pays");
        assert!(threads_pay(49152, 576, 1, 2, MIN_THREADED_MADDS), "unembedding pays");
        assert!(threads_pay(576, 576, 4, 8, MIN_THREADED_MADDS), "more inputs make the same weight pay");
        assert!(threads_pay(1, 1, 1, 2, 0), "a zero threshold always threads");
    }

    /// The blocked product's definition (FR-46): every element a sum over the shared dimension in index
    /// order, taken in chunks of 64 products, each chunk summed with fused multiply-adds and the chunk sums
    /// added in order. Layout `[count][rows]`, like the products'.
    fn fused_sum_reference(weight: &Tensor, input: &[f32], count: usize) -> Vec<f32> {
        let (rows, cols) = (weight.shape()[0], weight.shape()[1]);
        let mut out = vec![0f32; count * rows];
        for t in 0..count {
            for i in 0..rows {
                let mut total = 0f32;
                for start in (0..cols).step_by(64) {
                    let mut chunk = 0f32;
                    for k in start..(start + 64).min(cols) {
                        chunk = weight.row(i)[k].mul_add(input[t * cols + k], chunk);
                    }
                    total += chunk;
                }
                out[t * rows + i] = total;
            }
        }
        out
    }

    const BLOCKED_SHAPES: [(usize, usize); 16] = [
        (192, 576), (576, 576), (1536, 576), (576, 1536), (1, 37), (5, 16), (3, 7), (100, 33), (49, 15),
        (4, 16), (7, 17), (8, 1), (9, 64), (6, 65), (5, 129), (10, 192),
    ];
    const BLOCKED_COUNTS: [usize; 11] = [1, 2, 7, 8, 15, 16, 17, 31, 32, 33, 67];

    // FR-46, Spec §4: the blocked product is exactly the chunked fused sum, at the model's shapes
    // and at row counts, input counts and lengths that do not divide the tile, are smaller than it, or are 1.
    #[test]
    fn blocked_matmul_is_exactly_the_chunked_fused_sum() {
        for (i, &(rows, cols)) in BLOCKED_SHAPES.iter().enumerate() {
            let w = weight(rows, cols, 500 + i as u64);
            for count in BLOCKED_COUNTS {
                let x = random(600 + i as u64, count * cols);
                let mut got = vec![f32::NAN; count * rows];
                matmul_blocked(&w, &x, count, &mut got);
                assert_eq!(got, fused_sum_reference(&w, &x, count), "shape {rows}x{cols} count {count}");
            }
        }
    }

    // FR-46, Spec §4: a tile edge forgotten would leave elements unwritten; none is.
    #[test]
    fn blocked_matmul_writes_every_output_element() {
        for (rows, cols, count) in [(5, 16, 17), (7, 33, 33), (3, 7, 1), (9, 20, 31), (4, 16, 16)] {
            let w = weight(rows, cols, 7);
            let x = random(8, count * cols);
            let mut got = vec![f32::NAN; count * rows];
            matmul_blocked(&w, &x, count, &mut got);
            assert!(got.iter().all(|v| v.is_finite()), "{rows}x{cols} count {count}");
        }
    }

    // FR-46, FR-47, HC-2: the blocked product agrees with the test-only linear algebra crate far
    // inside S-1's tolerance, and is not the 16-partial-sum product's arithmetic: a comparison that
    // could not see a changed order of additions would be caught here.
    #[test]
    fn blocked_matmul_agrees_with_matrixmultiply_and_differs_from_the_partial_sum_loop() {
        let (rows, cols, count) = (192, 576, 8);
        let w = weight(rows, cols, 41);
        let x = random(42, count * cols);
        let mut blocked = vec![0f32; count * rows];
        matmul_blocked(&w, &x, count, &mut blocked);
        for t in 0..count {
            close(&blocked[t * rows..(t + 1) * rows], &reference(&w, &x[t * cols..(t + 1) * cols]), 1e-4);
        }
        let mut sixteen = vec![0f32; count * rows];
        matmul(&w, &x, count, &mut sixteen);
        assert_ne!(blocked, sixteen, "a different order of additions must show in some bit");
        for (a, b) in blocked.iter().zip(&sixteen) {
            assert!((a - b).abs() < 1e-4, "but only in the last bits: {a} {b}");
        }
    }

    // FR-46, FR-43, Spec §4: the pooled blocked product equals the single-threaded one exactly at every
    // count, so neither the tile nor the division of rows changes a bit.
    #[test]
    fn pooled_blocked_matmul_equals_blocked_matmul_exactly_at_every_count() {
        for threads in [1, 2, 3, 7, 16] {
            let pool = Pool::new(threads).unwrap();
            for (i, &(rows, cols)) in BLOCKED_SHAPES.iter().enumerate() {
                for count in [1, 8, 17, 33] {
                    let w = weight(rows, cols, 900 + i as u64);
                    let x = random(950 + i as u64, count * cols);
                    let mut want = vec![0f32; count * rows];
                    matmul_blocked(&w, &x, count, &mut want);
                    for min in [0, usize::MAX] {
                        let mut got = vec![f32::NAN; count * rows];
                        matmul_blocked_pooled_min(&pool, &w, &x, count, &mut got, min);
                        assert_eq!(got, want, "threads {threads} shape {rows}x{cols} count {count} min {min}");
                    }
                }
            }
        }
    }
}
