//! The tensor operations the forward pass needs (Spec §4). Each is a plain
//! loop over slices with f32 accumulation in index order: no `unsafe`, SIMD
//! or threading (HC-1). The reference sums in a different order, so results
//! differ in the last bits; S-1 and S-2 bound that difference.

// The forward pass (task 4.5) is the first caller of these, and removes this.
#![allow(dead_code)]

use crate::Tensor;

/// `output[i]` = dot product of weight row `i` with `input`, for a weight in
/// the published `[out, in]` layout (Spec §3), so each row is contiguous.
///
/// Panics if the weight is not 2-D or the lengths disagree: the engine builds
/// every call from the configuration, so a mismatch is an internal bug and
/// never the effect of user input (Spec §14).
pub(crate) fn matvec(weight: &Tensor, input: &[f32], output: &mut [f32]) {
    let &[rows, cols] = weight.shape() else {
        panic!("matvec needs a 2-D weight, got shape {:?}", weight.shape());
    };
    assert_eq!(input.len(), cols, "input length must equal the weight's columns");
    assert_eq!(output.len(), rows, "output length must equal the weight's rows");
    for (i, out) in output.iter_mut().enumerate() {
        let mut sum = 0f32;
        for (w, x) in weight.row(i).iter().zip(input) {
            sum += w * x;
        }
        *out = sum;
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

    #[test]
    #[should_panic(expected = "output length")]
    fn wrong_output_length_panics() {
        matvec(&weight(2, 3, 1), &[1., 2., 3.], &mut [0.; 3]);
    }

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
}
