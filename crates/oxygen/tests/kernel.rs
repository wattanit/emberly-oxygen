//! Tests of the matrix-vector kernel (FR-25, Spec §4, §17), through the public
//! `kernels` module that the `bench-kernels` feature adds. Plain `cargo test`
//! has the feature on through the library's self-referencing dev-dependency.

use emberly_oxygen::kernels;
use emberly_oxygen::Tensor;

/// A small seeded generator (xorshift64*), so inputs are reproducible without
/// a dependency. Values are uniform in [-1, 1).
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_f491_4f6c_dd1d) >> 40;
        bits as f32 / (1u64 << 23) as f32 - 1.0
    }

    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

type Kernel = fn(&Tensor, &[f32], &mut [f32]);

/// The partial-sum kernel at each width the release tries (Spec §4).
fn lanes(n: usize) -> Kernel {
    match n {
        4 => kernels::matvec_lanes::<4>,
        8 => kernels::matvec_lanes::<8>,
        16 => kernels::matvec_lanes::<16>,
        32 => kernels::matvec_lanes::<32>,
        _ => panic!("no kernel at width {n}"),
    }
}

const WIDTHS: [usize; 4] = [4, 8, 16, 32];

fn run(kernel: Kernel, weight: &Tensor, input: &[f32]) -> Vec<f32> {
    let mut out = vec![0f32; weight.shape()[0]];
    kernel(weight, input, &mut out);
    out
}

/// `matrixmultiply`'s single-column product, the test-only baseline (HC-2).
fn baseline(weight: &Tensor, input: &[f32]) -> Vec<f32> {
    let (m, k) = (weight.shape()[0], weight.shape()[1]);
    let mut out = vec![0f32; m];
    // SAFETY: out (m x 1) = weight (m x k, row-major) * input (k x 1); every
    // pointer covers exactly the elements those strides reach.
    unsafe {
        matrixmultiply::sgemm(
            m, k, 1, 1.0,
            weight.data().as_ptr(), k as isize, 1,
            input.as_ptr(), 1, 1,
            0.0,
            out.as_mut_ptr(), 1, 1,
        );
    }
    out
}

/// Fails unless every element of `got` is within `64 * f32::EPSILON * sum |w * x|`
/// of `want`: the summation-error bound scaled by how much the row's terms
/// could have cancelled (plan 0.1.1 §5, decision of 2026-11-07).
fn assert_close(what: &str, weight: &Tensor, input: &[f32], got: &[f32], want: &[f32]) {
    let cols = weight.shape()[1];
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let scale: f32 = if cols == 0 { 0.0 } else { weight.row(i).iter().zip(input).map(|(a, b)| (a * b).abs()).sum() };
        let bound = 64.0 * f32::EPSILON * scale;
        assert!((g - w).abs() <= bound, "{what}: row {i} is {g}, expected {w} within {bound}");
    }
}

// FR-25, HC-2, S-1: at the model's five matrix shapes, every width agrees with
// the 0.1.0 loop and with matrixmultiply.
#[test]
fn fr25_agrees_with_the_references_at_the_models_shapes() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for (rows, cols) in [(576, 576), (192, 576), (1536, 576), (576, 1536), (49152, 576)] {
        let weight = Tensor::new(vec![rows, cols], rng.vec(rows * cols)).unwrap();
        let input = rng.vec(cols);
        let old = run(kernels::matvec_index_order, &weight, &input);
        let library = baseline(&weight, &input);
        for n in WIDTHS {
            let got = run(lanes(n), &weight, &input);
            assert_close(&format!("{rows}x{cols}, {n} sums, against the 0.1.0 loop"), &weight, &input, &got, &old);
            assert_close(&format!("{rows}x{cols}, {n} sums, against matrixmultiply"), &weight, &input, &got, &library);
        }
    }
}

// FR-25: every column count from 0 to 40, so lengths below the width (only a
// tail), equal to it, one over it and not divisible by it are all covered.
#[test]
fn fr25_every_length_from_0_to_40() {
    let mut rng = Rng(0xdead_beef_0bad_f00d);
    for cols in 0..=40 {
        let rows = 5;
        let weight = Tensor::new(vec![rows, cols], rng.vec(rows * cols)).unwrap();
        let input = rng.vec(cols);
        let old = run(kernels::matvec_index_order, &weight, &input);
        for n in WIDTHS {
            let got = run(lanes(n), &weight, &input);
            assert_close(&format!("{cols} columns, {n} sums"), &weight, &input, &got, &old);
        }
    }
    // Zero rows are valid and give an empty output.
    let none = Tensor::new(vec![0, 9], vec![]).unwrap();
    for n in WIDTHS {
        assert!(run(lanes(n), &none, &[0.0; 9]).is_empty());
    }
}

// FR-25: with small-integer weights and inputs every product and every partial
// sum is exactly representable, so all variants give the same bits; a wrong
// index shows here and cannot be mistaken for rounding.
#[test]
fn fr25_exact_on_exact_inputs() {
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let mut small_ints = |n: usize| -> Vec<f32> { (0..n).map(|_| (rng.next() * 8.0).floor()).collect() };
    for cols in [0, 1, 7, 8, 9, 15, 16, 17, 31, 40, 576] {
        let rows = 6;
        let weight = Tensor::new(vec![rows, cols], small_ints(rows * cols)).unwrap();
        let input = small_ints(cols);
        let old = run(kernels::matvec_index_order, &weight, &input);
        for n in WIDTHS {
            let got = run(lanes(n), &weight, &input);
            assert_eq!(got, old, "{cols} columns, {n} sums");
        }
        assert_eq!(run(kernels::matvec, &weight, &input), old, "{cols} columns, the engine's kernel");
    }
}

// S-6, FR-25: two calls on the same input are bit-identical.
#[test]
fn s6_calls_are_bit_identical() {
    let mut rng = Rng(0xfeed_face_cafe_beef);
    let weight = Tensor::new(vec![192, 577], rng.vec(192 * 577)).unwrap();
    let input = rng.vec(577);
    for kernel in [kernels::matvec as Kernel, lanes(4), lanes(8), lanes(16), lanes(32)] {
        let (a, b) = (run(kernel, &weight, &input), run(kernel, &weight, &input));
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()));
    }
}

// FR-25, Spec §4: pins the partial-sum structure. With 16 columns, an input of
// ones and weights [1e8, 1 x7, -1e8, 1 x7] (1e8 is exact in f32 and the spacing
// there is 8, so adding 1 to it changes nothing):
//   index order:  1e8 absorbs the seven ones, cancels against -1e8, then the
//                 last seven ones survive: 7.
//   4 sums:       lane 0 holds 1e8, 1, -1e8, 1 and gives 1; lanes 1 to 3 give
//                 4 each: 1 + 12 = 13.
//   8 sums:       lane 0 holds 1e8 + -1e8 = 0; lanes 1 to 7 give 2 each: 14.
//   32 sums:      16 columns are all tail: no chunk, every lane 0, and the tail is
//                 added in index order: 7.
//   16 sums:      one chunk; the combine takes lane 0 (1e8), absorbs seven
//                 ones, cancels against lane 8 (-1e8), then adds seven ones: 7.
// The 0.1.0 loop gives 7, so this fails against it for every width but 16 and
// for the engine's kernel (8 sums).
#[test]
fn fr25_order_pin_partial_sum_structure() {
    let mut w = vec![1.0f32; 16];
    w[0] = 1e8;
    w[8] = -1e8;
    let weight = Tensor::new(vec![1, 16], w).unwrap();
    let input = vec![1.0f32; 16];
    assert_eq!(run(kernels::matvec_index_order, &weight, &input), [7.0]);
    assert_eq!(run(lanes(4), &weight, &input), [13.0]);
    assert_eq!(run(lanes(8), &weight, &input), [14.0]);
    assert_eq!(run(lanes(16), &weight, &input), [7.0]);
    assert_eq!(run(lanes(32), &weight, &input), [7.0], "32 sums on 16 columns is all tail, so index order");
    assert_eq!(run(kernels::matvec, &weight, &input), [14.0], "the engine uses 8 sums");
}

// FR-25, Spec §4, plan 0.1.1 §5: pins the combining order (lane 0 first, one
// lane at a time). With 8 columns each lane holds one product: [1e8, 3 x7].
//   lane 0 first:   1e8 + 3 rounds back to 1e8 seven times: 100000000.
//   last lane first: 3 x7 = 21, then + 1e8 = 100000021, nearest multiple of 8
//                    is 100000024.
//   pairwise tree:   (1e8+3) = 1e8, (3+3) = 6, 1e8+6 -> 100000008, (3+3)+(3+3)
//                    = 12, 100000008+12 = 100000020, a tie that rounds to even:
//                    100000016.
// All three differ, so a different combining order fails this test.
#[test]
fn fr25_order_pin_combining_order() {
    let mut w = vec![3.0f32; 8];
    w[0] = 1e8;
    let weight = Tensor::new(vec![1, 8], w).unwrap();
    let input = vec![1.0f32; 8];
    assert_eq!(run(lanes(8), &weight, &input), [100_000_000.0]);
    assert_eq!(run(kernels::matvec, &weight, &input), [100_000_000.0]);
}

// Spec §14: a mismatch in a kernel call is an internal bug and panics, as the
// engine's own kernel does.
#[test]
#[should_panic(expected = "2-D")]
fn fr25_lanes_rejects_a_weight_that_is_not_2d() {
    let weight = Tensor::new(vec![6], vec![0.0; 6]).unwrap();
    kernels::matvec_lanes::<8>(&weight, &[0.0; 6], &mut [0.0; 1]);
}

// FR-25, Spec §14: an input whose length is not the weight's columns is an internal bug and panics.
#[test]
#[should_panic(expected = "input length")]
fn fr25_lanes_rejects_a_wrong_input_length() {
    let weight = Tensor::new(vec![2, 3], vec![0.0; 6]).unwrap();
    kernels::matvec_lanes::<8>(&weight, &[0.0; 4], &mut [0.0; 2]);
}

// FR-25, Spec §14: an output whose length is not the weight's rows is an internal bug and panics.
#[test]
#[should_panic(expected = "output length")]
fn fr25_lanes_rejects_a_wrong_output_length() {
    let weight = Tensor::new(vec![2, 3], vec![0.0; 6]).unwrap();
    kernels::matvec_lanes::<8>(&weight, &[0.0; 3], &mut [0.0; 3]);
}
