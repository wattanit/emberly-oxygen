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

// FR-25: the kept reference is the 0.1.0 loop, so until the engine's kernel
// changes (task 7.3) the two give identical bits. Task 7.3 replaces this test
// with the agreement and order tests of task 7.2.
#[test]
fn index_order_reference_matches_the_engine_until_the_kernel_changes() {
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for (rows, cols) in [(7, 13), (192, 576), (64, 1536)] {
        let weight = Tensor::new(vec![rows, cols], rng.vec(rows * cols)).unwrap();
        let input = rng.vec(cols);
        let mut engine = vec![0f32; rows];
        let mut reference = vec![0f32; rows];
        kernels::matvec(&weight, &input, &mut engine);
        kernels::matvec_index_order(&weight, &input, &mut reference);
        let same = engine.iter().zip(&reference).all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(same, "{rows}x{cols}: the reference differs from the engine's kernel");
    }
}
