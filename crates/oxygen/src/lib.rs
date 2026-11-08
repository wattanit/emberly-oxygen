//! Emberly Oxygen: a decoder-only LLM inference engine, built from scratch.
//!
//! The surface below is the minimum the correctness harness and the
//! command-line tool need: loading, tokenization, the forward pass one token
//! at a time, and greedy generation (Spec §1).

mod config;
mod error;
mod fault;
mod forward;
mod generate;
mod kv_cache;
mod model;
mod ops;
mod rng;
mod sampling;
mod safetensors;
mod tensor;
#[cfg(test)]
mod test_log;
mod tokenizer;
mod weights;

pub use config::Config;
pub use error::Error;
#[cfg(feature = "negative-controls")]
pub use fault::Fault;
pub use generate::{generate, generate_with, Generation, Stats};
pub use model::{Capture, Model, Session};
pub use sampling::{distribution, Sampling};
#[cfg(feature = "negative-controls")]
pub use sampling::distribution_with_fault;
pub use tensor::Tensor;
pub use tokenizer::{StreamDecoder, Tokenizer};

/// The engine's own kernels, reachable for the benchmark's kernel group, which
/// times them against a mature implementation (HC-2, Spec §12). Only exists
/// with the `bench-kernels` feature, which a normal build never enables.
#[cfg(feature = "bench-kernels")]
pub mod kernels {
    use crate::Tensor;

    /// The matrix-vector product every projection in the forward pass uses:
    /// `output[i]` is the dot product of weight row `i` with `input`.
    pub fn matvec(weight: &Tensor, input: &[f32], output: &mut [f32]) {
        crate::ops::matvec(weight, input, output);
    }

    /// The partial-sum matrix-vector product with `N` independent sums (FR-25,
    /// Spec §4); the engine's own `matvec` is this at the width `ops::LANES` names. Public so the
    /// tests and the benchmark can try other widths.
    pub fn matvec_lanes<const N: usize>(weight: &Tensor, input: &[f32], output: &mut [f32]) {
        crate::ops::matvec_lanes::<N>(weight, input, output);
    }

    /// The matrix-vector product exactly as release 0.1.0 computed it: one
    /// running sum per output element, added in index order. Kept as a
    /// reference point for tests and the benchmark's kernel group (FR-25,
    /// Spec §12, §17); the engine never calls it, so changing the engine's
    /// own kernel cannot change it.
    pub fn matvec_index_order(weight: &Tensor, input: &[f32], output: &mut [f32]) {
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
}
