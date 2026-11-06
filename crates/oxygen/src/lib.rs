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
pub use generate::{generate, Generation, Stats};
pub use model::{Capture, Model, Session};
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
}
