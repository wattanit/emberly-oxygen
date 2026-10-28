//! Emberly Oxygen: a decoder-only LLM inference engine, built from scratch.
//!
//! The surface below is the minimum the correctness harness needs. Its
//! implementation is a stub that returns wrong values, so the harness can be
//! seen to fail (Spec §18, M2); later phases replace it piece by piece.

mod config;
mod error;
mod generate;
mod model;
mod safetensors;
mod tensor;
mod tokenizer;

pub use config::Config;
pub use error::Error;
pub use generate::generate;
pub use model::{Capture, Model, Session};
pub use tensor::Tensor;
pub use tokenizer::Tokenizer;
