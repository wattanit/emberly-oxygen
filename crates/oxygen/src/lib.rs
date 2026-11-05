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
mod tokenizer;
mod weights;

pub use config::Config;
pub use error::Error;
#[cfg(feature = "negative-controls")]
pub use fault::Fault;
pub use generate::generate;
pub use model::{Capture, Model, Session};
pub use tensor::Tensor;
pub use tokenizer::{StreamDecoder, Tokenizer};
