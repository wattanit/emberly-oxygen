use std::path::PathBuf;

/// One variant per failure class (Spec §14). User input never causes a panic.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("malformed file {path}: {reason}")]
    Malformed { path: PathBuf, reason: String },
    #[error("unsupported {component}: {value}")]
    Unsupported { component: String, value: String },
    #[error("tensor {name}: {reason}")]
    TensorMismatch { name: String, reason: String },
    #[error("request rejected: {0}")]
    Rejected(String),
}
