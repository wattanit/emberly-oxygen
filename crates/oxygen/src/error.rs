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

#[cfg(test)]
mod tests {
    use super::*;

    // FR-12, Spec §14: each variant's message names what the Specification says it names.
    #[test]
    fn messages_name_path_component_and_tensor() {
        let io = Error::Io {
            path: "models/x/config.json".into(),
            source: std::io::Error::other("boom"),
        };
        assert!(io.to_string().contains("models/x/config.json"));

        let malformed = Error::Malformed {
            path: "a/b.json".into(),
            reason: "bad header".into(),
        };
        let text = malformed.to_string();
        assert!(text.contains("a/b.json") && text.contains("bad header"));

        let unsupported = Error::Unsupported {
            component: "hidden_act".into(),
            value: "gelu".into(),
        };
        let text = unsupported.to_string();
        assert!(text.contains("hidden_act") && text.contains("gelu"));

        let mismatch = Error::TensorMismatch {
            name: "model.norm.weight".into(),
            reason: "missing".into(),
        };
        assert!(mismatch.to_string().contains("model.norm.weight"));

        assert!(Error::Rejected("empty prompt".into()).to_string().contains("empty prompt"));
    }
}
