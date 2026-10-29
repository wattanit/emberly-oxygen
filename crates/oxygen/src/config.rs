use std::path::Path;

use serde_json::Value;

use crate::Error;

/// The model's shape, read from `config.json` (Spec §3). Sizes and constants
/// come from the file; the supported set is checked at load (FR-12).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    pub hidden_size: usize,
    pub layers: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub context_length: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f64,
}

impl Config {
    /// Reads `config.json` from a model directory.
    pub fn load(model_dir: &Path) -> Result<Config, Error> {
        let path = model_dir.join("config.json");
        let text = std::fs::read_to_string(&path).map_err(|source| Error::Io {
            path: path.clone(),
            source,
        })?;
        Config::parse(&text, &path)
    }

    pub(crate) fn parse(text: &str, path: &Path) -> Result<Config, Error> {
        let malformed = |reason: String| Error::Malformed {
            path: path.to_owned(),
            reason,
        };
        let json: Value = serde_json::from_str(text).map_err(|e| malformed(e.to_string()))?;
        let obj = json
            .as_object()
            .ok_or_else(|| malformed("expected a JSON object".into()))?;

        check_supported(obj)?;

        let size = |field: &str| -> Result<usize, Error> {
            let n = obj
                .get(field)
                .and_then(Value::as_u64)
                .ok_or_else(|| malformed(format!("field `{field}` missing or not a whole number")))?;
            if n == 0 {
                return Err(malformed(format!("field `{field}` is zero")));
            }
            Ok(n as usize)
        };
        let float = |field: &str| -> Result<f64, Error> {
            obj.get(field)
                .and_then(Value::as_f64)
                .filter(|x| x.is_finite() && *x > 0.0)
                .ok_or_else(|| malformed(format!("field `{field}` missing or not a positive number")))
        };

        let hidden_size = size("hidden_size")?;
        let heads = size("num_attention_heads")?;
        let kv_heads = size("num_key_value_heads")?;
        if heads % kv_heads != 0 {
            return Err(malformed(format!(
                "num_attention_heads ({heads}) is not a multiple of num_key_value_heads ({kv_heads})"
            )));
        }
        let head_dim = match obj.get("head_dim").filter(|v| !v.is_null()) {
            Some(_) => size("head_dim")?,
            None if hidden_size % heads == 0 => hidden_size / heads,
            None => {
                return Err(malformed(format!(
                    "hidden_size ({hidden_size}) is not divisible by num_attention_heads ({heads}) and no head_dim is given"
                )));
            }
        };

        Ok(Config {
            hidden_size,
            layers: size("num_hidden_layers")?,
            heads,
            kv_heads,
            head_dim,
            intermediate_size: size("intermediate_size")?,
            vocab_size: size("vocab_size")?,
            context_length: size("max_position_embeddings")?,
            rope_theta: float("rope_theta")?,
            rms_norm_eps: float("rms_norm_eps")?,
        })
    }
}

/// Rejects anything outside the Spec §3 supported set, naming field and value.
fn check_supported(obj: &serde_json::Map<String, Value>) -> Result<(), Error> {
    // (field, the one supported value, whether the field may be absent)
    let table = [
        ("model_type", Value::from("llama"), false),
        ("hidden_act", Value::from("silu"), false),
        ("attention_bias", Value::Bool(false), false),
        ("mlp_bias", Value::Bool(false), true),
        ("rope_scaling", Value::Null, true),
        ("rope_interleaved", Value::Bool(false), true),
        ("tie_word_embeddings", Value::Bool(true), false),
        ("pretraining_tp", Value::from(1), true),
    ];
    for (field, supported, optional) in table {
        match obj.get(field) {
            Some(found) if *found == supported => {}
            None if optional => {}
            found => {
                return Err(Error::Unsupported {
                    component: field.into(),
                    value: found.map_or("absent".into(), Value::to_string),
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REAL: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/smollm2-135m/config.json");

    fn real_json() -> Value {
        let text = std::fs::read_to_string(REAL).unwrap_or_else(|e| {
            panic!("{REAL}: {e}; run `cd oracle && uv run python download.py`")
        });
        serde_json::from_str(&text).unwrap()
    }

    fn parse(v: &Value) -> Result<Config, Error> {
        Config::parse(&v.to_string(), Path::new("config.json"))
    }

    fn altered(field: &str, value: Option<Value>) -> Value {
        let mut v = real_json();
        let obj = v.as_object_mut().unwrap();
        match value {
            Some(x) => obj.insert(field.into(), x),
            None => obj.remove(field),
        };
        v
    }

    fn message(r: Result<Config, Error>) -> String {
        r.expect_err("expected rejection").to_string()
    }

    // FR-12, Spec §3: sizes and constants come from the file.
    #[test]
    fn real_config_parses() {
        let c = parse(&real_json()).unwrap();
        assert_eq!(c.hidden_size, 576);
        assert_eq!(c.layers, 30);
        assert_eq!((c.heads, c.kv_heads, c.head_dim), (9, 3, 64));
        assert_eq!(c.intermediate_size, 1536);
        assert_eq!(c.vocab_size, 49152);
        assert_eq!(c.context_length, 8192);
        assert_eq!(c.rope_theta, 100000.0);
        assert_eq!(c.rms_norm_eps, 1e-5);
    }

    // FR-11, FR-12: the configuration is read from the published file in the model directory.
    #[test]
    fn load_reads_a_model_directory() {
        let dir = Path::new(REAL).parent().unwrap();
        assert_eq!(Config::load(dir).unwrap(), parse(&real_json()).unwrap());
        let err = Config::load(Path::new("/nonexistent-model-dir")).unwrap_err();
        assert!(matches!(err, Error::Io { .. }) && err.to_string().contains("config.json"));
    }

    // FR-12, Spec §3: each supported-set row rejects naming field and value.
    #[test]
    fn unsupported_values_name_field_and_value() {
        let cases = [
            ("model_type", Value::from("gpt2"), "gpt2"),
            ("hidden_act", Value::from("gelu"), "gelu"),
            ("attention_bias", Value::from(true), "true"),
            ("mlp_bias", Value::from(true), "true"),
            ("rope_scaling", serde_json::json!({"type": "linear", "factor": 2.0}), "linear"),
            ("rope_interleaved", Value::from(true), "true"),
            ("tie_word_embeddings", Value::from(false), "false"),
            ("pretraining_tp", Value::from(2), "2"),
        ];
        for (field, value, shown) in cases {
            let msg = message(parse(&altered(field, Some(value))));
            assert!(msg.contains(field) && msg.contains(shown), "{field}: {msg}");
        }
    }

    // FR-12, Spec §3: absent-field cases follow the table.
    #[test]
    fn absent_fields_follow_the_table() {
        for field in ["mlp_bias", "rope_interleaved", "pretraining_tp", "rope_scaling"] {
            parse(&altered(field, None)).unwrap_or_else(|e| panic!("{field} absent: {e}"));
        }
        for field in ["model_type", "hidden_act", "attention_bias", "tie_word_embeddings"] {
            let msg = message(parse(&altered(field, None)));
            assert!(msg.contains(field) && msg.contains("absent"), "{field}: {msg}");
        }
    }

    // FR-12: head dimension from the file when present, else derived; bad sizes name the field.
    #[test]
    fn head_dim_and_size_checks() {
        assert_eq!(parse(&altered("head_dim", Some(Value::from(32)))).unwrap().head_dim, 32);
        assert_eq!(parse(&altered("head_dim", Some(Value::Null))).unwrap().head_dim, 64);

        let msg = message(parse(&altered("hidden_size", Some(Value::from(577)))));
        assert!(msg.contains("hidden_size") && msg.contains("num_attention_heads"), "{msg}");

        let msg = message(parse(&altered("num_key_value_heads", Some(Value::from(2)))));
        assert!(msg.contains("num_key_value_heads"), "{msg}");

        let msg = message(parse(&altered("num_hidden_layers", Some(Value::from(0)))));
        assert!(msg.contains("num_hidden_layers"), "{msg}");

        let msg = message(parse(&altered("vocab_size", None)));
        assert!(msg.contains("vocab_size"), "{msg}");

        let msg = message(parse(&altered("rms_norm_eps", Some(Value::from("x")))));
        assert!(msg.contains("rms_norm_eps"), "{msg}");
    }

    // FR-12, Spec §14: a broken configuration file is an error, never a panic.
    #[test]
    fn non_json_is_malformed_not_a_panic() {
        for text in ["", "[]", "{", "null"] {
            let err = Config::parse(text, Path::new("config.json")).unwrap_err();
            assert!(matches!(err, Error::Malformed { .. }), "{text}: {err}");
        }
    }
}
