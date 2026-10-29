//! Binds the tensors of a safetensors file to the model's weights by their
//! published names (Spec §3, §5). A missing, mis-shaped or unexpected tensor
//! means the file is not the model the configuration describes, so each is
//! rejected naming the tensor (FR-12).

use std::collections::BTreeSet;

use crate::safetensors::SafeTensors;
use crate::{Config, Error, Tensor};

/// One decoder layer's weights. Matrices keep the published `[out, in]` layout.
pub struct LayerWeights {
    pub attn_norm: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub mlp_norm: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
}

/// Every learned parameter of the model. The embedding table doubles as the
/// output projection because embeddings are tied (Spec §3).
pub struct Weights {
    pub embed_tokens: Tensor,
    pub layers: Vec<LayerWeights>,
    pub final_norm: Tensor,
}

impl Weights {
    pub fn bind(file: &SafeTensors, config: &Config) -> Result<Weights, Error> {
        let mut expected = BTreeSet::new();
        let mut take = |name: String, shape: &[usize]| -> Result<Tensor, Error> {
            let found = file.shape(&name).ok_or_else(|| Error::TensorMismatch {
                name: name.clone(),
                reason: "missing from the weights file".into(),
            })?;
            if found != shape {
                return Err(Error::TensorMismatch {
                    name,
                    reason: format!("shape is {found:?}, the configuration requires {shape:?}"),
                });
            }
            let tensor = file.tensor(&name)?;
            expected.insert(name);
            Ok(tensor)
        };

        let hidden = config.hidden_size;
        let q_out = config.heads * config.head_dim;
        let kv_out = config.kv_heads * config.head_dim;
        let inter = config.intermediate_size;

        let embed_tokens = take("model.embed_tokens.weight".into(), &[config.vocab_size, hidden])?;
        let mut layers = Vec::with_capacity(config.layers);
        for i in 0..config.layers {
            let p = format!("model.layers.{i}");
            layers.push(LayerWeights {
                attn_norm: take(format!("{p}.input_layernorm.weight"), &[hidden])?,
                q_proj: take(format!("{p}.self_attn.q_proj.weight"), &[q_out, hidden])?,
                k_proj: take(format!("{p}.self_attn.k_proj.weight"), &[kv_out, hidden])?,
                v_proj: take(format!("{p}.self_attn.v_proj.weight"), &[kv_out, hidden])?,
                o_proj: take(format!("{p}.self_attn.o_proj.weight"), &[hidden, q_out])?,
                mlp_norm: take(format!("{p}.post_attention_layernorm.weight"), &[hidden])?,
                gate_proj: take(format!("{p}.mlp.gate_proj.weight"), &[inter, hidden])?,
                up_proj: take(format!("{p}.mlp.up_proj.weight"), &[inter, hidden])?,
                down_proj: take(format!("{p}.mlp.down_proj.weight"), &[hidden, inter])?,
            });
        }
        let final_norm = take("model.norm.weight".into(), &[hidden])?;

        // Tied embeddings: a separate output matrix (`lm_head.weight`) is not
        // part of this model and lands here with any other extra tensor.
        if let Some(extra) = file.names().find(|n| !expected.contains(*n)) {
            return Err(Error::TensorMismatch {
                name: extra.into(),
                reason: "not a tensor of this model".into(),
            });
        }
        Ok(Weights { embed_tokens, layers, final_norm })
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::safetensors::test_util::{f32_image, temp_file};

    fn tiny_config() -> Config {
        Config {
            hidden_size: 4,
            layers: 2,
            heads: 2,
            kv_heads: 1,
            head_dim: 2,
            intermediate_size: 6,
            vocab_size: 5,
            context_length: 16,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
        }
    }

    /// The tensors the tiny configuration requires, in a file order of our choosing.
    fn tiny_tensors() -> Vec<(String, Vec<usize>)> {
        let mut t = vec![("model.embed_tokens.weight".to_string(), vec![5, 4])];
        for i in 0..2 {
            let p = format!("model.layers.{i}");
            for (name, shape) in [
                ("input_layernorm", vec![4]),
                ("self_attn.q_proj", vec![4, 4]),
                ("self_attn.k_proj", vec![2, 4]),
                ("self_attn.v_proj", vec![2, 4]),
                ("self_attn.o_proj", vec![4, 4]),
                ("post_attention_layernorm", vec![4]),
                ("mlp.gate_proj", vec![6, 4]),
                ("mlp.up_proj", vec![6, 4]),
                ("mlp.down_proj", vec![4, 6]),
            ] {
                t.push((format!("{p}.{name}.weight"), shape));
            }
        }
        t.push(("model.norm.weight".into(), vec![4]));
        t
    }

    fn bind(tensors: &[(String, Vec<usize>)]) -> Result<Weights, Error> {
        let file = SafeTensors::open(&temp_file(&f32_image(tensors)))?;
        Weights::bind(&file, &tiny_config())
    }

    fn rejection(tensors: &[(String, Vec<usize>)]) -> String {
        bind(tensors).err().expect("expected rejection").to_string()
    }

    // FR-12, Spec §5: a file with exactly the expected tensors binds.
    #[test]
    fn complete_file_binds_with_published_layout() {
        let w = bind(&tiny_tensors()).unwrap();
        assert_eq!(w.embed_tokens.shape(), &[5, 4]);
        assert_eq!(w.layers.len(), 2);
        assert_eq!(w.layers[1].k_proj.shape(), &[2, 4]);
        assert_eq!(w.layers[0].down_proj.shape(), &[4, 6]);
        assert_eq!(w.final_norm.shape(), &[4]);
    }

    // FR-12, Spec §5: a missing tensor is rejected naming it.
    #[test]
    fn missing_tensor_is_named() {
        let mut t = tiny_tensors();
        t.retain(|(n, _)| n != "model.layers.1.mlp.up_proj.weight");
        let msg = rejection(&t);
        assert!(msg.contains("model.layers.1.mlp.up_proj.weight") && msg.contains("missing"), "{msg}");
    }

    // FR-12, Spec §5: a mis-shaped tensor is rejected naming it, with both shapes.
    #[test]
    fn mis_shaped_tensor_is_named() {
        let mut t = tiny_tensors();
        t.iter_mut().find(|(n, _)| n == "model.layers.0.self_attn.v_proj.weight").unwrap().1 = vec![4, 4];
        let msg = rejection(&t);
        assert!(msg.contains("model.layers.0.self_attn.v_proj.weight"), "{msg}");
        assert!(msg.contains("[4, 4]") && msg.contains("[2, 4]"), "{msg}");
    }

    // FR-12, Spec §5: an unexpected tensor is rejected naming it, including a separate lm_head.
    #[test]
    fn unexpected_tensor_is_named() {
        for extra in ["lm_head.weight", "model.layers.2.mlp.up_proj.weight"] {
            let mut t = tiny_tensors();
            t.push((extra.into(), vec![5, 4]));
            let msg = rejection(&t);
            assert!(msg.contains(extra) && msg.contains("not a tensor"), "{msg}");
        }
    }

    // FR-11, FR-12: the real published model binds against its own configuration.
    #[test]
    fn real_model_binds() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        assert!(
            dir.exists(),
            "{}: missing; run `cd oracle && uv run python download.py`",
            dir.display()
        );
        let config = Config::load(&dir).unwrap();
        let file = SafeTensors::open(&dir.join("model.safetensors")).unwrap();
        let w = Weights::bind(&file, &config).unwrap();
        assert_eq!(w.layers.len(), 30);
        assert_eq!(w.layers[0].gate_proj.shape(), &[1536, 576]);
        assert_eq!(w.layers[29].k_proj.shape(), &[192, 576]);
        assert!(w.embed_tokens.data().iter().all(|x| x.is_finite()));
    }
}
