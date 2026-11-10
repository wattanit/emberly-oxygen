use std::path::Path;
use std::time::Instant;

use crate::fault::Fault;
use crate::forward;
use crate::kv_cache::KvCache;
use crate::safetensors::SafeTensors;
use crate::weights::Weights;
use crate::{Config, Error};

/// A loaded model (Spec §3, §5): its configuration and weights, read from the
/// files as published.
pub struct Model {
    config: Config,
    pub(crate) weights: Weights,
    fault: Option<Fault>,
}

/// One sequence being processed: the key-value cache lives here (Spec §2).
pub struct Session<'a> {
    model: &'a Model,
    pub(crate) cache: KvCache,
    capture: Option<Capture>,
}

/// Which logits a batched prompt pass returns (Spec §7): the last position's
/// only, which is all generation uses, or every position's, which the harness
/// checks (S-1, S-3). Every row costs a 49152-way unembedding, so only the
/// harness asks for them all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Logits {
    Last,
    All,
}

/// The residual stream of the last processed token, for the harness (Spec §7).
/// `layers` holds the raw output of every decoder layer; `final_norm` is the
/// output after the final RMSNorm.
#[derive(Debug, Default, Clone)]
pub struct Capture {
    pub embedding: Vec<f32>,
    pub layers: Vec<Vec<f32>>,
    pub final_norm: Vec<f32>,
}

impl Model {
    /// Loads `config.json` and `model.safetensors` from a model directory.
    /// Unsupported configurations and files that do not match them are
    /// rejected by name (FR-12).
    pub fn load(model_dir: &Path) -> Result<Model, Error> {
        let start = Instant::now();
        let config = Config::load(model_dir)?;
        let file = SafeTensors::open(&model_dir.join("model.safetensors"))?;
        let weights = Weights::bind(&file, &config)?;
        let c = &config;
        tracing::info!(
            dir = %model_dir.display(),
            layers = c.layers,
            hidden_size = c.hidden_size,
            heads = c.heads,
            kv_heads = c.kv_heads,
            head_dim = c.head_dim,
            intermediate_size = c.intermediate_size,
            vocab_size = c.vocab_size,
            context_length = c.context_length,
            load_ms = %format_args!("{:.3}", start.elapsed().as_secs_f64() * 1000.0),
            "model loaded"
        );
        Ok(Model { config, weights, fault: None })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// A model built from parts already in memory, for tests that need a tiny one.
    #[cfg(test)]
    pub(crate) fn from_parts(config: Config, weights: Weights) -> Model {
        Model { config, weights, fault: None }
    }

    /// Makes the forward pass deliberately wrong in one known way, so a test can
    /// check that the harness notices (Spec §17). Only exists with the
    /// `negative-controls` feature, which a normal build never enables.
    #[cfg(feature = "negative-controls")]
    pub fn inject_fault(&mut self, fault: Fault) {
        self.fault = Some(fault);
    }

    /// EXPERIMENT (branch experiment/early-exit, never for develop): keeps only
    /// the first `n` decoder layers, so the final norm and unembedding read the
    /// residual after layer `n`.
    pub fn keep_layers(&mut self, n: usize) {
        self.weights.layers.truncate(n);
        self.config.layers = self.weights.layers.len();
    }

    /// EXPERIMENT: keeps exactly the layers whose indices are listed, in order.
    pub fn keep_layer_indices(&mut self, keep: &[usize]) {
        let mut i = 0;
        self.weights.layers.retain(|_| {
            i += 1;
            keep.contains(&(i - 1))
        });
        self.config.layers = self.weights.layers.len();
    }

    /// Starts an empty sequence. Capture is off unless asked for.
    pub fn session(&self, capture: bool) -> Session<'_> {
        let c = &self.config;
        Session {
            model: self,
            cache: KvCache::new(c.layers, c.kv_heads, c.head_dim, c.context_length),
            capture: capture.then(Capture::default),
        }
    }
}

impl Session<'_> {
    /// Feeds the next token and returns the logits for that position (Spec §7).
    /// A token outside the vocabulary or a full context is rejected and leaves
    /// the session as it was.
    pub fn step(&mut self, token: u32) -> Result<Vec<f32>, Error> {
        forward::step(
            &self.model.config,
            &self.model.weights,
            &mut self.cache,
            token,
            self.capture.as_mut(),
            self.model.fault,
        )
    }

    /// Processes a whole prompt in one batched pass and returns its logits: the
    /// last position's (`vocab_size` values) or every position's, row after row
    /// (`tokens.len() × vocab_size`). The session must be empty: every caller
    /// starts a sequence with its prompt (Spec §2). An empty prompt, a token
    /// outside the vocabulary or a prompt longer than the context is rejected
    /// and leaves the session as it was (FR-18, FR-40).
    pub fn prefill(&mut self, tokens: &[u32], logits: Logits) -> Result<Vec<f32>, Error> {
        forward::prefill(
            &self.model.config,
            &self.model.weights,
            &mut self.cache,
            tokens,
            logits,
            self.capture.as_mut(),
            self.model.fault,
        )
    }

    /// Capture of the last step; `None` when capture is off.
    pub fn capture(&self) -> Option<&Capture> {
        self.capture.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use tracing::Level;

    use super::*;
    use crate::test_log::collect;

    // FR-23, Spec §14: loading logs one info event with the configuration summary and the
    // load time, and nothing at the default level.
    #[test]
    fn load_logs_the_configuration_and_load_time_at_info() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        let (model, events) = collect(Some(Level::INFO), || Model::load(&dir).unwrap());
        assert_eq!(events.len(), 1, "{events:?}");
        let event = &events[0];
        assert_eq!((event.level, event.message()), (Level::INFO, "model loaded"));
        let c = model.config();
        for (field, value) in [
            ("layers", c.layers),
            ("hidden_size", c.hidden_size),
            ("heads", c.heads),
            ("kv_heads", c.kv_heads),
            ("head_dim", c.head_dim),
            ("intermediate_size", c.intermediate_size),
            ("vocab_size", c.vocab_size),
            ("context_length", c.context_length),
        ] {
            assert_eq!(event.get(field), Some(value.to_string().as_str()), "{field}");
        }
        assert!(event.get("load_ms").unwrap().parse::<f64>().unwrap() > 0.0);
        assert!(event.get("dir").unwrap().contains("smollm2-135m"));

        assert!(collect(Some(Level::WARN), || Model::load(&dir).unwrap()).1.is_empty());
    }
}
