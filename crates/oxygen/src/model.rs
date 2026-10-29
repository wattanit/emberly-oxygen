use std::path::Path;

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
        let config = Config::load(model_dir)?;
        let file = SafeTensors::open(&model_dir.join("model.safetensors"))?;
        let weights = Weights::bind(&file, &config)?;
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

    /// Capture of the last step; `None` when capture is off.
    pub fn capture(&self) -> Option<&Capture> {
        self.capture.as_ref()
    }
}
