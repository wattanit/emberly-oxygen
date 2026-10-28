use std::path::Path;

use crate::safetensors::SafeTensors;
use crate::weights::Weights;
use crate::{Config, Error};

/// A loaded model (Spec §3, §5): its configuration and weights, read from the
/// files as published. The forward pass is still a stub until Phase 4.
pub struct Model {
    config: Config,
    // Read by the forward pass in Phase 4.
    #[allow(dead_code)]
    weights: Weights,
}

/// One sequence being processed: the key-value cache lives here (Spec §2).
pub struct Session<'a> {
    _model: &'a Model,
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
        Ok(Model { config, weights })
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// Starts an empty sequence. Capture is off unless asked for.
    pub fn session(&self, capture: bool) -> Session<'_> {
        Session {
            _model: self,
            capture: capture.then(Capture::default),
        }
    }
}

impl Session<'_> {
    /// Feeds the next token and returns the logits for that position.
    /// Stub: returns no logits.
    pub fn step(&mut self, _token: u32) -> Result<Vec<f32>, Error> {
        Ok(Vec::new())
    }

    /// Capture of the last step; `None` when capture is off.
    pub fn capture(&self) -> Option<&Capture> {
        self.capture.as_ref()
    }
}
