use std::path::Path;

use crate::Error;

/// A loaded model (Spec §3, §5). Stub: loads nothing.
pub struct Model;

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
    pub fn load(_model_dir: &Path) -> Result<Model, Error> {
        Ok(Model)
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
