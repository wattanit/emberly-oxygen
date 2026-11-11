use std::path::Path;
use std::time::Instant;

use crate::fault::Fault;
use crate::forward;
use crate::kv_cache::KvCache;
use crate::pool::Pool;
use crate::safetensors::SafeTensors;
use crate::weights::Weights;
use crate::{Config, Error};

/// A loaded model (Spec §3, §5): its configuration and weights, read from the
/// files as published.
pub struct Model {
    config: Config,
    pub(crate) weights: Weights,
    fault: Option<Fault>,
    pool: Pool,
}

/// The thread count `Model::load` uses: the cores the machine offers, or 1 when
/// that is unknown. Initial (FR-44): the release's thread sweep decides it.
fn default_threads() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
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
        Model::load_with_threads(model_dir, default_threads())
    }

    /// As [`Model::load`], with `threads` threads in all for the matrix
    /// products (FR-44, Spec §2): the calling thread counts as one, so
    /// `threads - 1` persistent workers are created, and 1 reproduces the
    /// single-threaded engine. 0 is rejected naming `threads`.
    pub fn load_with_threads(model_dir: &Path, threads: usize) -> Result<Model, Error> {
        let pool = Pool::new(threads)?;
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
        Ok(Model { config, weights, fault: None, pool })
    }

    /// The number of threads the products use, the caller included (FR-44).
    pub fn threads(&self) -> usize {
        self.pool.threads()
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// A model built from parts already in memory, for tests that need a tiny one.
    #[cfg(test)]
    pub(crate) fn from_parts(config: Config, weights: Weights) -> Model {
        Model { config, weights, fault: None, pool: Pool::new(1).expect("one thread") }
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
            &self.model.pool,
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
            &self.model.pool,
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

    // FR-44, Spec §2: 0 threads is rejected naming `threads`, before any file is read.
    #[test]
    fn zero_threads_is_rejected_naming_threads() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        let error = Model::load_with_threads(&dir, 0).err().expect("0 must be rejected");
        assert!(error.to_string().contains("threads"), "{error}");
        let error = Model::load_with_threads(Path::new("/no/such/dir"), 0).err().unwrap();
        assert!(error.to_string().contains("threads"), "rejected before the files are read: {error}");
    }

    // FR-44: the count is stored; `load` uses the cores the machine offers.
    #[test]
    fn the_thread_count_is_kept() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        for threads in [1, 3] {
            assert_eq!(Model::load_with_threads(&dir, threads).unwrap().threads(), threads);
        }
        let cores = std::thread::available_parallelism().unwrap().get();
        assert_eq!(Model::load(&dir).unwrap().threads(), cores);
    }

    // FR-43, S-6: the real model's logits are bit-identical at every thread count, through
    // `step` and through `prefill`.
    #[test]
    fn the_real_models_logits_do_not_depend_on_the_thread_count() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
        let tokens = [504u32, 3575, 282, 4649, 314, 1056, 30];
        let run = |threads: usize| {
            let model = Model::load_with_threads(&dir, threads).unwrap();
            let mut stepped = model.session(false);
            let by_step: Vec<Vec<f32>> = tokens.iter().map(|&t| stepped.step(t).unwrap()).collect();
            let by_prefill = model.session(false).prefill(&tokens, Logits::All).unwrap();
            (by_step, by_prefill)
        };
        let want = run(1);
        for threads in [2, 3, 7] {
            assert!(run(threads) == want, "threads {threads} changed a logit");
        }
    }
}
