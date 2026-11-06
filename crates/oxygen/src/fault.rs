//! Negative controls (Spec §17): deliberate faults that make the engine wrong
//! in a known way, so a test can check that the harness notices.
//!
//! Without the `negative-controls` feature `Fault` has no values at all, so a
//! normal build contains no faulty arithmetic and nothing can switch one on;
//! the hooks below then compile to nothing. With the feature, a `Model` can be
//! given one fault (`Model::inject_fault`), and with none injected the engine
//! computes exactly what it does without the feature.

/// A way of making the forward pass wrong.
#[cfg(feature = "negative-controls")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// RoPE pairs dimension `2i` with `2i + 1` instead of `i` with `i + d/2`.
    InterleavedRope,
    /// The first logit at one position is 2e-3 too high: just over S-1's 1e-3.
    PerturbedLogit { position: usize },
    /// The logits come from the residual stream without the final RMSNorm.
    NoFinalNorm,
}

/// Uninhabited: there is no way to build one.
#[cfg(not(feature = "negative-controls"))]
#[derive(Debug, Clone, Copy)]
pub(crate) enum Fault {}

#[cfg(feature = "negative-controls")]
impl Fault {
    pub(crate) fn interleaves_rope(self) -> bool {
        self == Fault::InterleavedRope
    }

    pub(crate) fn skips_final_norm(self) -> bool {
        self == Fault::NoFinalNorm
    }
}

#[cfg(not(feature = "negative-controls"))]
impl Fault {
    pub(crate) fn interleaves_rope(self) -> bool {
        match self {}
    }

    pub(crate) fn skips_final_norm(self) -> bool {
        match self {}
    }
}

/// Adds 2e-3, just over S-1's 1e-3, to the first logit at the faulty position.
#[cfg(feature = "negative-controls")]
pub(crate) fn perturb_logits(fault: Option<Fault>, position: usize, logits: &mut [f32]) {
    if fault == Some(Fault::PerturbedLogit { position }) {
        logits[0] += 2e-3;
    }
}

#[cfg(not(feature = "negative-controls"))]
pub(crate) fn perturb_logits(_fault: Option<Fault>, _position: usize, _logits: &mut [f32]) {}

/// RoPE with the wrong pairing: dimension `2i` rotates with `2i + 1`, at the
/// same angles as the correct pairing gives pair `i`.
#[cfg(feature = "negative-controls")]
pub(crate) fn rope_interleaved(_fault: Fault, x: &mut [f32], position: usize, heads: usize, head_dim: usize, theta: f32) {
    assert_eq!(x.len(), heads * head_dim);
    let angles = crate::ops::rope_angles(position, head_dim, theta);
    for head in x.chunks_exact_mut(head_dim) {
        for (i, &(cos, sin)) in angles.iter().enumerate() {
            let (a, b) = (head[2 * i], head[2 * i + 1]);
            head[2 * i] = a * cos - b * sin;
            head[2 * i + 1] = b * cos + a * sin;
        }
    }
}

#[cfg(not(feature = "negative-controls"))]
pub(crate) fn rope_interleaved(fault: Fault, _x: &mut [f32], _position: usize, _heads: usize, _head_dim: usize, _theta: f32) {
    match fault {}
}
