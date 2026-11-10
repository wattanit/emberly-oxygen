//! The batched pass against the one-token pass (FR-40, FR-41, Spec §11). This
//! is not one of the S-judged checks: those compare the engine with the 64-bit
//! reference. It compares the engine with itself, `prefill` against one `step`
//! per token, because the two do the same arithmetic in a different loop order,
//! so a difference above the bound is a defect in one of them and not rounding.
//! The largest differences are printed (`cargo test --test batched -- --nocapture`).

mod common;

use std::thread;

use common::fixtures::{discover, require_model_dir, workspace_root};
use common::harness::{engine_difference, run_engine, run_engine_by_steps, EngineRun};
use emberly_oxygen::{Fault, Model};

/// The bound on the largest absolute logit difference (Spec §11, initial). The
/// release measured 0 on every prompt (plan 0.4.0 §5), so this guards against a
/// defect and is not a measured margin.
const LOGIT_BOUND: f64 = 1e-5;

fn run(logits: Vec<f32>, embeddings: Vec<f32>, layers: Vec<Vec<f32>>, final_norm: Vec<f32>) -> EngineRun {
    EngineRun { logits, embeddings, layers, final_norm }
}

// FR-41: two identical runs differ by nothing.
#[test]
fn fr41_identical_runs_differ_by_nothing() {
    let a = run(vec![1.0, 2.0], vec![3.0], vec![vec![4.0], vec![5.0]], vec![6.0]);
    let b = run(vec![1.0, 2.0], vec![3.0], vec![vec![4.0], vec![5.0]], vec![6.0]);
    let d = engine_difference(&a, &b);
    assert_eq!((d.logits, d.intermediates), (0.0, 0.0));
}

// FR-41: the logit difference is the largest absolute one, measured exactly (the values are widened
// to f64 before the subtraction, so the difference adds no rounding of its own).
#[test]
fn fr41_the_logit_difference_is_the_largest_absolute_one() {
    let a = run(vec![1.0, 2.0, 3.0], vec![], vec![], vec![]);
    let b = run(vec![1.0, 2.0 + 0.5, 3.0 - 0.25], vec![], vec![], vec![]);
    assert_eq!(engine_difference(&a, &b).logits, 0.5);
}

// FR-41: a difference in the embeddings, any layer or the final norm shows as an intermediate difference,
// and not as a logit one.
#[test]
fn fr41_an_intermediate_difference_is_measured_in_any_tensor() {
    let base = || run(vec![1.0], vec![1.0], vec![vec![2.0], vec![3.0]], vec![4.0]);
    for (name, changed) in [
        ("embeddings", run(vec![1.0], vec![1.5], vec![vec![2.0], vec![3.0]], vec![4.0])),
        ("layer 1", run(vec![1.0], vec![1.0], vec![vec![2.0], vec![3.25]], vec![4.0])),
        ("final norm", run(vec![1.0], vec![1.0], vec![vec![2.0], vec![3.0]], vec![4.75])),
    ] {
        let d = engine_difference(&base(), &changed);
        assert_eq!(d.logits, 0.0, "{name}");
        assert!(d.intermediates > 0.0, "{name}");
    }
}

// FR-41: runs of different shapes are infinitely far apart, so a pass that returns too few values fails
// the bound instead of comparing a prefix.
#[test]
fn fr41_runs_of_different_shapes_are_infinitely_far_apart() {
    let a = run(vec![1.0, 2.0], vec![], vec![], vec![]);
    let b = run(vec![1.0], vec![], vec![], vec![]);
    assert_eq!(engine_difference(&a, &b).logits, f64::INFINITY);
    let c = run(vec![], vec![], vec![vec![1.0]], vec![]);
    let d = run(vec![], vec![], vec![vec![1.0], vec![1.0]], vec![]);
    assert_eq!(engine_difference(&c, &d).intermediates, f64::INFINITY);
}

fn load() -> (common::fixtures::Fixtures, std::path::PathBuf) {
    let root = workspace_root();
    let fixtures = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}")).into_iter().next().expect("no fixtures");
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    (fixtures, model_dir)
}

// FR-41, Spec §11: on every prompt of the set the batched pass and the one-token pass agree on every
// position's logits within the bound. Each prompt runs in a thread of its own (test code only; each
// has its own sessions), and the table is printed in prompt order.
#[test]
fn fr41_batched_and_one_token_passes_agree_on_every_prompt() {
    let (fixtures, model_dir) = load();
    let model = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let rows: Vec<(String, usize, f64, f64)> = thread::scope(|scope| {
        let handles: Vec<_> = fixtures
            .manifest
            .prompts
            .iter()
            .map(|prompt| {
                let model = &model;
                scope.spawn(move || {
                    let batched = run_engine(model, &prompt.token_ids).unwrap();
                    let stepped = run_engine_by_steps(model, &prompt.token_ids).unwrap();
                    let d = engine_difference(&batched, &stepped);
                    (prompt.id.clone(), prompt.token_ids.len(), d.logits, d.intermediates)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    println!("{:<18} {:>9} {:>18} {:>22}", "prompt", "positions", "largest logit diff", "largest interm. diff");
    for (id, n, logits, intermediates) in &rows {
        println!("{id:<18} {n:>9} {logits:>18e} {intermediates:>22e}");
    }
    assert!(!rows.is_empty());
    for (id, _, logits, _) in &rows {
        assert!(*logits <= LOGIT_BOUND, "{id}: the batched and one-token logits differ by {logits:e}, over {LOGIT_BOUND:e}");
    }
}

// FR-41, HC-4: the comparison can fail. A batched pass with no causal mask differs from the one-token
// pass by far more than the bound, on both the logits and the intermediates.
#[test]
fn fr41_a_faulty_batched_pass_is_over_the_bound() {
    let (fixtures, model_dir) = load();
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == "english-short").expect("prompt missing");
    let mut faulted = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    faulted.inject_fault(Fault::UnmaskedAttention);
    let clean = Model::load(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    let d = engine_difference(&run_engine(&faulted, &prompt.token_ids).unwrap(), &run_engine_by_steps(&clean, &prompt.token_ids).unwrap());
    assert!(d.logits > LOGIT_BOUND * 1e3, "logits differ by only {:e}", d.logits);
    assert!(d.intermediates > LOGIT_BOUND * 1e3, "intermediates differ by only {:e}", d.intermediates);
}
