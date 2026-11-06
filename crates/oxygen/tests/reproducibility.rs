//! Reproducibility (S-6): the same input gives the same output, run after run
//! in one process and across separate loads of the model. Verifies S-6.
//!
//! S-6 allows logits to differ by up to S-1 and continuations to differ at
//! near-ties, because a later version may add in parallel. This engine is
//! single-threaded and adds in index order, so the runs are expected to be
//! bit-identical; the bound S-6 states is asserted, whether they are
//! identical is printed (`cargo test --test reproducibility -- --nocapture`),
//! and the continuations are required to be exactly equal.

mod common;

use common::compare::LOGIT_TOLERANCE;
use common::fixtures::{discover, workspace_root};
use emberly_oxygen::{generate, Model, Tokenizer};

const NEW_TOKENS: usize = 16;

/// Every position's logits for the prompt's token IDs, from a fresh session.
fn prompt_logits(model: &Model, ids: &[u32]) -> Vec<f32> {
    let mut session = model.session(false);
    ids.iter().flat_map(|&id| session.step(id).unwrap()).collect()
}

// S-6, S-4: two runs on one loaded model and a run on a second load agree within S-1 at every
// position, and their greedy continuations are identical.
#[test]
fn runs_and_loads_give_the_same_output() {
    let root = workspace_root();
    let all = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}"));
    let fixtures = &all[0];
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == "english-short").unwrap();
    let dir = root.join("models").join(&fixtures.manifest.model_id);
    let tokenizer = Tokenizer::load(&dir).unwrap();

    let first_load = Model::load(&dir).unwrap();
    let second_load = Model::load(&dir).unwrap();
    let runs = [("first run", &first_load), ("second run, same load", &first_load), ("run on a second load", &second_load)];

    let (_, model) = runs[0];
    let logits = prompt_logits(model, &prompt.token_ids);
    let continuation = generate(model, &tokenizer, &prompt.text, NEW_TOKENS, &mut |_| {}).unwrap().ids;
    assert_eq!(continuation.len(), NEW_TOKENS);

    for (name, model) in &runs[1..] {
        let again = prompt_logits(model, &prompt.token_ids);
        assert_eq!(again.len(), logits.len());
        let max_diff = logits.iter().zip(&again).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
        assert!(max_diff <= LOGIT_TOLERANCE, "{name}: logits differ by {max_diff:e}, more than S-1 allows");
        let identical = logits.iter().zip(&again).all(|(a, b)| a.to_bits() == b.to_bits());
        println!(
            "{name}: {} logits over {} positions, largest difference {max_diff:e}, bit-identical: {identical}",
            again.len(),
            prompt.token_ids.len()
        );

        let ids = generate(model, &tokenizer, &prompt.text, NEW_TOKENS, &mut |_| {}).unwrap().ids;
        assert_eq!(ids, continuation, "{name}: the continuation differs");
    }
}
