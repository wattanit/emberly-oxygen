//! Calls every method of the engine surface once. The tokenizer is real since
//! Phase 3 and the forward pass since Phase 4; generation is still a stub, and
//! this file is removed when the correctness harness covers it (Phase 5).

use std::path::Path;

use emberly_oxygen::{generate, Model, Tokenizer};

#[test]
fn stub_surface_runs_without_panicking() {
    let dir = Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../../models/smollm2-135m"));
    let tokenizer = Tokenizer::load(dir).unwrap();
    let model = Model::load(dir).unwrap();

    let ids = tokenizer.encode("hello").unwrap();
    assert!(!ids.is_empty());
    assert_eq!(tokenizer.decode(&ids).unwrap(), "hello");

    let mut session = model.session(true);
    let logits = session.step(0).unwrap();
    assert_eq!(logits.len(), model.config().vocab_size);
    assert!(logits.iter().all(|x| x.is_finite()));
    assert_eq!(session.capture().unwrap().layers.len(), model.config().layers);
    assert!(model.session(false).capture().is_none());

    let mut pieces = Vec::new();
    let ids = generate(&model, &tokenizer, "hello", 4, &mut |t| pieces.push(t.to_owned())).unwrap();
    assert!(ids.is_empty() && pieces.is_empty());
}
