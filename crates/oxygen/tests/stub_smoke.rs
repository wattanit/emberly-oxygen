//! Calls every method of the engine surface once. The tokenizer is real since
//! Phase 3; the forward pass and generation are still stubs, and this file is
//! removed when the correctness harness covers them.

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
    assert!(session.step(0).unwrap().is_empty());
    assert!(session.capture().unwrap().layers.is_empty());
    assert!(model.session(false).capture().is_none());

    let mut pieces = Vec::new();
    let ids = generate(&model, &tokenizer, "hello", 4, &mut |t| pieces.push(t.to_owned())).unwrap();
    assert!(ids.is_empty() && pieces.is_empty());
}
