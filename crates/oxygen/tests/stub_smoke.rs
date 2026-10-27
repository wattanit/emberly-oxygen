//! Calls every method of the stub surface once; removed when the real
//! implementations land and the correctness harness covers them.

use std::path::Path;

use emberly_oxygen::{generate, Model, Tokenizer};

#[test]
fn stub_surface_runs_without_panicking() {
    let dir = Path::new("models/none");
    let tokenizer = Tokenizer::load(dir).unwrap();
    let model = Model::load(dir).unwrap();

    assert!(tokenizer.encode("hello").unwrap().is_empty());
    assert_eq!(tokenizer.decode(&[1, 2, 3]).unwrap(), "");

    let mut session = model.session(true);
    assert!(session.step(0).unwrap().is_empty());
    assert!(session.capture().unwrap().layers.is_empty());
    assert!(model.session(false).capture().is_none());

    let mut pieces = Vec::new();
    let ids = generate(&model, &tokenizer, "hello", 4, &mut |t| pieces.push(t.to_owned())).unwrap();
    assert!(ids.is_empty() && pieces.is_empty());
}
