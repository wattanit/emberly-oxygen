//! EXPERIMENT: free-running greedy text for several layer subsets and prompts.
use emberly_oxygen::{generate, Model, Tokenizer};
use std::path::Path;

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
    let tok = Tokenizer::load(&dir).unwrap();
    let prompts = [
        "The capital of France is",
        "Once upon a time, in a small village by the sea, there lived an old fisherman who",
        "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
        "Q: What is 12 plus 30?\nA:",
        "Photosynthesis is the process by which plants convert",
        "1, 2, 3, 4, 5, 6,",
    ];
    let cases: Vec<(&str, Vec<usize>)> = vec![
        ("all 30 layers", (0..30).collect()),
        ("first 28", (0..28).collect()),
        ("first 25", (0..25).collect()),
        ("first 24 + last 1", (0..24).chain([29]).collect()),
        ("first 15 + last 10 (drop 15-19)", (0..15).chain(20..30).collect()),
        ("drop 3-7", (0..30).filter(|i| !(3..8).contains(i)).collect()),
        ("drop 12-16", (0..30).filter(|i| !(12..17).contains(i)).collect()),
        ("drop layer 11 only", (0..30).filter(|&i| i != 11).collect()),
    ];
    for p in prompts {
        println!("PROMPT {p:?}");
        for (name, keep) in &cases {
            let mut m = Model::load(&dir).unwrap();
            m.keep_layer_indices(keep);
            let ids = generate(&m, &tok, p, 40, &mut |_| {}).unwrap().ids;
            println!("  [{name}] {:?}", tok.decode(&ids).unwrap());
        }
        println!();
    }
}
