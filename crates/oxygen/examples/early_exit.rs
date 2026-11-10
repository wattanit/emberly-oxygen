//! EXPERIMENT: read the logits off the residual after layer n instead of the last layer.
use emberly_oxygen::{generate, Logits, Model, Tokenizer};
use std::path::Path;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
    let tok = Tokenizer::load(&dir).unwrap();
    let prompt = "The quick brown fox jumps over the lazy dog, and then";
    let full = Model::load(&dir).unwrap();
    let reference = generate(&full, &tok, prompt, 24, &mut |_| {}).unwrap().ids;
    let mut all = tok.encode(prompt).unwrap();
    let plen = all.len();
    all.extend(&reference);
    println!("full (30 layers): {:?}\n", tok.decode(&reference).unwrap());
    println!("{:<24} {:>8} {:>8}  free-running greedy text", "kept", "agree@1", "ms/tok");
    let cases: Vec<(String, Vec<usize>)> = vec![
        ("all 30".into(), (0..30).collect()),
        ("first 24 + last 1 (25)".into(), (0..24).chain([29]).collect()),
        ("first 25 (no last)".into(), (0..25).collect()),
        ("first 23 + last 2 (25)".into(), (0..23).chain(28..30).collect()),
        ("first 22 + last 3 (25)".into(), (0..22).chain(27..30).collect()),
        ("first 20 + last 5 (25)".into(), (0..20).chain(25..30).collect()),
        ("first 15 + last 10 (25)".into(), (0..15).chain(20..30).collect()),
        ("first 10 + last 15 (25)".into(), (0..10).chain(15..30).collect()),
        ("first 5 + last 20 (25)".into(), (0..5).chain(10..30).collect()),
        ("drop layers 10-14 (25)".into(), (0..10).chain(15..30).collect()),
        ("drop every 6th (25)".into(), (0..30).filter(|i| i % 6 != 5).collect()),
        ("first 28 + last 1 (29)".into(), (0..28).chain([29]).collect()),
        ("first 26 + last 1 (27)".into(), (0..26).chain([29]).collect()),
    ];
    for (name, keep) in &cases {
        let mut m = Model::load(&dir).unwrap();
        m.keep_layer_indices(keep);
        // Teacher-forced: at each generated position, does this depth pick the full model's token?
        let mut s = m.session(false);
        let logits = s.prefill(&all[..all.len() - 1], Logits::All).unwrap();
        let v = m.config().vocab_size;
        let agree = (0..reference.len())
            .filter(|&i| argmax(&logits[(plen - 1 + i) * v..(plen + i) * v]) == reference[i])
            .count();
        let run = generate(&m, &tok, prompt, 24, &mut |_| {}).unwrap();
        let ms = run.stats.generation_tokens_per_second().map(|t| 1000.0 / t).unwrap_or(0.0);
        println!("{name:<24} {:>5}/{} {ms:>8.2}  {:?}", agree, reference.len(), tok.decode(&run.ids).unwrap());
    }
}
