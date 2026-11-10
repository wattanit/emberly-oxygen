//! EXPERIMENT: logit lens. After each layer, apply the final norm and the unembedding
//! to the residual of the last prompt token and show what it would predict.
use emberly_oxygen::{Logits, Model, Tokenizer};
use std::path::Path;

fn softmax_top(logits: &[f32], k: usize) -> (Vec<(usize, f32)>, Vec<f32>) {
    let max = logits.iter().cloned().fold(f32::MIN, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let p: Vec<f32> = exps.iter().map(|e| e / sum).collect();
    let mut idx: Vec<usize> = (0..p.len()).collect();
    idx.sort_by(|&a, &b| p[b].partial_cmp(&p[a]).unwrap());
    (idx[..k].iter().map(|&i| (i, p[i])).collect(), p)
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
    let tok = Tokenizer::load(&dir).unwrap();
    let prompts = ["The capital of France is", "The quick brown fox jumps over the lazy dog, and then", "Q: What is 12 plus 30?\nA:", "def fibonacci(n):\n    if n < 2:\n        return n\n    return"];
    for prompt in prompts {
        let ids = tok.encode(prompt).unwrap();
        let mut final_top = 0usize;
        println!("=== {prompt:?} ({} tokens) ===", ids.len());
        println!("{:>5}  {:<46} {:>9}", "after", "top-3 predictions (probability)", "final's top");
        for n in (0..=30).rev() {
            let mut m = Model::load(&dir).unwrap();
            m.keep_layers(n);
            let mut s = m.session(false);
            let logits = s.prefill(&ids, Logits::Last).unwrap();
            let (top, p) = softmax_top(&logits, 3);
            if n == 30 { final_top = top[0].0; }
            let show = |&(i, pr): &(usize, f32)| format!("{:?} {:.0}%", tok.decode(&[i as u32]).unwrap(), pr * 100.0);
            let line = top.iter().map(show).collect::<Vec<_>>().join(", ");
            println!("{n:>5}  {line:<46} {:>8.1}%", p[final_top] * 100.0);
        }
        println!();
    }
}
