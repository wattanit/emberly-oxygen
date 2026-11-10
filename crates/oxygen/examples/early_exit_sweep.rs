//! EXPERIMENT: teacher-forced agreement with the full model, over many prompts, for layer subsets.
use emberly_oxygen::{generate, Logits, Model, Tokenizer};
use std::path::Path;

const PROMPTS: [&str; 12] = [
    "The quick brown fox jumps over the lazy dog, and then",
    "The capital of France is",
    "In 1969, the first humans landed on the Moon. The mission was called",
    "def fibonacci(n):\n    if n < 2:\n        return n\n    return",
    "Once upon a time, in a small village by the sea, there lived an old fisherman who",
    "The three primary colors are red, blue, and",
    "Photosynthesis is the process by which plants convert",
    "Q: What is 12 plus 30?\nA:",
    "Dear Sir or Madam,\n\nI am writing to complain about",
    "The mitochondria is the powerhouse of the",
    "Water boils at 100 degrees Celsius at sea level, but on a high mountain it",
    "1, 2, 3, 4, 5, 6,",
];
const NEW: usize = 32;

fn argmax(v: &[f32]) -> u32 {
    v.iter().enumerate().fold((0, f32::MIN), |b, (i, &x)| if x > b.1 { (i, x) } else { b }).0 as u32
}

fn main() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../models/smollm2-135m");
    let tok = Tokenizer::load(&dir).unwrap();
    let full = Model::load(&dir).unwrap();
    let cases_in: Vec<(Vec<u32>, usize)> = PROMPTS
        .iter()
        .map(|p| {
            let r = generate(&full, &tok, p, NEW, &mut |_| {}).unwrap().ids;
            let mut all = tok.encode(p).unwrap();
            let plen = all.len();
            all.extend(&r);
            (all, plen)
        })
        .collect();
    let total: usize = cases_in.iter().map(|(a, p)| a.len() - p).sum();
    let score = |keep: &[usize]| -> (usize, Vec<usize>) {
        let mut m = Model::load(&dir).unwrap();
        m.keep_layer_indices(keep);
        let v = m.config().vocab_size;
        let mut per = vec![];
        for (all, plen) in &cases_in {
            let mut s = m.session(false);
            let lg = s.prefill(&all[..all.len() - 1], Logits::All).unwrap();
            let n = all.len() - plen;
            per.push((0..n).filter(|&i| argmax(&lg[(plen - 1 + i) * v..(plen + i) * v]) == all[plen + i]).count());
        }
        (per.iter().sum(), per)
    };
    let pct = |a: usize| 100.0 * a as f64 / total as f64;
    println!("{} prompts, {} scored tokens\n", PROMPTS.len(), total);

    let named: Vec<(String, Vec<usize>)> = vec![
        ("all 30".into(), (0..30).collect()),
        ("first 28 (no last 2)".into(), (0..28).collect()),
        ("first 25".into(), (0..25).collect()),
        ("first 24 + last 1".into(), (0..24).chain([29]).collect()),
        ("first 20 + last 5".into(), (0..20).chain(25..30).collect()),
        ("first 15 + last 10".into(), (0..15).chain(20..30).collect()),
        ("first 10 + last 15".into(), (0..10).chain(15..30).collect()),
        ("drop every 6th".into(), (0..30).filter(|i| i % 6 != 5).collect()),
    ];
    for (n, k) in &named {
        let (s, per) = score(k);
        println!("{n:<22} {:>5.1}%  per-prompt {:?}", pct(s), per);
    }
    println!("\ndrop ONE layer (agreement %):");
    for l in 0..30 {
        let k: Vec<usize> = (0..30).filter(|&i| i != l).collect();
        println!("  drop {l:>2}: {:>5.1}%", pct(score(&k).0));
    }
    println!("\ndrop a block of 5 starting at s (agreement %):");
    for s in 0..=25 {
        let k: Vec<usize> = (0..30).filter(|&i| i < s || i >= s + 5).collect();
        println!("  drop {s:>2}..{:>2}: {:>5.1}%", s + 4, pct(score(&k).0));
    }
}
