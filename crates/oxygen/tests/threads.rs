//! The engine does not depend on the thread count (FR-43, S-6, Spec §11): the
//! logits and every intermediate are bit-identical at 1, 2, 3, 7 threads and at
//! the default, through the batched pass and through one token at a time. The
//! counts 3 and 7 divide none of the model's sizes evenly, on purpose. The
//! bound is exactly 0, because no sum is split across threads.

mod common;

use common::fixtures::{discover, require_model_dir, workspace_root};
use common::harness::{engine_difference, run_engine, run_engine_by_steps, EngineRun};
use emberly_oxygen::{Fault, Model};

/// Thread counts compared with one thread; `None` is the default count.
const COUNTS: [Option<usize>; 5] = [Some(1), Some(2), Some(3), Some(7), None];

fn load() -> (common::fixtures::Fixtures, std::path::PathBuf) {
    let root = workspace_root();
    let fixtures = discover(&root.join("fixtures")).unwrap_or_else(|e| panic!("{e}")).into_iter().next().expect("no fixtures");
    let model_dir = root.join("models").join(&fixtures.manifest.model_id);
    require_model_dir(&model_dir).unwrap_or_else(|e| panic!("{e}"));
    (fixtures, model_dir)
}

fn model(dir: &std::path::Path, threads: Option<usize>) -> Model {
    match threads {
        Some(n) => Model::load_with_threads(dir, n),
        None => Model::load(dir),
    }
    .unwrap_or_else(|e| panic!("{e}"))
}

fn same(a: &EngineRun, b: &EngineRun) -> bool {
    let d = engine_difference(a, b);
    d.logits == 0.0 && d.intermediates == 0.0
}

// FR-43, S-6: every prompt of the set, through the batched pass, gives exactly the one-thread
// logits and intermediates at every count (the long prompts' decode-sized attention is
// threaded in the test below).
#[test]
fn fr43_s6_the_batched_pass_is_bit_identical_at_every_thread_count() {
    let (fixtures, dir) = load();
    let one = model(&dir, Some(1));
    let baseline: Vec<EngineRun> = fixtures.manifest.prompts.iter().map(|p| run_engine(&one, &p.token_ids).unwrap()).collect();
    for threads in COUNTS.into_iter().skip(1) {
        let m = model(&dir, threads);
        for (prompt, base) in fixtures.manifest.prompts.iter().zip(&baseline) {
            let got = run_engine(&m, &prompt.token_ids).unwrap();
            assert!(same(&got, base), "{}: {threads:?} threads changed a value ({:?})", prompt.id, engine_difference(&got, base).logits);
        }
    }
}

// FR-43, S-6: one token at a time, on the shortest prompt and on the longest (where the
// cache passes the attention threshold, so the heads are divided as well as the products).
#[test]
fn fr43_s6_one_token_at_a_time_is_bit_identical_at_every_thread_count() {
    let (fixtures, dir) = load();
    let prompts = &fixtures.manifest.prompts;
    let shortest = prompts.iter().min_by_key(|p| p.token_ids.len()).unwrap();
    let longest = prompts.iter().max_by_key(|p| p.token_ids.len()).unwrap();
    assert!(longest.token_ids.len() > 300, "the longest prompt must pass the attention threshold");
    let one = model(&dir, Some(1));
    for prompt in [shortest, longest] {
        let base = run_engine_by_steps(&one, &prompt.token_ids).unwrap();
        for threads in COUNTS {
            let got = run_engine_by_steps(&model(&dir, threads), &prompt.token_ids).unwrap();
            assert!(same(&got, &base), "{}: {threads:?} threads changed a value", prompt.id);
        }
    }
}

// FR-43, HC-4: the comparison can fail. With the last worker's range lost, 4 threads differ from
// one thread by far more than 0 on both the logits and the intermediates.
#[test]
fn fr43_a_skipped_range_is_over_the_bound() {
    let (fixtures, dir) = load();
    let prompt = fixtures.manifest.prompts.iter().find(|p| p.id == "english-short").expect("prompt missing");
    let base = run_engine(&model(&dir, Some(1)), &prompt.token_ids).unwrap();
    let mut faulted = model(&dir, Some(4));
    faulted.inject_fault(Fault::SkippedRows);
    let d = engine_difference(&run_engine(&faulted, &prompt.token_ids).unwrap(), &base);
    assert!(d.logits > 1e-3 && d.intermediates > 1e-3, "{:e} {:e}", d.logits, d.intermediates);
}
