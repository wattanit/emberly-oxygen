//! The worker pool behind the threaded products and attention (FR-43, FR-44,
//! Spec §2, §4). Workers are `std::thread`s created once and kept waiting, in
//! safe code: a job is an `Arc`ed closure over the range of output elements a
//! worker owns, and its result comes back over a channel, so nothing borrowed
//! crosses a thread and no `unsafe` is needed (owner's decision, 2026-11-11).
//!
//! The pool only divides *output elements*: a range is a run of whole outputs,
//! never a part of a sum, so the result of a job is the same at every thread
//! count (FR-43).

use std::any::Any;
use std::ops::Range;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::Error;

/// A job: computes the outputs of one contiguous range of output elements.
/// It returns `range.len() * width` values, `width` per element.
pub(crate) type Job = Arc<dyn Fn(Range<usize>) -> Vec<f32> + Send + Sync>;

pub(crate) struct Pool {
    threads: usize,
    /// One lock for a whole call, so two callers sharing a pool cannot
    /// interleave their hand-overs and receive each other's results.
    workers: Mutex<Vec<Worker>>,
    /// The negative control `Fault::SkippedRows` (Spec §17): the last worker's
    /// answer is replaced by zeros. Only exists with the `negative-controls` feature.
    #[cfg(feature = "negative-controls")]
    skip_last: std::sync::atomic::AtomicBool,
}

type Outcome = Result<Vec<f32>, Box<dyn Any + Send>>;

struct Worker {
    jobs: Option<Sender<(Job, Range<usize>)>>,
    results: Receiver<Outcome>,
    handle: Option<JoinHandle<()>>,
}

/// Divides `0..n` into `parts` contiguous ranges whose sizes differ by at most
/// one, the larger ones first. Some ranges are empty when `n < parts`.
pub(crate) fn split(n: usize, parts: usize) -> Vec<Range<usize>> {
    let (each, extra) = (n / parts, n % parts);
    let mut start = 0;
    (0..parts)
        .map(|i| {
            let end = start + each + usize::from(i < extra);
            let range = start..end;
            start = end;
            range
        })
        .collect()
}

impl Pool {
    /// A pool of `threads` threads in all: the caller counts as one, so
    /// `threads - 1` workers are created. 0 is rejected naming `threads`.
    pub(crate) fn new(threads: usize) -> Result<Pool, Error> {
        if threads == 0 {
            return Err(Error::Rejected("threads must be at least 1, got 0".into()));
        }
        let workers = (1..threads)
            .map(|_| {
                let (jobs, job_rx) = mpsc::channel::<(Job, Range<usize>)>();
                let (result_tx, results) = mpsc::channel::<Outcome>();
                let handle = thread::spawn(move || {
                    while let Ok((job, range)) = job_rx.recv() {
                        let outcome = catch_unwind(AssertUnwindSafe(|| job(range)));
                        // The job holds handles on shared data (the key-value cache);
                        // dropping it before the answer goes back means that when the
                        // caller has every answer, no worker still holds any.
                        drop(job);
                        if result_tx.send(outcome).is_err() {
                            break;
                        }
                    }
                });
                Worker { jobs: Some(jobs), results, handle: Some(handle) }
            })
            .collect();
        Ok(Pool {
            threads,
            workers: Mutex::new(workers),
            #[cfg(feature = "negative-controls")]
            skip_last: std::sync::atomic::AtomicBool::new(false),
        })
    }

    pub(crate) fn threads(&self) -> usize {
        self.threads
    }

    /// Makes every call lose its last worker's range (`Fault::SkippedRows`).
    #[cfg(feature = "negative-controls")]
    pub(crate) fn skip_last_range(&self) {
        self.skip_last.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Runs `job` over `0..n` divided into `threads` ranges, the calling
    /// thread taking the first, and returns the outputs in range order.
    /// Every worker's result is collected before anything is returned or
    /// re-raised, so a panic in one range leaves the pool ready for the next call.
    pub(crate) fn run(&self, n: usize, job: &Job) -> Vec<f32> {
        // No more parts than elements: a worker with nothing to do is not woken.
        let ranges = split(n, self.threads.min(n).max(1));
        let workers = self.workers.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let workers = &workers[..ranges.len() - 1];
        for (worker, range) in workers.iter().zip(&ranges[1..]) {
            worker.jobs.as_ref().expect("pool is alive").send((job.clone(), range.clone())).expect("worker is alive");
        }
        let mine = catch_unwind(AssertUnwindSafe(|| job(ranges[0].clone())));
        let theirs: Vec<Outcome> = workers.iter().map(|w| w.results.recv().expect("worker answers")).collect();
        let mut out = Vec::new();
        #[cfg(feature = "negative-controls")]
        let last = theirs.len();
        for (_i, outcome) in std::iter::once(mine).chain(theirs).enumerate() {
            match outcome {
                #[cfg(feature = "negative-controls")]
                Ok(part) if _i == last && last > 0 && self.skip_last.load(std::sync::atomic::Ordering::SeqCst) => {
                    out.extend(std::iter::repeat(0.0).take(part.len()))
                }
                Ok(part) => out.extend(part),
                Err(payload) => resume_unwind(payload),
            }
        }
        out
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        let workers = self.workers.get_mut().unwrap_or_else(|poisoned| poisoned.into_inner());
        for worker in workers.iter_mut() {
            worker.jobs = None;
        }
        for worker in workers.iter_mut() {
            if let Some(handle) = worker.handle.take() {
                let _ = handle.join();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::thread::{self, ThreadId};

    use super::*;

    fn job(f: impl Fn(Range<usize>) -> Vec<f32> + Send + Sync + 'static) -> Job {
        Arc::new(f)
    }

    // FR-43, Spec §4: ranges are contiguous, cover 0..n once, differ in size by at most one.
    #[test]
    fn split_covers_everything_once_with_sizes_within_one() {
        for n in [0, 1, 2, 7, 16, 192, 1536, 49152] {
            for parts in [1, 2, 3, 4, 7, 8, 16] {
                let ranges = split(n, parts);
                assert_eq!(ranges.len(), parts);
                assert_eq!(ranges[0].start, 0);
                assert_eq!(ranges[parts - 1].end, n);
                for pair in ranges.windows(2) {
                    assert_eq!(pair[0].end, pair[1].start, "contiguous");
                }
                let sizes: Vec<usize> = ranges.iter().map(|r| r.len()).collect();
                assert!(sizes.iter().max().unwrap() - sizes.iter().min().unwrap() <= 1, "{n} {parts} {sizes:?}");
                assert!(sizes.windows(2).all(|p| p[0] >= p[1]), "larger first");
            }
        }
    }

    // FR-44, Spec §2: 0 is rejected naming `threads`.
    #[test]
    fn zero_threads_is_rejected_naming_threads() {
        let error = Pool::new(0).err().expect("0 must be rejected");
        assert!(error.to_string().contains("threads"), "{error}");
        assert!(error.to_string().contains('0'), "{error}");
    }

    // FR-43: every element is computed exactly once and the outputs come back in order,
    // at every count, including counts that do not divide the work and exceed it.
    #[test]
    fn every_element_runs_once_and_in_order() {
        for threads in [1, 2, 3, 4, 7, 16] {
            let pool = Pool::new(threads).unwrap();
            assert_eq!(pool.threads(), threads);
            for n in [0, 1, 5, 100] {
                let calls = Arc::new(AtomicUsize::new(0));
                let seen = calls.clone();
                let out = pool.run(
                    n,
                    &job(move |r| {
                        seen.fetch_add(r.len(), Ordering::SeqCst);
                        r.map(|i| i as f32 * 2.0).collect()
                    }),
                );
                assert_eq!(calls.load(Ordering::SeqCst), n, "threads {threads} n {n}");
                assert_eq!(out, (0..n).map(|i| i as f32 * 2.0).collect::<Vec<_>>());
            }
        }
    }

    // FR-43: more than one value per element (the attention's head_dim) keeps its order.
    #[test]
    fn a_range_may_return_several_values_per_element() {
        let pool = Pool::new(3).unwrap();
        let out = pool.run(7, &job(|r| r.flat_map(|i| [i as f32, i as f32 + 0.5]).collect()));
        let want: Vec<f32> = (0..7).flat_map(|i| [i as f32, i as f32 + 0.5]).collect();
        assert_eq!(out, want);
    }

    // FR-44, Spec §2: a count of 1 runs on the calling thread and spawns nothing.
    #[test]
    fn one_thread_runs_on_the_caller() {
        let pool = Pool::new(1).unwrap();
        let ids = Arc::new(Mutex::new(Vec::<ThreadId>::new()));
        let sink = ids.clone();
        pool.run(10, &job(move |r| {
            sink.lock().unwrap().push(thread::current().id());
            r.map(|i| i as f32).collect()
        }));
        assert_eq!(*ids.lock().unwrap(), vec![thread::current().id()]);
    }

    // FR-43, Spec §4: the calling thread takes a range itself, and the others run elsewhere.
    #[test]
    fn the_caller_takes_one_range_and_workers_the_rest() {
        let pool = Pool::new(4).unwrap();
        let ids = Arc::new(Mutex::new(Vec::<ThreadId>::new()));
        let sink = ids.clone();
        pool.run(100, &job(move |r| {
            sink.lock().unwrap().push(thread::current().id());
            r.map(|i| i as f32).collect()
        }));
        let ids = ids.lock().unwrap();
        assert_eq!(ids.len(), 4);
        assert_eq!(ids.iter().filter(|id| **id == thread::current().id()).count(), 1);
    }

    // Spec §2: workers are persistent, so many calls reuse the same threads.
    #[test]
    fn workers_are_reused_across_calls() {
        let pool = Pool::new(3).unwrap();
        let ids = Arc::new(Mutex::new(std::collections::BTreeSet::new()));
        for _ in 0..50 {
            let sink = ids.clone();
            pool.run(30, &job(move |r| {
                sink.lock().unwrap().insert(format!("{:?}", thread::current().id()));
                r.map(|i| i as f32).collect()
            }));
        }
        assert_eq!(ids.lock().unwrap().len(), 3, "the caller and two workers, no more");
    }

    // Spec §2: workers are joined when the pool drops, so none outlives it.
    #[test]
    fn workers_are_joined_on_drop() {
        let probe = Arc::new(());
        let pool = Pool::new(4).unwrap();
        let held = probe.clone();
        pool.run(8, &job(move |r| {
            let _ = &held;
            r.map(|i| i as f32).collect()
        }));
        drop(pool);
        assert_eq!(Arc::strong_count(&probe), 1, "no worker still holds the job");
    }

    // FR-43: a panic in a worker reaches the caller and does not hang.
    #[test]
    #[should_panic(expected = "worker boom")]
    fn a_worker_panic_reaches_the_caller() {
        let pool = Pool::new(4).unwrap();
        pool.run(100, &job(|r| {
            if r.start != 0 {
                panic!("worker boom");
            }
            r.map(|i| i as f32).collect()
        }));
    }

    // A pool survives a panicked call: the next call still works.
    #[test]
    fn the_pool_is_usable_after_a_worker_panic() {
        let pool = Pool::new(3).unwrap();
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            pool.run(9, &job(|r| if r.start == 3 { panic!("once") } else { r.map(|i| i as f32).collect() }));
        }));
        assert!(failed.is_err());
        assert_eq!(pool.run(9, &job(|r| r.map(|i| i as f32).collect())), (0..9).map(|i| i as f32).collect::<Vec<_>>());
    }
}
