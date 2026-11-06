# Phase 6 — Baseline: To-do

Working checklist for Phase 6 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M6: the progression has a starting point). Tick items as they land; one task is one commit with a body paragraph and no co-author line, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read**. IDs cite the Requirements and Technical Specification.

Proves: the first benchmark record exists on the designated machine, every row carrying its conditions, and release 0.1.0 is described and tagged. Satisfies FR-7 to FR-10, S-7, HC-2, HC-4.

Layout: the benchmark harness is `crates/oxygen/benches/benchmarks.rs` with `harness = false` (Spec §1, §12), run by `cargo bench`; its pure helpers (statistics, the record row, machine description) live in `crates/oxygen/benches/support/` and are unit-tested from `crates/oxygen/tests/` through a `#[path]` module, because `cargo test` does not run a `harness = false` bench's tests. The record is `results/benchmarks.jsonl` at the workspace root. The release note is `docs/releases/0.1.0.md`.

Dependencies, all already in Spec §15 so no Specification bump: `libc` as a dev-dependency of the library, for benches only (peak resident memory, machine description); `matrixmultiply` is already a dev-dependency; `serde_json` is already a library dependency. `git` is run as a program for the commit and dirty flag, not linked. Nothing else.

The benchmark machine is **this machine** (owner, 2026-11-06): Apple M4 Max, 16 cores (12 performance, 4 efficiency), 128 GB, macOS 26.3 (Darwin 25.3.0), arm64. Its fixtures were generated on it (manifest `generated_at` 2026-11-05), so the second open question of plan §10 is answered too.

Out of scope: any performance change. This phase measures the engine as it is; techniques start in the next release (Requirements §12).

## Decision points for the owner

Each has a recommendation; none is decided by this to-do. The first blocks 6.2 and is asked there.

1. **The two benchmark prompts. (blocks 6.2.)** *Resolved by the owner: the recommendation, prompts written for the benchmark.* Spec §12 fixes their lengths (32 and 512 tokens) but not their text. *Recommendation:* two English prose passages committed in `crates/oxygen/benches/prompts.json`, written for the purpose; the harness encodes them with the engine's tokenizer and refuses to run unless they are exactly 32 and 512 tokens, and refuses a run whose continuation stops at EOS before 128 tokens, so every row measures the same work. English prose because it is the model's main language and its continuations are the least likely to end early; the oracle's prompts are not reused, since none is exactly 32 or 512 tokens and changing them would invalidate the fixtures.
2. **One warm-up run per workload.** *Resolved by the owner: yes.* The first process to read `model.safetensors` after a while pays the disk, not the engine, which inflates its load time. *Recommendation:* one unrecorded warm-up run per workload before the five recorded ones, stated in each row as `"warmup_runs": 1`. Spec §12 says five runs and this keeps five; it adds one that is not counted.
3. **How the kernel group reaches the in-house matrix-vector product.** *Resolved by the owner: the recommendation, a `bench-kernels` feature (option b of making it public, a test-only feature, or copying it).* `ops::matvec` is private to the library, and a bench is a separate crate. *Recommendation:* a `bench-kernels` Cargo feature, off by default, that makes a hidden `kernels` module public with `matvec` in it, enabled only through the library's existing self-referencing dev-dependency, as `negative-controls` is (plan §9, task 4.7); a normal build cannot see it. Recorded in plan §9 for Spec §1 and §12.
4. **Where the kernel results go.** *Resolved by the owner: the recommendation, the same record with a `group` field.* *Recommendation:* in the same `results/benchmarks.jsonl`, one row per matrix size with `"group": "kernels"` and the same conditions as the workload rows (`"group": "workloads"`), so HC-2's baseline is comparable release to release like everything else.
5. **A dirty tree.** *Resolved by the owner: the recommendation.* *Recommendation:* the harness always records, with `"dirty": true` when `git status --porcelain` is non-empty, so exploratory runs stay honest; the baseline rows of 6.6 must come from a clean tree at the commit that is tagged, and are rerun otherwise.
6. **Pushing and tagging.** *Decided by the owner as a standing rule:* nothing is pushed or tagged, and nothing is merged to `main` for release, without the owner's explicit order at the time. 6.8 stops and asks.

## 6.1 Benchmark machine (S-7, Requirements §13)

- [x] Record the owner's decision in plan §10 (both open questions resolved: the machine, and that its fixtures are its own) and in plan §9 for Requirements §13's open item on S-7
- [x] Note in plan §9 that Phase 4's measurements are recorded as "Apple M5" while Phase 5's and this phase's machine is an Apple M4 Max; ask the owner whether Phase 4 ran on a different machine or the label is wrong, and correct it only on the owner's word. The owner confirmed Phase 4 ran on an M5 and development moved to the M4 Max before task 5.1, so the label stands; plan §9 records the move and that Phase 4's worst errors reproduced to the last digit on the M4 Max

## 6.2 Benchmark harness and workloads (FR-7, FR-8, FR-10, S-7, Spec §12)

- [x] **Resolve decision point 1 with the owner before writing the prompts.** Written: `short` is one sentence about a lighthouse keeper (32 tokens), `long` four paragraphs about a coastal town (512 tokens), fitted with the engine's tokenizer through the CLI (`--max-new-tokens 0 -v` logs the count); both generate the full 128 tokens without EOS
- [x] `[[bench]] name = "benchmarks", harness = false` in the library's manifest; `cargo bench` builds with the `bench` profile (optimized, no debug assertions)
- [x] The harness refuses to run when `cfg!(debug_assertions)` is true, printing why and exiting non-zero (FR-10); checked by hand with debug assertions forced on (`CARGO_PROFILE_BENCH_DEBUG_ASSERTIONS=true cargo bench`): exit 1 with "refusing to benchmark a build with debug assertions on ... (FR-10)"
- [x] Runs offline and without Python: it reads only `models/` and the committed prompts, and if the model is missing it fails naming the download command (FR-7, HC-3)
- [x] Workloads `short` (32 prompt tokens) and `long` (512), each generating 128 tokens (Spec §12); the token counts are checked before any timing (decision point 1)
- [x] Each repetition is a fresh child process: the bench re-executes itself (`std::env::current_exe`) with `--child <workload>`, the child loads the model, generates, and prints one JSON object of its measurements on standard output; the parent collects them (Spec §12 isolation)
- [x] One unrecorded warm-up run, then 5 recorded runs per workload (S-7, decision point 2)
- [x] The child runs at the default log level: no subscriber is installed (S-8, Spec §14)
- [x] A child that fails or prints something unparseable stops the whole run with its standard error shown; nothing is recorded from a partial run
- [x] Check: the whole `cargo bench` is timed and the figure written here (estimate: the long workload is 640 steps at about 50 ms, so about 32 s a run and about 3.5 minutes for six; the short one about 1 minute). Measured: 4 min 5 s for both workloads. Runs are within about 1% of each other; the short workload gives about 1.57 s to first token and 20.3 tokens/s generation, the long one about 25.9 s and 18.8 tokens/s, with prompt throughput the same as generation (one token per step, Spec §2). These are not recorded yet: medians and memory are 6.3, the record 6.4

## 6.3 Metrics (FR-8, Spec §12)

- [x] From `generate`'s `Stats` (plan §9, tasks 5.3 and 5.4): prompt throughput, generation throughput and time to first token, with the definitions recorded there (the first token counts as prompt work)
- [x] Model load time, timed in the child around `Model::load` and `Tokenizer::load`, reported separately (Spec §12)
- [x] Peak resident memory: `getrusage(RUSAGE_SELF)` in the child after generation; `ru_maxrss` is bytes on macOS and kibibytes on Linux, so the unit is converted by target and recorded in bytes (`None` on other systems). The call is the harness's one `unsafe` block, commented; the engine's own code stays free of `unsafe`. Measured: 811.5 MB in every run of both workloads, because the peak is set while loading: the bf16 file (269 MB, memory-mapped and resident while it is read) plus its weights widened to f32 (538 MB); the key-value cache grows with the positions used and is small by comparison
- [x] Each metric over the 5 runs as median, minimum and maximum (S-7); the median of an odd count is the middle value
- [x] Unit tests (from `tests/`, decision on layout above) for the statistics on known lists, including ties and an unsorted input; every test names FR-8 or S-7. `tests/bench_support.rs` includes `benches/support/stats.rs` by path: 4 tests, also covering an even count (mean of the middle two) and empty or NaN input (no spread)
- [x] Measured (4 min 4 s): short, median 1.566 s to first token, 20.44 prompt and 20.29 generation tokens/s, load 57 ms; long, 26.0 s, 19.70 and 18.74 tokens/s, load 59 ms; every spread within about 1%. Not yet recorded; the record is 6.4

## 6.4 The record (FR-9, Spec §12)

- [x] One JSON line per workload appended to `results/benchmarks.jsonl` (created if absent; never rewritten): package version, commit, dirty flag (decision point 5), CPU model, core count (and the performance and efficiency split where the OS reports it), memory, OS and version, architecture, build profile (`bench`, with debug assertions off as checked), timestamp (UTC, RFC 3339), workload (prompt and generated token counts), warm-up and recorded run counts, and the results. Rows are written once, after every workload has finished, so a run that stops early records nothing. `cargo bench -- --dry-run` prints the rows instead of appending them, so development runs never enter the record
- [x] Machine description read through `libc::sysctlbyname` on macOS (`machdep.cpu.brand_string`, `hw.ncpu`, `hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu`, `hw.memsize`, `kern.osproductversion`) ; a field the OS does not give is recorded as `null`, never guessed. Changed while writing it: there is no Linux branch, because it could not be checked on this machine; on any other system the core count, OS name and architecture come from the standard library and the rest is `null`. The two `sysctlbyname` calls are in one commented `unsafe` block
- [x] Commit and dirty flag from `git rev-parse HEAD` and `git status --porcelain`; without `git` they are `null`
- [x] Unit tests: a row built from fixed inputs serializes to the expected JSON line with every condition field present; every test names FR-9. Three tests in `tests/bench_support.rs`: the full row against an expected value, unknown conditions written as `null`, and RFC 3339 timestamps at the epoch, a leap day and a known instant (checked with `date -u`)
- [x] A full dry run showed both rows complete: commit, `dirty: true` (correct, the change was uncommitted), Apple M4 Max with 16 cores (12 + 4), 128 GB, macOS 26.3, aarch64, the bench profile without debug assertions, and the timestamp; `results/` was left untouched. That run was about 2.5% slower than 6.3's with a spread of about 3%, the ordinary noise S-7's spread is there to show

## 6.5 Kernel baseline (HC-2, Spec §12)

- [x] **Decision points 3 and 4** applied: the `bench-kernels` feature and its hidden module, and `"group": "kernels"` rows. The module `emberly_oxygen::kernels` holds a `matvec` that calls the engine's own; it is feature-gated but not `#[doc(hidden)]`, so the documentation check below can show it absent and present
- [x] The in-house `matvec` against `matrixmultiply::sgemm` with one column, at every matrix size in the model: 576×576 (query and output projections), 192×576 (key and value), 1536×576 (gate and up), 576×1536 (down) and 49152×576 (the tied output projection)
- [x] Same inputs for both, and their outputs compared within a tolerance before timing, so a broken baseline is never timed
- [x] Repetitions sized to about a second per measurement, 5 measurements each, median, minimum and maximum of nanoseconds per call and of GFLOP/s
- [x] Check: a normal `cargo build` does not contain the `kernels` module (generated-documentation check, as for `negative-controls`); recorded in plan §9. `cargo doc --no-deps` lists no `kernels`; with `--features bench-kernels` it lists `kernels::matvec`. No warnings with and without the features
- [x] Dry run (the whole bench now takes about 5 minutes; the kernel group about 1). Medians, ns per call and GFLOP/s, in-house then `matrixmultiply`: 576×576 115 µs 5.77 / 92 µs 7.20; 192×576 38 µs 5.81 / 31 µs 7.05; 1536×576 304 µs 5.82 / 243 µs 7.28; 576×1536 382 µs 4.64 / 244 µs 7.25; 49152×576 9.72 ms 5.82 / 12.29 ms 4.61. Findings for the release note: (1) per token these add up to about 48.6 ms of the 49.3 ms step, so the matrix-vector product is about 98.6% of the work; (2) `matrixmultiply` is a general matrix-matrix routine, and with one column it is only about 1.25× faster at most sizes and slower at the 49152×576 output projection, so it is a weak yardstick for this kernel: the Phase 4 partial-sum variants (14 to 16 G multiply-adds/s with 8 sums) already beat it; (3) 5.8 GFLOP/s is 2.9 G multiply-adds/s, matching Phase 4's 2.3 to 3.0 G, which counted multiply-adds

## 6.6 The baseline run (HC-4, S-7, S-8, Spec §16)

- [x] On this machine, on AC power, with no other heavy work running; from a clean tree at the commit that will be tagged. Measured at `0b4a5f3`; the commits after it change only `results/` and `docs/`, so the code tagged in 6.8 is the code measured
- [x] `cargo test --no-fail-fast` first; a failure stops here and nothing is benchmarked (HC-4, Spec §16 step 1). Passed in 86 s. A first attempt piped the test output through `grep`, whose exit status would have hidden a failure; it was stopped during the tests, before any benchmark, and rerun gated on `cargo test`'s own exit status
- [x] Then `cargo bench` at the default log level (S-8), appending the workload and kernel rows (Spec §16 step 2). 306 s, 7 rows. The machine was still warm from the tests, which put the workloads about 6% below idle-machine runs; the owner kept the rows as the baseline (plan §8)
- [x] The rows are committed; plan §8 gets a short table of the baseline medians with their spread. Short: 1.68 s to first token, 18.97 generation tokens/s; long: 26.5 s, 18.48 tokens/s; 811.6 MB peak; in-house matvec 4.6 to 5.8 GFLOP/s against matrixmultiply's 4.3 to 7.2
- [x] Check: every row has every condition field, the dirty flag is false and the commit is the one to be tagged

## 6.7 Release note (Requirements §12)

- [x] `docs/releases/0.1.0.md`: what the release is (a by-the-book f32, single-threaded engine for SmolLM2-135M); why it was expected to work (the oracle and the harness, the measured errors, the negative controls); what was measured (the baseline rows and the kernel comparison); what was learned (the plan §9 findings, including the near-tie and reproducibility figures); negative results; and that, with no earlier release, this one is the baseline
- [x] Written, with its figures checked against `results/benchmarks.jsonl`, plan §8 and §9 before committing (three corrections: the test count is 203 with the bench helpers' tests; the partial-sum speed-ups were timed but never run through the harness, so the note says they are expected to stay within tolerance, not that they do; the top-token check "passes" rather than being "equal", since S-3 allows near-ties)
- [x] README: status line and a short benchmark section (`cargo bench`, where results go, the benchmark machine). The status line says what remains: the owner's review and the tag; the layout lists release notes under `docs/`

## 6.8 Tag (Spec §16)

- [x] **The owner reviews the results and the release note first.** Reviewed and accepted 2026-11-06
- [x] Then, **only on the owner's explicit order**, tag the reviewed commit `v0.1.0` (annotated, message naming the release note); merging to `main` and pushing likewise wait for the owner's order (decision point 6). The owner ordered the tag and a `--no-ff` merge of `develop` into `main`; the tag is on this commit. Nothing is pushed

## Phase exit check

- [x] `results/benchmarks.jsonl` holds the baseline rows from a clean tree, every row carrying its conditions (FR-9, S-7): 7 rows at `0b4a5f3`, `dirty: false`
- [x] The harness refused to run with debug assertions on (FR-10), checked by hand (task 6.2)
- [x] Every test added in this phase names the item it verifies (checked mechanically, as in Phase 5): 7 new tests, none without an ID
- [x] Full test time recorded, `cargo build` has no warnings, and the summary of every test binary was read: 203 tests across 14 binaries in about 86 s, exit 0; no warnings in the normal build, the bench build or with both test features
- [x] The release note exists and the README is updated (task 6.7)
- [x] The owner has reviewed the results and the note, and `v0.1.0` is tagged
