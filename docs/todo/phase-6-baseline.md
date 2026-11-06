# Phase 6 — Baseline: To-do

Working checklist for Phase 6 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M6: the progression has a starting point). Tick items as they land; one task is one commit with a body paragraph and no co-author line, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read**. IDs cite the Requirements and Technical Specification.

Proves: the first benchmark record exists on the designated machine, every row carrying its conditions, and release 0.1.0 is described and tagged. Satisfies FR-7 to FR-10, S-7, HC-2, HC-4.

Layout: the benchmark harness is `crates/oxygen/benches/benchmarks.rs` with `harness = false` (Spec §1, §12), run by `cargo bench`; its pure helpers (statistics, the record row, machine description) live in `crates/oxygen/benches/support/` and are unit-tested from `crates/oxygen/tests/` through a `#[path]` module, because `cargo test` does not run a `harness = false` bench's tests. The record is `results/benchmarks.jsonl` at the workspace root. The release note is `docs/releases/0.1.0.md`.

Dependencies, all already in Spec §15 so no Specification bump: `libc` as a dev-dependency of the library, for benches only (peak resident memory, machine description); `matrixmultiply` is already a dev-dependency; `serde_json` is already a library dependency. `git` is run as a program for the commit and dirty flag, not linked. Nothing else.

The benchmark machine is **this machine** (owner, 2026-11-06): Apple M4 Max, 16 cores (12 performance, 4 efficiency), 128 GB, macOS 26.3 (Darwin 25.3.0), arm64. Its fixtures were generated on it (manifest `generated_at` 2026-11-05), so the second open question of plan §10 is answered too.

Out of scope: any performance change. This phase measures the engine as it is; techniques start in the next release (Requirements §12).

## Decision points for the owner

Each has a recommendation; none is decided by this to-do. The first blocks 6.2 and is asked there.

1. **The two benchmark prompts. (blocks 6.2.)** Spec §12 fixes their lengths (32 and 512 tokens) but not their text. *Recommendation:* two English prose passages committed in `crates/oxygen/benches/prompts.json`, written for the purpose; the harness encodes them with the engine's tokenizer and refuses to run unless they are exactly 32 and 512 tokens, and refuses a run whose continuation stops at EOS before 128 tokens, so every row measures the same work. English prose because it is the model's main language and its continuations are the least likely to end early; the oracle's prompts are not reused, since none is exactly 32 or 512 tokens and changing them would invalidate the fixtures.
2. **One warm-up run per workload.** The first process to read `model.safetensors` after a while pays the disk, not the engine, which inflates its load time. *Recommendation:* one unrecorded warm-up run per workload before the five recorded ones, stated in each row as `"warmup_runs": 1`. Spec §12 says five runs and this keeps five; it adds one that is not counted.
3. **How the kernel group reaches the in-house matrix-vector product.** `ops::matvec` is private to the library, and a bench is a separate crate. *Recommendation:* a `bench-kernels` Cargo feature, off by default, that makes a hidden `kernels` module public with `matvec` in it, enabled only through the library's existing self-referencing dev-dependency, as `negative-controls` is (plan §9, task 4.7); a normal build cannot see it. Recorded in plan §9 for Spec §1 and §12.
4. **Where the kernel results go.** *Recommendation:* in the same `results/benchmarks.jsonl`, one row per matrix size with `"group": "kernels"` and the same conditions as the workload rows (`"group": "workloads"`), so HC-2's baseline is comparable release to release like everything else.
5. **A dirty tree.** *Recommendation:* the harness always records, with `"dirty": true` when `git status --porcelain` is non-empty, so exploratory runs stay honest; the baseline rows of 6.6 must come from a clean tree at the commit that is tagged, and are rerun otherwise.
6. **Pushing.** Nothing has been pushed this phase or the last. *Recommendation:* the owner pushes `develop`, `main` and the tag after the 6.8 review, or asks for it then.

## 6.1 Benchmark machine (S-7, Requirements §13)

- [ ] Record the owner's decision in plan §10 (both open questions resolved: the machine, and that its fixtures are its own) and in plan §9 for Requirements §13's open item on S-7
- [ ] Note in plan §9 that Phase 4's measurements are recorded as "Apple M5" while Phase 5's and this phase's machine is an Apple M4 Max; ask the owner whether Phase 4 ran on a different machine or the label is wrong, and correct it only on the owner's word

## 6.2 Benchmark harness and workloads (FR-7, FR-8, FR-10, S-7, Spec §12)

- [ ] **Resolve decision point 1 with the owner before writing the prompts**
- [ ] `[[bench]] name = "benchmarks", harness = false` in the library's manifest; `cargo bench` builds with the `bench` profile (optimized, no debug assertions)
- [ ] The harness refuses to run when `cfg!(debug_assertions)` is true, printing why and exiting non-zero (FR-10); checked by hand with debug assertions forced on (`CARGO_PROFILE_BENCH_DEBUG_ASSERTIONS=true cargo bench`)
- [ ] Runs offline and without Python: it reads only `models/` and the committed prompts, and if the model is missing it fails naming the download command (FR-7, HC-3)
- [ ] Workloads `short` (32 prompt tokens) and `long` (512), each generating 128 tokens (Spec §12); the token counts are checked before any timing (decision point 1)
- [ ] Each repetition is a fresh child process: the bench re-executes itself (`std::env::current_exe`) with `--child <workload>`, the child loads the model, generates, and prints one JSON object of its measurements on standard output; the parent collects them (Spec §12 isolation)
- [ ] One unrecorded warm-up run, then 5 recorded runs per workload (S-7, decision point 2)
- [ ] The child runs at the default log level: no subscriber is installed (S-8, Spec §14)
- [ ] A child that fails or prints something unparseable stops the whole run with its standard error shown; nothing is recorded from a partial run
- [ ] Check: the whole `cargo bench` is timed and the figure written here (estimate: the long workload is 640 steps at about 50 ms, so about 32 s a run and about 3.5 minutes for six; the short one about 1 minute)

## 6.3 Metrics (FR-8, Spec §12)

- [ ] From `generate`'s `Stats` (plan §9, tasks 5.3 and 5.4): prompt throughput, generation throughput and time to first token, with the definitions recorded there (the first token counts as prompt work)
- [ ] Model load time, timed in the child around `Model::load` and `Tokenizer::load`, reported separately (Spec §12)
- [ ] Peak resident memory: `getrusage(RUSAGE_SELF)` in the child after generation; `ru_maxrss` is bytes on macOS and kibibytes on Linux, so the unit is converted by target and recorded in bytes
- [ ] Each metric over the 5 runs as median, minimum and maximum (S-7); the median of an odd count is the middle value
- [ ] Unit tests (from `tests/`, decision on layout above) for the statistics on known lists, including ties and an unsorted input; every test names FR-8 or S-7

## 6.4 The record (FR-9, Spec §12)

- [ ] One JSON line per workload appended to `results/benchmarks.jsonl` (created if absent; never rewritten): package version, commit, dirty flag (decision point 5), CPU model, core count (and the performance and efficiency split where the OS reports it), memory, OS and version, architecture, build profile (`bench`, with debug assertions off as checked), timestamp (UTC, RFC 3339), workload (prompt and generated token counts), warm-up and recorded run counts, and the results
- [ ] Machine description read through `libc::sysctlbyname` on macOS (`machdep.cpu.brand_string`, `hw.ncpu`, `hw.perflevel0.physicalcpu`, `hw.perflevel1.physicalcpu`, `hw.memsize`, `kern.osproductversion`) and from `/proc` on Linux; a field the OS does not give is recorded as `null`, never guessed
- [ ] Commit and dirty flag from `git rev-parse HEAD` and `git status --porcelain`; without `git` they are `null`
- [ ] Unit tests: a row built from fixed inputs serializes to the expected JSON line with every condition field present; every test names FR-9

## 6.5 Kernel baseline (HC-2, Spec §12)

- [ ] **Decision points 3 and 4** applied: the `bench-kernels` feature and its hidden module, and `"group": "kernels"` rows
- [ ] The in-house `matvec` against `matrixmultiply::sgemm` with one column, at every matrix size in the model: 576×576 (query and output projections), 192×576 (key and value), 1536×576 (gate and up), 576×1536 (down) and 49152×576 (the tied output projection)
- [ ] Same inputs for both, and their outputs compared within a tolerance before timing, so a broken baseline is never timed
- [ ] Repetitions sized to about a second per measurement, 5 measurements each, median, minimum and maximum of nanoseconds per call and of GFLOP/s
- [ ] Check: a normal `cargo build` does not contain the `kernels` module (generated-documentation check, as for `negative-controls`); recorded in plan §9

## 6.6 The baseline run (HC-4, S-7, S-8, Spec §16)

- [ ] On this machine, on AC power, with no other heavy work running; from a clean tree at the commit that will be tagged
- [ ] `cargo test --no-fail-fast` first; a failure stops here and nothing is benchmarked (HC-4, Spec §16 step 1)
- [ ] Then `cargo bench` at the default log level (S-8), appending the workload and kernel rows (Spec §16 step 2)
- [ ] The rows are committed; plan §8 gets a short table of the baseline medians with their spread
- [ ] Check: every row has every condition field, the dirty flag is false and the commit is the one to be tagged

## 6.7 Release note (Requirements §12)

- [ ] `docs/releases/0.1.0.md`: what the release is (a by-the-book f32, single-threaded engine for SmolLM2-135M); why it was expected to work (the oracle and the harness, the measured errors, the negative controls); what was measured (the baseline rows and the kernel comparison); what was learned (the plan §9 findings, including the near-tie and reproducibility figures); negative results; and that, with no earlier release, this one is the baseline
- [ ] README: status line and a short benchmark section (`cargo bench`, where results go, the benchmark machine)

## 6.8 Tag (Spec §16)

- [ ] **The owner reviews the results and the release note first**
- [ ] Then tag the reviewed commit `v0.1.0` (annotated, message naming the release note); merging to `main` and pushing follow decision point 6

## Phase exit check

- [ ] `results/benchmarks.jsonl` holds the baseline rows from a clean tree, every row carrying its conditions (FR-9, S-7)
- [ ] The harness refused to run with debug assertions on (FR-10), checked by hand
- [ ] Every test added in this phase names the item it verifies (checked mechanically, as in Phase 5)
- [ ] Full test time recorded, `cargo build` has no warnings, and the summary of every test binary was read
- [ ] The release note exists and the README is updated
- [ ] The owner has reviewed the results and the note, and `v0.1.0` is tagged
