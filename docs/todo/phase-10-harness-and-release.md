# Phase 10 — Harness and release: To-do

Working checklist for Phase 10 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.2.md) (the second half of milestone M8: the gate judges against the f64 reference, can fail, and the release is measured). Tick items as they land; one task is one commit with a body paragraph and no attribution line, each ending with a passing `cargo test --no-fail-fast` whose **whole summary is read** (the measurement tasks add the procedure's own steps).

Proves: S-1 to S-4 are judged against the f64 fixtures with the engine unchanged; the f32 distance is reported and never gates; the negative controls fail against the new reference; the release is recorded. Satisfies FR-5, FR-6, FR-8, FR-9, S-1 to S-8, HC-4.

Layout: the harness is `crates/oxygen/tests/` (`common/fixtures.rs` the reader, `common/compare.rs` the checks, `common/harness.rs` the runs, `common/report.rs` the table; tests in `fixture_reader.rs`, `comparison.rs`, `report.rs`, `negative_controls.rs`, `correctness.rs`). The benchmark is `crates/oxygen/benches/benchmarks.rs`; the record is `results/benchmarks.jsonl`.

Dependencies: none new.

Out of scope: any change to `crates/oxygen/src` other than the version number (task 10.8). Any change to a threshold before task 10.7. Any change to the benchmark harness or its workloads.

The benchmark machine is the owner's Apple M4 Max (Requirements §13).

## Decision points for the owner

Each has a recommendation; **the owner agreed to all of them on 2026-11-08**. Point 1 blocks 10.7 and must be settled before the measurement in 10.6 is seen.

1. **How the thresholds are decided, written before the numbers. (blocks 10.7.)** *Recommendation:* for each of S-1 (logit) and S-2 (intermediate), take the worst error over all categories against the f64 reference. If it is at most **half the limit**, keep the limit: the margin is at least 2×, which is what 0.1.0 had at 22% and 0.1.1 lost. If it is above half and below the limit, the owner chooses between loosening and investigating; if it is at or above the limit, that is a failure to investigate, not a reason to loosen. A limit is not tightened to follow the engine's error, because a real bug smaller than the new limit could then hide, and the negative controls were sized for the present values. The near-tie gap (S-3) stays equal to the S-1 limit.
2. **What the f32 distance column shows. (applies to 10.4.)** *Recommendation:* the same two figures as the gating columns (worst logit error and worst relative intermediate error) per category, labelled "vs f32 (reported)", with no pass or fail mark, so the table cannot be misread as gating on it.
3. **Whether the old bit-exact tests move. (applies to 10.3.)** *Recommendation:* tests that assert a value against the f32 fixtures for a reason other than the gate (for example reproducibility, S-6, which compares the engine with itself) are untouched; tests that assert the gate's verdict move to the f64 data. If a test cannot be classified, it goes to the owner rather than being changed.
4. **Merging, tagging, pushing.** *Standing rule of the owner:* none happens without the owner's explicit order at the time, and pushing is the owner's own act. 10.12 stops and asks.

## 10.1 The reader accepts F64 (FR-4, FR-7, Spec §11)

- [x] **Tests first**, in `fixture_reader.rs`, seen to fail: an F64 file parses with exact values; a dtype different from the one asked for is refused naming both dtypes; a truncated or miscounted f64 file fails with a message, not a panic. *The tests that need `load_reference` (a missing f64 file and an F32 file where F64 is named, each naming the command; the real f64 shapes equal the manifest's) were written with task 10.2, where that function is added; this box was ticked too early in the 10.1 commit.*
- [x] `read_safetensors` takes the dtype it expects and returns an f32 or f64 tensor accordingly; the tensor type carries which it is.

## 10.2 The manifest's f64 fields (FR-4, FR-7)

- [x] `Prompt` gains the `f64_` fields of Specification §10; `discover` fails naming the oracle command if any is missing, so a stale fixture folder is told to regenerate.
- [x] `PromptTensors` loads both precisions for a prompt. Existing checks keep running on the f32 data until 10.4.

## 10.3 Judge against an f64 reference (S-1 to S-4, FR-2)

- [x] Apply decision point 3. **Tests first**, in `comparison.rs`, seen to fail against the old functions: a reference value that f32 cannot represent, where narrowing the reference gives a different error than measuring against the unrounded value; the near-tie rule reads f64 gaps; the greedy comparison reads an f64 continuation and stops at an f64 near-tie.
- [x] The comparison functions take the f64 reference and widen the engine's f32 values before subtracting. `LOGIT_TOLERANCE`, `INTERMEDIATE_TOLERANCE` and `NEAR_TIE_GAP` keep their values (now f64 constants of the same decimals, so "an f32 value exactly at the tolerance" can no longer be built and the boundary test is just under and just over). *The shared `common` harness module is compiled by every test file, so the harness's wiring to the f64 fixtures had to land in this same commit; 10.4 then adds the report and its tests.*

## 10.4 Wire the gate and report the f32 distance (FR-5, FR-6, S-5)

- [x] Apply decision point 2. The harness runs S-1 to S-4 on the f64 fixtures and measures the distance to the f32 fixtures for the logit and intermediate checks.
- [x] The category table shows both, the second labelled as reported, with no verdict. A test in `report.rs`: a prompt whose f32 distance is over a limit but whose f64 error is under it passes. *Also an integration test in `correctness.rs` (the f32 logits moved by 0.5; the prompt still passes against f64 and the report shows 0.5). Both were written after the wiring, which task 10.3 forced, so each was shown able to fail by temporarily judging against the f32 reference.*
- [x] A failure still names the first diverging position and layer (FR-6), judged against the f64 data.

## 10.5 Negative controls (HC-4, S-1, S-2)

- [x] All three controls fail the harness against the f64 reference, each as before (position named for `PerturbedLogit`).
- [x] A test asserts the `PerturbedLogit` size exceeds `LOGIT_TOLERANCE` and stays within the range the control was sized for, so a later change of limit makes it fail loudly. *The size is measured, not read from the engine (the same prompt with and without the fault, so `crates/oxygen/src` is untouched): 1.9999e-3 in f32. Shown able to fail with S-1 temporarily at 2e-3.*

## 10.6 The full gate (HC-4, S-1 to S-6, S-8)

- [ ] `cargo test --no-fail-fast`, whole summary read; record the time against 28 s.
- [ ] Record, per category, the worst logit and intermediate error against the f64 reference and against the f32 one for the 16-sum engine, and the margin to each limit, in plan §4.
- [ ] Optional, scratchpad only, for the release note: the same table with the engine at `LANES = 1` (0.1.0's order) and 8, so the three kernels' distances to the truth sit side by side. The constant is restored and `git status` is clean afterwards.
- [ ] Reproducibility (S-6) and output at every verbosity level (S-8) still hold.

## 10.7 The owner decides the thresholds (S-1 to S-4, Requirements §13)

- [ ] **Stop.** Present plan §4's table against decision point 1's rule. The owner decides: keep, tighten or loosen S-1 to S-4.
- [ ] If kept: record the decision in plan §4 and §6; the Open Question moves to resolved in the next foundation bump.
- [ ] If changed: a Requirements bump (0.4) is drafted and approved first, the Specification pins it, the constants and the control's size follow it, and tasks 10.5 and 10.6 are repeated.

## 10.8 Prepare the measurement (FR-9, Spec §16)

- [ ] The workspace version becomes 0.1.2 in its own commit, so every row of this release carries it.
- [ ] A clean tree, the measured commit recorded.

## 10.9 Test, then bench (HC-4, S-7, S-8, Spec §16)

- [ ] `cargo test --no-fail-fast`, then `cargo bench` straight after on the M4 Max at the default log level; a failing test run stops here. No cool-down, so the rows stay comparable with 0.1.1's.
- [ ] The rows land in `results/benchmarks.jsonl` and are committed on their own.

## 10.10 Compare with 0.1.1 (FR-8, FR-9)

- [ ] In plan §4: workloads and kernel shapes against 0.1.1's rows. The engine is unchanged, so the difference is run-to-run and machine-state variation; say so, with the spread.

## 10.11 Release note and README (Requirements §12)

- [ ] `docs/releases/0.1.2.md`: what changed (the reference), why it was expected to help, what was measured (the tables of §4, the thresholds decision), what was learned, negative results. It does not claim a speed change.
- [ ] README: status, the oracle's longer run and about 2 GB of fixtures, the test suite's time, and where the f32 distance appears.

## 10.12 Owner's review (Spec §16)

- [ ] The owner reviews the rows and the note. Then **stop and ask** before merging or tagging `v0.1.2`; pushing is the owner's own act.

## Exit check

- [ ] The harness passes every prompt in every category against the f64 reference with the thresholds the owner decided; the f32 distance is reported and gates nothing.
- [ ] All three negative controls fail; the `PerturbedLogit` size is checked against the limit.
- [ ] `results/benchmarks.jsonl` holds the 0.1.2 rows from a clean tree at the measured commit.
- [ ] The release note exists and matches the record; the owner has reviewed; merge and tag happen only on the owner's order.
