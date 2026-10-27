# Phase 2 — Harness: To-do

Working checklist for Phase 2 of the [implementation plan](../emberly-oxygen-implementation-plan-0.1.0.md) (milestone M2: the gate can fail). Tick items as they land; one task is one commit. IDs cite the Requirements and Technical Specification.

Layout used throughout: the harness lives in `crates/oxygen/tests/`, with shared code in `tests/common/` so each test file can `mod common;`. It adds no dependency: `serde` and `serde_json` are already library dependencies (Spec §15), and the fixture reader is hand-written.

## 2.1 Stub engine surface (Spec §2, §7, §8)

- [x] Decide the smallest surface the harness needs and write it down in the module docs: load a model directory; encode text to IDs; decode IDs to text; a session that feeds one token and returns that position's logits, with optional capture of the embedding, each layer's output and the final-norm output; a greedy continuation of `n` tokens
- [x] Add the dependencies the surface needs (`thiserror` now; `serde` and `serde_json` arrive with the fixture reader in 2.2) and the error type with only the variants Spec §14 lists
- [x] Implement the surface as a stub that compiles and returns wrong values (empty token list, zero logits, no capture), never panics on valid input
- [x] Keep the surface minimal: nothing for a later version's needs, and no design for batching or sampling (Requirements §2.3)
- [x] Check: `cargo build` passes; a smoke test calls every method on the stub

## 2.2 Fixture reader (FR-4, FR-7, Spec §11)

- [x] `tests/common/fixtures.rs`: parse `manifest.json` into typed structs (prompts, token IDs, continuation IDs, both gap lists, dims, eos id)
- [x] Hand-written safetensors reader for the fixtures: 8-byte header length, JSON header, F32 slices by name; reject any other dtype with a clear error
- [x] Return tensors as `Vec<f32>` plus shape, loaded one prompt at a time so a run never holds all fixtures in memory (the long ones are about 135 MB each)
- [x] Discover fixtures by globbing `fixtures/*/manifest.json` from the workspace root, so a new model's folder is picked up with no code change (Requirements §2.2)
- [x] The reader never touches engine code, so the oracle's output is not read by the code under test
- [x] Check: unit test reads a tiny safetensors file built in the test, and one real fixture's shapes match the manifest's `dims`

## 2.3 Comparison functions (S-1 to S-4, FR-6)

Pure functions with their own tests, in `tests/common/compare.rs`.

- [x] `max_abs_error(engine, reference)` for logits (S-1, threshold 1e-3), returning the worst value and its position
- [x] `relative_error(engine, reference)` for one intermediate tensor: largest absolute difference divided by the largest magnitude of the reference tensor (S-2, threshold 1e-4); define "tensor" as one stored intermediate for one prompt, meaning one layer's `[positions, hidden]` matrix, and record that reading in the plan's upstream-feedback table
- [x] Guard against an all-zero reference tensor (relative error undefined) and against NaN or infinite engine values, which must always fail
- [x] `top1_check(engine_logits, reference_logits, gaps)` (S-3): first position where the argmax differs and the reference gap is at least 1e-3; positions with a smaller gap pass either way
- [x] `greedy_check(engine_ids, reference_ids, gaps)` (S-4): first differing index; if a near-tie (gap under 1e-3) occurs at or before it, stop comparing there and report the near-tie instead of a failure
- [x] Every check returns first-diverging position, not just a boolean (FR-6)
- [x] Shape mismatches are failures with a message, never a panic (user-visible data never causes a panic, Spec §14)
- [x] Unit tests at each boundary: just under and just over each threshold, an exact tie, a near-tie at the end versus the start of a continuation, NaN, mismatched length
- [x] Check: every unit test names the S-n it verifies

## 2.4 Per-prompt tests driven by the manifest (FR-5, Spec §11)

- [x] One integration test, `tests/correctness.rs`, loops over every manifest and every prompt and records a result per check: token IDs (encode), decode of the reference IDs, logits, top-1, intermediates, greedy continuation
- [x] A check the stub cannot perform yet is run anyway and fails; nothing is skipped or marked ignored, because the gate must be seen to fail
- [x] Run the engine over the prompt one token at a time with capture on; the harness owns the loop so it is the same path later phases use
- [x] Results are collected into a structure and asserted after the table is printed (task 2.6), so one failure does not hide the others
- [x] Check: with the stub, every check for every prompt fails, and the test itself does not panic

## 2.5 First diverging layer (FR-6)

- [x] Compare embeddings first, then layer 0 through the last layer, then the final-norm output against the engine's normalized output
- [x] Report the first tensor that exceeds the S-2 threshold and, within it, the first position where the error exceeds it
- [x] Do not compare later layers once one has diverged in the failure message, but still record the outcome of the whole check
- [x] Check: unit test feeds a synthetic engine capture that diverges at a chosen layer and position and asserts the reported layer and position

## 2.6 Pass/fail table (S-5, FR-5)

- [x] Print a table by prompt (rows) and check (columns), then a summary by category
- [x] A category passes only if every prompt in it passes every check; no averages anywhere (S-5)
- [x] Any failing category fails the run with a non-zero exit code
- [x] Each failure line gives the first diverging position and, for intermediates, the layer (FR-6)
- [x] Note in the module docs that the table appears on failure by default and on success with `cargo test -- --nocapture`
- [x] Check: with the stub, the table prints all five categories failing

## 2.7 Missing inputs fail loudly (FR-7, HC-3, Spec §11)

- [x] Missing `models/<model-id>/` fails the run naming the path and the download command (`cd oracle && uv run python download.py`)
- [x] Missing `fixtures/` or manifest fails naming the oracle command (`cd oracle && uv run python generate.py`)
- [x] A manifest that names a missing tensors file fails naming that file
- [x] The harness never downloads, never runs Python, and never opens a network connection
- [x] Check: tests for each failure point at a temporary empty directory

## Phase exit check

- [x] `cargo test --no-fail-fast` against the stub runs offline, prints the table with every category failing, and exits non-zero
- [x] Unit tests of the comparison functions all pass, including each threshold boundary and the near-tie cases
- [x] A synthetic-divergence test proves the layer and position report is correct
- [x] Every test names the HC, FR or S item it verifies
- [x] `cargo test` time on the stub is recorded: about 0.3 s with a warm build (`cargo test --no-fail-fast`, 12 prompts, 669 MB of fixtures read); the correctness test itself takes about 0.2 s. Fixture loading will dominate once the real engine runs.
- [ ] Owner reviews the table output and the comparison functions' tests before Phase 3 starts

## Notes to carry forward

- Requirements S-3 was corrected in place by the owner (no version bump); its wording is "there either token passes".
- Plain `cargo test` stops at the first failing test binary, so while the engine is a stub the other test files do not run; use `cargo test --no-fail-fast` until the correctness harness passes.
- The stub is thrown away piece by piece: Phase 3 replaces encode and decode, Phase 4 replaces the forward pass, Phase 5 replaces generation.
