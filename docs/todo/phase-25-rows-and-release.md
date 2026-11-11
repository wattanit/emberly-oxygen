# Phase 25 — The kernel rows, the sweeps and the release: To-do

Working checklist for Phase 25 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.2.md) (milestone M18). Tick items as they land; one task is one commit, tests first where code is written.

Proves: what the blocked product is worth against 0.4.1 at every length and count, and that decode did not change. Satisfies FR-48, FR-8, FR-9, S-7, HC-4.

Layout: `crates/oxygen/benches/`, `results/benchmarks.jsonl`, `docs/releases/`, `README.md`. Behavior: Specification §12, §16.

Dependencies: none new. Out of scope: SIMD intrinsics, `unsafe`, lower-precision weights.

## 25.1 Batched kernel rows (FR-48, HC-2, FR-9)

- [x] Tests first for the support code; then `matmul`, `matmul_blocked` and sgemm at the three shapes, 128 inputs, one thread; the blocked kernel checked before timing.

## 25.2 Version (FR-44)

- [x] Bump the workspace version to 0.4.2; clean tree. The default thread count is re-chosen by the rule after 25.3, changed only if the rule picks another.

## 25.3 Test, then benchmark (FR-8, FR-9)

- [x] `cargo test --no-fail-fast`, then `cargo bench` at the default log level, under `caffeinate`.

## 25.4 Record the measurements (FR-48)

- [ ] Kernel rows; prefill sweep and workloads against 0.4.1; thread sweep against 0.4.1; decode unchanged; peak memory; where the speed-up stops and why.

## 25.5 Release note and README

- [ ] `docs/releases/0.4.2.md`; README status, version note and the equality rule.

## 25.6 Settle the SFD

- [ ] §6 rows absorbed as the owner permits; compliance review; pins checked; statuses.

## 25.7 Owner's review

- [ ] Rows and note reviewed; then stop and ask before merge or tag.
