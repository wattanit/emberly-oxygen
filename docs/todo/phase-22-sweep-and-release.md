# Phase 22 — The thread sweep and the release: To-do

Working checklist for Phase 22 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.1.md) (milestone M16). Tick items as they land; one task is one commit, tests first where code is written.

Proves: what threads are worth against 0.4.0 at every count, and the default count. Satisfies FR-45, FR-44, FR-8, FR-9, S-7, HC-4.

Layout: `crates/oxygen/benches/`, `results/benchmarks.jsonl`, `docs/releases/`, `README.md`. Behavior: Specification §12, §16.

Dependencies: none new. Out of scope: SIMD, register blocking, lower-precision weights.

## 22.1 The thread sweep group (FR-45, FR-9, S-7)

- [x] Tests first for the support code: counts, prompts, the equal-to-one-thread check, the row with `threads`.
- [x] Then the group: 1, 2, 4, 8, 12, 16 threads; prompt at 128 and 644; decode after 32 and 512.

## 22.2 Default count and version (FR-44)

- [ ] Choose the default by the rule of Specification §12; set it.
- [ ] Bump the workspace version to 0.4.1; clean tree.

## 22.3 Test, then benchmark (FR-8, FR-9)

- [ ] `cargo test --no-fail-fast`, then `cargo bench` at the default log level, under `caffeinate`.

## 22.4 Record the measurements (FR-45)

- [ ] Sweep per count, both workloads against 0.4.0, peak memory, one thread against 0.4.0, where the speed-up stops and why.

## 22.5 Release note and README

- [ ] `docs/releases/0.4.1.md`; README status line and version note.

## 22.6 Settle the SFD

- [ ] §6 rows absorbed as the owner permits; compliance review; pins checked; statuses approved by the owner.

## 22.7 Owner's review

- [ ] Rows and note reviewed; then stop and ask before merge or tag.
