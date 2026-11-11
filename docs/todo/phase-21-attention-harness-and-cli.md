# Phase 21 — Threaded attention, the harness and the command line: To-do

Working checklist for Phase 21 of the [implementation plan](../emberly-oxygen-implementation-plan-0.4.1.md) (second part of milestone M15). Tick items as they land; one task is one commit, tests first.

Proves: the whole engine is bit-identical at every thread count, the harness passes unchanged in its thresholds, and `SkippedRows` fails it. Satisfies FR-43, FR-44, S-1 to S-4, S-6, S-8, HC-4.

Layout: `crates/oxygen/src/forward.rs`, `fault.rs`, the CLI crate, `tests/`. Behavior: Specification §4, §7, §9, §11, §14, §17.

Dependencies: none new. Out of scope: finer attention splits (only if the sweep shows heads alone fall short), `unsafe`.

## 21.1 Threaded attention in `step` (FR-43)

- [ ] Tests first: logits, capture, cache exactly equal at counts 1, 2, 3, 7, default over a decode run.
- [ ] Then heads divided among workers.

## 21.2 Threaded attention in `prefill` (FR-43, FR-41)

- [ ] Tests first: exactly equal at every count; still equal to `step`; causal test holds.
- [ ] Then rows divided among workers.

## 21.3 `Fault::SkippedRows` (HC-4)

- [ ] Negative control first: with an explicit 4 workers it changes the output; with no fault nothing changes.
- [ ] Then the fault.

## 21.4 Thread-count independence test (FR-43, S-6)

- [ ] Prompt set, both paths, exact comparison across counts; a faulted run is over the bound.

## 21.5 The harness through the threaded engine (S-1 to S-4, HC-4)

- [ ] Correctness test at the default count, table read; thresholds unchanged; `SkippedRows` fails it.
- [ ] Guard tests that pin ids before and after: every id equal.

## 21.6 `--threads` (FR-44, S-8)

- [ ] Tests first: flag on `generate` and `chat`; count in `-v` and the run summary; 0 rejected naming option and value; output identical at every count and verbosity.
- [ ] Then the flag.

## 21.7 The full gate

- [ ] `cargo test --no-fail-fast`, whole summary read; harness table recorded in §5.
